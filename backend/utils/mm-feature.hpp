#pragma once

/// @file mm-feature.hpp
/// @brief 多模态项的载体（wire 解码 ← → scheduler ↔ runner）。
///
/// 放在 **utils/** 的理由：它是跨层共享的**纯数据叶子**——只用到 forch 的 dtype
/// 类型，不碰图/执行栈。放这里，wire 层（`server/protocol.hpp`、
/// `server/protocol.hpp` 的 mm 段）就能在"双方都能 include
/// 的最低层"拿到它，而不必去 include
/// 引擎的执行栈（`engine/model-runner.hpp`）。
///
/// dtype 用 `forch::DType` 而不是自定义枚举：payload 的终点就是 forch 张量，
/// 全引擎（models/*）也只用这一个词汇表；"wire 只放行哪三种"的约束由解码器
/// （`server/protocol.hpp` 的 mm 段）保证，不必再拿一个新类型复述一遍。
///
/// 三件约定：struct 无行为；跨界只走 struct；wire 词汇（dtype 名、kwarg
/// 名）留在 protocol 层，这里只装"已经翻译好、可以直接用"的数据。

#include <cstdint>
#include <string>
#include <vector>

#include <forch/forch.hpp> // forch::DType（utils 本来就坐在 forch 上）

namespace finer::mm {

/// 一个多模态**项**（一项 = 一张图 / 一段视频）。纯文本请求为空数组。
///
/// 字段按 vLLM wire + Qwen-VL 系的预处理约定定死（qwen3.5/3.8/qwen3-vl 同构：
/// 图像处理器 = Qwen2VLImageProcessorFast，patch_size 16、temporal_patch_size
/// 2、 merge_size 2）：
///   * 视觉 token 数 = (H/16)×(W/16) / merge_size²（视频再除以
///   temporal_patch_size
///     的时间维）——每项一个占位 token，由前端按 grid 展开成 mm_length 个
///     `<|image_pad|>`(248056) / `<|video_pad|>`(248057)，两侧是
///     `<|vision_start|>`(248053) / `<|vision_end|>`(248054)；
///   * 重嵌入在 runner 侧：把编码输出按行号散写回 token embedding 流
///     （Qwen3-VL 还有 deepstack
///     多层注入，所以缓存的是**编码输出集合**，不是单个矩阵）；
///   * M-RoPE 的 3 轴位置与 rope_deltas 由 runner 从 grid_thw 算出，scheduler
///   不知情。
struct MmFeatureSpec {
  /// 0 = image，1 = video，2 = 其他模态（不支持；准入会拒）
  int32_t modality = 0;

  /// 项内容的哈希：embedding 缓存的键，也进 KV 前缀的块哈希（同文本不同图不能
  /// 复用 KV）。由前端算，两侧都只当不透明字符串。
  std::string identifier;

  /// 占位 token 在 prompt 中的区间：[mm_offset, mm_offset + mm_length)
  int64_t mm_offset = 0;
  int64_t mm_length = 0;

  /// patch 网格 [T, H, W]（T=1 表示单帧图像）。定视觉 token 数，也是 M-RoPE
  /// 位置的依据；视频时配合 second_per_grid_ts。
  std::vector<int64_t> image_grid_thw;

  /// 视频的时间轴尺度（每帧/每 temporal 网格占多少秒）；图像为 0。
  float second_per_grid_ts = 0.0F;

  /// 像素负载（原始 wire 数据）。**重负载只归 runner**：scheduler 在该项准入
  /// 之后即可丢弃这份数据，runner 需要时自己拷贝持有。
  std::vector<int64_t> pixel_shape;
  forch::DType         pixel_dtype = forch::DType::BF16;
  std::vector<uint8_t> pixel_bytes;

  /// wire 的 `mm_position.is_embed`：占位区间内**哪些位置真吃 embedding**。
  /// 空 = wire 未提供（等价全 true）。Qwen-VL 系恒为全 true；像 Llama4 那样
  /// "只有 patch 位置吃 embedding" 的模型需要 runner 逐位置处理。
  std::vector<uint8_t> is_embed;

  /// 注意：编码代价就是 `mm_length`（视觉 token 数），**直接读字段**——struct
  /// 不带行为。
};

/// 本步要保证可用的多模态项（纯数据，scheduler 依"本步窗口 + encoder 预算"
/// 给出；scheduler 只知道项的**下标、区间与代价**，不知道项是什么）。
struct VisionJob {
  int64_t slot = 0;
  /// 该槽请求里的 mm 项下标（**自持有**：每步至多几项，分配可忽略；用 span 反而
  /// 要求调用方维护"下标数组活到调用结束"，平白多一趟循环）。
  std::vector<int32_t> item_indices;
};

} // namespace finer::mm
