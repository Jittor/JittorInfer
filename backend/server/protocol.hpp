#pragma once

/// @file protocol.hpp
/// @brief Finer engine-server wire layer: protocol types, msgpack codec.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#include "../utils/mm-feature.hpp" // mm::MmFeatureSpec（wire 的载荷）

namespace finer::server {

// Protocol constants

enum class RequestType : uint8_t {
  Add         = 0x00,
  Abort       = 0x01,
  StartDpWave = 0x02,
  Utility     = 0x03,
};

enum class FinishReason : uint8_t {
  Stop       = 0,
  Length     = 1,
  Abort      = 2,
  Error      = 3,
  Repetition = 4,
};

enum class SlotStatus : uint8_t {
  FREE       = 0,
  PREFILLING = 1,
  DECODING   = 2,
};

// Protocol data structures

struct SamplingParams {
  float   temperature  = -1.0f; // -1 = use config default
  float   top_p        = -1.0f;
  int64_t top_k        = -1;
  int64_t seed         = -1;
  int64_t max_tokens   = 0; // 0 = use config default
  int64_t min_tokens   = 0;
  int64_t eos_token_id = -1;
  // Request-level stop tokens (client-specified).
  std::unordered_set<int64_t> stop_token_ids;
  // Template-level stop set (vLLM `_all_stop_token_ids`), typically the
  // chat template's terminators including eos.
  std::unordered_set<int64_t> all_stop_token_ids;
  float                       frequency_penalty  = 0.0f;
  float                       presence_penalty   = 0.0f;
  float                       repetition_penalty = 1.0f;

  inline bool is_stop_token(int64_t token) const {
    return (eos_token_id >= 0 && token == eos_token_id)
      || all_stop_token_ids.count(token) || stop_token_ids.count(token);
  }

  static SamplingParams from_msgpack(const nlohmann::json &j);
};

struct Request {
  // Input (from msgpack)
  std::string                    request_id;
  std::vector<int64_t>           prompt_token_ids;
  std::vector<mm::MmFeatureSpec> mm_features; // wire [2]；纯文本为空
  SamplingParams                 sampling;
  /// 非空 = 这条 Add 解不成可服务的请求（**wire 层读不懂**：未知 dtype、不认识
  /// 的处理器 kwarg、张量不在本帧内、网格不是 int64{T,H,W}…）。
  /// scheduler 据此直接以 FinishReason::Error 收尾——不占槽、不排队。
  /// decode 绝不抛异常：抛了会带着整条请求炸掉主循环。
  std::string                decode_error;
  uint32_t                   client_index = 0;
  uint32_t                   current_wave = 0;
  int32_t                    priority     = 0;
  double                     arrival_time = 0.0;
  std::optional<std::string> cache_salt;
  std::optional<std::string> external_req_id;

  // Output (accumulated during generation)
  std::vector<int64_t> generated_ids;
  int64_t              next_token = 0;

  static Request from_msgpack(const nlohmann::json &j);
};

struct Output {
  std::string                 request_id;
  std::vector<int64_t>        new_token_ids;
  std::optional<FinishReason> finish_reason;
  std::optional<int64_t>      stop_reason_token;
  uint32_t                    num_nans_in_logits = 0;
};

// Counters accumulated since the previous stats message (vLLM semantics:
// the frontend adds each report to monotonic metrics).
struct PrefixCacheStats {
  bool     reset              = false;
  uint64_t requests           = 0;
  uint64_t queries            = 0;
  uint64_t hits               = 0;
  uint64_t preempted_requests = 0;
  uint64_t preempted_queries  = 0;
  uint64_t preempted_hits     = 0;
};

struct SchedulerStats {
  uint64_t         num_running_reqs = 0;
  uint64_t         num_waiting_reqs = 0;
  uint64_t         step_counter     = 0;
  uint64_t         current_wave     = 0;
  double           kv_cache_usage   = 0.0;
  PrefixCacheStats prefix_cache;
};

struct OutputBatch {
  uint32_t                      engine_index = 0;
  double                        timestamp    = 0.0;
  std::vector<Output>           outputs;
  std::vector<std::string>      finished_requests;
  std::optional<SchedulerStats> scheduler_stats;
};

struct ReadyMessage {
  std::string status;
  bool        headless = true;
  bool        local    = false;
};

// msgpack codec (via nlohmann::json)

inline std::vector<uint8_t> encode_msgpack(const nlohmann::json &v) {
  return nlohmann::json::to_msgpack(v);
}

inline nlohmann::json decode_msgpack(const void *data, size_t size) {
  auto *p = static_cast<const uint8_t *>(data);
  return nlohmann::json::from_msgpack(p, p + size);
}

// msgpack encode/decode for protocol types

inline SamplingParams SamplingParams::from_msgpack(const nlohmann::json &j) {
  SamplingParams p;
  p.temperature  = j.value("temperature", p.temperature);
  p.top_p        = j.value("top_p", p.top_p);
  p.top_k        = j.value("top_k", p.top_k);
  p.seed         = j.value("seed", p.seed);
  p.max_tokens   = j.value("max_tokens", p.max_tokens);
  p.min_tokens   = j.value("min_tokens", p.min_tokens);
  p.eos_token_id = j.value("_eos_token_id", p.eos_token_id);
  // TODO(sampling): frequency/presence/repetition_penalty 与 min_tokens 只解析
  // 未实现——scheduler 侧对非默认值会告警。
  p.frequency_penalty  = j.value("frequency_penalty", p.frequency_penalty);
  p.presence_penalty   = j.value("presence_penalty", p.presence_penalty);
  p.repetition_penalty = j.value("repetition_penalty", p.repetition_penalty);
  for (const auto &id : j.value("_all_stop_token_ids", nlohmann::json::array()))
    p.all_stop_token_ids.insert(id.get<int64_t>());
  for (const auto &id : j.value("stop_token_ids", nlohmann::json::array()))
    p.stop_token_ids.insert(id.get<int64_t>());
  return p;
}

// ── mm_features（Add 请求 msgpack 索引 [2]）的 wire 解码 ──────────────────
//
// 这里全是 **wire 词汇**（frontend 的 dtype 名、处理器 kwarg 名、vLLM 的
// MmFeatureSpec 布局）；payload 本身在 utils/mm-feature.hpp。
// 只翻译**真实存在的东西**：dtype 取 frontend 实发的三个名，kwarg 取 Qwen-VL 系
// 实发的键——多写一条没人发的分支，就多一条没人测过的代码。
// **不做业务校验**（区间是否落在 prompt 内、项是否重叠、像素与 shape
// 是否自洽）： 前端是我们自己的、同仓库同版本，那类问题是我们自己代码的
// bug，真错了在前端改。 解不通（未知
// dtype/kwarg、张量不在本帧内…）不抛异常，首个原因写进出参 err， 由
// Request::decode_error 带走，scheduler 以 FinishReason::Error 收尾。

/// wire 张量 = `[dtype, shape, data]`（msgpack tuple），data 是**本帧内**的
/// msgpack ext(tag 3) 原始字节。整数 data = Python vLLM 前端的 aux frame 索引
/// （本 Rust 前端不发多帧），拿不到字节就返回 false。
struct WireTensor {
  std::string          dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

inline bool read_wire_tensor(const nlohmann::json &d, WireTensor &t) {
  if (!d.is_array() || d.size() < 3 || !d[0].is_string() || !d[1].is_array()
    || !d[2].is_binary())
    return false;
  t.dtype = d[0].get<std::string>();
  t.shape.clear();
  for (const auto &s : d[1])
    t.shape.push_back(s.get<int64_t>());
  const auto &bin = d[2].get_binary();
  t.bytes.assign(bin.begin(), bin.end());
  return true;
}

/// 像素 dtype：frontend 只发这三个（简名，不是 numpy
/// dtype.str）。**放行哪几种由 这里说了算**，所以 payload 里的 dtype
/// 一定喂得进视觉图。
inline bool mm_pixel_dtype(const std::string &name, forch::DType &out) {
  if (name == "bfloat16")
    out = forch::DType::BF16;
  else if (name == "float16")
    out = forch::DType::FP16;
  else if (name == "float32")
    out = forch::DType::FP32;
  else
    return false;
  return true;
}

/// 解码一项 mm_features（msgpack **map**，vLLM MmFeatureSpec 布局）：
///   { data: { kwarg: {data: [dtype, shape, ext3], field: [...]} },
///     modality: "image"|"video", identifier, mm_position: {offset, length,
///     is_embed}, mm_hash }
/// `data` 为 null = 该项已在 engine 侧缓存（wire 只给 identifier），合法；
/// `patches_per_*` 是分片大小（每项已按项切好），忽略；其余 kwarg 记错。
/// 解码自身可能抛（nlohmann 类型错误），由 `Request::from_msgpack` 的边界
/// try/catch 兜成 decode_error。
inline mm::MmFeatureSpec mm_feature_from_msgpack(
  const nlohmann::json &f, std::string &err) {
  auto fail = [&err](const std::string &m) {
    if (err.empty())
      err = m; // 首个原因最接近根因
  };
  mm::MmFeatureSpec spec;
  const std::string modality = f.value("modality", std::string("image"));
  spec.identifier            = f.value("identifier", std::string(""));
  if (modality == "video")
    spec.modality = 1;
  else if (modality != "image")
    fail("unsupported modality '" + modality + "'");

  if (f.contains("mm_position")) {
    const auto &pos = f["mm_position"];
    spec.mm_offset  = pos.value("offset", static_cast<int64_t>(-1));
    spec.mm_length  = pos.value("length", static_cast<int64_t>(0));
    if (pos.contains("is_embed") && pos["is_embed"].is_array()) {
      WireTensor t;
      if (read_wire_tensor(pos["is_embed"], t))
        spec.is_embed = std::move(t.bytes);
      else
        fail("is_embed: not an inline wire tensor");
    }
  }
  if (!f.contains("data") || !f["data"].is_object())
    return spec;

  for (const auto &[key, elem] : f["data"].items()) {
    // MmFieldElem = {data, field}；也容忍裸的 MmKwargValue::Tensor。
    const auto &d =
      (elem.is_object() && elem.contains("data")) ? elem["data"] : elem;
    if (d.is_null()) // 已缓存，无负载
      continue;
    if (key == "patches_per_image" || key == "patches_per_video")
      continue;
    WireTensor t;
    if (!read_wire_tensor(d, t)) {
      fail(key + ": not an inline wire tensor");
      continue;
    }
    if (key == "pixel_values" || key == "pixel_values_videos") {
      if (!mm_pixel_dtype(t.dtype, spec.pixel_dtype)) {
        fail(key + ": unsupported dtype '" + t.dtype + "'");
        continue;
      }
      spec.pixel_shape = std::move(t.shape);
      spec.pixel_bytes = std::move(t.bytes);
    } else if (key == "image_grid_thw" || key == "video_grid_thw") {
      if (t.dtype != "int64" || t.bytes.size() != 3 * sizeof(int64_t)) {
        fail(key + ": expected int64 {T,H,W}");
        continue;
      }
      spec.image_grid_thw.resize(3);
      std::memcpy(spec.image_grid_thw.data(), t.bytes.data(), t.bytes.size());
    } else if (key == "video_second_per_grid") {
      if (t.bytes.size() != sizeof(float))
        fail(key + ": expected one float32");
      else
        std::memcpy(&spec.second_per_grid_ts, t.bytes.data(), sizeof(float));
    } else {
      fail("unsupported processor kwarg '" + key + "'");
    }
  }
  return spec;
}

inline std::vector<mm::MmFeatureSpec> mm_features_from_msgpack(
  const nlohmann::json &arr, std::string &err) {
  std::vector<mm::MmFeatureSpec> out;
  for (const auto &f : arr)
    out.push_back(mm_feature_from_msgpack(f, err));
  return out;
}

inline Request Request::from_msgpack(const nlohmann::json &j) {
  // mm 项级的问题（未知 kwarg / dtype / 张量不在本帧内）仍走
  // decode_error，由 scheduler 以 Error 收尾。
  static const nlohmann::json kNull;
  auto                        at = [&](size_t i) -> const nlohmann::json                        &{
    return i < j.size() ? j[i] : kNull;
  };

  Request r;
  r.request_id = at(0).get<std::string>();
  for (const auto &id : at(1))
    r.prompt_token_ids.push_back(id.get<int64_t>());
  if (at(2).is_array()) {
    std::string mm_err;
    r.mm_features = mm_features_from_msgpack(at(2), mm_err);
    // 本层不日志（沿用原有约定，日志归 scheduler）：解不通的原因装进
    // decode_error 这个纯数据字段，scheduler 统一收尾 + 打日志。
    if (!mm_err.empty()) {
      r.decode_error = "mm_features: " + mm_err;
      r.mm_features.clear(); // 半成品不给下游
    }
  }
  r.sampling     = SamplingParams::from_msgpack(at(3));
  r.arrival_time = at(5).is_number() ? at(5).get<double>() : 0.0;
  r.cache_salt =
    at(7).is_string() ? std::optional(at(7).get<std::string>()) : std::nullopt;
  r.client_index    = at(11).is_number_unsigned() ? at(11).get<uint32_t>() : 0;
  r.current_wave    = at(12).is_number_unsigned() ? at(12).get<uint32_t>() : 0;
  r.priority        = at(13).is_number_integer() ? at(13).get<int32_t>() : 0;
  r.external_req_id = at(16).is_string()
    ? std::optional(at(16).get<std::string>())
    : std::nullopt;
  return r;
}

inline std::vector<uint8_t> encode_outputs(const OutputBatch &batch) {
  using nlohmann::json;

  // Per-output tuple: [request_id, tokens, logprobs, plogprobs, pooling,
  //   finish_reason, stop_reason, events, kv, ec, trace, prefill, experts,
  //   nans]
  auto make_output = [](const Output &o) {
    return json::array(
      {o.request_id, o.new_token_ids, nullptr, nullptr, nullptr,
        o.finish_reason ? json(static_cast<uint8_t>(*o.finish_reason))
                        : json(nullptr),
        o.stop_reason_token ? json(static_cast<uint32_t>(*o.stop_reason_token))
                            : json(nullptr),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        o.num_nans_in_logits});
  };

  auto outputs_arr = json::array();
  for (const auto &o : batch.outputs)
    outputs_arr.push_back(make_output(o));

  // Envelope: [engine_index, outputs, stats, timestamp, utility_output,
  //   finished_requests, wave_complete, start_wave]
  return encode_msgpack(json::array({
    batch.engine_index, std::move(outputs_arr),
    batch.scheduler_stats
      ? json::object(
          {{"num_running_reqs", batch.scheduler_stats->num_running_reqs},
            {"num_waiting_reqs", batch.scheduler_stats->num_waiting_reqs},
            {"num_skipped_waiting_reqs", 0},
            {"step_counter", batch.scheduler_stats->step_counter},
            {"current_wave", batch.scheduler_stats->current_wave},
            {"kv_cache_usage", batch.scheduler_stats->kv_cache_usage},
            {"prefix_cache_stats",
              {{"reset", batch.scheduler_stats->prefix_cache.reset},
                {"requests", batch.scheduler_stats->prefix_cache.requests},
                {"queries", batch.scheduler_stats->prefix_cache.queries},
                {"hits", batch.scheduler_stats->prefix_cache.hits},
                {"preempted_requests",
                  batch.scheduler_stats->prefix_cache.preempted_requests},
                {"preempted_queries",
                  batch.scheduler_stats->prefix_cache.preempted_queries},
                {"preempted_hits",
                  batch.scheduler_stats->prefix_cache.preempted_hits}}},
            {"kv_cache_eviction_events", json::array()}})
      : json(nullptr),
    batch.timestamp,
    nullptr, // utility_output
    batch.finished_requests.empty() ? json(nullptr)
                                    : json(batch.finished_requests),
    nullptr, nullptr // wave_complete, start_wave
  }));
}

inline OutputBatch make_output_batch() {
  OutputBatch batch; // engine_index already 0
  batch.timestamp = std::chrono::duration<double>(
    std::chrono::system_clock::now().time_since_epoch())
                      .count();
  return batch;
}

inline std::vector<uint8_t> encode_ready_message(const ReadyMessage &msg) {
  return encode_msgpack(
    {{"status", msg.status}, {"headless", msg.headless}, {"local", msg.local}});
}

inline std::vector<uint8_t> encode_utility_response(
  uint64_t call_id, const nlohmann::json &result) {
  using nlohmann::json;
  return encode_msgpack(json::array({0, json::array(), nullptr, 0.0,
    json::array({call_id, nullptr, json::array({nullptr, result})}), nullptr,
    nullptr, nullptr}));
}

inline auto decode_request_message(const uint8_t *f0_data, size_t f0_size,
  const uint8_t *f1_data,
  size_t         f1_size) -> std::pair<RequestType, nlohmann::json> {
  if (f0_size != 1)
    throw std::runtime_error("expected single-byte request type frame");
  return {
    static_cast<RequestType>(f0_data[0]), decode_msgpack(f1_data, f1_size)};
}

inline std::vector<std::string> decode_abort_message(
  const nlohmann::json &payload) {
  std::vector<std::string> ids;
  if (payload.is_array())
    for (const auto &id : payload)
      ids.push_back(id.get<std::string>());
  return ids;
}

} // namespace finer::server
