#pragma once

#include "forch/exception.hpp"
#include "forch/ops/function.hpp"

namespace forch::op {

struct RotaryPositionEmbedding : public Function {
  enum class Mode : int64_t {
    Half       = 0,
    Interleave = 1,
  };

  Mode mode;

  explicit RotaryPositionEmbedding(Mode mode = Mode::Half) : mode(mode) {}

  /**
   * x: [S, N, D]
   * cos, sin: [S, N, D] or [S, 1, D]
   * output: [S, N, D]
   */
  Shape shape(const TensorList &input) const override {
    const auto &x   = input[0];
    const auto &cos = input[1];
    const auto &sin = input[2];

    if (x.dim() != 3 || cos.dim() != 3 || sin.dim() != 3) {
      throw exception(
        "rotary_position_embedding: x, cos and sin must have dim == 3");
    }

    if (cos.shape() != sin.shape()) {
      throw exception("rotary_position_embedding: cos and sin must have the "
                      "same shape");
    }

    auto S = x.shape()[0];
    auto N = x.shape()[1];
    auto D = x.shape()[2];

    if (cos.shape()[0] != S) {
      throw exception("rotary_position_embedding: cos and sin must have the "
                      "same sequence length as x");
    }
    if (cos.shape()[1] != 1 && cos.shape()[1] != N) {
      throw exception("rotary_position_embedding: cos and sin must have the "
                      "same batch size as x or batch size of 1");
    }
    // partial rotary（Qwen3.5：rotary_dim=64 < head_dim=256）kernel 不支持窄表
    // （2026-09-10 探针实测：tiling 的 JudgeLayoutByShape 要求 cos/sin 与 x
    // 可广播，见 tests/rope_partial.cpp case3）。partial 走
    // slice → 全宽 kRot 的 rope → cat 组合，已由同测试 case2 验证。
    if (cos.shape()[2] != D) {
      throw exception("rotary_position_embedding: cos and sin must have the "
                      "same hidden size as x");
    }

    return x.shape();
  }
};

inline auto rotary_position_embedding(
  RotaryPositionEmbedding::Mode mode = RotaryPositionEmbedding::Mode::Half) {
  return TensorBuildCurry<3>{std::make_unique<RotaryPositionEmbedding>(mode)};
}

} // namespace forch::op