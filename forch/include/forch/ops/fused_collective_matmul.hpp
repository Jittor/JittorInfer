#pragma once

#include "forch/exception.hpp"
#include "forch/ops/collective.hpp"
#include "forch/ops/function.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace forch::op {

/**
 * A matmul and one collective in a *single* node, so the communication
 * overlaps the compute instead of following it. Named after upstream
 * PyTorch's `torch.ops.symm_mem.fused_all_gather_matmul` family; the two with
 * no upstream counterpart follow the same wording.
 *
 * The constraints below come from the op protos, not from API taste:
 *
 * - x1 cannot be transposed, so only transpose_b exists.
 * - One output per op. A second output class would make the matmul run twice,
 *   so take y and recompute: norm_out = rms_norm(eps)(y, gamma) is equivalent
 *   because y is post-residual-add, and gather_out = all_gather(x1) is only
 *   worth materializing when asked for.
 *
 * Which dtypes a backend can actually fuse is the backend's business and is
 * checked there, not here.
 */

/**
 * x1 is (..., K), 2D or 3D, and x2 is 2D, (N,K) when transpose_b else
 * (K,N). The output keeps x1's leading dims and replaces the last one with N,
 * so a 3D x1 gives a 3D output.
 */
inline Shape fused_collective_matmul_shape(
  const char *who, const Tensor &x1, const Tensor &x2, bool transpose_b) {
  auto s1 = x1.shape();
  auto s2 = x2.shape();

  if (s1.size() != 2 && s1.size() != 3) {
    throw exception(std::string(who)
      + ": x1 must be 2D (m,k) or 3D (b,s,k), got rank "
      + std::to_string(s1.size()));
  }
  if (s2.size() != 2) {
    throw exception(std::string(who) + ": x2 must be 2D, got rank "
      + std::to_string(s2.size()));
  }

  const int64_t k1 = s1.back();
  const int64_t k2 = transpose_b ? s2[1] : s2[0];
  const int64_t n  = transpose_b ? s2[0] : s2[1];

  if (k1 != k2) {
    throw exception(std::string(who) + ": x1's last dim (" + std::to_string(k1)
      + ") must equal x2's K (" + std::to_string(k2) + ")");
  }

  s1.back() = n;
  return s1;
}

/** y = all_reduce(x1 @ x2), optionally plus bias: inputs (x1, x2[, bias]). */
struct FusedMatmulAllReduce : public Function {
  ProcessGroup group;
  ReduceOp     reduce_op;
  bool         transpose_b;

  FusedMatmulAllReduce(ProcessGroup group, ReduceOp reduce_op, bool transpose_b)
      : group(group), reduce_op(reduce_op), transpose_b(transpose_b) {}

  Shape shape(const TensorList &input) const override {
    if (input.size() != 2 && input.size() != 3) {
      throw exception(
        "fused_matmul_all_reduce: expects 2 or 3 inputs (x1, x2[, bias])");
    }

    auto out = fused_collective_matmul_shape(
      "fused_matmul_all_reduce", input[0], input[1], transpose_b);

    if (input.size() == 3) {
      const auto bs = input[2].shape();
      if (bs.size() != 1 || bs[0] != out.back()) {
        throw exception("fused_matmul_all_reduce: bias must be 1D of size "
          + std::to_string(out.back()));
      }
    }
    return out;
  }
};

/**
 * y = all_reduce(x1 @ x2) + residual, where the proto also returns norm_out =
 * rms_norm(y, gamma). Inputs are fixed at (x1, x2, residual, gamma).
 *
 * residual must be 3D (b,s,n) and y takes residual's shape, so even a 2D x1
 * produces a 3D output. norm_out is not exposed; recompute it as
 * rms_norm(epsilon)(y, gamma).
 */
struct FusedMatmulAllReduceAddRmsNorm : public Function {
  ProcessGroup group;
  ReduceOp     reduce_op;
  bool         transpose_b;
  float        epsilon;

  FusedMatmulAllReduceAddRmsNorm(
    ProcessGroup group, ReduceOp reduce_op, bool transpose_b, float epsilon)
      : group(group), reduce_op(reduce_op), transpose_b(transpose_b),
        epsilon(epsilon) {}

  Shape shape(const TensorList &input) const override {
    if (input.size() != 4) {
      throw exception("fused_matmul_all_reduce_add_rms_norm: expects 4 inputs "
                      "(x1, x2, residual, gamma)");
    }
    if (!(epsilon > 0.0F && epsilon < 1.0F)) {
      throw exception(
        "fused_matmul_all_reduce_add_rms_norm: epsilon must be in (0,1), got "
        + std::to_string(epsilon));
    }

    auto mm = fused_collective_matmul_shape(
      "fused_matmul_all_reduce_add_rms_norm", input[0], input[1], transpose_b);
    const int64_t n = mm.back();

    const auto &res   = input[2];
    const auto &gamma = input[3];

    if (res.dim() != 3) {
      throw exception("fused_matmul_all_reduce_add_rms_norm: residual must be "
                      "3D (b,s,n), got rank "
        + std::to_string(res.dim()));
    }
    if (res.shape()[2] != n) {
      throw exception(
        "fused_matmul_all_reduce_add_rms_norm: residual's last dim ("
        + std::to_string(res.shape()[2]) + ") must equal N ("
        + std::to_string(n) + ")");
    }
    if (gamma.dim() != 1 || gamma.shape()[0] != n) {
      throw exception(
        "fused_matmul_all_reduce_add_rms_norm: gamma must be 1D of size "
        + std::to_string(n));
    }

    // a 2D x1 counts as (b*s, k) here.
    int64_t lead = 1;
    for (size_t i = 0; i + 1 < mm.size(); ++i) {
      lead *= mm[i];
    }
    if (lead != res.shape()[0] * res.shape()[1]) {
      throw exception("fused_matmul_all_reduce_add_rms_norm: x1's rows ("
        + std::to_string(lead) + ") must equal residual's b*s ("
        + std::to_string(res.shape()[0] * res.shape()[1]) + ")");
    }

    return res.shape();
  }
};

/**
 * y = all_gather(x1) @ x2. x1 must be 2D (M,K), the most the proto allows,
 * and the output is (M*world_size, N). gather_out is not exposed; see above.
 */
struct FusedAllGatherMatmul : public Function {
  ProcessGroup group;
  bool         transpose_b;

  FusedAllGatherMatmul(ProcessGroup group, bool transpose_b)
      : group(group), transpose_b(transpose_b) {}

  Shape shape(const TensorList &input) const override {
    if (input.size() != 2) {
      throw exception("fused_all_gather_matmul: expects 2 inputs (x1, x2)");
    }
    if (input[0].dim() != 2) {
      throw exception("fused_all_gather_matmul: x1 must be 2D (M,K), got rank "
        + std::to_string(input[0].dim()));
    }

    auto out = fused_collective_matmul_shape(
      "fused_all_gather_matmul", input[0], input[1], transpose_b);
    out[0] *= group.world_size();
    return out;
  }
};

/**
 * y = reduce_scatter(x1 @ x2): matmul, then keep this rank's block of dim 0.
 * x1 must be 2D (M,K) with M divisible by world_size; output is
 * (M/world_size, N).
 */
struct FusedMatmulReduceScatter : public Function {
  ProcessGroup group;
  ReduceOp     reduce_op;
  bool         transpose_b;

  FusedMatmulReduceScatter(
    ProcessGroup group, ReduceOp reduce_op, bool transpose_b)
      : group(group), reduce_op(reduce_op), transpose_b(transpose_b) {}

  Shape shape(const TensorList &input) const override {
    if (input.size() != 2) {
      throw exception("fused_matmul_reduce_scatter: expects 2 inputs (x1, x2)");
    }
    if (input[0].dim() != 2) {
      throw exception(
        "fused_matmul_reduce_scatter: x1 must be 2D (M,K), got rank "
        + std::to_string(input[0].dim()));
    }

    auto out = fused_collective_matmul_shape(
      "fused_matmul_reduce_scatter", input[0], input[1], transpose_b);

    if (out[0] % group.world_size() != 0) {
      throw exception("fused_matmul_reduce_scatter: M ("
        + std::to_string(out[0]) + ") must be divisible by world_size ("
        + std::to_string(group.world_size()) + ")");
    }
    out[0] /= group.world_size();
    return out;
  }
};

// transpose_b defaults to true, opposite of op::mat_mul: TP weights are stored
// (N,K), so these ops compute x @ W^T.

inline Tensor fused_matmul_all_reduce(const Tensor &x1, const Tensor &x2,
  ProcessGroup group, ReduceOp reduce_op = ReduceOp::Sum,
  bool transpose_b = true) {
  return TensorBuildCurry<2>{std::make_unique<FusedMatmulAllReduce>(
    group, reduce_op, transpose_b)}(x1, x2);
}

inline Tensor fused_matmul_all_reduce(const Tensor &x1, const Tensor &x2,
  const Tensor &bias, ProcessGroup group, ReduceOp reduce_op = ReduceOp::Sum,
  bool transpose_b = true) {
  return TensorBuildCurry<3>{std::make_unique<FusedMatmulAllReduce>(
    group, reduce_op, transpose_b)}(x1, x2, bias);
}

inline Tensor fused_matmul_all_reduce_add_rms_norm(const Tensor &x1,
  const Tensor &x2, const Tensor &residual, const Tensor &gamma,
  ProcessGroup group, float epsilon = 1e-6F, ReduceOp reduce_op = ReduceOp::Sum,
  bool transpose_b = true) {
  return TensorBuildCurry<4>{std::make_unique<FusedMatmulAllReduceAddRmsNorm>(
    group, reduce_op, transpose_b, epsilon)}(x1, x2, residual, gamma);
}

inline Tensor fused_all_gather_matmul(const Tensor &x1, const Tensor &x2,
  ProcessGroup group, bool transpose_b = true) {
  return TensorBuildCurry<2>{
    std::make_unique<FusedAllGatherMatmul>(group, transpose_b)}(x1, x2);
}

inline Tensor fused_matmul_reduce_scatter(const Tensor &x1, const Tensor &x2,
  ProcessGroup group, ReduceOp reduce_op = ReduceOp::Sum,
  bool transpose_b = true) {
  return TensorBuildCurry<2>{std::make_unique<FusedMatmulReduceScatter>(
    group, reduce_op, transpose_b)}(x1, x2);
}

} // namespace forch::op
