#include "forch/ops/recurrent_gated_delta_rule.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace forch;

// 9B 实际维度(Nk=16, Nv=32, Dk=Dv=128)的等比缩小版,kernel 对维度是通用的
// (对齐到 16),小尺寸让每个用例秒级跑完。scale = 1/sqrt(Dk)。
constexpr int64_t kNk    = 2;
constexpr int64_t kNv    = 4;
constexpr int64_t kDk    = 32;
constexpr int64_t kDv    = 32;
constexpr float   kScale = 0.17677669529663687F;

static uint16_t fp32_to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb           = (bits >> 16U) & 1U;
  const uint32_t rounding_bias = 0x7fffU + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16U);
}

static float bf16_to_fp32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16U;
  float          out  = 0.0F;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

static float quant_bf16(float value) {
  return bf16_to_fp32(fp32_to_bf16(value));
}

static bool close_enough(
  float actual, float expected, float abs_tol, float rel_tol) {
  // 实际值若为 NaN/inf 直接判失败:fabs 相减无法察觉 NaN
  if (!std::isfinite(actual)) {
    return false;
  }
  const auto diff = std::fabs(actual - expected);
  return diff <= abs_tol + rel_tol * std::fabs(expected);
}

// ---------------------------------------------------------------------------
// CPU 参考实现:与 modeling_qwen3_5.py::torch_recurrent_gated_delta_rule
// 逐 token 循环一致(状态衰减 exp(g) → kv_mem → delta → 外积更新 → out),
// 再适配 kernel 的 packed 布局约定:
//   - 逐 token 的 ssm_state_indices:每个 token 把更新后的状态写回
//     state 块 indices[t];
//   - 序列 b 的初始状态读块 indices[start + accepted - 1](无
//     num_accepted_tokens 输入时为 indices[start]);
//   - v-head h 使用 k/q head h / (Nv/Nk)(kernel 的隐式 head repeat);
//   - 内部 fp32 计算,输入先经 bf16 量化,输出 out/ref_state 也量化到 bf16。
// 输入指针全部为量化后的 fp32 值(与 kernel 实际所见一致)。
static void rgdr_reference(const float *query, const float *key,
  const float *value, const float *beta, const float *state0,
  const int32_t *seq_lens, const int32_t *indices, const float *g,
  const int32_t *accepted, int64_t batch, int64_t block_num, float scale,
  float *out, float *ref_state) {
  const auto head_repeat = static_cast<size_t>(kNv / kNk);

  // ref_state 以量化后的初始状态为底(未被写到的块保持不变)
  const auto state_elems = static_cast<size_t>(block_num * kNv * kDv * kDk);
  for (size_t i = 0; i < state_elems; ++i) {
    ref_state[i] = state0[i];
  }

  auto q_at = [&](size_t t, size_t hk, size_t d) {
    return query[(t * static_cast<size_t>(kNk) + hk) * static_cast<size_t>(kDk)
             + d]
      * scale;
  };
  auto k_at = [&](size_t t, size_t hk, size_t d) {
    return key[(t * static_cast<size_t>(kNk) + hk) * static_cast<size_t>(kDk)
      + d];
  };

  size_t start = 0;
  for (size_t bi = 0; bi < static_cast<size_t>(batch); ++bi) {
    const auto len      = static_cast<size_t>(seq_lens[bi]);
    const auto init_pos = accepted != nullptr
      ? start + static_cast<size_t>(accepted[bi]) - 1
      : start;
    const auto init_blk = static_cast<size_t>(indices[init_pos]);

    // 工作状态 (Nv, Dv, Dk)
    std::vector<float> st(static_cast<size_t>(kNv * kDv * kDk));
    for (size_t h = 0; h < static_cast<size_t>(kNv); ++h) {
      for (size_t x = 0; x < static_cast<size_t>(kDv); ++x) {
        for (size_t d = 0; d < static_cast<size_t>(kDk); ++d) {
          st[(h * static_cast<size_t>(kDv) + x) * static_cast<size_t>(kDk)
            + d] = state0[(init_blk * static_cast<size_t>(kNv) + h)
              * static_cast<size_t>(kDv * kDk)
            + x * static_cast<size_t>(kDk) + d];
        }
      }
    }

    for (size_t i = 0; i < len; ++i) {
      const auto t = start + i;

      // 1. 状态衰减 α = e^g(逐 token、逐 v-head 标量,广播到 (Dv, Dk))
      if (g != nullptr) {
        for (size_t h = 0; h < static_cast<size_t>(kNv); ++h) {
          const auto decay = std::exp(g[t * static_cast<size_t>(kNv) + h]);
          for (size_t x = 0; x < static_cast<size_t>(kDv); ++x) {
            for (size_t d = 0; d < static_cast<size_t>(kDk); ++d) {
              st[(h * static_cast<size_t>(kDv) + x) * static_cast<size_t>(kDk)
                + d] *= decay;
            }
          }
        }
      }

      // 2. kv_mem = Σ_dk state * k_t(逐 v-head/value 维)
      // 3. delta = (v_t - kv_mem) * beta_t
      std::vector<float> delta(static_cast<size_t>(kNv * kDv));
      for (size_t h = 0; h < static_cast<size_t>(kNv); ++h) {
        const auto hk = h / head_repeat;
        for (size_t x = 0; x < static_cast<size_t>(kDv); ++x) {
          float mem = 0.0F;
          for (size_t d = 0; d < static_cast<size_t>(kDk); ++d) {
            mem +=
              st[(h * static_cast<size_t>(kDv) + x) * static_cast<size_t>(kDk)
                + d]
              * k_at(t, hk, d);
          }
          delta[h * static_cast<size_t>(kDv) + x] =
            (value[(t * static_cast<size_t>(kNv) + h) * static_cast<size_t>(kDv)
               + x]
              - mem)
            * beta[t * static_cast<size_t>(kNv) + h];
        }
      }

      // 4. state += k_t ⊗ delta
      for (size_t h = 0; h < static_cast<size_t>(kNv); ++h) {
        const auto hk = h / head_repeat;
        for (size_t x = 0; x < static_cast<size_t>(kDv); ++x) {
          for (size_t d = 0; d < static_cast<size_t>(kDk); ++d) {
            st[(h * static_cast<size_t>(kDv) + x) * static_cast<size_t>(kDk)
              + d] += k_at(t, hk, d) * delta[h * static_cast<size_t>(kDv) + x];
          }
        }
      }

      // 5. 更新后的状态量化写回 indices[t] 块;out = Σ_dk state * (q_t·scale)
      const auto out_blk = static_cast<size_t>(indices[t]);
      for (size_t h = 0; h < static_cast<size_t>(kNv); ++h) {
        const auto hk = h / head_repeat;
        for (size_t x = 0; x < static_cast<size_t>(kDv); ++x) {
          float acc = 0.0F;
          for (size_t d = 0; d < static_cast<size_t>(kDk); ++d) {
            acc +=
              st[(h * static_cast<size_t>(kDv) + x) * static_cast<size_t>(kDk)
                + d]
              * q_at(t, hk, d);
          }
          out[(t * static_cast<size_t>(kNv) + h) * static_cast<size_t>(kDv)
            + x] = quant_bf16(acc);

          for (size_t d = 0; d < static_cast<size_t>(kDk); ++d) {
            ref_state[(out_blk * static_cast<size_t>(kNv) + h)
                * static_cast<size_t>(kDv * kDk)
              + x * static_cast<size_t>(kDk) + d] =
              quant_bf16(
                st[(h * static_cast<size_t>(kDv) + x) * static_cast<size_t>(kDk)
                  + d]);
          }
        }
      }
    }
    start += len;
  }
}

// ---------------------------------------------------------------------------
struct CaseSpec {
  const char          *name;
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> indices; // 逐 token 的 state block
  int64_t              block_num;
  bool                 with_g;
  bool                 with_accepted;
  std::vector<int32_t> accepted;
  uint32_t             seed;
};

struct CaseData {
  std::vector<uint16_t> query_bf, key_bf, value_bf, beta_bf, state_bf;
  std::vector<float>    query_q, key_q, value_q, beta_q, state_q;
  std::vector<float>    g;
  std::vector<int32_t>  seq_lens, indices, accepted;
  int64_t               t = 0, batch = 0;
};

/// 生成量化后的输入(bf16 位型 + fp32 量化值)。幅值刻意压小,使状态保持
/// 在 O(0.1~0.5),bf16 输出量化误差落在容差内。
static CaseData make_case_data(
  const CaseSpec &spec, const std::vector<float> &state0_f32 = {}) {
  std::mt19937                          rng(spec.seed);
  std::normal_distribution<float>       qk(0.0F, 0.4F);
  std::normal_distribution<float>       vdist(0.0F, 0.3F);
  std::normal_distribution<float>       sdist(0.0F, 0.1F);
  std::uniform_real_distribution<float> beta(0.1F, 0.8F);
  std::uniform_real_distribution<float> gdist(-1.0F, 0.0F);

  CaseData data;
  data.t =
    std::accumulate(spec.seq_lens.begin(), spec.seq_lens.end(), int64_t{0});
  data.batch    = static_cast<int64_t>(spec.seq_lens.size());
  data.seq_lens = spec.seq_lens;
  data.indices  = spec.indices;
  data.accepted = spec.accepted;

  const auto n_qk = static_cast<size_t>(data.t * kNk * kDk);
  const auto n_v  = static_cast<size_t>(data.t * kNv * kDv);
  const auto n_b  = static_cast<size_t>(data.t * kNv);
  const auto n_s  = static_cast<size_t>(spec.block_num * kNv * kDv * kDk);

  data.query_q.resize(n_qk);
  data.key_q.resize(n_qk);
  data.value_q.resize(n_v);
  data.beta_q.resize(n_b);
  data.state_q.resize(n_s);
  for (size_t i = 0; i < n_qk; ++i) {
    data.query_q[i] = quant_bf16(qk(rng));
    data.key_q[i]   = quant_bf16(qk(rng));
  }
  for (size_t i = 0; i < n_v; ++i) {
    data.value_q[i] = quant_bf16(vdist(rng));
  }
  for (size_t i = 0; i < n_b; ++i) {
    data.beta_q[i] = quant_bf16(beta(rng));
  }
  if (!state0_f32.empty()) {
    // 两步接力:初始状态用上一步 golden(已量化)状态,重写为 bf16 位型
    data.state_q = state0_f32;
  } else {
    for (size_t i = 0; i < n_s; ++i) {
      data.state_q[i] = quant_bf16(sdist(rng));
    }
  }
  auto to_bits = [](const std::vector<float> &src) {
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
      dst[i] = fp32_to_bf16(src[i]);
    }
    return dst;
  };
  data.query_bf = to_bits(data.query_q);
  data.key_bf   = to_bits(data.key_q);
  data.value_bf = to_bits(data.value_q);
  data.beta_bf  = to_bits(data.beta_q);
  data.state_bf = to_bits(data.state_q);

  if (spec.with_g) {
    data.g.resize(n_b);
    for (size_t i = 0; i < n_b; ++i) {
      data.g[i] = gdist(rng); // fp32 输入,无量化
    }
  }
  return data;
}

static Shape out_shape(int64_t t) {
  return {t, kNv, kDv};
}
static Shape state_shape(int64_t block_num) {
  return {block_num, kNv, kDv, kDk};
}

// ---------------------------------------------------------------------------
enum class Ports { Both, OutOnly, StateOnly };

static int run_rgdr_case(
  Backend &backend, const CaseSpec &spec, Ports ports = Ports::Both) {
  const auto data = make_case_data(spec);

  const auto in = [&](const char *suffix) {
    return std::string(spec.name) + suffix;
  };
  const auto query_name  = in("_query");
  const auto key_name    = in("_key");
  const auto value_name  = in("_value");
  const auto beta_name   = in("_beta");
  const auto state_name  = in("_state");
  const auto lens_name   = in("_lens");
  const auto index_name  = in("_index");
  const auto g_name      = in("_g");
  const auto accept_name = in("_accepted");

  auto query_in =
    op::create_input({data.t, kNk, kDk}, DType::BF16).set_name(query_name);
  auto key_in =
    op::create_input({data.t, kNk, kDk}, DType::BF16).set_name(key_name);
  auto value_in =
    op::create_input({data.t, kNv, kDv}, DType::BF16).set_name(value_name);
  auto beta_in =
    op::create_input({data.t, kNv}, DType::BF16).set_name(beta_name);
  auto state_in = op::create_input(state_shape(spec.block_num), DType::BF16)
                    .set_name(state_name);
  auto lens_in =
    op::create_input({data.batch}, DType::INT32).set_name(lens_name);
  auto index_in = op::create_input({data.t}, DType::INT32).set_name(index_name);

  auto rgdr =
    op::recurrent_gated_delta_rule(kScale, spec.with_g, spec.with_accepted);
  auto pair = spec.with_g
    ? (spec.with_accepted
          ? rgdr(query_in, key_in, value_in, beta_in, state_in, lens_in,
              index_in,
              op::create_input({data.t, kNv}, DType::FP32).set_name(g_name),
              op::create_input({data.batch}, DType::INT32)
                .set_name(accept_name))
          : rgdr(query_in, key_in, value_in, beta_in, state_in, lens_in,
              index_in,
              op::create_input({data.t, kNv}, DType::FP32).set_name(g_name)))
    : (spec.with_accepted ? rgdr(query_in, key_in, value_in, beta_in, state_in,
                              lens_in, index_in,
                              op::create_input({data.batch}, DType::INT32)
                                .set_name(accept_name))
                          : rgdr(query_in, key_in, value_in, beta_in, state_in,
                              lens_in, index_in));

  // 声明期元数据
  if (pair.out.shape() != out_shape(data.t)
    || pair.out.dtype() != DType::BF16) {
    std::cerr << spec.name << ": declared out metadata mismatch" << std::endl;
    return 1;
  }
  if (pair.ref_state.shape() != state_shape(spec.block_num)
    || pair.ref_state.dtype() != DType::BF16) {
    std::cerr << spec.name << ": declared ref_state metadata mismatch"
              << std::endl;
    return 1;
  }

  TensorList graph_outputs;
  const auto out_tag   = std::string(spec.name) + "_out";
  const auto state_tag = std::string(spec.name) + "_ref_state";
  if (ports == Ports::Both || ports == Ports::OutOnly) {
    graph_outputs.push_back(pair.out.set_name(out_tag));
  }
  if (ports == Ports::Both || ports == Ports::StateOnly) {
    graph_outputs.push_back(pair.ref_state.set_name(state_tag));
  }

  forch::Graph graph_ir(graph_outputs);
  graph_ir.set_name(spec.name);
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << spec.name << ": compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  // CPU 参考输出
  std::vector<float> ref_out(static_cast<size_t>(data.t * kNv * kDv));
  std::vector<float> ref_state(
    static_cast<size_t>(spec.block_num * kNv * kDv * kDk));
  rgdr_reference(data.query_q.data(), data.key_q.data(), data.value_q.data(),
    data.beta_q.data(), data.state_q.data(), data.seq_lens.data(),
    data.indices.data(), data.g.empty() ? nullptr : data.g.data(),
    spec.with_accepted ? data.accepted.data() : nullptr, data.batch,
    spec.block_num, kScale, ref_out.data(), ref_state.data());

  auto bf16_tensor = [&](const std::vector<uint16_t> &bits, const Shape &shape,
                       const std::string &name) {
    return op::load_from_buffer(
      reinterpret_cast<Buffer>(const_cast<uint16_t *>(bits.data())), shape,
      DType::BF16)
      .set_name(name);
  };
  auto i32_tensor = [&](const std::vector<int32_t> &vals, const Shape &shape,
                      const std::string &name) {
    return op::load_from_vector(std::vector<int32_t>(vals), shape)
      .set_name(name);
  };

  TensorList run_inputs;
  run_inputs.push_back(
    bf16_tensor(data.query_bf, {data.t, kNk, kDk}, query_name));
  run_inputs.push_back(bf16_tensor(data.key_bf, {data.t, kNk, kDk}, key_name));
  run_inputs.push_back(
    bf16_tensor(data.value_bf, {data.t, kNv, kDv}, value_name));
  run_inputs.push_back(bf16_tensor(data.beta_bf, {data.t, kNv}, beta_name));
  run_inputs.push_back(
    bf16_tensor(data.state_bf, state_shape(spec.block_num), state_name));
  run_inputs.push_back(i32_tensor(data.seq_lens, {data.batch}, lens_name));
  run_inputs.push_back(i32_tensor(data.indices, {data.t}, index_name));
  if (spec.with_g) {
    run_inputs.push_back(
      op::load_from_vector(std::vector<float>(data.g), {data.t, kNv})
        .set_name(g_name));
  }
  if (spec.with_accepted) {
    run_inputs.push_back(i32_tensor(data.accepted, {data.batch}, accept_name));
  }

  auto outputs = graph.run(run_inputs);

  auto check_bf16_output = [&](const std::string &tag, const Shape &shape,
                             const std::vector<float> &expected, float abs_tol,
                             float rel_tol) {
    auto out_tensor = outputs.get(tag);
    if (out_tensor.shape() != shape || out_tensor.dtype() != DType::BF16) {
      std::cerr << spec.name << ": " << tag << " runtime shape/dtype mismatch"
                << std::endl;
      return false;
    }
    const uint16_t *ptr =
      reinterpret_cast<const uint16_t *>(out_tensor.get_data());
    for (size_t i = 0; i < expected.size(); ++i) {
      if (!close_enough(bf16_to_fp32(ptr[i]), expected[i], abs_tol, rel_tol)) {
        std::cerr << spec.name << ": " << tag << " mismatch at " << i
                  << ": out=" << bf16_to_fp32(ptr[i])
                  << ", expected=" << expected[i] << std::endl;
        return false;
      }
    }
    return true;
  };

  // out(bf16):需求给相对误差 ≤ 2e-2;近零元素用 1e-3 绝对下限兜底
  if (ports == Ports::Both || ports == Ports::OutOnly) {
    if (!check_bf16_output(out_tag, out_shape(data.t), ref_out, 1e-3F, 2e-2F)) {
      return 3;
    }
  }
  // ref_state(bf16) vs fp32 参考:需求误差 ≤ 1e-3,另留 bf16 ulp(≈0.4%)
  // 的相对余量,容许量化边界上的 1 ulp 翻转
  if (ports == Ports::Both || ports == Ports::StateOnly) {
    if (!check_bf16_output(
          state_tag, state_shape(spec.block_num), ref_state, 1e-3F, 4e-3F)) {
      return 4;
    }
  }

  std::cout << "  [" << spec.name << "] ok" << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
/// 两步连续调用状态接力:同一张图跑两遍,第二遍的 state 输入直接用第一遍
/// 的设备端 ref_state 输出(GERuntime → GERuntime 的真实接力)。
static int test_two_step_chain(Backend &backend) {
  const CaseSpec spec{"two_step", /*seq_lens=*/{2}, /*indices=*/{1, 1},
    /*block_num=*/2, /*with_g=*/false, /*with_accepted=*/false,
    /*accepted=*/{}, /*seed=*/7};

  // step1:state0 中 block0 非零且不被本步写入,检验状态块隔离
  std::mt19937                    rng(6);
  std::normal_distribution<float> sdist(0.0F, 0.1F);
  std::vector<float>              state0(
    static_cast<size_t>(spec.block_num * kNv * kDv * kDk));
  for (auto &v : state0) {
    v = quant_bf16(sdist(rng));
  }

  auto step1 = make_case_data(spec, state0);

  // 建图(输入名与 run_rgdr_case 一致的简化版)
  const auto query_name = std::string("two_step_query");
  const auto key_name   = std::string("two_step_key");
  const auto value_name = std::string("two_step_value");
  const auto beta_name  = std::string("two_step_beta");
  const auto state_name = std::string("two_step_state");
  const auto lens_name  = std::string("two_step_lens");
  const auto index_name = std::string("two_step_index");

  auto query_in =
    op::create_input({step1.t, kNk, kDk}, DType::BF16).set_name(query_name);
  auto key_in =
    op::create_input({step1.t, kNk, kDk}, DType::BF16).set_name(key_name);
  auto value_in =
    op::create_input({step1.t, kNv, kDv}, DType::BF16).set_name(value_name);
  auto beta_in =
    op::create_input({step1.t, kNv}, DType::BF16).set_name(beta_name);
  auto state_in = op::create_input(state_shape(spec.block_num), DType::BF16)
                    .set_name(state_name);
  auto lens_in =
    op::create_input({step1.batch}, DType::INT32).set_name(lens_name);
  auto index_in =
    op::create_input({step1.t}, DType::INT32).set_name(index_name);

  auto pair = op::recurrent_gated_delta_rule(kScale)(
    query_in, key_in, value_in, beta_in, state_in, lens_in, index_in);
  const auto out_tag   = std::string("two_step_out");
  const auto state_tag = std::string("two_step_ref_state");
  pair.out.set_name(out_tag);
  pair.ref_state.set_name(state_tag);

  forch::Graph graph_ir({pair.out, pair.ref_state});
  graph_ir.set_name("two_step");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "two_step: compile/load failed: " << e.what() << std::endl;
    return 1;
  }

  auto bf16_tensor = [](const std::vector<uint16_t> &bits, const Shape &shape,
                       const std::string &name) {
    return op::load_from_buffer(
      reinterpret_cast<Buffer>(const_cast<uint16_t *>(bits.data())), shape,
      DType::BF16)
      .set_name(name);
  };

  auto make_inputs = [&](const CaseData &data) {
    TensorList inputs;
    inputs.push_back(
      bf16_tensor(data.query_bf, {data.t, kNk, kDk}, query_name));
    inputs.push_back(bf16_tensor(data.key_bf, {data.t, kNk, kDk}, key_name));
    inputs.push_back(
      bf16_tensor(data.value_bf, {data.t, kNv, kDv}, value_name));
    inputs.push_back(bf16_tensor(data.beta_bf, {data.t, kNv}, beta_name));
    inputs.push_back(
      bf16_tensor(data.state_bf, state_shape(spec.block_num), state_name));
    inputs.push_back(
      op::load_from_vector(std::vector<int32_t>(data.seq_lens), {data.batch})
        .set_name(lens_name));
    inputs.push_back(
      op::load_from_vector(std::vector<int32_t>(data.indices), {data.t})
        .set_name(index_name));
    return inputs;
  };

  std::vector<float> ref_out1(static_cast<size_t>(step1.t * kNv * kDv));
  std::vector<float> ref_state1(
    static_cast<size_t>(spec.block_num * kNv * kDv * kDk));
  rgdr_reference(step1.query_q.data(), step1.key_q.data(), step1.value_q.data(),
    step1.beta_q.data(), step1.state_q.data(), step1.seq_lens.data(),
    step1.indices.data(), nullptr, nullptr, step1.batch, spec.block_num, kScale,
    ref_out1.data(), ref_state1.data());

  auto outputs1 = graph.run(make_inputs(step1));

  auto check = [&](const Backend::Graph::Output &outputs,
                 const std::string &tag, const Shape &shape,
                 const std::vector<float> &expected, float abs_tol,
                 float rel_tol) {
    auto out_tensor = outputs.get(tag);
    if (out_tensor.shape() != shape || out_tensor.dtype() != DType::BF16) {
      std::cerr << "two_step: " << tag << " runtime shape/dtype mismatch"
                << std::endl;
      return false;
    }
    const uint16_t *ptr =
      reinterpret_cast<const uint16_t *>(out_tensor.get_data());
    for (size_t i = 0; i < expected.size(); ++i) {
      if (!close_enough(bf16_to_fp32(ptr[i]), expected[i], abs_tol, rel_tol)) {
        std::cerr << "two_step: " << tag << " mismatch at " << i
                  << ": out=" << bf16_to_fp32(ptr[i])
                  << ", expected=" << expected[i] << std::endl;
        return false;
      }
    }
    return true;
  };

  if (!check(outputs1, out_tag, out_shape(step1.t), ref_out1, 1e-3F, 2e-2F)) {
    return 2;
  }
  if (!check(outputs1, state_tag, state_shape(spec.block_num), ref_state1,
        1e-3F, 4e-3F)) {
    return 3;
  }

  // 状态块隔离:block0 未被 step1 写入,必须与初始状态逐位一致
  {
    const size_t    block0_elems = static_cast<size_t>(kNv * kDv * kDk);
    const uint16_t *ptr =
      reinterpret_cast<const uint16_t *>(outputs1.get(state_tag).get_data());
    for (size_t i = 0; i < block0_elems; ++i) {
      if (bf16_to_fp32(ptr[i]) != state0[i]) {
        std::cerr << "two_step: untouched state block 0 was modified at " << i
                  << std::endl;
        return 4;
      }
    }
  }

  // step2:state 输入直接取 step1 的设备端 ref_state(改名匹配输入占位)
  const CaseSpec spec2{"two_step2", /*seq_lens=*/{2}, /*indices=*/{1, 1},
    /*block_num=*/2, false, false, {}, /*seed=*/8};
  auto           step2          = make_case_data(spec2, ref_state1);
  auto           state_in_value = outputs1.get(state_tag).set_name(state_name);

  auto inputs2     = make_inputs(step2);
  bool state_found = false;
  for (auto &t : inputs2) {
    if (t.get_name() == state_name) {
      t           = state_in_value;
      state_found = true;
    }
  }
  if (!state_found) {
    std::cerr << "two_step: state input not found" << std::endl;
    return 5;
  }

  std::vector<float> ref_out2(static_cast<size_t>(step2.t * kNv * kDv));
  std::vector<float> ref_state2(
    static_cast<size_t>(spec.block_num * kNv * kDv * kDk));
  rgdr_reference(step2.query_q.data(), step2.key_q.data(), step2.value_q.data(),
    step2.beta_q.data(), step2.state_q.data(), step2.seq_lens.data(),
    step2.indices.data(), nullptr, nullptr, step2.batch, spec.block_num, kScale,
    ref_out2.data(), ref_state2.data());

  auto outputs2 = graph.run(inputs2);
  if (!check(outputs2, out_tag, out_shape(step2.t), ref_out2, 1e-3F, 2e-2F)) {
    return 6;
  }
  if (!check(outputs2, state_tag, state_shape(spec.block_num), ref_state2,
        1e-3F, 4e-3F)) {
    return 7;
  }

  std::cout << "  [two_step_chain] ok" << std::endl;
  return 0;
}

static int test_invalid_inputs() {
  // beta 形状错误:构建期(Tensor 构造)即抛 forch::exception
  auto query_in = op::create_input({1, kNk, kDk}, DType::BF16);
  auto key_in   = op::create_input({1, kNk, kDk}, DType::BF16);
  auto value_in = op::create_input({1, kNv, kDv}, DType::BF16);
  auto beta_bad = op::create_input({1, kNv + 1}, DType::BF16);
  auto state_in = op::create_input(state_shape(1), DType::BF16);
  auto lens_in  = op::create_input({1}, DType::INT32);
  auto index_in = op::create_input({1}, DType::INT32);

  bool threw = false;
  try {
    auto pair = op::recurrent_gated_delta_rule(kScale)(
      query_in, key_in, value_in, beta_bad, state_in, lens_in, index_in);
    (void)pair.out.shape();
  } catch (const std::exception &) {
    threw = true;
  }
  if (!threw) {
    std::cerr << "rgdr_invalid_beta: expected exception, got none" << std::endl;
    return 1;
  }

  // Nv 不是 Nk 的倍数
  auto value_bad = op::create_input({1, kNv + 1, kDv}, DType::BF16);
  threw          = false;
  try {
    auto pair = op::recurrent_gated_delta_rule(kScale)(
      query_in, key_in, value_bad, beta_bad, state_in, lens_in, index_in);
    (void)pair.out.shape();
  } catch (const std::exception &) {
    threw = true;
  }
  if (!threw) {
    std::cerr << "rgdr_invalid_nv: expected exception, got none" << std::endl;
    return 2;
  }

  // 输入个数与可选输入声明不符
  threw = false;
  try {
    auto pair = op::recurrent_gated_delta_rule(kScale)(query_in, key_in,
      value_in, beta_bad, state_in, lens_in, index_in, query_in);
    (void)pair.out.shape();
  } catch (const std::exception &) {
    threw = true;
  }
  if (!threw) {
    std::cerr << "rgdr_invalid_arity: expected exception, got none"
              << std::endl;
    return 3;
  }

  std::cout << "  [rgdr_invalid_inputs] ok" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  // 必测:B=1 L=1
  if (int rc = run_rgdr_case(backend,
        {"b1_l1", /*seq_lens=*/{1}, /*indices=*/{0}, /*block_num=*/1, false,
          false, {}, /*seed=*/1});
      rc != 0) {
    return rc;
  }

  // 必测:B=4 变长 {1,2,3,1}(常规 varlen:每序列一个 block)
  if (int rc = run_rgdr_case(backend,
        {"b4_varlen", {1, 2, 3, 1}, {0, 1, 1, 2, 2, 2, 3}, 4, false, false, {},
          2});
      rc != 0) {
    return rc;
  }

  // 必测:indices 乱序(非恒等映射,同一 block 被多个 token 命中)
  if (int rc = run_rgdr_case(backend,
        {"shuffled_indices", {1, 2, 3, 1}, {5, 2, 2, 0, 4, 4, 1}, 6, false,
          false, {}, 3});
      rc != 0) {
    return rc;
  }

  // 必测:g 非零衰减
  if (int rc = run_rgdr_case(backend,
        {"g_decay", {1, 2, 3, 1}, {0, 1, 1, 2, 2, 2, 3}, 4, true, false, {},
          4});
      rc != 0) {
    return rc;
  }

  // num_accepted_tokens(MTP 语义)+ g 同时在场
  if (int rc = run_rgdr_case(backend,
        {"num_accepted", {2, 1, 3}, {3, 3, 1, 5, 2, 2}, 6, true, true,
          {2, 1, 3}, 5});
      rc != 0) {
    return rc;
  }

  // 单端口消费(端口同源于一个共享节点)
  if (int rc = run_rgdr_case(backend,
        {"out_only", {1}, {0}, 1, false, false, {}, 11}, Ports::OutOnly);
      rc != 0) {
    return rc;
  }
  if (int rc = run_rgdr_case(backend,
        {"state_only", {1}, {0}, 1, false, false, {}, 12}, Ports::StateOnly);
      rc != 0) {
    return rc;
  }

  // 必测:两步连续调用状态接力
  if (int rc = test_two_step_chain(backend); rc != 0) {
    return rc;
  }

  if (int rc = test_invalid_inputs(); rc != 0) {
    return rc;
  }

  std::cout << "RecurrentGatedDeltaRule tests passed." << std::endl;
  return 0;
}
