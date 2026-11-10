#include "forch/ops/fused_infer_attention.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/batched_mat_mul.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/permute.hpp"
#include "forch/ops/softmax.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace forch;

static size_t numel(const Shape &shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

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

static std::vector<uint16_t> to_bf16(const std::vector<float> &data) {
  std::vector<uint16_t> out;
  out.reserve(data.size());
  for (float value : data) {
    out.push_back(fp32_to_bf16(value));
  }
  return out;
}

static bool almost_equal_bf16(
  const uint16_t *a, const uint16_t *b, size_t n, float tol = 1.5e-1F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(bf16_to_fp32(a[i]) - bf16_to_fp32(b[i])) > tol) {
      return false;
    }
  }
  return true;
}

int run_attention_case(Backend &backend, const char *case_name,
  const Shape &query_shape, const Shape &kv_shape,
  const std::vector<float> &query_data_f32,
  const std::vector<float> &key_data_f32,
  const std::vector<float> &value_data_f32, int64_t num_heads) {
  auto query_name = std::string(case_name) + "_query";
  auto key_name   = std::string(case_name) + "_key";
  auto value_name = std::string(case_name) + "_value";
  auto fused_name = std::string(case_name) + "_fused";
  auto ref_name   = std::string(case_name) + "_ref";

  op::FusedInferAttention::Options options(num_heads);
  options.input_layout = op::FusedInferAttention::InputLayout::BNSD_BSND;
  options.scale        = 1.0F;

  auto query = op::create_input(query_shape, DType::BF16).set_name(query_name);
  auto key   = op::create_input(kv_shape, DType::BF16).set_name(key_name);
  auto value = op::create_input(kv_shape, DType::BF16).set_name(value_name);

  auto fused = op::fused_infer_attention(options)(query, key, value);
  fused.set_name(fused_name);

  auto scores = op::batched_mat_mul(/*transpose_a=*/false,
    /*transpose_b=*/true)(query, key);
  auto probs  = op::softmax(3)(scores);
  auto ref    = op::batched_mat_mul()(probs, value);
  ref         = op::permute({0, 2, 1, 3})(ref);
  ref.set_name(ref_name);

  Shape output_shape = {
    query_shape[0], query_shape[2], query_shape[1], query_shape[3]};
  if (fused.shape() != output_shape || ref.shape() != output_shape) {
    std::cerr << case_name << ": output shape mismatch" << std::endl;
    return 1;
  }

  forch::Graph graph_ir({fused, ref});
  graph_ir.set_name(case_name);
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << case_name << ": compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  auto query_data = to_bf16(query_data_f32);
  auto key_data   = to_bf16(key_data_f32);
  auto value_data = to_bf16(value_data_f32);

  auto query_tensor = op::load_from_buffer(
    reinterpret_cast<Buffer>(query_data.data()), query_shape, DType::BF16)
                        .set_name(query_name);
  auto key_tensor = op::load_from_buffer(
    reinterpret_cast<Buffer>(key_data.data()), kv_shape, DType::BF16)
                      .set_name(key_name);
  auto value_tensor = op::load_from_buffer(
    reinterpret_cast<Buffer>(value_data.data()), kv_shape, DType::BF16)
                        .set_name(value_name);

  auto outputs = graph.run({query_tensor, key_tensor, value_tensor});
  if (outputs.size() != 2) {
    std::cerr << case_name << ": output count mismatch" << std::endl;
    return 3;
  }

  auto fused_tensor = outputs.get(fused_name);
  auto ref_tensor   = outputs.get(ref_name);
  if (fused_tensor.shape() != output_shape
    || ref_tensor.shape() != output_shape) {
    std::cerr << case_name << ": runtime shape mismatch" << std::endl;
    return 4;
  }

  const auto *fused_ptr =
    reinterpret_cast<const uint16_t *>(fused_tensor.get_data());
  const auto *ref_ptr =
    reinterpret_cast<const uint16_t *>(ref_tensor.get_data());
  const auto n = numel(output_shape);
  if (!almost_equal_bf16(fused_ptr, ref_ptr, n)) {
    std::cerr << case_name << ": values differ" << std::endl;
    for (size_t i = 0; i < n; ++i) {
      const auto fused_value = bf16_to_fp32(fused_ptr[i]);
      const auto ref_value   = bf16_to_fp32(ref_ptr[i]);
      if (std::fabs(fused_value - ref_value) > 1.5e-1F) {
        std::cerr << "first mismatch at " << i << ": fused=" << fused_value
                  << ", ref=" << ref_value << std::endl;
        break;
      }
    }
    return 5;
  }

  return 0;
}

int test_fused_infer_attention_matches_existing_ops(Backend &backend) {
  {
    Shape query_shape = {1, 1, 2, 16};
    Shape kv_shape    = {1, 1, 2, 16};

    std::vector<float> query_data = {
      0.20F,
      -0.10F,
      0.30F,
      0.00F,
      0.50F,
      -0.40F,
      0.10F,
      0.60F,
      -0.20F,
      0.70F,
      -0.30F,
      0.40F,
      0.80F,
      -0.50F,
      0.90F,
      -0.60F,
      0.10F,
      0.20F,
      -0.20F,
      0.30F,
      -0.30F,
      0.40F,
      -0.40F,
      0.50F,
      0.60F,
      -0.50F,
      0.70F,
      -0.60F,
      0.80F,
      -0.70F,
      0.90F,
      -0.80F,
    };
    std::vector<float> key_data = {
      0.30F,
      0.00F,
      -0.20F,
      0.40F,
      -0.10F,
      0.50F,
      -0.30F,
      0.60F,
      -0.40F,
      0.70F,
      -0.50F,
      0.80F,
      -0.60F,
      0.90F,
      -0.70F,
      1.00F,
      -0.10F,
      0.20F,
      0.10F,
      -0.20F,
      0.30F,
      -0.30F,
      0.40F,
      -0.40F,
      0.50F,
      -0.50F,
      0.60F,
      -0.60F,
      0.70F,
      -0.70F,
      0.80F,
      -0.80F,
    };
    std::vector<float> value_data = {
      1.00F,
      0.50F,
      -0.50F,
      0.25F,
      -0.25F,
      0.75F,
      -0.75F,
      0.10F,
      -0.10F,
      0.20F,
      -0.20F,
      0.30F,
      -0.30F,
      0.40F,
      -0.40F,
      0.60F,
      -0.60F,
      0.55F,
      -0.55F,
      0.45F,
      -0.45F,
      0.35F,
      -0.35F,
      0.15F,
      -0.15F,
      0.05F,
      -0.05F,
      0.65F,
      -0.65F,
      0.85F,
      -0.85F,
      0.95F,
    };

    int r = run_attention_case(backend, "fia_bnsd_prompt", query_shape,
      kv_shape, query_data, key_data, value_data, /*num_heads=*/1);
    if (r != 0) {
      return r;
    }
  }

  {
    Shape query_shape = {1, 1, 2, 16};
    Shape kv_shape    = {1, 1, 3, 16};

    std::vector<float> query_data = {
      0.10F,
      -0.20F,
      0.30F,
      -0.40F,
      0.50F,
      -0.60F,
      0.70F,
      -0.80F,
      0.90F,
      -1.00F,
      0.80F,
      -0.70F,
      0.60F,
      -0.50F,
      0.40F,
      -0.30F,
      -0.15F,
      0.25F,
      -0.35F,
      0.45F,
      -0.55F,
      0.65F,
      -0.75F,
      0.85F,
      -0.95F,
      0.75F,
      -0.65F,
      0.55F,
      -0.45F,
      0.35F,
      -0.25F,
      0.15F,
    };
    std::vector<float> key_data = {
      0.20F,
      0.10F,
      0.00F,
      -0.10F,
      -0.20F,
      0.30F,
      0.40F,
      -0.30F,
      -0.40F,
      0.50F,
      0.60F,
      -0.50F,
      -0.60F,
      0.70F,
      0.80F,
      -0.70F,
      0.30F,
      -0.20F,
      0.10F,
      0.00F,
      -0.10F,
      0.20F,
      -0.30F,
      0.40F,
      -0.50F,
      0.60F,
      -0.70F,
      0.80F,
      -0.90F,
      1.00F,
      -0.80F,
      0.70F,
      -0.20F,
      0.30F,
      -0.40F,
      0.50F,
      -0.60F,
      0.70F,
      -0.80F,
      0.90F,
      -1.00F,
      0.80F,
      -0.70F,
      0.60F,
      -0.50F,
      0.40F,
      -0.30F,
      0.20F,
    };
    std::vector<float> value_data = {
      0.10F,
      0.20F,
      0.30F,
      0.40F,
      0.50F,
      0.60F,
      0.70F,
      0.80F,
      0.90F,
      1.00F,
      0.90F,
      0.80F,
      0.70F,
      0.60F,
      0.50F,
      0.40F,
      -0.10F,
      -0.20F,
      -0.30F,
      -0.40F,
      -0.50F,
      -0.60F,
      -0.70F,
      -0.80F,
      -0.90F,
      -1.00F,
      -0.90F,
      -0.80F,
      -0.70F,
      -0.60F,
      -0.50F,
      -0.40F,
      0.05F,
      -0.05F,
      0.15F,
      -0.15F,
      0.25F,
      -0.25F,
      0.35F,
      -0.35F,
      0.45F,
      -0.45F,
      0.55F,
      -0.55F,
      0.65F,
      -0.65F,
      0.75F,
      -0.75F,
    };

    int r = run_attention_case(backend, "fia_bnsd_bsnd_kv_longer", query_shape,
      kv_shape, query_data, key_data, value_data, /*num_heads=*/1);
    if (r != 0) {
      return r;
    }
  }

  return 0;
}

static std::string shape_to_string(const Shape &shape) {
  std::string out = "{";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out += ",";
    }
    out += std::to_string(shape[i]);
  }
  out += "}";
  return out;
}

static std::vector<uint16_t> random_bf16_data(
  const Shape &shape, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<uint16_t>                 data;
  data.reserve(numel(shape));
  for (size_t i = 0; i < numel(shape); ++i) {
    data.push_back(fp32_to_bf16(dist(rng)));
  }
  return data;
}

static Tensor make_bf16_runtime_input(
  std::vector<uint16_t> &data, const Shape &shape, const std::string &name) {
  return op::load_from_buffer(
    reinterpret_cast<Buffer>(data.data()), shape, DType::BF16)
    .set_name(name);
}

static auto build_fused_benchmark_graph(Backend &backend,
  const Shape &query_shape, const Shape &kv_shape, int64_t num_heads) {
  auto query =
    op::create_input(query_shape, DType::BF16).set_name("bench_query");
  auto key   = op::create_input(kv_shape, DType::BF16).set_name("bench_key");
  auto value = op::create_input(kv_shape, DType::BF16).set_name("bench_value");

  op::FusedInferAttention::Options options(num_heads);
  options.input_layout = op::FusedInferAttention::InputLayout::BNSD_BSND;
  options.scale        = 1.0F;

  auto out = op::fused_infer_attention(options)(query, key, value);
  out.set_name("bench_fused_out");

  forch::Graph graph_ir({out});
  graph_ir.set_name("fia_bf16_bnsd_bsnd_benchmark_fused");
  auto graph = backend.build(graph_ir);
  graph.compile();
  graph.load();
  return graph;
}

static auto build_unfused_benchmark_graph(
  Backend &backend, const Shape &query_shape, const Shape &kv_shape) {
  auto query =
    op::create_input(query_shape, DType::BF16).set_name("bench_query");
  auto key   = op::create_input(kv_shape, DType::BF16).set_name("bench_key");
  auto value = op::create_input(kv_shape, DType::BF16).set_name("bench_value");

  auto scores = op::batched_mat_mul(/*transpose_a=*/false,
    /*transpose_b=*/true)(query, key);
  auto probs  = op::softmax(3)(scores);
  auto out    = op::batched_mat_mul()(probs, value);
  out         = op::permute({0, 2, 1, 3})(out);
  out.set_name("bench_unfused_out");

  forch::Graph graph_ir({out});
  graph_ir.set_name("fia_bf16_bnsd_bsnd_benchmark_unfused");
  auto graph = backend.build(graph_ir);
  graph.compile();
  graph.load();
  return graph;
}

static double benchmark_graph_ms(
  Backend::Graph &graph, const TensorList &inputs, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    auto outputs = graph.run(inputs);
    if (outputs.empty()) {
      throw exception("fused_infer_attention benchmark produced no output");
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    auto outputs = graph.run(inputs);
    if (outputs.empty()) {
      throw exception("fused_infer_attention benchmark produced no output");
    }
  }
  const auto end = std::chrono::steady_clock::now();

  const std::chrono::duration<double, std::milli> elapsed = end - start;
  return elapsed.count() / static_cast<double>(iterations);
}

int benchmark_fused_infer_attention_bf16_bnsd_bsnd(Backend &backend) {
  constexpr int     warmup     = 10;
  constexpr int     iterations = 100;
  constexpr int64_t batch      = 16;
  constexpr int64_t num_heads  = 16;
  constexpr int64_t query_seq  = 64;
  constexpr int64_t kv_seq     = 4096;
  constexpr int64_t head_dim   = 64;

  const Shape query_shape = {batch, num_heads, query_seq, head_dim};
  const Shape kv_shape    = {batch, num_heads, kv_seq, head_dim};

  try {
    std::mt19937 rng(20240704U);
    auto         query_data = random_bf16_data(query_shape, rng);
    auto         key_data   = random_bf16_data(kv_shape, rng);
    auto         value_data = random_bf16_data(kv_shape, rng);

    auto query_tensor =
      make_bf16_runtime_input(query_data, query_shape, "bench_query");
    auto key_tensor = make_bf16_runtime_input(key_data, kv_shape, "bench_key");
    auto value_tensor =
      make_bf16_runtime_input(value_data, kv_shape, "bench_value");
    TensorList inputs = {query_tensor, key_tensor, value_tensor};

    auto fused_graph =
      build_fused_benchmark_graph(backend, query_shape, kv_shape, num_heads);
    auto unfused_graph =
      build_unfused_benchmark_graph(backend, query_shape, kv_shape);

    const double fused_ms =
      benchmark_graph_ms(fused_graph, inputs, warmup, iterations);
    const double unfused_ms =
      benchmark_graph_ms(unfused_graph, inputs, warmup, iterations);

    std::cout << std::fixed << std::setprecision(3)
              << "FusedInferAttention BF16 BNSD_BSND benchmark: query="
              << shape_to_string(query_shape)
              << ", key/value=" << shape_to_string(kv_shape)
              << ", warmup=" << warmup << ", iterations=" << iterations << "\n"
              << "  fused: " << fused_ms << " ms/run\n"
              << "  batched_mat_mul + softmax + batched_mat_mul + permute: "
              << unfused_ms << " ms/run\n";
    if (fused_ms > 0.0) {
      std::cout << "  speedup: " << (unfused_ms / fused_ms) << "x\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "fused_infer_attention benchmark failed: " << e.what()
              << std::endl;
    return 1;
  }

  return 0;
}

int test_bnsd_to_bsnd_shape() {
  op::FusedInferAttention::Options options(4);
  options.input_layout        = op::FusedInferAttention::InputLayout::BNSD_BSND;
  options.num_key_value_heads = 2;

  auto query = op::create_input({2, 4, 3, 16}, DType::BF16);
  auto key   = op::create_input({2, 2, 5, 16}, DType::BF16);
  auto value = op::create_input({2, 2, 5, 16}, DType::BF16);

  auto out = op::fused_infer_attention(options)(query, key, value);
  if (out.shape() != Shape{2, 3, 4, 16}) {
    std::cerr << "fused_infer_attention BNSD_BSND shape mismatch" << std::endl;
    return 1;
  }
  if (out.dtype() != DType::BF16) {
    std::cerr << "fused_infer_attention BNSD_BSND dtype mismatch" << std::endl;
    return 2;
  }
  return 0;
}

int test_optional_inputs_shape() {
  op::FusedInferAttention::Options options(4);
  options.input_layout = op::FusedInferAttention::InputLayout::BSND;
  options.output_dtype = DType::BF16;

  auto query = op::create_input({2, 3, 4, 16}, DType::FP16);
  auto key   = op::create_input({2, 5, 4, 16}, DType::FP16);
  auto value = op::create_input({2, 5, 4, 16}, DType::FP16);
  auto mask  = op::create_input({3, 5}, DType::INT32);
  auto seq   = op::create_input({2}, DType::INT64);

  op::FusedInferAttention::OptionalInputs optional;
  optional.atten_mask            = mask;
  optional.actual_seq_lengths    = seq;
  optional.actual_seq_lengths_kv = seq;
  optional.query_padding_size    = seq;
  optional.kv_padding_size       = seq;

  auto out = op::fused_infer_attention(options)(query, key, value, optional);
  if (out.shape() != query.shape()) {
    std::cerr << "fused_infer_attention optional shape mismatch" << std::endl;
    return 1;
  }
  if (out.dtype() != DType::BF16) {
    std::cerr << "fused_infer_attention optional dtype mismatch" << std::endl;
    return 2;
  }
  return 0;
}

int test_dynamic_key_value_shape() {
  auto query  = op::create_input({2, 3, 16}, DType::FP16);
  auto key0   = op::create_input({1, 5, 16}, DType::FP16);
  auto key1   = op::create_input({1, 7, 16}, DType::FP16);
  auto value0 = op::create_input({1, 5, 16}, DType::FP16);
  auto value1 = op::create_input({1, 7, 16}, DType::FP16);

  auto out = op::fused_infer_attention(4)(
    query, TensorList{key0, key1}, TensorList{value0, value1});
  if (out.shape() != query.shape()) {
    std::cerr << "fused_infer_attention dynamic shape mismatch" << std::endl;
    return 1;
  }
  return 0;
}

int test_dynamic_key_value_batch_mismatch() {
  auto query  = op::create_input({3, 3, 16}, DType::FP16);
  auto key0   = op::create_input({1, 5, 16}, DType::FP16);
  auto key1   = op::create_input({1, 7, 16}, DType::FP16);
  auto value0 = op::create_input({1, 5, 16}, DType::FP16);
  auto value1 = op::create_input({1, 7, 16}, DType::FP16);

  try {
    (void)op::fused_infer_attention(4)(
      query, TensorList{key0, key1}, TensorList{value0, value1});
  } catch (const std::exception &) {
    return 0;
  }

  std::cerr << "fused_infer_attention dynamic batch mismatch was accepted"
            << std::endl;
  return 1;
}

int test_dynamic_key_value_requires_batch_one() {
  auto query  = op::create_input({2, 3, 16}, DType::FP16);
  auto key0   = op::create_input({2, 5, 16}, DType::FP16);
  auto key1   = op::create_input({1, 7, 16}, DType::FP16);
  auto value0 = op::create_input({2, 5, 16}, DType::FP16);
  auto value1 = op::create_input({1, 7, 16}, DType::FP16);

  try {
    (void)op::fused_infer_attention(4)(
      query, TensorList{key0, key1}, TensorList{value0, value1});
  } catch (const std::exception &) {
    return 0;
  }

  std::cerr << "fused_infer_attention dynamic batch > 1 was accepted"
            << std::endl;
  return 1;
}

int test_decode_mla_option() {
  using FIA = op::FusedInferAttention;
  FIA::Options options(16);
  if (options.decode_mla) {
    std::cerr << "decode_mla must default to false" << std::endl;
    return 1;
  }
  options.decode_mla          = true;
  options.input_layout        = FIA::InputLayout::BSND;
  options.num_key_value_heads = 1;
  auto                query   = op::create_input({2, 1, 16, 512}, DType::BF16);
  auto                key     = op::create_input({2, 128, 1, 512}, DType::BF16);
  FIA::OptionalInputs extra;
  extra.query_rope            = op::create_input({2, 1, 16, 64}, DType::BF16);
  extra.key_rope              = op::create_input({2, 128, 1, 64}, DType::BF16);
  extra.actual_seq_lengths_kv = op::create_input({2}, DType::INT64);
  extra.atten_mask            = op::create_input({2, 1, 128}, DType::INT8);
  auto out = op::fused_infer_attention(options)(query, key, key, extra);
  if (out.shape() != query.shape()) {
    std::cerr << "decode_mla output shape mismatch" << std::endl;
    return 1;
  }
  return 0;
}

// B64 exceeds 910B's Cube count. Uniform scan lengths plus slot masks must
// still return each slot's own values, without reading poisoned padding.
int test_decode_mla_large_batch(GEBackend &backend) {
  constexpr int64_t b = 64, c = 128, h = 16, d = 512, r = 64;
  auto q       = op::create_input({b, 1, h, d}, DType::BF16).set_name("mla_q");
  auto k       = op::create_input({b, c, 1, d}, DType::BF16).set_name("mla_k");
  auto qr      = op::create_input({b, 1, h, r}, DType::BF16).set_name("mla_qr");
  auto kr      = op::create_input({b, c, 1, r}, DType::BF16).set_name("mla_kr");
  auto lengths = op::create_input({b}, DType::INT64).set_name("mla_lengths");
  auto mask    = op::create_input({b, 1, c}, DType::INT8).set_name("mla_mask");
  op::FusedInferAttention::Options opt(h);
  opt.input_layout        = op::FusedInferAttention::InputLayout::BSND;
  opt.num_key_value_heads = 1;
  opt.decode_mla          = true;
  opt.inner_precise       = 2;
  op::FusedInferAttention::OptionalInputs extra;
  extra.query_rope            = qr;
  extra.key_rope              = kr;
  extra.actual_seq_lengths_kv = lengths;
  extra.atten_mask            = mask;
  auto out                    = op::fused_infer_attention(opt)(q, k, k, extra);
  out.set_name("mla_out");
  auto graph =
    backend.build(Graph({out}), {.enable_host_input_indexes = false});
  graph.compile();
  graph.load();
  std::vector<uint16_t> qv(b * h * d, 0), qrv(b * h * r, 0), krv(b * c * r, 0);
  std::vector<uint16_t> kv(b * c * d, fp32_to_bf16(100.0F));
  std::vector<int8_t>   mv(b * c, 1);
  for (int64_t slot = 0; slot < b; ++slot) {
    const auto n = slot % 2 ? 65 : 33;
    for (int64_t j = 0; j < n; ++j) {
      mv[slot * c + j] = 0;
      for (int64_t x = 0; x < d; ++x)
        kv[(slot * c + j) * d + x] = fp32_to_bf16(float(slot + 1) / 64);
    }
  }
  auto result =
    graph
      .run({op::load_from_buffer(
              reinterpret_cast<Buffer>(qv.data()), q.shape(), DType::BF16)
              .set_name("mla_q"),
        op::load_from_buffer(
          reinterpret_cast<Buffer>(kv.data()), k.shape(), DType::BF16)
          .set_name("mla_k"),
        op::load_from_buffer(
          reinterpret_cast<Buffer>(qrv.data()), qr.shape(), DType::BF16)
          .set_name("mla_qr"),
        op::load_from_buffer(
          reinterpret_cast<Buffer>(krv.data()), kr.shape(), DType::BF16)
          .set_name("mla_kr"),
        op::load_from_vector(std::vector<int64_t>(b, 65), {b})
          .set_name("mla_lengths"),
        op::load_from_vector(std::move(mv), {b, 1, c}).set_name("mla_mask")})
      .get("mla_out");
  if (result.shape() != q.shape() || result.dtype() != DType::BF16)
    return 1;
  const auto *values = static_cast<const uint16_t *>(result.get_data());
  for (int64_t i = 0; i < b * h * d; ++i) {
    const float actual   = bf16_to_fp32(values[i]);
    const float expected = float(i / (h * d) + 1) / 64;
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 0.01F) {
      std::cerr << "MLA B64 mismatch at " << i << ": " << actual << " vs "
                << expected << std::endl;
      return 1;
    }
  }
  return 0;
}

int main() {
  if (test_decode_mla_option() != 0)
    return 1;
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;
  if (test_decode_mla_large_batch(backend_impl) != 0)
    return 1;

  int ret = test_fused_infer_attention_matches_existing_ops(backend);
  if (ret != 0) {
    return ret;
  }
  ret = test_bnsd_to_bsnd_shape();
  if (ret != 0) {
    return ret;
  }
  ret = test_optional_inputs_shape();
  if (ret != 0) {
    return ret;
  }
  ret = test_dynamic_key_value_shape();
  if (ret != 0) {
    return ret;
  }
  ret = test_dynamic_key_value_batch_mismatch();
  if (ret != 0) {
    return ret;
  }
  ret = test_dynamic_key_value_requires_batch_one();
  if (ret != 0) {
    return ret;
  }
  // ret = benchmark_fused_infer_attention_bf16_bnsd_bsnd(backend);
  // if (ret != 0) {
  //   return ret;
  // }

  std::cout << "FusedInferAttention tests passed." << std::endl;
  return 0;
}
