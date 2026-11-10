#include "forch/ops/topk.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/cast.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/input.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace forch;

/// 每个用例取哪些端口作为图输出。
enum class TopKPorts { Values, Indices, Both };

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-5F) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

static size_t numel(const Shape &shape) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

static bool indices_match(const int64_t *a, const int64_t *b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

static std::vector<float> topk_reference_values_1d(
  const std::vector<float> &data, int64_t k) {
  size_t                                n = data.size();
  std::vector<std::pair<float, size_t>> pairs(n);
  for (size_t i = 0; i < n; ++i) {
    pairs[i] = {data[i], i};
  }
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
    [](const auto &a, const auto &b) { return a.first > b.first; });
  std::vector<float> out(static_cast<size_t>(k));
  for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
    out[i] = pairs[i].first;
  }
  return out;
}

static std::vector<int64_t> topk_reference_indices_1d(
  const std::vector<float> &data, int64_t k) {
  size_t                                n = data.size();
  std::vector<std::pair<float, size_t>> pairs(n);
  for (size_t i = 0; i < n; ++i) {
    pairs[i] = {data[i], i};
  }
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
    [](const auto &a, const auto &b) { return a.first > b.first; });
  std::vector<int64_t> out(static_cast<size_t>(k));
  for (size_t i = 0; i < static_cast<size_t>(k); ++i) {
    out[i] = static_cast<int64_t>(pairs[i].second);
  }
  return out;
}

/// 沿 dim 维做 top-k,返回展平的 values。
static std::vector<float> topk_reference_values(
  const std::vector<float> &data, const Shape &shape, int64_t k, int64_t dim) {
  int64_t rank = static_cast<int64_t>(shape.size());
  int64_t ndim = dim < 0 ? dim + rank : dim;

  size_t outer = 1;
  for (int64_t i = 0; i < ndim; ++i) {
    outer *= static_cast<size_t>(shape[i]);
  }
  size_t axis  = static_cast<size_t>(shape[ndim]);
  size_t inner = 1;
  for (int64_t i = ndim + 1; i < rank; ++i) {
    inner *= static_cast<size_t>(shape[i]);
  }

  std::vector<float> out(outer * static_cast<size_t>(k) * inner);
  for (size_t o = 0; o < outer; ++o) {
    for (size_t in = 0; in < inner; ++in) {
      std::vector<float> slice(axis);
      for (size_t a = 0; a < axis; ++a) {
        slice[a] = data[o * axis * inner + a * inner + in];
      }
      auto ref = topk_reference_values_1d(slice, k);
      for (size_t a = 0; a < static_cast<size_t>(k); ++a) {
        out[o * static_cast<size_t>(k) * inner + a * inner + in] = ref[a];
      }
    }
  }
  return out;
}

/// 沿 dim 维做 top-k,返回展平的 indices。
static std::vector<int64_t> topk_reference_indices(
  const std::vector<float> &data, const Shape &shape, int64_t k, int64_t dim) {
  int64_t rank = static_cast<int64_t>(shape.size());
  int64_t ndim = dim < 0 ? dim + rank : dim;

  size_t outer = 1;
  for (int64_t i = 0; i < ndim; ++i) {
    outer *= static_cast<size_t>(shape[i]);
  }
  size_t axis  = static_cast<size_t>(shape[ndim]);
  size_t inner = 1;
  for (int64_t i = ndim + 1; i < rank; ++i) {
    inner *= static_cast<size_t>(shape[i]);
  }

  std::vector<int64_t> out(outer * static_cast<size_t>(k) * inner);
  for (size_t o = 0; o < outer; ++o) {
    for (size_t in = 0; in < inner; ++in) {
      std::vector<float> slice(axis);
      for (size_t a = 0; a < axis; ++a) {
        slice[a] = data[o * axis * inner + a * inner + in];
      }
      auto ref = topk_reference_indices_1d(slice, k);
      for (size_t a = 0; a < static_cast<size_t>(k); ++a) {
        out[o * static_cast<size_t>(k) * inner + a * inner + in] = ref[a];
      }
    }
  }
  return out;
}

static int64_t norm_dim(int64_t dim, const Shape &shape) {
  return dim < 0 ? dim + static_cast<int64_t>(shape.size()) : dim;
}

static Shape topk_expected_shape(const Shape &shape, int64_t k, int64_t dim) {
  Shape out                 = shape;
  out[norm_dim(dim, shape)] = k;
  return out;
}

/**
 * 校验设备端同时返回的 values/indices 端口是否来自同一次 TopK:
 *   1. indices 全部落在输入 dim 维范围内;
 *   2. values 精确等于输入中对应 indices 指向的那个元素;
 *   3. sorted 时 values 沿 k 维不增。
 * 该检查不依赖 CPU 参考,只依赖 pair 自身的自洽性。
 */
static bool topk_pair_consistent(const std::vector<float> &data,
  const Shape &shape, int64_t k, int64_t dim, const float *vals,
  const int64_t *idx) {
  int64_t ndim  = norm_dim(dim, shape);
  size_t  outer = 1;
  for (int64_t i = 0; i < ndim; ++i) {
    outer *= static_cast<size_t>(shape[i]);
  }
  size_t axis  = static_cast<size_t>(shape[ndim]);
  size_t inner = 1;
  for (int64_t i = ndim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
    inner *= static_cast<size_t>(shape[i]);
  }

  const size_t nk = static_cast<size_t>(k);
  for (size_t o = 0; o < outer; ++o) {
    for (size_t j = 0; j < nk; ++j) {
      for (size_t in = 0; in < inner; ++in) {
        const size_t pos = o * nk * inner + j * inner + in;
        const auto   p   = idx[pos];
        if (p < 0 || static_cast<size_t>(p) >= axis) {
          return false; // 索引越界
        }
        const float src =
          data[o * axis * inner + static_cast<size_t>(p) * inner + in];
        if (src != vals[pos]) {
          return false; // values 与 indices 不对应
        }
        if (j > 0) {
          const size_t prev = o * nk * inner + (j - 1) * inner + in;
          if (vals[prev] < vals[pos]) {
            return false; // sorted 输出应沿 k 不增
          }
        }
      }
    }
  }
  return true;
}

static int run_topk_case(Backend &backend, const Shape &shape,
  const std::vector<float> &data, int64_t k, int64_t dim, bool sorted,
  TopKPorts ports, const char *case_name) {

  std::cout << "  [" << case_name << "] shape=";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << (i > 0 ? "x" : "") << shape[i];
  }
  std::cout << " k=" << k << " dim=" << dim << " sorted=" << sorted;
  if (ports == TopKPorts::Values || ports == TopKPorts::Both)
    std::cout << " [values]";
  if (ports == TopKPorts::Indices || ports == TopKPorts::Both)
    std::cout << " [indices]";
  std::cout << " ... " << std::flush;

  auto input_name = std::string(case_name) + "_input";
  auto input      = op::create_input(shape, DType::FP32).set_name(input_name);

  const auto  val_name = std::string(case_name) + "_values";
  const auto  idx_name = std::string(case_name) + "_indices";
  const Shape expected = topk_expected_shape(shape, k, dim);

  // 共享节点的端口写法:一次调用创建共享 TopKNode,再按用例选择端口作为
  // 图输出。单端口用例同样来自同一个 pair(只消费其中一路)。
  auto tk = op::topk(k, dim, sorted)(input);

  TensorList outputs;
  if (ports == TopKPorts::Values || ports == TopKPorts::Both) {
    outputs.push_back(tk.values.set_name(val_name));
  }
  if (ports == TopKPorts::Indices || ports == TopKPorts::Both) {
    outputs.push_back(tk.indices.set_name(idx_name));
  }

  for (const auto &t : outputs) {
    if (t.shape() != expected) {
      std::cerr << case_name << ": shape mismatch before build" << std::endl;
      return 1;
    }
  }

  forch::Graph graph_ir(outputs);
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

  const bool with_vals = ports == TopKPorts::Values || ports == TopKPorts::Both;
  const bool with_idxs =
    ports == TopKPorts::Indices || ports == TopKPorts::Both;
  const auto expected_vals = with_vals
    ? topk_reference_values(data, shape, k, dim)
    : std::vector<float>{};
  const auto expected_idxs = with_idxs
    ? topk_reference_indices(data, shape, k, dim)
    : std::vector<int64_t>{};

  // data 按值传入,这里拷贝一份搬进 const 张量后,后面的一致性检查仍可用。
  auto host_data = data;
  auto in_tensor =
    op::load_from_vector(std::move(host_data), shape).set_name(input_name);
  auto run_outputs = graph.run({in_tensor});

  if (with_vals) {
    auto out_tensor = run_outputs.get(val_name);
    if (out_tensor.shape() != expected) {
      std::cerr << case_name << ": values shape mismatch at runtime"
                << std::endl;
      return 3;
    }

    const float *out_ptr =
      reinterpret_cast<const float *>(out_tensor.get_data());
    constexpr float tol = 1e-3F;
    if (!almost_equal(
          out_ptr, expected_vals.data(), expected_vals.size(), tol)) {
      std::cerr << case_name << ": values differ" << std::endl;
      size_t n = numel(expected);
      for (size_t i = 0; i < n; ++i) {
        if (std::fabs(out_ptr[i] - expected_vals[i]) > tol) {
          std::cerr << "first mismatch at " << i << ": out=" << out_ptr[i]
                    << ", expected=" << expected_vals[i] << std::endl;
          break;
        }
      }
      return 4;
    }
  }

  if (with_idxs) {
    auto out_tensor = run_outputs.get(idx_name);
    if (out_tensor.shape() != expected) {
      std::cerr << case_name << ": indices shape mismatch at runtime"
                << std::endl;
      return 5;
    }

    const int64_t *out_ptr =
      reinterpret_cast<const int64_t *>(out_tensor.get_data());
    if (!indices_match(out_ptr, expected_idxs.data(), expected_idxs.size())) {
      std::cerr << case_name << ": indices differ" << std::endl;
      size_t n = numel(expected);
      for (size_t i = 0; i < n; ++i) {
        if (out_ptr[i] != expected_idxs[i]) {
          std::cerr << "first mismatch at " << i << ": out=" << out_ptr[i]
                    << ", expected=" << expected_idxs[i] << std::endl;
          break;
        }
      }
      return 6;
    }
  }

  // 双端口且 sorted 时,校验 values/indices 来自同一次 TopK(自洽性)。
  if (ports == TopKPorts::Both && sorted) {
    auto         vals_t   = run_outputs.get(val_name);
    auto         idxs_t   = run_outputs.get(idx_name);
    const float *vals_ptr = reinterpret_cast<const float *>(vals_t.get_data());
    const int64_t *idxs_ptr =
      reinterpret_cast<const int64_t *>(idxs_t.get_data());
    if (!topk_pair_consistent(data, shape, k, dim, vals_ptr, idxs_ptr)) {
      std::cerr << case_name << ": values/indices pair not consistent"
                << std::endl;
      return 7;
    }
  }

  std::cout << "ok" << std::endl;
  return 0;
}

/// 同一份输入分别跑 Both 与单端口(端口同源于一个共享 TopKNode)。
static int run_all_ports(Backend &backend, const Shape &shape,
  const std::vector<float> &data, int64_t k, int64_t dim, bool sorted,
  const char *tag) {
  int r;
  if ((r = run_topk_case(
         backend, shape, data, k, dim, sorted, TopKPorts::Both, tag))
    != 0) {
    return r;
  }
  const auto vals_only = std::string(tag) + "_vals_only";
  if ((r = run_topk_case(backend, shape, data, k, dim, sorted,
         TopKPorts::Values, vals_only.c_str()))
    != 0) {
    return r;
  }
  const auto idx_only = std::string(tag) + "_idx_only";
  return run_topk_case(
    backend, shape, data, k, dim, sorted, TopKPorts::Indices, idx_only.c_str());
}

/// 双端口拓扑用例:values 端口先被下游 op 消费,再与 indices 端口一起输出
/// (端口作为中间值喂给其它算子)。
static int test_port_as_intermediate(Backend &backend) {
  const Shape              shape = {4};
  const std::vector<float> data  = {4.0F, 1.0F, 3.0F, 2.0F};
  auto                     input =
    op::create_input(shape, DType::FP32).set_name("port_inter_input");
  auto tk       = op::topk(2, 0, true)(input);
  auto consumed = tk.values | op::cast(forch::DType::FP32);
  consumed.set_name("port_inter_values");
  tk.indices.set_name("port_inter_indices");

  forch::Graph graph_ir({consumed, tk.indices});
  graph_ir.set_name("port_inter");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "port_inter: compile/load failed: " << e.what() << std::endl;
    return 2;
  }

  auto host_data = data;
  auto in_tensor = op::load_from_vector(std::move(host_data), shape)
                     .set_name("port_inter_input");
  auto outputs = graph.run({in_tensor});
  auto vals_t  = outputs.get("port_inter_values");
  auto idxs_t  = outputs.get("port_inter_indices");
  if (!topk_pair_consistent(data, shape, 2, 0,
        reinterpret_cast<const float *>(vals_t.get_data()),
        reinterpret_cast<const int64_t *>(idxs_t.get_data()))) {
    std::cerr << "port_inter: values/indices pair not consistent" << std::endl;
    return 3;
  }
  std::cout << "  [port_intermediate] ok" << std::endl;
  return 0;
}

/// builder 复用回归:同一个 builder 对象在同一张图里调用两次,两次必须
/// 各自以自己的输入物化(修复前共享 node,第二次的输入会被第一次的
/// 静默替换,后端按 node 指针去重)。
static int test_builder_reuse(Backend &backend) {
  const Shape shape = {4};

  auto in_a = op::create_input(shape, DType::FP32).set_name("reuse_a");
  auto in_b = op::create_input(shape, DType::FP32).set_name("reuse_b");

  auto b  = op::topk(2, 0, true); // 同一 builder,两次调用
  auto t1 = b(in_a);
  auto t2 = b(in_b);
  t1.values.set_name("reuse_vals1");
  t2.values.set_name("reuse_vals2");

  forch::Graph graph_ir(TensorList{t1.values, t2.values});
  graph_ir.set_name("builder_reuse");
  auto graph = backend.build(graph_ir);
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "builder_reuse: compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  const std::vector<float> data1 = {4.0F, 1.0F, 3.0F, 2.0F}; // top2: 4 3
  const std::vector<float> data2 = {9.0F, 0.0F, 7.0F, 5.0F}; // top2: 9 7
  const std::vector<float> exp1  = {4.0F, 3.0F};
  const std::vector<float> exp2  = {9.0F, 7.0F};

  auto a_t =
    op::load_from_vector(std::vector<float>(data1), shape).set_name("reuse_a");
  auto b_t =
    op::load_from_vector(std::vector<float>(data2), shape).set_name("reuse_b");
  auto        outputs = graph.run(TensorList{a_t, b_t});
  const auto *v1 =
    reinterpret_cast<const float *>(outputs.get("reuse_vals1").get_data());
  const auto *v2 =
    reinterpret_cast<const float *>(outputs.get("reuse_vals2").get_data());

  if (!almost_equal(v1, exp1.data(), exp1.size())
    || !almost_equal(v2, exp2.data(), exp2.size())) {
    std::cerr << "builder_reuse: second call did not get its own inputs"
              << std::endl;
    return 3;
  }
  std::cout << "  [builder_reuse] ok" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  std::cout << "--- 1D tests ---" << std::endl;
  {
    const Shape              shape = {6};
    const std::vector<float> data  = {5.0F, 2.0F, 8.0F, 1.0F, 9.0F, 3.0F};
    if (int r = run_all_ports(backend, shape, data, /*k=*/3, /*dim=*/0,
          /*sorted=*/true, "topk_1d");
        r != 0) {
      return r;
    }
  }

  std::cout << "--- 2D tests ---" << std::endl;
  {
    const Shape              shape = {2, 3};
    const std::vector<float> data  = {
      1.0F,
      5.0F,
      3.0F,
      7.0F,
      2.0F,
      9.0F,
    };
    if (int r = run_all_ports(backend, shape, data, /*k=*/2, /*dim=*/1,
          /*sorted=*/true, "topk_2d_dim1");
        r != 0) {
      return r;
    }
  }

  std::cout << "--- 3D tests (non-last dim) ---" << std::endl;
  {
    const Shape              shape = {2, 3, 2};
    const std::vector<float> data  = {
      1.0F,
      2.0F,
      0.0F,
      -1.0F,
      3.0F,
      1.0F,
      4.0F,
      -2.0F,
      2.0F,
      0.5F,
      -1.0F,
      3.0F,
    };
    if (int r = run_all_ports(backend, shape, data, /*k=*/2, /*dim=*/1,
          /*sorted=*/true, "topk_3d_dim1");
        r != 0) {
      return r;
    }
  }

  std::cout << "--- negative dim tests ---" << std::endl;
  {
    const Shape              shape = {2, 3};
    const std::vector<float> data  = {
      1.0F,
      5.0F,
      3.0F,
      7.0F,
      2.0F,
      9.0F,
    };
    if (int r = run_all_ports(backend, shape, data, /*k=*/2, /*dim=*/-1,
          /*sorted=*/true, "topk_neg_dim");
        r != 0) {
      return r;
    }
  }

  std::cout << "--- k equals dim tests ---" << std::endl;
  {
    const Shape              shape = {3};
    const std::vector<float> data  = {3.0F, 1.0F, 2.0F};
    if (int r = run_all_ports(backend, shape, data, /*k=*/3, /*dim=*/0,
          /*sorted=*/true, "topk_k_eq_dim");
        r != 0) {
      return r;
    }
  }

  if (int r = test_port_as_intermediate(backend); r != 0) {
    return r;
  }

  if (int r = test_builder_reuse(backend); r != 0) {
    return r;
  }

  std::cout << "TopK tests passed." << std::endl;
  return 0;
}
