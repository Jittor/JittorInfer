#include "forch/ops/scatter_update.hpp"
#include "forch/backend/forch-ge.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"

#include <cmath>
#include <iostream>
#include <vector>

using namespace forch;

static bool almost_equal(
  const float *a, const float *b, size_t n, float tol = 1e-5f) {
  for (size_t i = 0; i < n; ++i) {
    if (std::fabs(a[i] - b[i]) > tol) {
      return false;
    }
  }
  return true;
}

// 测试1: 1D张量的基本散点更新
int test_scatter_update_1d(Backend &backend) {
  // 初始变量: [0, 0, 0, 0, 0]
  Shape              var_shape = {5};
  std::vector<float> var_data  = {0, 0, 0, 0, 0};

  // 索引: 在位置 1, 3, 4 处更新
  Shape                indices_shape = {3};
  std::vector<int32_t> indices_data  = {1, 3, 4};

  // 更新值: [10, 30, 40]
  Shape              updates_shape = {3};
  std::vector<float> updates_data  = {10, 30, 40};

  // 期望结果: [0, 10, 0, 30, 40]
  Shape              out_shape = {5};
  std::vector<float> expected  = {0, 10, 0, 30, 40};

  auto var = op::create_input(var_shape, DType::FP32).set_name("var_1d");
  auto indices =
    op::create_input(indices_shape, DType::INT32).set_name("indices_1d");
  auto updates =
    op::create_input(updates_shape, DType::FP32).set_name("updates_1d");
  auto out = op::scatter_update()(var, indices, updates);
  out.set_name(std::string("scatter_update_output_1d"));

  auto graph =
    backend.build(forch::Graph({out}).set_name("test_scatter_update_1d"));
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "scatter_update_1d: compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  auto var_tensor =
    op::load_from_vector(std::move(var_data), var_shape).set_name("var_1d");
  auto indices_tensor =
    op::load_from_vector(std::move(indices_data), indices_shape)
      .set_name("indices_1d");
  auto updates_tensor =
    op::load_from_vector(std::move(updates_data), updates_shape)
      .set_name("updates_1d");

  auto outputs = graph.run({var_tensor, indices_tensor, updates_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("scatter_update_output_1d")).get_data());

  if (outputs.get(std::string("scatter_update_output_1d")).shape()
    != out_shape) {
    std::cerr << "scatter_update_1d: shape mismatch" << std::endl;
    auto shape = outputs.get(std::string("scatter_update_output_1d")).shape();
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cerr << shape[i] << (i + 1 < shape.size() ? "," : "\n");
    }
    return 4;
  }

  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "scatter_update_1d: values differ" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      std::cerr << "Expected: " << expected[i] << ", Got: " << out_ptr[i]
                << std::endl;
    }
    return 5;
  }

  std::cout << "scatter_update_1d: passed" << std::endl;
  return 0;
}

// 测试2: 2D张量在第一个维度上的散点更新
int test_scatter_update_2d_dim0(Backend &backend) {
  // 初始变量: 3x4 的矩阵，初始化为0
  Shape              var_shape = {3, 4};
  std::vector<float> var_data(12, 0.0f); // 全0矩阵

  // 索引: 更新第0行和第2行
  Shape                indices_shape = {2};
  std::vector<int32_t> indices_data  = {0, 2};

  // 更新值: 2x4 的矩阵
  Shape              updates_shape = {2, 4};
  std::vector<float> updates_data  = {
    1, 2, 3, 4,   // 第0行
    9, 10, 11, 12 // 第2行
  };

  // 期望结果:
  // 第0行: [1,2,3,4]
  // 第1行: [0,0,0,0]
  // 第2行: [9,10,11,12]
  Shape              out_shape = {3, 4};
  std::vector<float> expected  = {1, 2, 3, 4, 0, 0, 0, 0, 9, 10, 11, 12};

  auto var = op::create_input(var_shape, DType::FP32).set_name("var_2d_dim0");
  auto indices =
    op::create_input(indices_shape, DType::INT32).set_name("indices_2d_dim0");
  auto updates =
    op::create_input(updates_shape, DType::FP32).set_name("updates_2d_dim0");
  auto out = op::scatter_update()(var, indices, updates);
  out.set_name(std::string("scatter_update_output_2d_dim0"));

  auto graph =
    backend.build(forch::Graph({out}).set_name("test_scatter_update_2d_dim0"));
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "scatter_update_2d_dim0: compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  auto var_tensor = op::load_from_vector(std::move(var_data), var_shape)
                      .set_name("var_2d_dim0");
  auto indices_tensor =
    op::load_from_vector(std::move(indices_data), indices_shape)
      .set_name("indices_2d_dim0");
  auto updates_tensor =
    op::load_from_vector(std::move(updates_data), updates_shape)
      .set_name("updates_2d_dim0");

  auto outputs = graph.run({var_tensor, indices_tensor, updates_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("scatter_update_output_2d_dim0")).get_data());

  if (outputs.get(std::string("scatter_update_output_2d_dim0")).shape()
    != out_shape) {
    std::cerr << "scatter_update_2d_dim0: shape mismatch" << std::endl;
    return 4;
  }

  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "scatter_update_2d_dim0: values differ" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(out_ptr[i] - expected[i]) > 1e-5f) {
        std::cerr << "At index " << i << ": Expected " << expected[i]
                  << ", Got " << out_ptr[i] << std::endl;
      }
    }
    return 5;
  }

  std::cout << "scatter_update_2d_dim0: passed" << std::endl;
  return 0;
}

// 测试3: 重复索引的情况（后面的更新会覆盖前面的）
int test_scatter_update_duplicate_indices(Backend &backend) {
  // 测试当索引重复时的行为（后面的更新应该覆盖前面的）
  Shape              var_shape = {5};
  std::vector<float> var_data  = {0, 0, 0, 0, 0};

  // 索引包含重复: 位置1被更新两次
  Shape                indices_shape = {3};
  std::vector<int32_t> indices_data  = {1, 2, 1}; // 索引1出现两次

  // 更新值: 最后索引1的值应该是30
  Shape              updates_shape = {3};
  std::vector<float> updates_data  = {10, 20, 30};

  // 期望结果: 位置1最后被更新为30
  Shape              out_shape = {5};
  std::vector<float> expected  = {0, 30, 20, 0, 0};

  auto var = op::create_input(var_shape, DType::FP32).set_name("var_duplicate");
  auto indices =
    op::create_input(indices_shape, DType::INT32).set_name("indices_duplicate");
  auto updates =
    op::create_input(updates_shape, DType::FP32).set_name("updates_duplicate");
  auto out = op::scatter_update()(var, indices, updates);
  out.set_name(std::string("scatter_update_output_duplicate"));

  auto graph = backend.build(
    forch::Graph({out}).set_name("test_scatter_update_duplicate_indices"));
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "scatter_update_duplicate_indices: compile/load failed: "
              << e.what() << std::endl;
    return 2;
  }

  auto var_tensor = op::load_from_vector(std::move(var_data), var_shape)
                      .set_name("var_duplicate");
  auto indices_tensor =
    op::load_from_vector(std::move(indices_data), indices_shape)
      .set_name("indices_duplicate");
  auto updates_tensor =
    op::load_from_vector(std::move(updates_data), updates_shape)
      .set_name("updates_duplicate");

  auto outputs = graph.run({var_tensor, indices_tensor, updates_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("scatter_update_output_duplicate")).get_data());

  if (outputs.get(std::string("scatter_update_output_duplicate")).shape()
    != out_shape) {
    std::cerr << "scatter_update_duplicate_indices: shape mismatch"
              << std::endl;
    return 4;
  }

  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "scatter_update_duplicate_indices: values differ" << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      if (std::fabs(out_ptr[i] - expected[i]) > 1e-5f) {
        std::cerr << "At index " << i << ": Expected " << expected[i]
                  << ", Got " << out_ptr[i] << std::endl;
      }
    }
    return 5;
  }

  std::cout << "scatter_update_duplicate_indices: passed" << std::endl;
  return 0;
}

// 测试4: 边界测试 - 更新最后一个元素
int test_scatter_update_boundary(Backend &backend) {
  Shape              var_shape = {10};
  std::vector<float> var_data(10, 0.0f);

  // 更新第一个和最后一个元素
  Shape                indices_shape = {2};
  std::vector<int32_t> indices_data  = {0, 9};

  Shape              updates_shape = {2};
  std::vector<float> updates_data  = {100, 999};

  Shape              out_shape = {10};
  std::vector<float> expected(10, 0.0f);
  expected[0] = 100;
  expected[9] = 999;

  auto var = op::create_input(var_shape, DType::FP32).set_name("var_boundary");
  auto indices =
    op::create_input(indices_shape, DType::INT32).set_name("indices_boundary");
  auto updates =
    op::create_input(updates_shape, DType::FP32).set_name("updates_boundary");
  auto out = op::scatter_update()(var, indices, updates);
  out.set_name(std::string("scatter_update_output_boundary"));

  auto graph =
    backend.build(forch::Graph({out}).set_name("test_scatter_update_boundary"));
  try {
    graph.compile();
    graph.load();
  } catch (const std::exception &e) {
    std::cerr << "scatter_update_boundary: compile/load failed: " << e.what()
              << std::endl;
    return 2;
  }

  auto var_tensor = op::load_from_vector(std::move(var_data), var_shape)
                      .set_name("var_boundary");
  auto indices_tensor =
    op::load_from_vector(std::move(indices_data), indices_shape)
      .set_name("indices_boundary");
  auto updates_tensor =
    op::load_from_vector(std::move(updates_data), updates_shape)
      .set_name("updates_boundary");

  auto outputs = graph.run({var_tensor, indices_tensor, updates_tensor});
  if (outputs.empty())
    return 3;

  const float *out_ptr = reinterpret_cast<const float *>(
    outputs.get(std::string("scatter_update_output_boundary")).get_data());

  if (outputs.get(std::string("scatter_update_output_boundary")).shape()
    != out_shape) {
    std::cerr << "scatter_update_boundary: shape mismatch" << std::endl;
    return 4;
  }

  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "scatter_update_boundary: values differ" << std::endl;
    return 5;
  }

  std::cout << "scatter_update_boundary: passed" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  int r = 0;

  r = test_scatter_update_1d(backend);
  if (r != 0)
    return r;

  r = test_scatter_update_2d_dim0(backend);
  if (r != 0)
    return r;

  r = test_scatter_update_duplicate_indices(backend);
  if (r != 0)
    return r;

  r = test_scatter_update_boundary(backend);
  if (r != 0)
    return r;

  std::cout << "\nAll scatter_update tests passed successfully!" << std::endl;
  return 0;
}