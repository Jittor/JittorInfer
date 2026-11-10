#include "forch/backend/forch-ge.hpp"
#include "forch/exception.hpp"
#include "forch/forch.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include "forch/ops/input.hpp"
#include "forch/ops/scatter_update.hpp"

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

// device 上的 var 作为 scatter_update 的输入被更新
int test_device_tensor_scatter_update(Backend &backend) {
  // 初始 var: [1, 2, 3, 4, 5]
  Shape              var_shape = {5};
  std::vector<float> var_data  = {1, 2, 3, 4, 5};

  // indices: 在位置 1 和 4 处更新
  Shape                indices_shape = {2};
  std::vector<int32_t> indices_data  = {1, 4};

  // updates: 新值 100 和 300
  Shape              updates_shape = {2};
  std::vector<float> updates_data  = {100, 300};

  // 期望: [1, 100, 3, 4, 300]
  std::vector<float> expected = {1, 100, 3, 4, 300};

  // 1. 将 var 放到 device 上
  auto host_var =
    op::load_from_vector(std::move(var_data), var_shape).set_name("var_dev");
  auto var_device = host_var | forch::to_device(backend);

  // 2. 构建 scatter_update 图：device var 将被就地更新
  auto var_ph = op::create_input(var_shape, DType::FP32).set_name("var_dev");
  auto indices =
    op::create_input(indices_shape, DType::INT32).set_name("indices_dev");
  auto updates =
    op::create_input(updates_shape, DType::FP32).set_name("updates_dev");
  auto out = op::scatter_update()(var_ph, indices, updates);
  out.set_name("scatter_dev_out");

  auto graph =
    backend.build(forch::Graph({out}).set_name("test_device_scatter_update"));
  graph.compile();
  graph.load();

  // 3. 以 device tensor 为 var 输入运行 graph
  auto indices_tensor =
    op::load_from_vector(std::move(indices_data), indices_shape)
      .set_name("indices_dev");
  auto updates_tensor =
    op::load_from_vector(std::move(updates_data), updates_shape)
      .set_name("updates_dev");

  auto outputs = graph.run({var_device, indices_tensor, updates_tensor});

  // 4. 验证输出
  const float *out_ptr =
    reinterpret_cast<const float *>(outputs.get("scatter_dev_out").get_data());

  if (outputs.get("scatter_dev_out").shape() != var_shape) {
    std::cerr << "FAIL: shape mismatch" << std::endl;
    return 1;
  }

  if (!almost_equal(out_ptr, expected.data(), expected.size())) {
    std::cerr << "FAIL: values differ (device tensor not updated correctly)"
              << std::endl;
    for (size_t i = 0; i < expected.size(); ++i) {
      std::cerr << "  [" << i << "] expected=" << expected[i]
                << " got=" << out_ptr[i] << std::endl;
    }
    return 2;
  }

  std::cout << "test_device_tensor_scatter_update: passed" << std::endl;
  return 0;
}

// 同一个 device var 多次 scatter_update，更新效果叠加累积
int test_device_tensor_multi_run(Backend &backend) {
  Shape              var_shape = {5};
  std::vector<float> var_data  = {1, 2, 3, 4, 5};

  auto host_var =
    op::load_from_vector(std::move(var_data), var_shape).set_name("var_dev");
  auto var_device = host_var | forch::to_device(backend);

  // 构建图：固定占位符名字，每次 run 用不同数据填充
  auto var_ph = op::create_input(var_shape, DType::FP32).set_name("var_dev");
  auto indices_ph = op::create_input({-1}, DType::INT32).set_name("indices");
  auto updates_ph = op::create_input({-1}, DType::FP32).set_name("updates");
  auto out        = op::scatter_update()(var_ph, indices_ph, updates_ph);
  out.set_name("out");

  auto graph = backend.build(forch::Graph({out}).set_name("test_multi_run"));
  graph.compile();
  graph.load();

  // 第 1 次：初始 [1,2,3,4,5]，更新位置 0 → 100
  //         期望 [100,2,3,4,5]
  {
    std::vector<int32_t> indices_data = {0};
    std::vector<float>   updates_data = {100};
    std::vector<float>   expected     = {100, 2, 3, 4, 5};

    auto indices_t =
      op::load_from_vector(std::move(indices_data), {1}).set_name("indices");
    auto updates_t =
      op::load_from_vector(std::move(updates_data), {1}).set_name("updates");

    auto         outputs = graph.run({var_device, indices_t, updates_t});
    const float *out_ptr =
      reinterpret_cast<const float *>(outputs.get("out").get_data());

    if (!almost_equal(out_ptr, expected.data(), expected.size())) {
      std::cerr << "FAIL multi_run[0]: values differ" << std::endl;
      for (size_t i = 0; i < expected.size(); ++i)
        std::cerr << "  [" << i << "] expected=" << expected[i]
                  << " got=" << out_ptr[i] << std::endl;
      return 1;
    }
  }

  // 第 2 次：var_device 已被更新为 [100,2,3,4,5]，再更新位置 4 → 500
  //         期望 [100,2,3,4,500]
  {
    std::vector<int32_t> indices_data = {4};
    std::vector<float>   updates_data = {500};
    std::vector<float>   expected     = {100, 2, 3, 4, 500};

    auto indices_t =
      op::load_from_vector(std::move(indices_data), {1}).set_name("indices");
    auto updates_t =
      op::load_from_vector(std::move(updates_data), {1}).set_name("updates");

    auto         outputs = graph.run({var_device, indices_t, updates_t});
    const float *out_ptr =
      reinterpret_cast<const float *>(outputs.get("out").get_data());

    if (!almost_equal(out_ptr, expected.data(), expected.size())) {
      std::cerr << "FAIL multi_run[1]: values differ" << std::endl;
      for (size_t i = 0; i < expected.size(); ++i)
        std::cerr << "  [" << i << "] expected=" << expected[i]
                  << " got=" << out_ptr[i] << std::endl;
      return 2;
    }
  }

  // 第 3 次：var_device 已被更新为 [100,2,3,4,500]，再更新位置 1,3 → 200,400
  //         期望 [100,200,3,400,500]
  {
    std::vector<int32_t> indices_data = {1, 3};
    std::vector<float>   updates_data = {200, 400};
    std::vector<float>   expected     = {100, 200, 3, 400, 500};

    auto indices_t =
      op::load_from_vector(std::move(indices_data), {2}).set_name("indices");
    auto updates_t =
      op::load_from_vector(std::move(updates_data), {2}).set_name("updates");

    auto         outputs = graph.run({var_device, indices_t, updates_t});
    const float *out_ptr =
      reinterpret_cast<const float *>(outputs.get("out").get_data());

    if (!almost_equal(out_ptr, expected.data(), expected.size())) {
      std::cerr << "FAIL multi_run[2]: values differ" << std::endl;
      for (size_t i = 0; i < expected.size(); ++i)
        std::cerr << "  [" << i << "] expected=" << expected[i]
                  << " got=" << out_ptr[i] << std::endl;
      return 3;
    }
  }

  std::cout << "test_device_tensor_multi_run: passed" << std::endl;
  return 0;
}

// 析构测试：反复创建/销毁 device tensor，验证 aclrtFree 路径不崩溃
int test_device_tensor_destructor(Backend &backend) {
  // 使用较大的 tensor (64MB) 少量迭代：若未正确释放，累积分配会触发 OOM
  constexpr int kIters = 20;
  constexpr int kElems = 16 * 1024 * 1024; // 16M floats = 64MB
  Shape         shape  = {kElems};

  for (int iter = 0; iter < kIters; ++iter) {
    // 在内部 scope 创建并销毁 device tensor
    {
      std::vector<float> data(kElems, static_cast<float>(iter));
      auto               host_t =
        op::load_from_vector(std::move(data), shape).set_name("lifecycle");
      auto dev_t = host_t | forch::to_device(backend);
      (void)dev_t;
      // dev_t 离开 scope → RtTensorPtr 析构 → forch_ge_rt_tensor_destroy
      //   → delete handle → gert::Tensor 析构 → manager(kFreeTensor) →
      //   aclrtFree
    }
    // 此时 device 内存应已被释放，下一轮迭代复用相同大小的显存
  }

  // 循环结束后再分配一个同样大小的 tensor：若之前释放正常，应成功
  {
    std::vector<float> data(kElems, 42.0f);
    auto               host_t =
      op::load_from_vector(std::move(data), shape).set_name("final");
    auto dev_t = host_t | forch::to_device(backend);
    (void)dev_t;
  }

  std::cout << "test_device_tensor_destructor: passed (" << kIters
            << " create/destroy cycles)" << std::endl;
  return 0;
}

// 非 Const tensor 调用 to_device(backend) 应抛出异常
int test_to_rejects_non_const(Backend &backend) {
  // Input placeholder 不是 Const
  auto input_t = op::create_input({10}, DType::FP32).set_name("not_const");
  try {
    input_t | forch::to_device(backend);
    std::cerr << "FAIL: should have thrown for non-Const tensor" << std::endl;
    return 1;
  } catch (const forch::exception &) {
    // expected
  }

  // add 的输出（Function）也不是 Const
  auto a   = op::create_input({5}, DType::FP32).set_name("a");
  auto sum = op::add(a, a).set_name("sum");
  try {
    sum | forch::to_device(backend);
    std::cerr << "FAIL: should have thrown for Function tensor" << std::endl;
    return 2;
  } catch (const forch::exception &) {
    // expected
  }

  std::cout << "test_to_rejects_non_const: passed" << std::endl;
  return 0;
}

// FP16 和 INT32 两种 dtype 的 device tensor 基本功能
int test_device_tensor_multi_dtype(Backend &backend) {
  // --- FP16 scatter_update ---
  {
    // FP16 常量: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400, 5.0=0x4500
    Shape                 var_shape = {5};
    std::vector<uint16_t> var_data{
      0x3C00, 0x4000, 0x4200, 0x4400, 0x4500}; // 1,2,3,4,5

    // load_from_buffer 不接管 buffer 所有权，vector 需在此 scope 内保持存活
    auto host_var = op::load_from_buffer(
      reinterpret_cast<Buffer>(var_data.data()), var_shape, DType::FP16)
                      .set_name("var_fp16");
    auto var_device = host_var | forch::to_device(backend);

    // scatter_update: indices=[0,4], updates=[10.0=0x4900, 50.0=0x5240]
    auto var_ph = op::create_input(var_shape, DType::FP16).set_name("var_fp16");
    Shape                 indices_shape = {2};
    std::vector<int32_t>  indices_data  = {0, 4};
    Shape                 updates_shape = {2};
    std::vector<uint16_t> updates_data{0x4900, 0x5240};
    std::vector<uint16_t> expected{0x4900, 0x4000, 0x4200, 0x4400, 0x5240};

    auto idx =
      op::create_input(indices_shape, DType::INT32).set_name("idx_fp16");
    auto upd =
      op::create_input(updates_shape, DType::FP16).set_name("upd_fp16");
    auto out = op::scatter_update()(var_ph, idx, upd);
    out.set_name("out_fp16");

    auto graph = backend.build(forch::Graph({out}).set_name("test_dtype_fp16"));
    graph.compile();
    graph.load();

    auto idx_t = op::load_from_vector(std::move(indices_data), indices_shape)
                   .set_name("idx_fp16");
    auto upd_t = op::load_from_buffer(
      reinterpret_cast<Buffer>(updates_data.data()), updates_shape, DType::FP16)
                   .set_name("upd_fp16");

    auto            outputs = graph.run({var_device, idx_t, upd_t});
    const uint16_t *out_ptr =
      reinterpret_cast<const uint16_t *>(outputs.get("out_fp16").get_data());

    for (size_t i = 0; i < expected.size(); ++i) {
      if (out_ptr[i] != expected[i]) {
        std::cerr << "FAIL dtype FP16: values differ" << std::endl;
        for (size_t j = 0; j < expected.size(); ++j)
          std::cerr << std::hex << "  [" << j << "] expected=0x" << expected[j]
                    << " got=0x" << out_ptr[j] << std::dec << std::endl;
        return 1;
      }
    }
  }

  // --- INT32 scatter_update ---
  {
    Shape                var_shape = {4};
    std::vector<int32_t> var_data  = {100, 200, 300, 400};

    auto host_var =
      op::load_from_vector(std::move(var_data), var_shape).set_name("var_i32");
    auto var_device = host_var | forch::to_device(backend);

    auto var_ph = op::create_input(var_shape, DType::INT32).set_name("var_i32");
    Shape                indices_shape = {2};
    std::vector<int32_t> indices_data  = {1, 2};
    Shape                updates_shape = {2};
    std::vector<int32_t> updates_data  = {999, 888};
    std::vector<int32_t> expected      = {100, 999, 888, 400};

    auto idx =
      op::create_input(indices_shape, DType::INT32).set_name("idx_i32");
    auto upd =
      op::create_input(updates_shape, DType::INT32).set_name("upd_i32");
    auto out = op::scatter_update()(var_ph, idx, upd);
    out.set_name("out_i32");

    auto graph = backend.build(forch::Graph({out}).set_name("test_dtype_i32"));
    graph.compile();
    graph.load();

    auto idx_t = op::load_from_vector(std::move(indices_data), indices_shape)
                   .set_name("idx_i32");
    auto upd_t = op::load_from_vector(std::move(updates_data), updates_shape)
                   .set_name("upd_i32");

    auto           outputs = graph.run({var_device, idx_t, upd_t});
    const int32_t *out_ptr =
      reinterpret_cast<const int32_t *>(outputs.get("out_i32").get_data());

    for (size_t i = 0; i < expected.size(); ++i) {
      if (out_ptr[i] != expected[i]) {
        std::cerr << "FAIL dtype INT32: values differ" << std::endl;
        for (size_t j = 0; j < expected.size(); ++j)
          std::cerr << "  [" << j << "] expected=" << expected[j]
                    << " got=" << out_ptr[j] << std::endl;
        return 2;
      }
    }
  }

  std::cout << "test_device_tensor_multi_dtype: passed" << std::endl;
  return 0;
}

int main() {
  GEBackend backend_impl(AscendDevice::from_env());
  Backend   backend = backend_impl;

  int r = 0;

  r = test_device_tensor_destructor(backend);
  if (r != 0)
    return r;

  r = test_to_rejects_non_const(backend);
  if (r != 0)
    return r;

  r = test_device_tensor_multi_dtype(backend);
  if (r != 0)
    return r;

  r = test_device_tensor_scatter_update(backend);
  if (r != 0)
    return r;

  r = test_device_tensor_multi_run(backend);
  if (r != 0)
    return r;

  return 0;
}
