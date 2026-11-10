# Device 与设备内存的生命周期

一个 rank 构造一个 `AscendDevice`，再把句柄传给 `GEBackend`。Device 负责选定本进程的卡、分配显存和同步拷贝字节。GE 负责解释 Tensor 的形状和 dtype、构造运行时视图以及管理图和 session。将来接入 ACLNN 时，另一个 backend 可以接收同一设备句柄，显存的释放也无需依赖 GE。

## 设备由谁绑定

`AscendDevice(id)` 初始化 ACL，并将 `id` 作为**进程可见的逻辑设备号**传给 `aclrtSetDevice`。`AscendDevice::from_env()` 从 `FORCH_GE_DEVICE_ID` 读取同一个参数，未设置或为空时取 0。例如，launcher 将某个 rank 的 `ASCEND_RT_VISIBLE_DEVICES` 设为物理卡 3 后，该进程构造的是逻辑设备 0。全局 rank 和 rank table 由通信配置提供。参见 [Ascend 实现](../src/forch-ascend/device.cpp)和[多卡通信层](collectives.md)。

一进程一卡是调用方的约定。构造后通过复制句柄共享同一实现，初始化、使用和析构都串行执行，调用方须保持该设备的默认 context 为当前 context。GE 在建图、运行和卸载时直接使用它。GEBackend 的一次初始化限制属于 GE，Device 本身没有单例检查。

## 一次 H2D 拷贝由谁持有

下例在 GE backend 析构后读回四个浮点数。返回的 Tensor 持有复制到设备上的数据，`get_data()` 取得的设备地址只在存储有效期间可用。

```cpp
#include "forch/backend/forch-ge.hpp"
#include "forch/ops/const.hpp"
#include "forch/ops/function.hpp"
#include <array>

std::array<float, 4> round_trip() {
  auto device = forch::AscendDevice::from_env();
  auto tensor = [&] {
    forch::GEBackend backend(device);
    auto host = forch::op::load_from_vector<float>({1, 2, 3, 4}, {2, 2});
    return host | forch::to_device(backend);
  }();

  std::array<float, 4> values{};
  device.device_to_host(values.data(), tensor.get_data(),
                        values.size() * sizeof(float));
  return values;
}
```

`to_device(backend)` 接受 `op::Const`。GE 检查形状和字节数后调用 `Device::allocate(bytes)`，将返回的可移动 `Device::Buffer` 放入 `shared_ptr`，同步复制数据，再创建借用设备地址的 GE view。Tensor 保留原来的名称、形状和 dtype。直接调用字节拷贝接口时，调用方保证两端容量足够，设备地址属于当前 Device。零字节拷贝允许空指针。[Device 接口](../include/forch/device/device.hpp) · [GE 实现](../src/forch-ge/backend.cpp)

| 持有关系 | 释放时的行为 |
| --- | --- |
| H2D Tensor → GERuntime → GE view 和共享 Buffer | 复制 Tensor 共享同一节点。GERuntime 先声明 Buffer、后声明 view，C++ 逆序析构保证先销毁借用地址的 view。 |
| Buffer → 分配它的 Device 实现 | 最后一个持有者释放 Buffer 时，由该实现释放显存。H2D 拷贝或创建 view 抛错时，局部持有者也会回收分配。将来增加缓存时，应避免从 Device 强引用回 Buffer。 |
| GE Graph → GE backend → session 和设备句柄 | 已加载的图在析构时先卸载。最后一个 backend 句柄释放后销毁 session、终结 GE。外部 ACL 使用方须让 ACL 和默认 context 存活到最后一次显存释放。 |

这套 Buffer 所有权适用于 H2D 输入。图运行返回的输出由 GE 管理。另一个容易混淆的名称是 `forch::Buffer`，它是既有的裸指针别名；持有设备分配的类型全名是 `forch::Device::Buffer`。

## 验证改动

Buffer 所有权和 GE 析构后的读回共用一个测试目标。前半段注入拷贝异常并统计释放次数，检查异常回收、Device 句柄先析构及最后一个持有者释放。后半段在真卡上拷贝 Tensor，销毁 backend 后读回并核对数值。[合并测试源码](../src/forch-ge/tests/device_memory.cpp)

在已配置 CANN 的环境中，从已配置好的 GE 构建目录运行：

```bash
cmake --build . --target forch-ge-test-device_memory -j8
ASCEND_WORK_PATH=ascend_work srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 \
  ./src/forch-ge/tests/forch-ge-test-device_memory
```

修改 GE 初始化或图执行路径时，相关回归分别是 `forch-ge-test-backend_lifecycle` 和 `forch-ge-test-move_to_device`。多 rank 验证使用现有[集合通信测试入口](collectives.md#怎么跑)。
