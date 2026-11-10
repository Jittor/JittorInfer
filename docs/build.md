# 构建与依赖

本文记录项目构建契约与依赖关系，不指定开发者的环境管理器、环境名称或安装目录。
本机激活命令与资源分配规则放在不入库的 `AGENTS.local.md` 中；
可使用 [本地说明模板](../AGENTS.local.example.md) 建立自己的记录。

## 构建前提

- 后端目标平台为 aarch64 Linux / Ascend；构建需要 CANN Toolkit、匹配芯片的算子包、
  C++20 编译器和 CMake。执行模型和算子还需要已分配的 NPU。
- 后端即使单卡也依赖 MPI 开发库。构建和运行应加载同一套 MPI，
  当前 HMPI 启动流程见 [HMPI 指南](hmpi.md)。
- Rust 前端工具链以 `rust-frontend/rust-toolchain.toml` 为准。首次构建可能需要联网
  获取工具链、crates 和 git 依赖。
- 在当前 shell 中准备好依赖后，再执行下面的命令；CANN 的环境脚本应设置
  `ASCEND_HOME_PATH`。不要把某台机器已有的环境视为已验证基线。

[910B 环境配方](env/910b.yaml) 是 Atlas A2 / 910B 与 CANN 9.1.0.beta3 的一种安装方案，
不是完整锁文件，也不要求所有开发者使用相同环境名称。其他硬件需选择相应算子包并验证。
源码安装环境与 [Docker 部署](docker.md) 是不同路径，不要混用容器和宿主的工具链目录。

## 构建顺序

在任务 worktree 根目录初始化该 worktree 的子模块，随后依次构建：

```bash
git submodule update --init --recursive
CMAKE_BUILD_TYPE=Release bash script/build-third-party.sh
CMAKE_BUILD_TYPE=Release bash script/build-forch.sh
CMAKE_BUILD_TYPE=Release bash script/build-backend.sh
(cd rust-frontend && cargo build --release --bin vllm-rs)
```

前端构建前还必须设置下节说明的 `PROTOC_INCLUDE`。
第三方库和 Forch 会先安装头文件、库，供后端配置阶段消费。
默认产物分别在 `build/third_party/`、`build/forch/`、`build/backend/`；
前端可执行文件在 `rust-frontend/target/release/`。

`CMAKE_BUILD_TYPE` 默认 `Debug`；`NPROC` 默认 `nproc`，应按已分配的 CPU 数调整。
`BUILD_DIR` 是共同构建根目录，不是某个组件的输出目录；三个阶段必须使用相同根目录。
启动脚本仍使用默认相对路径，调整构建目录时要同时检查启动方式。
不同 worktree 不应复用同一个 CMake 构建目录，也不要为省事改动另一个 worktree 的子模块。

## 三种 protobuf

| 依赖 | 用途 |
| --- | --- |
| `protoc` 和标准 `.proto` 文件 | Rust 构建时生成 gRPC 类型 |
| CANN 的 `libascend_protobuf` | C++ 运行时依赖 |
| Python `protobuf` / `google.protobuf` | CANN 辅助 Python 工具 |

前后端通信是 ZMQ + MessagePack，不使用 protobuf。Rust 前端继承了可选的北向 gRPC
服务，但当前 `build.rs` 仍无条件编译 `proto/vllm_grpc.proto`；它导入
`google/protobuf/struct.proto`，因此即使部署不启动 gRPC，也需要编译器和标准 proto 文件。

安装供应商提供的 protobuf 开发包后，将 `PROTOC_INCLUDE` 设为包含
`google/protobuf/struct.proto` 的 include 根目录。在实际激活的环境中检查：

```bash
command -v protoc
protoc --version
test -f "${PROTOC_INCLUDE:?set PROTOC_INCLUDE}/google/protobuf/struct.proto"
```

本仓库的 910B 配方用 conda `libprotobuf` 提供编译器和标准 proto 文件，
用 pip `protobuf` 提供 Python 包。不要为了对齐 Python 包而降级 native `libprotobuf`。
不要让 conda 和 pip 同时安装 Python `protobuf`，它们会覆盖相同的 `google/protobuf` 文件。
配方未固定 Python protobuf 的版本，见 [兼容性说明](env/protobuf.md)。

## CANN 与 ABI

CANN 的部分 C++ 接口使用旧 libstdc++ ABI。Forch 将这类调用隔离在 C shim 中，
不要把 `_GLIBCXX_USE_CXX11_ABI=0` 扩散为全项目编译选项，也不要未经验证就替换编译器。
修改 Forch 时先阅读它自己的 `AGENTS.md` / `CLAUDE.md`。

包管理器可能把 CANN 随包提供的 Python 模块（如 `te`、`auto_tune`、`hccl`、`ge-py`）
标记为来自 pip；这不说明它们存在于公共 PyPI。核对安装位置与实际 imports，
不要将这些模块名直接加入 pip 依赖。

构建后，后端与前端运行入口见 [README](../README.md)；
测试目标和设备需求见 [测试与验证](testing.md)。
