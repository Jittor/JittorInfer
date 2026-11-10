# HMPI 安装与 JittorInfer 多进程启动

`engine_server` 用 MPI 建立 TP 进程组；`script/mpirun-ge.py` 复用
`forch/script/forch-ge-run.py` 的设备列表解析和 rank table 生成，再 exec 华为
`mpirun` 启动各 rank。HCCL 仍由 GEBackend 根据 rank table 初始化。仅支持单机。

## 安装 HMPI

从华为取得 `HPCKit_26.1.RC1_Linux-aarch64.tar.gz` 及其校验文件，选择匹配操作系统
和网卡驱动的安装包。安装流程参见[华为 HPCKit 安装指南](https://www.hikunpeng.com/doc_center/source/en/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_012.html)。

使用**交互安装**，在组件菜单中只勾选 GCC 版 HMPI，不采用 `-y` 默认安装。
在安装包所在目录执行：

```bash
sha256sum -c HPCKit_26.1.RC1_Linux-aarch64.tar.gz.sha256sum
tar -xzf HPCKit_26.1.RC1_Linux-aarch64.tar.gz

bash HPCKit_26.1.RC1_Linux-aarch64/install.sh
```

按提示完成以下选择：

1. 安装根目录填写 `$HOME/software` 对应的实际路径，例如 `/path/to/software`。
   安装器会在其下追加 `HPCKit/26.1.RC1`。
2. 编译器/变体菜单只保留 **GCC Compiler**，取消 **Bisheng Compiler**。
3. 在 **GCC Compiled Component** 菜单中，只保留 **HMPI**；取消 KML、KUPL、
   KuLitho、KuRTM、KuDNN、KuQCD 等其他组件。输入对应编号切换勾选，确认仅 HMPI
   显示为 `[v]`，然后输入 `y` 或回车继续。
4. HMPI 类型选择与网卡驱动匹配的选项：华为网卡驱动使用 `hns`，Mellanox 驱动
   使用 `mlx`。具体支持情况以安装器对当前操作系统的检查为准。
5. HIO 菜单取消 HIO，不安装其他组件。

这里的“只装 HMPI”指组件菜单只选择 HMPI。26.1 RC1 安装器把编译器选择与组件变体
绑定，选择 GCC 时还会安装包内的 GCC，无法通过原版交互菜单单独取消它。后续只
source HMPI 的组件级环境脚本，继续使用项目原有的系统编译器，不加载 HPCKit
编译器环境。HMPI 所需的 HUCX/XUCG 随其组件包安装。

本文后续使用以下安装目录变量，按实际位置调整：

```bash
HPCKIT_ROOT="$HOME/software/HPCKit/26.1.RC1"
```

## 加载环境

运行 Forch GE 程序前，先激活 CANN 环境，再加载 HMPI：

```bash
micromamba activate forch
source "$CONDA_PREFIX/Ascend/cann/set_env.sh"
source "$HPCKIT_ROOT/hmpi/gcc/env/setvars.sh" --mode=release
```

组件级 `setvars.sh` 设置以下环境：

| 变量 | 用途 |
| --- | --- |
| `OPAL_PREFIX` | HMPI 安装前缀，`mpirun-ge.py` 使用其中的 `bin/mpirun` |
| `PATH` | 加入 HMPI、HUCX、XUCG 的可执行文件目录 |
| `LD_LIBRARY_PATH` | 加入对应的动态库目录 |
| `INCLUDE` | 加入对应的头文件目录 |

确认编译器包装器、运行库来自同一套 HMPI：

```bash
command -v mpicc
command -v mpicxx
command -v mpirun
printf 'OPAL_PREFIX=%s\n' "$OPAL_PREFIX"
mpirun --version
```

使用固定版本目录便于复现；若使用安装器创建的 `latest` 链接，需要确认其实际指向。
只需加载官方组件环境，不需要手动设置 PMIx、ESS 或 rank 相关变量。

## 启动

完整服务由 `run-server.sh` 按配置的 `engine.tensor_parallel_size` 自动选择：

```bash
DEVICES=2,3 bash script/run-server.sh backend/config/example_qwen3_tp.json
```

只启动后端或运行其他 GE/MPI 程序：

```bash
DEVICES=2,3 python3 script/mpirun-ge.py 2 -- \
  ./build/backend/engine_server backend/config/example_qwen3_tp.json
```

`mpirun-ge.py` 的流程：

1. 用 forch 的逻辑生成 rank table，写到 `$TMPDIR/finer-mpirun-<uid>/`，并设置共享的
   `ASCEND_WORK_PATH`、`FORCH_GE_RANK_TABLE_FILE` 和 `FORCH_GE_DEVICE_ID=0`。
2. exec `$OPAL_PREFIX/bin/mpirun`，每个 rank 一个 MPMD 段，各自的
   `ASCEND_RT_VISIBLE_DEVICES` 只含一张卡。信号和退出码由 mpirun 直接处理。
3. 程序从 MPI 获取 rank 和 world size。

SLURM 只需分配卡并启动一个 launcher task，HMPI 在本机启动 N 个 rank。
输出由 mpirun 按 rank 标记。

## 常见问题

- `load HMPI's env/setvars.sh first`：先加载华为 HMPI 组件级 `setvars.sh`。
- 共享库缺失符号：检查是否混入系统 Open MPI、其他 HMPI 版本或其他 PMIx 库。
  编译和运行应使用同一套华为 HMPI。
- `hccn_tool not found` 或设备 IP 查询失败：加载 CANN 环境。

官方源码：[HMPI 仓库](https://github.com/kunpengcompute/hmpi)。
安装后的环境变量也可参见[华为环境配置说明](https://www.hikunpeng.com/doc_center/source/en/kunpenghpcs/hpckit/instg/KunpengHPCKit_install_014.html)。
