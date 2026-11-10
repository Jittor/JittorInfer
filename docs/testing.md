# 测试与验证

测试源码在 `backend/tests/`，默认不构建。CMake 会自动发现目录下所有 `.cpp`，
每个文件一个可执行文件。所有命令都需要先激活 CANN 环境。
配置前应已构建安装 Forch 和第三方库，并加载 MPI 开发环境（见 [HMPI](hmpi.md)）。
下面的命令在仓库根目录执行，测试二进制位于 `build/backend/tests/`，不会自动加入 PATH。

```bash
cmake -S backend -B build/backend -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/build/forch/install;$PWD/build/third_party/install"
cmake --build build/backend -j8
ctest --test-dir build/backend --output-on-failure
```

## 主机测试（CTest，无需 NPU）

| 测试 | 覆盖 |
|---|---|
| `test_runner_contract` | token 打包与输出行；dense/paged 写入行（96 种布局）；eager/FIA mask；TND/BSND/MLA 长度、mask 与块表 |
| `test_prefix_cache` | SHA-256 编码；完整块发布时机；保留最后 token；共享页引用计数；用量；prefix reset 的拒绝与成功 |
| `test_scheduler_preemption` | 真实 `PagedSlotMap` + 主机递推模型：三种淘汰策略、随机采样下的抢占重算、decode 阶段抢占、取消、超长拒绝、满上下文结束；统计上报与 `reset_prefix_cache` |
| `test_attention_contract` | 各 attention 方案的输出形状、必需输入、dtype、MLA RoPE、算子属性，dense/paged 存储 |
| `test_model_tp` | Llama/DeepSeek 的 TP 权重切片、局部维度、归约位置、MLA 吸收与 LM head |

## NPU 测试

需要先分配设备（本机通过 SLURM `npu` 分区）。

| 命令 | 覆盖 |
|---|---|
| `test_static_attention [paged \| slot [paged] \| prefix \| long paged]` | full attention 写入 + 计算对照 CPU double 参考：不均匀分配、空槽、padding、容量上限、槽复用、多 query 槽块、共享前缀页、16 槽长上下文 |
| `test_static_mla_attention 128 [paged]` | MLA latent/RoPE 写入与 TND/BSND 计算对照 CPU double 参考 |
| `test_gdn_slot_lifecycle <config>` | Qwen3.5 槽回收与首次使用：其他请求持续运行产生 padding 后，新准入槽输出与干净请求一致 |
| `test_model_serving <config> <report.json>` | 完整 checkpoint + 生产 scheduler（内存 OutputSink 代替 HTTP）：三条自然语言 smoke、逐请求串行与并发混合生成比较、抢占次数、步数与耗时 |
| `test_model_forward <config> --ids 1,2,3` | 调试入口：不起前端直接驱动 runner；`--graph` 单图冒烟，`--json` 落盘 ids 与每步耗时 |

`test_model_serving` 是 MPI 程序，本机单卡也要经 mpirun 启动；TP 配置通过
`script/mpirun-ge.py` 启动（见 [HMPI](hmpi.md)）。

非 MPI 算子测试可直接交给 SLURM，例如在上述环境激活后运行：

```bash
ASCEND_WORK_PATH="$PWD/build/ascend_work" srun -p npu --gres=gpu:1 \
  --ntasks=1 --cpus-per-task=8 ./build/backend/tests/test_static_attention paged
```

### 模型回归配置

```bash
python script/test-paged-serving.py --tokenize --extra
```

在 `build/paged-validation/` 生成每个模型的配置：

| 后缀 | 含义 |
|---|---|
| `pressure` | 2K 上下文、4 槽、页池只容纳一个最大上下文，必然抢占 |
| `roomy` | 同负载、4 倍页池，不抢占 |
| `dense` | 关闭分页的对照 |
| `eager`（`--extra`） | 关闭 fused attention 的参考路径 |
| `no-prefix`、`decode-replay`、`extreme`（`--extra`） | 关闭前缀复用；decode 阶段抢占；8K × 16 槽 × 24 请求、总池 8192 token |

`--tokenize` 需要 Python `tokenizers` 包，用本地 `tokenizer.json` 生成 smoke prompt，
不下载模型。生成器读取 `backend/config/` 中的本机 checkpoint 路径，其他机器需替换路径。
`python script/summarize-paged-validation.py build/paged-validation --decode`
汇总报告。汇总不替代检查进程退出码，也不会自动判定数值差异为正确。

## 本机注意事项

- 本机 HMPI 在 SLURM 下直接启动会崩溃，需要
  `mpirun --bind-to none --mca plm isolated --mca ras ^slurm`，并设置
  `OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp`。
- TP launcher 的 `TMPDIR` 要用短路径（如 `/tmp/<job>`），否则 CANN Python
  multiprocessing 的 AF_UNIX 路径会超长。
- `test_model_serving` 在 `MPI_Finalize` 后直接退出以绕过 GE 异步清理，TBE 子进程
  可能打印清理警告；以报告、断言和退出码判断结果。
- 本机缺少 `libasan` / `libubsan`，sanitizer 构建无法链接。

## 当前验证记录

最近一次完整验证在 2026-11-06，针对共享 `SlotMap` / `CacheRunner` 重构
（`bd58cdf`）与统计上报修复（`caf38ee`），910B、CANN 9.1.0.beta3、Release 构建、
完整 checkpoint。

- 主机 CTest 5/5 通过。
- NPU 数值检查 10/10 通过：full/MLA × dense/paged × compact/slot、共享前缀、16 槽
  长上下文 8 个 oracle，以及 fused（容差 0）/eager GDN 槽生命周期。
- 模型配置 24/24 通过：Qwen3-0.6B、Llama-2-7b-chat、DeepSeek-V2-Lite-Chat、
  Qwen3.5-0.8B（TP1）与 Qwen3-30B-A3B、Qwen3.5-35B-A3B、Qwen3.8-27B（TP2）的
  pressure 与 dense；四个 TP1 模型的 eager；Qwen3/Qwen3.5 roomy、Qwen3 no-prefix、
  两个 decode-replay 和 Qwen3 extreme。串行与并发逐 token 一致；与重构前报告比较，
  完整结果、smoke 输出、抢占次数和步数全部相同。extreme 为 751 步、240 次抢占。
- 真实 Rust 前端 + `engine_server` 的 HTTP 检查 8/8 通过：单步请求计数、空闲时
  gauge 归零、前缀命中、空闲 reset 成功、reset 后冷 miss、8 并发长请求与串行一致、流式 `[DONE]`、
  压力后仍可服务。

粗略耗时（不含建图，不作显著性承诺）：

| 配置 | 重构前 mixed | 重构后 mixed |
|---|---:|---:|
| Qwen3 no-prefix，53 步 | 0.550 s | 0.461 s |
| Qwen3.5 roomy，263 步 | 2.543 s | 2.219 s |
| Qwen3 extreme，751 步 | 10.981 s | 8.206 s |

分页相对 dense 的开销在等工作量下约为 Qwen3 +7.5%、Qwen3.5 +1.9%（重构前测量）。
小页池带来的重算是容量换计算的预期代价，不应与宽裕的 dense 总耗时直接比较。

未覆盖：全量模型质量评测（见 `script/benchmark/`）、多模态、sanitizer。
以上是指定配置下的回归结果，不是对所有输入或所有硬件的正确性保证；没有重新构建 Rust 前端。
原始日志与报告在本地 `build/revalidation-bd58cdf/`（不入库）。
