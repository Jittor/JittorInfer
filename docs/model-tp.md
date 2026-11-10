# 单机 Tensor Parallel

## 启动

编译后端和启动前，在已激活的 CANN 环境中加载 HMPI（见 [HMPI 文档](hmpi.md)）：

```bash
source /path/to/HPCKit/latest/hmpi/gcc/env/setvars.sh --mode=release
DEVICES=2,3 bash script/run-server.sh backend/config/example_qwen3_tp.json
```

`engine.tensor_parallel_size` 决定 rank 数。大于 1 时 `run-server.sh` 通过
`script/mpirun-ge.py` 启动后端：它复用 forch 的 rank table 生成逻辑，再 exec
HMPI 的 `mpirun`，每个 rank 只看到自己的卡。卡号取 `DEVICES`，缺省为
`ASCEND_RT_VISIBLE_DEVICES`。

示例配置：`example_{llama,deepseek_v2,qwen3,qwen3_5}_tp.json` 为 TP2，
`example_qwen3_8_tp.json` 为 Qwen3.8-27B 的 TP8。

## 进程组

`engine_server` 总是调用 `MPI_Init`；直接运行时是单进程 MPI world。
`bin/` 校验 `tensor_parallel_size` 等于 MPI world size，再把
`ProcessGroup(rank, size)` 写入 `EngineConfig::tp`。模型和 GEBackend 都只从
`EngineConfig::tp` 取并行信息。

## 请求和同步

`server::Transport` 是 server 层唯一感知 rank 的组件，所有 rank 以相同顺序调用它：

- 只有 rank0 持有 ZMQ socket，完成前端握手、发送输出和 Utility 响应。
  所有 rank 建完图后经一次 `MPI_Barrier`，rank0 才发送 READY。
- 每轮 step 前 `recv()`：rank0 收取当前可读的原始请求帧，先广播批次长度再广播
  字节数据（空批次也走相同轮次），各 rank 解码同一份字节。
- 计划执行前 `all_have_capacity()`：通过 `MPI_Allreduce(MIN)` 对各 rank 的
  `runner.can_run` 结果取交集，保证任何一个 rank 缺页时都一起抢占并重排。
- 采样后 `sync_tokens()`：各 rank 都按本地候选采样，再以 rank0 的结果为准。
  TopK 在并列值上的选择不保证跨 rank 一致，所以必须同步。

Scheduler 不分支处理 rank，通过 `OutputSink` 使用容量协调和 token 同步接口；
`Server` 在每轮规划前调用 `Transport::recv()`。

## 模型切分

| 模型 | Attention | FFN |
|---|---|---|
| Qwen3 / Llama | Q 按 head 切分，KV 切分或复制，O 行并行归约 | gate/up 列并行，down 行并行归约 |
| Qwen3.5 | full attention 同上；GDN 按 K/V head 切分，状态留在本地 | 同上 |
| DeepSeek-V2 | q_proj 或 q_b_proj、kv_b_proj 按 head 切分，o_proj 行并行归约 | dense/shared/routed expert 的中间通道切分 |

- 所有模型共用 `models::MLP`（列并行 gate/up + 行并行 down）。
- 本地 head 数由并行线性层给出，runner 从模型读取后分配 cache。KV 页分配在每个
  rank 上独立进行，但所有 rank 收到相同的请求与 token，因此映射一致；
  `can_run` 取各 rank 结果的交集。
- Embedding、norm 保持复制；行并行 bias 在归约后只加一次。
- `VocabParallelLMHead` 按词表行均分，要求 `vocab_size % tp == 0`。每卡做本地
  TopK，AllGather 分数和全局 token ID 后取全局 TopK；模型 forward 直接返回
  values/indices。共享 embedding 权重时仍从原 checkpoint key 加载本地行。
- Q heads、FFN 中间维度必须被 TP size 整除。KV heads 不小于 TP size 时同样
  需要整除，否则 TP size 必须是 KV heads 的整数倍，相邻 rank 复制同一个 KV head。
- DeepSeek 的 q_a_proj、kv_a_proj_with_mqa、latent cache 保持复制；MoE 使用 TP，
  每个 rank 保留全部 expert 和 router，routed 输出合并后归约一次，shared expert
  独立归约。

## 验证

`test_model_tp`（主机 CTest）检查 Llama/DeepSeek 的权重切片、局部维度、归约位置和
LM head。端到端回归使用 `test_model_serving` 的 TP2 配置（Qwen3-30B-A3B、
Qwen3.5-35B-A3B、Qwen3.8-27B），最近结果见 [测试与验证](testing.md)。
