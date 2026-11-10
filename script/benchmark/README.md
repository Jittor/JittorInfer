# benchmark

JittorInfer 的评测脚本：**性能测速**（huggingface/inference-benchmarker）+ **正确性评测**（OpenCompass）。
两者都指向"已启动的 server"的 OpenAI 兼容接口，server 用 `bash script/run-server.sh` 启动。

| 文件 | 作用 |
|---|---|
| `env.local.sh` | 本机配置：conda、server 地址、模型名、tokenizer 路径 |
| `bench-performance.sh` | 性能测速入口：指定 prefill / decode 长度 |
| `bench-correctness.sh` | 正确性评测入口：gsm8k / mrcr / quick |
| `install-inference-benchmarker.sh` | 装 inference-benchmarker（clone CoderOJ fork + cargo build） |
| `install-opencompass.sh` | 装 OpenCompass 0.5.4（pip 到 `env.local.sh` 里的 conda 环境） |
| `opencompass/eval_gsm8k.py` | GSM8K 全量配置（1319 题，4-shot CoT） |
| `opencompass/eval_mrcr.py` | MRCR 长上下文配置（默认 3 个 ≤32k 档） |
| `opencompass/eval_quick.py` | 16 条 gsm8k 冒烟配置 |
| `opencompass/oc_limit.py` | 抽样数据集类（固定取前 16 条） |

结果目录：性能 `bench-results/p<prefill>-d<decode>-v<vus>-<时间戳>/`，正确性 `eval-results/<项>/<时间戳>/`。

本机相关的配置只写在 [`env.local.sh`](env.local.sh)：conda 环境、server 地址、模型名、tokenizer。换机器只改这个文件。

## 安装

```bash
bash script/benchmark/install-inference-benchmarker.sh   # 装到 build/inference-benchmarker
bash script/benchmark/install-opencompass.sh             # 按 env.local.sh 的 conda 环境 pip 安装
```


## 性能测速

`--prefill`、`--decode`、`--vus` 必填。地址、模型名、tokenizer 用 `env.local.sh`；`--url`、`--tokenizer` 可临时覆盖。`--warmup` 默认 `30s`，`--duration` 默认 `120s`。**需要重启 server** 对齐 `max_num_request` 和上下文长度：

```bash
bash script/benchmark/bench-performance.sh --prefill 32 --decode 96 --vus 4
bash script/benchmark/bench-performance.sh --prefill 4096 --decode 1 --vus 1
```

- 指标从工具表格里读：decode 看 `Throughput (tokens/sec)`；prefill 看 `TTFT (avg)`
  除以 prompt 长度得 prompt tok/s。

- 工具没装先跑 `install-inference-benchmarker.sh`

## 正确性评测


```bash
bash script/benchmark/bench-correctness.sh quick   # 16 条 gsm8k 冒烟，快速验证
bash script/benchmark/bench-correctness.sh gsm8k   # 全量 1319 题
bash script/benchmark/bench-correctness.sh mrcr    # 长上下文 MRCR，3 个 ≤32k 档，约 300 条
```

- 并发、长度档在 `opencompass/eval_<项>.py` 顶部改。地址、模型名、tokenizer 在 `env.local.sh`。
- `quick`/`gsm8k` 用同一套 4-shot CoT 模板，quick 只是取前 16 条（`oc_limit.py`，固定 16），
  只用于验证链路。


## 已知坑


1. **服务端默认关 thinking**（`script/run-server.sh` 的
   `--default-chat-template-kwargs '{"enable_thinking":false}'`）：思考 token 会污染吞吐/TTFT，
   且思考内容走 `delta.reasoning` 通道、评测工具只统计 `delta.content`。要开思考：
   请求里带 `chat_template_kwargs={"enable_thinking":true}`，或
   `CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}' bash script/run-server.sh` 全局开回。

2. MRCR 要完整 24 个子集（4/8-needle、64k~1M 档）的话，`eval_mrcr.py` 的 `MRCR_SUBSETS`
   补全即可，但需要后端支持对应上下文长度，且长档单条推理要分钟级。

## 参考基线（Qwen3-0.6B，910B，n_context=8192；16k 档用 32768）

| 项 | 结果 |
|---|---|
| decode tps | 1 并发 104；4 并发 390 tok/s |
| prefill | 1k：2.2k tok/s（TTFT 460ms）；4k：2.3k（1.8s）；16k：830（19.8s） |
| gsm8k 全量 | accuracy 53.15 |
| mrcr 2needle 4k-8k | score 20.50 |
