# vLLM Rust 前端 — 引擎端接口协议

本文档描述外部推理后端接入 vLLM Rust 前端所需的完整协议。协议完全基于 ZMQ + MessagePack，**语言无关**——你可以用 Python、C++、Go 或任何支持 ZMQ 和 MessagePack 的语言来实现。

## 架构概览

```
Rust 前端                          你的推理后端
─────────                         ────────────
                                  
启动时:                           
  bind handshake ROUTER  ←─────── DEALER connect + HELLO
  send INIT              ────────→ 获取 data plane 地址
  bind input ROUTER      ←─────── DEALER connect + ReadyResponse 注册
  bind output PULL       ←─────── PUSH connect
  收到 READY             ←─────── DEALER send READY

运行时:                           
  input ROUTER send      ────────→ DEALER recv (2帧: request_type + payload)
                         帧1: 请求类型 (1字节)
                          0x00 = Add 请求
                          0x01 = Abort 请求  
                          0x02 = StartDpWave (可忽略)
                          0x03 = Utility 请求
                         帧2: MessagePack 编码的负载

  output PULL recv       ←──────── PUSH send (1帧: MessagePack 编码的输出批次)
```

## 第一步: 握手协议

### 1.1 连接到 Handshake Socket

Rust 前端启动时会在 `handshake_address` (例如 `tcp://127.0.0.1:29550`) 上绑定一个 ZMQ ROUTER socket。

你的引擎需要:

1. 创建 ZMQ DEALER socket, 设置 identity 为 engine index 的 2 字节小端编码
2. 连接到 `handshake_address`
3. 发送 HELLO 消息

**ZMQ Identity 编码**: engine index 0 → `b'\x00\x00'`, index 1 → `b'\x01\x00'`, index 2 → `b'\x02\x00'`

**HELLO 消息** (单帧 MessagePack 映射):

```json
{
    "status": "HELLO",
    "headless": true,
    "local": false
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定为 `"HELLO"` |
| `headless` | bool | 固定为 `true` (外部引擎) |
| `local` | bool | 是否为本地引擎, 通常为 `false` |

### 1.2 接收 INIT 响应

前端收到 HELLO 后会回复一个单帧 MessagePack 消息:

```json
{
    "addresses": {
        "inputs": ["tcp://127.0.0.1:XXXXX"],
        "outputs": ["tcp://127.0.0.1:YYYYY"],
        "coordinator_input": null,
        "coordinator_output": null,
        "frontend_stats_publish_address": null
    },
    "parallel_config": {}
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `addresses.inputs` | [string] | Engine 输入 socket 地址 (DEALER → ROUTER) |
| `addresses.outputs` | [string] | Engine 输出 socket 地址 (PUSH → PULL) |
| `addresses.coordinator_input` | string\|null | DP coordinator 地址, 单引擎为 null |
| `addresses.coordinator_output` | string\|null | DP coordinator 地址, 单引擎为 null |
| `parallel_config` | object | 一般为空 `{}` |

对于单引擎部署, `inputs` 和 `outputs` 各一个元素; 多 API server 场景可能有多个。

### 1.3 连接 Data Plane Socket

**Input Socket (接收前端请求)**:

1. 创建 ZMQ DEALER socket, 设置与握手相同的 identity
2. 连接到 `addresses.inputs[0]`
3. 发送 **ReadyResponse** 作为注册帧 (单帧 MessagePack):

```json
{
    "max_model_len": 131072,
    "num_gpu_blocks": 65536,
    "block_size": 16,
    "dp_stats_address": null,
    "dtype": "bfloat16",
    "vllm_version": "my-backend-1.0",
    "world_size": 1,
    "data_parallel_size": 1,
    "kv_cache_size_tokens": null,
    "kv_cache_max_concurrency": null
}
```

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `max_model_len` | u64 | 是 | 最大上下文长度 (token 数), 前端用此值计算 `max_tokens` |
| `num_gpu_blocks` | u64 | 是 | GPU KV cache block 数量, 用于负载均衡 |
| `block_size` | u64 | 是 | 每个 block 的 token 数 |
| `dp_stats_address` | string\|null | 否 | 数据并行 stats 地址, 单引擎为 null |
| `dtype` | string | 是 | 模型数据类型: `"float32"`, `"float16"`, `"bfloat16"`, `"float8"`, `"auto"` |
| `vllm_version` | string | 是 | 引擎版本标识符 |
| `world_size` | u64 | 是 | TP × PP 规模 |
| `data_parallel_size` | u64 | 是 | 数据并行副本数 |
| `kv_cache_size_tokens` | u64\|null | 否 | 此引擎的 token 容量 |
| `kv_cache_max_concurrency` | f64\|null | 否 | 估算的最大并发数 |

**Output Socket (推送结果回前端)**:

1. 创建 ZMQ PUSH socket
2. 连接到 `addresses.outputs[0]`

### 1.4 发送 READY

Data plane 连接完成后的最后一步——在 **handshake DEALER** 上发送:

```json
{
    "status": "READY",
    "headless": true,
    "local": false
}
```

此后可以断开 handshake DEALER——不再需要它。

> **注意**: 如果启用了 DP coordinator (非 null), 引擎需要在发送 READY 之前等待 coordinator 的 `READY` barrier。参见 [数据并行](#数据并行) 章节。

---

## 第二步: 请求协议 (前端 → 引擎)

引擎在 input DEALER socket 上接收来自前端的请求, 格式为 **2 帧 ZMQ 消息**:

```
帧 1: 请求类型 (1 字节)
帧 2: MessagePack 编码的负载
```

### 2.1 Add 请求 (0x00)

发起一个新的文本生成请求。

**负载**: MessagePack 数组 (msgpack 元组), 字段按位置:

```python
# 伪代码, msgpack 元组/数组按索引访问
[
    request_id,          # [0] String: "chatcmpl-abc123-a1b2c3d4"
    prompt_token_ids,    # [1] [u32] 或 null: [101, 202, 303]
    mm_features,         # [2] 多模态特征 或 null
    sampling_params,     # [3] 采样参数 Map 或 null
    pooling_params,      # [4] 池化参数 或 null (embedding 模型使用)
    arrival_time,        # [5] f64: Unix 秒级时间戳
    lora_request,        # [6] LoRA 请求 或 null
    cache_salt,          # [7] String 或 null: 前缀缓存分区键
    data_parallel_rank,  # [8] u32 或 null: DP rank 路由覆盖
    prompt_embeds,       # [9] 嵌入向量 或 null (暂未使用)
    prompt_is_token_ids, # [10] 混合模式掩码 或 null (暂未使用)
    client_index,        # [11] u32: 前端客户端索引, 输出路由用
    current_wave,        # [12] u32: DP wave 编号
    priority,            # [13] i32: 调度优先级 (越小越优先)
    trace_headers,       # [14] Map 或 null: 追踪头
    resumable,           # [15] bool: 是否支持续传
    external_req_id,     # [16] String 或 null: 用户可见请求 ID
    reasoning_ended,     # [17] bool 或 null: reasoning 是否已结束
    reasoning_parser_kwargs, # [18] Map 或 null: reasoning 解析器参数
    abort_immediately,   # [19] bool: 是否立即中止
]
```

#### sampling_params (索引 [3]) 详细定义

MessagePack Map, 字段可以省略 (使用默认值):

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | f32 | 1.0 | 温度, 0 为贪婪采样 |
| `top_p` | f32 | 1.0 | 核采样累计概率阈值 |
| `top_k` | u32 | 0 | 只考虑 top-k token, 0=全部 |
| `seed` | i64\|null | null | 随机种子 |
| `max_tokens` | u32 | 16 | 最大生成 token 数 |
| `min_tokens` | u32 | 0 | 最小生成 token 数 (EOS 被屏蔽) |
| `thinking_token_budget` | u64\|null | null | 推理 token 预算, null=无限制 |
| `logprobs` | i32\|null | null | 每 token 返回的 logprob 数, -1=全词表 |
| `prompt_logprobs` | i32\|null | null | 每 prompt token 返回的 logprob 数 |
| `min_p` | f32 | 0.0 | 最小概率阈值 |
| `frequency_penalty` | f32 | 0.0 | 频率惩罚 |
| `presence_penalty` | f32 | 0.0 | 存在惩罚 |
| `repetition_penalty` | f32 | 1.0 | 重复惩罚 |
| `repetition_detection` | Map\|null | null | N-gram 重复检测参数 |
| `stop_token_ids` | [u32] | [] | 停止 token ID 列表 |
| `_eos_token_id` | u32\|null | null | 主 EOS token ID |
| `_all_stop_token_ids` | [u32] | [] | 完整停止 token 集合 (含用户+模型 token) |
| `logit_bias` | {u32: f32}\|null | null | token→logit 偏置映射 |
| `allowed_token_ids` | [u32]\|null | null | 生成时限制使用的 token |
| `_bad_words_token_ids` | [[u32]]\|null | null | 禁止出现的 token 序列 |
| `structured_outputs` | Map\|null | null | 结构化输出/guided decoding 参数 |
| `logprob_token_ids` | [u32]\|null | null | 返回指定 token 的 logprob |
| `skip_reading_prefix_cache` | bool\|null | null | 跳过前缀缓存读取 |
| `extra_args` | Map\|null | null | 自定义扩展参数 |

#### 你需要从 Add 请求中提取的关键字段

```
prompt_token_ids   ← 已分词的输入, 这是你需要送进模型的
max_tokens         ← 最多生成多少个 token
temperature, top_p, top_k, seed  ← 采样控制
stop_token_ids, _eos_token_id   ← 停止条件
request_id         ← 输出时用来匹配回这个请求
client_index       ← 输出时用来路由到正确的 PUSH socket
```

### 2.2 Abort 请求 (0x01)

中止一个或多个进行中的请求。

**负载**: MessagePack 字符串数组:

```json
["request-id-1", "request-id-2"]
```

收到后立即停止指定请求的生成并返回终止输出。

### 2.3 Utility 请求 (0x03)

引擎管理/查询调用。

**负载**: MessagePack 数组:

```
[client_index: u32, call_id: u64, method_name: String, args: ...]
```

需要支持的方法:

| 方法 | args | 返回值 | 说明 |
|------|------|--------|------|
| `get_supported_tasks` | `()` | `["generate"]` | 声明引擎支持的任务 |
| `is_sleeping` | `()` | `false` | 查询引擎是否休眠 |
| `reset_prefix_cache` | `(reset_running_requests: bool, reset_connector: bool)` | `true` | 清理前缀缓存 |
| `reset_mm_cache` | `()` | `null` | 清理多模态缓存 |
| `reset_encoder_cache` | `()` | `null` | 清理 encoder 缓存 |
| `sleep` | `(level: u32, mode: PauseMode)` | `null` | 将引擎置为休眠 |
| `wake_up` | `(tags: [String]\|null)` | `null` | 唤醒引擎 |
| `profile` | `(start: bool, profile_prefix: String\|null)` | `null` | 开始/停止 profiling |
| `is_scheduler_paused` | `()` | `false` | 查询调度器是否暂停 |
| `pause_scheduler` | `(mode: PauseMode, clear_cache: bool)` | `null` | 暂停调度器 |
| `resume_scheduler` | `()` | `null` | 恢复调度器 |
| `execute_dummy_batch` | `()` | `null` | 执行一次空 batch (用于 GPU warmup) |
| `collective_rpc` | `(method, timeout, args, kwargs)` | 引擎返回 | 自定义 RPC |
| `add_lora` | `(lora_request)` | `true` | 加载 LoRA adapter |
| `remove_lora` | `(lora_id: u64)` | `true` | 卸载 LoRA adapter |

`PauseMode` 是一个 u8: `0 = NO_ABORT` (等待已有请求完成), `1 = ABORT` (立即中止)。

**Utility 返回值编码**: 返回值需要包装在 `UtilityResultEnvelope` 中:
```json
{
    "result": <你的返回值>,
    "failure_message": null
}
```

对于最简单的实现, 你可以:
- `is_sleeping` → `false`
- `get_supported_tasks` → `["generate"]`
- `is_scheduler_paused` → `false`
- 所有其他方法 → 返回成功 (空值或 `true`)

### 2.4 StartDpWave 请求 (0x02)

仅用于数据并行模式。单引擎部署可以安全忽略此消息。

---

## 第三步: 输出协议 (引擎 → 前端)

引擎通过 **PUSH socket** 将输出发送回前端。格式为 **单个 MessagePack 帧**:

### EngineCoreOutputs 信封

```json
{
    "engine_index": 0,
    "timestamp": 1752931200.5,
    "outputs": [
        {
            "request_id": "chatcmpl-abc123-a1b2c3d4",
            "new_token_ids": [42, 99],
            "new_logprobs": null,
            "new_prompt_logprobs_tensors": null,
            "pooling_output": null,
            "finish_reason": 0,
            "stop_reason": 151645,
            "events": null,
            "kv_transfer_params": null,
            "ec_transfer_params": null,
            "trace_headers": null,
            "prefill_stats": null,
            "routed_experts": null,
            "num_nans_in_logits": 0
        }
    ],
    "finished_requests": ["chatcmpl-abc123-a1b2c3d4"],
    "scheduler_stats": null,
    "utility_output": null
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `engine_index` | u32 | 产生此输出的引擎索引 |
| `timestamp` | f64 | Unix 秒级时间戳 (引擎侧) |
| `outputs` | [OutputObject] | 本批次的请求输出列表 |
| `finished_requests` | [String]\|null | 本批次完成的 request_id 集合 |
| `scheduler_stats` | Map\|null | 调度器统计 (用于前端负载均衡) |
| `utility_output` | Map\|null | Utility 调用结果 (见下文) |

### OutputObject 详细字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `request_id` | String | - | 必须匹配 Add 请求中的 request_id |
| `new_token_ids` | [u32] | [] | 此 step 新生成的 token ID |
| `new_logprobs` | 见下方 | null | 采样 logprobs |
| `new_prompt_logprobs_tensors` | 见下方 | null | Prompt logprobs (仅第一个输出) |
| `finish_reason` | u8\|null | null | 终止原因 (非 null = 最后一个输出) |
| `stop_reason` | u32\|String\|null | null | 触发停止的 token ID 或字符串 |
| `prefill_stats` | Map\|null | null | Prefill 统计 |
| `num_nans_in_logits` | u32 | 0 | NaN 检测, 非零表示异常 |

### finish_reason 取值

| 值 | 含义 | 示例场景 |
|----|------|----------|
| `null` | 未终止, 还有更多 token | 正常生成中的中间步骤 |
| `0` | Stop | 遇到 EOS/stop token/stop string |
| `1` | Length | 达到 max_tokens |
| `2` | Abort | 客户端主动中止 |
| `3` | Error | 请求级内部错误 (可重试) |
| `4` | Repetition | 检测到重复模式 |

### stop_reason 取值

当 `finish_reason == 0 (Stop)` 或 `4 (Repetition)` 时提供:

- `u32`: 触发停止的 token ID (例如 EOS token = 151645)
- `String`: 触发停止的字符串 (例如 `"</s>"`)
- `null`: EOS 触发的停止, 无具体 stop_reason

### 输出发送策略

1. **中间步骤**: 每步发送 `outputs` 非空但 `finish_reason` 为 null
2. **最终步骤**: `finish_reason` 非 null, `finished_requests` 包含此请求 ID
3. **可以批量**: `outputs` 可以包含多个请求的输出
4. **心跳/空批次**: `outputs` 为空但 `scheduler_stats` 非 null (可选, 用于负载均衡)
5. **Abort 响应**: 收到 Abort 后发送 `finish_reason: 2` 并加入 `finished_requests`

### Utility 响应

当引擎收到 Utility 请求并完成处理后, 通过同一 PUSH socket 发送:

```json
{
    "engine_index": 0,
    "timestamp": 1752931200.5,
    "utility_output": {
        "call_id": 42,
        "failure_message": null,
        "result": {
            "result": <具体返回值>
        }
    }
}
```

`call_id` 必须匹配 Utility 请求中的 `call_id`。

### Scheduler Stats (可选, 推荐)

用于前端的负载均衡, 在 `scheduler_stats` 字段中发送:

```json
{
    "num_running_reqs": 5,
    "num_waiting_reqs": 2,
    "gpu_cache_usage": 0.35,
    "waiting_lora_adapters": {},
    "running_lora_adapters": {}
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `num_running_reqs` | u32 | 当前正在执行的请求数 |
| `num_waiting_reqs` | u32 | 排队等待的请求数 |
| `gpu_cache_usage` | f64 | GPU KV cache 使用率 (0.0-1.0) |
| `waiting_lora_adapters` | Map | 等待中的 LoRA adapter |
| `running_lora_adapters` | Map | 运行中的 LoRA adapter |

---

## 第四步: Logprobs (可选)

当 `sampling_params.logprobs` 不为 null 时, 需要在输出中包含 logprobs。

### new_prompt_logprobs_tensors (仅第一个输出)

```json
[
    {
        "prompt_token_ids": [1, 2, 3],
        "logprobs": {
            "0": {
                "1": {"logprob": -0.1, "rank": 1},
                "5": {"logprob": -2.3, "rank": 2}
            },
            "1": {
                "2": {"logprob": -0.05, "rank": 1}
            }
        }
    }
]
```

### new_logprobs (每步输出可选)

```json
[
    {
        "logprobs": {
            "0": {
                "42": {"logprob": -0.1, "rank": 1},
                "7":  {"logprob": -2.3, "rank": 2}
            }
        }
    }
]
```

其中 `"0"`, `"1"` 是生成位置索引 (字符串形式的 u32), key 为 token ID, value 含 `logprob` (f32) 和 `rank` (u32)。

---

## 完整示例: Python 最小实现

```python
import zmq
import msgpack
import time
import uuid

ctx = zmq.Context()

# ===== 1. 握手 =====
HANDSHAKE_ADDR = "tcp://127.0.0.1:29550"
ENGINE_INDEX = 0
ENGINE_ID = ENGINE_INDEX.to_bytes(2, "little")

hs = ctx.socket(zmq.DEALER)
hs.setsockopt(zmq.IDENTITY, ENGINE_ID)
hs.connect(HANDSHAKE_ADDR)

# 发送 HELLO
hs.send(msgpack.packb({"status": "HELLO", "headless": True, "local": False}))

# 接收 INIT
init = msgpack.unpackb(hs.recv())
input_addr = init["addresses"]["inputs"][0]
output_addr = init["addresses"]["outputs"][0]

# ===== 2. 连接数据平面 =====
dealer = ctx.socket(zmq.DEALER)
dealer.setsockopt(zmq.IDENTITY, ENGINE_ID)
dealer.connect(input_addr)

# 发送 ReadyResponse 注册帧
dealer.send(msgpack.packb({
    "max_model_len": 131072,
    "num_gpu_blocks": 65536,
    "block_size": 16,
    "dtype": "bfloat16",
    "vllm_version": "my-backend-1.0",
    "world_size": 1,
    "data_parallel_size": 1,
    "dp_stats_address": None,
    "kv_cache_size_tokens": None,
    "kv_cache_max_concurrency": None,
}))

push = ctx.socket(zmq.PUSH)
push.connect(output_addr)

# 发送 READY
hs.send(msgpack.packb({"status": "READY", "headless": True, "local": False}))
hs.close()  # 不再需要 handshake

print(f"Engine {ENGINE_INDEX} connected, ready for requests")

# ===== 3. 主循环 =====
active_requests = {}  # request_id -> 推理状态

while True:
    # 非阻塞轮询, timeout 后可以做一步推理
    if dealer.poll(timeout=10):  # 10ms 超时
        frames = dealer.recv_multipart()
        req_type = frames[0][0]
        payload = msgpack.unpackb(frames[1])

        if req_type == 0x00:  # Add
            request_id = payload[0]
            prompt_tokens = payload[1] or []
            sampling_params = payload[3] or {}
            max_tokens = sampling_params.get("max_tokens", 16)
            client_index = payload[11]

            active_requests[request_id] = {
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "generated": 0,
                "client_index": client_index,
            }
            print(f"  New request: {request_id}, prompt_len={len(prompt_tokens)}, max_tokens={max_tokens}")

        elif req_type == 0x01:  # Abort
            for rid in payload:
                if rid in active_requests:
                    # 发送 Abort 终止输出
                    push.send(msgpack.packb({
                        "engine_index": ENGINE_INDEX,
                        "timestamp": time.time(),
                        "outputs": [{
                            "request_id": rid,
                            "new_token_ids": [],
                            "finish_reason": 2,  # Abort
                        }],
                        "finished_requests": [rid],
                    }))
                    del active_requests[rid]

        elif req_type == 0x03:  # Utility
            client_index, call_id, method_name = payload[0], payload[1], payload[2]
            result = None
            if method_name == "is_sleeping":
                result = False
            elif method_name == "get_supported_tasks":
                result = ["generate"]
            elif method_name == "is_scheduler_paused":
                result = False
            # ... 其他方法返回 null/true

            push.send(msgpack.packb({
                "engine_index": ENGINE_INDEX,
                "timestamp": time.time(),
                "utility_output": {
                    "call_id": call_id,
                    "failure_message": None,
                    "result": {"result": result},
                },
            }))

    # ===== 4. 推理步进 =====
    finished = []
    outputs = []
    for rid, req in list(active_requests.items()):
        # 调用你的推理引擎生成一个 token
        token = your_inference_engine_step(req)  # 返回 u32

        req["generated"] += 1
        is_finished = req["generated"] >= req["max_tokens"]

        outputs.append({
            "request_id": rid,
            "new_token_ids": [token],
            "finish_reason": 1 if is_finished else None,  # Length
        })
        if is_finished:
            finished.append(rid)

    # 清理已完成的请求
    for rid in finished:
        del active_requests[rid]

    if outputs:
        push.send(msgpack.packb({
            "engine_index": ENGINE_INDEX,
            "timestamp": time.time(),
            "outputs": outputs,
            "finished_requests": finished if finished else None,
        }))
```

---

## ZMQ Socket 类型总结

| Socket | 类型 | 方向 | 说明 |
|--------|------|------|------|
| Handshake | DEALER | 双向 | STARTUP 时连接, READY 后断开 |
| Input | DEALER | recv only | 接收前端的 Add/Abort/Utility 请求 |
| Output | PUSH | send only | 推送生成结果和 Utility 响应 |

ZMQ DEALER socket 必须设置 identity (2 字节小端编码的 engine index)。ROUTER socket 使用此 identity 将请求路由到正确的引擎。

---

## 启动顺序

1. **首先**启动 Rust 前端 (它会创建 handshake socket):
   ```bash
   vllm-rs serve <model> --data-parallel-size 1 --data-parallel-size-local 0 --handshake-port 29550
   ```

2. **然后**启动你的引擎, 连接到 `tcp://127.0.0.1:29550`

3. 对于多引擎部署: 设置 `--data-parallel-size N`, 启动 N 个引擎实例 (每个使用不同的 engine index 作为 ZMQ identity)

---

## 数据并行

多引擎部署下:

- `data_parallel_size` 告诉前端预期有多少个引擎
- 每个引擎使用唯一 engine_index 作为 ZMQ identity
- 前端通过 `data_parallel_rank` 字段路由请求
- 启用了 DP coordinator 时, 引擎需要:
  1. 连接 coordinator SUB socket
  2. 等待 coordinator 广播 `READY` barrier (单帧, 内容为 `b"READY"`)
  3. 然后才能发送 READY 到 handshake socket

对于最简单的场景 (单引擎), 这些都可以忽略。

---

## 常见问题

**Q: 可以跳过 handshake 直接用 data plane socket 吗?**

如果你能预先知道 data plane 地址, 可以使用 "bootstrapped" 模式。先启动前端, 然后引擎直接连接 input/output socket 并发送 ReadyResponse 注册帧, 跳过 handshake socket。这适用于固定端口部署。

**Q: 引擎输出频率应该是多少?**

每生成一个 token 就发一个 output, 或把一小批 token (speculative decoding) 放在一个 output 里。不要缓存太多 token 再发, 否则会削弱流式体验。

**Q: 如何测试?**

用 vLLM 自带的 mock engine 作为参考:
```bash
# 终端 1: 启动 Rust 前端
vllm-rs serve Qwen/Qwen3-0.6B --data-parallel-size 1 --data-parallel-size-local 0 --handshake-port 29550

# 终端 2: 启动 mock engine 测试连通性
cargo run -p vllm-mock-engine -- --handshake-address tcp://127.0.0.1:29550

# 终端 3: 发请求
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

**Q: 我需要实现 logprobs 吗?**

不必。如果你的后端不支持 logprobs, 将 `new_logprobs` 和 `new_prompt_logprobs_tensors` 字段设为 null 即可。前端会处理用户请求了 logprobs 但引擎未返回的情况。

**Q: 多模态输入怎么处理?**

多模态在前端侧通过 `mm_features` 字段传递。如果你只做纯文本推理, 忽略此字段 (始终为 null)。
