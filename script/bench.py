#!/usr/bin/env python3
"""Simple benchmark: TTFT, throughput, concurrency for vLLM-compatible API."""

import argparse
import concurrent.futures
import json
import threading
import time

import requests

# First-token arrival times across all requests (streaming mode), used to
# compute the decode-throughput window. Guarded by first_token_lock.
first_token_times = []
first_token_lock = threading.Lock()


def send_request(
    base_url: str, model: str, prompt: str, max_tokens: int, stream: bool,
    enable_thinking: bool = False
):
    """Return (ttft_ms, total_ms, tokens, text, prompt)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if not enable_thinking:
        # 关闭 thinking。Qwen3 / Qwen3.5 / Qwen3.8 的 HF 模板可能默认开启思考
        # （Qwen3.8 默认 effort=xhigh），推理 token 会混进吞吐/TTFT 指标；模板里的
        # `enable_thinking` 是标准开关，模板不支持时该 kwarg 被忽略。
        # 注意：`reasoning_effort` 只对 Harmony 渲染器（gpt-oss）生效，对 HF 渲染的
        # qwen 系无效——之前这里写它就是没起作用。
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    first_token = None
    count = 0
    text = ""

    if stream:
        with requests.post(
            f"{base_url}/v1/chat/completions", json={**payload, "stream": True}, stream=True
        ) as resp:
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                data = line[6:]
                if data == b"[DONE]":
                    break
                if first_token is None:
                    first_token = time.time()
                    with first_token_lock:
                        first_token_times.append(first_token)
                try:
                    chunk = data.decode()
                    if '"content":"' in chunk:
                        text += chunk.split('"content":"')[1].split('"')[0]
                    count += 1
                except Exception:
                    pass
    else:
        resp = requests.post(
            f"{base_url}/v1/chat/completions", json={**payload, "stream": False}
        )
        data = resp.json()
        count = data["usage"]["completion_tokens"]
        text = data["choices"][0]["message"]["content"]
        first_token = time.time()

    t1 = time.time()
    ttft = (first_token - t0) * 1000 if first_token else (t1 - t0) * 1000
    return ttft, (t1 - t0) * 1000, count, text, prompt


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM API")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Base URL")
    parser.add_argument("--model", default="qwen3-0.6b", help="Model name")
    parser.add_argument("--prompt", default="", help="Single prompt (overrides built-in list)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens per request")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests per round")
    parser.add_argument("--rounds", type=int, default=4, help="Number of rounds")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Let the model think (default: off, i.e. "
                             "chat_template_kwargs.enable_thinking=false; only "
                             "affects templates that support the switch)")
    parser.add_argument("--repeat", action="store_true", help="Use the same prompt for all requests")
    parser.add_argument("--json", action="store_true",
                        help="Print a single JSON metrics object as the final stdout line")
    args = parser.parse_args()

    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the three laws of thermodynamics?",
        "Describe the process of photosynthesis.",
        "Who was Alan Turing and what did he contribute?",
        "List five healthy breakfast ideas.",
        "How does a transformer model work in deep learning?",
        "What is the difference between Python and C++?",
        "Explain the significance of the Turing Test.",
        "Write a recipe for chocolate chip cookies.",
        "What causes the northern lights?",
        "Summarize the plot of Romeo and Juliet.",
        "How do airplanes stay in the air?",
        "What is blockchain technology?",
        "Describe the water cycle.",
        "What is the speed of light?",
        "Explain the Big Bang theory.",
        "Write a haiku about nature.",
        "What are black holes?",
        "How does DNA replication work?",
        "Describe the solar system.",
        "What is machine learning?",
        "Explain the greenhouse effect.",
        "Who wrote the Odyssey?",
        "What is Pythagoras' theorem?",
        "How do vaccines work?",
        "Describe the Great Wall of China.",
        "What is the periodic table?",
        "Explain supply and demand.",
        "Write a limerick about coding.",
        "What causes earthquakes?",
    ]

    if args.prompt:
        prompts = [args.prompt]

    total_requests = args.rounds * args.concurrency
    stream = not args.no_stream
    print(f"Benchmark: {args.rounds} rounds × {args.concurrency} = {total_requests} requests, "
          f"stream={stream}, max_tokens={args.max_tokens}, "
          f"thinking={'on' if args.enable_thinking else 'off'}")
    print(f"Prompts: {len(prompts)} unique, repeat={'yes' if args.repeat else 'no'}")

    ttfts = []
    totals = []
    tokens = []

    t_start = time.time()
    first_token_times.clear()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for i in range(total_requests):
            prompt = prompts[0] if args.repeat else prompts[i % len(prompts)]
            futures.append(pool.submit(send_request, args.url, args.model,
                prompt, args.max_tokens, stream, args.enable_thinking))
        results = []
        for f in concurrent.futures.as_completed(futures):
            r = f.result()
            results.append(r)
            ttfts.append(r[0])
            totals.append(r[1])
            tokens.append(r[2])
    t_end = time.time()
    elapsed = t_end - t_start
    with first_token_lock:
        ft_times = list(first_token_times)
    # decode window: first token arrival (across all requests) → end
    decode_time = (t_end - min(ft_times)) if ft_times else elapsed

    # Print first N responses for correctness check
    show_n = min(5, len(results))
    print(f"\n--- First {show_n} Responses ---")
    for i, (ttft, total, n, text, prompt) in enumerate(results[:show_n]):
        print(f"[{i}] Q: {prompt[:60]}")
        print(f"    A: {text.strip()}")
        print(f"    ({n} tokens, ttft={ttft:.0f}ms, total={total:.0f}ms)\n")
    if len(results) > show_n:
        print(f"... ({len(results) - show_n} more responses omitted)\n")

    ttfts.sort()
    totals.sort()
    total_tokens = sum(tokens)

    tp = total_tokens / elapsed
    qps = total_requests / elapsed
    avg_lat = sum(totals) / len(totals)
    avg_ttft = sum(ttfts) / len(ttfts)

    decode_tp = total_tokens / max(decode_time, 0.001)

    print(f"\n{'='*60}")
    print(f"Duration:            {elapsed:.2f}s")
    print(f"Requests:            {total_requests} (concurrency={args.concurrency})")
    print(f"Total tokens:        {total_tokens}")
    print(f"Throughput:          {tp:.1f} tok/s  (total)")
    print(f"Decode throughput:    {decode_tp:.1f} tok/s  (first-token→end)")
    print(f"QPS:                 {qps:.2f} req/s")
    print(f"Per-request:         {avg_lat:.0f}ms avg, {totals[len(totals)//2]:.0f}ms p50")
    avg_tokens = max(1, total_tokens // total_requests)
    print(f"Per-token:           {avg_lat / avg_tokens:.1f}ms avg (decode)")
    print(f"TTFT:                {avg_ttft:.0f}ms avg, {ttfts[len(ttfts)//2]:.0f}ms p50, "
          f"{ttfts[int(len(ttfts)*0.95)]:.0f}ms p95")

    if args.json:
        metrics = {
            "url": args.url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "rounds": args.rounds,
            "stream": stream,
            "thinking": args.enable_thinking,
            "duration_s": round(elapsed, 3),
            "requests": total_requests,
            "total_tokens": total_tokens,
            "throughput_tok_s": round(tp, 3),
            "decode_throughput_tok_s": round(decode_tp, 3),
            "qps": round(qps, 4),
            "avg_latency_ms": round(avg_lat, 3),
            "p50_latency_ms": round(totals[len(totals) // 2], 3),
            "per_token_ms": round(avg_lat / avg_tokens, 4),
            "ttft_avg_ms": round(avg_ttft, 3),
            "ttft_p50_ms": round(ttfts[len(ttfts) // 2], 3),
            "ttft_p95_ms": round(ttfts[min(int(len(ttfts) * 0.95), len(ttfts) - 1)], 3),
        }
        print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
