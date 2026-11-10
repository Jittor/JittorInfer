#!/usr/bin/env python3
"""Smoke-test a running server: send a few chat requests, check each returns text.

    python script/check-serving.py --model qwen3-0.6b [-n 4]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request

PROMPTS = ["What is 2 + 3?", "Name the capital of France.", "Say hello.", "Count to five."]


def ask(url, model, prompt):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32, "temperature": 0}
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    # Bypass any HTTP proxy for local servers.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=300) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("-n", type=int, default=4, help="concurrent requests")
    args = parser.parse_args()

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.n)]
    with ThreadPoolExecutor(args.n) as pool:
        results = pool.map(lambda p: ask(args.url.rstrip("/"), args.model, p), prompts)
        failures = 0
        for prompt, text in zip(prompts, results):
            ok = bool(text and text.strip())
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'} {prompt!r} -> {text!r}")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
