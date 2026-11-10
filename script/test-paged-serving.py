#!/usr/bin/env python3
"""Generate reproducible model-serving regression configurations (no downloads).

Run each CONFIG with tests/test_model_serving under the site's NPU launcher.
The pressure case fits one maximal context; the roomy control checks the same
workload without recomputation. Dense is the pre-paging storage reference.
"""

import argparse
import copy
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/paged-validation"))
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--tokenize", action="store_true",
                        help="include natural-language probes using local tokenizers")
    parser.add_argument("--extra", action="store_true",
                        help="also generate eager, no-prefix performance and 8K/16-slot stress cases")
    args = parser.parse_args()
    if args.context < 384:
        parser.error("context must be at least 384")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    def save(name, cfg):
        path = args.output_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(path)

    models = {
        "qwen3": ("example_qwen3_server.json", 1, False),
        "llama": ("example_llama.json", 1, False),
        "deepseek": ("example_deepseek_v2_server.json", 1, False),
        "qwen35": ("example_qwen3_5_server.json", 1, True),
        "qwen3-moe": ("example_qwen3_moe_server.json", 2, False),
        "qwen35-moe": ("example_qwen3_5_moe_tp.json", 2, True),
        "qwen38": ("example_qwen3_8_server.json", 2, True),
    }
    for name, (source, tp, recurrent) in models.items():
        base = json.loads((root / "backend/config" / source).read_text())
        for mode in ("pressure", "roomy", "dense"):
            cfg = copy.deepcopy(base)
            engine = cfg["engine"]
            engine.update(n_context=args.context, max_num_request=4,
                          max_num_batched_tokens=32 if recurrent else 128,
                          tensor_parallel_size=tp)
            engine["optimization"].update(fuse_attention=True)
            pages = (args.context + 127) // 128
            engine["kv_cache"] = dict(paged=mode != "dense", prefix_caching=True,
                                      block_size=128, num_pages=pages * (4 if mode == "roomy" else 1))
            cfg["test"] = dict(requests=6, generated=12, expect_preemption=mode == "pressure")
            if args.tokenize:
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(Path(cfg["model"]) / "tokenizer.json"))
                probes = ["The capital of France is", "Question: What is 2 + 2?\nAnswer:",
                          "Question: Name the three primary colors of light.\nAnswer:"]
                cfg["test"]["smoke_prompts"] = [tokenizer.encode(text).ids for text in probes]
            save(f"{name}-{mode}", cfg)
            if not args.extra:
                continue
            if mode == "dense" and tp == 1:
                eager = copy.deepcopy(cfg)
                eager["engine"].update(n_context=384,
                                       max_num_batched_tokens=4 if recurrent else 32)
                eager["engine"]["optimization"]["fuse_attention"] = False
                if recurrent:
                    eager["engine"]["optimization"].update(
                        fused_gdn_attention=False, ge_precision_mode="origin")
                eager["test"]["requests"] = 4
                save(f"{name}-eager", eager)
            if name == "qwen3" and mode == "roomy":
                cfg["engine"]["kv_cache"]["prefix_caching"] = False
                save("qwen3-no-prefix", cfg)
            if name in ("qwen3", "qwen35") and mode == "pressure":
                replay = copy.deepcopy(cfg)
                replay["engine"]["n_context"] = 384
                replay["engine"]["kv_cache"]["num_pages"] = 3
                replay["test"].update(requests=2, generated=128, lengths=[200, 1])
                save(f"{name}-decode-replay", replay)
            if name == "qwen3" and mode == "pressure":
                cfg["engine"].update(n_context=8192, max_num_request=16,
                                     max_num_batched_tokens=256)
                cfg["engine"]["kv_cache"]["num_pages"] = 64
                cfg["test"].update(requests=24, generated=16,
                                   lengths=[8160, 7936, 8128, 7680])
                save("qwen3-extreme", cfg)


if __name__ == "__main__":
    main()
