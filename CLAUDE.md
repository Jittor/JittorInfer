# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

JittorInfer is a C++ LLM inference framework forked/derived from llama.cpp, retargeted to **Huawei Ascend NPUs** via the CANN toolkit. It serves OpenAI-compatible HTTP endpoints (`llama-server`) and supports DeepSeek-V2/V2-Lite, Qwen2/Qwen3, and Qwen3-MoE models. Multi-NPU execution uses MPI (`llama-mpi-server`) for tensor parallelism (TP) and a simple form of expert parallelism (EP).

Primary working tree is `/beegfs/home/yzhang/jittorinfer-online/jittorinfer`. There is a mirror at `/home/yzhang/jittorinfer-online/jittorinfer` listed as an additional working directory.

## Build / format / test

Standard build (Debug type, `-O2` C/CXX flags — this is what `build.sh` uses):

```bash
cmake -B build -DGGML_CUDA=OFF -DGGML_CANN=on -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-O2" -DCMAKE_C_FLAGS="-O2"
cmake --build build -j64
```

Release builds use `-DCMAKE_BUILD_TYPE=Release` (see `examples/README.md`). A commented-out Bisheng compiler invocation in `build.sh` exists but is **not supported** because Bisheng currently lacks MPI.

Formatting (clang-format-18 must be on PATH):

```bash
./format.sh   # formats common/, examples/, ggml/, src/, tests/
```

Run the server (single NPU):

```bash
ASCEND_RT_VISIBLE_DEVICES=0 ./build/bin/llama-server --config configs/config_<model>.yaml
```

Multi-NPU TP (layer split, default):

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./build/bin/llama-server --config configs/config_deepseek_v2.yaml
```

Multi-NPU EP+TP (uses MPI launcher and a separate binary):

```bash
mpirun -np 2 ./build/bin/llama-mpi-server --config configs/config_deepseek_v2_lite_ep.yaml
```

Tests (enabled by `LLAMA_BUILD_TESTS=ON`, default ON). The `tests/main/` binaries hard-code model paths and other hyperparameters inside the source files — edit and rebuild to change them:

```bash
./build/bin/llama-test-single-sequence   # tests/main/single_sequence.cpp
./build/bin/llama-test-multi-sequence    # tests/main/multi_sequence.cpp
```

Operator tests live in `tests/ops/` (e.g. `flash_attn_aclnn.cpp`, `ops_test.cpp`, `test_ops_speed.cpp`) and are also produced under `build/bin/`.

Benchmarking is driven by `script/test_benchmark.sh`, which shells out to an external `inference-benchmarker` binary against a running server.

## Architecture

The codebase keeps llama.cpp's three-layer split: `ggml` (tensor library + backends), `src/llama-*` (model loading + per-architecture graph construction + scheduling), and `examples/{server,mpi_server}` (HTTP entry points). The non-trivial parts to know:

### Per-architecture graph builders are duplicated for "eager" vs "GE" execution

For each supported model there are two builder translation units:

- `llama-graph-<arch>.cpp` — builds a ggml compute graph that runs op-by-op via the CANN backend (ACLNN ops).
- `llama-graph-<arch>ge.cpp` — builds the same forward pass for the **Ascend Graph Engine** path, which compiles the whole graph ahead of time and is selected via the YAML key `enable_ge: true`.

Architectures present today: `deepseek2`, `qwen2`, `qwen3`, plus `qwen3moe` (GE-only file `llama-graph-qwen3moege.cpp`). When adding or modifying a model, expect to touch both the eager and GE builder, plus shared helpers in `llama-graph-utils.cpp`, `llama-graph-attn.cpp`, and `llama-graph-ffn.cpp`. The dispatcher entry point is `llama_graph_builder::llama_build_graph` in `src/llama-graph-builder.{h,cpp}`.

DeepSeek-specific accelerations toggled from YAML: `enable_mla` (multi-head latent attention), `enable_fused_moe`, `enable_ge`. Qwen3 currently relies on `enable_ge` for the GE path; recent commits also added merged-QKV support for Qwen3-MoE GE (see `git log` around `qwen3 moe ge`).

### Execution loop

`llama-context.cpp` plus `llama-execute-decode.cpp` and `llama-execute-kvupdate.cpp` drive prompt processing and token-by-token decode. KV cache and defrag live in `llama-kv-cache.cpp` and `llama-graph-defrag.cpp` (defrag threshold is `defrag_thold` in the YAML). Continuous batching is enabled via `cont_batching: true` in the config, and `n_parallel` caps concurrent sequences (currently must be `< 32`).

Sampling can run on-NPU via `presample_count` (set to `-1` to disable). Set `offload_input: true` to keep input tensors on the device.

### CANN backend

Lives entirely under `ggml/src/ggml-cann/`:

- `ggml-cann.cpp` is the backend registration entry point (selected by `-DGGML_CANN=on`).
- `aclnn_ops.{h,cpp}` and `acl_tensor.{h,cpp}` wrap ACLNN ops and tensor descriptors used by the eager path.
- `ascend_graph*.{h,cpp}` implement the Graph Engine path used when `enable_ge: true`. `fusion_switch.cfg` and `ge_check_op.json` (repo root) tune GE fusion behavior.
- `kernels/` contains hand-written AscendC kernels for ops not covered by ACLNN (custom RoPE, dup, get_row variants, quantization to/from q4_0/q8_0). They are compiled separately — see `ggml/src/ggml-cann/kernels/CMakeLists.txt`.
- `rope_cache.{h,cpp}` precomputes RoPE tables shared across the graph.

### Server layer

`examples/server/server.cpp` is the single-process HTTP server; `examples/mpi_server/server.cpp` is the MPI-aware variant launched via `mpirun` for TP/EP. Both expose `/v1/chat/completions` and `/v1/completions` and load all parameters from a YAML file via `--config` (parsed in `src/llama-params.cpp`).

`common/` contains the locally vendored chat templating, sampling, and JSON-schema-to-grammar code (the `*_local.cpp` naming is intentional — these are decoupled from llama.cpp's upstream `common/`).

### Config files

`configs/*.yaml` are the source of truth for runtime knobs. Naming convention is `config_<model>[_tp|_ep].yaml`. The full parameter list is documented in `examples/README.md`; key flags worth knowing when debugging perf are `enable_mla`, `enable_fused_moe`, `enable_ge`, `enable_cann_flash_attention`, `presample_count`, and `defrag_thold`.

### Third-party

`third_party/` is on the include path for `llama` (see `src/CMakeLists.txt`). MPI is a hard dependency — `find_package(MPI REQUIRED)` is at the top of the root `CMakeLists.txt` even when MPI features are unused.

## Conventions

- Source layout follows llama.cpp upstream as closely as possible; new model support should follow the existing `llama-graph-<arch>.cpp` + `llama-graph-<arch>ge.cpp` split rather than introducing a new shape.
- The repo runs an "Auto-format code with clang-format" commit periodically — keep `format.sh` clean before pushing.
- Models are GGUF format. `examples/convert_qwen3_to_gguf.py` is the helper for Qwen3 conversion; see `examples/README_gguf.md` for context.
