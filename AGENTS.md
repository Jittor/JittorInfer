# AGENTS.md

Guidance for AI coding agents working in this repository.

## Shared and local guidance

This tracked file contains repository-wide workflow rules, engineering constraints,
and documentation entry points. Do not add personal environment names, installation
paths, scheduler partitions, machine inventories, or timing estimates here.

- Read `AGENTS.local.md` when present before running commands. It is untracked and
  contains this machine's activation commands, resource allocation, and workarounds.
- In a linked worktree without that file, read the primary checkout's
  `AGENTS.local.md` if available. Find the primary checkout from the first worktree
  record in `git worktree list --porcelain`, not from an assumed parent directory.
  A worktree-specific local file may reference that profile and override local settings.
- Use [AGENTS.local.example.md](AGENTS.local.example.md) to document a new machine.
  Local settings do not change repository invariants or grant execution permissions.
  If missing, inspect the actual environment; do not guess paths or environment names.
- Reusable installation and dependency details belong in [docs/build.md](docs/build.md)
  and deployment docs. Recipes are examples, not proof of an installed environment.

## Worktree workflow

**Keep the primary checkout on `master`.** Tracked edits, commits, tests, and PR work
belong in a task-specific linked worktree, not the primary checkout.

1. Inspect `git worktree list --porcelain` and `git status --short --branch` before
   changing branches. Preserve unrelated edits, untracked files, and submodule state.
2. Reuse the worktree already owning a PR branch. For a new task, create a separate
   branch and worktree from the intended base; use the `codex/` branch prefix by default.
   `.worktrees/` is ignored for local worktree storage; sibling locations also work.
3. Never force-checkout a branch already in use elsewhere. Do not remove another
   task's worktree, share its build directory, or reset a submodule to clean status.
4. New worktrees do not inherit initialized submodules, build outputs, or ignored
   local files. Initialize dependencies at their recorded commits when needed, and
   link or create an ignored `AGENTS.local.md` without overwriting an existing one.
5. Keep local profiles and worktree directories out of commits. Commit a submodule
   change in its own repository first, then update the parent pointer deliberately.

## Project and layout

**JittorInfer** is a pre-1.0 LLM inference engine for Huawei Ascend NPUs. APIs and wire
formats may change during development. The Rust frontend serves the OpenAI-compatible
HTTP API, tokenizes, and renders chat templates. The C++ backend owns models, compiles
Ascend graphs, and runs batched inference. They communicate over ZMQ + MessagePack
using vLLM's engine-core protocol; this explains the `vllm-compat` naming.

| Path | Role |
| --- | --- |
| `backend/` | C++ inference engine, the main code owned by this repository |
| `forch/` | Submodule: C++20 tensor/graph library targeting Ascend GE |
| `rust-frontend/` | Submodule: Rust HTTP frontend forked from vLLM |
| `third_party/libzmq` | Submodule: ZeroMQ |
| `script/` | Build, run, benchmark and regression-config scripts |
| `docs/` | Architecture, cache, configuration, deployment and testing references |

`frontend/` is a dead legacy Python demo. Do not read, modify, extend, or cite
`frontend/main.py`; the working frontend is `rust-frontend/`.

## Build, run, and verify

Prepare the environment described by the local profile before build, test, format,
or run commands. C++ builds need CANN, a C++20 toolchain, and MPI development libraries;
model/operator execution needs an allocated NPU. Host CTest checks do not need an NPU.
Do not change dependencies based solely on a package listing: check actual imports
and tool paths. Compiler ABI and protobuf ownership constraints are in
[docs/build.md](docs/build.md).

Run commands from the task worktree root. Build order is third-party libraries,
Forch, then backend; frontend builds separately. Keep each worktree's build outputs
isolated. `BUILD_DIR` is a common root with `third_party`, `forch`, and `backend`
subdirectories, not a single-component override. Details: [build guide](docs/build.md).

| Task | Entry point |
| --- | --- |
| Server | `bash script/run-server.sh <config.json>`; [configuration](docs/configuration.md) and [Docker](docs/docker.md) |
| Host tests | Enable `BUILD_TESTING=ON`, then `ctest --test-dir build/backend --output-on-failure` |
| NPU verification | [Testing guide](docs/testing.md): attention oracles, GDN lifecycle, full-checkpoint serving and preemption |
| Model debugging | `test_model_forward` drives a runner directly, without HTTP/ZMQ |
| Benchmarks | [Benchmark guide](script/benchmark/README.md); keep host settings in local profiles |
| Backend formatting | `bash script/format.sh`, requiring `clang-format-18` |
| Forch / Rust tests | Follow submodule instructions; Rust uses `cargo nextest run` |

Load MPI for single-card backend builds and runs as well as TP. The Rust build needs
`protoc` and its standard `.proto` files, with `PROTOC_INCLUDE` set to their include
root, even when gRPC is unused. Read the checked-out frontend's `rust-toolchain.toml`
for its version pin. Do not treat the submodule's standalone environment as the
parent project's deployment environment.

Scale verification to the change. Report missing tools and skipped tests honestly;
do not claim an old build or another worktree's result validates new code.

**CI can push formatting commits back to the branch.** Run `script/format.sh` before
pushing and inspect the diff. It uses `backend/.clang-format` (2-space indent,
80-column limit, aligned declarations/assignments). Do not hand-align declarations.
`backend/.clang-tidy` is not wired into a script and its header filter does not match
the current source layout; its presence does not mean lint ran.

## Backend invariants

Start with [architecture](docs/architecture.md) and [KV cache](docs/kv-cache.md).
Implementation in `backend/cache/`, `engine/`, `server/`, `models/` and `utils/` is
header-only with inline definitions. Translation units live in `bin/` and `tests/`.
CMake creates a binary for each `bin/*.cpp`; tests are opt-in. Only explicitly linked
targets get MPI/ZMQ/Threads, so check linkage when adding a binary.

### Layering and ownership

```text
bin/ -> server/ -> engine/ <- models/common/runner.hpp <- models/<name>/
                                       |
                                       v
                                    cache/ -> forch
```

1. `bin/` is the composition root for server configuration. `bin::Config` parses
   JSON; `EngineConfig` and `ServerConfig` consume their slices. Scheduler limits
   come from the already-built engine config, never a second independent parse.
   Model loaders separately read HF `config.json`; wire decoding is also separate.
2. `engine/` and `server/` do not know concrete models. The scheduler talks to
   `ModelRunner` using `GraphInfo`, `SlotStep`, and `StepOutput`, without tensors or
   physical page numbers. Engine-side packing/output helpers may use tensors.
3. `CacheRunner::make_owned` keeps the mmapped loader, model, and layer caches alive.
   Register the loader before the model: weight tensors borrow the mapping.
   Unload graphs before releasing resources in reverse registration order.
4. TP has one source: `engine_server` initializes MPI, checks configured size against
   the MPI world, and writes `EngineConfig::tp`. Models and GEBackend use that field;
   never infer a second group from environment variables. `Transport` owns rank-aware
   request broadcasts, sampling synchronization, and capacity reduction. Keep the
   scheduler rank-agnostic; see [Tensor Parallel](docs/model-tp.md).

### Graphs, cache, and scheduling

- Graphs compile once at startup and every step pads to their token budget.
  Shape or KV configuration changes require a restart. Model-specific capacities,
  including GDN's per-slot limits, are declared by the runner.
- One `SlotMap` per runner owns cursors, page allocation, prefix hashes, and block
  tables for all token-cache layers. One `AttentionMetadata` builds write rows,
  masks, lengths, and block tables per step. Each layer keeps its own data Pools;
  do not reintroduce per-layer/per-pool addressing. GDN state remains per-slot.
- The scheduler probes `can_run` / `SlotMap::can_fit` without side effects, then
  executes `reserve -> feed -> run -> commit`. No capacity-consuming operation may
  intervene. Generate masks/tables after reservation; publish prefix blocks only
  after the graph successfully writes them.
- Models dispatch through `full_attention` / `mla_attention` in
  `models/common/attention/attention.hpp`. Eager, DenseFIA, TND, BSND and PagedBSND
  are separate operator schemes. Preserve eager reference paths and the intentional
  DenseFIA ablation baseline; see [attention layering](backend/models/common/attention/README.md).
- Continuous batching uses a fixed slot pool and a waiting queue. When capacity
  is insufficient, preempt the newest request and recompute later. No new request
  is admitted until survivors drain, avoiding repeated prefix-attach/preempt cycles.
- NPU TopK reduces transfer size; CPU sampling applies temperature and top-p.
  Return vocab token IDs from `indices`, never positions in `vals`. Preserve request
  RNG state during preemption; replay must not advance it for intermediate tokens.

### Wire protocol

The frontend binds and the backend connects: DEALER requests, PUSH outputs, and a
short-lived DEALER handshake (HELLO -> INIT -> ReadyResponse -> READY). The engine
identity is two-byte little-endian. Requests are a one-byte type frame followed by
a MessagePack payload. ReadyResponse block counts reflect the runner's real KV pool.

**Engine-core envelopes are positional arrays; field order is the contract.** Read
`rust-frontend/ENGINE_PROTOCOL.md` before touching `protocol.hpp`. Mirror contract
changes in the frontend, without changing nested map payloads into arrays.

### Adding models

Add the model config, graph, and `CacheRunner` subclass under `models/<name>/`, reuse
`models/common/` modules, and register it in `models/loader.hpp`. Honor optimization
flags and `n_model_layers`, and preserve the eager reference. The checklist is in
[architecture](docs/architecture.md); model support is defined by the loader and
implementation, not by historical agent notes.

## Conventions

- Namespaces and compatibility strings still use `finer`. Do not mass-rename them:
  the version string is part of the handshake payload.
- The launcher fixes the frontend handshake port at 29550; the backend can read
  `server.handshake_address`, but changing only one side breaks the connection.
  `engine_index` is fixed. Coordinate both sides if changing transport setup.
- Use spdlog: startup/handshake at info, request/step detail at debug;
  `SPDLOG_LEVEL` controls verbosity.
- SIGINT deliberately uses `_Exit(0)` to avoid GE cleanup subprocess errors;
  SIGTERM requests normal shutdown. Register handlers after GE initialization,
  which may overwrite them.
- Comments explain constraints and decisions, not obvious operations. Follow the
  surrounding file's language; docs are primarily Chinese, code comments English.

## Submodules

Before working inside a submodule, read its instructions:

- `forch/CLAUDE.md` (or its `AGENTS.md` target): PIMPL facades, the CANN ABI shim,
  operator registry, weak-symbol link ordering, and operator development.
- `rust-frontend/AGENTS.md`: Rust error handling, parser style, and snapshot tests.

Submodules are separate repositories with separate remotes. Do not update their
pointers accidentally while editing parent documentation or switching worktrees.
