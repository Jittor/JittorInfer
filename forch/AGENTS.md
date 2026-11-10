# AGENTS.md

Guidance for AI coding agents working in this repository.

`AGENTS.md` is the shared source of truth. Read `AGENTS.local.md`, when present,
for machine-specific environment and execution settings; it is ignored by Git.
`CLAUDE.md` and `CLAUDE.local.md` are compatibility symlinks to the corresponding
`AGENTS` files, not separate documents. Edit the targets and preserve this direction.

## Overview

Forch is a C++20 tensor abstraction library targeting the Ascend GE (Graph Engine) backend. It uses a deferred execution model: users build a DAG of `Tensor` objects, then `compile`, `load`, and `run` it via a `Backend`.

## Documentation and task skills

- Keep repository-wide constraints and common commands here. Put machine paths,
  environment names, and resource settings in the untracked `AGENTS.local.md`.
- Keep reusable task workflows under `.agents/skills/<task>/SKILL.md`, committed
  with this repository. Give each skill a precise `name` and `description` in
  YAML frontmatter. Add `references/` only for substantial conditional details;
  add scripts only when they perform useful repeatable checks or operations.
- Read the relevant skill when starting its task, then only the references that
  apply. Agents without skill discovery should follow this index directly.
  When working from the parent JittorInfer checkout, use this index as well;
  do not assume a parent session discovers skills inside a child submodule.
- Keep each procedure in one place. Link to live source examples instead of
  maintaining copied implementations. Update the relevant skill in the same
  change when its workflow or contracts change; do not recreate the removed
  `docs/` tree for parallel copies of agent instructions.

| Task | Read |
| --- | --- |
| Add, change, or debug a Forch operator's GE mapping and tests | [forch-ge-operator](.agents/skills/forch-ge-operator/SKILL.md) |

## Common Commands

Run these commands from the Forch root after activating the environment described
in `AGENTS.local.md` (or your deployment's equivalent). The default GE build needs
the CANN toolkit and `ASCEND_HOME_PATH`; GE execution also needs an allocated NPU.

- **Build** — `bash script/build.sh` hardcodes Debug and writes to `build/`.
  Setting `CMAKE_BUILD_TYPE` in the environment does not override this script.
  For Release, use `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`, then
  `cmake --build build -j8` (adjust parallelism to the allocated CPU budget).
  If generated `es_all` sources unexpectedly rebuild serially, inspect the build
  first; only clean the affected generated build directory with user approval.
- **Format** — `bash script/format.sh` runs the unversioned `clang-format` from
  `PATH` over `example/`, `include/`, and `src/`. Neither the script nor CI pins
  version 18. Run it for C/C++ changes before pushing: CI formats and pushes an
  automatic commit when it finds differences.
- **Lint** — `bash script/tidy.sh` uses `clang-tidy-18 -p build/ src/**.cpp`.
  This is not recursive under the script's default Bash settings: it only checks
  top-level `src/*.cpp`, excluding backend subdirectories as well as examples.
  For backend changes, pass the changed `.cpp` paths explicitly to
  `clang-tidy-18 -p <actual-build-dir>`.
- **Run all tests** — `python script/run-test.py` discovers executables under
  `build/src/forch-ge/tests/` and launches each via
  `srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8`. SLURM allocates the devices;
  the launcher removes inherited `ASCEND_RT_VISIBLE_DEVICES`. There is no
  `--devices` option. Use `--srun-arg=--partition=<partition>` for site settings.
  `--timeout-sec` defaults to 600 seconds, includes SLURM queue time, and accepts
  `<=0` to disable the timeout. Use `--build-dir` for a non-default build tree.
  All tests share `ASCEND_WORK_PATH` (`ascend_work` relative to the build tree
  by default, configurable with `--work-path`); preserve this runner convention.
- **Run a single test** — from `build/`, use
  `ASCEND_WORK_PATH=ascend_work srun --gres=gpu:1 --ntasks=1 --cpus-per-task=8 ./src/forch-ge/tests/forch-ge-test-softmax`.
  Direct execution is appropriate only when an NPU has already been allocated
  and its visibility is configured. Test target names are
  `forch-ge-test-<source-stem>`, not just `<source-stem>`.
- **Run anything on several cards** — `script/forch-ge-run.py` is the multi-card
  launcher, forch-ge's equivalent of `torchrun`. It is not a test runner: it
  picks the cards, writes the rank table, exports the environment contract that
  `GEBackend::Options::from_env()` consumes, and supervises one process per
  rank. Everything after `--` is passed through untouched.

  ```bash
  python script/forch-ge-run.py --devices 2,3 -- \
    ./build/src/forch-ge/tests/multirank/forch-ge-mr-collectives --dtype bf16
  python script/forch-ge-run.py --devices 0,2,3,5 -- \
    ./build/backend/engine_server config.json
  ```

  It is deliberately not coupled to SLURM: SLURM allocates the cards and sets
  `ASCEND_RT_VISIBLE_DEVICES`, which is the default device list, but nothing is
  launched through `srun`. `--devices` names *physical* card ids; each rank then
  sees exactly its own card and drives logical device 0. Use `--keep` or
  `--log-dir` to retain per-rank logs and the generated rank table, which
  otherwise live in a temporary directory under `$XDG_RUNTIME_DIR`.
- **Multi-rank tests** live in
  [src/forch-ge/tests/multirank/](src/forch-ge/tests/multirank/) and are named
  `forch-ge-mr-<source-stem>`. `run-test.py`'s glob is not recursive, so it
  never picks them up — launch them with `forch-ge-run.py`. Each needs at least
  two cards and reports a clean skip with one.
- **Launcher regression checks** — `python script/test-forch-ge-run.py -v`
  tests the launcher itself with host subprocesses, without allocating NPUs.
  This lives beside the other `script/` commands and is separate from the GE
  operator tests run by `run-test.py`.

The minimal multi-card example is [example/ge-collective.cpp](example/ge-collective.cpp).
It sums one value per rank with `all_reduce`; the file starts with its run command.

When this checkout is the `forch/` submodule of JittorInfer, its parent build script
uses a separate tree: from the parent root, run
`CMAKE_BUILD_TYPE=Release bash script/build-forch.sh`. It builds and installs into
`build/forch/` and `build/forch/install/`. From the Forch root, test those binaries
with `python script/run-test.py --build-dir ../build/forch`. Do not set `BUILD_DIR`
for the parent scripts: they reuse it for otherwise separate component trees.

## Architecture

### PIMPL facades

`Device`, `Tensor`, `Backend`, `forch::Graph`, and `Backend::Graph` (in [include/forch/](include/forch/)) are lightweight handles holding a `shared_ptr<Impl>`. Backend-specific Impls (e.g. `GEBackend::Impl`, `GEBackend::GEGraph::Impl`) live in [src/forch-ge/](src/forch-ge/) and own GE session / graph state. Copying a `Tensor` aliases the same node and metadata; it does not deep-copy storage.

**Public headers in `include/` must not expose backend-specific types** (GE/ACL). Use `Impl` forward declarations for that.

### Backends as separate compilation units

Each backend in `src/{backend-name}/` is a separately compiled dynamic library.

GE backend ABI boundary: the C++ session/runtime APIs used here (`ge::GeSession`,
`ge::GEInitializeV2`, and `gert::Tensor`) are isolated in a single-TU C shim at
[src/forch-ge/shim/ge_session_shim.h](src/forch-ge/shim/ge_session_shim.h) /
[.cpp](src/forch-ge/shim/ge_session_shim.cpp). Its CMake object target
`forch-ge-shim` alone sets `_GLIBCXX_USE_CXX11_ABI=0` and
`google=ascend_private` to match the CANN ABI; these are not project-wide flags.
It is linked privately into `libforch-ge.so`, and its header is not installed.
Other Forch GE translation units stay on the default ABI and use CANN's C APIs
(`esb_funcs.h`, `es_*_c.h`, `graph/c_types.h`, ACL) or the shim's opaque handles.

`GEBackend::Impl` and `GEBackend::GEGraph::Impl` are declared in the private
[src/forch-ge/forch-ge-common.hpp](src/forch-ge/forch-ge-common.hpp) so the backend
does not have to live in one file: [init.cpp](src/forch-ge/init.cpp) owns
process-wide GEInitializeV2 options,
[backend.cpp](src/forch-ge/backend.cpp) owns the session, `move_to_device` and
the execution path, and [graph.cpp](src/forch-ge/graph.cpp) owns graph
construction. Keep new code in the file that matches its scope.
The common header contains shared Graph/Backend definitions and declarations
for GE utilities, not their implementations. Dtype mapping and shim error
handling live in `common.cpp`. Runtime tensors, their memory owners and
metadata methods stay private to `backend.cpp`. Operator dispatch lives in
`op-registry.hpp`; HCCL attribute translation lives in `hccl-common.hpp`.
Backend-independent implementation helpers such as strict environment parsing
belong in the private `src/utils.hpp`, not the installed public utils header.

### Device memory and synchronous transfers

`Device` manages byte allocations and synchronous transfers. Its move-only
`Device::Buffer` retains the allocating Device implementation and frees the
allocation on destruction. An H2D runtime tensor holds a shared pointer to the
Buffer and a GE view that borrows its address. Declare the Buffer first so the
view is destroyed first. Tensor copies alias storage. H2D tensors can outlive
the backend and still be read back and released. GE owns its RunGraph outputs.

Construct one `AscendDevice` per rank and pass copies of its handle to backends.
Construction initializes ACL and binds the logical device for this process.
The caller keeps its default context current throughout serial use and teardown.
A GE graph retains its backend until the graph is unloaded; the backend owns
the session and finalizes GE on teardown. Neither GEBackend nor AscendDevice
resets the device or finalizes ACL. External ACL owners must keep the runtime
and context alive until the last device allocation has been released.

Use `host_tensor | to_device(backend)` (only `op::Const` is accepted). For D2H,
provide a host buffer to `device.device_to_host(dst, tensor.get_data(), bytes)`;
there is no `ToHost`/`to_host` or Backend host-tensor construction API yet.
Both raw transfer buffers must hold at least `bytes`, and device pointers must
belong to the Device. Zero-byte operations accept null pointers.

`FORCH_ENABLE_ASCEND` defaults to `FORCH_ENABLE_GE`; GE requires Ascend support.
Ascend-only builds set GE OFF and Ascend ON. Installed Ascend consumers use
`find_package(forch REQUIRED COMPONENTS Ascend)` and `forch::forch-ascend`;
requesting GE also resolves Ascend dependencies. Core-only consumers need no CANN.
On an allocated NPU, `forch-ge-test-device_memory` checks Buffer ownership
and reads back H2D tensors after GE teardown. The GE tests `backend_lifecycle`
and `move_to_device` cover GE initialization and graph execution with device inputs.

### Operator system

- Ops derive from `op::Base` or `op::Function`. `op::Function` is for ops taking `Tensor` inputs.
- `Visitor` pattern is used for backend traversal of the op DAG.
- `TensorBuildCurry<N>` (in [include/forch/ops/function.hpp](include/forch/ops/function.hpp)) provides the user-facing builder API (e.g. `add(a, b)`).
- `GEOpRegistry` in `src/forch-ge/op-registry.hpp` dispatches by concrete C++ type; common `op::Function::Type`
  values have a separate switch in `ops/common.cpp`. Multi-output ops share an
  `op::MultiOutputBase` node with `MultiOutputPort` tensors and use
  `register_multi_op`. The operator skill covers choosing and extending these
  paths.
- Every emitter, single- or multi-output, has the same shape:
  `(const GEOpContext &ctx, const ConcreteOp *op)`. The context carries the
  builder, the input `EsCTensorHolder`s, and `input_tensors`, the forch tensors
  behind them. `input_tensors` is there because an `EsCTensorHolder` can be told
  its dtype (`EsSetDataType`) but never asked, so an op with an explicit
  output-dtype attribute recovers it by running its own `dtype()` over the
  inputs — see `y_dtype` in `ops/fused_collective_matmul.cpp`.
- `register_multi_op` checks the returned holder count against `n_outputs()`,
  so the promise and its enforcement sit together rather than in the graph
  builder.

### Collectives

Collectives are backend-agnostic in [include/forch/ops/collective.hpp](include/forch/ops/collective.hpp)
and [include/forch/ops/fused_collective_matmul.hpp](include/forch/ops/fused_collective_matmul.hpp):
they take a `forch::ProcessGroup` (`rank_id()`, `world_size()`, `group_id()`, see
[include/forch/distributed.hpp](include/forch/distributed.hpp)) and a
`forch::op::ReduceOp` (defined in `ops/collective.hpp`). `ProcessGroup(rank, size,
group_id = 0)` validates membership at construction; its accessors preserve
the invariant, so operators do not revalidate the group. Only the default
world group (id 0) is provisioned today. The launcher's rank table records it
as Forch `group_id` metadata; native HCCL v1 tables omit that field and describe
the same implicit world group. Nonzero ids do not create communicators.

Those headers carry only rules that belong to the operation itself: output
shapes, divisibility, group and root membership. **What a particular
collective library can move is not one of them.** The dtype whitelists
(no FP64/BOOL for HCCL, FP16/BF16 only for the fused matmul) live in
the corresponding `src/forch-ge/ops/collective.cpp` and
`ops/fused_collective_matmul.cpp`, and the emitters call them, so they are reported
when the graph is built, not at `Tensor` construction. Requiring the inputs to
share a dtype needs no code at all: the inherited `Function::dtype()` already
does it.

Running them takes process-wide setup that GE reads *at init*, never through a
graph option: a rank table, the rank id, and a bound device. Construct
`AscendDevice::from_env()` and pass it to `GEBackend(device)`; this default
GE constructor reads `Options::from_env()`. Use `GEBackend(device, options)`
for explicit GE configuration. `AscendDevice::from_env()` reads
`FORCH_GE_DEVICE_ID` (logical id, default 0); GE options read `FORCH_RANK_ID`,
`FORCH_WORLD_SIZE`, `FORCH_GROUP_ID`, `FORCH_GE_RANK_TABLE_FILE` and
`FORCH_GE_HCCL_TIMEOUT`. The launcher exports the same environment as before.
Rank and topology configure GE communication independently of the logical
device selected by each process. The
`ge.exec.*` option keys exist only in [src/forch-ge/init.cpp](src/forch-ge/init.cpp).

**Construct exactly one GEBackend and copy its handle to share it.** Every
second construction throws, including identical options and construction after
teardown. The shim exposes one `forch_ge_init(options, count)` and rejects
reinitialization too. A failed validation does not consume the initialization.
Backend initialization and teardown are single-threaded; there are no locks or
atomic initialization guards. Do not call them concurrently.

Two constraints worth knowing before debugging a failure here: an *inline* rank
table is not an option, because CANN 9.0.0 has no init key for one and passing
`ge.exec.rankTable` is rejected at graph compile time; and the rank table's
`device_id` entries are physical card ids, while `aclrtSetDevice` takes a
logical one.

### Graph execution flow (GE backend)

1. Build inputs with `op::Input` / `op::Const`, then chain ops to produce a DAG of `Tensor`s.
2. Construct `GEBackend`, wrap the outputs in `forch::Graph(outputs)`, and call
   `backend.build(graph)` → returns a `Backend::Graph`. The overload taking
   `GEBuildOptions` returns `GEBackend::GEGraph`.
3. `graph.compile()` → `graph.load()` → `graph.run(inputs)`.
4. `detail::GEGraphBuilder` in [src/forch-ge/graph.cpp](src/forch-ge/graph.cpp) walks the Forch DAG via Visitor and emits an `EsCGraph` using the C interface in `esb_funcs.h` (`EsCreateGraphBuilder` / `EsCreateGraphInputWithDetails` / `EsConst` / `EsSetGraphOutput` / `EsBuildGraphAndReset`).
5. `GEBackend::Impl::run_graph` in [src/forch-ge/backend.cpp](src/forch-ge/backend.cpp) converts Forch tensors to runtime tensor handles via `forch_ge_rt_tensor_from_ptr` and submits them to the shim's `forch_ge_session_run`.
6. The GE run implementation matches inputs by `Tensor::get_name()`. Input tensors
   must have **unique names** matching the graph's `op::Input` names. Runtime
   tensors can be passed to `run`, but cannot be embedded as graph-building nodes;
   use `op::Input` placeholders for them.

`op::load_from_buffer` borrows host storage, whereas `op::load_from_vector`
owns its moved vector. The shim's pointer-backed runtime views are non-owning.
Hugging Face weights use borrowed mmapped safetensors storage: keep the
`huggingface::Pretrained` / `Safetensors` owner alive while those views are used.

### Linker note: weak symbols in CANN

`libes_all.so` (generated by `add_es_library`) provides the strong implementations
for op functions like `EsAdd`, while `libeager_style_graph_builder_base.so` carries
weak placeholders. Preserve the dependency ordering with `libes_all.so` first to
avoid "Weak symbol implementation called" errors. `src/forch-ge/CMakeLists.txt`
links `es_all` before the other CANN libraries and does not directly link
`eager_style_graph_builder_base`; the latter arrives transitively. Preserve the
same ordering in `cmake/forchConfig.cmake.in` for installed consumers.

Installed GE consumers must request `find_package(forch REQUIRED COMPONENTS GE)`
and link `forch::forch-ge`. The package resolves CANN dependencies using the
consumer's `ASCEND_HOME_PATH` without exposing CANN headers in the public API.

## Code organization conventions

Use `logger()` for library diagnostic output, not `fprintf`. Tests and examples
may keep ordinary console output; they do not need to use `logger()`. Explicit
throws outside the C shim use `forch::exception`; follow the existing error message prefixes,
format and capitalization. Inside the shim, report failures with `set_error`
(or `set_ge_error` for GE status codes) and the existing failure return values.

- `include/` — public headers; backend-agnostic only. Inline functions are okay if they don't pull in backend types.
- `src/{backend-name}/` — backend implementation; one dynamic library per backend.
- `src/{backend-name}/tests/` — backend tests (mostly AI-generated).
- `example/` — usage examples, mostly AI-generated. The maintainer's stance: if AI can't produce a working example, the API design is wrong.
