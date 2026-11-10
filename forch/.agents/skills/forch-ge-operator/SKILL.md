---
name: forch-ge-operator
description: Add, modify, or debug Forch tensor operators backed by Ascend GE, including public shape/dtype rules, generated Es C API mappings, registry and CMake integration, and operator tests. Use for forch-ge operator work, not for implementing new AscendC device kernels or unrelated backend infrastructure.
---

# Forch GE Operator Development

Read [AGENTS.md](../../../AGENTS.md) and `AGENTS.local.md` if present first.
Paths below are relative to the Forch root, not the parent JittorInfer root.
This skill covers wrapping existing GE operations. If the requested operation
needs a new device kernel or a toolkit upgrade, explain the missing capability
and confirm that expanded scope before proceeding.

## Establish the contract

Identify the intended call site and nearest existing operator before editing.
For a new API, establish input count, ranks/shapes, broadcasting, accepted dtypes,
attributes and defaults, output shapes/dtypes, and any ordering or aliasing
semantics. Ask about ambiguities that affect callers rather than silently
assuming PyTorch behavior. For a fix, preserve the existing contract unless its
change is requested or necessary and explained.

Use live sources as examples, not as proof that every edge case is handled:

| Need | Read |
| --- | --- |
| Attribute-bearing single output | `include/forch/ops/softmax.hpp`, `src/forch-ge/ops/softmax.cpp` |
| Output dtype differs from input | `include/forch/ops/cast.hpp`, `src/forch-ge/ops/cast.cpp` |
| Tensor-valued shape parameter | `src/forch-ge/ops/reshape.cpp` |
| Collective or multi-card operation | [Collective operators](references/collectives.md) |
| Simple common functions | `include/forch/ops/function.hpp`, `src/forch-ge/ops/common.cpp` |
| Multiple outputs from one operation | [Multi-output operators](references/multi-output.md) |

Negative axes, dtype promotion, and invalid-argument handling are not uniform
across existing operators. For example, Softmax currently rejects negative axes;
TopK normalizes them for shape inference. Do not generalize either behavior to
an unspecified new API.

## Check the installed GE interface

Choose the actual build tree (`build` standalone, `../build/forch` for the parent
integration). Inspect its generated `output/include/es_all/es_<Op>_c.h` and, when
needed, corresponding `src/forch-ge/es_all_build/generated_code/` implementation.
Use `rg --files --hidden --no-ignore <build-tree> -g 'es_*_c.h'` to locate headers
that normal repository searches may skip. If absent, configure/build using the
environment and commands in `AGENTS.md`; do not hand-create generated headers.

Check the exact symbol, argument order, tensor versus attribute parameters,
optional inputs, output structure, and status/handle conventions. The generated
signature establishes calling conventions, not all supported shapes or dtypes;
check the installed op definitions and matching official CANN documentation for
those constraints, then verify on the target hardware. Do not use an API from a
different CANN version just because it appears in a search result.

Concrete distinctions in the current bindings:

- `EsSoftmaxV2` takes an axes array and length as attributes.
- `EsReshape` takes its shape as a tensor, constructed using
  `EsCreateVectorInt64(builder, ...)`.
- `EsTopK` takes tensor-valued `k`, constructed using `EsCreateScalarInt64`,
  and returns an output struct. See the multi-output reference for registration.

Keep CANN types out of public headers and preserve the existing C ABI shim and
link ordering described in `AGENTS.md`. A normal operator mapping should not
need new C++ GE session calls, project-wide ABI flags, or graph visitor changes.

## Implement the public operation

For a parameterized single-output operation, add or edit
`include/forch/ops/<name>.hpp`: derive from `op::Function`, store attributes,
implement `shape()`, and expose a `TensorBuildCurry<N>` builder. A dedicated
subclass inherits the `Function` visitor route; it normally needs no new visitor.

`Tensor::Impl` computes and caches shape and dtype at construction. Validate
supported ranks, axes, dimensions, and attributes before indexing shapes, and
throw `forch::exception` for invalid inputs. Do not mutate shape/dtype-affecting
attributes after a Tensor is constructed. Check input arity before indexing
when the public path permits arbitrary input lists.

The inherited `Function::dtype()` requires matching input dtypes; it is not a
general type-promotion rule. Override it for casts, indices, mixed-type inputs,
or other output-type changes. Metadata must describe what the GE mapping emits.
Do not inherit broadcasting shape rules for a non-broadcasting operation.

A dynamic `TensorBuildCurry` is single-use: calling it consumes its stored op.
Return a fresh builder for each application. For a genuinely simple common
function, extending `Function::Type` is possible, but requires matching shape
and dtype rules plus the GE switch in `src/forch-ge/ops/common.cpp`.

For multiple outputs, follow [Multi-output operators](references/multi-output.md)
instead of splitting the operation into independent single-output functions.

## Map and register it

Add or edit `src/forch-ge/ops/<name>.cpp`. Include the public operation header,
`op-registry.hpp`, and the generated `es_<Op>_c.h`. Include
`forch-ge-common.hpp` only when using shared GE utilities, or
`hccl-common.hpp` for collective attribute translation. Register the concrete
function type once with `detail::GEOpRegistry::register_op<YourOp>` inside an
anonymous-namespace `__attribute__((constructor))` function, following Softmax.
The callback receives the active builder, ordered input holders, and typed op.

Match public input order, normalized attributes, and dtypes to the generated
signature. Construct tensor-valued scalar/vector inputs using that callback's
builder; pass ordinary attributes directly. Handle failures according to the
actual API. Do not continue with a failed helper result or output handle, and
do not invent an error accessor from another CANN interface.

Add every new mapping `.cpp` to `src/forch-ge/ops/CMakeLists.txt`:
`target_sources(forch-ge PRIVATE ...)` is explicit. A file omitted here can
compile nowhere and leave the registry entry missing at runtime. For changes to
`Function::Type`, update the existing common-function switch instead of adding
a second registration for `op::Function`.

## Test the contract

Add or extend `src/forch-ge/tests/<name>.cpp`. CMake discovers these sources with
`CONFIGURE_DEPENDS` and creates `forch-ge-test-<name>` executables. There is no
need to invent a test framework or use the parent backend's debug harnesses.
Use `src/forch-ge/tests/softmax.cpp` for graph lifecycle and named input/output
structure, not as a complete numerical-comparison template.

Cover the parts changed: public metadata and invalid inputs, then GE numerical
results with a small independent reference or hand-computed case. Include
nontrivial ranks/axes, boundary attributes, supported dtype variations, and
broadcasting only where relevant to the contract. For reductions or approximate
math, choose a justified absolute/relative tolerance. Reject unexpected NaNs
and handle expected infinities explicitly: `fabs(actual - expected) > tol`
alone can silently accept NaNs. Check integers and indices exactly where the
contract makes them deterministic. Tests must fail via a nonzero exit status;
do not rely only on `assert`, which Release builds may disable.

Exercise `forch::Graph` -> `backend.build` -> `compile` -> `load` -> `run`, with
unique input names matching the runtime data. Assert both declared and runtime
shape/dtype before interpreting output memory. Reuse the established host or
device input path; enabling auto-H2D requires host inputs. Preserve borrowed
buffer owners through execution, or use owning `load_from_vector` inputs.
For multi-output changes, also use the reference's topology test checklist.

## Verify and report

Activate the environment before every build, format, lint, or test invocation.
Use the build tree selected above consistently, including its compile database.
After configuration, a focused build is
`cmake --build <build-tree> --target forch-ge-test-<name> -j8`; adjust parallelism
to the CPU allocation. Run the built binary through the single-test SLURM command
in `AGENTS.md`, substituting its actual name and build tree. The suite runner has
no test-name filter; it runs every executable it discovers.

Run formatting as documented in `AGENTS.md`. For lint, supply the changed GE
`.cpp` paths explicitly; the existing tidy script does not recurse into them.
Run related operators or the suite when changing shared broadcasting, dispatch,
multi-output graph handling, or another cross-operator contract. For parent
integration, rebuild/install Forch through the parent script before rebuilding
its consumers so they do not use stale installed headers or libraries.

If verification fails, first separate environment or SLURM allocation failures
from missing registration, invalid graph metadata, GE compile failures, and
numerical mismatches. Check generated signatures and linked source lists before
changing architecture. Do not fix an ABI or weak-symbol failure by applying old
ABI flags globally or adding the weak-placeholder library directly.

Report the API/mapping change, actual build and test commands and results, and
remaining unsupported cases or verification gaps. Without a toolkit, allocation,
or successful runtime test, explicitly report what was not verified; successful
compilation or metadata checks alone do not establish NPU correctness.
