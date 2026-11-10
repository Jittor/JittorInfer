# Collective GE Operators

Read this when an operation communicates between cards, whether a pure
collective or a matmul fused with one. Paths are relative to the Forch root.
The live examples are:

- `include/forch/distributed.hpp`: `ProcessGroup`, `process_group_from_env`.
- `include/forch/ops/collective.hpp`: `forch::op::ReduceOp`, six primitives and shape rules.
- `include/forch/ops/fused_collective_matmul.hpp`: the fused matmul family.
- `src/forch-ge/hccl-common.hpp`: `hccl_group_name` and `hccl_reduce_op`, the
  only place forch's vocabulary becomes HCCL's.
- `src/forch-ge/ops/collective.cpp`, `src/forch-ge/ops/fused_collective_matmul.cpp`.
- `src/forch-ge/init.cpp`: device binding and the `ge.exec.*` option map.
- `src/forch-ge/tests/multirank/`: both tests.

## Keep the public API backend-agnostic

The participants arrive as a `ProcessGroup` parameter after the tensors:

```cpp
forch::ProcessGroup group(0, 8);
Tensor y = forch::op::all_reduce(x, group);
Tensor z = forch::op::fused_matmul_all_reduce(x1, x2, group);
```

`ProcessGroup` validates its rank, size and nonnegative group id at construction,
and exposes read-only accessors. Forch holds no process-wide copy, so
nothing in a public header may name an HCCL group, a reduction string, or a
`ge.exec.*` key. A new collective takes `ProcessGroup` and, if it reduces,
`forch::op::ReduceOp`; translate both in `src/forch-ge/hccl-common.hpp`. Group id 0 names
the default world group. Nonzero ids do not create groups; multiple worlds and
subgroup provisioning remain unsupported. The launcher records `group_id=0`
as Forch metadata alongside the native HCCL table, and exports `FORCH_GROUP_ID`.

The same line applies to dtypes. A public op header states what the operation
means — shapes, divisibility, group membership — not what the hardware under
it happens to support. Keep dtype checks local to the respective emitters in
`ops/collective.cpp` and `ops/fused_collective_matmul.cpp`.
Do not add a `check_*_dtype` to `include/forch/`. "All inputs share a dtype"
is the one dtype rule that is genuinely the operation's, and the inherited
`Function::dtype()` already enforces it, so do not override `dtype()` to
restate it.

Name the operation after `torch.distributed` where it has a counterpart, and
after `torch.ops.symm_mem.fused_*` for a fusion of a matmul and a collective.

## Validate in shape(), not at compile time

`Tensor::Impl` calls `shape()` then `dtype()` at construction, both without a
device, so this is where a caller still has the context to fix the problem.
Group validity is guaranteed by its constructor; check `root_rank` against
`group.world_size()` and check
the divisibility the collective needs — `reduce_scatter` and `all_to_all`
require `dim 0 % world_size == 0`. Left to GE, these surface as a tiling or
HCCL error that names none of it.

Backend limits are the exception: they are checked in the emitter, so they
surface from `backend.build()` instead. That is still well before GE compiles
anything, and it keeps the public header honest.

## Probe dtypes, do not read them

The op protos are not authoritative about dtype. `hcom_ops.h` omits BF16 for
`all_reduce` and `reduce_scatter`, yet both work on 910B3 and run faster than
FP16, so GE is not silently widening to FP32. All six primitives are measured
working in BF16; the fused ops accept FP16/BF16 only, because anything else
needs the antiquant/dequant-scale inputs this library does not expose.

Measure on the target before you widen or narrow a dtype list, and if you do
narrow one, quote the GE message that made you.

## Output-dtype attributes

`EsMatmulAllReduce` has a `y_dtype` attribute that must be filled: left at
`C_DT_UNDEFINED`, GE infers `DT_FLOAT` for y and tiling then fails with a
message that never mentions `y_dtype`. An `EsCTensorHolder` can be told its
dtype (`EsSetDataType`) but never asked, so recover it by running the op's own
rule over the context's forch tensors — `op->dtype(ctx.input_tensors)` — rather
than restating the rule in the emitter. See
`src/forch-ge/ops/fused_collective_matmul.cpp`.

## Device and backend bring-up

A process running collectives needs ACL initialization/device binding and a
rank table reaching `GEInitializeV2`; these cannot be passed through `AddGraph`.
Construct `auto device = forch::AscendDevice::from_env()` followed by
`forch::GEBackend backend(device)`. Device reads `FORCH_GE_DEVICE_ID`; Backend
reads the remaining GE/distributed environment via `Options::from_env()`.
For explicit GE configuration, use `GEBackend(device, options)`. Read the group
with `forch::process_group_from_env()` as before; the launcher contract is unchanged.

Construct one backend and copy the handle; a second construction is rejected
even for identical options or after teardown. Initialization and teardown are
single-threaded. Do not reach into `ge_session_shim.h` or ACL from an operator
test. Device initialization belongs to AscendDevice; GE/HCCL options belong to
`GEBackend::Options`. Neither Device nor Backend resets the device or finalizes
ACL; Backend still finalizes GE.

## Test with two branches, not a host oracle

Put multi-card tests in `src/forch-ge/tests/multirank/`, which the
non-recursive glob in `tests/CMakeLists.txt` keeps out of `run-test.py`. Target
names are `forch-ge-mr-<source-stem>`. Launch with:

```bash
python script/forch-ge-run.py --devices 2,3 -- \
  ./build/src/forch-ge/tests/multirank/forch-ge-mr-collectives --dtype bf16
```

Two complementary shapes of test, both worth having:

- `collectives.cpp` compares each op against a host reference built from what
  every rank contributed.
- `tp_align.cpp` puts a sharded branch and a replicated branch in the *same*
  graph and compares them element-wise. This is what catches a wrong assembly
  — a shard offset off by one, `all_reduce` where `all_gather` belonged, a
  fused op's y taken pre- instead of post-residual. A host oracle cannot: it
  assumes the same thing the sharded branch does.

Choose values that make the comparison exact rather than approximate. With 0/1
matmul inputs and `K <= 256`, every partial sum is an exact integer in any
summation order, so the check is bit-exact and a tolerance never has to be
tuned. Report a skip, not a failure, when the group is too small for the case.
