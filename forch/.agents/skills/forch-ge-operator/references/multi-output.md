# Multi-output GE Operators

Read this when one GE operation produces multiple related tensors. Paths are
relative to the Forch root. The live example is TopK:

- `include/forch/ops/multi_output.hpp`: node and port interfaces.
- `include/forch/ops/topk.hpp`: public outputs and builder.
- `src/forch-ge/ops/topk.cpp`: generated output struct to registry output order.
- `src/forch-ge/op-registry.hpp`: `GEOpContext`, `register_multi_op`, `emit`.
- `src/forch-ge/graph.cpp`: `visit(const op::MultiOutputPort *, ...)` and
  `built_multi_` cache.
- `src/forch-ge/tests/topk.cpp`: both-output, single-port, and intermediate cases.

## Model one logical operation

Derive a node from `op::MultiOutputBase`, implementing `n_outputs()`, per-output
`shape(inputs, index)` / `dtype(inputs, index)`, and `dump()`. Define a stable
output order. The node itself is not a Tensor or a `Function`: expose each
output using `make_port(shared_node, index, same_inputs)` and a named result
structure, following `TopKOutputs`.

All ports of one invocation must share the same node and ordered input list.
Different invocations must have different nodes, even if their attributes are
equal. GE graph building caches generated holders by node-pointer identity,
not by input identity. Sharing a node across different inputs in one graph
would silently select the first materialized operation's outputs.

The existing `TopKBuilder` stores its node before application, so use a fresh
`op::topk(...)(input)` expression for each logical invocation. Do not copy a
builder and apply the copies to different inputs. If a new builder is intended
to be reusable, allocate its shared node inside each call so only that call's
ports share identity. Do not change the existing TopK API as incidental cleanup.

Shape and dtype rules must cover every output. In particular, do not infer an
indices dtype from the values dtype. Verify metadata against the actual GE
result and check runtime dtype before casting data pointers.

## Register once and preserve output order

Use `GEOpRegistry::register_multi_op<YourNode>` in the GE mapping translation
unit, which must be listed in `src/forch-ge/ops/CMakeLists.txt`.

The callback has the same shape as a single-output one,
`(const GEOpContext &ctx, const YourNode *node)`, and returns a
`std::vector<EsCTensorHolder *>` in exactly the public node's output order.
For TopK, that is `{out.values, out.indices}`. Check the generated header for
the actual fields and API-specific failure behavior; do not assume every op
uses TopK's return structure. Per-output metadata, when the emitter needs it,
comes from `node->dtype(ctx.input_tensors, i)`.

The graph builder materializes all holders on the first reachable port and
reuses them for the others; the registry checks their count against
`n_outputs()`.
`MultiOutputPort` is a zero-computation selector with an existing visitor route:
do not register it as a `Function`, make a separate GE call per port, or add a
visitor per new multi-output node type.

## Validate topology as well as values

Select cases relevant to the new operation, including:

- All outputs requested together, and each output requested alone.
- A port used by a downstream operation while a sibling is a graph output.
- Reversed graph-output order, to catch assumptions about traversal order.
- Two independent invocations with different inputs in the same graph, to catch
  accidental sharing of one node or cache entry.
- Per-output declared/runtime shape and dtype, plus cross-output consistency.

For TopK-like outputs, check that indices are in range and values correspond
to the indexed inputs. Check sorting only when requested. With ties, compare
the documented guarantees rather than demanding one reference implementation's
arbitrary index order; with unsorted results, do not require sorted order.
Use the main skill's finite-value and tolerance guidance in these checks.

Numerical equality by itself does not prove a single backend materialization.
When modifying deduplication or traversal, also inspect generated graph nodes
or use a focused build-time callback counter to check one emission per logical
node. Keep such instrumentation scoped to the test or debugging task.
