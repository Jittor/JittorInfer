# Local Agent Instructions Template

Create an untracked `AGENTS.local.md` from this template in your checkout, or link
to an existing machine profile. Do not edit this template with personal settings.
Do not store credentials or private keys in either file.

## Workspace

- Primary checkout path and task-worktree location:
- Existing worktrees or artifacts that must be preserved:
- Shared machine profile path, if this worktree has local overrides:

Keep the primary checkout on `master`; perform tracked changes and PR work in a
task worktree. Initialize that worktree's submodules and use its own build tree.

## Activation

- Shell and noninteractive activation method:
- Environment manager and environment name, or container image:
- CANN environment script and chip-specific ops package:
- MPI environment script (matching build and runtime):
- Compiler, Rust toolchain availability, `protoc` and well-known proto include path:

Record actual activation commands here only after verifying them on this machine.
Do not infer local installation paths or installed versions from a recipe.

## Resources

- Resource allocation procedure, scheduler partition and CPU/NPU limits:
- Model storage and per-job log/output directories:
- Network/port isolation and temporary-directory requirements:

## Observations

Record the date, commands used to inspect packages/imports, known missing tools,
and machine-specific workarounds. Distinguish observations from passing tests;
recheck before changing dependencies or relying on a workaround.
