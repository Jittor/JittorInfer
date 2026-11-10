#!/usr/bin/env python3
"""Launch an N-rank forch-ge program on this host through HMPI's mpirun.

    python3 script/mpirun-ge.py 2 -- ./build/backend/engine_server cfg.json

Cards come from $DEVICES or ASCEND_RT_VISIBLE_DEVICES; rank i drives the i-th.
The program takes rank/size from MPI. This script only writes the HCCL rank
table and gives each rank its card, then becomes mpirun, so signals and exit
status are mpirun's own. Load CANN and HMPI's env/setvars.sh first.
"""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile

# Reuse forch's device parsing and rank table writer (a hyphenated script).
_spec = importlib.util.spec_from_file_location(
    "forch_ge_run", Path(__file__).resolve().parents[1] / "forch/script/forch-ge-run.py")
forch = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = forch  # its dataclasses look themselves up here
_spec.loader.exec_module(forch)


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit(__doc__)
    nproc, command = int(sys.argv[1]), sys.argv[3:]
    raw = os.environ.get("DEVICES") or os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not raw:
        raise SystemExit("set DEVICES or ASCEND_RT_VISIBLE_DEVICES")
    devices = forch.parse_device_list(raw)[:nproc]
    if len(devices) != nproc:
        raise SystemExit(f"need {nproc} cards, got {raw}")
    prefix = os.environ.get("OPAL_PREFIX")
    if not prefix:
        raise SystemExit("load HMPI's env/setvars.sh first (OPAL_PREFIX)")

    work = Path(tempfile.gettempdir()) / f"finer-mpirun-{os.getuid()}"
    (work / "ascend_work").mkdir(parents=True, exist_ok=True)
    table = work / "rank_table.json"
    forch.write_rank_table(table, devices)
    os.environ.update(ASCEND_WORK_PATH=str(work / "ascend_work"),
                      FORCH_GE_RANK_TABLE_FILE=str(table), FORCH_GE_DEVICE_ID="0")

    mpirun = str(Path(prefix) / "bin/mpirun")
    # SLURM allocates one launcher task; HMPI starts the local ranks itself.
    launch = [mpirun, "--bind-to", "none", "--mca", "plm", "isolated",
              "--mca", "ras", "^slurm", "--tag-output"]
    for i, card in enumerate(devices):
        # One MPMD segment per rank, so each sees only its own card.
        launch += [":"] * (i > 0) + ["-n", "1", "-x", f"ASCEND_RT_VISIBLE_DEVICES={card}",
                                     *command]
    print(f"mpirun-ge: devices={devices} rank_table={table}", flush=True)
    os.execv(mpirun, launch)


if __name__ == "__main__":
    main()
