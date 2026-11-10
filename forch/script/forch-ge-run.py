#!/usr/bin/env python3
"""Launch one forch-ge process per NPU, like torchrun does for torch.

This is not a test runner. It knows nothing about what it starts: it picks the
cards, writes the rank table, exports the environment contract that
GEBackend::Options::from_env() reads, and supervises the processes. Running the
multi-rank tests and running a tensor-parallel engine_server are the same path.

    forch-ge-run.py --devices 2,3 -- ./build/.../forch-ge-mr-collectives
    forch-ge-run.py --devices 0,2,3,5 -- ./build/backend/engine_server cfg.json

Everything after `--` is passed through untouched.

Deliberately not coupled to SLURM: SLURM allocates the cards and exports
ASCEND_RT_VISIBLE_DEVICES, which is the default device list here, but nothing
is launched through srun.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# The rank table wants physical card ids, while each rank is handed a single
# card through ASCEND_RT_VISIBLE_DEVICES and so drives logical device 0.
LOGICAL_DEVICE = "0"
DEFAULT_GROUP_ID = 0


def parse_device_list(raw: str) -> list[int]:
	devices: list[int] = []
	for item in raw.replace(" ", ",").split(","):
		if not item:
			continue
		if not item.isdigit():
			raise SystemExit(f"error: '{item}' is not a card id (in '{raw}')")
		devices.append(int(item))
	if not devices:
		raise SystemExit(f"error: no card ids in '{raw}'")
	if len(set(devices)) != len(devices):
		raise SystemExit(f"error: duplicate card ids in '{raw}'")
	return devices


def idle_devices() -> list[int]:
	"""Cards npu-smi reports no process on. Advisory only, for error messages."""
	if shutil.which("npu-smi") is None:
		return []
	try:
		out = subprocess.run(
			["npu-smi", "info"], capture_output=True, text=True, timeout=30
		).stdout
	except (OSError, subprocess.SubprocessError):
		return []
	return sorted(int(m) for m in re.findall(r"No running processes found in NPU (\d+)", out))


def device_ip(card: int) -> str:
	"""The card's RoCE address. HCCL link setup needs it; 127.0.0.1 does not work."""
	if shutil.which("hccn_tool") is None:
		raise SystemExit(
			"error: hccn_tool not found, cannot read device IPs for the rank table.\n"
			"       Pass a ready-made table with --rank-table instead."
		)
	proc = subprocess.run(
		["hccn_tool", "-i", str(card), "-ip", "-g"],
		capture_output=True,
		text=True,
	)
	for line in proc.stdout.splitlines():
		if line.startswith("ipaddr:"):
			return line.split(":", 1)[1].strip()
	raise SystemExit(
		f"error: 'hccn_tool -i {card} -ip -g' reported no address for card {card}.\n"
		f"       stdout: {proc.stdout.strip()!r}\n"
		f"       stderr: {proc.stderr.strip()!r}"
	)


def write_rank_table(path: Path, devices: list[int]) -> None:
	table = {
		"version": "1.0",
		# Forch metadata; HCCL's v1 table describes the implicit world group.
		"group_id": str(DEFAULT_GROUP_ID),
		"server_count": "1",
		"server_list": [
			{
				"server_id": "127.0.0.1",
				"device": [
					{
						"device_id": str(card),
						"device_ip": device_ip(card),
						"rank_id": str(rank),
					}
					for rank, card in enumerate(devices)
				],
			}
		],
		"status": "completed",
	}
	path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")


def validate_rank_table(path: Path, devices: list[int]) -> None:
	"""Reject a stale table before peers enter HCCL with different topologies."""
	try:
		table = json.loads(path.read_text(encoding="utf-8"))
		if int(table.get("group_id", DEFAULT_GROUP_ID)) != DEFAULT_GROUP_ID:
			raise ValueError("only default group_id=0 is supported")
		servers = table["server_list"]
		if len(servers) != 1:
			raise ValueError("only single-server rank tables are supported")
		entries = servers[0]["device"]
		actual = sorted((int(d["rank_id"]), int(d["device_id"])) for d in entries)
		if actual != list(enumerate(devices)):
			raise ValueError("rank/device mapping does not match the selected devices")
	except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
		raise SystemExit(f"error: invalid rank table {path}: {exc}") from exc


def run_dir(keep: bool, log_dir: str | None) -> tuple[Path, bool]:
	"""Returns the working directory and whether the caller must remove it."""
	if log_dir is not None:
		path = Path(log_dir).resolve()
		path.mkdir(parents=True, exist_ok=True)
		return path, False
	# XDG_RUNTIME_DIR is per-user and tmpfs-backed, which is where transient
	# run state belongs; /tmp is shared with everyone else on the node.
	parent = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
	path = Path(tempfile.mkdtemp(prefix="forch-ge-run.", dir=parent))
	return path, not keep


@dataclass
class Rank:
	rank: int
	card: int
	proc: subprocess.Popen
	log: Path
	returncode: int | None = None


def pump(rank: Rank, stream_to_stdout: bool) -> None:
	"""Copy one rank's merged output into its log, optionally echoing it."""
	with rank.log.open("w", encoding="utf-8") as sink:
		assert rank.proc.stdout is not None
		for line in rank.proc.stdout:
			sink.write(line)
			sink.flush()
			if stream_to_stdout:
				sys.stdout.write(f"[rank {rank.rank}] {line}")
				sys.stdout.flush()


def terminate(ranks: list[Rank]) -> None:
	# Signal whole process groups, including helpers left by an exited leader.
	for rank in ranks:
		try:
			os.killpg(rank.proc.pid, signal.SIGTERM)
		except ProcessLookupError:
			pass
	deadline = time.monotonic() + 10
	for rank in ranks:
		try:
			rank.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
		except subprocess.TimeoutExpired:
			pass
	for rank in ranks:
		try:
			os.killpg(rank.proc.pid, signal.SIGKILL)
		except ProcessLookupError:
			pass
		rank.proc.wait()


def main() -> int:
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument(
		"--devices",
		help=(
			"physical card ids, e.g. '2,3'. Defaults to ASCEND_RT_VISIBLE_DEVICES; "
			"rank i gets the i-th card in the list"
		),
	)
	parser.add_argument(
		"--nproc",
		type=int,
		help="use only the first N cards of the device list",
	)
	parser.add_argument(
		"--rank-table",
		help="reuse this rank table instead of generating one",
	)
	parser.add_argument(
		"--log-dir",
		help=(
			"keep per-rank logs and the rank table here instead of a temporary "
			"directory under $XDG_RUNTIME_DIR"
		),
	)
	parser.add_argument(
		"--keep",
		action="store_true",
		help="do not delete the temporary run directory on exit",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		help="kill every rank after this many seconds (default: wait forever)",
	)
	parser.add_argument(
		"--tee",
		action="store_true",
		help="echo every rank's output, not just rank 0's",
	)
	parser.add_argument(
		"command",
		nargs=argparse.REMAINDER,
		help="-- followed by the command to run on every rank",
	)
	args = parser.parse_args()
	if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0):
		parser.error("--timeout must be finite and positive")

	command = args.command
	if command and command[0] == "--":
		command = command[1:]
	if not command:
		parser.error("no command given; put it after '--'")

	raw_devices = args.devices or os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
	if not raw_devices:
		idle = idle_devices()
		hint = f" Currently idle: {','.join(str(d) for d in idle)}." if idle else ""
		raise SystemExit(
			"error: no devices given and ASCEND_RT_VISIBLE_DEVICES is unset.\n"
			f"       Pass --devices <ids>, e.g. --devices 0,1.{hint}"
		)
	devices = parse_device_list(raw_devices)
	if args.nproc is not None:
		if args.nproc < 1 or args.nproc > len(devices):
			raise SystemExit(
				f"error: --nproc {args.nproc} is outside 1..{len(devices)} "
				f"(device list: {','.join(str(d) for d in devices)})"
			)
		devices = devices[: args.nproc]

	work, cleanup = run_dir(args.keep, args.log_dir)
	world_size = len(devices)
	ranks: list[Rank] = []
	pumps: list[threading.Thread] = []

	def on_signal(signum, _frame):  # noqa: ANN001
		raise SystemExit(128 + signum)

	previous_handlers = {
		sig: signal.signal(sig, on_signal) for sig in (signal.SIGINT, signal.SIGTERM)
	}

	try:
		if args.rank_table is not None:
			rank_table = Path(args.rank_table).resolve()
			if not rank_table.is_file():
				raise SystemExit(f"error: no such rank table: {rank_table}")
		else:
			rank_table = work / "rank_table.json"
			write_rank_table(rank_table, devices)
		validate_rank_table(rank_table, devices)

		# One shared ASCEND_WORK_PATH, as script/run-test.py also does: the
		# Ascend runtime expects it shared, not isolated per process.
		ascend_work = work / "ascend_work"
		ascend_work.mkdir(parents=True, exist_ok=True)

		print(
			f"forch-ge-run: world_size={world_size} devices={','.join(str(d) for d in devices)}",
			flush=True,
		)
		print(f"forch-ge-run: run dir {work}", flush=True)

		for rank, card in enumerate(devices):
			env = os.environ.copy()
			env["ASCEND_RT_VISIBLE_DEVICES"] = str(card)
			env["ASCEND_WORK_PATH"] = str(ascend_work)
			env["FORCH_RANK_ID"] = str(rank)
			env["FORCH_WORLD_SIZE"] = str(world_size)
			env["FORCH_GROUP_ID"] = str(DEFAULT_GROUP_ID)
			env["FORCH_GE_DEVICE_ID"] = LOGICAL_DEVICE
			env["FORCH_GE_RANK_TABLE_FILE"] = str(rank_table)

			proc = subprocess.Popen(
				command,
				env=env,
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				text=True,
				bufsize=1,
				# Own session, so a Ctrl-C in this terminal does not race us to
				# the children and we can signal each whole tree ourselves.
				start_new_session=True,
			)
			ranks.append(Rank(rank, card, proc, work / f"rank{rank}.log"))
			thread = threading.Thread(
				target=pump, args=(ranks[-1], args.tee or rank == 0), daemon=True
			)
			pumps.append(thread)
			thread.start()

		deadline = None if args.timeout is None else time.monotonic() + args.timeout
		failure = 0
		while True:
			for r in ranks:
				r.returncode = r.proc.poll()
			failed = [r for r in ranks if r.returncode not in (None, 0)]
			if failed:
				failure = failed[0].returncode
				break
			if all(r.returncode is not None for r in ranks):
				break
			if deadline is not None and time.monotonic() >= deadline:
				print(f"forch-ge-run: timeout after {args.timeout:g}s", flush=True)
				failure = 124
				break
			time.sleep(0.05)
		terminate(ranks)
		for r in ranks:
			r.returncode = r.proc.returncode

		for thread in pumps:
			thread.join(timeout=5)

		print("forch-ge-run: exit codes " + ", ".join(
			f"rank{r.rank}(card {r.card})={r.returncode}" for r in ranks
		), flush=True)

		failed = [r for r in ranks if r.returncode != 0]
		for r in failed:
			# rank 0 was echoed live already unless --tee covered everything.
			if r.rank == 0 and not args.tee:
				continue
			print(f"--- rank {r.rank} (card {r.card}), last 40 lines ---", flush=True)
			if r.log.is_file():
				for line in r.log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]:
					print(f"  {line}")

		if not failure:
			return 0
		if cleanup:
			print(f"forch-ge-run: rerun with --keep to inspect {work}", flush=True)
		return failure if failure > 0 else 128 - failure
	finally:
		# Also covers a failed Popen partway through launching the ranks.
		for sig in previous_handlers:
			signal.signal(sig, signal.SIG_IGN)
		terminate(ranks)
		for thread in pumps:
			thread.join(timeout=5)
		for sig, handler in previous_handlers.items():
			signal.signal(sig, handler)
		if cleanup:
			shutil.rmtree(work, ignore_errors=True)
		else:
			print(f"forch-ge-run: logs kept in {work}", flush=True)


if __name__ == "__main__":
	raise SystemExit(main())
