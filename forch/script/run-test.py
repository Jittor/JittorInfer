#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestResult:
	test_path: Path
	return_code: int
	duration_sec: float
	stdout: str
	stderr: str
	timed_out: bool = False

	@property
	def passed(self) -> bool:
		return self.return_code == 0


def discover_tests(build_dir: Path, backend_name: str) -> list[Path]:
	tests_dir = build_dir / "src" / backend_name / "tests"
	if not tests_dir.exists():
		raise FileNotFoundError(f"Tests directory not found: {tests_dir}")

	test_paths: list[Path] = []
	for path in sorted(tests_dir.glob("*")):
		if not path.is_file():
			continue
		if os.access(path, os.X_OK):
			test_paths.append(path)

	return test_paths


def run_single_test(build_dir: Path, test_path: Path, work_path: str,
	timeout_sec: float | None, srun_extra: list[str]) -> TestResult:
	rel_test_path = test_path.relative_to(build_dir)

	env = os.environ.copy()
	# IMPORTANT: Ascend runtime expects a shared ASCEND_WORK_PATH.
	# Do not isolate this path per test.
	env["ASCEND_WORK_PATH"] = work_path
	# Let SLURM allocate the device via --gres=gpu:1; don't pin manually.
	env.pop("ASCEND_RT_VISIBLE_DEVICES", None)

	cmd = [
		"srun",
		"--gres=gpu:1",
		"--ntasks=1",
		"--cpus-per-task=8",
		f"--job-name=test-{rel_test_path.stem}",
		*srun_extra,
		f"./{rel_test_path.as_posix()}",
	]

	start = time.perf_counter()
	with tempfile.TemporaryDirectory(prefix="run-test-", dir=build_dir) as tmp_dir:
		tmp_dir_path = Path(tmp_dir)
		stdout_path = tmp_dir_path / "stdout.log"
		stderr_path = tmp_dir_path / "stderr.log"

		# Avoid capture_output pipes here: if a crashed test leaves descendants
		# inheriting pipe fds, communicate() may wait until timeout.
		with stdout_path.open("w", encoding="utf-8") as stdout_file, \
				stderr_path.open("w", encoding="utf-8") as stderr_file:
			proc = subprocess.Popen(
				cmd,
				cwd=build_dir,
				env=env,
				stdout=stdout_file,
				stderr=stderr_file,
				text=True,
				start_new_session=True,
			)

			timed_out = False
			try:
				return_code = proc.wait(timeout=timeout_sec)
			except subprocess.TimeoutExpired:
				timed_out = True
				# SIGINT first lets srun cancel the step cleanly; escalate if needed.
				try:
					os.killpg(proc.pid, signal.SIGINT)
				except ProcessLookupError:
					pass
				try:
					proc.wait(timeout=10)
				except subprocess.TimeoutExpired:
					try:
						os.killpg(proc.pid, signal.SIGKILL)
					except ProcessLookupError:
						pass
					proc.wait()
				return_code = 124

		duration_sec = time.perf_counter() - start
		stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
		stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

		if timed_out:
			timeout_msg = f"timed out after {timeout_sec:.1f}s"
			stderr = f"{stderr}\n{timeout_msg}" if stderr else timeout_msg

		if return_code < 0:
			sig_num = -return_code
			try:
				sig_name = signal.Signals(sig_num).name
			except ValueError:
				sig_name = f"SIG{sig_num}"
			sig_msg = f"terminated by signal {sig_name}"
			stderr = f"{stderr}\n{sig_msg}" if stderr else sig_msg

		return TestResult(
			test_path=test_path,
			return_code=return_code,
			duration_sec=duration_sec,
			stdout=stdout,
			stderr=stderr,
			timed_out=timed_out,
		)


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Run tests in parallel via SLURM. Each test is launched as its own "
			"srun --gres=gpu:1 step; SLURM handles queueing across available NPUs. "
			"Default tests path: src/forch-ge/tests/*"
		)
	)
	parser.add_argument(
		"--build-dir",
		default=str(Path(__file__).resolve().parents[1] / "build"),
		help="Build directory",
	)
	parser.add_argument(
		"--work-path",
		default="ascend_work",
		help="Value for ASCEND_WORK_PATH (relative to build dir)",
	)
	parser.add_argument(
		"--timeout-sec",
		type=float,
		default=600.0,
		help="Per-test timeout in seconds; <=0 disables timeout",
	)
	parser.add_argument(
		"--srun-arg",
		action="append",
		default=[],
		metavar="ARG",
		help=(
			"Extra argument forwarded to srun. Repeat to pass multiple, e.g. "
			"--srun-arg=--partition=foo --srun-arg=--time=00:30:00"
		),
	)
	args = parser.parse_args()

	build_dir = Path(args.build_dir).resolve()
	if not build_dir.exists():
		print(f"error: build dir not found: {build_dir}", file=sys.stderr)
		return 2

	try:
		tests = discover_tests(build_dir=build_dir, backend_name="forch-ge")
	except FileNotFoundError as exc:
		print(f"error: {exc}", file=sys.stderr)
		return 2

	if not tests:
		print("No executable tests found.")
		return 0

	timeout_sec = args.timeout_sec if args.timeout_sec > 0 else None

	print(f"Build dir: {build_dir}")
	print(f"ASCEND_WORK_PATH: {args.work_path}")
	print(
		f"Per-test timeout: {'disabled' if timeout_sec is None else f'{timeout_sec:.1f}s'}"
	)
	if args.srun_arg:
		print(f"Extra srun args: {' '.join(args.srun_arg)}")
	print(f"Discovered tests: {len(tests)}")
	for t in tests:
		print(f"  - {t.relative_to(build_dir)}")

	results: list[TestResult] = []
	# All tests are submitted concurrently; SLURM queues them based on
	# available --gres=gpu resources.
	with ThreadPoolExecutor(max_workers=max(len(tests), 1)) as executor:
		future_to_test = {}
		for test_path in tests:
			rel_path = test_path.relative_to(build_dir)
			print(f"[SUBMIT] {rel_path}", flush=True)
			future = executor.submit(
				run_single_test,
				build_dir,
				test_path,
				args.work_path,
				timeout_sec,
				args.srun_arg,
			)
			future_to_test[future] = test_path

		for future in as_completed(future_to_test):
			test_path = future_to_test[future]
			rel_path = test_path.relative_to(build_dir)
			try:
				result = future.result()
			except Exception as exc:  # pragma: no cover
				print(f"[FAIL] {rel_path}: {exc}", flush=True)
				results.append(
					TestResult(
						test_path=test_path,
						return_code=1,
						duration_sec=0.0,
						stdout="",
						stderr=f"srun launcher crashed: {exc}",
					)
				)
				continue
			if result.passed:
				status = "PASS"
			elif result.timed_out:
				status = "TIME"
			else:
				status = "FAIL"
			print(
				f"[{status}] {rel_path} ({result.duration_sec:.2f}s)",
				flush=True,
			)
			results.append(result)

	results.sort(key=lambda x: x.test_path.as_posix())

	passed = [r for r in results if r.passed]
	failed = [r for r in results if not r.passed]
	total = len(results)

	print("\n=== Summary ===")
	print(f"Total:  {total}")
	print(f"Passed: {len(passed)}")
	print(f"Failed: {len(failed)}")

	if total > 0:
		pass_rate = len(passed) * 100.0 / total
		print(f"Pass rate: {pass_rate:.2f}%")

	if failed:
		print("\n=== Failed Tests ===")
		for item in failed:
			rel_path = item.test_path.relative_to(build_dir)
			print(f"- {rel_path} (rc={item.return_code})")
			if item.stdout.strip():
				print("  stdout:")
				print(item.stdout.rstrip())
			if item.stderr.strip():
				print("  stderr:")
				print(item.stderr.rstrip())

	return 0 if not failed else 1


if __name__ == "__main__":
	raise SystemExit(main())
