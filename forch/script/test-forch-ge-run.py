#!/usr/bin/env python3
"""Host-only launcher regression checks; no CANN runtime or NPU is used.

Run from the Forch root: python script/test-forch-ge-run.py -v
Tests cover rank-table validation, environment propagation, and child cleanup
on failure, timeout, or partial startup. Only Python's standard library is used.
"""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().with_name("forch-ge-run.py")
spec = importlib.util.spec_from_file_location("forch_ge_run", SCRIPT)
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


class LauncherTest(unittest.TestCase):
	def setUp(self):
		self.tmp = tempfile.TemporaryDirectory()
		self.addCleanup(self.tmp.cleanup)
		self.root = Path(self.tmp.name)
		self.table = self.root / "table.json"
		with mock.patch.object(launcher, "device_ip", return_value="192.0.2.1"):
			launcher.write_rank_table(self.table, [2, 3])

	def args(self, *command, timeout="5"):
		return [
			sys.executable, str(SCRIPT), "--devices", "2,3",
			"--rank-table", str(self.table), "--log-dir", str(self.root / "logs"),
			"--timeout", timeout, "--", *command,
		]

	def test_device_list(self):
		self.assertEqual(launcher.parse_device_list("2, 3"), [2, 3])
		for raw in ("", "1,1", "-1", "a"):
			with self.assertRaises(SystemExit):
				launcher.parse_device_list(raw)

	def test_table(self):
		table = json.loads(self.table.read_text())
		self.assertEqual(table["group_id"], "0")
		launcher.validate_rank_table(self.table, [2, 3])
		with self.assertRaises(SystemExit):
			launcher.validate_rank_table(self.table, [3, 2])
		table["group_id"] = "1"
		self.table.write_text(json.dumps(table))
		with self.assertRaises(SystemExit):
			launcher.validate_rank_table(self.table, [2, 3])
		del table["group_id"]
		self.table.write_text(json.dumps(table))
		launcher.validate_rank_table(self.table, [2, 3])

	def test_environment_and_json(self):
		code = (
			"import json,os; print(json.dumps({k:v for k,v in os.environ.items() "
			"if k.startswith('FORCH_') or k == 'ASCEND_RT_VISIBLE_DEVICES'}))"
		)
		env = dict(os.environ, FORCH_GROUP_ID="99")
		proc = subprocess.run(
			self.args(sys.executable, "-c", code), env=env,
			capture_output=True, text=True, timeout=15,
		)
		self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
		for rank, card in enumerate([2, 3]):
			values = json.loads((self.root / "logs" / f"rank{rank}.log").read_text())
			self.assertEqual(values["FORCH_RANK_ID"], str(rank))
			self.assertEqual(values["FORCH_WORLD_SIZE"], "2")
			self.assertEqual(values["FORCH_GROUP_ID"], "0")
			self.assertEqual(values["FORCH_GE_DEVICE_ID"], "0")
			self.assertEqual(values["ASCEND_RT_VISIBLE_DEVICES"], str(card))

	def test_failure_of_later_rank_stops_first(self):
		code = (
			"import os,time,sys; time.sleep(30) "
			"if os.environ['FORCH_RANK_ID']=='0' else sys.exit(7)"
		)
		start = time.monotonic()
		proc = subprocess.run(
			self.args(sys.executable, "-c", code, timeout="20"),
			capture_output=True, text=True, timeout=15,
		)
		self.assertEqual(proc.returncode, 7, proc.stdout + proc.stderr)
		self.assertLess(time.monotonic() - start, 10)

	def test_timeout_and_invalid_timeout(self):
		proc = subprocess.run(
			self.args(sys.executable, "-c", "import time; time.sleep(30)", timeout="0.2"),
			capture_output=True, text=True, timeout=15,
		)
		self.assertEqual(proc.returncode, 124, proc.stdout + proc.stderr)
		for value in ("0", "-1", "nan", "inf"):
			proc = subprocess.run(
				self.args("unused", timeout=value),
				capture_output=True, text=True, timeout=5,
			)
			self.assertEqual(proc.returncode, 2)

	def test_partial_launch_is_cleaned_up(self):
		real_popen = subprocess.Popen
		processes = []

		def popen(*args, **kwargs):
			if processes:
				raise OSError("injected second launch failure")
			proc = real_popen(*args, **kwargs)
			processes.append(proc)
			return proc

		argv = self.args(sys.executable, "-c", "import time; time.sleep(30)")[1:]
		with (
			mock.patch.object(sys, "argv", argv),
			mock.patch.object(launcher.subprocess, "Popen", side_effect=popen),
		):
			with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(OSError):
				launcher.main()
		self.assertIsNotNone(processes[0].poll())
		processes[0].stdout.close()


if __name__ == "__main__":
	unittest.main()
