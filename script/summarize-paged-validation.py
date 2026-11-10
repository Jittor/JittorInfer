#!/usr/bin/env python3
"""Summarize model reports and compare paging with the dense storage control.

Optional --decode uses each local tokenizer.json; no model download occurs.
Report creation does not replace checking the launcher/test exit status.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    reports = {}
    for path in sorted(args.directory.glob("*-result.json")):
        report = json.loads(path.read_text())
        name = path.name.removesuffix("-result.json")
        if "mixed_seconds" not in report:
            print(f"{name}: non-model report (inspect separately)")
            continue
        reports[name] = report
        per_step = report["mixed_seconds"] * 1000 / report["mixed_steps"]
        print(f"{name}: requests={report['requests']} context={report['context']} "
              f"slots={report['slots']} TP={report['tp']} "
              f"preemptions={report['preemptions']} mismatches={report['mismatching_requests']} "
              f"mixed={report['mixed_seconds']:.3f}s step={per_step:.2f}ms")
        if args.decode and report.get("smoke"):
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(Path(report["model"]) / "tokenizer.json"))
            for ids in report["smoke"]:
                print("  " + repr(tokenizer.decode(ids)))
    for name, pressure in reports.items():
        if not name.endswith("-pressure"):
            continue
        dense = reports.get(name.removesuffix("-pressure") + "-dense")
        if dense is None:
            print(f"{name}: dense control MISSING")
            continue
        a = [row["serial"] for row in pressure["results"]]
        b = [row["serial"] for row in dense["results"]]
        print(f"{name}: dense generations exact={a == b}, "
              f"smoke exact={pressure.get('smoke') == dense.get('smoke')}")


if __name__ == "__main__":
    main()
