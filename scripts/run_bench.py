#!/usr/bin/env python3
"""Repeated benchmark runner for the native Isaac knock-pick executor."""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="problems/*.yaml")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--isaac-python", default=os.environ.get("ISAACSIM_PYTHON_EXE", ""))
    return parser.parse_args()


def action_counts(path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    actions = document.get("actions", [])
    return len(actions), sum(a.get("type") in ("Knock", "Smash") for a in actions)


def main():
    args = parse_args()
    if not args.isaac_python or not Path(args.isaac_python).is_file():
        raise SystemExit("Set ISAACSIM_PYTHON_EXE or pass --isaac-python /path/to/python.sh")
    problems = sorted(Path().glob(args.problems))
    if not problems:
        raise SystemExit(f"No problem files match {args.problems!r}")

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch = Path(args.output) / stamp
    batch.mkdir(parents=True)
    records = []
    executor = Path("src/isaac_executor/scripts/isaac_executor.py").resolve()

    for problem in problems:
        actions, knocks = action_counts(problem)
        for repetition in range(args.runs):
            run_dir = batch / f"{problem.stem}__run{repetition:02d}"
            run_dir.mkdir()
            copied = run_dir / problem.name
            shutil.copy2(problem, copied)
            log_path = run_dir / "isaac_executor.log"
            started = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(
                        [args.isaac_python, str(executor), "--yaml", str(copied),
                         "--headless", "--no-wait"],
                        stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False,
                    )
                    returncode = result.returncode
                    failure = None if returncode == 0 else f"exit_{returncode}"
                except subprocess.TimeoutExpired:
                    returncode = None
                    failure = "timeout"
            record = {
                "problem": str(problem), "run": repetition,
                "num_actions": actions, "num_knocks": knocks,
                "success": returncode == 0,
                "failure": failure,
                "execution_time_sec": round(time.perf_counter() - started, 6),
                "log": str(log_path),
            }
            records.append(record)
            (run_dir / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"{problem.name} run {repetition + 1}/{args.runs}: "
                  f"{'success' if record['success'] else failure}", flush=True)

    successes = [record for record in records if record["success"]]
    summary = {
        "runs": records,
        "n_runs": len(records),
        "successes": len(successes),
        "success_rate": len(successes) / len(records),
        "avg_execution_time_sec": (
            sum(record["execution_time_sec"] for record in successes) / len(successes)
            if successes else None
        ),
    }
    output = batch / "overall_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
