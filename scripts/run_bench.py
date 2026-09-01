#!/usr/bin/env python3
"""Repeated benchmark runner for the native Isaac knock-pick executor."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import statistics
import time

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default="problems/*.yaml")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", default="runs")
    parser.add_argument("--isaac-python", default=os.environ.get("ISAACSIM_PYTHON_EXE", ""))
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent Isaac processes to run concurrently; start with 1")
    return parser.parse_args()


def action_counts(path):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    actions = document.get("actions", [])
    return (
        len(actions),
        sum(a.get("type") in ("Clearance", "Knock", "Smash") for a in actions),
        float(document.get("planning_time_sec", 0.0)),
    )


def run_one(task):
    args, executor, problem, repetition, batch = task
    actions, clearances, planning_time = action_counts(problem)
    run_dir = batch / f"{problem.stem}__run{repetition:02d}"
    run_dir.mkdir()
    copied = run_dir / problem.name
    shutil.copy2(problem, copied)
    log_path = run_dir / "isaac_executor.log"
    metrics_path = run_dir / "execution_metrics.json"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                [args.isaac_python, str(executor), "--yaml", str(copied),
                 "--headless", "--no-wait", "--metrics-json", str(metrics_path)],
                stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, check=False,
            )
            returncode = result.returncode
            failure = None if returncode == 0 else f"exit_{returncode}"
        except subprocess.TimeoutExpired:
            returncode = None
            failure = "timeout"
    executor_metrics = {}
    if metrics_path.exists():
        executor_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    record = {
        "problem": str(problem), "case": problem.stem, "run": repetition,
        "num_actions": actions, "num_clearances": clearances,
        "num_knocks": clearances,  # backward-compatible analysis column
        "planning_time_sec": planning_time,
        "success": returncode == 0 and executor_metrics.get("status") == "success",
        "failure": failure or executor_metrics.get("failure"),
        "exec_wall_time_sec": round(time.perf_counter() - started, 6),
        "log": str(log_path), **executor_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def aggregate(records):
    fields = ["planning_time_sec", "exec_wall_time_sec", "execution_time_sec", "clearance_attempts",
              "clearance_retries", "actions_completed", "objects_removed"]
    cases = []
    for case in sorted({r["case"] for r in records}):
        runs = [r for r in records if r["case"] == case]
        successes = [r for r in runs if r["success"]]
        row = {"case": case, "n_runs": len(runs), "successes": len(successes),
               "success_rate": len(successes) / len(runs), "runs": runs}
        for field in fields:
            values = [float(r[field]) for r in runs if isinstance(r.get(field), (int, float))]
            row[f"{field}_mean"] = statistics.fmean(values) if values else None
            row[f"{field}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
        cases.append(row)
    return cases


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

    tasks = [(args, executor, problem, repetition, batch)
             for problem in problems for repetition in range(args.runs)]
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.workers > 1:
        print(f"Running {args.workers} isolated Isaac processes concurrently; "
              "reduce to --workers 1 if GPU memory or physics throughput is unstable.")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_task = {pool.submit(run_one, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_task):
            record = future.result()
            records.append(record)
            print(f"{record['case']} run {record['run'] + 1}/{args.runs}: "
                  f"{'success' if record['success'] else record['failure']}", flush=True)

    successes = [record for record in records if record["success"]]
    cases = aggregate(records)
    summary = {
        "cases": cases, "runs": records,
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
    csv_rows = [{k: v for k, v in case.items() if k != "runs"} for case in cases]
    if csv_rows:
        with (batch / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader(); writer.writerows(csv_rows)
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
