#!/usr/bin/env python3
"""Sample random feasible-Knock plans and execute each plan in Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import zlib

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cuboid_knock_pick_solvers as solver  # noqa: E402
from scripts.baseline_compare import random_clearance_trial  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems",
                        default="results/experiment_40_planning/isaac_problems/*.yaml")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--isaac-python", required=True)
    parser.add_argument("--output", default="runs/random-knock-isaac")
    parser.add_argument("--batch-name",
                        help="Use this exact subdirectory instead of a timestamp")
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")

    source_paths = sorted(ROOT.glob(args.problems))
    if not source_paths:
        raise SystemExit(f"No problem files match {args.problems!r}")
    names = {path.stem for path in source_paths}
    instances = {instance.name: instance for instance in solver.all_benchmarks()
                 if instance.name in names}
    missing = sorted(names - instances.keys())
    if missing:
        raise SystemExit("No geometry instances for: " + ", ".join(missing))

    stamp = args.batch_name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch = (ROOT / args.output / stamp).resolve()
    plans = batch / "sampled_plans"
    plans.mkdir(parents=True)
    records = []
    for case_index, name in enumerate(sorted(names)):
        rule = solver.preprocess(instances[name], prune_dominated=True)
        feasible_knocks = solver.isaac_feasible_knock_actions(rule)
        for repetition in range(args.runs):
            plan_seed = (args.seed + case_index * args.runs + repetition +
                         zlib.crc32(name.encode("utf-8"))) & 0x7fffffff
            sampled = random_clearance_trial(
                rule, plan_seed, include_solution=True,
                feasible_knocks=feasible_knocks)
            if not sampled["success"]:
                raise RuntimeError(f"random Knock planning failed for {name}: {sampled}")
            result = solver.SolveResult(
                solver="RandomFeasibleKnock", status="feasible",
                optimum=int(sampled["knocks"]),
                runtime_sec=float(sampled["runtime_sec"]),
                selected_actions=sampled["selected_actions"],
                order=sampled["order"],
            )
            path = plans / f"{name}__random_knock_run{repetition:02d}.yaml"
            document = solver.write_isaac_problem(rule, result, path)
            document.update({
                "benchmark_case": name,
                "benchmark_repetition": repetition,
                "policy": "random-knock",
                "random_plan_seed": plan_seed,
            })
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            records.append({
                "case": name, "run": repetition, "seed": plan_seed,
                "planning_time_sec": sampled["runtime_sec"],
                "knocks": sampled["knocks"],
                "physical_actions": sampled["physical_actions"],
                "yaml": str(path),
            })

    with (batch / "sampled_plan_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (batch / "sampled_plan_metrics.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8")

    command = [
        sys.executable, str(ROOT / "scripts/run_bench.py"),
        "--problems", str(plans / "*.yaml"), "--runs", "1",
        "--policy", "planned", "--workers", str(args.workers),
        "--timeout", str(args.timeout), "--seed", str(args.seed),
        "--isaac-python", args.isaac_python, "--output", str(batch),
        "--batch-name", "execution",
    ]
    print("[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    manifest = {
        "policy": "random-feasible-knock", "cases": len(names),
        "runs_per_case": args.runs, "seed": args.seed,
        "sampled_plans": str(plans),
        "execution_summary": str(batch / "execution/overall_summary.json"),
    }
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                          encoding="utf-8")
    print(f"[DONE] Random-Knock planning and Isaac benchmark: {batch}")


if __name__ == "__main__":
    main()
