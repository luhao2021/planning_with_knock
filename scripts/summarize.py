#!/usr/bin/env python3
import argparse
import json
import math
from typing import Any, Dict, List, Tuple

METRICS: List[str] = [
    "num_actions",
    "task_and_motion_planning_time_sec",
    "gap_task_planning_time_sec",
    "execution_time_sec_internal",
    "single_goal_height",
]

def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))

def summarize_success_runs(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    runs = data.get("runs", [])
    # Filter only successful runs
    successes = [r for r in runs if r.get("success") is True]

    summary: Dict[str, Dict[str, float]] = {m: {"sum": 0.0, "count": 0, "avg": float("nan")} for m in METRICS}

    for r in successes:
        for m in METRICS:
            val = r.get(m, None)
            if is_number(val):
                summary[m]["sum"] += float(val)
                summary[m]["count"] += 1

    for m in METRICS:
        c = summary[m]["count"]
        summary[m]["avg"] = (summary[m]["sum"] / c) if c > 0 else float("nan")

    summary["__meta__"] = {
        "total_runs": len(runs),
        "success_runs": len(successes),
    }
    return summary

def fmt(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "NA"
    # Keep a sensible precision
    return f"{x:.6f}" if not float(x).is_integer() else f"{int(x)}"

def main():
    parser = argparse.ArgumentParser(description="Summarize metrics over successful runs in a scenario JSON.")
    parser.add_argument("json_path", nargs="?", default="/mnt/data/scenario_summary.json",
                        help="Path to the scenario summary JSON file.")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    summary = summarize_success_runs(data)

    print(f"Total runs: {summary['__meta__']['total_runs']}")
    print(f"Successful runs counted: {summary['__meta__']['success_runs']}\n")

    header = ["metric", "sum", "count", "average"]
    print("{:<36} {:>15} {:>10} {:>15}".format(*header))
    print("-" * 80)
    for m in METRICS:
        row = summary[m]
        print("{:<36} {:>15} {:>10} {:>15}".format(
            m,
            fmt(row["sum"]),
            fmt(row["count"]),
            fmt(row["avg"]),
        ))

if __name__ == "__main__":
    main()
