#!/usr/bin/env python3
"""Summarize one case or an entire Isaac Sim benchmark batch."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRICS: List[str] = [
    "num_knocks",
    "num_actions",
    "knock_retry",
    "tamp_time_sec",
    "execution_time_sec",
    "total_time_sec",
    "exec_wall_time_sec",
    "exec_run_retries",
]


def is_number(value: Any) -> bool:
    """Return true for finite numeric metric values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def summarize_success_runs(data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Aggregate current batch-runner metrics over successful runs only."""
    runs = data.get("runs", [])
    successes = [run for run in runs if run.get("success") is True]
    summary: Dict[str, Dict[str, float]] = {
        metric: {"sum": 0.0, "count": 0, "avg": float("nan")}
        for metric in METRICS
    }

    for run in successes:
        for metric in METRICS:
            value = run.get(metric)
            if is_number(value):
                summary[metric]["sum"] += float(value)
                summary[metric]["count"] += 1

    for metric in METRICS:
        count = summary[metric]["count"]
        if count:
            summary[metric]["avg"] = summary[metric]["sum"] / count

    summary["__meta__"] = {
        "total_runs": len(runs),
        "success_runs": len(successes),
    }
    return summary


def extract_cases(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize an overall batch or one case summary to a list of cases."""
    if isinstance(data.get("cases"), list):
        return [case for case in data["cases"] if isinstance(case, dict)]
    if isinstance(data.get("runs"), list):
        return [data]
    raise ValueError(
        "Unsupported JSON schema: expected overall_summary.json with 'cases' "
        "or case_summary.json with 'runs'."
    )


def combine_case_runs(cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine case run records for a batch-wide successful-run summary."""
    runs: List[Dict[str, Any]] = []
    for case in cases:
        runs.extend(run for run in case.get("runs", []) if isinstance(run, dict))
    return {"runs": runs}


def fmt(value: float, precision: int = 3) -> str:
    if not is_number(value):
        return "NA"
    return f"{float(value):.{precision}f}"


def print_case_table(cases: List[Dict[str, Any]]) -> None:
    headers = ["case", "runs", "success", "rate", "knocks", "actions", "TAMP(s)", "exec(s)", "total(s)", "knRetry"]
    rows: List[List[str]] = []
    for index, case in enumerate(cases):
        summary = summarize_success_runs(case)
        meta = summary["__meta__"]
        total = int(meta["total_runs"])
        successful = int(meta["success_runs"])
        rows.append([
            str(case.get("case", f"case_{index}")),
            str(total),
            str(successful),
            fmt(successful / total if total else float("nan"), 3),
            fmt(summary["num_knocks"]["avg"]),
            fmt(summary["num_actions"]["avg"]),
            fmt(summary["tamp_time_sec"]["avg"]),
            fmt(summary["execution_time_sec"]["avg"]),
            fmt(summary["total_time_sec"]["avg"]),
            fmt(summary["knock_retry"]["avg"]),
        ])

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def print_metric_summary(summary: Dict[str, Dict[str, float]]) -> None:
    meta = summary["__meta__"]
    print(f"\nTotal runs: {int(meta['total_runs'])}")
    print(f"Successful runs counted: {int(meta['success_runs'])}\n")
    print(f"{'metric':<26} {'sum':>15} {'count':>8} {'average':>15}")
    print("-" * 70)
    for metric in METRICS:
        row = summary[metric]
        print(
            f"{metric:<26} {fmt(row['sum'], 6):>15} "
            f"{int(row['count']):>8} {fmt(row['avg'], 6):>15}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a run_bench.py case or overall batch JSON."
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to case_summary.json or overall_summary.json.",
    )
    args = parser.parse_args()

    data = json.loads(args.json_path.read_text(encoding="utf-8"))
    cases = extract_cases(data)

    print_case_table(cases)
    heading = "Batch-wide successful-run metrics" if len(cases) > 1 else "Successful-run metrics"
    print(f"\n{heading}")
    print_metric_summary(summarize_success_runs(combine_case_runs(cases)))


if __name__ == "__main__":
    main()
