#!/usr/bin/env python3
"""Run and report the proposed method and both random baselines end to end."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from paper_tables import generate_paper_tables


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def execution_cases(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case"]): {k: v for k, v in row.items() if k != "runs"}
            for row in data["cases"]}


def prefixed(row: dict, prefix: str, skip=("case", "family")) -> dict:
    return {f"{prefix}{key}": value for key, value in row.items()
            if key not in skip}


def require_same_cases(named_sets: dict[str, set[str]]) -> list[str]:
    """Fail loudly instead of silently dropping a case from a comparison."""
    union = set().union(*named_sets.values())
    problems = []
    for label, names in named_sets.items():
        missing = sorted(union - names)
        extra = sorted(names - set.intersection(*named_sets.values()))
        if missing:
            problems.append(f"{label} is missing: {', '.join(missing)}")
        if extra:
            problems.append(f"{label} has unmatched cases: {', '.join(extra)}")
    if problems:
        raise SystemExit("Benchmark inputs do not contain identical case sets:\n  " +
                         "\n  ".join(problems))
    return sorted(union)


def scalar(value) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_table(title: str, rows: list[dict], base: Path) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |",
             "|" + "|".join("---" if i < 2 else "---:"
                              for i in range(len(fields))) + "|"]
    lines.extend("| " + " | ".join(scalar(row.get(key)) for key in fields) + " |"
                 for row in rows)
    markdown = "\n".join(lines) + "\n"
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown, flush=True)


def run(command: list[str]) -> None:
    print("\n[RUN] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--random-knock-trials", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--in-process-parallel", type=int, default=0,
                        help="Opt-in Panda workspaces in one Isaac simulation")
    parser.add_argument("--workspace-spacing", type=float, default=2.5)
    parser.add_argument("--show-gui", action="store_true",
                        help="Visualize the in-process Panda workspace grid")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--isaac-python",
                        default=os.environ.get("ISAACSIM_PYTHON_EXE", ""))
    parser.add_argument("--output", default="runs/all-methods")
    parser.add_argument("--problems",
                        default="results/experiment_40_planning/isaac_problems/*.yaml")
    parser.add_argument("--planner-output", default="results/experiment_40_planning",
                        help="Planning artifact directory for the primary 40-case set")
    parser.add_argument("--no-regenerate-plans", action="store_true")
    parser.add_argument("--proposed-summary", type=Path,
                        help="Reuse a proposed-method summary instead of executing Isaac")
    parser.add_argument("--random-push-summary", type=Path,
                        help="Reuse a random-push summary instead of executing Isaac")
    args = parser.parse_args()
    if not args.isaac_python or not Path(args.isaac_python).is_file():
        raise SystemExit("Set ISAACSIM_PYTHON_EXE or pass --isaac-python")
    if (args.proposed_summary is None) != (args.random_push_summary is None):
        raise SystemExit("Provide both summary reuse arguments or neither")

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    batch = ROOT / args.output / stamp
    batch.mkdir(parents=True, exist_ok=False)

    if not args.no_regenerate_plans:
        run([args.isaac_python, "cuboid_knock_pick_solvers.py",
             "--output-dir", args.planner_output, "--primary-experiment-only",
             "--isaac-exportable-only"])

    if args.proposed_summary is None:
        process_workers = 1 if args.in_process_parallel else args.workers
        common = [sys.executable, "scripts/run_bench.py", "--problems", args.problems,
                  "--runs", str(args.runs), "--workers", str(process_workers),
                  "--timeout", str(args.timeout), "--seed", str(args.seed),
                  "--isaac-python", args.isaac_python, "--output", str(batch)]
        if args.in_process_parallel:
            common += ["--in-process-parallel", str(args.in_process_parallel),
                       "--workspace-spacing", str(args.workspace_spacing)]
            if args.show_gui:
                common.append("--show-gui")
        run(common + ["--policy", "planned", "--batch-name", "proposed"])
        run(common + ["--policy", "random-push", "--batch-name", "random_push"])
        run([args.isaac_python, "scripts/random_knock_benchmark.py",
             "--problems", args.problems, "--runs", str(args.runs),
             "--workers", str(args.workers), "--timeout", str(args.timeout),
             "--seed", str(args.seed), "--isaac-python", args.isaac_python,
             "--output", str(batch), "--batch-name", "random_knock_isaac"])
        proposed_summary = batch / "proposed" / "overall_summary.json"
        push_summary = batch / "random_push" / "overall_summary.json"
        random_knock_isaac_summary = (batch / "random_knock_isaac" /
                                      "execution" / "overall_summary.json")
    else:
        proposed_summary = args.proposed_summary.resolve()
        push_summary = args.random_push_summary.resolve()
        random_knock_isaac_summary = None

    random_dir = batch / "random_knock"
    run([args.isaac_python, "scripts/baseline_compare.py",
         "--problems", args.problems,
         "--trials", str(args.random_knock_trials), "--seed", str(args.seed),
         "--proposed-runs", str(proposed_summary),
         "--random-push-runs", str(push_summary), "--output", str(random_dir)])

    planner = {row["instance"]: row for row in
               read_csv(ROOT / args.planner_output / "solver_results.csv")}
    proposed = execution_cases(proposed_summary)
    push = execution_cases(push_summary)
    random_knock = {row["case"]: row for row in read_csv(random_dir / "comparison.csv")}
    random_knock_isaac = (execution_cases(random_knock_isaac_summary)
                          if random_knock_isaac_summary else {})
    compared_sets = {
        "solver_results.csv": set(planner),
        "proposed Isaac summary": set(proposed),
        "random-push Isaac summary": set(push),
        "random-knock comparison": set(random_knock),
    }
    if random_knock_isaac:
        compared_sets["random-knock Isaac summary"] = set(random_knock_isaac)
    names = require_same_cases(compared_sets)

    table1 = []
    table2 = []
    table3 = []
    for name in names:
        identity = {
            "case": name,
            "family": planner[name].get("experiment_category",
                                         planner[name].get("family", "")),
        }
        table1.append({**identity, **prefixed(planner[name], "planner_", ("instance", "family")),
                       **prefixed(proposed[name], "isaac_")})
        random_fields = {k: v for k, v in random_knock[name].items()
                         if k.startswith("random_clearance_") or
                         k in {"optimal_knocks", "optimal_solver_time_sec"}}
        table2.append({**identity,
                       **prefixed(planner[name], "ours_", ("instance", "family", "isaac_export")),
                       **random_fields,
                       **prefixed(random_knock_isaac.get(name, {}),
                                  "random_knock_isaac_")})
        table3.append({**identity,
                       "ours_policy": "optimal-knock-pick",
                       "random_push_policy": "online-random-push",
                       **prefixed(planner[name], "ours_planner_", ("instance", "family")),
                       **prefixed(proposed[name], "ours_isaac_"),
                       **prefixed(push[name], "random_push_isaac_")})

    write_table("Table 1 — Proposed method: planning and Isaac execution",
                table1, batch / "table_1_our_method_full")
    write_table("Table 2 — Optimal planner versus random feasible Knock",
                table2, batch / "table_2_planning_vs_random_knock")
    write_table("Table 3 — Proposed method versus random-push baseline",
                table3, batch / "table_3_ours_vs_random_push")
    paper_tables = generate_paper_tables(batch)
    full_tables = [batch / f"table_{i}{suffix}" for i, suffix in (
        (1, "_our_method_full"), (2, "_planning_vs_random_knock"),
        (3, "_ours_vs_random_push"))]
    manifest = {
        "created": dt.datetime.now().isoformat(), "cases": len(names),
        "runs_per_isaac_policy": args.runs,
        "random_knock_trials_per_case": args.random_knock_trials,
        "seed": args.seed, "workers": args.workers,
        "in_process_parallel": args.in_process_parallel,
        "workspace_spacing": args.workspace_spacing,
        "proposed_summary": str(proposed_summary),
        "random_push_summary": str(push_summary),
        "random_knock_isaac_summary": (str(random_knock_isaac_summary)
                                        if random_knock_isaac_summary else None),
        "tables": [str(path) for path in full_tables + paper_tables],
        "full_tables": [str(path) for path in full_tables],
        "paper_tables": [str(path) for path in paper_tables],
    }
    (batch / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[DONE] Complete three-method report (3 full + 3 paper tables): {batch}",
          flush=True)


if __name__ == "__main__":
    main()
