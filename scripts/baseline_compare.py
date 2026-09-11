#!/usr/bin/env python3
"""Run the random-feasible-clearance baseline and compare all three methods."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cuboid_knock_pick_solvers as solver  # noqa: E402


def random_clearance_trial(rule, seed: int, include_solution: bool = False,
                           feasible_knocks=None) -> dict:
    """Exhaust Picks; at deadlock choose one uniformly random feasible Knock."""
    rng = random.Random(seed)
    cleared: set[int] = set()
    actions: list[dict] = []
    knocks = 0
    selected_actions = {}
    order = []
    started = time.perf_counter()
    while len(cleared) < rule.n:
        # Exhaustive picking is deterministic only for reporting convenience;
        # closure itself is order independent.
        progressed = True
        while progressed:
            progressed = False
            for i in range(rule.n):
                if i in cleared:
                    continue
                picks = [a for a in rule.actions[i]
                         if a.kind == "pick" and a.blockers <= cleared]
                if picks:
                    chosen = min(picks, key=lambda a: a.witness)
                    cleared.add(i)
                    selected_actions[i] = chosen
                    order.append(i)
                    actions.append({"type": "pick", "object": rule.names[i],
                                    "witness": chosen.witness})
                    progressed = True
        if len(cleared) == rule.n:
            break
        feasible = [a for i in range(rule.n) if i not in cleared
                    for a in rule.actions[i]
                    if a.kind == "knock" and a.blockers <= cleared and
                    (feasible_knocks is None or a in feasible_knocks)]
        if not feasible:
            return {"success": False, "failure": "no_feasible_knock",
                    "knocks": knocks, "actions": actions,
                    "objects_cleared": len(cleared),
                    "runtime_sec": time.perf_counter() - started}
        # The baseline samples a feasible cuboid uniformly, then one of that
        # cuboid's feasible directions.  This avoids biasing the choice toward
        # objects that happen to expose more faces.
        by_target = {}
        for action in feasible:
            by_target.setdefault(action.target, []).append(action)
        target = rng.choice(sorted(by_target))
        chosen = rng.choice(by_target[target])
        cleared.add(chosen.target)
        selected_actions[chosen.target] = chosen
        order.append(chosen.target)
        knocks += 1
        actions.append({"type": "knock", "object": rule.names[chosen.target],
                        "direction": chosen.direction,
                        "witness": chosen.witness})
    result = {"success": True, "failure": None, "knocks": knocks,
            "physical_actions": rule.n + knocks, "actions": actions,
            "objects_cleared": len(cleared),
            "runtime_sec": time.perf_counter() - started}
    if include_solution:
        result["selected_actions"] = selected_actions
        result["order"] = order
    return result


def load_execution_cases(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["case"]): row for row in data.get("cases", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="results/isaac_problems/*.yaml")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--proposed-runs", type=Path,
                        default=Path("runs/20260909-230426/overall_summary.json"))
    parser.add_argument("--random-push-runs", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("results/baselines"))
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    problem_names = {p.stem for p in ROOT.glob(args.problems)}
    instances = {i.name: i for i in solver.all_benchmarks()
                 if i.name in problem_names}
    missing = sorted(problem_names - instances.keys())
    if missing:
        raise SystemExit(f"No geometry instances for: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    proposed = load_execution_cases(args.proposed_runs)
    random_push = load_execution_cases(args.random_push_runs)
    trials: list[dict] = []
    rows: list[dict] = []
    for case_index, name in enumerate(sorted(instances)):
        rule = solver.preprocess(instances[name], prune_dominated=True)
        feasible_knocks = solver.isaac_feasible_knock_actions(rule)
        optimal_result = solver.solve_ilp(rule)
        optimum = optimal_result.optimum
        case_trials = []
        for trial in range(args.trials):
            trial_seed = args.seed + case_index * args.trials + trial
            result = random_clearance_trial(
                rule, trial_seed, feasible_knocks=feasible_knocks)
            record = {"case": name, "trial": trial, "seed": trial_seed,
                      **result}
            trials.append(record)
            case_trials.append(record)
        values = [r["knocks"] for r in case_trials if r["success"]]
        runtimes = [r["runtime_sec"] for r in case_trials]
        physical_actions = [r["physical_actions"] for r in case_trials
                            if r["success"]]
        proposed_case = proposed.get(name, {})
        push_case = random_push.get(name, {})
        rows.append({
            "case": name,
            "family": instances[name].metadata.get(
                "experiment_category", instances[name].metadata.get("family", "")),
            "n": rule.n,
            "optimal_knocks": optimum,
            "optimal_solver_time_sec": optimal_result.runtime_sec,
            "random_clearance_trials": len(case_trials),
            "random_clearance_success_rate": sum(r["success"] for r in case_trials) / len(case_trials),
            "random_clearance_knocks_mean": statistics.fmean(values) if values else None,
            "random_clearance_knocks_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "random_clearance_knocks_min": min(values) if values else None,
            "random_clearance_knocks_max": max(values) if values else None,
            "random_clearance_regret_mean": (statistics.fmean(values) - optimum) if values else None,
            "random_clearance_runtime_sec_mean": statistics.fmean(runtimes),
            "random_clearance_runtime_sec_std": statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0,
            "random_clearance_physical_actions_mean": statistics.fmean(physical_actions) if physical_actions else None,
            "proposed_execution_runs": proposed_case.get("n_runs"),
            "proposed_execution_success_rate": proposed_case.get("success_rate"),
            "proposed_object_completion_mean": proposed_case.get("object_completion_percentage_mean"),
            "proposed_execution_time_sec_mean": proposed_case.get("execution_time_sec_mean"),
            "random_push_runs": push_case.get("n_runs"),
            "random_push_success_rate": push_case.get("success_rate"),
            "random_push_count_mean": push_case.get("push_actions_mean"),
            "random_push_successful_count_mean": push_case.get("successful_pushes_mean"),
            "random_push_failed_count_mean": push_case.get("failed_pushes_mean"),
            "random_push_pick_attempts_mean": push_case.get("pick_attempts_mean"),
            "random_push_successful_picks_mean": push_case.get("successful_picks_mean"),
            "random_push_deadlocks_mean": push_case.get("deadlocks_mean"),
            "random_push_replan_time_sec_mean": push_case.get("live_replan_time_sec_mean"),
            "random_push_execution_time_sec_mean": push_case.get("execution_time_sec_mean"),
            "random_push_object_completion_mean": push_case.get("object_completion_percentage_mean"),
        })

    (args.output / "random_clearance_trials.json").write_text(
        json.dumps(trials, indent=2), encoding="utf-8")
    with (args.output / "random_clearance_trials.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [k for k in trials[0] if k != "actions"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(trials)
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Baseline comparison", "",
             "| Case | n | Optimal K | Random K mean±std | Regret | Proposed success | Random-push success | Random-push completion |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        fmt = lambda x, p=3: "NA" if x is None else f"{float(x):.{p}f}"
        lines.append(
            f"| {r['case']} | {r['n']} | {r['optimal_knocks']} | "
            f"{fmt(r['random_clearance_knocks_mean'])}±{fmt(r['random_clearance_knocks_std'])} | "
            f"{fmt(r['random_clearance_regret_mean'])} | "
            f"{fmt(r['proposed_execution_success_rate'])} | "
            f"{fmt(r['random_push_success_rate'])} | "
            f"{fmt(r['random_push_object_completion_mean'])} |"
        )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nDetailed outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
