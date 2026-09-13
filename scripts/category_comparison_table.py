#!/usr/bin/env python3
"""Merge the three full benchmark tables into a paper-ready category table.

Each reported mean and sample standard deviation is across case-level means,
not across pooled Isaac runs. The input tables must contain the same cases.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


FIELDS = [
    "Category", "Method", "Plan time (ms)", "Total actions",
    "Clearance actions", "Clearance attempts", "Execution time (s)",
    "Failed actions", "Success (%)", "Objects (%)",
]
METHODS = ("Proposed", "Random Knock", "Random Push")


def read_cases(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    cases = {row["case"]: row for row in rows}
    if len(cases) != len(rows):
        raise ValueError(f"Duplicate cases in {path}")
    return cases


def value(row: dict[str, str], key: str) -> float:
    raw = row.get(key)
    if raw in (None, "", "NA"):
        raise ValueError(f"Missing required metric {key!r} for {row['case']}")
    return float(raw)


def summarize(values: list[float], digits: int = 2) -> str:
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def category_rows(batch: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    ours = read_cases(batch / "table_1_our_method_full.csv")
    knock = read_cases(batch / "table_2_planning_vs_random_knock.csv")
    push = read_cases(batch / "table_3_ours_vs_random_push.csv")
    if not ours or set(ours) != set(knock) or set(ours) != set(push):
        raise ValueError("The three full tables must have the same nonempty case set")
    groups: dict[str, list[str]] = {}
    for case, row in ours.items():
        family = row["family"]
        if family not in ("axis-aligned", "non-axis-aligned"):
            raise ValueError(f"Unexpected category {family!r} for {case}")
        if knock[case]["family"] != family or push[case]["family"] != family:
            raise ValueError(f"Category mismatch for {case}")
        groups.setdefault(family, []).append(case)

    def metrics(method: str, case: str) -> dict[str, float]:
        a, b, c = ours[case], knock[case], push[case]
        if method == "Proposed":
            return {
                "Plan time (ms)": 1000 * (value(a, "planner_preprocess_sec") + value(a, "planner_ILP_sec")),
                "Total actions": value(a, "isaac_actions_planned_mean"),
                "Clearance actions": value(a, "isaac_clearance_actions_mean"),
                "Clearance attempts": value(a, "isaac_clearance_attempts_mean"),
                "Execution time (s)": value(a, "isaac_execution_time_sec_mean"),
                "Failed actions": value(a, "isaac_failed_picks_mean") + value(a, "isaac_failed_knocks_mean"),
                "Success (%)": 100 * value(a, "isaac_success_rate"),
                "Objects (%)": value(a, "isaac_object_completion_percentage_mean"),
            }
        if method == "Random Knock":
            prefix = "random_knock_isaac_"
            return {
                "Plan time (ms)": 1000 * (value(b, "ours_preprocess_sec") + value(b, "random_clearance_runtime_sec_mean")),
                "Total actions": value(b, prefix + "actions_planned_mean"),
                "Clearance actions": value(b, prefix + "clearance_actions_mean"),
                "Clearance attempts": value(b, prefix + "clearance_attempts_mean"),
                "Execution time (s)": value(b, prefix + "execution_time_sec_mean"),
                "Failed actions": value(b, prefix + "failed_picks_mean") + value(b, prefix + "failed_knocks_mean"),
                "Success (%)": 100 * value(b, prefix + "success_rate"),
                "Objects (%)": value(b, prefix + "object_completion_percentage_mean"),
            }
        prefix = "random_push_isaac_"
        return {
            "Plan time (ms)": 1000 * value(c, prefix + "live_replan_time_sec_mean"),
            "Total actions": value(c, prefix + "pick_attempts_mean") + value(c, prefix + "push_attempts_mean"),
            "Clearance actions": value(c, prefix + "push_actions_mean"),
            "Clearance attempts": value(c, prefix + "push_attempts_mean"),
            "Execution time (s)": value(c, prefix + "execution_time_sec_mean"),
            "Failed actions": value(c, prefix + "failed_picks_mean") + value(c, prefix + "failed_pushes_mean"),
            "Success (%)": 100 * value(c, prefix + "success_rate"),
            "Objects (%)": value(c, prefix + "object_completion_percentage_mean"),
        }

    rows = []
    order = ("axis-aligned", "non-axis-aligned")
    for category in order:
        if category not in groups or len(groups[category]) < 2:
            raise ValueError(f"At least two cases required in {category}")
        for method in METHODS:
            cases = [metrics(method, case) for case in sorted(groups[category])]
            row = {"Category": category, "Method": method}
            for field in FIELDS[2:]:
                values = [entry[field] for entry in cases]
                row[field] = (f"{statistics.mean(values):.1f}"
                              if field == "Success (%)" else summarize(values, 3 if field == "Failed actions" else 2))
            rows.append(row)
    return rows, {category: len(cases) for category, cases in groups.items()}


def write_tables(batch: Path, rows: list[dict[str, str]], counts: dict[str, int]) -> Path:
    base = batch / "table_7_category_comparison"
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(FIELDS) + " |"
    divider = "|" + "|".join("---" if i < 2 else "---:" for i in range(len(FIELDS))) + "|"
    lines = ["# Category-level comparison", "", header, divider]
    lines.extend("| " + " | ".join(row[field] for field in FIELDS) + " |" for row in rows)
    base.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def math(value: str, bold: bool = False) -> str:
        latex = value.replace(" ± ", "\\pm ")
        return f"$\\mathbf{{{latex}}}$" if bold else f"${latex}$"

    tex = [
        "% Requires booktabs, multirow, and graphicx.",
        "\\begin{table*}[t]", "\\centering", "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.3pt}",
        ("\\caption{Category-level comparison of the proposed method and both baselines. "
         "Values are mean $\\pm$ sample standard deviation across the case-level means "
         "in each category. $T_{\\rm plan}$ is preprocessing plus MILP for Proposed, "
         "preprocessing plus random selection for Random Knock, and cumulative online "
         "replanning for Random Push. Total actions are high-level Pick-plus-clearance "
         "actions for Proposed and Random Knock, and attempted Pick-plus-Push actions "
         "for Random Push. Clearance attempts count physical Knock or Push contacts.}" ),
        "\\label{tab:category_comparison}",
        "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        ("Category & Method & $T_{\\rm plan}$ (ms) & Total actions & Clearance actions "
         "& Clearance attempts & $T_{\\rm exec}$ (s) & Failed actions & Success (\\%) & Objects (\\%) " + "\\\\"),
        "\\midrule",
    ]
    for index, category in enumerate(("axis-aligned", "non-axis-aligned")):
        if index:
            tex.append("\\midrule")
        label = "Axis-aligned" if index == 0 else "Non-axis-aligned"
        category_rows = [row for row in rows if row["Category"] == category]
        for method_index, method in enumerate(METHODS):
            row = next(row for row in category_rows if row["Method"] == method)
            values = []
            for field in FIELDS[2:]:
                mean = float(row[field].split(" ± ")[0])
                case_means = [float(other[field].split(" ± ")[0]) for other in category_rows]
                best = max(case_means) if field in ("Success (%)", "Objects (%)") else min(case_means)
                bold = abs(mean - best) < 1e-9
                if field == "Success (%)":
                    values.append(f"\\textbf{{{row[field]}}}" if bold else row[field])
                else:
                    values.append(math(row[field], bold))
            prefix = f"\\multirow{{3}}{{*}}{{{label}}}" if method_index == 0 else ""
            tex.append(" & ".join([prefix, method, *values]) + " " + "\\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""])
    base.with_suffix(".tex").write_text("\n".join(tex), encoding="utf-8")

    manifest_path = batch / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["category_comparison_table"] = str(base.resolve())
        manifest["category_case_counts"] = counts
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved category comparison: {base}.csv/.md/.tex")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="Directory containing the three full-result CSV tables")
    args = parser.parse_args()
    batch = args.batch.resolve()
    rows, counts = category_rows(batch)
    write_tables(batch, rows, counts)


if __name__ == "__main__":
    main()
