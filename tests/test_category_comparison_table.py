"""Regression checks for the three-method category report."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from category_comparison_table import category_rows, write_tables  # noqa: E402


class CategoryComparisonTests(unittest.TestCase):
    def test_category_means_use_case_means_and_all_three_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            for index, filename in enumerate((
                    "table_1_our_method_full.csv",
                    "table_2_planning_vs_random_knock.csv",
                    "table_3_ours_vs_random_push.csv")):
                rows = []
                for category in ("axis-aligned", "non-axis-aligned"):
                    for case_number in (1, 2):
                        case = f"{category}-{case_number}"
                        common = {"case": case, "family": category}
                        if index == 0:
                            rows.append({**common, "planner_preprocess_sec": "0.01",
                                         "planner_ILP_sec": "0.02",
                                         "isaac_actions_planned_mean": str(case_number + 2),
                                         "isaac_clearance_actions_mean": "1",
                                         "isaac_clearance_attempts_mean": "2",
                                         "isaac_execution_time_sec_mean": "10",
                                         "isaac_failed_picks_mean": "0",
                                         "isaac_failed_knocks_mean": "0",
                                         "isaac_success_rate": "1",
                                         "isaac_object_completion_percentage_mean": "100"})
                        elif index == 1:
                            rows.append({**common, "ours_preprocess_sec": "0.01",
                                         "random_clearance_runtime_sec_mean": "0.001",
                                         "random_knock_isaac_actions_planned_mean": "5",
                                         "random_knock_isaac_clearance_actions_mean": "3",
                                         "random_knock_isaac_clearance_attempts_mean": "4",
                                         "random_knock_isaac_execution_time_sec_mean": "12",
                                         "random_knock_isaac_failed_picks_mean": "0",
                                         "random_knock_isaac_failed_knocks_mean": "0",
                                         "random_knock_isaac_success_rate": "1",
                                         "random_knock_isaac_object_completion_percentage_mean": "100"})
                        else:
                            rows.append({**common, "random_push_isaac_live_replan_time_sec_mean": "0.1",
                                         "random_push_isaac_pick_attempts_mean": "4",
                                         "random_push_isaac_push_attempts_mean": "3",
                                         "random_push_isaac_push_actions_mean": "3",
                                         "random_push_isaac_execution_time_sec_mean": "20",
                                         "random_push_isaac_failed_picks_mean": "1",
                                         "random_push_isaac_failed_pushes_mean": "2",
                                         "random_push_isaac_success_rate": "0.5",
                                         "random_push_isaac_object_completion_percentage_mean": "80"})
                fields = list(rows[0])
                with (batch / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            rows, counts = category_rows(batch)
            self.assertEqual(counts, {"axis-aligned": 2, "non-axis-aligned": 2})
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["Plan time (ms)"], "30.00 ± 0.00")
            self.assertEqual(rows[0]["Total actions"], "3.50 ± 0.71")
            self.assertEqual(rows[1]["Clearance attempts"], "4.00 ± 0.00")
            self.assertEqual(rows[2]["Failed actions"], "3.000 ± 0.000")
            write_tables(batch, rows, counts)
            tex = (batch / "table_7_category_comparison.tex").read_text()
            self.assertIn("\\multirow{3}{*}{Axis-aligned}", tex)
            self.assertIn("\\textbf{100.0}", tex)


if __name__ == "__main__":
    unittest.main()
