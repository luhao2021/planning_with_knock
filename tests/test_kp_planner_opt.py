"""Regression tests for the public optimal-planner workflow."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest

import yaml

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "kp_planner_opt.py"
SPEC = importlib.util.spec_from_file_location("kp_planner_opt", MODULE_PATH)
assert SPEC and SPEC.loader
PLANNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLANNER
SPEC.loader.exec_module(PLANNER)

class OptimalPlannerRegressionTest(unittest.TestCase):
    def test_full_grid_optimal_knock_counts(self) -> None:
        expected = {(3, 3): 2, (5, 5): 8, (10, 5): 18}
        for (rows, columns), knock_count in expected.items():
            with self.subTest(grid=(rows, columns)):
                metrics, plan = PLANNER.run_matching_solver(
                    PLANNER.occ_full(rows, columns), rows, columns
                )
                self.assertTrue(metrics.ok)
                self.assertEqual(metrics.knocks, knock_count)
                self.assertEqual(metrics.picks, rows * columns - knock_count)
                self.assertIsNotNone(plan)

    def test_pick_and_knock_predicates(self) -> None:
        graph = PLANNER.build_graph(PLANNER.occ_full(2, 2))
        self.assertFalse(any(PLANNER.pickable(graph, node) for node in graph))
        self.assertTrue(all(PLANNER.knockable(graph, node, 2, 2) for node in graph))

    def test_problem_yaml_uses_knock_schema(self) -> None:
        occupancy = PLANNER.occ_full(3, 3)
        metrics, plan = PLANNER.run_matching_solver(occupancy, 3, 3)
        self.assertTrue(metrics.ok)
        self.assertIsNotNone(plan)

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "problem.yaml"
            PLANNER.write_problem_yaml(
                output, 3, 3, occupancy, 0.05, 0.005, 0.45, 0.0, plan
            )
            document = yaml.safe_load(output.read_text(encoding="utf-8"))

        action_types = [action["type"] for action in document["actions"]]
        self.assertIn("Knock", action_types)
        self.assertNotIn("Smash", action_types)
        self.assertEqual(action_types.count("Knock"), metrics.knocks)
        self.assertEqual(action_types.count("Move"), metrics.picks + metrics.knocks)

if __name__ == "__main__":
    unittest.main()
