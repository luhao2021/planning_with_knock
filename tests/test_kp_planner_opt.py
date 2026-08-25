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

PROBLEM_MODULE_PATH = (
    Path(__file__).parents[1] / "src" / "isaac_executor" / "scripts" / "problem.py"
)
PROBLEM_SPEC = importlib.util.spec_from_file_location("isaac_problem", PROBLEM_MODULE_PATH)
assert PROBLEM_SPEC and PROBLEM_SPEC.loader
PROBLEM = importlib.util.module_from_spec(PROBLEM_SPEC)
sys.modules[PROBLEM_SPEC.name] = PROBLEM
PROBLEM_SPEC.loader.exec_module(PROBLEM)

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

    def test_variable_height_cuboids_drive_manipulation_geometry(self) -> None:
        occupancy = PLANNER.occ_full(3, 3)
        metrics, plan = PLANNER.run_matching_solver(occupancy, 3, 3)
        self.assertTrue(metrics.ok)
        heights = [0.05, 0.10, 0.15]

        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cuboids.yaml"
            PLANNER.write_problem_yaml(
                output, 3, 3, occupancy, 0.05, 0.005, 0.45, 0.0, plan,
                heights=heights,
            )
            problem = PROBLEM.load_problem(output)

        actual_heights = [cuboid.height for cuboid in problem.cuboids.values()]
        self.assertEqual(actual_heights[:3], heights)
        short = next(iter(problem.cuboids.values()))
        tall = list(problem.cuboids.values())[2]
        self.assertLess(PROBLEM.grasp_height(short), PROBLEM.grasp_height(tall))
        self.assertLess(PROBLEM.contact_height(short), PROBLEM.contact_height(tall))

    def test_cuboid_height_must_be_side_multiple(self) -> None:
        occupancy = PLANNER.occ_full(2, 2)
        _, plan = PLANNER.run_matching_solver(occupancy, 2, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "integer multiple"):
                PLANNER.write_problem_yaml(
                    Path(temp_dir) / "invalid.yaml", 2, 2, occupancy,
                    0.05, 0.005, 0.45, 0.0, plan, heights=[0.075],
                )

    def test_ros_knock_directions_are_rotated_into_isaac(self) -> None:
        expected = {
            "+x": (0.0, 1.0, 0.0),
            "-x": (0.0, -1.0, 0.0),
            "+y": (-1.0, 0.0, 0.0),
            "-y": (1.0, 0.0, 0.0),
        }
        for direction, isaac_vector in expected.items():
            with self.subTest(direction=direction):
                self.assertEqual(
                    PROBLEM.knock_direction_to_isaac(direction), isaac_vector
                )

if __name__ == "__main__":
    unittest.main()
