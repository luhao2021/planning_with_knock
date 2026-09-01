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

    def test_mixed_height_degree_uses_candidate_height_slice(self) -> None:
        occupancy = PLANNER.occ_full(2, 2)
        heights = {
            (0, 0): 0.05,
            (0, 1): 0.10,
            (1, 0): 0.15,
            (1, 1): 0.05,
        }
        graph = PLANNER.build_graph(occupancy, node_heights=heights)

        self.assertEqual(
            {node: PLANNER.height_aware_degree(graph, node) for node in graph},
            {(0, 0): 2, (0, 1): 0, (1, 0): 0, (1, 1): 2},
        )
        self.assertTrue(PLANNER.pickable(graph, (0, 1)))
        self.assertTrue(PLANNER.pickable(graph, (1, 0)))
        self.assertFalse(PLANNER.pickable(graph, (0, 0)))
        self.assertFalse(PLANNER.pickable(graph, (1, 1)))
        # Knocking deliberately retains the original occupancy degree/ray.
        self.assertTrue(all(graph.degree(node) == 2 for node in graph))
        self.assertTrue(all(PLANNER.knockable(graph, node, 2, 2) for node in graph))

        metrics, plan = PLANNER.run_matching_solver(
            occupancy, 2, 2, node_heights=heights
        )
        self.assertTrue(metrics.ok)
        self.assertEqual(metrics.knocks, 0)
        self.assertEqual(metrics.picks, 4)
        self.assertTrue(all(step.kind == "pick" for step in plan))

    def test_equal_height_2x2_still_requires_one_knock(self) -> None:
        occupancy = PLANNER.occ_full(2, 2)
        heights = {node: 0.15 for node in occupancy}
        metrics, plan = PLANNER.run_matching_solver(
            occupancy, 2, 2, node_heights=heights
        )
        self.assertTrue(metrics.ok)
        self.assertEqual(metrics.knocks, 1)
        self.assertEqual(metrics.picks, 3)
        self.assertEqual(len(plan), 4)

    def test_protected_random_subgraph_preserves_forced_gadget(self) -> None:
        protected = PLANNER._corner_gadget("2x3")
        occupancy = PLANNER.random_hole_free_subgraph_by_removal(
            8, 8, 38, seed=19, protected=protected
        )
        self.assertEqual(len(occupancy), 38)
        self.assertTrue(protected.issubset(occupancy))
        self.assertTrue(PLANNER._is_connected_occ(occupancy, 8, 8))
        self.assertTrue(PLANNER._is_hole_free(occupancy, 8, 8))

        heights = PLANNER.random_height_assignment(
            occupancy, seed=23, forced_equal_cells=protected
        )
        self.assertEqual({heights[node] for node in protected}, {4.0})
        metrics, _ = PLANNER.run_matching_solver(
            occupancy, 8, 8, node_heights=heights
        )
        self.assertGreaterEqual(metrics.knocks, 1)

    def test_degree_sequence_plot_contains_both_degree_models(self) -> None:
        occupancy = PLANNER.occ_full(2, 2)
        heights = {node: 3.0 for node in occupancy}
        _, plan = PLANNER.run_matching_solver(
            occupancy, 2, 2, node_heights=heights
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "degrees.png"
            PLANNER.plot_degree_sequence(
                occupancy, heights, plan, str(output), "test 2x2"
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)

            graph_output = Path(temp_dir) / "graphs.png"
            PLANNER.plot_graph_sequence(
                occupancy, heights, plan, 2, 2, str(graph_output), "test 2x2"
            )
            self.assertTrue(graph_output.is_file())
            self.assertGreater(graph_output.stat().st_size, 1000)

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
        heights = [0.05, 0.10, 0.15]
        node_heights = PLANNER.assign_node_heights(occupancy, heights)
        metrics, plan = PLANNER.run_matching_solver(
            occupancy, 3, 3, node_heights=node_heights
        )
        self.assertTrue(metrics.ok)

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
