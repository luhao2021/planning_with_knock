"""Regression tests for heterogeneous cuboid planning and Isaac export."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

import yaml


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cuboid_solver", ROOT / "cuboid_knock_pick_solvers.py"
)
SOLVER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SOLVER
SPEC.loader.exec_module(SOLVER)
PROBLEM_SPEC = importlib.util.spec_from_file_location(
    "isaac_problem", ROOT / "src/isaac_executor/scripts/problem.py"
)
PROBLEM = importlib.util.module_from_spec(PROBLEM_SPEC)
assert PROBLEM_SPEC.loader is not None
sys.modules[PROBLEM_SPEC.name] = PROBLEM
PROBLEM_SPEC.loader.exec_module(PROBLEM)


class CuboidSolverTests(unittest.TestCase):
    def test_primary_experiment_has_two_balanced_categories(self):
        suite = SOLVER.primary_experiment_instances()
        categories = [instance.metadata["experiment_category"] for instance in suite]
        self.assertEqual(categories.count("axis-aligned"), 20)
        self.assertEqual(categories.count("non-axis-aligned"), 20)
        self.assertFalse(any(instance.metadata.get("family") in
                             {"two-clump", "yaw-single-clump", "yaw-multi-clump"}
                             for instance in suite))

    def test_dense_independent_yaw_primary_cases_are_locked_and_exportable(self):
        cases = SOLVER.non_axis_aligned_experiment_instances()
        self.assertEqual(len(cases), 20)
        self.assertGreater(len({len(instance.rects) for instance in cases}), 4)
        for instance in cases:
            metrics = SOLVER.clutter_metrics(instance)
            self.assertGreater(metrics["packing_fraction"], 0.55)
            self.assertGreater(metrics["near_neighbor_density"], 0.30)
            SOLVER.isaac_scale(instance)
            self.assertGreaterEqual(
                SOLVER.solve_ilp(SOLVER.preprocess(instance)).optimum, 1)

    def test_partial_window_reserves_physical_finger_corridor(self):
        instance = next(i for i in SOLVER.extended_contact_instances()
                        if i.name == "partial_window_8")
        scale = SOLVER.isaac_scale(instance)
        self.assertGreaterEqual(instance.finger_clearance * scale, 0.022)
        rule = SOLVER.preprocess(instance)
        self.assertEqual(SOLVER.solve_ilp(rule).optimum, 1)

    def test_expanded_suite_contains_partial_contacts_and_two_clumps(self):
        suite = SOLVER.all_benchmarks()
        families = [instance.metadata.get("family") for instance in suite]
        self.assertGreaterEqual(families.count("partial-contact"), 4)
        self.assertGreaterEqual(families.count("two-clump"), 4)

    def test_yaw_suite_covers_all_three_models(self):
        suite = SOLVER.yaw_instances()
        self.assertEqual(len(suite), 15)
        families = {i.metadata["family"] for i in suite}
        self.assertEqual(families, {"yaw-single-clump", "yaw-multi-clump",
                                    "yaw-independent"})
        for instance in suite:
            SOLVER.validate_geometry(instance)
            self.assertTrue(any(abs(rect.yaw) > 1e-8 for rect in instance.rects))

    def test_compact_missing_family_cases_are_isaac_exportable(self):
        expected = {
            "two_clumps_compact_2x2_8",
            "two_clumps_compact_2x2_3x2_10",
            "two_clumps_compact_3x2_2x2_10",
            "guillotine_compact_n6_s2",
            "guillotine_compact_n7_s6",
            "guillotine_compact_n8_s5",
            "yaw_multi_clump_compact_1_8",
            "yaw_multi_clump_compact_2_8",
            "yaw_multi_clump_compact_3_8",
        }
        instances = {i.name: i for i in SOLVER.all_benchmarks()}
        with tempfile.TemporaryDirectory() as directory:
            for name in sorted(expected):
                rule = SOLVER.preprocess(instances[name])
                result = SOLVER.solve_ilp(rule)
                SOLVER.write_isaac_problem(
                    rule, result, Path(directory) / f"{name}.yaml")

    def test_rotated_dense_clump_preserves_nontrivial_optimum(self):
        instance = SOLVER.yaw_instances()[0]
        rule = SOLVER.preprocess(instance)
        ilp = SOLVER.solve_ilp(rule)
        sat = SOLVER.solve_sat(rule)
        self.assertGreater(ilp.optimum, 0)
        self.assertEqual(ilp.optimum, sat.optimum)

    def test_arbitrary_yaw_projection_creates_finite_clearance_rules(self):
        instance = next(i for i in SOLVER.yaw_instances()
                        if i.name == "yaw_independent_4_9")
        self.assertGreater(instance.finger_clearance, 0)
        self.assertTrue(any(SOLVER.contact_intervals(instance, i, o)
                            for i, rect in enumerate(instance.rects)
                            for o in SOLVER.admissible_orientations(
                                rect, instance.D, instance.jaw_margin)))

    def test_locked_independent_yaw_cases_require_clearance(self):
        cases = [i for i in SOLVER.yaw_instances()
                 if i.name.startswith("yaw_independent_locked_")]
        self.assertEqual(len(cases), 2)
        for instance in cases:
            rule = SOLVER.preprocess(instance)
            ilp = SOLVER.solve_ilp(rule)
            sat = SOLVER.solve_sat(rule)
            self.assertIsNotNone(ilp.optimum)
            self.assertGreaterEqual(ilp.optimum, 1)
            self.assertEqual(ilp.optimum, sat.optimum)

    def test_yaw_round_trips_through_isaac_problem(self):
        instance = SOLVER.yaw_instances()[0]
        result = SOLVER.solve_ilp(SOLVER.preprocess(instance))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yaw.yaml"
            document = SOLVER.write_isaac_problem(
                SOLVER.preprocess(instance), result, path)
            problem = PROBLEM.load_problem(path)
        first = instance.rects[0]
        self.assertAlmostEqual(problem.cuboids[first.name].yaw_rad, first.yaw)
        self.assertEqual(document["yaw_type"], "common")

    def test_two_clumps_need_multiple_clearances(self):
        instance = SOLVER.two_clump_instances()[0]
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.optimum, 2)

    def test_isaac_export_uses_clearance_and_swaps_rotated_axes(self):
        instance = next(i for i in SOLVER.canonical_instances()
                        if i.name == "long_spine_11")
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problem.yaml"
            document = SOLVER.write_isaac_problem(rule, result, path)
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(document["schema_version"], 2)
        self.assertIn("Clearance", [a["type"] for a in loaded["actions"]])
        self.assertNotIn("Knock", [a["type"] for a in loaded["actions"]])
        first = instance.rects[0]
        scale = document["scale"]
        dims = loaded["objects"][first.name]["dimensions"]
        self.assertAlmostEqual(dims[0], first.h * scale - document["contact_gap"])
        self.assertAlmostEqual(dims[1], first.w * scale - document["contact_gap"])
        self.assertAlmostEqual(document["contact_gap"], 0.005)
        self.assertAlmostEqual(document["ros_x_workspace_shift"], 0.05)
        goal_pads = {item["pad"] for item in document["goal"]}
        xmin = min(rect.xmin for rect in instance.rects)
        xmax = max(rect.xmax for rect in instance.rects)
        expected_first_x = (
            0.45 + scale * (
                0.5 * (first.xmin + first.xmax) - 0.5 * (xmin + xmax)
            )
        )
        self.assertAlmostEqual(
            document["pads"][document["initial"][0]["pad"]]["x"],
            expected_first_x,
        )
        self.assertEqual(goal_pads, {
            document["sorting"]["chutes"]["red"]["pad"],
            document["sorting"]["chutes"]["blue"]["pad"],
        })
        self.assertEqual(
            document["sorting"]["chutes"]["red"]["center_isaac"],
            [-0.13, 0.82],
        )
        self.assertEqual(
            document["sorting"]["chutes"]["blue"]["center_isaac"],
            [0.13, 0.82],
        )
        self.assertEqual(
            document["sorting"]["table_bounds_isaac"]["ymax"], 0.78
        )
        for item in document["goal"]:
            self.assertEqual(
                item["pad"],
                document["sorting"]["chutes"][item["sort_color"]]["pad"],
            )
        self.assertGreaterEqual(document["execution"]["max_clearance_attempts"], 10)

    def test_exported_rectangular_problem_loads(self):
        instance = next(i for i in SOLVER.canonical_instances()
                        if i.name == "long_spine_11")
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "problem.yaml"
            SOLVER.write_isaac_problem(rule, result, path)
            problem = PROBLEM.load_problem(path)
        self.assertTrue(any(c.dimensions[0] != c.dimensions[1]
                            for c in problem.cuboids.values()))
        self.assertTrue(any(a["type"] == "Clearance" for a in problem.actions))
        self.assertGreaterEqual(problem.max_clearance_attempts, 10)

    def test_physically_unscalable_case_is_not_exported(self):
        instance = SOLVER.two_clump_instances()[0]
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no physical scale"):
                SOLVER.write_isaac_problem(
                    rule, result, Path(directory) / "problem.yaml")

    def test_every_exported_finger_window_is_full_width_and_inside_side(self):
        with tempfile.TemporaryDirectory() as directory:
            for instance in SOLVER.all_benchmarks():
                rule = SOLVER.preprocess(instance)
                result = SOLVER.solve_ilp(rule)
                try:
                    document = SOLVER.write_isaac_problem(
                        rule, result, Path(directory) / f"{instance.name}.yaml")
                except ValueError:
                    continue
                width = document["execution"]["finger_side_contact_width"]
                gap = document["contact_gap"]
                for action in document["actions"]:
                    if action["type"] != "Move":
                        continue
                    grasp = action["grasp"]
                    start, end = grasp["finger_window"]
                    self.assertAlmostEqual(end - start, width, places=7)
                    planning_w, planning_h = document["objects"][
                        action["block"]]["planning_dimensions"]
                    tangential = (planning_h if grasp["closing_axis"] == "x"
                                  else planning_w) - gap
                    self.assertGreaterEqual(start, -0.5*tangential - 1e-7)
                    self.assertLessEqual(end, 0.5*tangential + 1e-7)

    def test_double_rail_boundary_witness_keeps_whole_finger_on_rail(self):
        instance = next(i for i in SOLVER.all_benchmarks()
                        if i.name == "double_rail_7")
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        with tempfile.TemporaryDirectory() as directory:
            document = SOLVER.write_isaac_problem(
                rule, result, Path(directory) / "double_rail.yaml")
        moves = {a["block"]: a for a in document["actions"]
                 if a["type"] == "Move"}
        rail_span = document["objects"]["Top"]["planning_dimensions"][0]
        physical_span = rail_span - document["contact_gap"]
        for name in ("Top", "Bot"):
            grasp = moves[name]["grasp"]
            start, end = grasp["finger_window"]
            self.assertEqual(moves[name]["solver_witness"], "pick-y@-3")
            self.assertAlmostEqual(
                end - start, SOLVER.PANDA_FINGER_TANGENTIAL_EXTENT, places=7)
            self.assertGreaterEqual(
                start - (-0.5 * physical_span),
                SOLVER.PANDA_FINGER_EDGE_MARGIN - 1e-7)
            self.assertAlmostEqual(grasp["offset"][0], 0.5 * (start + end),
                                   places=7)

    def test_exported_clearances_are_object_local(self):
        instance = next(i for i in SOLVER.yaw_instances()
                        if i.name == "yaw_independent_locked_1_4")
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        with tempfile.TemporaryDirectory() as directory:
            document = SOLVER.write_isaac_problem(
                rule, result, Path(directory) / "problem.yaml")
        clearances = [a for a in document["actions"]
                      if a["type"] == "Clearance"]
        self.assertTrue(clearances)
        self.assertTrue(all(a["direction_frame"] == "object_local"
                            for a in clearances))

    def test_export_preserves_noncentral_grasp_witness(self):
        instance = next(i for i in SOLVER.canonical_instances()
                        if i.name == "gap_grasp_5")
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        with tempfile.TemporaryDirectory() as directory:
            document = SOLVER.write_isaac_problem(
                rule, result, Path(directory) / "problem.yaml"
            )
        direct_moves = [
            a for a in document["actions"]
            if "grasp" in a and not a.get("after_clearance")
        ]
        self.assertTrue(any(
            abs(value) > 1e-9
            for move in direct_moves for value in move["grasp"]["offset"]
        ))
        for move in direct_moves:
            witness = move["solver_witness"]
            self.assertIn("@", witness)
            self.assertIn("placement_witness", move["grasp"])

    def test_post_clearance_move_uses_exposed_end_grasp(self):
        instance = next(i for i in SOLVER.canonical_instances()
                        if i.name == "long_spine_11")
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        with tempfile.TemporaryDirectory() as directory:
            document = SOLVER.write_isaac_problem(
                rule, result, Path(directory) / "problem.yaml"
            )
        move_index, move = next(
            (i, action) for i, action in enumerate(document["actions"])
            if action.get("after_clearance")
        )
        clearance = document["actions"][move_index - 1]
        self.assertEqual(move["grasp"]["derived_from_clearance_witness"],
                         clearance["solver_witness"])
        self.assertGreater(max(abs(v) for v in move["grasp"]["offset"]),
                           0.05)
        self.assertEqual(move["grasp"]["closing_axis"], "x")


if __name__ == "__main__":
    unittest.main()
