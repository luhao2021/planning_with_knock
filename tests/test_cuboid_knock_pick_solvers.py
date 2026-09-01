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
    def test_expanded_suite_contains_partial_contacts_and_two_clumps(self):
        suite = SOLVER.all_benchmarks()
        families = [instance.metadata.get("family") for instance in suite]
        self.assertGreaterEqual(families.count("partial-contact"), 4)
        self.assertGreaterEqual(families.count("two-clump"), 4)

    def test_two_clumps_need_multiple_clearances(self):
        instance = SOLVER.two_clump_instances()[0]
        rule = SOLVER.preprocess(instance)
        result = SOLVER.solve_ilp(rule)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.optimum, 2)

    def test_isaac_export_uses_clearance_and_swaps_rotated_axes(self):
        instance = next(i for i in SOLVER.canonical_instances()
                        if i.name == "nonuniform_2x3_6")
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
        self.assertGreaterEqual(document["execution"]["max_clearance_attempts"], 10)

    def test_exported_rectangular_problem_loads(self):
        instance = SOLVER.two_clump_instances()[0]
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


if __name__ == "__main__":
    unittest.main()
