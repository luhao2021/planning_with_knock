#!/usr/bin/env python3
"""Exact solvers and benchmark generator for heterogeneous cuboid knock-pick planning.

The implementation follows the geometric model in the companion manuscript:
  * equal-height axis-aligned cuboids represented by 2-D rectangles;
  * contact-interface parallel-jaw grasps with tangential contact length F;
  * state-independent finite Pick blocker sets obtained by exact 1-D discretization;
  * full-footprint cardinal escape corridors for directional Knock blocker sets;
  * a contracted monotone planning model in which every rule clears exactly one object.

Two exact optimizers are implemented:
  1) ILP/MILP: one action rule is selected per object and continuous topological
     rank variables enforce acyclicity. Uses scipy.optimize.milp (HiGHS).
  2) SAT: action-selection CNF with an at-most-k Knock bound and lazy cycle
     clauses. If python-sat is installed, it is used automatically; otherwise a
     compact pure-Python DPLL fallback is used. The fallback is intended for the
     included moderate benchmarks; for larger instances install python-sat.

The script also generates deterministic representative and random geometric
benchmarks, solves them with both methods, verifies the returned schedules, and
writes CSV/Markdown summaries plus 2-D layout and solution-sequence figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple
import argparse
import csv
import itertools
import math
import random
import time
import yaml

import numpy as np
import networkx as nx
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

try:
    from pysat.solvers import Solver as PySATSolver  # type: ignore
    HAVE_PYSAT = True
except Exception:
    HAVE_PYSAT = False


EPS = 1e-9
DIRECTIONS = ("+x", "-x", "+y", "-y")


@dataclass(frozen=True)
class Rect:
    name: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def w(self) -> float:
        return self.xmax - self.xmin

    @property
    def h(self) -> float:
        return self.ymax - self.ymin

    @property
    def min_side(self) -> float:
        return min(self.w, self.h)

    @property
    def max_side(self) -> float:
        return max(self.w, self.h)


@dataclass(frozen=True)
class Action:
    target: int
    kind: str                    # "pick" or "knock"
    blockers: FrozenSet[int]
    cost: int
    witness: str                 # e.g., "pick-x@1.200" or "knock-+x"
    orientation: Optional[str] = None
    placement: Optional[float] = None
    direction: Optional[str] = None


@dataclass
class GeometryInstance:
    name: str
    rects: List[Rect]
    F: float
    D: float
    jaw_margin: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RuleInstance:
    geometry: GeometryInstance
    actions: Dict[int, List[Action]]

    @property
    def n(self) -> int:
        return len(self.geometry.rects)

    @property
    def names(self) -> List[str]:
        return [r.name for r in self.geometry.rects]


@dataclass
class SolveResult:
    solver: str
    status: str
    optimum: Optional[int]
    runtime_sec: float
    selected_actions: Optional[Dict[int, Action]] = None
    order: Optional[List[int]] = None
    extra: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Geometry and rule preprocessing
# ---------------------------------------------------------------------------

def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def validate_geometry(inst: GeometryInstance) -> None:
    if inst.F <= 0 or inst.D <= 0:
        raise ValueError("F and D must be positive")
    n = len(inst.rects)
    for i, r in enumerate(inst.rects):
        if r.w <= EPS or r.h <= EPS:
            raise ValueError(f"Degenerate rectangle {r.name}")
        if r.min_side + EPS < inst.F:
            raise ValueError(
                f"{r.name}: min side {r.min_side:.4g} < F={inst.F:.4g}; "
                "singleton finger-contact admissibility fails"
            )
        if r.min_side + inst.jaw_margin > inst.D + EPS:
            raise ValueError(
                f"{r.name}: neither closing axis fits D={inst.D:.4g}"
            )
        for j in range(i):
            s = inst.rects[j]
            if overlap_len(r.xmin, r.xmax, s.xmin, s.xmax) > EPS and \
               overlap_len(r.ymin, r.ymax, s.ymin, s.ymax) > EPS:
                raise ValueError(f"Interior overlap: {r.name} and {s.name}")


def admissible_orientations(r: Rect, D: float, jaw_margin: float) -> List[str]:
    out = []
    if r.w + jaw_margin <= D + EPS:
        out.append("x")
    if r.h + jaw_margin <= D + EPS:
        out.append("y")
    return out


def contact_intervals(inst: GeometryInstance, i: int, orientation: str) -> Dict[int, List[Tuple[float, float]]]:
    """Return 1-D contact intervals for blockers on the two grasped sides.

    For x-closing, the finger windows lie on the left/right vertical sides and
    intervals are expressed in y. For y-closing, use bottom/top sides and x.
    A neighbor can in principle contribute on only one side in a valid
    non-overlapping rectangle scene, but the list representation keeps the code
    robust and explicit.
    """
    ri = inst.rects[i]
    ans: Dict[int, List[Tuple[float, float]]] = {}
    for j, rj in enumerate(inst.rects):
        if j == i:
            continue
        ivals: List[Tuple[float, float]] = []
        if orientation == "x":
            ov0, ov1 = max(ri.ymin, rj.ymin), min(ri.ymax, rj.ymax)
            if ov1 - ov0 > EPS:
                if abs(rj.xmax - ri.xmin) <= EPS or abs(rj.xmin - ri.xmax) <= EPS:
                    ivals.append((ov0, ov1))
        elif orientation == "y":
            ov0, ov1 = max(ri.xmin, rj.xmin), min(ri.xmax, rj.xmax)
            if ov1 - ov0 > EPS:
                if abs(rj.ymax - ri.ymin) <= EPS or abs(rj.ymin - ri.ymax) <= EPS:
                    ivals.append((ov0, ov1))
        else:
            raise ValueError(orientation)
        if ivals:
            ans[j] = ivals
    return ans


def _positive_window_overlap(t: float, F: float, a: float, b: float) -> bool:
    return min(t + F, b) - max(t, a) > EPS


def pick_actions_for_object(inst: GeometryInstance, i: int) -> List[Action]:
    ri = inst.rects[i]
    candidates: Dict[FrozenSet[int], Action] = {}

    for o in admissible_orientations(ri, inst.D, inst.jaw_margin):
        if o == "x":
            lo, hi = ri.ymin, ri.ymax - inst.F
        else:
            lo, hi = ri.xmin, ri.xmax - inst.F
        if hi < lo - EPS:
            continue

        contacts = contact_intervals(inst, i, o)
        critical = [lo, hi]
        for ivals in contacts.values():
            for a, b in ivals:
                critical.extend([a - inst.F, b])

        pts = sorted({max(lo, min(hi, x)) for x in critical if lo - EPS <= x <= hi + EPS})
        reps = set(pts)
        for a, b in zip(pts, pts[1:]):
            if b - a > EPS:
                reps.add(0.5 * (a + b))
        if not reps:
            reps.add(lo)

        for t in sorted(reps):
            blockers = frozenset(
                j for j, ivals in contacts.items()
                if any(_positive_window_overlap(t, inst.F, a, b) for a, b in ivals)
            )
            a = Action(
                target=i,
                kind="pick",
                blockers=blockers,
                cost=0,
                witness=f"pick-{o}@{t:.6g}",
                orientation=o,
                placement=float(t),
            )
            # One witness per distinct blocker set is sufficient.
            candidates.setdefault(blockers, a)

    # Inclusion-minimal blocker sets only.
    items = list(candidates.items())
    minimal: List[Action] = []
    for B, a in items:
        if any(B2 < B for B2, _ in items):
            continue
        minimal.append(a)
    minimal.sort(key=lambda a: (len(a.blockers), a.orientation or "", a.placement or 0.0))
    return minimal


def knock_blockers(inst: GeometryInstance, i: int, d: str) -> FrozenSet[int]:
    ri = inst.rects[i]
    out = set()
    for j, rj in enumerate(inst.rects):
        if j == i:
            continue
        if d == "+x":
            if overlap_len(ri.ymin, ri.ymax, rj.ymin, rj.ymax) > EPS and rj.xmin >= ri.xmax - EPS:
                out.add(j)
        elif d == "-x":
            if overlap_len(ri.ymin, ri.ymax, rj.ymin, rj.ymax) > EPS and rj.xmax <= ri.xmin + EPS:
                out.add(j)
        elif d == "+y":
            if overlap_len(ri.xmin, ri.xmax, rj.xmin, rj.xmax) > EPS and rj.ymin >= ri.ymax - EPS:
                out.add(j)
        elif d == "-y":
            if overlap_len(ri.xmin, ri.xmax, rj.xmin, rj.xmax) > EPS and rj.ymax <= ri.ymin + EPS:
                out.add(j)
        else:
            raise ValueError(d)
    return frozenset(out)


def nondominated_actions(actions: Sequence[Action]) -> List[Action]:
    """Dominance across both Pick and Knock rules for one target.

    Action a dominates b if a has no greater cost and a blocker subset. Replacing
    b by a can only make execution easier and cannot increase the objective.
    """
    keep: List[Action] = []
    for idx, a in enumerate(actions):
        dominated = False
        for jdx, b in enumerate(actions):
            if idx == jdx:
                continue
            if b.cost <= a.cost and b.blockers.issubset(a.blockers):
                strict = (b.cost < a.cost) or (b.blockers < a.blockers)
                if strict:
                    dominated = True
                    break
                # Exact duplicate: keep deterministic first witness only.
                if b.cost == a.cost and b.blockers == a.blockers and jdx < idx:
                    dominated = True
                    break
        if not dominated:
            keep.append(a)
    keep.sort(key=lambda a: (a.cost, len(a.blockers), a.kind, a.witness))
    return keep


def preprocess(inst: GeometryInstance, prune_dominated: bool = True) -> RuleInstance:
    validate_geometry(inst)
    actions: Dict[int, List[Action]] = {}
    for i in range(len(inst.rects)):
        acts = pick_actions_for_object(inst, i)
        acts += [
            Action(i, "knock", knock_blockers(inst, i, d), 1, f"knock-{d}", direction=d)
            for d in DIRECTIONS
        ]
        actions[i] = nondominated_actions(acts) if prune_dominated else acts
        if not actions[i]:
            raise RuntimeError(f"No action rule for object {inst.rects[i].name}")
    return RuleInstance(inst, actions)


# ---------------------------------------------------------------------------
# Rule-system utilities and verification
# ---------------------------------------------------------------------------

def dependency_graph(selected: Dict[int, Action], n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i, a in selected.items():
        for j in a.blockers:
            G.add_edge(j, i)
    return G


def verify_solution(rule: RuleInstance, selected: Dict[int, Action], order: Sequence[int]) -> Tuple[bool, str]:
    n = rule.n
    if set(selected) != set(range(n)):
        return False, "selected actions do not cover all objects"
    if len(order) != n or set(order) != set(range(n)):
        return False, "order is not a permutation"
    cleared = set()
    for i in order:
        a = selected[i]
        if not a.blockers.issubset(cleared):
            missing = sorted(a.blockers - cleared)
            return False, f"{rule.names[i]} executed before blockers {missing}"
        cleared.add(i)
    return True, "ok"


def physical_plan(rule: RuleInstance, selected: Dict[int, Action], order: Sequence[int]) -> List[str]:
    lines = []
    for i in order:
        a = selected[i]
        name = rule.names[i]
        if a.kind == "pick":
            lines.append(f"Pick({name}; {a.witness})")
        else:
            lines.append(f"Knock({name},{a.direction}) + mandatory PickPlace({name})")
    return lines


# ---------------------------------------------------------------------------
# Exact MILP solver (SciPy/HiGHS)
# ---------------------------------------------------------------------------

def solve_ilp(rule: RuleInstance, time_limit: Optional[float] = None) -> SolveResult:
    t0 = time.perf_counter()
    n = rule.n
    action_list: List[Action] = []
    action_vars_by_target: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for a in rule.actions[i]:
            idx = len(action_list)
            action_list.append(a)
            action_vars_by_target[i].append(idx)

    A = len(action_list)
    N = A + n
    c = np.zeros(N)
    integrality = np.zeros(N, dtype=int)
    lb = np.zeros(N)
    ub = np.ones(N)
    for q, a in enumerate(action_list):
        c[q] = a.cost
        integrality[q] = 1
    # Continuous topological ranks in [0, n-1].
    lb[A:] = 0.0
    ub[A:] = max(0, n - 1)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    row = 0

    # Exactly one action per target.
    for i in range(n):
        for q in action_vars_by_target[i]:
            rows.append(row); cols.append(q); data.append(1.0)
        lower.append(1.0); upper.append(1.0); row += 1

    # If x_{i,a}=1 and j in tail(a): u_j + 1 <= u_i.
    # Equivalent: u_j - u_i + n*x_{i,a} <= n-1.
    M = float(max(1, n))
    for q, a in enumerate(action_list):
        i = a.target
        for j in a.blockers:
            rows.extend([row, row, row])
            cols.extend([A + j, A + i, q])
            data.extend([1.0, -1.0, M])
            lower.append(-np.inf)
            upper.append(M - 1.0)
            row += 1

    mat = coo_matrix((data, (rows, cols)), shape=(row, N)).tocsr()
    constraints = LinearConstraint(mat, np.asarray(lower), np.asarray(upper))
    options = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    res = milp(c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints, options=options)
    runtime = time.perf_counter() - t0

    if res.x is None or not res.success:
        status = f"failed(status={res.status}, message={res.message})"
        return SolveResult("ILP-HiGHS", status, None, runtime, extra={"raw_status": int(res.status)})

    selected: Dict[int, Action] = {}
    for i in range(n):
        q = max(action_vars_by_target[i], key=lambda k: res.x[k])
        selected[i] = action_list[q]
    G = dependency_graph(selected, n)
    if not nx.is_directed_acyclic_graph(G):
        return SolveResult("ILP-HiGHS", "internal-error-cycle", None, runtime, selected)
    order = list(nx.topological_sort(G))
    ok, msg = verify_solution(rule, selected, order)
    if not ok:
        return SolveResult("ILP-HiGHS", f"verification-failed:{msg}", None, runtime, selected, order)
    optimum = sum(a.cost for a in selected.values())
    return SolveResult(
        "ILP-HiGHS", "optimal", int(optimum), runtime, selected, order,
        extra={"mip_objective": float(res.fun), "n_action_vars": A, "n_constraints": row}
    )


# ---------------------------------------------------------------------------
# CNF utilities and SAT engines
# ---------------------------------------------------------------------------

class VarPool:
    def __init__(self):
        self.next_var = 1
        self.labels: Dict[int, str] = {}

    def new(self, label: str) -> int:
        v = self.next_var
        self.next_var += 1
        self.labels[v] = label
        return v


def add_exactly_one(cnf: List[List[int]], vars_: Sequence[int]) -> None:
    if not vars_:
        cnf.append([])
        return
    cnf.append(list(vars_))
    for a, b in itertools.combinations(vars_, 2):
        cnf.append([-a, -b])


def add_at_most_k_sequential(cnf: List[List[int]], xs: Sequence[int], k: int, pool: VarPool) -> None:
    """Sinz-style sequential counter encoding for sum(xs) <= k.

    We use s[i,j] = among x_1..x_i at least j are true, for
    i=1..m and j=1..k. This direct monotone encoding is compact enough for the
    benchmark sizes here and handles k=0 separately.
    """
    xs = list(xs)
    m = len(xs)
    if k < 0:
        cnf.append([]); return
    if k >= m:
        return
    if k == 0:
        for x in xs:
            cnf.append([-x])
        return

    s: Dict[Tuple[int, int], int] = {}
    for i in range(1, m + 1):
        for j in range(1, min(i, k) + 1):
            s[(i, j)] = pool.new(f"seq_{i}_{j}")

    # x_i -> s(i,1)
    for i, x in enumerate(xs, start=1):
        cnf.append([-x, s[(i, 1)]])

    # s(i-1,j) -> s(i,j)
    for i in range(2, m + 1):
        for j in range(1, min(i - 1, k) + 1):
            cnf.append([-s[(i - 1, j)], s[(i, j)]])

    # x_i & s(i-1,j-1) -> s(i,j)
    for i in range(2, m + 1):
        x = xs[i - 1]
        for j in range(2, min(i, k) + 1):
            if (i - 1, j - 1) in s:
                cnf.append([-x, -s[(i - 1, j - 1)], s[(i, j)]])

    # At most k: x_i & s(i-1,k) impossible.
    for i in range(k + 1, m + 1):
        x = xs[i - 1]
        if (i - 1, k) in s:
            cnf.append([-x, -s[(i - 1, k)]])


class SimpleDPLL:
    """Small CNF SAT solver used only when PySAT is unavailable.

    It implements repeated unit propagation plus a Jeroslow-Wang-like branching
    score. It is intentionally compact rather than competitive with CDCL.
    """

    def __init__(self, cnf: Sequence[Sequence[int]], nvars: int):
        self.cnf = [tuple(c) for c in cnf]
        self.nvars = nvars
        self.nodes = 0

    @staticmethod
    def _simplify(clauses: Sequence[Tuple[int, ...]], assignment: Dict[int, bool]):
        out = []
        for clause in clauses:
            sat = False
            rem = []
            for lit in clause:
                v = abs(lit)
                if v in assignment:
                    val = assignment[v]
                    if (lit > 0 and val) or (lit < 0 and not val):
                        sat = True
                        break
                else:
                    rem.append(lit)
            if sat:
                continue
            if not rem:
                return None
            out.append(tuple(rem))
        return out

    def _propagate(self, clauses, assignment):
        while True:
            unit = None
            for c in clauses:
                if len(c) == 1:
                    unit = c[0]
                    break
            if unit is None:
                return clauses, assignment
            v, val = abs(unit), unit > 0
            if v in assignment and assignment[v] != val:
                return None, assignment
            assignment[v] = val
            clauses = self._simplify(clauses, assignment)
            if clauses is None:
                return None, assignment

    def _choose_var(self, clauses, assignment):
        scores: Dict[int, float] = {}
        signs: Dict[int, float] = {}
        for c in clauses:
            weight = 2.0 ** (-len(c))
            for lit in c:
                v = abs(lit)
                if v in assignment:
                    continue
                scores[v] = scores.get(v, 0.0) + weight
                signs[v] = signs.get(v, 0.0) + (weight if lit > 0 else -weight)
        v = max(scores, key=scores.get)
        preferred = signs.get(v, 0.0) >= 0
        return v, preferred

    def _search(self, clauses, assignment):
        self.nodes += 1
        clauses, assignment = self._propagate(clauses, dict(assignment))
        if clauses is None:
            return None
        if not clauses:
            return assignment
        v, pref = self._choose_var(clauses, assignment)
        for val in (pref, not pref):
            a2 = dict(assignment); a2[v] = val
            c2 = self._simplify(clauses, a2)
            if c2 is None:
                continue
            ans = self._search(c2, a2)
            if ans is not None:
                return ans
        return None

    def solve(self) -> Optional[Dict[int, bool]]:
        if any(len(c) == 0 for c in self.cnf):
            return None
        ans = self._search(self.cnf, {})
        if ans is None:
            return None
        return {v: ans.get(v, False) for v in range(1, self.nvars + 1)}


def sat_solve_cnf(cnf: List[List[int]], nvars: int) -> Tuple[Optional[Dict[int, bool]], Dict[str, object]]:
    if HAVE_PYSAT:
        with PySATSolver(name="glucose4", bootstrap_with=cnf) as s:
            ok = s.solve()
            if not ok:
                return None, {"backend": "PySAT-Glucose4"}
            model_lits = s.get_model()
            model = {abs(l): l > 0 for l in model_lits if abs(l) <= nvars}
            return model, {"backend": "PySAT-Glucose4"}
    dpll = SimpleDPLL(cnf, nvars)
    model = dpll.solve()
    return model, {"backend": "SimpleDPLL", "dpll_nodes": dpll.nodes}


def first_selected_cycle(selected: Dict[int, Action], n: int) -> Optional[List[int]]:
    G = dependency_graph(selected, n)
    try:
        cyc = nx.find_cycle(G, orientation="original")
    except nx.NetworkXNoCycle:
        return None
    # nx returns edges; cycle vertices as targets of each edge, then normalize.
    verts = [int(e[0]) for e in cyc]
    return verts


def solve_sat(rule: RuleInstance, time_limit: Optional[float] = None) -> SolveResult:
    """Cost-optimal SAT with lazy acyclicity clauses.

    Base CNF: exactly one action per object. For each Knock budget k we append an
    at-most-k sequential counter. A satisfying action selection is checked for a
    directed dependency cycle; a cycle clause forbidding that simultaneous set of
    selected rules is then added and SAT is repeated. Cycle clauses are globally
    valid and retained across budgets.
    """
    t0 = time.perf_counter()
    n = rule.n
    pool = VarPool()
    action_var: Dict[Tuple[int, int], int] = {}
    var_action: Dict[int, Action] = {}
    base_cnf: List[List[int]] = []

    for i in range(n):
        vars_i = []
        for aidx, a in enumerate(rule.actions[i]):
            v = pool.new(f"a_{i}_{aidx}_{a.witness}")
            action_var[(i, aidx)] = v
            var_action[v] = a
            vars_i.append(v)
        add_exactly_one(base_cnf, vars_i)

    knock_vars = [v for v, a in var_action.items() if a.cost == 1]
    lazy_cycle_clauses: List[List[int]] = []
    total_sat_calls = 0
    total_cycle_cuts = 0
    total_dpll_nodes = 0
    backend = None

    # A trivial admissible lower bound: zero; iterate to first feasible budget.
    for k in range(0, n + 1):
        budget_pool_start = pool.next_var
        # Rebuild the cardinality auxiliaries for this budget so each CNF is self-contained.
        budget_pool = VarPool()
        budget_pool.next_var = pool.next_var
        budget_pool.labels.update(pool.labels)
        cnf = [list(c) for c in base_cnf] + [list(c) for c in lazy_cycle_clauses]
        add_at_most_k_sequential(cnf, knock_vars, k, budget_pool)
        nvars = budget_pool.next_var - 1

        while True:
            if time_limit is not None and time.perf_counter() - t0 > time_limit:
                return SolveResult(
                    "SAT-LazyCycles", "time-limit", None, time.perf_counter() - t0,
                    extra={"backend": backend, "sat_calls": total_sat_calls,
                           "cycle_cuts": total_cycle_cuts, "dpll_nodes": total_dpll_nodes}
                )
            model, info = sat_solve_cnf(cnf, nvars)
            total_sat_calls += 1
            backend = info.get("backend")
            total_dpll_nodes += int(info.get("dpll_nodes", 0))
            if model is None:
                break

            selected: Dict[int, Action] = {}
            selected_var_by_target: Dict[int, int] = {}
            for i in range(n):
                for aidx, a in enumerate(rule.actions[i]):
                    v = action_var[(i, aidx)]
                    if model.get(v, False):
                        selected[i] = a
                        selected_var_by_target[i] = v
                        break
                if i not in selected:
                    # Defensive fallback if an external solver omits false/unassigned literals.
                    true_candidates = [
                        (action_var[(i, aidx)], a)
                        for aidx, a in enumerate(rule.actions[i])
                        if model.get(action_var[(i, aidx)], False)
                    ]
                    if true_candidates:
                        selected_var_by_target[i], selected[i] = true_candidates[0]

            G = dependency_graph(selected, n)
            if nx.is_directed_acyclic_graph(G):
                order = list(nx.topological_sort(G))
                ok, msg = verify_solution(rule, selected, order)
                if not ok:
                    return SolveResult("SAT-LazyCycles", f"verification-failed:{msg}", None,
                                       time.perf_counter() - t0, selected, order)
                optimum = sum(a.cost for a in selected.values())
                return SolveResult(
                    "SAT-LazyCycles", "optimal", int(optimum), time.perf_counter() - t0,
                    selected, order,
                    extra={"backend": backend, "sat_calls": total_sat_calls,
                           "cycle_cuts": total_cycle_cuts, "dpll_nodes": total_dpll_nodes,
                           "budget": k, "action_vars": len(var_action)}
                )

            cyc_edges = nx.find_cycle(G, orientation="original")
            cycle_vertices = [int(e[1]) for e in cyc_edges]
            # For each cycle target i, forbid the exact selected action that supplies
            # its incoming cycle arc. At least one must change to break this cycle.
            clause = sorted({-selected_var_by_target[i] for i in cycle_vertices})
            if not clause:
                return SolveResult("SAT-LazyCycles", "internal-error-empty-cycle-cut", None,
                                   time.perf_counter() - t0)
            lazy_cycle_clauses.append(clause)
            cnf.append(clause)
            total_cycle_cuts += 1

    return SolveResult("SAT-LazyCycles", "infeasible", None, time.perf_counter() - t0,
                       extra={"backend": backend, "sat_calls": total_sat_calls,
                              "cycle_cuts": total_cycle_cuts, "dpll_nodes": total_dpll_nodes})


# ---------------------------------------------------------------------------
# Reference closure-state BFS for validation of small benchmark instances
# ---------------------------------------------------------------------------

def pick_closure(rule: RuleInstance, cleared: FrozenSet[int]) -> FrozenSet[int]:
    R = set(cleared)
    changed = True
    while changed:
        changed = False
        for i in range(rule.n):
            if i in R:
                continue
            for a in rule.actions[i]:
                if a.kind == "pick" and a.blockers.issubset(R):
                    R.add(i); changed = True; break
    return frozenset(R)


def solve_bfs_reference(rule: RuleInstance, max_n: int = 14) -> Optional[int]:
    if rule.n > max_n:
        return None
    from collections import deque
    start = pick_closure(rule, frozenset())
    if len(start) == rule.n:
        return 0
    q = deque([(start, 0)])
    seen = {start}
    while q:
        R, cost = q.popleft()
        for i in range(rule.n):
            if i in R:
                continue
            feasible_knock = any(
                a.kind == "knock" and a.blockers.issubset(R)
                for a in rule.actions[i]
            )
            if not feasible_knock:
                continue
            Rp = pick_closure(rule, frozenset(set(R) | {i}))
            if len(Rp) == rule.n:
                return cost + 1
            if Rp not in seen:
                seen.add(Rp); q.append((Rp, cost + 1))
    return None


# ---------------------------------------------------------------------------
# Benchmark generation
# ---------------------------------------------------------------------------

def canonical_instances() -> List[GeometryInstance]:
    out = []

    # 1. Continuous grasp-placement example: two contacts leave a usable window.
    out.append(GeometryInstance(
        "gap_grasp_5",
        [
            Rect("A", 0, 1, 0, 3),
            Rect("B", 1, 2, 0, 1),
            Rect("C", 1, 2, 2, 3),
            Rect("D", 2, 2.8, 0, 1),
            Rect("E", 2, 2.9, 2, 3),
        ],
        F=0.6, D=1.5,
        metadata={"family": "representative", "description": "long cuboid with separated side contacts"}
    ))

    # 2. Proper partial side overlaps: neither contacting side contains the other.
    out.append(GeometryInstance(
        "partial_contacts_3",
        [
            Rect("A", 0, 1, 0, 3),
            Rect("B", 1, 2, 1, 4),
            Rect("C", -1, 0, -1, 1.5),
        ],
        F=0.7, D=1.5,
        metadata={"family": "representative", "description": "non-nested positive-length side contacts"}
    ))

    # 3. Non-uniform full 2x3 tiling.
    xs = [0, 1.0, 2.4, 3.2]
    ys = [0, 0.9, 2.1]
    rects = []
    names = [["A", "B", "C"], ["D", "E", "F"]]
    for r in range(2):
        for c in range(3):
            rects.append(Rect(names[r][c], xs[c], xs[c+1], ys[r], ys[r+1]))
    out.append(GeometryInstance(
        "nonuniform_2x3_6", rects, F=0.4, D=1.6,
        metadata={"family": "representative", "description": "heterogeneous 2x3 packed tiling"}
    ))

    # 4. Non-uniform full 3x3 tiling.
    xs = [0, 0.9, 2.2, 3.2]
    ys = [0, 0.8, 1.9, 2.8]
    names = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]]
    rects = []
    for r in range(3):
        for c in range(3):
            rects.append(Rect(names[r][c], xs[c], xs[c+1], ys[r], ys[r+1]))
    out.append(GeometryInstance(
        "nonuniform_3x3_9", rects, F=0.4, D=1.5,
        metadata={"family": "representative", "description": "heterogeneous 3x3 packed tiling"}
    ))

    # 5. One long vertical spine contacted by many small cuboids on both sides.
    rects = [Rect("S", 1, 2, 0, 5)]
    for k in range(5):
        rects.append(Rect(f"L{k}", 0, 1, k, k + 1))
        rects.append(Rect(f"R{k}", 2, 3, k, k + 1))
    out.append(GeometryInstance(
        "long_spine_11", rects, F=0.6, D=1.5,
        metadata={"family": "representative", "description": "one long cuboid with ten smaller side contacts"}
    ))

    # 6. Two long horizontal rails and staggered middle blocks.
    rects = [
        Rect("Top", 0, 6, 2, 3),
        Rect("Bot", 0, 6, 0, 1),
        Rect("M0", 0, 1.2, 1, 2),
        Rect("M1", 1.2, 2.0, 1, 2),
        Rect("M2", 2.0, 3.5, 1, 2),
        Rect("M3", 3.5, 4.4, 1, 2),
        Rect("M4", 4.4, 6.0, 1, 2),
    ]
    out.append(GeometryInstance(
        "double_rail_7", rects, F=0.55, D=1.7,
        metadata={"family": "representative", "description": "two long cuboids enclosing heterogeneous short cuboids"}
    ))

    # 7. A sparse mixed cluster: packed 2x2 core plus isolated/easy objects.
    rects = [
        Rect("A", 0, 1.1, 0, 1), Rect("B", 1.1, 2.0, 0, 1),
        Rect("C", 0, 1.1, 1, 2.2), Rect("D", 1.1, 2.0, 1, 2.2),
        Rect("E", 3.0, 4.2, 0, 0.8), Rect("F", 5.0, 5.8, 1.2, 2.3),
    ]
    out.append(GeometryInstance(
        "mixed_sparse_6", rects, F=0.45, D=1.35,
        metadata={"family": "representative", "description": "locked core with immediately pickable isolated cuboids"}
    ))
    return out


def guillotine_partition(n: int, seed: int, W: float = 8.0, H: float = 6.0) -> List[Rect]:
    rng = random.Random(seed)
    cells: List[Tuple[float, float, float, float]] = [(0.0, W, 0.0, H)]
    while len(cells) < n:
        # Prefer splitting the largest-area cell, with randomization among top candidates.
        ranked = sorted(range(len(cells)), key=lambda q: (cells[q][1]-cells[q][0])*(cells[q][3]-cells[q][2]), reverse=True)
        idx = rng.choice(ranked[: min(3, len(ranked))])
        x0, x1, y0, y1 = cells.pop(idx)
        w, h = x1 - x0, y1 - y0
        # Prefer splitting along the longer dimension, but sometimes do the other.
        vertical = (w >= h)
        if rng.random() < 0.25:
            vertical = not vertical
        if vertical and w > 0.8:
            frac = rng.uniform(0.32, 0.68)
            xm = x0 + frac * w
            cells.extend([(x0, xm, y0, y1), (xm, x1, y0, y1)])
        elif h > 0.8:
            frac = rng.uniform(0.32, 0.68)
            ym = y0 + frac * h
            cells.extend([(x0, x1, y0, ym), (x0, x1, ym, y1)])
        else:
            # Fallback split along whichever has more room.
            if w >= h:
                xm = 0.5 * (x0 + x1)
                cells.extend([(x0, xm, y0, y1), (xm, x1, y0, y1)])
            else:
                ym = 0.5 * (y0 + y1)
                cells.extend([(x0, x1, y0, ym), (x0, x1, ym, y1)])
    cells = sorted(cells, key=lambda c: (c[2], c[0], c[3], c[1]))
    return [Rect(f"R{i}", *c) for i, c in enumerate(cells)]


def random_instances() -> List[GeometryInstance]:
    configs = [
        (8, 11, 7.0, 5.0),
        (9, 23, 9.0, 4.5),   # wide aspect ratio
        (10, 37, 6.0, 8.0),  # tall aspect ratio
        (11, 51, 8.0, 6.0),
        (12, 71, 10.0, 5.0),
        (13, 89, 7.0, 7.0),
        (15, 101, 10.0, 7.0),
    ]
    out = []
    for n, seed, W, H in configs:
        rects = guillotine_partition(n, seed, W, H)
        min_side = min(r.min_side for r in rects)
        max_min_side = max(r.min_side for r in rects)
        F = 0.55 * min_side
        D = 1.05 * max_min_side
        out.append(GeometryInstance(
            f"guillotine_n{n}_s{seed}", rects, F=F, D=D,
            metadata={"family": "random-guillotine", "seed": str(seed), "description": f"random dense guillotine tiling {W}x{H}"}
        ))
    return out


def extended_contact_instances() -> List[GeometryInstance]:
    """Hand-built proper partial-contact cases with varied blocker structure."""
    cases = [
        ("partial_offset_chain_5", [
            Rect("A", 0, 1.0, 0, 3.2), Rect("B", 1.0, 2.1, 0.6, 1.8),
            Rect("C", 1.0, 2.4, 2.1, 3.8), Rect("D", -1.2, 0, -0.4, 1.1),
            Rect("E", -0.9, 0, 1.7, 3.0)], 0.55, 1.55),
        ("partial_staggered_6", [
            Rect("A", 0, 1.2, 0, 2.8), Rect("B", 1.2, 2.1, -0.5, 0.9),
            Rect("C", 1.2, 2.5, 1.2, 2.2), Rect("D", 1.2, 2.0, 2.45, 3.5),
            Rect("E", -1.0, 0, 0.5, 1.55), Rect("F", -1.4, 0, 1.9, 3.1)],
         0.45, 1.45),
        ("partial_cross_7", [
            Rect("C", 0, 1.0, 0, 4.0), Rect("L0", -1.4, 0, -0.3, 1.0),
            Rect("L1", -1.0, 0, 1.35, 2.45), Rect("L2", -1.3, 0, 2.8, 4.4),
            Rect("R0", 1.0, 2.1, 0.45, 1.6), Rect("R1", 1.0, 2.5, 1.9, 2.8),
            Rect("R2", 1.0, 2.0, 3.15, 4.3)], 0.42, 1.55),
        ("partial_window_8", [
            Rect("A", 0, 1, 0, 4), Rect("B0", 1, 2.0, -0.2, 0.8),
            Rect("B1", 1, 2.2, 1.15, 2.0), Rect("B2", 1, 1.9, 2.45, 3.15),
            Rect("B3", 1, 2.3, 3.55, 4.4), Rect("L0", -1.1, 0, 0.5, 1.45),
            Rect("L1", -1.4, 0, 2.0, 2.8), Rect("L2", -0.9, 0, 3.1, 4.1)],
         0.50, 1.55),
    ]
    return [GeometryInstance(
        name, rects, F=F, D=D,
        metadata={"family": "partial-contact", "description": "proper partial side contacts"},
    ) for name, rects, F, D in cases]


def two_clump_instances() -> List[GeometryInstance]:
    """Separated packed components, including zero-, one-, and two-Knock cases."""
    out = []
    layouts = [
        ("two_clumps_2x2_8", 2, 2, 2, 2),
        ("two_clumps_2x3_10", 2, 2, 2, 3),
        ("two_clumps_3x3_13", 2, 2, 3, 3),
        ("two_clumps_2x3_pair_12", 2, 3, 2, 3),
    ]
    for name, rows_a, cols_a, rows_b, cols_b in layouts:
        rects = []
        widths = (0.85, 1.10, 0.95)
        heights = (0.90, 1.15, 0.80)
        for prefix, rows, cols, xshift in (
            ("A", rows_a, cols_a, 0.0), ("B", rows_b, cols_b, 5.2)
        ):
            xs = [xshift]
            ys = [0.0]
            for c in range(cols): xs.append(xs[-1] + widths[c % len(widths)])
            for r in range(rows): ys.append(ys[-1] + heights[r % len(heights)])
            for r in range(rows):
                for c in range(cols):
                    rects.append(Rect(f"{prefix}{r}{c}", xs[c], xs[c + 1], ys[r], ys[r + 1]))
        out.append(GeometryInstance(
            name, rects, F=0.38, D=1.25,
            metadata={"family": "two-clump", "description": "two independent packed components"},
        ))
    return out


def all_benchmarks() -> List[GeometryInstance]:
    return (canonical_instances() + extended_contact_instances()
            + two_clump_instances() + random_instances())


# ---------------------------------------------------------------------------
# Plotting and reporting
# ---------------------------------------------------------------------------

def _plot_bounds(inst: GeometryInstance) -> Tuple[float, float, float, float, float, float]:
    xmin = min(r.xmin for r in inst.rects); xmax = max(r.xmax for r in inst.rects)
    ymin = min(r.ymin for r in inst.rects); ymax = max(r.ymax for r in inst.rects)
    padx = max(0.2, 0.04 * max(EPS, xmax - xmin))
    pady = max(0.2, 0.04 * max(EPS, ymax - ymin))
    return xmin, xmax, ymin, ymax, padx, pady


def _dir_vec(direction: str) -> Tuple[float, float]:
    return {
        "+x": (1.0, 0.0),
        "-x": (-1.0, 0.0),
        "+y": (0.0, 1.0),
        "-y": (0.0, -1.0),
    }[direction]


def _configure_axes(ax, inst: GeometryInstance, title: Optional[str] = None) -> None:
    xmin, xmax, ymin, ymax, padx, pady = _plot_bounds(inst)
    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)


def _draw_layout(ax, inst: GeometryInstance, present: Optional[Sequence[bool]] = None,
                 highlight_idx: Optional[int] = None,
                 step_map: Optional[Dict[int, int]] = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    n = len(inst.rects)
    if present is None:
        present = [True] * n
    cmap = plt.get_cmap("viridis", max(2, n))

    for i, r in enumerate(inst.rects):
        is_present = bool(present[i])
        fill = None
        alpha = 1.0
        edgecolor = "black"
        linewidth = 1.5
        linestyle = "-"
        if step_map is not None and i in step_map:
            fill = cmap(step_map[i] - 1)
            alpha = 0.18 if is_present else 0.10
        if not is_present:
            edgecolor = "0.65"
            linewidth = 1.0
            linestyle = "--"
        if highlight_idx == i:
            fill = "gold"
            alpha = 0.35 if is_present else 0.18
            edgecolor = "black"
            linewidth = 2.2
        patch = Rectangle((r.xmin, r.ymin), r.w, r.h, fill=fill is not None,
                          facecolor=fill if fill is not None else None,
                          edgecolor=edgecolor, linewidth=linewidth,
                          linestyle=linestyle, alpha=alpha if fill is not None else 1.0)
        ax.add_patch(patch)
        ax.text(0.5 * (r.xmin + r.xmax), 0.5 * (r.ymin + r.ymax), r.name,
                ha="center", va="center", fontsize=8,
                color="black" if is_present else "0.45")


def _annotate_step_action(ax, rect: Rect, action: Action, step: int, dim_scale: float,
                          emphasize: bool = False) -> None:
    cx = 0.5 * (rect.xmin + rect.xmax)
    cy = 0.5 * (rect.ymin + rect.ymax)
    badge_x = rect.xmin + 0.18 * rect.w
    badge_y = rect.ymax - 0.18 * rect.h
    label = f"{step}{'P' if action.kind == 'pick' else 'K'}"
    ax.text(badge_x, badge_y, label, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="circle,pad=0.23",
                      fc=(0.88, 1.0, 0.88) if action.kind == 'pick' else (1.0, 0.9, 0.9),
                      ec="black", lw=1.0))
    if action.kind == "pick":
        span = 0.32 * rect.w if action.orientation == "x" else 0.32 * rect.h
        if action.orientation == "x":
            ax.annotate("", xy=(cx + span, cy), xytext=(cx - span, cy),
                        arrowprops=dict(arrowstyle="<->", lw=1.6))
            ax.text(cx, cy + 0.12 * rect.h + 0.05 * dim_scale,
                    f"P({action.orientation})", ha="center", va="bottom", fontsize=7)
        else:
            ax.annotate("", xy=(cx, cy + span), xytext=(cx, cy - span),
                        arrowprops=dict(arrowstyle="<->", lw=1.6))
            ax.text(cx + 0.12 * rect.w + 0.05 * dim_scale, cy,
                    f"P({action.orientation})", ha="left", va="center", fontsize=7)
    else:
        dx, dy = _dir_vec(action.direction or "+x")
        length = 0.52 * max(rect.w, rect.h) + 0.18 * dim_scale
        sx = cx + 0.12 * rect.w * dx
        sy = cy + 0.12 * rect.h * dy
        ex = sx + length * dx
        ey = sy + length * dy
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="->", lw=2.0))
        tx = ex + 0.05 * dim_scale * dx
        ty = ey + 0.05 * dim_scale * dy
        ax.text(tx, ty, f"K({action.direction})", fontsize=7,
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top")


def plot_instance(inst: GeometryInstance, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    _draw_layout(ax, inst)
    _configure_axes(ax, inst, f"{inst.name}  (n={len(inst.rects)}, F={inst.F:.3g}, D={inst.D:.3g})")
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_solution_overlay(rule: RuleInstance, selected: Dict[int, Action], order: Sequence[int], path: Path,
                          solver_name: str = "") -> None:
    import matplotlib.pyplot as plt
    inst = rule.geometry
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    step_map = {obj: step for step, obj in enumerate(order, start=1)}
    _draw_layout(ax, inst, present=[True] * rule.n, step_map=step_map)
    _, _, _, _, padx, pady = _plot_bounds(inst)
    dim_scale = max(padx, pady)
    for step, obj in enumerate(order, start=1):
        _annotate_step_action(ax, inst.rects[obj], selected[obj], step, dim_scale)
    title = f"{inst.name} — {solver_name} solution overlay" if solver_name else f"{inst.name} — solution overlay"
    _configure_axes(ax, inst, title)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_solution_sequence(rule: RuleInstance, selected: Dict[int, Action], order: Sequence[int], path: Path,
                           solver_name: str = "") -> None:
    import matplotlib.pyplot as plt
    inst = rule.geometry
    n = len(order)
    cols = min(4, max(1, math.ceil(math.sqrt(n))))
    rows = math.ceil(n / cols)
    fig_w = 4.3 * cols
    fig_h = 3.8 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(np.asarray(axes).reshape(-1))

    removed: set[int] = set()
    step_map = {obj: step for step, obj in enumerate(order, start=1)}
    _, _, _, _, padx, pady = _plot_bounds(inst)
    dim_scale = max(padx, pady)
    for step, obj in enumerate(order, start=1):
        ax = axes_list[step - 1]
        present = [i not in removed for i in range(rule.n)]
        _draw_layout(ax, inst, present=present, highlight_idx=obj, step_map=step_map)
        _annotate_step_action(ax, inst.rects[obj], selected[obj], step, dim_scale, emphasize=True)
        a = selected[obj]
        desc = f"Pick {rule.names[obj]} ({a.orientation})" if a.kind == 'pick' else f"Knock {rule.names[obj]} {a.direction}"
        _configure_axes(ax, inst, f"Step {step}: {desc}")
        removed.add(obj)

    for ax in axes_list[n:]:
        ax.axis("off")
    fig.suptitle(f"{inst.name} — {solver_name} execution sequence" if solver_name else f"{inst.name} — execution sequence",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_rule_instance(rule: RuleInstance) -> Dict[str, object]:
    picks = sum(sum(1 for a in acts if a.kind == "pick") for acts in rule.actions.values())
    knocks = sum(sum(1 for a in acts if a.kind == "knock") for acts in rule.actions.values())
    pick_tail = sum(len(a.blockers) for acts in rule.actions.values() for a in acts if a.kind == "pick")
    knock_tail = sum(len(a.blockers) for acts in rule.actions.values() for a in acts if a.kind == "knock")
    max_pick_rank = max([0] + [len(a.blockers) for acts in rule.actions.values() for a in acts if a.kind == "pick"])
    max_knock_rank = max([0] + [len(a.blockers) for acts in rule.actions.values() for a in acts if a.kind == "knock"])
    return {
        "n": rule.n, "pick_rules": picks, "knock_rules": knocks,
        "M_P": pick_tail, "M_K": knock_tail,
        "max_pick_rank": max_pick_rank, "max_knock_rank": max_knock_rank,
    }


def isaac_scale(inst: GeometryInstance, workspace_size: float = 0.25,
                workspace_margin: float = 0.02, max_jaw_width: float = 0.075) -> float:
    """Uniformly fit geometry in the workspace and within Panda jaw opening."""
    span_x = max(r.xmax for r in inst.rects) - min(r.xmin for r in inst.rects)
    span_y = max(r.ymax for r in inst.rects) - min(r.ymin for r in inst.rects)
    usable = workspace_size - workspace_margin
    if usable <= 0:
        raise ValueError("workspace margin must be smaller than workspace size")
    return min(usable / max(span_x, span_y), max_jaw_width / inst.D)


def _pick_grasp_payload(inst: GeometryInstance, rect: Rect, action: Action,
                        scale: float) -> Dict[str, object]:
    if action.orientation == "x":
        offset = [0.0, (float(action.placement) + 0.5 * inst.F
                        - 0.5 * (rect.ymin + rect.ymax)) * scale]
    else:
        offset = [(float(action.placement) + 0.5 * inst.F
                   - 0.5 * (rect.xmin + rect.xmax)) * scale, 0.0]
    return {
        "closing_axis": action.orientation,
        "offset": [round(v, 8) for v in offset],
        "placement_witness": round(float(action.placement) * scale, 8),
    }


def write_isaac_problem(rule: RuleInstance, result: SolveResult, path: Path,
                        contact_gap: float = 0.001, common_height: float = 0.05,
                        max_clearance_attempts: int = 12) -> Dict[str, object]:
    """Serialize an exact cuboid plan for the native Isaac executor.

    Planning contacts remain exact. Physical rectangles are shrunk symmetrically
    by ``contact_gap`` so nominally contacting bodies have a small robustness gap.
    """
    if result.status != "optimal" or result.selected_actions is None or result.order is None:
        raise ValueError("an optimal solved plan is required for Isaac export")
    inst = rule.geometry
    scale = isaac_scale(inst)
    xmin = min(r.xmin for r in inst.rects); xmax = max(r.xmax for r in inst.rects)
    ymin = min(r.ymin for r in inst.rects); ymax = max(r.ymax for r in inst.rects)
    source_origin = np.array([0.45, 0.0])
    plan_mid = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
    pads: Dict[int, Dict[str, float]] = {}
    objects: Dict[str, Dict[str, object]] = {}
    initial = []
    goal = []
    initial_pad_by_index = {}
    destination_pad_by_index = {}

    for i, rect in enumerate(inst.rects):
        center = source_origin + scale * (
            np.array([0.5 * (rect.xmin + rect.xmax), 0.5 * (rect.ymin + rect.ymax)])
            - plan_mid
        )
        pad = i + 1
        initial_pad_by_index[i] = pad
        pads[pad] = {"x": round(float(center[0]), 8), "y": round(float(center[1]), 8)}
        physical_w = rect.w * scale - contact_gap
        physical_h = rect.h * scale - contact_gap
        if min(physical_w, physical_h) <= 0.010:
            raise ValueError(
                f"{inst.name}/{rect.name}: scaled side too small for reliable execution "
                f"({physical_w:.4f} x {physical_h:.4f} m)"
            )
        objects[rect.name] = {
            # The Rz(+90 deg) ROS-plan -> Isaac transform swaps footprint axes.
            "dimensions": [round(physical_h, 8), round(physical_w, 8), common_height],
            "plan_center": [round(float(center[0]), 8), round(float(center[1]), 8)],
            "planning_dimensions": [round(rect.w * scale, 8), round(rect.h * scale, 8)],
        }
        initial.append({"name": rect.name, "pad": pad, "height": 0})

        destination_pad = len(inst.rects) + 1 + i
        destination_pad_by_index[i] = destination_pad
        # A bank of disposal locations beyond the packed 25 cm scene.
        pads[destination_pad] = {
            "x": round(0.68 + 0.025 * (i % 3), 8),
            "y": round(-0.18 + 0.045 * ((i // 3) % 9), 8),
        }
        goal.append({"name": rect.name, "pad": destination_pad, "height": 0})

    actions = []
    for i in result.order:
        selected = result.selected_actions[i]
        rect = inst.rects[i]
        if selected.kind == "knock":
            actions.append({
                "type": "Clearance", "blocks": [rect.name],
                "dir": selected.direction,
                "required_displacement": round(inst.F * scale + 0.005, 8),
                "solver_witness": selected.witness,
            })
            move = {
                "type": "Move", "block": rect.name,
                "to": {"pad": destination_pad_by_index[i], "height": 0},
                "after_clearance": True,
                "remove_after_place": True,
            }
        else:
            move = {
                "type": "Move", "block": rect.name,
                "to": {"pad": destination_pad_by_index[i], "height": 0},
                "grasp": _pick_grasp_payload(inst, rect, selected, scale),
                "solver_witness": selected.witness,
                "remove_after_place": True,
            }
        actions.append(move)

    document = {
        "schema_version": 2,
        "planner": "heterogeneous-cuboid-exact",
        "instance": inst.name,
        "solver": result.solver,
        "planning_time_sec": result.runtime_sec,
        "optimal_clearances": result.optimum,
        "scale": scale,
        "finger_contact_length": inst.F * scale,
        "jaw_opening": inst.D * scale,
        "contact_gap": contact_gap,
        "execution": {
            "clearance_margin": 0.005,
            "panda_finger_clearance": 0.022,
            "max_clearance_attempts": max(10, max_clearance_attempts),
        },
        "pads": pads, "objects": objects, "initial": initial,
        "goal": goal, "actions": actions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return document


def run_benchmarks(output_dir: Path, sat_time_limit: Optional[float] = 60.0,
                   ilp_time_limit: Optional[float] = 60.0, make_plots: bool = True,
                   export_isaac: bool = True) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        (output_dir / "layouts").mkdir(exist_ok=True)
        (output_dir / "solution_overlays").mkdir(exist_ok=True)
        (output_dir / "solution_sequences").mkdir(exist_ok=True)
    if export_isaac:
        (output_dir / "isaac_problems").mkdir(exist_ok=True)

    rows: List[Dict[str, object]] = []
    for inst in all_benchmarks():
        print(f"[benchmark] {inst.name}")
        preprocess_started = time.perf_counter()
        rule = preprocess(inst, prune_dominated=True)
        preprocess_sec = time.perf_counter() - preprocess_started
        stats = summarize_rule_instance(rule)
        if make_plots:
            plot_instance(inst, output_dir / "layouts" / f"{inst.name}.png")

        ilp_res = solve_ilp(rule, ilp_time_limit)
        sat_res = solve_sat(rule, sat_time_limit)
        if make_plots:
            for res in (ilp_res, sat_res):
                if res.selected_actions is not None and res.order is not None and res.status == "optimal":
                    tag = res.solver.lower().replace("/", "_").replace(" ", "_").replace("-", "_")
                    plot_solution_overlay(rule, res.selected_actions, res.order,
                                          output_dir / "solution_overlays" / f"{inst.name}_{tag}.png",
                                          res.solver)
                    plot_solution_sequence(rule, res.selected_actions, res.order,
                                           output_dir / "solution_sequences" / f"{inst.name}_{tag}.png",
                                           res.solver)
        bfs_opt = solve_bfs_reference(rule, max_n=13)
        isaac_export = "disabled"
        if export_isaac and ilp_res.status == "optimal":
            try:
                write_isaac_problem(
                    rule, ilp_res,
                    output_dir / "isaac_problems" / f"{inst.name}.yaml",
                )
                isaac_export = "ok"
            except ValueError as exc:
                isaac_export = f"unsuitable:{exc}"

        if ilp_res.optimum is not None and sat_res.optimum is not None and ilp_res.optimum != sat_res.optimum:
            raise AssertionError(f"Solver disagreement on {inst.name}: ILP={ilp_res.optimum}, SAT={sat_res.optimum}")
        if bfs_opt is not None and ilp_res.optimum is not None and bfs_opt != ilp_res.optimum:
            raise AssertionError(f"BFS disagreement on {inst.name}: BFS={bfs_opt}, ILP={ilp_res.optimum}")

        row: Dict[str, object] = {
            "instance": inst.name,
            "family": inst.metadata.get("family", ""),
            "n": rule.n,
            "F": inst.F,
            "D": inst.D,
            "preprocess_sec": preprocess_sec,
            **stats,
            "ILP_status": ilp_res.status,
            "ILP_opt": ilp_res.optimum,
            "ILP_sec": ilp_res.runtime_sec,
            "SAT_status": sat_res.status,
            "SAT_opt": sat_res.optimum,
            "SAT_sec": sat_res.runtime_sec,
            "SAT_backend": sat_res.extra.get("backend"),
            "SAT_calls": sat_res.extra.get("sat_calls"),
            "SAT_cycle_cuts": sat_res.extra.get("cycle_cuts"),
            "SAT_dpll_nodes": sat_res.extra.get("dpll_nodes"),
            "BFS_opt_small": bfs_opt,
            "physical_actions": rule.n + (ilp_res.optimum or 0),
            "isaac_export": isaac_export,
        }
        rows.append(row)

        # Save one readable plan per solver (they can differ while sharing optimum).
        plan_path = output_dir / f"{inst.name}_plans.txt"
        with plan_path.open("w", encoding="utf-8") as f:
            f.write(f"Instance: {inst.name}\n\n")
            for res in (ilp_res, sat_res):
                f.write(f"[{res.solver}] status={res.status}, optimum={res.optimum}, runtime={res.runtime_sec:.6f}s\n")
                if res.selected_actions is not None and res.order is not None:
                    ok, msg = verify_solution(rule, res.selected_actions, res.order)
                    f.write(f"verification={ok} ({msg})\n")
                    for step, text in enumerate(physical_plan(rule, res.selected_actions, res.order), start=1):
                        f.write(f"  {step:02d}. {text}\n")
                f.write("\n")

    # CSV
    csv_path = output_dir / "solver_results.csv"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

    # Markdown summary
    md_path = output_dir / "solver_results.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Cuboid Knock-Pick Solver Benchmark Summary\n\n")
        f.write("Both exact methods solve the same preprocessed geometric rule systems. ")
        f.write("The SAT backend used in this run is reported explicitly; when `python-sat` is unavailable the script uses its bundled DPLL fallback.\n\n")
        f.write("| Instance | n | Pick rules | Knock rules | max B | max K | OPT | ILP (s) | SAT (s) | SAT cuts | BFS check |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            opt = r["ILP_opt"] if r["ILP_opt"] is not None else r["SAT_opt"]
            f.write(
                f"| {r['instance']} | {r['n']} | {r['pick_rules']} | {r['knock_rules']} | "
                f"{r['max_pick_rank']} | {r['max_knock_rank']} | {opt} | "
                f"{float(r['ILP_sec']):.4f} | {float(r['SAT_sec']):.4f} | "
                f"{r['SAT_cycle_cuts']} | {r['BFS_opt_small']} |\n"
            )
        f.write("\n## Notes\n\n")
        f.write("- `OPT` is the minimum number of directional Knocks; the mandatory Pick/Place of each object contributes the fixed additive term n to physical action count.\n")
        f.write("- Dominated action rules are removed before optimization: a rule is dominated when another rule for the same object has no greater cost and a subset of blockers.\n")
        f.write("- The small-instance closure-state BFS is used only as an independent verification oracle; it is not one of the two proposed solvers.\n")
        f.write(f"- SAT backend in this environment: {'PySAT/Glucose4' if HAVE_PYSAT else 'bundled SimpleDPLL fallback'}.\n")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cuboid_kp_results")
    parser.add_argument("--sat-time-limit", type=float, default=60.0)
    parser.add_argument("--ilp-time-limit", type=float, default=60.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-isaac-export", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    rows = run_benchmarks(
        out, args.sat_time_limit, args.ilp_time_limit,
        not args.no_plots, not args.no_isaac_export,
    )
    print(f"\nWrote {len(rows)} benchmark results to {out.resolve()}")
    for r in rows:
        print(f"{r['instance']:<24} n={r['n']:>2} OPT={r['ILP_opt']} "
              f"ILP={float(r['ILP_sec']):.4f}s SAT={float(r['SAT_sec']):.4f}s")


if __name__ == "__main__":
    main()
