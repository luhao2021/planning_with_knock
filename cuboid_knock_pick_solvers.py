#!/usr/bin/env python3
"""Exact solvers and benchmark generator for heterogeneous cuboid knock-pick planning.

The implementation follows the geometric model in the companion manuscript:
  * equal-height, arbitrarily yawed cuboids represented by oriented rectangles;
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
import json
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
    yaw: float = 0.0              # radians in the common planning frame

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

    @property
    def center(self) -> np.ndarray:
        return np.array([0.5 * (self.xmin + self.xmax),
                         0.5 * (self.ymin + self.ymax)], dtype=float)

    @property
    def corners(self) -> np.ndarray:
        local = np.array([[-self.w/2, -self.h/2], [self.w/2, -self.h/2],
                          [self.w/2, self.h/2], [-self.w/2, self.h/2]])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return local @ np.array([[c, s], [-s, c]]) + self.center


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
    finger_clearance: float = 0.0
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


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _clip_halfplane(poly: np.ndarray, axis: int, bound: float,
                    keep_less: bool) -> np.ndarray:
    if len(poly) == 0:
        return poly
    out = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        fp = (p[axis] - bound) * (1 if keep_less else -1)
        fq = (q[axis] - bound) * (1 if keep_less else -1)
        pin, qin = fp <= EPS, fq <= EPS
        if pin:
            out.append(p)
        if pin != qin:
            den = q[axis] - p[axis]
            if abs(den) > EPS:
                out.append(p + (bound - p[axis]) / den * (q - p))
    return np.asarray(out, dtype=float).reshape((-1, 2))


def _clip_box(poly: np.ndarray, xmin: float, xmax: float,
              ymin: float, ymax: float) -> np.ndarray:
    for axis, bound, less in ((0, xmin, False), (0, xmax, True),
                              (1, ymin, False), (1, ymax, True)):
        poly = _clip_halfplane(poly, axis, bound, less)
    return poly


def _polygon_area(poly: np.ndarray) -> float:
    return 0.0 if len(poly) < 3 else abs(sum(_cross(p, q) for p, q in
        zip(poly, np.roll(poly, -1, axis=0)))) * 0.5


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)
    def build(seq):
        out = []
        for p in seq:
            p = np.asarray(p)
            while len(out) >= 2 and _cross(out[-1]-out[-2], p-out[-1]) <= EPS:
                out.pop()
            out.append(p)
        return out
    return np.asarray(build(pts)[:-1] + build(reversed(pts))[:-1])


def _convex_intersection(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    out = np.asarray(subject, dtype=float)
    for a, b in zip(clipper, np.roll(clipper, -1, axis=0)):
        if len(out) == 0:
            break
        nxt = []
        for p, q in zip(out, np.roll(out, -1, axis=0)):
            fp, fq = _cross(b-a, p-a), _cross(b-a, q-a)
            pin, qin = fp >= -EPS, fq >= -EPS
            if pin: nxt.append(p)
            if pin != qin:
                den = fp - fq
                if abs(den) > EPS: nxt.append(p + fp / den * (q-p))
        out = np.asarray(nxt, dtype=float).reshape((-1, 2))
    return out


def _to_local(points: np.ndarray, rect: Rect) -> np.ndarray:
    c, s = math.cos(rect.yaw), math.sin(rect.yaw)
    return (np.asarray(points) - rect.center) @ np.array([[c, -s], [s, c]])


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
            if _polygon_area(_convex_intersection(r.corners, s.corners)) > EPS:
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
        local = _to_local(rj.corners, ri)
        eta = max(inst.finger_clearance, 1e-8)
        if orientation == "x":
            boxes = [(-ri.w/2-eta, -ri.w/2+1e-8, -ri.h/2, ri.h/2),
                     (ri.w/2-1e-8, ri.w/2+eta, -ri.h/2, ri.h/2)]
            tangential = 1
        elif orientation == "y":
            boxes = [(-ri.w/2, ri.w/2, -ri.h/2-eta, -ri.h/2+1e-8),
                     (-ri.w/2, ri.w/2, ri.h/2-1e-8, ri.h/2+eta)]
            tangential = 0
        else:
            raise ValueError(orientation)
        for box in boxes:
            clipped = _clip_box(local, *box)
            if len(clipped):
                a, b = float(clipped[:, tangential].min()), float(clipped[:, tangential].max())
                if b-a > EPS: ivals.append((a, b))
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
            lo, hi = -ri.h/2, ri.h/2 - inst.F
        else:
            lo, hi = -ri.w/2, ri.w/2 - inst.F
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
    # Direction labels are object-local. Rotate the selected local axis into
    # the common planning frame before constructing the swept footprint.
    local_direction = np.asarray(_dir_vec(d), dtype=float)
    c, s = math.cos(ri.yaw), math.sin(ri.yaw)
    direction = np.array([[c, -s], [s, c]]) @ local_direction
    support = max(float(p @ direction) for r in inst.rects for p in r.corners)
    delta = support - min(float(p @ direction) for p in ri.corners) + 1e-6
    sweep = _convex_hull(np.vstack((ri.corners, ri.corners + delta * direction)))
    out = set()
    for j, rj in enumerate(inst.rects):
        if j == i:
            continue
        if _polygon_area(_convex_intersection(rj.corners, sweep)) > EPS:
            out.add(j)
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
    # Compact guillotine partitions retained specifically for physical Isaac
    # coverage under the 25 cm workspace and Panda jaw constraints.
    for n, seed in ((6, 2), (7, 6), (8, 5)):
        rects = guillotine_partition(n, seed, 4.0, 3.0)
        out.append(GeometryInstance(
            f"guillotine_compact_n{n}_s{seed}", rects, F=0.4, D=1.4,
            metadata={"family": "random-guillotine", "seed": str(seed),
                      "description": "compact random dense guillotine tiling 4x3"},
        ))
    return out


def axis_aligned_experiment_instances() -> List[GeometryInstance]:
    """The 20-case primary axis-aligned experiment set.

    It retains the requested canonical/contact cases and adds deterministic
    random compact guillotine layouts.  Seeds are part of the instance names,
    making the supposedly random dataset exactly reproducible.
    """
    canonical = {instance.name: instance for instance in canonical_instances()}
    partial = {instance.name: instance for instance in extended_contact_instances()}
    retained_names = (
        "double_rail_7", "gap_grasp_5", "long_spine_11",
        "partial_contacts_3", "partial_offset_chain_5",
        "partial_staggered_6", "partial_window_8",
    )
    retained = [canonical.get(name, partial.get(name)) for name in retained_names]
    retained.extend(instance for instance in random_instances()
                    if instance.name.startswith("guillotine_compact_"))

    # Additional dense partitions are selected from this deterministic seed
    # stream by the physical-exportability/solver validation step below.
    generated = []
    for index, (n, seed, width, height) in enumerate((
        (6, 709, 4.0, 3.0), (6, 712, 4.0, 3.0),
        (7, 804, 4.0, 3.0), (7, 824, 4.0, 3.0),
        (8, 926, 4.0, 3.0), (8, 121, 4.0, 3.0),
        (9, 100, 4.0, 3.0), (9, 152, 4.0, 3.0),
        (10, 162, 4.0, 3.0), (10, 113, 4.0, 3.0),
    ), 1):
        rects = guillotine_partition(n, seed, width, height)
        min_side = min(rect.min_side for rect in rects)
        max_min_side = max(rect.min_side for rect in rects)
        generated.append(GeometryInstance(
            f"guillotine_dense_{index:02d}_n{n}_s{seed}", rects,
            F=0.55 * min_side, D=1.05 * max_min_side,
            metadata={"family": "random-guillotine", "seed": str(seed),
                      "description": "dense compact random guillotine partition"},
        ))

    result = [instance for instance in retained if instance is not None] + generated
    if len(result) != 20:
        raise AssertionError(f"expected 20 axis-aligned cases, generated {len(result)}")
    for instance in result:
        instance.metadata["experiment_category"] = "axis-aligned"
    return result


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
        # partial_window_8 deliberately contains narrow geometric windows.
        # Reserve a physical normal corridor (~24 mm after scaling) so its
        # zero-clearance theoretical Pick is not exported as a fragile grasp
        # whose open finger can sweep a neighboring cuboid.
        finger_clearance=0.50 if name == "partial_window_8" else 0.0,
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
    # Compact variants keep the two components independent while avoiding the
    # large empty inter-clump span that made every original variant physically
    # unscalable in a 25 cm workspace.
    compact_layouts = [
        ("two_clumps_compact_2x2_8", 2, 2, 2, 2),
        ("two_clumps_compact_2x2_3x2_10", 2, 2, 3, 2),
        ("two_clumps_compact_3x2_2x2_10", 3, 2, 2, 2),
    ]
    for name, rows_a, cols_a, rows_b, cols_b in compact_layouts:
        rects = []
        for prefix, rows, cols, xshift in (
            ("A", rows_a, cols_a, 0.0),
            ("B", rows_b, cols_b, 2.3),
        ):
            rects.extend(Rect(f"{prefix}{r}{c}", xshift+c, xshift+c+1,
                              r, r+1)
                         for r in range(rows) for c in range(cols))
        out.append(GeometryInstance(
            name, rects, F=0.4, D=1.2,
            metadata={"family": "two-clump",
                      "description": "compact independent packed components"},
        ))
    return out


def all_benchmarks() -> List[GeometryInstance]:
    primary = primary_experiment_instances()
    primary_names = {instance.name for instance in primary}
    legacy = (canonical_instances() + extended_contact_instances()
              + two_clump_instances() + random_instances() + yaw_instances())
    others = []
    for instance in legacy:
        if instance.name in primary_names:
            continue
        instance.metadata["experiment_category"] = "others"
        others.append(instance)
    return primary + others


def _rect_at(name: str, cx: float, cy: float, w: float, h: float,
             yaw_deg: float = 0.0) -> Rect:
    return Rect(name, cx-w/2, cx+w/2, cy-h/2, cy+h/2,
                math.radians(yaw_deg))


def _rotate_clump(rects: Sequence[Rect], yaw_deg: float,
                  origin: Optional[Tuple[float, float]] = None,
                  prefix: str = "") -> List[Rect]:
    centers = np.asarray([r.center for r in rects])
    pivot = centers.mean(axis=0) if origin is None else np.asarray(origin, dtype=float)
    angle = math.radians(yaw_deg); c, s = math.cos(angle), math.sin(angle)
    Q = np.array([[c, -s], [s, c]])
    out = []
    for r in rects:
        center = pivot + Q @ (r.center-pivot)
        out.append(_rect_at(prefix+r.name, *center, r.w, r.h,
                            math.degrees(r.yaw)+yaw_deg))
    return out


def yaw_instances() -> List[GeometryInstance]:
    """Twelve deterministic arbitrary-yaw cases covering the paper's 3 classes."""
    out: List[GeometryInstance] = []
    # One contact-connected common-yaw clump. Exact contacts are preserved by
    # rigidly rotating dense heterogeneous tilings.
    bases = [(2, 3, 17.0), (3, 3, -24.0), (2, 4, 33.0)]
    for k, (rows, cols, yaw) in enumerate(bases, 1):
        xs = np.cumsum([0.0] + [0.9 + 0.15*(j % 3) for j in range(cols)])
        ys = np.cumsum([0.0] + [0.8 + 0.18*(j % 2) for j in range(rows)])
        raw = [Rect(f"C{r}{c}", xs[c], xs[c+1], ys[r], ys[r+1])
               for r in range(rows) for c in range(cols)]
        rects = _rotate_clump(raw, yaw)
        out.append(GeometryInstance(
            f"yaw_single_clump_{k}_{len(rects)}", rects, F=0.38, D=1.35,
            metadata={"family": "yaw-single-clump", "yaw_type": "common"}))

    # Multiple separated clumps; each clump has a distinct common yaw.
    multi_specs = [((18, -27), (2, 2)), ((31, -16), (2, 3)),
                   ((12, -34, 43), (2, 2, 2))]
    for k, (angles, cols_per) in enumerate(multi_specs, 1):
        rects = []
        for q, (yaw, cols) in enumerate(zip(angles, cols_per)):
            raw = [Rect(f"M{q}_{r}{c}", c, c+1.0, 0.9*r, 0.9*(r+1))
                   for r in range(2) for c in range(cols)]
            rotated = _rotate_clump(raw, yaw)
            shift = np.array([q*5.0, (q % 2)*0.5])
            for r in rotated:
                cc = r.center + shift
                rects.append(_rect_at(r.name, *cc, r.w, r.h,
                                      math.degrees(r.yaw)))
        out.append(GeometryInstance(
            f"yaw_multi_clump_{k}_{len(rects)}", rects, F=0.38, D=1.25,
            metadata={"family": "yaw-multi-clump", "yaw_type": "per-clump"}))

    # Physically scalable two-clump yaw cases. Each component is a rigidly
    # rotated contacted 2x2 clump; the compact separation preserves distinct
    # components without consuming most of the robot workspace as empty span.
    for k, (yaw_a, yaw_b, separation) in enumerate(
            ((12.0, -12.0, 2.8),
             (18.0, -10.0, 2.9),
             (8.0, 22.0, 2.9)), 1):
        raw = [Rect(f"C{r}{c}", c, c+1.0, r, r+1.0)
               for r in range(2) for c in range(2)]
        first = _rotate_clump(raw, yaw_a, prefix="A")
        second = _rotate_clump(raw, yaw_b, prefix="B")
        rects = list(first)
        for rect in second:
            center = rect.center + np.array([separation, 0.0])
            rects.append(_rect_at(rect.name, *center, rect.w, rect.h,
                                  math.degrees(rect.yaw)))
        out.append(GeometryInstance(
            f"yaw_multi_clump_compact_{k}_8", rects, F=0.5, D=1.4,
            metadata={"family": "yaw-multi-clump",
                      "yaw_type": "per-clump",
                      "description": "compact two-clump common-yaw layout"},
        ))

    # Independently oriented non-overlapping rectangles. Small gaps are less
    # than eta_F and therefore create the finite-clearance blocker relations
    # approved for this class, without pretending they are exact contacts.
    independent = [
        [(0,0,17),(1.15,0.05,-11),(2.32,-0.03,29),(0.2,1.18,51),(1.4,1.22,-37),(2.6,1.15,8)],
        [(0,0,-22),(1.25,0.0,14),(2.5,0.12,41),(0.05,1.3,63),(1.35,1.25,-46),(2.65,1.35,25),(1.3,2.5,5)],
        [(0,0,9),(1.3,0.1,-31),(2.6,0,54),(3.9,0.1,-12),(0.2,1.35,38),(1.5,1.4,-57),(2.8,1.32,21),(4.0,1.4,67)],
        [(0,0,-8),(1.2,0.15,36),(2.45,0.0,-49),(0.1,1.25,19),(1.35,1.4,58),(2.6,1.25,-27),(0.2,2.55,71),(1.5,2.6,-15),(2.75,2.5,43)],
    ]
    for k, spec in enumerate(independent, 1):
        rects = [_rect_at(f"I{i}", x, y, 0.82+0.05*(i%2),
                          0.62+0.06*(i%3), yaw)
                 for i, (x, y, yaw) in enumerate(spec)]
        out.append(GeometryInstance(
            f"yaw_independent_{k}_{len(rects)}", rects, F=0.30, D=0.95,
            finger_clearance=0.35,
            metadata={"family": "yaw-independent", "yaw_type": "individual",
                      "clearance_model": "eta_F=0.35"}))

    # Dense, independently yawed finite-clearance cycles. Every retained Pick
    # option initially has a blocker, while a boundary workspace-direction
    # Knock breaks the cycle. These explicitly prevent the independent-yaw
    # benchmark family from containing only zero-Knock examples.
    locked_specs = [
        ([(0, 0), (1.2, 0), (0, 1.2), (1.2, 1.2)],
         [-5, 3, 8, -2], 0.50),
        ([(0, 0), (1.15, 0), (2.3, 0), (0.58, 1.15), (1.73, 1.15)],
         [-6, 4, 11, -3, 7], 0.48),
    ]
    for k, (centers, yaws, eta) in enumerate(locked_specs, 1):
        rects = [_rect_at(f"L{i}", x, y, 0.8, 0.8, yaw)
                 for i, ((x, y), yaw) in enumerate(zip(centers, yaws))]
        out.append(GeometryInstance(
            f"yaw_independent_locked_{k}_{len(rects)}", rects,
            F=0.50, D=0.90, finger_clearance=eta,
            metadata={"family": "yaw-independent", "yaw_type": "individual",
                      "clearance_model": f"eta_F={eta:.2f}",
                      "description": "finite-clearance Pick dependency cycle"}))
    return out


def _segments_distance(a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, d: np.ndarray) -> float:
    """Euclidean distance between two closed planar line segments."""
    def point_segment(p: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        delta = y - x
        denom = float(delta @ delta)
        t = 0.0 if denom <= EPS else min(1.0, max(0.0, float((p-x) @ delta) / denom))
        return float(np.linalg.norm(p - (x + t*delta)))

    ab, cd = b-a, d-c
    if (_cross(ab, c-a) * _cross(ab, d-a) <= EPS and
            _cross(cd, a-c) * _cross(cd, b-c) <= EPS):
        return 0.0
    return min(point_segment(a, c, d), point_segment(b, c, d),
               point_segment(c, a, b), point_segment(d, a, b))


def footprint_distance(first: Rect, second: Rect) -> float:
    if len(_convex_intersection(first.corners, second.corners)):
        return 0.0
    return min(_segments_distance(a, b, c, d)
               for a, b in zip(first.corners, np.roll(first.corners, -1, axis=0))
               for c, d in zip(second.corners, np.roll(second.corners, -1, axis=0)))


def clutter_metrics(inst: GeometryInstance) -> Dict[str, float]:
    """Rotation-independent density and proximity descriptors for a scene."""
    hull = _convex_hull(np.vstack([rect.corners for rect in inst.rects]))
    hull_area = _polygon_area(hull)
    footprint_area = sum(rect.w * rect.h for rect in inst.rects)
    distances = [footprint_distance(inst.rects[i], inst.rects[j])
                 for i in range(len(inst.rects)) for j in range(i)]
    threshold = inst.finger_clearance
    near_edges = sum(distance <= threshold + EPS for distance in distances)
    possible = len(inst.rects) * (len(inst.rects)-1) / 2
    nearest = [min(footprint_distance(rect, other)
                   for other in inst.rects if other is not rect)
               for rect in inst.rects] if len(inst.rects) > 1 else [0.0]
    return {
        "packing_fraction": footprint_area / hull_area if hull_area > EPS else 1.0,
        "mean_nearest_gap": float(np.mean(nearest)),
        "minimum_gap": min(distances, default=0.0),
        "near_neighbor_edges": float(near_edges),
        "near_neighbor_density": near_edges / possible if possible else 0.0,
    }


def _move_rect(rect: Rect, delta: np.ndarray) -> Rect:
    center = rect.center + delta
    return _rect_at(rect.name, float(center[0]), float(center[1]),
                    rect.w, rect.h, math.degrees(rect.yaw))


def _compact_oriented_rectangles(rects: List[Rect], rng: random.Random,
                                 sweeps: int = 90) -> List[Rect]:
    """Move collision-free rectangles toward the clump center until jammed."""
    current = list(rects)
    for sweep in range(sweeps):
        center = np.mean([rect.center for rect in current], axis=0)
        changed = False
        order = list(range(len(current)))
        rng.shuffle(order)
        step = max(0.006, 0.045 * (1.0 - sweep / sweeps))
        for index in order:
            direction = center - current[index].center
            norm = float(np.linalg.norm(direction))
            if norm <= EPS:
                continue
            # A small transverse component prevents all objects collapsing
            # onto a few radial chains while retaining deterministic density.
            tangent = np.array([-direction[1], direction[0]]) / norm
            direction = direction / norm + rng.uniform(-0.20, 0.20) * tangent
            direction /= np.linalg.norm(direction)
            candidate = _move_rect(current[index], step * direction)
            if all(_polygon_area(_convex_intersection(candidate.corners, other.corners))
                   <= EPS for j, other in enumerate(current) if j != index):
                current[index] = candidate
                changed = True
        if not changed and sweep > 20:
            break
    return current


def non_axis_aligned_experiment_instances() -> List[GeometryInstance]:
    """Twenty reproducible dense independently yawed locked scenes."""
    result = []
    for case_index in range(1, 21):
        seed = 4167 if case_index == 19 else 3100 + 97 * case_index
        rng = random.Random(seed)
        n = 4 + ((case_index * 3) % 8)  # cycles through 4--11 objects
        columns = math.ceil(math.sqrt(n))
        rows = math.ceil(n / columns)
        rects = []
        for index in range(n):
            row, column = divmod(index, columns)
            width = rng.uniform(0.72, 1.02)
            height = rng.uniform(0.68, 0.98)
            # Exclude almost-axis-aligned angles and make every orientation
            # independent, while limiting extreme bounding-box inflation.
            magnitude = rng.uniform(7.0, 32.0)
            yaw = magnitude if rng.random() < 0.5 else -magnitude
            yaw += 0.31 * index
            x = 1.35 * column + rng.uniform(-0.08, 0.08)
            y = 1.35 * row + rng.uniform(-0.08, 0.08)
            rects.append(_rect_at(f"Y{index}", x, y, width, height, yaw))
        rects = _compact_oriented_rectangles(rects, rng)
        if case_index == 19:
            # A rigid extra rotation preserves the locked rule structure of
            # this reproducible seed while making its yaws distinct from case
            # 11, which deliberately has the same object count.
            rects = _rotate_clump(rects, 4.5)
        min_side = min(rect.min_side for rect in rects)
        max_min_side = max(rect.min_side for rect in rects)
        # The finite-clearance band joins close but nonintersecting bodies into
        # Pick dependencies. Its physical value scales with the whole scene.
        eta = 0.60 if case_index in (3, 9) else 0.50
        instance = GeometryInstance(
            f"yaw_independent_locked_dense_{case_index:02d}_n{n}_s{seed}",
            rects, F=0.62 * min_side, D=1.08 * max_min_side,
            finger_clearance=eta,
            metadata={"family": "yaw-independent-locked",
                      "experiment_category": "non-axis-aligned",
                      "yaw_type": "individual", "seed": str(seed),
                      "description": "dense random independently yawed locked clutter"},
        )
        metrics = clutter_metrics(instance)
        instance.metadata.update({key: f"{value:.8g}" for key, value in metrics.items()})
        result.append(instance)
    return result


def primary_experiment_instances() -> List[GeometryInstance]:
    """Exactly 20 axis-aligned and 20 independently yawed primary cases."""
    result = (axis_aligned_experiment_instances() +
              non_axis_aligned_experiment_instances())
    counts = {category: sum(instance.metadata.get("experiment_category") == category
                            for instance in result)
              for category in ("axis-aligned", "non-axis-aligned")}
    if counts != {"axis-aligned": 20, "non-axis-aligned": 20}:
        raise AssertionError(f"invalid primary experiment composition: {counts}")
    return result


# ---------------------------------------------------------------------------
# Plotting and reporting
# ---------------------------------------------------------------------------

def _plot_bounds(inst: GeometryInstance) -> Tuple[float, float, float, float, float, float]:
    points = np.vstack([r.corners for r in inst.rects])
    xmin, ymin = points.min(axis=0); xmax, ymax = points.max(axis=0)
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
    from matplotlib.patches import Polygon

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
        patch = Polygon(r.corners, closed=True, fill=fill is not None,
                          facecolor=fill if fill is not None else None,
                          edgecolor=edgecolor, linewidth=linewidth,
                          linestyle=linestyle, alpha=alpha if fill is not None else 1.0)
        ax.add_patch(patch)
        ax.text(*r.center, r.name,
                ha="center", va="center", fontsize=8,
                color="black" if is_present else "0.45")


def _annotate_step_action(ax, rect: Rect, action: Action, step: int, dim_scale: float,
                          emphasize: bool = False) -> None:
    cx, cy = rect.center
    u = np.array([math.cos(rect.yaw), math.sin(rect.yaw)])
    v = np.array([-math.sin(rect.yaw), math.cos(rect.yaw)])
    badge_x, badge_y = rect.center - .32*rect.w*u + .32*rect.h*v
    label = f"{step}{'P' if action.kind == 'pick' else 'K'}"
    ax.text(badge_x, badge_y, label, ha="center", va="center", fontsize=8,
            bbox=dict(boxstyle="circle,pad=0.23",
                      fc=(0.88, 1.0, 0.88) if action.kind == 'pick' else (1.0, 0.9, 0.9),
                      ec="black", lw=1.0))
    if action.kind == "pick":
        span = 0.32 * rect.w if action.orientation == "x" else 0.32 * rect.h
        if action.orientation == "x":
            dv = span*u
            ax.annotate("", xy=(cx + dv[0], cy + dv[1]), xytext=(cx - dv[0], cy - dv[1]),
                        arrowprops=dict(arrowstyle="<->", lw=1.6))
            ax.text(cx, cy + 0.12 * rect.h + 0.05 * dim_scale,
                    f"P({action.orientation})", ha="center", va="bottom", fontsize=7)
        else:
            dv = span*v
            ax.annotate("", xy=(cx + dv[0], cy + dv[1]), xytext=(cx - dv[0], cy - dv[1]),
                        arrowprops=dict(arrowstyle="<->", lw=1.6))
            ax.text(cx + 0.12 * rect.w + 0.05 * dim_scale, cy,
                    f"P({action.orientation})", ha="left", va="center", fontsize=7)
    else:
        dx, dy = _dir_vec(action.direction or "+x")
        c, s = math.cos(rect.yaw), math.sin(rect.yaw)
        dx, dy = c*dx-s*dy, s*dx+c*dy
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
                workspace_margin: float = 0.01, max_jaw_width: float = 0.075,
                finger_width: float = 0.020, contact_gap: float = 0.005,
                minimum_physical_side: float = 0.020) -> float:
    """Find a physically valid uniform scale or reject the Isaac export.

    The lower bound ensures the complete tangential finger footprint fits in
    every planner window and that a tipped cuboid remains tall enough to grasp.
    The upper bound enforces both the 25 cm workspace and Panda jaw opening.
    """
    points = np.vstack([r.corners for r in inst.rects])
    span_x, span_y = points.max(axis=0) - points.min(axis=0)
    usable = workspace_size - workspace_margin
    if usable <= 0:
        raise ValueError("workspace margin must be smaller than workspace size")
    upper = float(min(usable / max(span_x, span_y), max_jaw_width / inst.D))
    lower = float(max(
        finger_width / inst.F,
        (minimum_physical_side + contact_gap)
        / min(r.min_side for r in inst.rects),
    ))
    if lower > upper + EPS:
        raise ValueError(
            f"no physical scale satisfies the 25 cm workspace, {max_jaw_width*1000:.0f} mm "
            f"jaw limit, {finger_width*1000:.0f} mm finger window, and "
            f"{minimum_physical_side*1000:.0f} mm minimum side "
            f"(required scale {lower:.5f}, maximum {upper:.5f})"
        )
    return max(lower, upper)  # largest feasible geometry improves contact robustness


PANDA_FINGER_TANGENTIAL_EXTENT = 0.0211
PANDA_FINGER_EDGE_MARGIN = 0.002
PANDA_BOUNDARY_GRASP_INWARD_NUDGE = 0.010


def _pick_grasp_payload(inst: GeometryInstance, rect: Rect, action: Action,
                        scale: float, contact_gap: float,
                        finger_width: float) -> Dict[str, object]:
    # ``offset`` remains the planner-derived centre of [S,S+W].  A boundary
    # nudge is serialized separately and applied by the executor at runtime,
    # so it can be tuned directly in the YAML without regenerating the plan.
    edge_margin = PANDA_FINGER_EDGE_MARGIN
    if action.orientation == "x":
        physical_span = rect.h * scale - contact_gap
        axis = 1
    else:
        physical_span = rect.w * scale - contact_gap
        axis = 0
    if finger_width + 2*edge_margin > physical_span + EPS:
        raise ValueError(f"{rect.name}: physical finger does not fit tangential side")
    # The theoretical witness is the start S of [S,S+F]. Place the physical
    # finger directly on [S,S+W]; its commanded point is the interval centre.
    start = float(action.placement) * scale
    low = -0.5*physical_span + edge_margin
    high = 0.5*physical_span - edge_margin - finger_width
    physical_start = min(max(start, low), high)
    center = physical_start + 0.5*finger_width
    inward_nudge = 0.0
    if abs(physical_start - low) <= 1e-8:
        inward_nudge = PANDA_BOUNDARY_GRASP_INWARD_NUDGE
    elif abs(physical_start - high) <= 1e-8:
        inward_nudge = PANDA_BOUNDARY_GRASP_INWARD_NUDGE
    offset = [0.0, 0.0]
    offset[axis] = center
    return {
        "closing_axis": action.orientation,
        "offset": [round(v, 8) for v in offset],
        "placement_witness": round(float(action.placement) * scale, 8),
        "finger_side_contact_width": finger_width,
        "physical_edge_margin": edge_margin,
        "inward_nudge": round(float(inward_nudge), 8),
        "finger_window": [round(physical_start, 8),
                          round(physical_start + finger_width, 8)],
    }


def _post_clearance_grasp_payload(inst: GeometryInstance, rect: Rect,
                                  action: Action,
                                  scale: float, contact_gap: float,
                                  finger_side_width: float) -> Dict[str, object]:
    """Serialize the end grasp exposed by the selected clearance action.

    A clearance in +/-Y exposes the corresponding Y end and permits an
    X-closing grasp; +/-X is the symmetric case.  The finger window has length
    F, so its center is F/2 inward from the selected leading end.
    """
    direction = str(action.direction)
    sign = 1.0 if direction[0] == "+" else -1.0
    axis = direction[1]
    edge_margin = PANDA_FINGER_EDGE_MARGIN
    if axis == "y":
        closing_axis = "x"
        physical_span = rect.h * scale - contact_gap
        contact_width = finger_side_width
        if contact_width + 2*edge_margin > physical_span + EPS:
            raise ValueError(f"{rect.name}: full finger does not fit exposed Y end")
        center = sign * max(
            0.0, 0.5 * physical_span - edge_margin - 0.5 * contact_width
        )
        offset = [0.0, center]
        closing_span = rect.w * scale - contact_gap
    elif axis == "x":
        closing_axis = "y"
        physical_span = rect.w * scale - contact_gap
        contact_width = finger_side_width
        if contact_width + 2*edge_margin > physical_span + EPS:
            raise ValueError(f"{rect.name}: full finger does not fit exposed X end")
        center = sign * max(
            0.0, 0.5 * physical_span - edge_margin - 0.5 * contact_width
        )
        offset = [center, 0.0]
        closing_span = rect.h * scale - contact_gap
    else:
        raise ValueError(f"unsupported clearance direction {direction!r}")
    if closing_span > 0.075 + EPS:
        raise ValueError(
            f"{rect.name}: clearance exposes an end whose required closing "
            f"span is {closing_span:.4f} m, exceeding the Panda's 0.075 m "
            "safe jaw span; no executor-invented fallback grasp is allowed"
        )
    return {
        "closing_axis": closing_axis,
        "offset": [round(float(v), 8) for v in offset],
        "derived_from_clearance_witness": action.witness,
        "finger_side_contact_width": round(float(contact_width), 8),
        "physical_edge_margin": edge_margin,
        "inward_nudge": round(float(PANDA_BOUNDARY_GRASP_INWARD_NUDGE), 8),
        "post_clearance_grasp_mode": "exposed_end",
        # Interval coordinates along the non-closing side, relative to the
        # cuboid center. This makes the planner's [S,S+W] convention explicit.
        "finger_window": [
            round(float(center - 0.5 * contact_width), 8),
            round(float(center + 0.5 * contact_width), 8),
        ],
    }


def isaac_feasible_knock_actions(rule: RuleInstance,
                                  contact_gap: float = 0.005,
                                  finger_side_width: float = PANDA_FINGER_TANGENTIAL_EXTENT
                                  ) -> FrozenSet[Action]:
    """Return Knock rules whose mandatory exposed-end Pick fits the Panda."""
    scale = isaac_scale(rule.geometry, finger_width=finger_side_width,
                        contact_gap=contact_gap)
    feasible = set()
    for index, actions in rule.actions.items():
        for action in actions:
            if action.kind != "knock":
                continue
            try:
                _post_clearance_grasp_payload(
                    rule.geometry, rule.geometry.rects[index], action, scale,
                    contact_gap, finger_side_width)
            except ValueError:
                continue
            feasible.add(action)
    return frozenset(feasible)


def write_isaac_problem(rule: RuleInstance, result: SolveResult, path: Path,
                        contact_gap: float = 0.005, common_height: float = 0.05,
                        max_clearance_attempts: int = 24,
                        finger_side_width: float = PANDA_FINGER_TANGENTIAL_EXTENT,
                        minimum_physical_side: float = 0.020) -> Dict[str, object]:
    """Serialize an exact cuboid plan for the native Isaac executor.

    Planning contacts remain exact. Physical rectangles are shrunk symmetrically
    by ``contact_gap`` so nominally contacting bodies have a small robustness gap.
    """
    if result.status not in {"optimal", "feasible"} or result.selected_actions is None or result.order is None:
        raise ValueError("a verified complete plan is required for Isaac export")
    inst = rule.geometry
    scale = isaac_scale(
        inst, finger_width=finger_side_width, contact_gap=contact_gap,
        minimum_physical_side=minimum_physical_side,
    )
    bounds = np.vstack([r.corners for r in inst.rects])
    xmin, ymin = bounds.min(axis=0); xmax, ymax = bounds.max(axis=0)
    # Keep low tabletop contact motions inside the Panda's comfortable
    # workspace.  Goal pads may be farther away because transport occurs high.
    plan_mid = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)])
    # Shift both source and goal workspaces 5 cm along ROS +X. Under the
    # repository's Rz(+90 deg) convention this is Isaac +Y, farther from the
    # Franka base, without changing any relative geometry.
    ros_x_workspace_shift = 0.05
    source_origin = np.array([0.40 + ros_x_workspace_shift, 0.0])
    pads: Dict[int, Dict[str, float]] = {}
    objects: Dict[str, Dict[str, object]] = {}
    initial = []
    goal = []
    initial_pad_by_index = {}
    destination_pad_by_index = {}

    # The current SimpleRoom task rests on an effectively unbounded support
    # plane.  Define a finite, explicit manipulation-table envelope and attach
    # two symmetric sorting chutes to its +Y (front/away-from-robot) edge.
    # These two destinations are fixed for every instance and therefore cannot
    # influence the planner's action order.
    table_bounds_isaac = {
        "xmin": -0.25, "xmax": 0.25, "ymin": 0.28, "ymax": 0.78,
    }
    chute_centers_isaac = {
        "red": np.array([-0.13, 0.82]),
        "blue": np.array([0.13, 0.82]),
    }
    # Invert plan_xy_to_isaac: (x_I,y_I)=(-y_R,x_R+0.1).
    chute_plan_xy = {
        color: np.array([center[1] - 0.1, -center[0]])
        for color, center in chute_centers_isaac.items()
    }
    chute_pad_by_color = {
        "red": len(inst.rects) + 1,
        "blue": len(inst.rects) + 2,
    }
    for color, pad in chute_pad_by_color.items():
        pads[pad] = {
            "x": round(float(chute_plan_xy[color][0]), 8),
            "y": round(float(chute_plan_xy[color][1]), 8),
        }

    shuffled = list(range(len(inst.rects)))
    random.Random(f"{inst.name}:two-color-sorting").shuffle(shuffled)
    sort_color_by_index = {
        index: ("red" if rank % 2 == 0 else "blue")
        for rank, index in enumerate(shuffled)
    }

    for i, rect in enumerate(inst.rects):
        center = source_origin + scale * (
            rect.center
            - plan_mid
        )
        pad = i + 1
        initial_pad_by_index[i] = pad
        pads[pad] = {"x": round(float(center[0]), 8), "y": round(float(center[1]), 8)}
        physical_w = rect.w * scale - contact_gap
        physical_h = rect.h * scale - contact_gap
        # Preserve the requested physical contact gap even for thin synthetic
        # rails.  Below 2 mm PhysX contact becomes unreliable at this scale;
        # the 5 mm gap leaves the thinnest current benchmark at about 2.9 mm.
        if min(physical_w, physical_h) <= 0.002:
            raise ValueError(
                f"{inst.name}/{rect.name}: scaled side too small for reliable execution "
                f"({physical_w:.4f} x {physical_h:.4f} m)"
            )
        objects[rect.name] = {
            # The Rz(+90 deg) ROS-plan -> Isaac transform swaps footprint axes.
            "dimensions": [round(physical_h, 8), round(physical_w, 8), common_height],
            "plan_center": [float(round(float(center[0]), 8)),
                            float(round(float(center[1]), 8))],
            "planning_dimensions": [round(rect.w * scale, 8), round(rect.h * scale, 8)],
            "yaw_rad": round(float(rect.yaw), 10),
            "sort_color": sort_color_by_index[i],
        }
        initial.append({"name": rect.name, "pad": pad, "height": 0,
                        "yaw_rad": round(float(rect.yaw), 10)})

        destination_pad = chute_pad_by_color[sort_color_by_index[i]]
        destination_pad_by_index[i] = destination_pad
        goal.append({"name": rect.name, "pad": destination_pad, "height": 0,
                     "sort_color": sort_color_by_index[i]})

    actions = []
    for i in result.order:
        selected = result.selected_actions[i]
        rect = inst.rects[i]
        if selected.kind == "knock":
            actions.append({
                "type": "Clearance", "blocks": [rect.name],
                "dir": selected.direction,
                "direction_frame": "object_local",
                "required_displacement": round(inst.F * scale + 0.005, 8),
                "solver_witness": selected.witness,
            })
            move = {
                "type": "Move", "block": rect.name,
                "to": {"pad": destination_pad_by_index[i], "height": 0},
                "grasp": _post_clearance_grasp_payload(
                    inst, rect, selected, scale, contact_gap,
                    finger_side_width,
                ),
                "after_clearance": True,
                "remove_after_place": True,
            }
        else:
            move = {
                "type": "Move", "block": rect.name,
                "to": {"pad": destination_pad_by_index[i], "height": 0},
                "grasp": _pick_grasp_payload(
                    inst, rect, selected, scale, contact_gap,
                    finger_side_width,
                ),
                "solver_witness": selected.witness,
                "remove_after_place": True,
            }
        actions.append(move)

    document = {
        "schema_version": 2,
        "planner": "heterogeneous-cuboid-exact",
        "instance": inst.name,
        "family": inst.metadata.get("family", ""),
        "experiment_category": inst.metadata.get("experiment_category", "others"),
        "yaw_type": inst.metadata.get("yaw_type", "axis-aligned"),
        "solver": result.solver,
        "planning_time_sec": result.runtime_sec,
        "optimal_clearances": result.optimum,
        "scale": scale,
        "finger_contact_length": inst.F * scale,
        "jaw_opening": inst.D * scale,
        "finger_clearance": inst.finger_clearance * scale,
        "contact_gap": contact_gap,
        "ros_x_workspace_shift": ros_x_workspace_shift,
        "execution": {
            "clearance_margin": 0.005,
            "panda_finger_clearance": 0.022,
            "finger_side_contact_width": finger_side_width,
            "minimum_physical_side": minimum_physical_side,
            "max_clearance_attempts": max(10, max_clearance_attempts),
        },
        "sorting": {
            "mode": "two-color-fixed-chutes",
            "table_bounds_isaac": table_bounds_isaac,
            "chute_size": [0.18, 0.08],
            "chutes": {
                color: {
                    "pad": chute_pad_by_color[color],
                    "center_isaac": chute_centers_isaac[color].tolist(),
                }
                for color in ("red", "blue")
            },
        },
        "pads": pads, "objects": objects, "initial": initial,
        "goal": goal, "actions": actions,
    }
    # Normalize NumPy scalar values introduced by rotated-center arithmetic;
    # PyYAML's safe dumper intentionally accepts only native Python scalars.
    serializable_document = json.loads(json.dumps(document))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(serializable_document, sort_keys=False), encoding="utf-8")
    # Return exactly the native-Python representation that was written.  Some
    # callers add benchmark metadata and dump the document again; returning
    # the pre-normalized dictionary leaked NumPy scalars into that second dump.
    return serializable_document


def run_benchmarks(output_dir: Path, sat_time_limit: Optional[float] = 60.0,
                   ilp_time_limit: Optional[float] = 60.0, make_plots: bool = True,
                   export_isaac: bool = True,
                   isaac_exportable_only: bool = False,
                   instances: Optional[Sequence[GeometryInstance]] = None
                   ) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if make_plots:
        (output_dir / "layouts").mkdir(exist_ok=True)
        (output_dir / "solution_overlays").mkdir(exist_ok=True)
        (output_dir / "solution_sequences").mkdir(exist_ok=True)
    if export_isaac:
        (output_dir / "isaac_problems").mkdir(exist_ok=True)

    rows: List[Dict[str, object]] = []
    for inst in (all_benchmarks() if instances is None else instances):
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
            isaac_path = output_dir / "isaac_problems" / f"{inst.name}.yaml"
            # Never leave a stale executable problem behind when a newly
            # enforced physical-scale check rejects this instance.
            isaac_path.unlink(missing_ok=True)
            try:
                write_isaac_problem(
                    rule, ilp_res, isaac_path,
                )
                isaac_export = "ok"
            except ValueError as exc:
                isaac_export = f"unsuitable:{exc}"

        if ilp_res.optimum is not None and sat_res.optimum is not None and ilp_res.optimum != sat_res.optimum:
            raise AssertionError(f"Solver disagreement on {inst.name}: ILP={ilp_res.optimum}, SAT={sat_res.optimum}")
        if bfs_opt is not None and ilp_res.optimum is not None and bfs_opt != ilp_res.optimum:
            raise AssertionError(f"BFS disagreement on {inst.name}: BFS={bfs_opt}, ILP={ilp_res.optimum}")

        if isaac_exportable_only and isaac_export != "ok":
            if make_plots:
                (output_dir / "layouts" / f"{inst.name}.png").unlink(
                    missing_ok=True)
                for folder in ("solution_overlays", "solution_sequences"):
                    for generated in (output_dir / folder).glob(f"{inst.name}_*.png"):
                        generated.unlink()
            continue

        row: Dict[str, object] = {
            "instance": inst.name,
            "family": inst.metadata.get("family", ""),
            "experiment_category": inst.metadata.get("experiment_category", "others"),
            "seed": inst.metadata.get("seed", ""),
            "n": rule.n,
            "F": inst.F,
            "D": inst.D,
            "eta_F": inst.finger_clearance,
            "yaw_objects": sum(abs(r.yaw) > EPS for r in inst.rects),
            "yaw_min_deg": min(math.degrees(r.yaw) for r in inst.rects),
            "yaw_max_deg": max(math.degrees(r.yaw) for r in inst.rects),
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
            "packing_fraction": inst.metadata.get("packing_fraction", ""),
            "mean_nearest_gap": inst.metadata.get("mean_nearest_gap", ""),
            "minimum_gap": inst.metadata.get("minimum_gap", ""),
            "near_neighbor_edges": inst.metadata.get("near_neighbor_edges", ""),
            "near_neighbor_density": inst.metadata.get("near_neighbor_density", ""),
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

    # Category-level planning summary for the restructured experiment. Empty
    # clutter fields on axis-aligned cases remain explicit rather than being
    # silently interpreted as zero density.
    category_rows = []
    for category in sorted({str(row["experiment_category"]) for row in rows}):
        group = [row for row in rows if row["experiment_category"] == category]
        numeric = lambda key: [float(row[key]) for row in group
                               if row.get(key) not in (None, "")]
        def avg(key: str) -> object:
            values = numeric(key)
            return float(np.mean(values)) if values else ""
        def std(key: str) -> object:
            values = numeric(key)
            return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else ""
        category_rows.append({
            "experiment_category": category,
            "cases": len(group),
            "n_mean": avg("n"), "n_std": std("n"),
            "planning_time_sec_mean": float(np.mean([
                float(row["preprocess_sec"]) + float(row["ILP_sec"]) for row in group])),
            "planning_time_sec_std": float(np.std([
                float(row["preprocess_sec"]) + float(row["ILP_sec"]) for row in group], ddof=1))
                if len(group) > 1 else 0.0,
            "optimal_knocks_mean": avg("ILP_opt"),
            "optimal_knocks_std": std("ILP_opt"),
            "physical_actions_mean": avg("physical_actions"),
            "physical_actions_std": std("physical_actions"),
            "packing_fraction_mean": avg("packing_fraction"),
            "packing_fraction_std": std("packing_fraction"),
            "mean_nearest_gap": avg("mean_nearest_gap"),
            "near_neighbor_density_mean": avg("near_neighbor_density"),
            "isaac_exportable": sum(row["isaac_export"] == "ok" for row in group),
        })
    category_csv = output_dir / "planning_category_summary.csv"
    with category_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(category_rows[0]))
        writer.writeheader(); writer.writerows(category_rows)
    with (output_dir / "planning_category_summary.md").open("w", encoding="utf-8") as stream:
        stream.write("# Primary Experiment Planning Summary\n\n")
        stream.write("| Category | Cases | n | Plan time (s) | OPT Knocks | Physical actions | Packing fraction | Near-neighbor density | Exportable |\n")
        stream.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in category_rows:
            fmt = lambda value: "--" if value == "" else f"{float(value):.3f}"
            stream.write(
                f"| {row['experiment_category']} | {row['cases']} | "
                f"{fmt(row['n_mean'])} $\\pm$ {fmt(row['n_std'])} | "
                f"{fmt(row['planning_time_sec_mean'])} $\\pm$ {fmt(row['planning_time_sec_std'])} | "
                f"{fmt(row['optimal_knocks_mean'])} $\\pm$ {fmt(row['optimal_knocks_std'])} | "
                f"{fmt(row['physical_actions_mean'])} $\\pm$ {fmt(row['physical_actions_std'])} | "
                f"{fmt(row['packing_fraction_mean'])} | {fmt(row['near_neighbor_density_mean'])} | "
                f"{row['isaac_exportable']}/{row['cases']} |\n")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--sat-time-limit", type=float, default=60.0)
    parser.add_argument("--ilp-time-limit", type=float, default=60.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-isaac-export", action="store_true")
    parser.add_argument(
        "--isaac-exportable-only", action="store_true",
        help="write summaries, plans, plots, and YAMLs only for cases that pass Isaac physical export",
    )
    parser.add_argument(
        "--primary-experiment-only", action="store_true",
        help="solve exactly the 20 axis-aligned and 20 independently yawed primary cases",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    rows = run_benchmarks(
        out, args.sat_time_limit, args.ilp_time_limit,
        not args.no_plots, not args.no_isaac_export,
        args.isaac_exportable_only,
        primary_experiment_instances() if args.primary_experiment_only else None,
    )
    print(f"\nWrote {len(rows)} benchmark results to {out.resolve()}")
    for r in rows:
        print(f"{r['instance']:<24} n={r['n']:>2} OPT={r['ILP_opt']} "
              f"ILP={float(r['ILP_sec']):.4f}s SAT={float(r['SAT_sec']):.4f}s")


if __name__ == "__main__":
    main()
