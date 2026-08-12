#!/usr/bin/env python3
from __future__ import annotations

"""
KP benchmark + planners + IsaacSim YAML generator

Keeps:
  --debug / --no-debug
  --visualize (optional matplotlib visualization)

Implements planners:
  (A) full grid exact: RectLayerPeel (layer-peeling optimal for full rectangles)
  (B) greedy approx: face-set-cover + outermost-preferred execution
  (C) subgraph "layer" heuristic: outermost-face gadget peel (same executor idea without a cover)

Synthetic benchmark:
  - Full grids: run exact+greedy once each size
  - Subgraphs: generate 20 random subsets each size; run greedy (and optionally layer heuristic)
    (default subset sizes follow your earlier list; change SUBSET_SIZES if needed)

IsaacSim generation:
  - full (3x3,5x5) + layer peeling: 2 files
  - full (3x3,5x5) + greedy:       2 files
  - sub (3x3,5x5)  + greedy:       5 each size  (guaranteed >=1 face => >=1 knock)
  - sub (3x3,5x5)  + layer:        5 each size  (same)

Outputs YAML under ./problems by default.
"""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict, Iterable
import argparse
import collections
import math
import os
import random
import time

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

Node = Tuple[int, int]   # (i,j)
Face = Tuple[int, int]   # (i,j) top-left of a 2x2 face

# grid neighbor directions in (di,dj)
DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # down, up, right, left


# ============================================================
# Occupancy helpers
# ============================================================

def occ_full(m: int, n: int) -> Set[Node]:
    return {(i, j) for i in range(m) for j in range(n)}

def random_subgraph_with_forced_face(m: int, n: int, k: int, rng: random.Random) -> Set[Node]:
    """
    Sample k nodes from full m×n occupancy, but FORCE at least one 2×2 face
    so that there is at least one gadget => at least one knock is needed.
    """
    assert 0 < k <= m * n
    if k < 4 or m < 2 or n < 2:
        # cannot force a face; fall back to random k nodes
        all_nodes = [(i, j) for i in range(m) for j in range(n)]
        rng.shuffle(all_nodes)
        return set(all_nodes[:k])

    # choose a random face top-left
    fi = rng.randrange(0, m - 1)
    fj = rng.randrange(0, n - 1)
    face_nodes = {(fi, fj), (fi + 1, fj), (fi, fj + 1), (fi + 1, fj + 1)}

    rest = [(i, j) for i in range(m) for j in range(n) if (i, j) not in face_nodes]
    rng.shuffle(rest)
    needed = k - 4
    return set(face_nodes | set(rest[:needed]))


# ============================================================
# Graph construction + hull-layer geometry
# ============================================================

def build_graph(occ: Set[Node]) -> nx.Graph:
    G = nx.Graph()
    for (i, j) in occ:
        G.add_node((i, j), pos=(float(j), float(-i)))
    for (i, j) in occ:
        for di, dj in DIRS4:
            u = (i + di, j + dj)
            if u in occ:
                G.add_edge((i, j), u)
    return G

def layer_index_hull(i: int, j: int, m: int, n: int) -> int:
    """Hull-padding layer index w.r.t. full m×n rectangle."""
    return 1 + min(i, j, (m - 1) - i, (n - 1) - j)


# ============================================================
# Pick + knock feasibility
# ============================================================

def is_I_shape_degree_two(G: nx.Graph, u: Node) -> bool:
    if G.degree(u) != 2:
        return False
    v1, v2 = list(G.neighbors(u))
    x1, y1 = G.nodes[v1]["pos"]
    x2, y2 = G.nodes[v2]["pos"]
    return (x1 == x2) or (y1 == y2)

def pickable(G: nx.Graph, u: Node) -> bool:
    d = G.degree(u)
    if d <= 1:
        return True
    if d == 2 and is_I_shape_degree_two(G, u):
        return True
    return False

def ray_clear_directions(G: nx.Graph, u: Node, m: int, n: int) -> List[Tuple[int, int]]:
    """
    Ray(u,d): along direction d until leaving the hull, there is no remaining node.
    """
    occ = set(G.nodes())
    ui, uj = u
    feasible: List[Tuple[int, int]] = []
    for di, dj in DIRS4:
        i, j = ui + di, uj + dj
        clear = True
        while 0 <= i < m and 0 <= j < n:
            if (i, j) in occ:
                clear = False
                break
            i += di
            j += dj
        if clear:
            feasible.append((di, dj))
    return feasible

def knockable(G: nx.Graph, u: Node, m: int, n: int) -> bool:
    # your constraint: degree <= 3 and must have at least one clear ray
    if G.degree(u) > 3:
        return False
    return len(ray_clear_directions(G, u, m, n)) > 0


# ============================================================
# Core nodes: iterative removal of pickable nodes
# ============================================================

def core_nodes(G: nx.Graph) -> Set[Node]:
    H = G.copy()
    changed = True
    while changed:
        changed = False
        P = [u for u in list(H.nodes()) if pickable(H, u)]
        if P:
            H.remove_nodes_from(P)
            changed = True
    return set(H.nodes())


# ============================================================
# Faces and gadgets (FACE universe)
# ============================================================

def face_blocks(f: Face) -> Set[Node]:
    i, j = f
    return {(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)}

def enumerate_faces(U: Set[Node], m: int, n: int) -> Set[Face]:
    Uset = set(U)
    F: Set[Face] = set()
    if m < 2 or n < 2:
        return F
    for i in range(m - 1):
        for j in range(n - 1):
            f = (i, j)
            if face_blocks(f).issubset(Uset):
                F.add(f)
    return F

def face_layer(f: Face, m: int, n: int) -> int:
    return min(layer_index_hull(i, j, m, n) for (i, j) in face_blocks(f))

@dataclass(frozen=True)
class FaceGadget:
    kind: str                 # "2x2", "2x3", "3x2"
    faces: Tuple[Face, ...]   # 1 or 2
    cells: Tuple[Node, ...]   # 4 or 6

def enumerate_face_gadgets(U: Set[Node], F: Set[Face], m: int, n: int) -> List[FaceGadget]:
    """
    Gadget instances in current core:
      - 2x2: one face
      - 2x3: two horizontally adjacent faces
      - 3x2: two vertically adjacent faces
    """
    Uset = set(U)
    Fset = set(F)
    gadgets: List[FaceGadget] = []

    def add(kind: str, faces: List[Face], cells: List[Node]) -> None:
        gadgets.append(FaceGadget(kind=kind,
                                  faces=tuple(sorted(faces)),
                                  cells=tuple(sorted(cells))))

    # 2x2
    for f in Fset:
        add("2x2", [f], list(face_blocks(f)))

    # 2x3
    if n >= 3 and m >= 2:
        for i in range(m - 1):
            for j in range(n - 2):
                f1, f2 = (i, j), (i, j + 1)
                if f1 in Fset and f2 in Fset:
                    cells = [(i + di, j + dj) for di in range(2) for dj in range(3)]
                    if all(c in Uset for c in cells):
                        add("2x3", [f1, f2], cells)

    # 3x2
    if m >= 3 and n >= 2:
        for i in range(m - 2):
            for j in range(n - 1):
                f1, f2 = (i, j), (i + 1, j)
                if f1 in Fset and f2 in Fset:
                    cells = [(i + di, j + dj) for di in range(3) for dj in range(2)]
                    if all(c in Uset for c in cells):
                        add("3x2", [f1, f2], cells)

    uniq: Dict[Tuple, FaceGadget] = {}
    for g in gadgets:
        uniq[(g.kind, g.faces, g.cells)] = g
    return list(uniq.values())


# ============================================================
# Greedy face cover (unweighted)
# ============================================================

def greedy_set_cover_faces(
    F: Set[Face],
    gadgets: List[FaceGadget],
    m: int,
    n: int,
    debug: bool = True
) -> List[FaceGadget]:
    """
    Greedy set cover with your tie-breaking:
      maximize newly covered faces,
      minimize waste,
      prefer smaller |faces(g)| when only 1 face remains (forces 2x2),
      prefer outer layers, stable tie-break.
    """
    U = set(F)
    chosen: List[FaceGadget] = []
    kind_rank = {"2x2": 0, "2x3": 1, "3x2": 2}
    step = 0

    while U:
        best = None  # (key, gadget, newly)
        for g in gadgets:
            newly = set(g.faces) & U
            if not newly:
                continue
            cover_size = len(newly)
            waste = len(g.faces) - cover_size
            min_faceL = min(face_layer(f, m, n) for f in newly)
            min_cellL = min(layer_index_hull(i, j, m, n) for (i, j) in g.cells)

            key = (
                -cover_size,
                waste,
                len(g.faces),
                min_faceL,
                min_cellL,
                kind_rank.get(g.kind, 9),
                g.faces,
                g.cells,
            )
            if best is None or key < best[0]:
                best = (key, g, newly)

        if best is None:
            if debug:
                print(f"[COVER-FACE] stuck: remaining faces={len(U)}", flush=True)
            break

        _, gbest, newly = best
        chosen.append(gbest)
        U -= set(newly)
        if debug:
            print(f"[COVER-FACE] step {step}: chose {gbest.kind} covers {len(newly)} face(s); remaining faces={len(U)}",
                  flush=True)
        step += 1

        if step > 2_000_000:
            print("[COVER-FACE] guard triggered (too many iterations)", flush=True)
            break

    return chosen


# ============================================================
# STRICT knock candidates per gadget kind (your requirement)
# ============================================================

def gadget_knock_candidates(g: FaceGadget) -> List[Node]:
    """
    Your rule:
      - 2x2: any of 4 nodes (but must pass Ray)
      - 2x3: ONLY the two middle-column nodes
      - 3x2: ONLY the two middle-row nodes
    """
    cells = list(g.cells)
    is_ = [u[0] for u in cells]
    js_ = [u[1] for u in cells]
    i0, j0 = min(is_), min(js_)

    if g.kind == "2x2":
        return sorted(cells)

    if g.kind == "2x3":
        return [(i0, j0 + 1), (i0 + 1, j0 + 1)]

    if g.kind == "3x2":
        return [(i0 + 1, j0), (i0 + 1, j0 + 1)]

    return sorted(cells)

def choose_knock_for_gadget(
    G: nx.Graph,
    g: FaceGadget,
    coreU: Set[Node],
    m: int,
    n: int
) -> Optional[Tuple[Node, Tuple[int, int]]]:
    """
    Choose (v,d) for gadget g:
      - candidates restricted by gadget kind (middle nodes for 2x3/3x2),
      - enforce candidate must be present in coreU,
      - prefer smaller layer; then higher degree; then coord,
      - pick the first candidate with some Ray direction feasible.
    """
    cand0 = [v for v in gadget_knock_candidates(g) if (v in coreU) and G.has_node(v)]
    if not cand0:
        return None

    cand0.sort(key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))

    for v in cand0:
        # IMPORTANT: for 2x3/3x2, the correct candidates are typically degree-3 "middle" nodes.
        # If v has degree 4, it's interior and often NOT the correct removal for that gadget.
        if g.kind in ("2x3", "3x2") and G.degree(v) != 3:
            continue
        if G.degree(v) <= 3:
            dirs = ray_clear_directions(G, v, m, n)
            if dirs:
                dirs.sort()
                return v, dirs[0]

    return None


# ============================================================
# Plan representation
# ============================================================

@dataclass(frozen=True)
class PlanStep:
    node: Node
    kind: str  # "pick" or "knock"
    direction: Optional[Tuple[int, int]] = None  # only for knock

def dir_to_isaac(di_dj: Tuple[int, int]) -> str:
    """
    Map grid direction (di,dj) to IsaacSim dir strings:
      up    (-1,0) => +y
      down  ( 1,0) => -y
      right (0, 1) => +x
      left  (0,-1) => -x
    """
    di, dj = di_dj
    if (di, dj) == (-1, 0):
        return "+y"
    if (di, dj) == (1, 0):
        return "-y"
    if (di, dj) == (0, 1):
        return "+x"
    if (di, dj) == (0, -1):
        return "-x"
    return "+x"


# ============================================================
# Execution (greedy cover + outermost preferred)
# ============================================================

@dataclass
class ExecSnapshot:
    remaining: Set[Node]
    picked: Set[Node]
    knocked: Set[Node]
    last_step: Optional[PlanStep]
    note: str = ""

def execute_from_cover_outermost(
    G_full: nx.Graph,
    cover: List[FaceGadget],
    m: int,
    n: int,
    debug: bool = True,
) -> Tuple[bool, List[PlanStep], List[ExecSnapshot], Dict[str, int]]:
    """
    Outermost-preferred executor:
      1) Cleanup picks
      2) Compute core + faces; determine outermost face-layer
      3) Among UNUSED cover gadgets that still exist, pick one that hits outermost faces and is executable
      4) If none: EXTRA = pick any executable gadget in current core that hits outermost faces
      5) If still none: FALLBACK = pick any ray-feasible core node
    """
    G = G_full.copy()
    picked: Set[Node] = set()
    knocked: Set[Node] = set()
    plan: List[PlanStep] = []
    snaps: List[ExecSnapshot] = []
    used = [False] * len(cover)
    stats = collections.Counter()

    def record(last: Optional[PlanStep], note: str = "") -> None:
        snaps.append(ExecSnapshot(
            remaining=set(G.nodes()),
            picked=set(picked),
            knocked=set(knocked),
            last_step=last,
            note=note
        ))

    record(None, "start")

    it = 0
    max_it = 50 * max(1, G.number_of_nodes()) + 200

    while G.number_of_nodes() > 0:
        it += 1
        if it > max_it:
            print("[EXEC] guard triggered: too many iterations (deadlock/bug).", flush=True)
            return False, plan, snaps, dict(stats)

        # cleanup picks
        progress = False
        while True:
            P = [u for u in G.nodes() if pickable(G, u)]
            if not P:
                break
            P.sort()
            u = P[0]
            step = PlanStep(node=u, kind="pick", direction=None)
            plan.append(step)
            picked.add(u)
            G.remove_node(u)
            record(step, "cleanup")
            progress = True

        if G.number_of_nodes() == 0:
            break

        coreU = core_nodes(G)
        F = enumerate_faces(coreU, m, n)

        # mark cover gadgets "used" if their faces are no longer present at all
        # (this fixes the “unused notion” you asked about)
        Fset = set(F)
        for idx, g in enumerate(cover):
            if used[idx]:
                continue
            if not set(g.faces).issubset(Fset):
                used[idx] = True
                stats["auto_used"] += 1

        if not coreU:
            # should not happen if nodes exist, but safe
            continue

        if debug:
            outerL = min((face_layer(f, m, n) for f in F), default=None)
            print(f"[EXEC] iter={it} remainingV={G.number_of_nodes()} coreV={len(coreU)} faces={len(F)} "
                  f"outerL={outerL} used={sum(used)}/{len(cover)}",
                  flush=True)

        # If there are no faces but still core nodes exist, we must break cycles by a ray-feasible knock.
        if not F:
            # fallback knock directly
            cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
            chosen = None
            for v in cand:
                if knockable(G, v, m, n):
                    dirs = ray_clear_directions(G, v, m, n)
                    dirs.sort()
                    chosen = (v, dirs[0])
                    break
            if chosen is None:
                print("[EXEC] DEADLOCK: no ray-feasible knock in face-less core.", flush=True)
                return False, plan, snaps, dict(stats)
            v, d = chosen
            step = PlanStep(node=v, kind="knock", direction=d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            stats["fallback"] += 1
            record(step, "fallback (no faces)")
            continue

        outerL = min(face_layer(f, m, n) for f in F)
        outerF = {f for f in F if face_layer(f, m, n) == outerL}

        # Try UNUSED cover gadgets that touch outermost faces
        best = None  # (key, idx, (v,d))
        for idx, g in enumerate(cover):
            if used[idx]:
                continue
            hit = set(g.faces) & outerF
            if not hit:
                continue
            pair = choose_knock_for_gadget(G, g, coreU, m, n)
            if pair is None:
                continue
            v, d = pair
            key = (layer_index_hull(v[0], v[1], m, n), -len(hit), g.kind, g.faces, v)
            if best is None or key < best[0]:
                best = (key, idx, (v, d))

        if best is not None:
            _, idx, (v, d) = best
            used[idx] = True
            step = PlanStep(node=v, kind="knock", direction=d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            stats["cover_knock"] += 1
            record(step, f"cover gid={idx} kind={cover[idx].kind} d={d}")
            continue

        # EXTRA: enumerate current gadgets and take one that hits outermost faces
        cur_gadgets = enumerate_face_gadgets(coreU, set(F), m, n)
        extra_best = None  # (key, gadget, (v,d))
        for g in cur_gadgets:
            hit = set(g.faces) & outerF
            if not hit:
                continue
            pair = choose_knock_for_gadget(G, g, coreU, m, n)
            if pair is None:
                continue
            v, d = pair
            key = (layer_index_hull(v[0], v[1], m, n), -len(hit), g.kind, g.faces, v)
            if extra_best is None or key < extra_best[0]:
                extra_best = (key, g, (v, d))

        if extra_best is not None:
            _, g, (v, d) = extra_best
            step = PlanStep(node=v, kind="knock", direction=d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            stats["extra"] += 1
            record(step, f"extra kind={g.kind} d={d}")
            continue

        # FINAL fallback: any ray-feasible core node
        cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
        chosen = None
        for v in cand:
            if knockable(G, v, m, n):
                dirs = ray_clear_directions(G, v, m, n)
                dirs.sort()
                chosen = (v, dirs[0])
                break
        if chosen is None:
            print("[EXEC] DEADLOCK: no ray-feasible knock at all.", flush=True)
            return False, plan, snaps, dict(stats)

        v, d = chosen
        step = PlanStep(node=v, kind="knock", direction=d)
        plan.append(step)
        knocked.add(v)
        G.remove_node(v)
        stats["fallback"] += 1
        record(step, f"fallback d={d}")

    if debug:
        knocks = sum(1 for s in plan if s.kind == "knock")
        picks = sum(1 for s in plan if s.kind == "pick")
        print(f"[EXEC DONE] knocks={knocks} picks={picks} steps={len(plan)} stats={dict(stats)}", flush=True)

    return True, plan, snaps, dict(stats)


# ============================================================
# Planner: greedy approx
# ============================================================

def plan_greedy(occ: Set[Node], m: int, n: int, debug: bool = True, visualize: bool = False):
    G = build_graph(occ)
    U0 = core_nodes(G)
    F0 = enumerate_faces(U0, m, n)
    gadgets = enumerate_face_gadgets(U0, F0, m, n)

    if debug:
        cnt = collections.Counter(g.kind for g in gadgets)
        print(f"[INIT] |V|={G.number_of_nodes()} |E|={G.number_of_edges()} |Core|={len(U0)} |Faces|={len(F0)}", flush=True)
        if len(F0) > 0:
            print(f"[INIT] face lower-bound ceil(|Faces|/2)={(len(F0)+1)//2}", flush=True)
        print(f"[GADGETS] total={len(gadgets)} counts={dict(cnt)}", flush=True)

    t0 = time.perf_counter()
    cover = greedy_set_cover_faces(F0, gadgets, m, n, debug=debug)
    t_cover = time.perf_counter() - t0

    if debug:
        print(f"[COVER] chosen={len(cover)}  time={t_cover*1000:.2f}ms", flush=True)

    t1 = time.perf_counter()
    ok, plan, snaps, stats = execute_from_cover_outermost(G, cover, m, n, debug=debug)
    t_exec = time.perf_counter() - t1

    if debug:
        knocks = sum(1 for s in plan if s.kind == "knock")
        picks = sum(1 for s in plan if s.kind == "pick")
        print(f"[RESULT] ok={ok} knocks={knocks} picks={picks} steps={len(plan)} "
              f"extra={stats.get('extra',0)} fallback={stats.get('fallback',0)} "
              f"time_exec={t_exec*1000:.2f}ms total={(t_cover+t_exec)*1000:.2f}ms",
              flush=True)

    if visualize:
        visualize_solution(G, snaps, cover=cover, title="Greedy: cover+sequence", show_faces=False)

    return ok, plan, cover, t_cover, t_exec, stats


# ============================================================
# Planner: subgraph "layer" heuristic (outermost gadgets, no cover)
# ============================================================

def plan_layer_subgraph(occ: Set[Node], m: int, n: int, debug: bool = True, visualize: bool = False):
    G = build_graph(occ)
    plan: List[PlanStep] = []
    snaps: List[ExecSnapshot] = []
    stats = collections.Counter()

    picked: Set[Node] = set()
    knocked: Set[Node] = set()

    def record(last: Optional[PlanStep], note: str = ""):
        snaps.append(ExecSnapshot(set(G.nodes()), set(picked), set(knocked), last, note))

    record(None, "start")

    it = 0
    max_it = 60 * max(1, G.number_of_nodes()) + 200

    while G.number_of_nodes() > 0:
        it += 1
        if it > max_it:
            print("[LAYER-SUB] guard triggered: too many iterations", flush=True)
            return False, plan, None, 0.0, 0.0, dict(stats)

        # cleanup picks
        while True:
            P = [u for u in G.nodes() if pickable(G, u)]
            if not P:
                break
            P.sort()
            u = P[0]
            step = PlanStep(u, "pick", None)
            plan.append(step)
            picked.add(u)
            G.remove_node(u)
            record(step, "cleanup")

        if G.number_of_nodes() == 0:
            break

        coreU = core_nodes(G)
        F = enumerate_faces(coreU, m, n)

        if debug:
            outerL = min((face_layer(f, m, n) for f in F), default=None)
            print(f"[LAYER-SUB] iter={it} remainingV={G.number_of_nodes()} coreV={len(coreU)} faces={len(F)} outerL={outerL}",
                  flush=True)

        if not coreU:
            continue

        if not F:
            # break a face-less cycle by ray-feasible knock
            cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
            chosen = None
            for v in cand:
                if knockable(G, v, m, n):
                    dirs = ray_clear_directions(G, v, m, n)
                    dirs.sort()
                    chosen = (v, dirs[0])
                    break
            if chosen is None:
                print("[LAYER-SUB] DEADLOCK: no ray-feasible knock in face-less core.", flush=True)
                return False, plan, None, 0.0, 0.0, dict(stats)
            v, d = chosen
            step = PlanStep(v, "knock", d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            stats["fallback"] += 1
            record(step, "fallback(no faces)")
            continue

        outerL = min(face_layer(f, m, n) for f in F)
        outerF = {f for f in F if face_layer(f, m, n) == outerL}

        gadgets = enumerate_face_gadgets(coreU, set(F), m, n)
        best = None  # (key, g, (v,d))
        for g in gadgets:
            hit = set(g.faces) & outerF
            if not hit:
                continue
            pair = choose_knock_for_gadget(G, g, coreU, m, n)
            if pair is None:
                continue
            v, d = pair
            key = (layer_index_hull(v[0], v[1], m, n), -len(hit), g.kind, g.faces, v)
            if best is None or key < best[0]:
                best = (key, g, (v, d))

        if best is None:
            # final fallback
            cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
            chosen = None
            for v in cand:
                if knockable(G, v, m, n):
                    dirs = ray_clear_directions(G, v, m, n)
                    dirs.sort()
                    chosen = (v, dirs[0])
                    break
            if chosen is None:
                print("[LAYER-SUB] DEADLOCK: no ray-feasible knock at all.", flush=True)
                return False, plan, None, 0.0, 0.0, dict(stats)
            v, d = chosen
            step = PlanStep(v, "knock", d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            stats["fallback"] += 1
            record(step, "fallback")
            continue

        _, g, (v, d) = best
        step = PlanStep(v, "knock", d)
        plan.append(step)
        knocked.add(v)
        G.remove_node(v)
        stats["layer_knock"] += 1
        record(step, f"outermost kind={g.kind} d={d}")

    if visualize:
        visualize_solution(build_graph(occ), snaps, cover=None, title="Layer-subgraph: sequence", show_faces=False)

    return True, plan, None, 0.0, 0.0, dict(stats)


# ============================================================
# Planner: full-rectangle exact layer peeling (optimal)
# ============================================================

def ring_faces_cyclic(t: int, m: int, n: int) -> List[Face]:
    """
    Cyclic order of ring faces at offset t for a full rectangle.
    Faces live on (m-1)×(n-1) grid of top-left indices.

    Only valid for m-2t >= 3 and n-2t >= 3.
    """
    top_i = t
    bot_i = (m - 2) - t
    left_j = t
    right_j = (n - 2) - t

    faces: List[Face] = []
    # top row
    for j in range(left_j, right_j + 1):
        faces.append((top_i, j))
    # right col (excluding top)
    for i in range(top_i + 1, bot_i + 1):
        faces.append((i, right_j))
    # bottom row (excluding rightmost)
    for j in range(right_j - 1, left_j - 1, -1):
        faces.append((bot_i, j))
    # left col (excluding bottom+top)
    for i in range(bot_i - 1, top_i, -1):
        faces.append((i, left_j))
    return faces

def plan_layer_peel_full(m: int, n: int, debug: bool = True, visualize: bool = False):
    """
    Full-grid optimal layer peeling.
    Target optimal knocks = ceil((m-1)(n-1)/2).
    """
    occ = occ_full(m, n)
    G = build_graph(occ)

    plan: List[PlanStep] = []
    snaps: List[ExecSnapshot] = []
    picked: Set[Node] = set()
    knocked: Set[Node] = set()

    def record(last: Optional[PlanStep], note: str = ""):
        snaps.append(ExecSnapshot(set(G.nodes()), set(picked), set(knocked), last, note))

    record(None, "start")

    it = 0
    max_it = 80 * (m * n) + 500

    if debug:
        opt = math.ceil(((m - 1) * (n - 1)) / 2) if (m >= 2 and n >= 2) else 0
        print(f"=== RectLayerPeel OPT on full {m}x{n} |V|={m*n} opt_knocks={opt} ===", flush=True)

    while G.number_of_nodes() > 0:
        it += 1
        if it > max_it:
            print("[RECT-LAYER] guard triggered: too many iterations", flush=True)
            return False, plan, None, 0.0, 0.0, {}

        # cleanup picks
        while True:
            P = [u for u in G.nodes() if pickable(G, u)]
            if not P:
                break
            P.sort()
            u = P[0]
            step = PlanStep(u, "pick", None)
            plan.append(step)
            picked.add(u)
            G.remove_node(u)
            record(step, "cleanup")

        if G.number_of_nodes() == 0:
            break

        # current offset (for full rectangle, after peeling, min layer shifts)
        cur_nodes = list(G.nodes())
        t_off = min(layer_index_hull(i, j, m, n) for (i, j) in cur_nodes) - 1
        mr = m - 2 * t_off
        nr = n - 2 * t_off

        if debug:
            print(f"[RECT-LAYER] iter={it} remainingV={G.number_of_nodes()} offset={t_off} mr={mr} nr={nr}",
                  flush=True)

        # If no faces exist, we should be pick-only from here, but safe fallback:
        coreU = core_nodes(G)
        F = enumerate_faces(coreU, m, n)
        if not F:
            # fallback knock if needed
            cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
            chosen = None
            for v in cand:
                if knockable(G, v, m, n):
                    dirs = ray_clear_directions(G, v, m, n)
                    dirs.sort()
                    chosen = (v, dirs[0])
                    break
            if chosen is None:
                # if even fallback not possible, something is wrong; break
                print("[RECT-LAYER] WARNING: no faces and no knockable core; stopping", flush=True)
                break
            v, d = chosen
            step = PlanStep(v, "knock", d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            record(step, "fallback(no faces)")
            continue

        # Special small-dimension handling
        if mr == 2 and nr == 2:
            # single 2x2 face
            f = (t_off, t_off)
            g = FaceGadget("2x2", (f,), tuple(sorted(face_blocks(f))))
            pair = choose_knock_for_gadget(G, g, coreU, m, n)
            if pair is None:
                # fallback knock anywhere ray-feasible
                cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
                for v in cand:
                    dirs = ray_clear_directions(G, v, m, n)
                    if dirs:
                        dirs.sort()
                        pair = (v, dirs[0])
                        break
            v, d = pair
            step = PlanStep(v, "knock", d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            record(step, "2x2 base")
            continue

        # If mr==2 or nr==2: do scanline pairing on the face-row/col
        if mr == 2 and nr >= 3:
            # faces are (t_off, j) for j=t_off..n-2-t_off
            face_js = list(range(t_off, (n - 1) - t_off))  # length nr-1
            j = 0
            while j < len(face_js) - 1:
                f1 = (t_off, face_js[j])
                f2 = (t_off, face_js[j + 1])
                # 2x3 gadget top-left at (t_off, face_js[j])
                gi, gj = t_off, face_js[j]
                cells = [(gi + di, gj + dj) for di in range(2) for dj in range(3)]
                g = FaceGadget("2x3", tuple(sorted([f1, f2])), tuple(sorted(cells)))
                pair = choose_knock_for_gadget(G, g, coreU, m, n)
                if pair is None:
                    # fallback: pick any candidate with ray
                    for v in gadget_knock_candidates(g):
                        if v in coreU and G.has_node(v):
                            dirs = ray_clear_directions(G, v, m, n)
                            if dirs:
                                dirs.sort()
                                pair = (v, dirs[0])
                                break
                v, d = pair
                step = PlanStep(v, "knock", d)
                plan.append(step)
                knocked.add(v)
                G.remove_node(v)
                record(step, "mr==2 peel")
                # cleanup picks after each knock
                coreU = core_nodes(G)
                j += 2
            # leftover single face
            if j == len(face_js) - 1:
                f = (t_off, face_js[j])
                g2 = FaceGadget("2x2", (f,), tuple(sorted(face_blocks(f))))
                pair2 = choose_knock_for_gadget(G, g2, coreU, m, n)
                if pair2 is None:
                    for v in gadget_knock_candidates(g2):
                        if v in coreU and G.has_node(v):
                            dirs = ray_clear_directions(G, v, m, n)
                            if dirs:
                                dirs.sort()
                                pair2 = (v, dirs[0])
                                break
                v, d = pair2
                step = PlanStep(v, "knock", d)
                plan.append(step)
                knocked.add(v)
                G.remove_node(v)
                record(step, "mr==2 leftover")
            continue

        if nr == 2 and mr >= 3:
            face_is = list(range(t_off, (m - 1) - t_off))  # length mr-1
            i = 0
            while i < len(face_is) - 1:
                f1 = (face_is[i], t_off)
                f2 = (face_is[i + 1], t_off)
                gi, gj = face_is[i], t_off
                cells = [(gi + di, gj + dj) for di in range(3) for dj in range(2)]
                g = FaceGadget("3x2", tuple(sorted([f1, f2])), tuple(sorted(cells)))
                pair = choose_knock_for_gadget(G, g, coreU, m, n)
                if pair is None:
                    for v in gadget_knock_candidates(g):
                        if v in coreU and G.has_node(v):
                            dirs = ray_clear_directions(G, v, m, n)
                            if dirs:
                                dirs.sort()
                                pair = (v, dirs[0])
                                break
                v, d = pair
                step = PlanStep(v, "knock", d)
                plan.append(step)
                knocked.add(v)
                G.remove_node(v)
                record(step, "nr==2 peel")
                coreU = core_nodes(G)
                i += 2
            if i == len(face_is) - 1:
                f = (face_is[i], t_off)
                g2 = FaceGadget("2x2", (f,), tuple(sorted(face_blocks(f))))
                pair2 = choose_knock_for_gadget(G, g2, coreU, m, n)
                if pair2 is None:
                    for v in gadget_knock_candidates(g2):
                        if v in coreU and G.has_node(v):
                            dirs = ray_clear_directions(G, v, m, n)
                            if dirs:
                                dirs.sort()
                                pair2 = (v, dirs[0])
                                break
                v, d = pair2
                step = PlanStep(v, "knock", d)
                plan.append(step)
                knocked.add(v)
                G.remove_node(v)
                record(step, "nr==2 leftover")
            continue

        # General ring peeling for mr>=3,nr>=3
        ring = ring_faces_cyclic(t_off, m, n)
        if len(ring) % 2 != 0:
            # should not happen; safe
            ring = ring[:-1]

        # Pair consecutive faces
        for kpair in range(0, len(ring), 2):
            f1, f2 = ring[kpair], ring[kpair + 1]
            (i1, j1), (i2, j2) = f1, f2
            if i1 == i2 and abs(j1 - j2) == 1:
                gi, gj = i1, min(j1, j2)
                cells = [(gi + di, gj + dj) for di in range(2) for dj in range(3)]
                g = FaceGadget("2x3", tuple(sorted([f1, f2])), tuple(sorted(cells)))
            elif j1 == j2 and abs(i1 - i2) == 1:
                gi, gj = min(i1, i2), j1
                cells = [(gi + di, gj + dj) for di in range(3) for dj in range(2)]
                g = FaceGadget("3x2", tuple(sorted([f1, f2])), tuple(sorted(cells)))
            else:
                # shouldn't happen in cyclic ring, but guard: treat f1 as 2x2
                g = FaceGadget("2x2", (f1,), tuple(sorted(face_blocks(f1))))

            coreU = core_nodes(G)
            pair = choose_knock_for_gadget(G, g, coreU, m, n)
            if pair is None:
                # fallback: any ray-feasible candidate inside gadget
                for v in gadget_knock_candidates(g):
                    if v in coreU and G.has_node(v):
                        dirs = ray_clear_directions(G, v, m, n)
                        if dirs:
                            dirs.sort()
                            pair = (v, dirs[0])
                            break
            if pair is None:
                # ultimate fallback: any ray-feasible core node
                cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
                for v in cand:
                    if knockable(G, v, m, n):
                        dirs = ray_clear_directions(G, v, m, n)
                        dirs.sort()
                        pair = (v, dirs[0])
                        break

            v, d = pair
            step = PlanStep(v, "knock", d)
            plan.append(step)
            knocked.add(v)
            G.remove_node(v)
            record(step, f"ring kind={g.kind}")

            # cleanup picks after each knock to avoid stalling
            while True:
                P = [u for u in G.nodes() if pickable(G, u)]
                if not P:
                    break
                P.sort()
                u = P[0]
                st2 = PlanStep(u, "pick", None)
                plan.append(st2)
                picked.add(u)
                G.remove_node(u)
                record(st2, "cleanup-after-knock")

            if G.number_of_nodes() == 0:
                break

    if visualize:
        visualize_solution(build_graph(occ), snaps, cover=None, title=f"RectLayerPeel full {m}x{n}", show_faces=False)

    return True, plan, None, 0.0, 0.0, {}


# ============================================================
# Visualization (optional)
# ============================================================

def gadget_patches(cover: List[FaceGadget]):
    patches = []
    for g in cover:
        cells = list(g.cells)
        is_ = [u[0] for u in cells]
        js_ = [u[1] for u in cells]
        i0, j0 = min(is_), min(js_)

        if g.kind == "2x2":
            f = g.faces[0]
            fi, fj = f
            cx, cy = fj + 0.5, -(fi + 0.5)
            s = 0.45
            patches.append(Rectangle((cx - s / 2, cy - s / 2), s, s))
        elif g.kind == "2x3":
            cx, cy = (j0 + 1.0), -(i0 + 0.5)
            w = 1.00
            h = 0.50
            patches.append(Rectangle((cx - w / 2, cy - h / 2), w, h))
        elif g.kind == "3x2":
            cx, cy = (j0 + 0.5), -(i0 + 1.0)
            w = 0.50
            h = 1.00
            patches.append(Rectangle((cx - w / 2, cy - h / 2), w, h))
    return patches

def visualize_solution(G_full: nx.Graph, snaps: List[ExecSnapshot], cover: Optional[List[FaceGadget]],
                       title: str = "", show_faces: bool = False, pause: float = 0.20):
    pos = nx.get_node_attributes(G_full, "pos")
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))

    cover_markers = []
    if cover:
        cover_markers = gadget_patches(cover)
        for p in cover_markers:
            p.set_facecolor("grey")
            p.set_edgecolor("none")
            p.set_alpha(0.22)

    for t, st in enumerate(snaps):
        ax.clear()

        for p in cover_markers:
            ax.add_patch(p)

        nx.draw_networkx_edges(G_full, pos=pos, ax=ax, alpha=0.10)

        rem = list(st.remaining)
        pk = list(st.picked)
        kn = list(st.knocked)

        if rem:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=rem, ax=ax, node_size=320, alpha=0.70)
        if pk:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=pk, ax=ax, node_size=220, node_color="green", alpha=0.85)
        if kn:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=kn, ax=ax, node_size=220, node_color="red",
                                   alpha=0.85, node_shape="s")

        if st.last_step is None:
            ax.set_title(f"{title} step {t}")
        else:
            s = st.last_step
            ax.set_title(f"{title} step {t}: {s.kind} {s.node} {st.note}")

        # annotate step indices
        for tt in range(t + 1):
            past = snaps[tt]
            if past.last_step is None:
                continue
            u = past.last_step.node
            act = past.last_step.kind
            if u not in pos:
                continue
            x, y = pos[u]
            ax.text(
                x + 0.08, y + 0.08, str(tt),
                fontsize=11, fontweight="bold",
                color=("green" if act == "pick" else "red"),
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                zorder=10,
            )

        ax.set_axis_off()
        plt.pause(pause)

    plt.ioff()
    plt.show()


# ============================================================
# IsaacSim YAML generator
# ============================================================

GOAL_PAD_POS = {
    "Red":   (0.790, 0.0),
    "Green": (0.790, 0.15),
    "Blue":  (0.790, -0.15),
}

def compute_initial_centers(m: int, n: int, size: float, gap_s: float, cx: float, cy: float) -> List[Tuple[float, float]]:
    """m×n grid around (cx,cy) with edge gap gap_s."""
    pitch = size + gap_s
    xs = [cx + (j - (n - 1) / 2.0) * pitch for j in range(n)]
    ys = [cy + ((m - 1) / 2.0 - i) * pitch for i in range(m)]
    return [(xs[j], ys[i]) for i in range(m) for j in range(n)]

def make_names(k: int) -> List[str]:
    colors = ["Red", "Green", "Blue"]
    return [f"{colors[i % 3]}{i // 3 + 1}" for i in range(k)]

def fmt(x: float, prec: int) -> str:
    return f"{x:.{prec}f}"

def node_to_pad_id(i: int, j: int, n: int) -> int:
    # row-major: i=0 top row
    return i * n + j + 1

def block_color(name: str) -> str:
    if name.startswith("Red"):
        return "Red"
    if name.startswith("Green"):
        return "Green"
    if name.startswith("Blue"):
        return "Blue"
    return "Red"

def write_problem_yaml(
    out_path: str,
    m: int,
    n: int,
    occ: Set[Node],
    size: float,
    gap_s: float,
    center_x: float,
    center_y: float,
    plan: List[PlanStep],
    prec: int = 3,
) -> None:
    """
    Write a problem.yaml that includes:
      pads (m*n initial pads + 3 goal pads),
      z_base / z_increment,
      initial / goal,
      actions (Smash+Move for knock, Move for pick).

    We assign block names to occupied nodes in row-major order.
    """
    # pads
    init_centers = compute_initial_centers(m, n, size, gap_s, center_x, center_y)
    k_pads = m * n
    goal_red_id = k_pads + 1
    goal_green_id = k_pads + 2
    goal_blue_id = k_pads + 3
    goal_pad_id = {"Red": goal_red_id, "Green": goal_green_id, "Blue": goal_blue_id}

    # block naming: only for occupied nodes, row-major
    occ_sorted = sorted(list(occ))
    names = make_names(len(occ_sorted))
    node_name: Dict[Node, str] = {u: names[idx] for idx, u in enumerate(occ_sorted)}
    name_node: Dict[str, Node] = {names[idx]: occ_sorted[idx] for idx in range(len(occ_sorted))}

    # initial pad mapping for each block
    block_pad: Dict[str, int] = {}
    for u, name in node_name.items():
        i, j = u
        block_pad[name] = node_to_pad_id(i, j, n)

    zb = size / 2.0
    zi = size

    lines: List[str] = []
    lines.append("pads:")
    # initial pads 1..m*n
    for pid in range(1, k_pads + 1):
        x, y = init_centers[pid - 1]
        lines.append(f"  {pid}: {{x: {fmt(x,prec)}, y: {fmt(y,prec)}}}")
    # 3 goal pads
    xr, yr = GOAL_PAD_POS["Red"]
    xg, yg = GOAL_PAD_POS["Green"]
    xb, yb = GOAL_PAD_POS["Blue"]
    lines.append(f"  {goal_red_id}: {{x: {fmt(xr,prec)}, y: {fmt(yr,prec)}}}")
    lines.append(f"  {goal_green_id}: {{x: {fmt(xg,prec)}, y: {fmt(yg,prec)}}}")
    lines.append(f"  {goal_blue_id}: {{x: {fmt(xb,prec)}, y: {fmt(yb,prec)}}}")

    lines.append("")
    lines.append(f"z_base:      {fmt(zb,prec)}       # table + 1/2 block")
    lines.append(f"z_increment: {fmt(zi,prec)}     # full block height")
    lines.append("# ----------------------------------------------------------")
    lines.append("")

    # initial list
    lines.append("initial:")
    for u in occ_sorted:
        name = node_name[u]
        pad = block_pad[name]
        lines.append(f"  - {{name: {name}, pad: {pad}, height: 0}}")

    # goal list: all red->red pad, etc
    lines.append("")
    lines.append("goal:")
    for u in occ_sorted:
        name = node_name[u]
        col = block_color(name)
        gpid = goal_pad_id[col]
        lines.append(f"  - {{name: {name}, pad: {gpid}, height: 0}}")

    # actions
    lines.append("")
    lines.append("actions:")
    for step in plan:
        name = node_name.get(step.node, None)
        if name is None:
            # should never happen (each node removed once)
            continue
        from_pad = block_pad[name]
        col = block_color(name)
        to_pad = goal_pad_id[col]

        if step.kind == "knock":
            dstr = dir_to_isaac(step.direction if step.direction else (0, 1))
            lines.append("  - type:  Smash")
            lines.append(f"    blocks: [{name}]")
            lines.append(f"    from: {{pad: {from_pad}, height: 0}}")
            lines.append(f"    dir:   {dstr}")
            lines.append("")
            lines.append("  - type:  Move")
            lines.append(f"    block: {name}")
            lines.append(f"    from: {{pad: {from_pad}, height: 0}}")
            lines.append(f"    to: {{pad: {to_pad}, height: 0}}")
            lines.append("")
            block_pad[name] = to_pad
        else:
            lines.append("  - type:  Move")
            lines.append(f"    block: {name}")
            lines.append(f"    from: {{pad: {from_pad}, height: 0}}")
            lines.append(f"    to: {{pad: {to_pad}, height: 0}}")
            lines.append("")
            block_pad[name] = to_pad

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")


# ============================================================
# Benchmark helpers
# ============================================================

def summarize_plan(plan: List[PlanStep]) -> Dict[str, int]:
    knocks = sum(1 for s in plan if s.kind == "knock")
    picks = sum(1 for s in plan if s.kind == "pick")
    steps = len(plan)
    primitive_actions = picks + 2 * knocks  # IsaacSim: knock = Smash + Move, pick = Move
    return {"knocks": knocks, "picks": picks, "steps": steps, "primitive_actions": primitive_actions}

def run_one_instance(occ: Set[Node], m: int, n: int, alg: str, debug: bool, visualize: bool):
    """
    alg:
      - "greedy"
      - "layer_full"  (only for full grids)
      - "layer_sub"   (heuristic)
    """
    if alg == "greedy":
        ok, plan, cover, t_cover, t_exec, stats = plan_greedy(occ, m, n, debug=debug, visualize=visualize)
        return ok, plan, t_cover, t_exec, stats
    if alg == "layer_full":
        # exact for full grid
        ok, plan, _, _, _, stats = plan_layer_peel_full(m, n, debug=debug, visualize=visualize)
        return ok, plan, 0.0, 0.0, stats
    if alg == "layer_sub":
        ok, plan, _, _, _, stats = plan_layer_subgraph(occ, m, n, debug=debug, visualize=visualize)
        return ok, plan, 0.0, 0.0, stats
    raise ValueError(f"Unknown alg={alg}")

def print_row(tag: str, metrics: Dict[str, float]):
    print(
        f"{tag:32s} | knocks={int(metrics['knocks']):4d} steps={int(metrics['steps']):4d} prim={int(metrics['primitive_actions']):4d} "
        f"| t_cover={metrics['t_cover_ms']:8.2f}ms t_exec={metrics['t_exec_ms']:8.2f}ms t_total={metrics['t_total_ms']:8.2f}ms",
        flush=True
    )


# ============================================================
# Synthetic benchmark
# ============================================================

FULL_CASES = [(3, 3), (5, 5), (10, 5), (10, 10), (20, 10), (20, 20)]
# You wrote: 5,16,25,50,100,20 correspondingly (for 9,25,50,100,200,400).
# Keep exactly as you said; change here if you meant 200 instead of 20.
SUBSET_SIZES = [5, 16, 25, 50, 100, 20]

def run_synthetic_benchmark(debug: bool, visualize: bool, seed: int = 0):
    rng = random.Random(seed)

    print("\n==================== SYNTHETIC BENCHMARK ====================", flush=True)

    # Full grids: run exact and greedy once each
    for (m, n) in FULL_CASES:
        occ = occ_full(m, n)

        print(f"\n--- FULL {m}x{n} ---", flush=True)

        # exact layer peeling
        t0 = time.perf_counter()
        ok1, plan1, _, _, _ = run_one_instance(occ, m, n, "layer_full", debug=debug, visualize=visualize)
        t_total = (time.perf_counter() - t0)
        s1 = summarize_plan(plan1)
        row1 = {
            **s1,
            "t_cover_ms": 0.0,
            "t_exec_ms": 0.0,
            "t_total_ms": 1000.0 * t_total,
        }
        print_row(f"full {m}x{n} (layer_full)", row1)

        # greedy
        t1 = time.perf_counter()
        ok2, plan2, t_cover, t_exec, stats2 = run_one_instance(occ, m, n, "greedy", debug=debug, visualize=visualize)
        t_total2 = (time.perf_counter() - t1)
        s2 = summarize_plan(plan2)
        row2 = {
            **s2,
            "t_cover_ms": 1000.0 * t_cover,
            "t_exec_ms": 1000.0 * t_exec,
            "t_total_ms": 1000.0 * t_total2,
        }
        print_row(f"full {m}x{n} (greedy)", row2)

    # Subgraphs: 20 random subsets each size, run greedy and average
    print("\n==================== SUBGRAPH (RANDOM SUBSETS) ====================", flush=True)
    for idx, (m, n) in enumerate(FULL_CASES):
        k = SUBSET_SIZES[idx]
        trials = 20
        print(f"\n--- SUBGRAPH from {m}x{n} |k|={k} trials={trials} ---", flush=True)

        acc = collections.Counter()
        acc_time_cover = 0.0
        acc_time_exec = 0.0
        acc_time_total = 0.0

        for t in range(trials):
            occ = random_subgraph_with_forced_face(m, n, k, rng)

            # greedy
            t0 = time.perf_counter()
            ok, plan, t_cover, t_exec, stats = run_one_instance(occ, m, n, "greedy", debug=False, visualize=False)
            t_total = time.perf_counter() - t0
            s = summarize_plan(plan)
            acc.update(s)
            acc_time_cover += t_cover
            acc_time_exec += t_exec
            acc_time_total += t_total

        avg = {
            "knocks": acc["knocks"] / trials,
            "picks": acc["picks"] / trials,
            "steps": acc["steps"] / trials,
            "primitive_actions": acc["primitive_actions"] / trials,
            "t_cover_ms": 1000.0 * (acc_time_cover / trials),
            "t_exec_ms": 1000.0 * (acc_time_exec / trials),
            "t_total_ms": 1000.0 * (acc_time_total / trials),
        }
        print(
            f"sub {m}x{n} k={k:4d} (greedy avg) | knocks={avg['knocks']:.2f} steps={avg['steps']:.2f} prim={avg['primitive_actions']:.2f} "
            f"| t_cover={avg['t_cover_ms']:.2f}ms t_exec={avg['t_exec_ms']:.2f}ms t_total={avg['t_total_ms']:.2f}ms",
            flush=True
        )


# ============================================================
# IsaacSim problem generation
# ============================================================

def generate_isaac_problems(
    outdir: str,
    debug: bool,
    visualize: bool,
    seed: int = 0,
    size: float = 0.05,
    gap_s: float = 0.005,
    center_x: float = 0.50,
    center_y: float = 0.00,
    prec: int = 3,
):
    rng = random.Random(seed)
    os.makedirs(outdir, exist_ok=True)

    print("\n==================== ISAACSIM PROBLEM GENERATION ====================", flush=True)

    sizes = [(3, 3), (5, 5)]
    # Use your synthetic subset sizes for these: 3x3 -> 5, 5x5 -> 16
    sub_k = {(3, 3): 5, (5, 5): 16}

    # 1) full grid + layer peeling: 2 files
    for (m, n) in sizes:
        occ = occ_full(m, n)
        print(f"\n[ISAAC] full {m}x{n} + layer_full", flush=True)
        ok, plan, _, _, _ = run_one_instance(occ, m, n, "layer_full", debug=debug, visualize=visualize)
        out = os.path.join(outdir, f"full_{m}x{n}_layer.yaml")
        write_problem_yaml(out, m, n, occ, size, gap_s, center_x, center_y, plan, prec=prec)
        print(f"  wrote: {out}", flush=True)

    # 3) full grid + greedy: 2 files
    for (m, n) in sizes:
        occ = occ_full(m, n)
        print(f"\n[ISAAC] full {m}x{n} + greedy", flush=True)
        ok, plan, _, _, _ = run_one_instance(occ, m, n, "greedy", debug=debug, visualize=visualize)
        out = os.path.join(outdir, f"full_{m}x{n}_greedy.yaml")
        write_problem_yaml(out, m, n, occ, size, gap_s, center_x, center_y, plan, prec=prec)
        print(f"  wrote: {out}", flush=True)

    # 2) subgraph + greedy: 5 each size
    for (m, n) in sizes:
        k = sub_k[(m, n)]
        for sidx in range(5):
            occ = random_subgraph_with_forced_face(m, n, k, rng)
            print(f"\n[ISAAC] sub {m}x{n} k={k} seed={seed}#{sidx} + greedy", flush=True)
            ok, plan, _, _, _ = run_one_instance(occ, m, n, "greedy", debug=debug, visualize=visualize)
            # ensure at least one knock (forced by construction, but double-check)
            if summarize_plan(plan)["knocks"] == 0:
                print("  WARNING: got 0 knocks unexpectedly; regenerating once", flush=True)
                occ = random_subgraph_with_forced_face(m, n, k, rng)
                ok, plan, _, _, _ = run_one_instance(occ, m, n, "greedy", debug=debug, visualize=visualize)
            out = os.path.join(outdir, f"sub_{m}x{n}_k{k}_greedy_{sidx}.yaml")
            write_problem_yaml(out, m, n, occ, size, gap_s, center_x, center_y, plan, prec=prec)
            print(f"  wrote: {out}", flush=True)

    # 4) subgraph + layer heuristic: 5 each size
    for (m, n) in sizes:
        k = sub_k[(m, n)]
        for sidx in range(5):
            occ = random_subgraph_with_forced_face(m, n, k, rng)
            print(f"\n[ISAAC] sub {m}x{n} k={k} seed={seed}#{sidx} + layer_sub", flush=True)
            ok, plan, _, _, _ = run_one_instance(occ, m, n, "layer_sub", debug=debug, visualize=visualize)
            if summarize_plan(plan)["knocks"] == 0:
                print("  WARNING: got 0 knocks unexpectedly; regenerating once", flush=True)
                occ = random_subgraph_with_forced_face(m, n, k, rng)
                ok, plan, _, _, _ = run_one_instance(occ, m, n, "layer_sub", debug=debug, visualize=visualize)
            out = os.path.join(outdir, f"sub_{m}x{n}_k{k}_layer_{sidx}.yaml")
            write_problem_yaml(out, m, n, occ, size, gap_s, center_x, center_y, plan, prec=prec)
            print(f"  wrote: {out}", flush=True)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=str, default="synthetic", choices=["isaac", "synthetic"],
                    help="Run IsaacSim YAML generation or synthetic benchmark.")
    ap.add_argument("--outdir", type=str, default="problems", help="Output directory for IsaacSim YAMLs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visualize", action="store_true", help="Visualize each planned instance (matplotlib).")
    ap.add_argument("--debug", action="store_true", help="Enable debug prints.")
    # Isaac parameters
    ap.add_argument("--size", type=float, default=0.05)
    ap.add_argument("--gap_s", type=float, default=0.005)
    ap.add_argument("--center_x", type=float, default=0.50)
    ap.add_argument("--center_y", type=float, default=0.00)
    ap.add_argument("--precision", type=int, default=3)
    args = ap.parse_args()

    if args.bench == "synthetic":
        run_synthetic_benchmark(debug=args.debug, visualize=args.visualize, seed=args.seed)
    else:
        generate_isaac_problems(
            outdir=args.outdir,
            debug=args.debug,
            visualize=args.visualize,
            seed=args.seed,
            size=args.size,
            gap_s=args.gap_s,
            center_x=args.center_x,
            center_y=args.center_y,
            prec=args.precision,
        )

if __name__ == "__main__":
    main()
