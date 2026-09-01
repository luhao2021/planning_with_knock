#!/usr/bin/env python3

"""Optimal knock-pick planner and benchmark entry point.

This module implements the paper's graph abstraction without modeling robot
motion: feasible picks clean the object-location graph, a maximum-cardinality
matching selects a minimum gadget cover, and the cover is converted into a
ray-feasible sequence of knocks interleaved with exhaustive picks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict, Iterable
import argparse
import collections
import random
import os
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

Node = Tuple[int, int]   # (i,j)
Face = Tuple[int, int]   # (i,j) top-left of a 2x2 face

DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # +i, -i, +j, -j


# ============================================================
# Occupancy helpers
# ============================================================

def occ_full(m: int, n: int) -> Set[Node]:
    return {(i, j) for i in range(m) for j in range(n)}

def apply_removals(occ: Set[Node], rects: List[Tuple[int, int, int, int]]) -> Set[Node]:
    """Remove rectangles [i0,i1)×[j0,j1) from occ."""
    out = set(occ)
    for (i0, j0, i1, j1) in rects:
        for i in range(i0, i1):
            for j in range(j0, j1):
                out.discard((i, j))
    return out

def occ_concave_L(m: int, n: int, cut_i0: int, cut_j0: int, cut_i1: int, cut_j1: int) -> Set[Node]:
    occ = occ_full(m, n)
    return apply_removals(occ, [(cut_i0, cut_j0, cut_i1, cut_j1)])

def occ_with_hole(m: int, n: int, hole_i0: int, hole_j0: int, hole_i1: int, hole_j1: int) -> Set[Node]:
    occ = occ_full(m, n)
    return apply_removals(occ, [(hole_i0, hole_j0, hole_i1, hole_j1)])


# ============================================================
# Graph construction + hull-layer geometry
# ============================================================

def build_graph(
    occ: Set[Node],
    node_heights: Optional[Dict[Node, float]] = None,
) -> nx.Graph:
    """Build the height-agnostic occupancy graph and attach cuboid heights.

    Edges always represent planar four-neighbor occupancy and are therefore
    the original degrees used for knocking. Picking computes a separate,
    asymmetric degree from the per-node ``height`` attributes.
    """
    node_heights = node_heights or {}
    G = nx.Graph()
    for (i, j) in occ:
        # position: x=j, y=-i (so i grows downward)
        height = float(node_heights.get((i, j), 1.0))
        if height <= 0.0:
            raise ValueError(f"node {(i, j)} has non-positive height {height}")
        G.add_node((i, j), pos=(float(j), float(-i)), height=height)
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

def height_aware_neighbors(G: nx.Graph, u: Node) -> List[Node]:
    """Neighbors reaching the candidate's height and therefore blocking it."""
    candidate_height = float(G.nodes[u].get("height", 1.0))
    return [
        v for v in G.neighbors(u)
        if float(G.nodes[v].get("height", 1.0)) >= candidate_height
    ]


def height_aware_degree(G: nx.Graph, u: Node) -> int:
    """Picking degree at the horizontal slice defined by ``u``'s height."""
    return len(height_aware_neighbors(G, u))


def is_I_shape_degree_two(G: nx.Graph, u: Node) -> bool:
    if height_aware_degree(G, u) != 2:
        return False
    v1, v2 = height_aware_neighbors(G, u)
    x1, y1 = G.nodes[v1]["pos"]
    x2, y2 = G.nodes[v2]["pos"]
    return (x1 == x2) or (y1 == y2)

def pickable(G: nx.Graph, u: Node) -> bool:
    d = height_aware_degree(G, u)
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

def face_center_pos(f: Face) -> Tuple[float, float]:
    i, j = f
    return (j + 0.5, -(i + 0.5))

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

    # 2x2 gadgets
    for f in Fset:
        add("2x2", [f], list(face_blocks(f)))

    # 2x3 gadgets
    if n >= 3 and m >= 2:
        for i in range(m - 1):
            for j in range(n - 2):
                f1, f2 = (i, j), (i, j + 1)
                if f1 in Fset and f2 in Fset:
                    cells = [(i + di, j + dj) for di in range(2) for dj in range(3)]
                    if all(c in Uset for c in cells):
                        add("2x3", [f1, f2], cells)

    # 3x2 gadgets
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
# STRICT knock candidates per gadget kind (your requirement)
# ============================================================

def gadget_knock_candidates(g: FaceGadget) -> List[Node]:
    """
    - 2x2: any of 4 nodes
    - 2x3: ONLY two middle-column nodes
    - 3x2: ONLY two middle-row nodes
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

def choose_knock_for_gadget(G: nx.Graph, g: FaceGadget, coreU: Set[Node], m: int, n: int) -> Optional[Tuple[Node, Tuple[int, int]]]:
    """
    Choose (v,d) for this gadget:
      - candidates restricted by gadget kind (middle nodes for 2x3/3x2),
      - require degree<=3 and some Ray direction feasible,
      - prefer outer layers first (smaller hull-layer index),
      - tie-break by higher degree then coordinates.
    """
    cand0 = [v for v in gadget_knock_candidates(g) if (v in coreU) and G.has_node(v)]
    if not cand0:
        return None

    cand0.sort(key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
    for v in cand0:
        if G.degree(v) <= 3:
            dirs = ray_clear_directions(G, v, m, n)
            if dirs:
                dirs.sort()
                return v, dirs[0]
    return None


# ============================================================
# Maximum Matching cover on face graph
# ============================================================

def build_face_adjacency_graph_and_map(gadgets: List[FaceGadget]) -> Tuple[nx.Graph, Dict[Tuple[Face, Face], FaceGadget]]:
    """
    Create graph on faces using ONLY 2-face gadgets (2x3/3x2).
    Also return a map (f,g)->gadget to convert matching edges back to gadgets.
    """
    H = nx.Graph()
    edge2g: Dict[Tuple[Face, Face], FaceGadget] = {}

    # Collect faces as nodes
    all_faces: Set[Face] = set()
    for g in gadgets:
        for f in g.faces:
            all_faces.add(f)
    H.add_nodes_from(all_faces)

    for g in gadgets:
        if len(g.faces) != 2:
            continue
        f1, f2 = g.faces
        H.add_edge(f1, f2)
        key = tuple(sorted((f1, f2)))
        edge2g[key] = g

    return H, edge2g

def maximum_matching_cover_faces(F: Set[Face], gadgets: List[FaceGadget], debug: bool = True) -> Tuple[List[FaceGadget], Set[Tuple[Face, Face]], Set[Face]]:
    """
    Compute maximum matching on face-adjacency graph, then cover:
      - each matched edge => its 2-face gadget (2x3/3x2),
      - each unmatched face => 2x2 gadget on that face.
    Returns:
      cover list,
      matching edge set as sorted pairs,
      unmatched face set
    """
    # Build adjacency graph among faces using 2-face gadgets only
    H, edge2g = build_face_adjacency_graph_and_map(gadgets)
    # Ensure all faces in F exist as nodes
    for f in F:
        if f not in H:
            H.add_node(f)

    # Maximum cardinality matching
    M = nx.algorithms.matching.max_weight_matching(H, maxcardinality=True)
    matching_edges: Set[Tuple[Face, Face]] = set(tuple(sorted((a, b))) for (a, b) in M)

    matched_faces: Set[Face] = set()
    for a, b in matching_edges:
        matched_faces.add(a)
        matched_faces.add(b)

    unmatched = set(F) - matched_faces

    cover: List[FaceGadget] = []
    # Matched edges -> gadgets
    for a, b in sorted(list(matching_edges)):
        key = tuple(sorted((a, b)))
        g = edge2g.get(key, None)
        if g is None:
            # Safety: if somehow missing, fallback to reconstruct union (should not happen)
            cells = sorted(list(face_blocks(a) | face_blocks(b)))
            # Determine kind by relative position
            (ia, ja), (ib, jb) = a, b
            if ia == ib and abs(ja - jb) == 1:
                kind = "2x3"
            elif ja == jb and abs(ia - ib) == 1:
                kind = "3x2"
            else:
                kind = "2x3"
            g = FaceGadget(kind=kind, faces=tuple(sorted((a, b))), cells=tuple(cells))
        cover.append(g)

    # Unmatched vertices -> 2x2 gadgets
    for f in sorted(list(unmatched)):
        cover.append(FaceGadget(kind="2x2", faces=(f,), cells=tuple(sorted(face_blocks(f)))))

    if debug:
        print(f"[MATCH] |F|={len(F)} |E_F|={H.number_of_edges()} |M|={len(matching_edges)} => |C|=|F|-|M|={len(F)-len(matching_edges)}")
        if len(cover) != len(F) - len(matching_edges):
            print(f"[MATCH] WARNING: cover size {len(cover)} != |F|-|M| {len(F)-len(matching_edges)}")

    # Sort cover for more stable execution: outer faces first, then prefer 2-face gadgets
    def _cover_sort_key(g: FaceGadget) -> Tuple[int, int, Tuple]:
        minL = min(face_layer(f, 10**9, 10**9) for f in g.faces)  # dummy; not used
        # We'll re-sort later in execution using current m,n if needed; keep stable here:
        # prefer multi-face gadgets
        return (len(g.faces), g.faces)
    cover.sort(key=_cover_sort_key)

    return cover, matching_edges, unmatched


# ============================================================
# Execution dataclass (for visualization)
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

@dataclass
class ExecStep:
    remaining: Set[Node]
    picked: Set[Node]
    knocked: Set[Node]
    last_action: Optional[PlanStep]
    note: str = ""


# ============================================================
# Cleanup picks helper (deterministic)
# ============================================================

def _cleanup_picks(G: nx.Graph,
                   actions: List[PlanStep],
                   picked: Set[Node],
                   hist: List[ExecStep],
                   knocked: Set[Node],
                   note_prefix: str = "") -> None:
    while True:
        P = [u for u in G.nodes() if pickable(G, u)]
        if not P:
            break
        P.sort()
        u = P[0]
        step = PlanStep(node=u, kind="pick", direction=None)
        actions.append(step)
        picked.add(u)
        G.remove_node(u)
        hist.append(ExecStep(remaining=set(G.nodes()),
                             picked=set(picked),
                             knocked=set(knocked),
                             last_action=step,
                             note=f"{note_prefix}cleanup"))

# ============================================================
# Executor: prefer OUTERMOST faces + Ray feasibility
#   (and clarify "used": unused/used/inactive)
# ============================================================

def execute_from_cover(
    G_full: nx.Graph,
    cover: List[FaceGadget],
    m: int,
    n: int,
    debug: bool = True,
) -> Tuple[bool, List[PlanStep], List[ExecStep], List[bool], Dict[str, int]]:
    """
    Loop:
      1) Cleanup picks
      2) Compute core + faces
      3) Choose executable UNUSED cover gadget
      4) Else try extra / fallback (kept as in your pipeline)
    """
    G = G_full.copy()
    picked: Set[Node] = set()
    knocked: Set[Node] = set()
    actions: List[PlanStep] = []
    hist: List[ExecStep] = []
    used = [False] * len(cover)
    stats = collections.Counter()

    def record(last: Optional[PlanStep], note: str = "") -> None:
        hist.append(ExecStep(remaining=set(G.nodes()),
                             picked=set(picked),
                             knocked=set(knocked),
                             last_action=last,
                             note=note))

    record(None, "start")

    it = 0
    max_it = 80 * max(1, G.number_of_nodes()) + 500

    while G.number_of_nodes() > 0:
        it += 1
        if it > max_it:
            print("[EXEC] guard triggered: too many iterations (deadlock/bug).")
            return False, actions, hist, used, dict(stats)

        # cleanup picks
        while True:
            P = [u for u in G.nodes() if pickable(G, u)]
            if not P:
                break
            P.sort()
            u = P[0]
            step = PlanStep(node=u, kind="pick", direction=None)
            actions.append(step)
            picked.add(u)
            G.remove_node(u)
            record(step, "cleanup")

        if G.number_of_nodes() == 0:
            break

        coreU = core_nodes(G)
        F = enumerate_faces(coreU, m, n)
        outerL = min((face_layer(f, m, n) for f in F), default=None)
        outerF = {f for f in F if outerL is not None and face_layer(f, m, n) == outerL}

        if debug:
            coreE = G.subgraph(coreU).number_of_edges()
            print(f"[EXEC] iter={it} remainingV={G.number_of_nodes()} coreV={len(coreU)} coreE={coreE} faces={len(F)} outerL={outerL} used={sum(used)}/{len(cover)}")

        if not coreU:
            continue

        # If core exists but no faces: in theory should not happen for your "clean" instances.
        # For robustness in benchmarking, treat as failure.
        if not F:
            print("[EXEC] ERROR: core nonempty but no faces; instance outside FACE-gadget model.")
            return False, actions, hist, used, dict(stats)

        # executable UNUSED cover gadget
        next_knockable = None
        for idx, g in enumerate(cover):
            if used[idx]:
                continue
            pair = choose_knock_for_gadget(G, g, coreU, m, n)
            if pair is None:
                continue

            v, d = pair
            used[idx] = True
            step = PlanStep(node=v, kind="knock", direction=d)
            actions.append(step)
            knocked.add(v)
            G.remove_node(v)
            stats["cover_knock"] += 1
            record(step, f"cover gid={idx} kind={cover[idx].kind} d={d}")
            next_knockable = (idx, (v, d))
            break

        if next_knockable is not None:
            continue

        # If no executable unused gadget, try EXTRA gadget on current outermost faces
        if debug:
            print("[EXEC] no executable UNUSED cover gadget; searching EXTRA gadget on current core...")

        cur_gadgets = enumerate_face_gadgets(coreU, F, m, n)
        extra_best = None  # (key, gadget, (v,d))
        for g in cur_gadgets:
            hit = set(g.faces) & outerF
            if not hit:
                continue
            pair = choose_knock_for_gadget(G, g, coreU, m, n)
            if pair is None:
                continue
            v, d = pair
            key = (layer_index_hull(v[0], v[1], m, n), -len(hit), -G.degree(v), g.kind, g.faces)
            if extra_best is None or key < extra_best[0]:
                extra_best = (key, g, pair)

        if extra_best is not None:
            _, g, (v, d) = extra_best
            step = PlanStep(node=v, kind="knock", direction=d)
            actions.append(step)
            knocked.add(v)
            G.remove_node(v)
            stats["extra"] += 1
            record(step, f"extra kind={g.kind} d={d}")
            continue

        # final fallback: any ray-feasible core node
        cand = sorted(list(coreU), key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
        chosen = None
        for v in cand:
            if knockable(G, v, m, n):
                dirs = ray_clear_directions(G, v, m, n)
                dirs.sort()
                chosen = (v, dirs[0])
                break
        if chosen is None:
            print("[EXEC] DEADLOCK: no ray-feasible knock at all.")
            return False, actions, hist, used, dict(stats)

        v, d = chosen
        step = PlanStep(node=v, kind="knock", direction=d)
        actions.append(step)
        knocked.add(v)
        G.remove_node(v)
        stats["fallback"] += 1
        record(step, f"d={d}")

    if debug:
        knocks = sum(1 for s in actions if s.kind == "knock")
        picks = sum(1 for s in actions if s.kind == "pick")
        print(f"[EXEC DONE] knocks={knocks} picks={picks} steps={len(actions)} used={sum(used)}/{len(cover)} stats={dict(stats)}")

    return True, actions, hist, used, dict(stats)


# ============================================================
# Sanity replay
# ============================================================

def replay_sanity(
    occ: Set[Node],
    m: int,
    n: int,
    actions: List[PlanStep],
    verbose_fail: bool = True,
    node_heights: Optional[Dict[Node, float]] = None,
) -> bool:
    G = build_graph(occ, node_heights=node_heights)
    for t, step in enumerate(actions):
        u = step.node
        act = step.kind
        d = step.direction
        if not G.has_node(u):
            if verbose_fail:
                print(f"[REPLAY FAIL] step {t}: node {u} missing")
            return False
        if act == "pick":
            if not pickable(G, u):
                if verbose_fail:
                    print(
                        f"[REPLAY FAIL] step {t}: pick infeasible at {u}, "
                        f"height_deg={height_aware_degree(G, u)} "
                        f"occupancy_deg={G.degree(u)}"
                    )
                return False
            G.remove_node(u)
        else:
            if not knockable(G, u, m, n):
                if verbose_fail:
                    print(f"[REPLAY FAIL] step {t}: knock infeasible at {u}, {d}, deg={G.degree(u)} act={act}")
                    print("  ray dirs:", ray_clear_directions(G, u, m, n))
                return False
            G.remove_node(u)
    ok = (G.number_of_nodes() == 0)
    if not ok and verbose_fail:
        print("[REPLAY FAIL] ended with remaining nodes:", G.number_of_nodes())
    return ok


# ============================================================
# Visualization helpers
# ============================================================

def gadget_patches(cover: List[FaceGadget]):
    """
    Visualization rule (your spec):
      - 2x2: small square at face center.
      - 2x3 / 3x2: small rectangle centered at middle point between two middle nodes.
    """
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

def visualize_two_panel(
    G_full: nx.Graph,
    hist: List[ExecStep],
    cover: List[FaceGadget],
    m: int,
    n: int,
    F0: Set[Face],
    H_face: nx.Graph,
    matching_edges: Set[Tuple[Face, Face]],
    unmatched_faces: Set[Face],
    pause: float = 0.20,
    title: str = "",
) -> None:
    """
    Two panels:
      Left: block graph animation + cover markers + action indices
      Right: face adjacency graph with matching edges highlighted + unmatched faces highlighted
    """
    pos_nodes = nx.get_node_attributes(G_full, "pos")
    pos_faces = {f: face_center_pos(f) for f in H_face.nodes()}

    plt.ion()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(22, 11))

    # Draw right panel once (static)
    axR.clear()
    nx.draw_networkx_edges(H_face, pos=pos_faces, ax=axR, alpha=0.20, width=1.0)
    # all face nodes
    nx.draw_networkx_nodes(H_face, pos=pos_faces, ax=axR,
                           nodelist=list(H_face.nodes()),
                           node_size=220, alpha=0.55)
    # unmatched highlighted
    if unmatched_faces:
        nx.draw_networkx_nodes(H_face, pos=pos_faces, ax=axR,
                               nodelist=list(unmatched_faces),
                               node_size=320, node_color="orange", alpha=0.85)
    # matching edges highlighted
    if matching_edges:
        edgelist = list(matching_edges)
        nx.draw_networkx_edges(H_face, pos=pos_faces, ax=axR,
                               edgelist=edgelist, width=3.0, alpha=0.9)

    axR.set_title(f"{title}\nFace graph: |F|={len(F0)} |M|={len(matching_edges)} unmatched={len(unmatched_faces)}")
    axR.set_axis_off()

    # Left panel cover markers
    cover_markers = gadget_patches(cover)
    for p in cover_markers:
        p.set_facecolor("grey")
        p.set_edgecolor("none")
        p.set_alpha(0.22)

    for t, st in enumerate(hist):
        axL.clear()

        for p in cover_markers:
            axL.add_patch(p)

        nx.draw_networkx_edges(G_full, pos=pos_nodes, ax=axL, alpha=0.10)

        rem = list(st.remaining)
        pk = list(st.picked)
        kn = list(st.knocked)

        if rem:
            nx.draw_networkx_nodes(G_full, pos=pos_nodes, nodelist=rem, ax=axL, node_size=360, alpha=0.70)
        if pk:
            nx.draw_networkx_nodes(G_full, pos=pos_nodes, nodelist=pk, ax=axL, node_size=220, node_color="green", alpha=0.85)
        if kn:
            nx.draw_networkx_nodes(G_full, pos=pos_nodes, nodelist=kn, ax=axL, node_size=220, node_color="red", alpha=0.85, node_shape="s")

        if st.last_action is None:
            axL.set_title(f"{title}  step {t}")
        else:
            step = st.last_action
            axL.set_title(f"{title}  step {t}: {step.kind} {step.node}   {st.note}")

        # action indices
        for tt in range(t + 1):
            past = hist[tt]
            if past.last_action is None:
                continue
            step = past.last_action
            u = step.node
            act = step.kind
            if u not in pos_nodes:
                continue
            x, y = pos_nodes[u]
            axL.text(
                x + 0.08, y + 0.08, str(tt),
                fontsize=12, fontweight="bold",
                color=("green" if act == "pick" else "red"),
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                zorder=10,
            )

        axL.set_axis_off()
        plt.pause(pause)

    plt.ioff()
    #plt.show()
    plt.pause(2.0)


# ============================================================
# Random hole-free connected subgraph generator (by removals)
# ============================================================

def _neighbors4(i:int, j:int, m:int, n:int) -> Iterable[Node]:
    for di,dj in DIRS4:
        ni,nj = i+di, j+dj
        if 0 <= ni < m and 0 <= nj < n:
            yield (ni,nj)

def _is_connected_occ(occ: Set[Node], m: int, n: int) -> bool:
    if not occ:
        return True
    start = next(iter(occ))
    q = collections.deque([start])
    seen = {start}
    while q:
        u = q.popleft()
        ui,uj = u
        for v in _neighbors4(ui,uj,m,n):
            if v in occ and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(occ)

def _is_hole_free(occ: Set[Node], m: int, n: int) -> bool:
    """
    Hole-free: every empty cell is connected (4-neigh) to boundary empties.
    """
    empties = {(i,j) for i in range(m) for j in range(n)} - set(occ)
    if not empties:
        return True

    seeds = []
    for i in range(m):
        for j in range(n):
            if i in (0,m-1) or j in (0,n-1):
                if (i,j) in empties:
                    seeds.append((i,j))
    if not seeds:
        return False

    q = collections.deque(seeds)
    seen = set(seeds)
    while q:
        u = q.popleft()
        ui,uj = u
        for v in _neighbors4(ui,uj,m,n):
            if v in empties and v not in seen:
                seen.add(v)
                q.append(v)
    return len(seen) == len(empties)

def random_hole_free_subgraph_by_removal(
    m: int,
    n: int,
    k: int,
    seed: int,
    max_restarts: int = 200,
    protected: Optional[Set[Node]] = None,
) -> Set[Node]:
    rng = random.Random(seed)
    protected = set(protected or set())
    if not protected.issubset(occ_full(m, n)):
        raise ValueError("protected cells must lie inside the grid")
    if len(protected) > k:
        raise ValueError("k must be at least the number of protected cells")
    if k >= m*n:
        return occ_full(m,n)
    if k <= 0:
        return set()

    def current_boundary(occ: Set[Node]) -> List[Node]:
        return [
            (i, j) for i, j in occ
            if i in (0, m - 1) or j in (0, n - 1)
            or any(neighbor not in occ for neighbor in _neighbors4(i, j, m, n))
        ]

    for _ in range(max_restarts):
        occ = occ_full(m,n)
        attempts = 0
        while len(occ) > k and attempts < 200000:
            attempts += 1
            boundary = current_boundary(occ)
            cand_pool = [u for u in (boundary if boundary else list(occ))
                         if u not in protected]
            if not cand_pool:
                break
            u = rng.choice(cand_pool)
            occ2 = set(occ); occ2.remove(u)
            if not _is_connected_occ(occ2, m, n):
                continue
            if not _is_hole_free(occ2, m, n):
                continue
            occ = occ2
        if len(occ) == k:
            return occ

    # fallback deterministic boundary peel
    occ = occ_full(m,n)
    while len(occ) > k:
        removed = False
        boundary_now = set(current_boundary(occ))
        for u in sorted(list(occ)):
            if u in protected:
                continue
            if u not in boundary_now:
                continue
            occ2 = set(occ); occ2.remove(u)
            if _is_connected_occ(occ2,m,n) and _is_hole_free(occ2,m,n):
                occ = occ2
                removed = True
                break
        if not removed:
            break
    return occ


# ============================================================
# Benchmark + metrics
# ============================================================

@dataclass
class Metrics:
    ok: bool
    mode: str
    knocks: int
    picks: int
    steps: int
    t_cover: float
    t_exec: float
    t_total: float
    extra: int = 0
    fallback: int = 0

def _count_knocks(actions: List[PlanStep]) -> int:
    return sum(1 for s in actions if s.kind == "knock")

def _count_picks(actions: List[PlanStep]) -> int:
    return sum(1 for s in actions if s.kind == "pick")

def _fmt_sec(x: float) -> str:
    return f"{x*1000.0:.2f}ms" if x < 1.0 else f"{x:.3f}s"

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs) if xs else 0.0


def assign_node_heights(
    occ: Set[Node], heights: Optional[List[float]] = None
) -> Dict[Node, float]:
    """Assign the YAML height cycle to occupied nodes in row-major order."""
    values = [1.0] if heights is None else [float(value) for value in heights]
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("heights must contain positive cuboid heights")
    return {
        node: values[index % len(values)]
        for index, node in enumerate(sorted(occ))
    }

def run_matching_solver(
    occ: Set[Node], m: int, n: int,
    debug: bool = False,
    visualize: bool = False,
    pause: float = 0.15,
    title: str = "",
    node_heights: Optional[Dict[Node, float]] = None,
) -> Tuple[Metrics, List[PlanStep]]:
    G0 = build_graph(occ, node_heights=node_heights)
    t0 = time.perf_counter()

    U0 = core_nodes(G0)
    if len(U0) == 0:
        # pick-only
        actions: List[PlanStep] = []
        hist: List[ExecStep] = [ExecStep(remaining=set(G0.nodes()), picked=set(), knocked=set(), last_action=None, note="start")]
        picked: Set[Node] = set()
        knocked: Set[Node] = set()
        _cleanup_picks(G0, actions, picked, hist, knocked, note_prefix="pick-only/")
        t1 = time.perf_counter()
        ok = (G0.number_of_nodes() == 0)
        return Metrics(ok=ok, mode="match_pick_only",
                       knocks=0, picks=len(actions), steps=len(actions),
                       t_cover=0.0, t_exec=(t1-t0), t_total=(t1-t0),
                       extra=0, fallback=0), actions

    F0 = enumerate_faces(U0, m, n)
    gadgets0 = enumerate_face_gadgets(U0, F0, m, n)

    if debug:
        cnt = collections.Counter(g.kind for g in gadgets0)
        print(f"[INIT] |V|={G0.number_of_nodes()} |E|={G0.number_of_edges()} |Core|={len(U0)} |Faces|={len(F0)} gadgets={dict(cnt)}")

    cover, matching_edges, unmatched_faces = maximum_matching_cover_faces(F0, gadgets0, debug=debug)
    t_cover_end = time.perf_counter()

    # For right-panel graph: build adjacency graph from gadgets0 (2-face gadgets)
    H_face, _ = build_face_adjacency_graph_and_map(gadgets0)

    ok, actions, hist, used, stats = execute_from_cover(G0, cover, m, n, debug=debug)
    t_exec_end = time.perf_counter()

    sanity = replay_sanity(
        occ, m, n, actions, verbose_fail=True, node_heights=node_heights
    )
    if debug:
        print(f"[SANITY] replay={sanity}")

    if visualize:
        visualize_two_panel(G0, hist, cover, m, n,
                            F0=F0, H_face=H_face,
                            matching_edges=matching_edges,
                            unmatched_faces=unmatched_faces,
                            pause=pause, title=title)

    knocks = _count_knocks(actions)
    picks = _count_picks(actions)

    return Metrics(ok=ok and sanity,
                   mode="edge cover",
                   knocks=knocks,
                   picks=picks,
                   steps=len(actions),
                   t_cover=(t_cover_end - t0),
                   t_exec=(t_exec_end - t_cover_end),
                   t_total=(t_exec_end - t0),
                   extra=int(stats.get("extra", 0)),
                   fallback=int(stats.get("fallback", 0))), actions

def print_full_grid_table(results: List[Tuple[str,int,int,Metrics]]) -> None:
    print("\n=== FULL GRID RESULTS (edge-cover) ===")
    header = f"{'case':16s} {'|V|':>5s} {'knocks':>7s} {'steps':>7s} {'t_cover':>10s} {'t_exec':>10s} {'t_total':>10s} {'extra':>6s} {'fb':>4s} {'ok':>4s}"
    print(header)
    print("-"*len(header))
    for name,m,n,met in results:
        V = m*n
        print(f"{name:16s} {V:5d} {met.knocks:7d} {met.steps:7d} "
              f"{_fmt_sec(met.t_cover):>10s} {_fmt_sec(met.t_exec):>10s} {_fmt_sec(met.t_total):>10s} "
              f"{met.extra:6d} {met.fallback:4d} {str(met.ok):>4s}")

def print_subgraph_table(rows: List[Dict]) -> None:
    print("\n=== MIXED-HEIGHT SUBGRAPH AVERAGES (edge-cover) ===")
    header = f"{'hull':10s} {'k':>5s} {'N':>3s} {'knockN':>7s} {'knocks':>9s} {'steps':>9s} {'t_cover':>10s} {'t_exec':>10s} {'t_total':>10s} {'extra':>8s} {'fb':>6s} {'ok%':>6s}"
    print(header)
    print("-"*len(header))
    for r in rows:
        print(f"{r['hull']:10s} {r['k']:5d} {r['N']:3d} {r['knock_required']:7d} "
              f"{r['knocks_mean']:9.2f} {r['steps_mean']:9.2f} "
              f"{_fmt_sec(r['t_cover_mean']):>10s} {_fmt_sec(r['t_exec_mean']):>10s} {_fmt_sec(r['t_total_mean']):>10s} "
              f"{r['extra_mean']:8.2f} {r['fallback_mean']:6.2f} {r['ok_rate']*100:5.1f}%")


def _corner_gadget(kind: str) -> Set[Node]:
    if kind == "2x2":
        return {(i, j) for i in range(2) for j in range(2)}
    if kind == "2x3":
        return {(i, j) for i in range(2) for j in range(3)}
    if kind == "3x2":
        return {(i, j) for i in range(3) for j in range(2)}
    raise ValueError(f"unsupported forced gadget kind {kind!r}")


def random_height_assignment(
    occ: Set[Node],
    seed: int,
    levels: Tuple[int, ...] = (1, 2, 3, 4),
    forced_equal_cells: Optional[Set[Node]] = None,
) -> Dict[Node, float]:
    """Assign reproducible integer cuboid heights, optionally forcing a gadget."""
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("height levels must be positive")
    rng = random.Random(seed)
    result = {node: float(rng.choice(levels)) for node in sorted(occ)}
    forced_equal_cells = set(forced_equal_cells or set())
    if not forced_equal_cells.issubset(occ):
        raise ValueError("forced equal-height gadget must be occupied")
    for node in forced_equal_cells:
        result[node] = float(max(levels))
    return result


def plot_degree_sequence(
    occ: Set[Node],
    node_heights: Dict[Node, float],
    actions: List[PlanStep],
    output_path: str,
    title: str,
) -> None:
    """Save both degree definitions across the complete manipulation plan."""
    nodes = sorted(occ)
    node_index = {node: index for index, node in enumerate(nodes)}
    shape = (len(nodes), len(actions) + 1)
    occupancy_degrees = np.full(shape, np.nan)
    height_degrees = np.full(shape, np.nan)
    graph = build_graph(occ, node_heights=node_heights)

    def record(column: int) -> None:
        for node in graph.nodes():
            row = node_index[node]
            occupancy_degrees[row, column] = graph.degree(node)
            height_degrees[row, column] = height_aware_degree(graph, node)

    record(0)
    for column, action in enumerate(actions, start=1):
        graph.remove_node(action.node)
        record(column)

    width = min(26.0, max(12.0, 0.055 * len(actions) + 10.0))
    height = min(18.0, max(8.0, 0.025 * len(nodes) + 7.0))
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True,
                             constrained_layout=True)
    cmap = plt.colormaps["viridis"].resampled(5)
    cmap.set_bad(color="#e6e6e6")
    images = []
    for axis, values, label in zip(
        axes,
        (occupancy_degrees, height_degrees),
        ("Height-agnostic occupancy degree", "Height-aware picking degree"),
    ):
        image = axis.imshow(values, aspect="auto", interpolation="nearest",
                            cmap=cmap, vmin=-0.5, vmax=4.5)
        images.append(image)
        axis.set_title(label, loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("block (row-major)")
        for step_index, action in enumerate(actions, start=1):
            if action.kind == "knock":
                axis.axvline(step_index, color="#d62728", linewidth=0.8, alpha=0.8)

    max_y_ticks = 20
    tick_stride = max(1, int(np.ceil(len(nodes) / max_y_ticks)))
    y_ticks = list(range(0, len(nodes), tick_stride))
    for axis in axes:
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([
            f"{nodes[index]} h={node_heights[nodes[index]]:g}" for index in y_ticks
        ], fontsize=7)
    max_x_ticks = 16
    x_stride = max(1, int(np.ceil((len(actions) + 1) / max_x_ticks)))
    x_ticks = list(range(0, len(actions) + 1, x_stride))
    if x_ticks[-1] != len(actions):
        x_ticks.append(len(actions))
    axes[-1].set_xticks(x_ticks)
    axes[-1].set_xlabel("plan state (0 = initial; red line = knock action)")
    colorbar = fig.colorbar(images[-1], ax=axes, ticks=range(5), shrink=0.82)
    colorbar.set_label("degree")
    knocks = _count_knocks(actions)
    fig.suptitle(
        f"{title}\n{len(nodes)} blocks, {len(actions)} actions, "
        f"{knocks} knocks, {len(actions) - knocks} picks",
        fontsize=14, fontweight="bold",
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_graph_sequence(
    occ: Set[Node],
    node_heights: Dict[Node, float],
    actions: List[PlanStep],
    m: int,
    n: int,
    output_path: str,
    title: str,
) -> None:
    """Plot start, compressed pick batches, every knock, and final state."""
    graph = build_graph(occ, node_heights=node_heights)
    checkpoints = [dict(
        title="start", remaining=set(graph.nodes()), removed=set(),
        kind="start", action=None,
    )]
    action_index = 0
    knock_count = 0
    while action_index < len(actions):
        if actions[action_index].kind == "pick":
            first_step = action_index + 1
            removed: Set[Node] = set()
            while action_index < len(actions) and actions[action_index].kind == "pick":
                node = actions[action_index].node
                removed.add(node)
                graph.remove_node(node)
                action_index += 1
            last_step = action_index
            checkpoints.append(dict(
                title=(f"after {len(removed)} picks\nsteps {first_step}–{last_step}"),
                remaining=set(graph.nodes()), removed=removed,
                kind="pick", action=None,
            ))
        else:
            action = actions[action_index]
            knock_count += 1
            graph.remove_node(action.node)
            action_index += 1
            checkpoints.append(dict(
                title=(f"after knock {knock_count}\nstep {action_index}, "
                       f"node {action.node}"),
                remaining=set(graph.nodes()), removed={action.node},
                kind="knock", action=action,
            ))

    # The last action checkpoint is already the final graph. Label it clearly
    # instead of duplicating an identical empty panel and wasting a grid cell.
    checkpoints[-1]["title"] += "\nfinal state"

    panel_count = len(checkpoints)
    columns = min(4, panel_count)
    rows = int(np.ceil(panel_count / columns))
    fig, axes_array = plt.subplots(
        rows, columns, figsize=(4.6 * columns, 4.4 * rows), squeeze=False,
        constrained_layout=True,
    )
    axes = list(axes_array.flat)
    positions = {node: (float(node[1]), float(-node[0])) for node in occ}
    unique_heights = sorted({float(node_heights[node]) for node in occ})
    height_colors = {
        height: plt.colormaps["viridis"](
            0.12 if len(unique_heights) == 1
            else 0.12 + 0.76 * index / (len(unique_heights) - 1)
        )
        for index, height in enumerate(unique_heights)
    }
    node_size = max(120.0, min(620.0, 2800.0 / max(m, n)))
    font_size = max(5.0, min(10.0, 15.0 - 0.45 * max(m, n)))

    for axis, checkpoint in zip(axes, checkpoints):
        remaining = checkpoint["remaining"]
        subgraph = build_graph(remaining, node_heights=node_heights)
        if remaining:
            nx.draw_networkx_edges(
                subgraph, positions, ax=axis, edge_color="#9aa0a6",
                width=1.0, alpha=0.75,
            )
            nodes = sorted(remaining)
            nx.draw_networkx_nodes(
                subgraph, positions, nodelist=nodes, ax=axis,
                node_color=[height_colors[node_heights[node]] for node in nodes],
                node_size=node_size, edgecolors="#202124", linewidths=0.8,
            )
            nx.draw_networkx_labels(
                subgraph, positions,
                labels={node: f"{node_heights[node]:g}" for node in nodes},
                ax=axis, font_size=font_size, font_color="white",
                font_weight="bold",
            )
        else:
            axis.text(0.5, 0.5, "no blocks remaining", transform=axis.transAxes,
                      ha="center", va="center", color="#666666", fontsize=11)

        removed = sorted(checkpoint["removed"])
        if removed:
            marker_color = "#d62728" if checkpoint["kind"] == "knock" else "#2ca02c"
            axis.scatter(
                [positions[node][0] for node in removed],
                [positions[node][1] for node in removed],
                marker="x", s=node_size * 0.75, linewidths=2.2,
                color=marker_color, zorder=5,
            )
        action = checkpoint["action"]
        if action is not None and action.direction is not None:
            x, y = positions[action.node]
            di, dj = action.direction
            axis.annotate(
                "", xy=(x + 0.75 * dj, y - 0.75 * di), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=2.0),
                zorder=6,
            )

        axis.set_title(checkpoint["title"], fontsize=11, fontweight="bold")
        axis.set_xlim(-0.8, n - 0.2)
        axis.set_ylim(-m + 0.2, 0.8)
        axis.set_aspect("equal")
        axis.set_axis_off()

    for axis in axes[panel_count:]:
        axis.set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", label=f"height {height:g}",
               markerfacecolor=color, markeredgecolor="#202124", markersize=9)
        for height, color in height_colors.items()
    ]
    legend_handles.extend([
        Line2D([0], [0], marker="x", color="#2ca02c", linestyle="none",
               markersize=9, markeredgewidth=2, label="picked batch"),
        Line2D([0], [0], marker="x", color="#d62728", linestyle="none",
               markersize=9, markeredgewidth=2, label="knocked"),
    ])
    fig.legend(handles=legend_handles, loc="outside lower center", ncol=min(6, len(legend_handles)))
    fig.suptitle(
        f"{title}: compressed graph solution sequence\n"
        f"{len(occ)} blocks, {_count_knocks(actions)} knocks, "
        f"{_count_picks(actions)} picks",
        fontsize=15, fontweight="bold",
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_synthetic_benchmark(
    visualize: bool = False,
    pause: float = 0.15,
    seed: int = 0,
    debug: bool = False,
    problems_per_size: int = 20,
    degree_plot_dir: str = "results/degree_sequences",
    graph_plot_dir: str = "results/graph_sequences",
) -> None:
    if problems_per_size <= 0:
        raise ValueError("problems_per_size must be positive")
    rng = random.Random(seed)
    square_sizes = list(range(3, 21))
    if visualize:
        os.makedirs(degree_plot_dir, exist_ok=True)
        os.makedirs(graph_plot_dir, exist_ok=True)

    # One randomly height-labeled full grid for every size from 3x3 to 20x20.
    full_results: List[Tuple[str,int,int,Metrics]] = []
    for size in square_sizes:
        name = f"full_{size}x{size}_mixed_height"
        occ = occ_full(size, size)
        height_seed = rng.randrange(2**31)
        node_heights = random_height_assignment(occ, height_seed)
        print(f"\n==================== FULL GRID: {name} ====================")
        met, actions = run_matching_solver(
            occ, size, size, debug=debug, visualize=False, pause=pause,
            title=name, node_heights=node_heights,
        )
        full_results.append((name, size, size, met))
        if visualize:
            plot_degree_sequence(
                occ, node_heights, actions,
                os.path.join(degree_plot_dir, f"{name}.png"), name,
            )
            plot_graph_sequence(
                occ, node_heights, actions, size, size,
                os.path.join(graph_plot_dir, f"{name}.png"), name,
            )

    print_full_grid_table(full_results)

    # Multiple connected, hole-free mixed-height subgraphs per square hull.
    # Odd-indexed problems contain a protected maximum-height gadget, which
    # guarantees at least one non-pickable core while even-indexed problems
    # remain unbiased and may legitimately require zero knocks.
    sub_rows: List[Dict] = []
    for size in square_sizes:
        hull_name = f"{size}x{size}"
        m = n = size
        k = max(5, int(round(0.60 * m * n)))
        print(f"\n==================== SUBGRAPHS: hull {hull_name}, k={k} ====================")
        N = problems_per_size
        knocks_list: List[float] = []
        steps_list: List[float] = []
        tc_list: List[float] = []
        te_list: List[float] = []
        tt_list: List[float] = []
        extra_list: List[float] = []
        fb_list: List[float] = []
        ok_count = 0
        knock_required_count = 0

        for s in range(N):
            force_knock = (s % 2 == 1)
            forced_kind = None
            forced_cells: Set[Node] = set()
            if force_knock:
                available_kinds = ["2x2"]
                if k >= 6:
                    available_kinds.extend(["2x3", "3x2"])
                forced_kind = available_kinds[(s // 2) % len(available_kinds)]
                forced_cells = _corner_gadget(forced_kind)

            graph_seed = rng.randrange(2**31)
            height_seed = rng.randrange(2**31)
            occ = random_hole_free_subgraph_by_removal(
                m, n, k, graph_seed, protected=forced_cells
            )
            node_heights = random_height_assignment(
                occ, height_seed, forced_equal_cells=forced_cells
            )
            case_name = (
                f"sub_{hull_name}_k{k}_{s:02d}_"
                f"{'forced_' + forced_kind if forced_kind else 'random'}"
            )
            met, actions = run_matching_solver(
                occ, m, n, debug=debug, visualize=False, pause=pause,
                title=case_name, node_heights=node_heights,
            )
            if force_knock and met.knocks == 0:
                raise RuntimeError(
                    f"forced {forced_kind} case unexpectedly required zero knocks: "
                    f"{case_name}"
                )
            if visualize:
                plot_degree_sequence(
                    occ, node_heights, actions,
                    os.path.join(degree_plot_dir, f"{case_name}.png"),
                    case_name,
                )
                plot_graph_sequence(
                    occ, node_heights, actions, m, n,
                    os.path.join(graph_plot_dir, f"{case_name}.png"),
                    case_name,
                )

            ok_count += int(met.ok)
            knock_required_count += int(met.knocks > 0)

            knocks_list.append(float(met.knocks))
            steps_list.append(float(met.steps))
            tc_list.append(float(met.t_cover))
            te_list.append(float(met.t_exec))
            tt_list.append(float(met.t_total))
            extra_list.append(float(met.extra))
            fb_list.append(float(met.fallback))

        sub_rows.append(dict(
            hull=hull_name,
            k=k,
            N=N,
            knocks_mean=mean(knocks_list),
            steps_mean=mean(steps_list),
            t_cover_mean=mean(tc_list),
            t_exec_mean=mean(te_list),
            t_total_mean=mean(tt_list),
            extra_mean=mean(extra_list),
            fallback_mean=mean(fb_list),
            ok_rate=ok_count / N,
            knock_required=knock_required_count,
        ))

    print_subgraph_table(sub_rows)

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
    heights: Optional[List[float]] = None,
) -> None:
    """
    Write a problem.yaml that includes:
      pads (m*n initial pads + 3 goal pads),
      object geometry (configurable cuboids),
      initial / goal,
      actions (Knock+Move for a knocked block, Move for a directly picked block).

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

    if heights is None:
        heights = [size]
    if not heights or any(height <= 0.0 for height in heights):
        raise ValueError("heights must contain positive cuboid heights")
    for height in heights:
        multiple = height / size
        if abs(multiple - round(multiple)) > 1e-8:
            raise ValueError(
                f"cuboid height {height} must be an integer multiple of side {size}"
            )
    node_heights = assign_node_heights(occ, heights)
    dimensions = {
        node_name[node]: (size, size, node_heights[node])
        for node in occ_sorted
    }

    # Retained for old problem readers. New Isaac execution uses objects[].dimensions.
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
    lines.append("")
    lines.append("objects:")
    for name in names:
        dx, dy, dz = dimensions[name]
        lines.append(
            f"  {name}: {{dimensions: [{fmt(dx,prec)}, {fmt(dy,prec)}, {fmt(dz,prec)}]}}"
        )
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
            lines.append("  - type:  Knock")
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
# IsaacSim problem generation
# ============================================================

def run_isaac_benchmark(
    outdir: str,
    debug: bool,
    visualize: bool,
    seed: int = 0,
    pause: float = 0.20,
    size: float = 0.05,
    gap_s: float = 0.005,
    center_x: float = 0.50,
    center_y: float = 0.00,
    prec: int = 3,
    heights: Optional[List[float]] = None,
):
    rng = random.Random(seed)
    os.makedirs(outdir, exist_ok=True)

    print("\n==================== ISAACSIM PROBLEM GENERATION ====================", flush=True)

    sizes = [(3, 3), (5, 5)]
    # Use your synthetic subset sizes for these: 3x3 -> 5, 5x5 -> 16
    sub_k = {(3, 3): 5, (5, 5): 16}

    # 1) full grid: 2 files
    for (m, n) in sizes:
        occ = occ_full(m, n)
        node_heights = assign_node_heights(occ, heights)
        _, plan = run_matching_solver(occ, m, n,
                                  debug=debug,
                                  visualize=visualize,
                                  pause=pause,
                                  title=f"full_{m}x{n} (matching)",
                                  node_heights=node_heights)
        print(f"\n[ISAAC] full {m}x{n}", flush=True)
        out = os.path.join(outdir, f"full_{m}x{n}.yaml")
        write_problem_yaml(out, m, n, occ, size, gap_s, center_x, center_y, plan,
                           prec=prec, heights=heights)
        print(f"  wrote: {out}", flush=True)

    # 2) subgraph + greedy: 5 each size
    for (m, n) in sizes:
        k = sub_k[(m, n)]
        for sidx in range(5):
            seed += 1
            occ = random_hole_free_subgraph_by_removal(m, n, k, seed)
            node_heights = assign_node_heights(occ, heights)
            print(f"\n[ISAAC] sub {m}x{n} k={k} seed={seed}#{sidx}", flush=True)

            met, plan = run_matching_solver(occ, m, n,
                                      debug=debug,
                                      visualize=visualize,
                                      pause=pause,
                                      title=f"sub_{m}_{n}_k{k} (matching)",
                                      node_heights=node_heights)

            out = os.path.join(outdir, f"sub_{m}x{n}_k{k}_{sidx}.yaml")
            write_problem_yaml(out, m, n, occ, size, gap_s, center_x, center_y, plan,
                               prec=prec, heights=heights)
            print(f"  wrote: {out}", flush=True)

# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Compute optimal knock-pick plans for synthetic or Isaac Sim benchmarks.")
    ap.add_argument("--bench", type=str, default="synthetic", choices=["isaac", "synthetic"],
                    help="Run IsaacSim YAML generation or synthetic benchmark.")
    ap.add_argument(
        "--visualize", action="store_true",
        help="save full-plan occupancy and height-aware degree plots",
    )
    ap.add_argument("--pause", type=float, default=0.15)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--one", type=str, default="", help="run one named instance, e.g. full_5x5 or concave7")

    # Isaac parameters
    ap.add_argument("--outdir", type=str, default="problems", help="Output directory for IsaacSim YAMLs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--synthetic-problems-per-size", type=int, default=20,
        help="Random connected subgraphs for each square size 3..20",
    )
    ap.add_argument(
        "--degree-plot-dir", default="results/degree_sequences",
        help="Output directory used by --bench synthetic --visualize",
    )
    ap.add_argument(
        "--graph-plot-dir", default="results/graph_sequences",
        help="Compressed graph-sequence output used by synthetic --visualize",
    )
    ap.add_argument("--size", type=float, default=0.05)
    ap.add_argument(
        "--height-multipliers", default="1,2,3",
        help="Comma-separated positive integer height/side ratios (Isaac problems only).",
    )
    ap.add_argument("--gap_s", type=float, default=0.005)
    ap.add_argument("--center_x", type=float, default=0.45)
    ap.add_argument("--center_y", type=float, default=0.00)
    ap.add_argument("--precision", type=int, default=3)
    args = ap.parse_args()
    try:
        height_multipliers = [
            int(value) for value in args.height_multipliers.split(",") if value.strip()
        ]
    except ValueError as exc:
        ap.error(f"invalid --height-multipliers value: {exc}")
    if not height_multipliers or any(value <= 0 for value in height_multipliers):
        ap.error("--height-multipliers must contain positive integers")
    cuboid_heights = [args.size * value for value in height_multipliers]

    if args.one:
        # quick one-off runner for debugging/visualization
        if args.one == "concave7":
            occ, m, n = occ_concave_L(7,7, 0,4,3,7), 7, 7
        elif args.one.startswith("full_"):
            _, ms, ns = args.one.split("_")
            m, n = int(ms), int(ns)
            occ = occ_full(m,n)
        else:
            print("Unknown --one option. Try: concave7 or full_5_5 style via full_m_n.")
            return
        print(f"\n==================== ONE: {args.one} ====================")
        met, _ = run_matching_solver(occ, m, n, debug=args.debug, visualize=args.visualize, pause=args.pause, title=args.one)
        print(f"[ONE RESULT] ok={met.ok} knocks={met.knocks} picks={met.picks} steps={met.steps} "
              f"t_cover={_fmt_sec(met.t_cover)} t_exec={_fmt_sec(met.t_exec)} t_total={_fmt_sec(met.t_total)} "
              f"extra={met.extra} fallback={met.fallback}")
        return

    if args.bench == "synthetic":
        run_synthetic_benchmark(
            visualize=args.visualize,
            pause=args.pause,
            seed=args.seed,
            debug=args.debug,
            problems_per_size=args.synthetic_problems_per_size,
            degree_plot_dir=args.degree_plot_dir,
            graph_plot_dir=args.graph_plot_dir,
        )
    else:
        run_isaac_benchmark(
            outdir=args.outdir,
            debug=args.debug,
            visualize=args.visualize,
            seed=args.seed,
            pause=args.pause,
            size=args.size,
            gap_s=args.gap_s,
            center_x=args.center_x,
            center_y=args.center_y,
            prec=args.precision,
            heights=cuboid_heights,
        )

if __name__ == "__main__":
    main()
