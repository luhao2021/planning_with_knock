#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict, Iterable
import argparse
import collections
import random
import time

import networkx as nx
import matplotlib.pyplot as plt
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

def occ_multi_concave_holes(m: int, n: int, concaves: List[Tuple[int,int,int,int]], holes: List[Tuple[int,int,int,int]]) -> Set[Node]:
    """
    Start full hull then remove:
      - concaves: rectangles that touch the boundary (notches)
      - holes: interior rectangles
    """
    occ = occ_full(m, n)
    occ = apply_removals(occ, concaves)
    occ = apply_removals(occ, holes)
    return occ


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
# Greedy face cover (unweighted)
# ============================================================

def greedy_set_cover_faces(F: Set[Face], gadgets: List[FaceGadget], m: int, n: int, debug: bool = True) -> List[FaceGadget]:
    """
    Greedy set cover:
      maximize newly covered faces,
      then minimize waste (|faces(g)|-newly),
      then minimize |faces(g)|  (forces 2x2 when only 1 face remains),
      then prefer outer layers, then stable tie-break.
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
                print(f"[COVER-FACE] stuck: remaining faces={len(U)}")
            break

        _, gbest, newly = best
        chosen.append(gbest)
        U -= set(newly)
        if debug:
            print(f"[COVER-FACE] step {step}: chose {gbest.kind} covers {len(newly)} face(s); remaining faces={len(U)}")
        step += 1
        if step > 5_000_000:
            print("[COVER-FACE] guard triggered")
            break

    return chosen


# ============================================================
# STRICT knock candidates per gadget kind
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
      - require degree<=3 and some Ray direction feasible.
    """
    cand0 = [v for v in gadget_knock_candidates(g) if (v in coreU) and G.has_node(v)]
    if not cand0:
        return None

    # Prefer outermost / knockable candidates implicitly by filtering degree<=3 first
    cand0.sort(key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))

    for v in cand0:
        if G.degree(v) <= 3:
            dirs = ray_clear_directions(G, v, m, n)
            if dirs:
                dirs.sort()
                return v, dirs[0]

    return None


# ============================================================
# Execution dataclass (for visualization)
# ============================================================

@dataclass
class ExecStep:
    remaining: Set[Node]
    picked: Set[Node]
    knocked: Set[Node]
    last_action: Optional[Tuple[Node, str]]
    note: str = ""


# ============================================================
# Approx execution from cover
# ============================================================

def execute_from_cover(
    G_full: nx.Graph,
    cover: List[FaceGadget],
    m: int,
    n: int,
    debug: bool = True,
) -> Tuple[bool, List[Tuple[Node, str]], List[ExecStep], List[bool], Dict[str, int]]:
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
    actions: List[Tuple[Node, str]] = []
    hist: List[ExecStep] = []
    used = [False] * len(cover)
    stats = collections.Counter()

    def record(last: Optional[Tuple[Node, str]], note: str = "") -> None:
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
            actions.append((u, "pick"))
            picked.add(u)
            G.remove_node(u)
            record((u, "pick"), "cleanup")

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
            actions.append((v, "knock"))
            knocked.add(v)
            G.remove_node(v)
            stats["cover_knock"] += 1
            record((v, "knock"), f"gid={idx} kind={cover[idx].kind} d={d}")
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
            actions.append((v, "knock_extra"))
            knocked.add(v)
            G.remove_node(v)
            stats["extra"] += 1
            record((v, "knock_extra"), f"extra kind={g.kind} d={d}")
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
        actions.append((v, "knock_fallback"))
        knocked.add(v)
        G.remove_node(v)
        stats["fallback"] += 1
        record((v, "knock_fallback"), f"d={d}")

    if debug:
        knocks = sum(1 for _, a in actions if "knock" in a)
        picks = sum(1 for _, a in actions if a == "pick")
        print(f"[EXEC DONE] knocks={knocks} picks={picks} steps={len(actions)} used={sum(used)}/{len(cover)} stats={dict(stats)}")

    return True, actions, hist, used, dict(stats)


# ============================================================
# Sanity replay
# ============================================================

def replay_sanity(occ: Set[Node], m: int, n: int, actions: List[Tuple[Node, str]], verbose_fail: bool = True) -> bool:
    G = build_graph(occ)
    for t, (u, act) in enumerate(actions):
        if not G.has_node(u):
            if verbose_fail:
                print(f"[REPLAY FAIL] step {t}: node {u} missing")
            return False
        if act == "pick":
            if not pickable(G, u):
                if verbose_fail:
                    print(f"[REPLAY FAIL] step {t}: pick infeasible at {u}, deg={G.degree(u)}")
                return False
            G.remove_node(u)
        else:
            if not knockable(G, u, m, n):
                if verbose_fail:
                    print(f"[REPLAY FAIL] step {t}: knock infeasible at {u}, deg={G.degree(u)} act={act}")
                    print("  ray dirs:", ray_clear_directions(G, u, m, n))
                return False
            G.remove_node(u)
    ok = (G.number_of_nodes() == 0)
    if not ok and verbose_fail:
        print("[REPLAY FAIL] ended with remaining nodes:", G.number_of_nodes())
    return ok


# ============================================================
# Visualization (cover-only + merged)
# ============================================================

def gadget_patches(cover: List[FaceGadget]):
    """
    Visualization rule:
      - 2x2 gadget: small square at face center.
      - 2x3 / 3x2: small rectangle centered at the middle point between the two middle nodes.
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

def visualize_exec_with_cover(
    G_full: nx.Graph,
    hist: List[ExecStep],
    cover: List[FaceGadget],
    m: int,
    n: int,
    pause: float = 0.20,
    title: str = "",
) -> None:
    pos = nx.get_node_attributes(G_full, "pos")
    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 11))

    cover_markers = gadget_patches(cover)
    for p in cover_markers:
        p.set_facecolor("grey")
        p.set_edgecolor("none")
        p.set_alpha(0.22)

    for t, st in enumerate(hist):
        ax.clear()

        for p in cover_markers:
            ax.add_patch(p)

        nx.draw_networkx_edges(G_full, pos=pos, ax=ax, alpha=0.10)

        rem = list(st.remaining)
        pk = list(st.picked)
        kn = list(st.knocked)

        if rem:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=rem, ax=ax, node_size=360, alpha=0.70)
        if pk:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=pk, ax=ax, node_size=220, node_color="green", alpha=0.85)
        if kn:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=kn, ax=ax, node_size=220, node_color="red", alpha=0.85, node_shape="s")

        if st.last_action is None:
            ax.set_title(f"{title} step {t}")
        else:
            u, act = st.last_action
            ax.set_title(f"{title} step {t}: {act} {u}  {st.note}")

        # action indices
        for tt in range(t + 1):
            past = hist[tt]
            if past.last_action is None:
                continue
            u, act = past.last_action
            if u not in pos:
                continue
            x, y = pos[u]
            ax.text(
                x + 0.08, y + 0.08, str(tt),
                fontsize=12, fontweight="bold",
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
# Exact full-grid solver: RectLayerPeel (layer peeling)
# ============================================================

def optimal_knocks_full_grid(m: int, n: int) -> int:
    return ((m - 1) * (n - 1) + 1) // 2

def _cleanup_picks(G: nx.Graph, m: int, n: int,
                   actions: List[Tuple[Node,str]],
                   picked: Set[Node],
                   hist: List[ExecStep],
                   knocked: Set[Node],
                   note_prefix: str = "") -> None:
    """Repeatedly pick any pickable node (deterministic smallest lexicographic)."""
    while True:
        P = [u for u in G.nodes() if pickable(G, u)]
        if not P:
            break
        P.sort()
        u = P[0]
        actions.append((u, "pick"))
        picked.add(u)
        G.remove_node(u)
        hist.append(ExecStep(remaining=set(G.nodes()),
                             picked=set(picked),
                             knocked=set(knocked),
                             last_action=(u, "pick"),
                             note=f"{note_prefix}cleanup"))

def _current_bbox(G: nx.Graph) -> Optional[Tuple[int,int,int,int]]:
    """Return (imin,imax,jmin,jmax) inclusive bounds of remaining nodes."""
    if G.number_of_nodes() == 0:
        return None
    nodes = list(G.nodes())
    imin = min(i for i,j in nodes)
    imax = max(i for i,j in nodes)
    jmin = min(j for i,j in nodes)
    jmax = max(j for i,j in nodes)
    return imin, imax, jmin, jmax

def _rectlayer_targets(imin:int, imax:int, jmin:int, jmax:int) -> List[Node]:
    """
    Produce the intended knock targets for the current rectangle.
    Handles:
      - m_cur,n_cur >= 3 : every other non-corner boundary node in cyclic order.
      - m_cur == 2       : knock along top row at columns jmin+1, jmin+3, ...
      - n_cur == 2       : knock along left col at rows imin+1, imin+3, ...
      - m_cur==2,n_cur==2 also works (single corner).
    """
    mcur = imax - imin + 1
    ncur = jmax - jmin + 1
    targets: List[Node] = []

    if mcur <= 1 or ncur <= 1:
        return targets

    if mcur == 2:
        # 2 x n strip
        for j in range(jmin + 1, jmax + 1, 2):
            targets.append((imin, j))
        return targets

    if ncur == 2:
        # m x 2 strip
        for i in range(imin + 1, imax + 1, 2):
            targets.append((i, jmin))
        return targets

    # mcur,ncur >= 3: cyclic non-corner boundary nodes
    cyc: List[Node] = []
    # top edge (exclude corners)
    for j in range(jmin + 1, jmax):
        cyc.append((imin, j))
    # right edge
    for i in range(imin + 1, imax):
        cyc.append((i, jmax))
    # bottom edge
    for j in range(jmax - 1, jmin, -1):
        cyc.append((imax, j))
    # left edge
    for i in range(imax - 1, imin, -1):
        cyc.append((i, jmin))

    # take every other
    targets = cyc[::2]
    return targets

def rect_layer_peel_execute(m: int, n: int, debug: bool = True) -> Tuple[bool, List[Tuple[Node,str]], List[ExecStep], Dict[str,int]]:
    """
    Exact solver for FULL m×n grid.
    Interleaves:
      - cleanup picks
      - choose the current rectangle bbox (should remain a rectangle)
      - knock the layer targets for that rectangle
    """
    occ = occ_full(m, n)
    G = build_graph(occ)

    actions: List[Tuple[Node,str]] = []
    hist: List[ExecStep] = []
    picked: Set[Node] = set()
    knocked: Set[Node] = set()
    stats = collections.Counter()

    hist.append(ExecStep(remaining=set(G.nodes()),
                         picked=set(),
                         knocked=set(),
                         last_action=None,
                         note="start"))

    it = 0
    max_it = 30 * max(1, m*n) + 500

    while G.number_of_nodes() > 0:
        it += 1
        if it > max_it:
            print("[RECT] guard triggered (deadlock/bug).")
            return False, actions, hist, dict(stats)

        # cleanup picks as much as possible
        _cleanup_picks(G, m, n, actions, picked, hist, knocked, note_prefix="rect/")

        if G.number_of_nodes() == 0:
            break

        bb = _current_bbox(G)
        if bb is None:
            break
        imin, imax, jmin, jmax = bb
        mcur = imax - imin + 1
        ncur = jmax - jmin + 1

        if debug:
            print(f"[RECT] it={it} remaining={G.number_of_nodes()} bbox=({imin}:{imax},{jmin}:{jmax}) size={mcur}x{ncur}")

        targets = _rectlayer_targets(imin, imax, jmin, jmax)
        if not targets:
            # if no targets, we should be able to pick everything; otherwise pick model mismatch
            if debug:
                print("[RECT] no knock targets in this bbox; trying cleanup again...")
            _cleanup_picks(G, m, n, actions, picked, hist, knocked, note_prefix="rect/")
            if G.number_of_nodes() == 0:
                break
            # If still nodes remain, pickability model + this bbox schedule mismatch → knock any feasible boundary node
            remaining = list(G.nodes())
            remaining.sort(key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
            forced = None
            for v in remaining:
                if knockable(G, v, m, n):
                    dirs = ray_clear_directions(G, v, m, n)
                    dirs.sort()
                    forced = (v, dirs[0])
                    break
            if forced is None:
                print("[RECT] DEADLOCK: no ray-feasible knock.")
                return False, actions, hist, dict(stats)
            v, d = forced
            actions.append((v, "knock"))
            knocked.add(v)
            G.remove_node(v)
            stats["knock"] += 1
            hist.append(ExecStep(remaining=set(G.nodes()),
                                 picked=set(picked),
                                 knocked=set(knocked),
                                 last_action=(v, "knock"),
                                 note=f"forced d={d}"))
            continue

        # Execute target knocks; skip targets already removed by picks
        did_knock = False
        for v in targets:
            if not G.has_node(v):
                continue
            # ensure we only knock knockable nodes w.r.t. Ray
            if not knockable(G, v, m, n):
                # Try other node on the same side near v (rare)
                continue
            dirs = ray_clear_directions(G, v, m, n)
            if not dirs:
                continue
            dirs.sort()
            d = dirs[0]

            actions.append((v, "knock"))
            knocked.add(v)
            G.remove_node(v)
            stats["knock"] += 1
            hist.append(ExecStep(remaining=set(G.nodes()),
                                 picked=set(picked),
                                 knocked=set(knocked),
                                 last_action=(v, "knock"),
                                 note=f"rect d={d} bbox={mcur}x{ncur}"))
            did_knock = True

            # After each knock, cleanup picks immediately (this is what makes the peel “work”)
            _cleanup_picks(G, m, n, actions, picked, hist, knocked, note_prefix="rect/")
            if G.number_of_nodes() == 0:
                break

        if not did_knock:
            # As a safety net: knock any ray-feasible node on current boundary
            boundary = []
            for (i,j) in G.nodes():
                if i == imin or i == imax or j == jmin or j == jmax:
                    boundary.append((i,j))
            boundary.sort(key=lambda v: (layer_index_hull(v[0], v[1], m, n), -G.degree(v), v))
            forced = None
            for v in boundary:
                if knockable(G, v, m, n):
                    dirs = ray_clear_directions(G, v, m, n)
                    dirs.sort()
                    forced = (v, dirs[0])
                    break
            if forced is None:
                print("[RECT] DEADLOCK: no boundary ray-feasible knock.")
                return False, actions, hist, dict(stats)
            v, d = forced
            actions.append((v, "knock"))
            knocked.add(v)
            G.remove_node(v)
            stats["knock_forced_boundary"] += 1
            hist.append(ExecStep(remaining=set(G.nodes()),
                                 picked=set(picked),
                                 knocked=set(knocked),
                                 last_action=(v, "knock"),
                                 note=f"boundary-forced d={d}"))
            _cleanup_picks(G, m, n, actions, picked, hist, knocked, note_prefix="rect/")

    ok = (G.number_of_nodes() == 0)
    if debug:
        knocks = sum(1 for _, a in actions if "knock" in a)
        picks = sum(1 for _, a in actions if a == "pick")
        print(f"[RECT DONE] ok={ok} knocks={knocks} picks={picks} steps={len(actions)} stats={dict(stats)}")
        if ok:
            print(f"[RECT OPT CHECK] expected_knocks={optimal_knocks_full_grid(m,n)}")
    return ok, actions, hist, dict(stats)


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
    Hole-free: every empty cell is connected (4-neigh) to the boundary empties.
    Flood fill empties from boundary; if some empty not reached => hole.
    """
    empties = {(i,j) for i in range(m) for j in range(n)} - set(occ)
    if not empties:
        return True

    # boundary empties
    seeds = []
    for i in range(m):
        for j in range(n):
            if i in (0,m-1) or j in (0,n-1):
                if (i,j) in empties:
                    seeds.append((i,j))
    if not seeds:
        # boundary fully occupied but empties exist -> must be interior hole(s)
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

def random_hole_free_subgraph_by_removal(m: int, n: int, k: int, seed: int, max_restarts: int = 200) -> Set[Node]:
    """
    Start from full hull and remove random cells until |occ|=k,
    while maintaining:
      - occ connected
      - empties hole-free
    """
    rng = random.Random(seed)

    if k >= m*n:
        return occ_full(m,n)
    if k <= 0:
        return set()

    for restart in range(max_restarts):
        occ = occ_full(m,n)
        cells = list(occ)
        rng.shuffle(cells)

        # removal candidates are shuffled each restart
        attempts = 0
        while len(occ) > k and attempts < 200000:
            attempts += 1
            # prefer removing boundary cells to avoid making holes
            boundary = []
            for (i,j) in occ:
                if i==0 or i==m-1 or j==0 or j==n-1:
                    boundary.append((i,j))
            # also include current boundary in occ (w.r.t. bbox), but hull boundary is enough.
            cand_pool = boundary if boundary else list(occ)
            u = rng.choice(cand_pool)

            occ2 = set(occ)
            occ2.remove(u)

            if not _is_connected_occ(occ2, m, n):
                continue
            if not _is_hole_free(occ2, m, n):
                continue

            occ = occ2

        if len(occ) == k:
            return occ

    # if fails, just return a deterministic peel (always remove boundary in a fixed sweep)
    occ = occ_full(m,n)
    while len(occ) > k:
        # remove smallest boundary that preserves constraints
        removed = False
        for u in sorted(list(occ)):
            i,j = u
            if not (i==0 or i==m-1 or j==0 or j==n-1):
                continue
            occ2 = set(occ); occ2.remove(u)
            if _is_connected_occ(occ2,m,n) and _is_hole_free(occ2,m,n):
                occ = occ2
                removed = True
                break
        if not removed:
            # give up and just stop
            break
    return occ


# ============================================================
# Benchmark runners
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

def _count_knocks(actions: List[Tuple[Node,str]]) -> int:
    return sum(1 for _,a in actions if "knock" in a)

def _count_picks(actions: List[Tuple[Node,str]]) -> int:
    return sum(1 for _,a in actions if a == "pick")

def run_approx_solver(occ: Set[Node], m: int, n: int, debug: bool = False, visualize: bool = False, pause: float = 0.15, title: str = "") -> Metrics:
    G = build_graph(occ)

    # If everything is pickable after cleanup: run pick-only (cover time 0)
    t0 = time.perf_counter()
    U0 = core_nodes(G)
    if len(U0) == 0:
        # pick-only execution
        actions: List[Tuple[Node,str]] = []
        hist: List[ExecStep] = [ExecStep(remaining=set(G.nodes()), picked=set(), knocked=set(), last_action=None, note="start")]
        picked: Set[Node] = set()
        knocked: Set[Node] = set()
        _cleanup_picks(G, m, n, actions, picked, hist, knocked, note_prefix="pick-only/")
        t1 = time.perf_counter()
        ok = (G.number_of_nodes() == 0)
        if visualize:
            visualize_exec_with_cover(build_graph(occ), hist, [], m, n, pause=pause, title=f"{title} approx(pick-only)")
        return Metrics(ok=ok, mode="approx_pick_only",
                       knocks=0, picks=len(actions), steps=len(actions),
                       t_cover=0.0, t_exec=(t1-t0), t_total=(t1-t0),
                       extra=0, fallback=0)

    # cover time includes faces+gadgets+greedy
    F0 = enumerate_faces(U0, m, n)
    gadgets = enumerate_face_gadgets(U0, F0, m, n)
    cover = greedy_set_cover_faces(F0, gadgets, m, n, debug=debug)
    t_cover_end = time.perf_counter()

    # exec time
    ok, actions, hist, used, stats = execute_from_cover(build_graph(occ), cover, m, n, debug=debug)
    t_exec_end = time.perf_counter()

    sanity = replay_sanity(occ, m, n, actions, verbose_fail=True)
    if debug:
        print(f"[SANITY] replay={sanity}")

    if visualize:
        visualize_exec_with_cover(build_graph(occ), hist, cover, m, n, pause=pause, title=f"{title} approx")

    knocks = _count_knocks(actions)
    picks = _count_picks(actions)
    return Metrics(ok=ok and sanity,
                   mode="approx",
                   knocks=knocks,
                   picks=picks,
                   steps=len(actions),
                   t_cover=(t_cover_end - t0),
                   t_exec=(t_exec_end - t_cover_end),
                   t_total=(t_exec_end - t0),
                   extra=int(stats.get("extra", 0)),
                   fallback=int(stats.get("fallback", 0)))

def run_exact_full_grid(m: int, n: int, debug: bool = False, visualize: bool = False, pause: float = 0.15, title: str = "") -> Metrics:
    t0 = time.perf_counter()
    ok, actions, hist, stats = rect_layer_peel_execute(m, n, debug=debug)
    t1 = time.perf_counter()

    sanity = replay_sanity(occ_full(m,n), m, n, actions, verbose_fail=True)
    if debug:
        print(f"[RECT SANITY] replay={sanity}")

    if visualize:
        visualize_exec_with_cover(build_graph(occ_full(m,n)), hist, [], m, n, pause=pause, title=f"{title} exact(RectLayerPeel)")

    knocks = _count_knocks(actions)
    picks = _count_picks(actions)

    # check optimal knock count (report but don't fail benchmark)
    optK = optimal_knocks_full_grid(m,n)
    if debug:
        print(f"[RECT OPT] knocks={knocks} expected={optK}")

    return Metrics(ok=ok and sanity,
                   mode="exact",
                   knocks=knocks,
                   picks=picks,
                   steps=len(actions),
                   t_cover=0.0,
                   t_exec=(t1-t0),
                   t_total=(t1-t0),
                   extra=0,
                   fallback=0)


def _fmt_sec(x: float) -> str:
    return f"{x*1000.0:.2f}ms" if x < 1.0 else f"{x:.3f}s"

def print_full_grid_table(results: List[Tuple[str,int,int,Metrics]]) -> None:
    """
    results: list of (name, m, n, metrics) for full grids (both exact+approx)
    """
    print("\n=== FULL GRID RESULTS ===")
    header = f"{'case':22s} {'algo':8s} {'|V|':>5s} {'knocks':>7s} {'steps':>7s} {'t_cover':>10s} {'t_exec':>10s} {'t_total':>10s} {'ok':>4s}"
    print(header)
    print("-"*len(header))
    for name,m,n,met in results:
        V = m*n
        print(f"{name:22s} {met.mode:8s} {V:5d} {met.knocks:7d} {met.steps:7d} "
              f"{_fmt_sec(met.t_cover):>10s} {_fmt_sec(met.t_exec):>10s} {_fmt_sec(met.t_total):>10s} {str(met.ok):>4s}")

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs) if xs else 0.0

def print_subgraph_table(rows: List[Dict]) -> None:
    print("\n=== SUBGRAPH AVERAGES (20 random each) ===")
    header = f"{'hull':10s} {'k':>5s} {'N':>3s} {'knocks':>9s} {'steps':>9s} {'t_cover':>10s} {'t_exec':>10s} {'t_total':>10s} {'ok%':>6s}"
    print(header)
    print("-"*len(header))
    for r in rows:
        print(f"{r['hull']:10s} {r['k']:5d} {r['N']:3d} "
              f"{r['knocks_mean']:9.2f} {r['steps_mean']:9.2f} "
              f"{_fmt_sec(r['t_cover_mean']):>10s} {_fmt_sec(r['t_exec_mean']):>10s} {_fmt_sec(r['t_total_mean']):>10s} "
              f"{r['ok_rate']*100:5.1f}%")

def run_benchmark(visualize: bool = False, pause: float = 0.15, debug: bool = False) -> None:
    # Full grid suite (run once each)
    full_cases = [
        ("full_3x3", 3, 3),
        ("full_5x5", 5, 5),
        ("full_10x5", 10, 5),
        ("full_10x10", 10, 10),
        ("full_20x10", 20, 10),
        ("full_20x20", 20, 20),
    ]

    full_results: List[Tuple[str,int,int,Metrics]] = []
    for name,m,n in full_cases:
        print(f"\n==================== FULL GRID: {name} ====================")
        print(f"[INFO] optimal_knocks = ceil((m-1)(n-1)/2) = {optimal_knocks_full_grid(m,n)}")

        met_exact = run_exact_full_grid(m, n, debug=debug, visualize=visualize, pause=pause, title=name)
        met_approx = run_approx_solver(occ_full(m,n), m, n, debug=debug, visualize=visualize, pause=pause, title=name)

        full_results.append((name,m,n,met_exact))
        full_results.append((name,m,n,met_approx))

    print_full_grid_table(full_results)

    # Subgraph suite
    # NOTE: user wrote last as "20"; we use 200 as the natural corresponding size for 20x20.
    # You can change this list easily if you really want k=20 on 20x20.
    sub_suites = [
        ("3x3", 3, 3, 5),
        ("5x5", 5, 5, 16),
        ("10x5", 10, 5, 25),
        ("10x10", 10, 10, 50),
        ("20x10", 20, 10, 100),
        ("20x20", 20, 20, 200),
    ]

    sub_rows: List[Dict] = []
    for hull_name, m, n, k in sub_suites:
        print(f"\n==================== SUBGRAPHS: hull {hull_name}, k={k} ====================")
        N = 20
        knocks_list: List[float] = []
        steps_list: List[float] = []
        tc_list: List[float] = []
        te_list: List[float] = []
        tt_list: List[float] = []
        ok_count = 0

        for s in range(N):
            occ = random_hole_free_subgraph_by_removal(m, n, k, seed=1000 + 97*s)

            # visualize only the first instance per size (if requested)
            viz_this = visualize and (s == 0)

            met = run_approx_solver(occ, m, n, debug=debug, visualize=viz_this, pause=pause, title=f"sub_{hull_name}_k{k}_#{s}")
            ok_count += int(met.ok)

            knocks_list.append(float(met.knocks))
            steps_list.append(float(met.steps))
            tc_list.append(float(met.t_cover))
            te_list.append(float(met.t_exec))
            tt_list.append(float(met.t_total))

        row = dict(
            hull=hull_name,
            k=k,
            N=N,
            knocks_mean=mean(knocks_list),
            steps_mean=mean(steps_list),
            t_cover_mean=mean(tc_list),
            t_exec_mean=mean(te_list),
            t_total_mean=mean(tt_list),
            ok_rate=ok_count / N,
        )
        sub_rows.append(row)

    print_subgraph_table(sub_rows)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-benchmark", action="store_true", help="run the full synthetic benchmark suite")
    ap.add_argument("--visualize", action="store_true", help="visualize solutions (full grids + first subgraph per size)")
    ap.add_argument("--pause", type=float, default=0.15)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.run_benchmark:
        print("Nothing to do. Run with: --run-benchmark  (optionally --visualize --debug)")
        return

    run_benchmark(visualize=args.visualize, pause=args.pause, debug=args.debug)

if __name__ == "__main__":
    main()

