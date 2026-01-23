#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict
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

def occ_narrow_concaves_20x20() -> Tuple[Set[Node], int, int]:
    """
    20x20 hull with very narrow concaves (1-cell-wide bays) + multiple holes.
    """
    m, n = 20, 20
    concaves = [
        (0, 6, 8, 7),    # narrow vertical bay from top
        (12, 0, 13, 9),  # narrow horizontal bay from left
        (19-7, 19-1, 20, 20),  # tiny corner shaving
        (8, 19-6, 9, 20),      # thin bay from right
    ]
    holes = [
        (6, 6, 9, 9),      # 3x3 hole
        (10, 12, 13, 15),  # 3x3 hole
        (3, 14, 5, 17),    # 2x3 hole
        (14, 3, 17, 5),    # 3x2 hole
    ]
    return occ_multi_concave_holes(m, n, concaves, holes), m, n

def occ_multi_notches_multi_holes_20x20(seed: int = 0) -> Tuple[Set[Node], int, int]:
    """
    Random-ish stress: several boundary notches + several interior holes, varied sizes.
    """
    rng = random.Random(seed)
    m, n = 20, 20
    concaves = []
    holes = []

    # boundary notches
    for _ in range(6):
        side = rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            j0 = rng.randrange(1, n-2)
            w = rng.randrange(1, 4)
            d = rng.randrange(1, 7)  # depth
            j1 = min(n-1, j0+w)
            if side == "top":
                concaves.append((0, j0, d, j1))
            else:
                concaves.append((m-d, j0, m, j1))
        else:
            i0 = rng.randrange(1, m-2)
            h = rng.randrange(1, 4)
            d = rng.randrange(1, 7)
            i1 = min(m-1, i0+h)
            if side == "left":
                concaves.append((i0, 0, i1, d))
            else:
                concaves.append((i0, n-d, i1, n))

    # interior holes
    for _ in range(6):
        i0 = rng.randrange(2, m-6)
        j0 = rng.randrange(2, n-6)
        h = rng.randrange(2, 6)
        w = rng.randrange(2, 6)
        holes.append((i0, j0, min(m-2, i0+h), min(n-2, j0+w)))

    return occ_multi_concave_holes(m, n, concaves, holes), m, n

def occ_only_2x2_islands(m: int, n: int, step: int = 4) -> Set[Node]:
    """
    Disconnected occupancy: only isolated 2x2 blocks spaced apart so there are faces but no adjacent faces.
    """
    occ: Set[Node] = set()
    for i in range(0, m-1, step):
        for j in range(0, n-1, step):
            occ |= {(i, j), (i+1, j), (i, j+1), (i+1, j+1)}
    return occ

def occ_rect_ring(m: int, n: int, thickness: int = 1) -> Set[Node]:
    """
    "No gadgets" extreme: a 1-thick (or k-thick) rectangular ring on the hull boundary.
    For thickness=1 => no 2x2 faces anywhere, all nodes degree=2 except small variations at corners (still 2).
    """
    assert thickness >= 1
    occ: Set[Node] = set()
    for i in range(m):
        for j in range(n):
            if i < thickness or i >= m-thickness or j < thickness or j >= n-thickness:
                occ.add((i, j))
    return occ

def occ_cycle_like_disc(m: int, n: int, r_inner: float, r_outer: float) -> Set[Node]:
    """
    Approx 'circle ring' using Euclidean distance around center; may have some degree !=2,
    but typically faces sparse. Useful stress.
    """
    cx, cy = (m-1)/2.0, (n-1)/2.0
    occ: Set[Node] = set()
    for i in range(m):
        for j in range(n):
            d = ((i-cx)**2 + (j-cy)**2) ** 0.5
            if r_inner <= d <= r_outer:
                occ.add((i, j))
    # make sure it's not empty
    if not occ:
        occ.add((int(cx), int(cy)))
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
      - prefer candidates that lie on nodes of faces in outerF that gadget touches,
      - sort by (layer, -degree, coord),
      - pick first with some Ray direction feasible.
    """
    cand0 = [v for v in gadget_knock_candidates(g) if (v in coreU) and G.has_node(v)]
    if not cand0:
        return None

    for v in cand0:
        if G.degree(v) <= 3:
            dirs = ray_clear_directions(G, v, m, n)
            if dirs:
                dirs.sort()
                return v, dirs[0]

    return None

# ============================================================
# Execution from cover
# ============================================================

@dataclass
class ExecStep:
    remaining: Set[Node]
    picked: Set[Node]
    knocked: Set[Node]
    last_action: Optional[Tuple[Node, str]]
    note: str = ""

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
      4) Else report infeasible
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

        if debug:
            coreE = G.subgraph(coreU).number_of_edges()
            outerL = min((face_layer(f, m, n) for f in F), default=None)
            print(f"[EXEC] iter={it} remainingV={G.number_of_nodes()} coreV={len(coreU)} coreE={coreE} faces={len(F)} outerL={outerL} used={sum(used)}/{len(cover)}")

        if not coreU:
            continue

        # no faces => no gadgets
        if not F:
            # this should not happen, as faces are the minimal non-pickable unit in a clean graph
            print("[EXEC] no available faces (2x2) in clean core graph...")
            exit(-1)

        # executable UNUSED cover gadget
        next_knockable = None  # (idx, (v,d))
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

        if next_knockable is None:
            # this should not happen, but in case it happens, print the info and do the following extra and fallback
            print("[EXEC] no executable UNUSED cover gadget; searching EXTRA gadget on current core...")
        else:
            continue

        # ----------------------- SHOULD NOT REACH HERE -------------------------

        # executable EXTRA gadget on outermost layer
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
                    print(f"[REPLAY FAIL] step {t}: knock infeasible at {u}, deg={G.degree(u)}")
                    print("  ray dirs:", ray_clear_directions(G, u, m, n))
                return False
            G.remove_node(u)
    ok = (G.number_of_nodes() == 0)
    if not ok and verbose_fail:
        print("[REPLAY FAIL] ended with remaining nodes:", G.number_of_nodes())
    return ok


# ============================================================
# Visualization
# ============================================================

def gadget_patches(cover: List[FaceGadget]):
    """
    Your visualization rule:
      - 2x2 gadget: draw a smaller square centered at face center.
      - 2x3 / 3x2: draw a smaller rectangle centered at the middle point between the two middle nodes.
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
            # center between the two middle nodes: (i0, j0+1) and (i0+1, j0+1)
            cx, cy = (j0 + 1.0), -(i0 + 0.5)
            w = 1.00
            h = 0.50
            patches.append(Rectangle((cx - w / 2, cy - h / 2), w, h))
        elif g.kind == "3x2":
            # center between the two middle nodes: (i0+1, j0) and (i0+1, j0+1)
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
    pause: float = 0.25,
    title: str = "",
    show_faces: bool = False,
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

        # gadget markers
        for p in cover_markers:
            ax.add_patch(p)

        # optional: outline current core faces
        if show_faces:
            Gtmp = G_full.subgraph(st.remaining).copy()
            U = core_nodes(Gtmp)
            F = enumerate_faces(U, m, n)
            for f in F:
                i, j = f
                x0, y0 = float(j), float(-(i + 1))
                ax.add_patch(Rectangle((x0, y0), 1.0, 1.0, fill=False, linewidth=2.0, alpha=0.98))

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
            if act == "knock_extra" or act == "knock_fallback":
                print(u, act)
                print("-----------------------")
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

def visualize_cover_only(G_full: nx.Graph, cover: List[FaceGadget], title: str = "cover") -> None:
    pos = nx.get_node_attributes(G_full, "pos")
    fig, ax = plt.subplots(figsize=(11, 11))
    nx.draw_networkx_edges(G_full, pos=pos, ax=ax, alpha=0.10)
    nx.draw_networkx_nodes(G_full, pos=pos, nodelist=list(G_full.nodes()), ax=ax, node_size=250, alpha=0.55)

    patches = gadget_patches(cover)
    for p in patches:
        p.set_facecolor("none")
        p.set_edgecolor("grey")
        p.set_linewidth(2.0)
        p.set_alpha(0.35)
        ax.add_patch(p)

    ax.set_title(title)
    ax.set_axis_off()
    plt.show()


# ============================================================
# Pipeline
# ============================================================

def run_face_cover_pipeline(occ: Set[Node], m: int, n: int, debug: bool = True):
    G = build_graph(occ)
    U0 = core_nodes(G)
    F0 = enumerate_faces(U0, m, n)
    gadgets = enumerate_face_gadgets(U0, F0, m, n)

    if debug:
        cnt = collections.Counter(g.kind for g in gadgets)
        print(f"[INIT] |V|={G.number_of_nodes()} |E|={G.number_of_edges()} |Core|={len(U0)} |Faces|={len(F0)}")
        if len(F0) > 0:
            print(f"[INIT] face lower-bound ceil(|Faces|/2) = {(len(F0) + 1)//2}")
        print(f"[GADGETS-FACE] total={len(gadgets)} counts={dict(cnt)}")

    cover = greedy_set_cover_faces(F0, gadgets, m, n, debug=debug)

    if debug:
        print(f"[COVER-FACE] chosen={len(cover)} (each gadget should cost 1 knock)")

    ok, actions, hist, used, stats = execute_from_cover(G, cover, m, n, debug=debug)
    sanity = replay_sanity(occ, m, n, actions, verbose_fail=True)

    if debug:
        knocks = sum(1 for _, a in actions if "knock" in a)
        picks = sum(1 for _, a in actions if a == "pick")
        print(f"[RESULT] ok={ok} knocks={knocks} picks={picks} steps={len(actions)} extra={stats.get('extra',0)} fallback={stats.get('fallback',0)}")
        print(f"[SANITY] replay={sanity}")

    return G, cover, ok, actions, hist, stats, sanity


# ============================================================
# Named tests (expanded)
# ============================================================

def named_tests():
    tests = {}

    # baseline full grids
    tests["full_3x3"] = (occ_full(3, 3), 3, 3)
    tests["full_5x5"] = (occ_full(5, 5), 5, 5)
    tests["full_6x4"] = (occ_full(6, 4), 6, 4)
    tests["full_9x9"] = (occ_full(9, 9), 9, 9)
    tests["full_20x20"] = (occ_full(20, 20), 20, 20)

    # concave (your 7x7)
    tests["concave_L_7x7_cut"] = (occ_concave_L(7, 7, 0, 4, 3, 7), 7, 7)

    # more concaves/holes
    tests["hole_7x7_center3x3"] = (occ_with_hole(7, 7, 2, 2, 5, 5), 7, 7)
    tests["hole_9x9_center3x3"] = (occ_with_hole(9, 9, 3, 3, 6, 6), 9, 9)

    occ20a, m20, n20 = occ_narrow_concaves_20x20()
    tests["stress_20x20_narrow_concaves_holes"] = (occ20a, m20, n20)

    occ20b, m20, n20 = occ_multi_notches_multi_holes_20x20(seed=1)
    tests["stress_20x20_multi_notches_holes_seed1"] = (occ20b, m20, n20)

    occ20c, m20, n20 = occ_multi_notches_multi_holes_20x20(seed=2)
    tests["stress_20x20_multi_notches_holes_seed2"] = (occ20c, m20, n20)

    # only 2x2 gadgets (disconnected islands)
    tests["only_2x2_islands_20x20_step4"] = (occ_only_2x2_islands(20, 20, step=4), 20, 20)
    tests["only_2x2_islands_12x12_step3"] = (occ_only_2x2_islands(12, 12, step=3), 12, 12)

    # no gadgets: ring/cycle-like (faces=0)
    tests["ring_rect_20x20_t1"] = (occ_rect_ring(20, 20, thickness=1), 20, 20)
    tests["ring_rect_20x20_t2"] = (occ_rect_ring(20, 20, thickness=2), 20, 20)

    # approximate circle ring (may have very few faces; good stress)
    tests["ring_circle_like_30x30"] = (occ_cycle_like_disc(30, 30, r_inner=10.0, r_outer=11.0), 30, 30)

    # corner cases
    tests["tiny_1x1"] = (occ_full(1, 1), 1, 1)
    tests["tiny_1x5"] = (occ_full(1, 5), 1, 5)
    tests["tiny_2x2"] = (occ_full(2, 2), 2, 2)
    tests["tiny_2x3"] = (occ_full(2, 3), 2, 3)
    tests["tiny_3x2"] = (occ_full(3, 2), 3, 2)
    tests["tiny_2x10"] = (occ_full(2, 10), 2, 10)
    tests["tiny_10x2"] = (occ_full(10, 2), 10, 2)

    return tests


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=str, default="concave_L_7x7_cut", help="test name (or 'all')")
    ap.add_argument("--show-tests", action="store_true", help="show all available tests")
    ap.add_argument("--viz-cover", action="store_true", help="show cover-only visualization")
    ap.add_argument("--viz-merged", action="store_true", help="show merged animation (cover markers + sequence)")
    ap.add_argument("--show-faces", action="store_true", help="in merged animation also show current core faces")
    ap.add_argument("--pause", type=float, default=0.20)
    ap.add_argument("--no-debug", action="store_true")
    args = ap.parse_args()

    debug = (not args.no_debug)
    tests = named_tests()

    if args.show_tests:
        print("Available tests:")
        for k in tests:
            print(" ", k)
        return

    keys = list(tests.keys()) if args.test == "all" else [args.test]
    for name in keys:
        if name not in tests:
            print("Unknown test:", name)
            print("Available tests:")
            for k in tests:
                print(" ", k)
            return

        occ, m, n = tests[name]
        print("\n==================== TEST:", name, "====================")
        G_full, cover, ok, actions, hist, stats, sanity = run_face_cover_pipeline(occ, m, n, debug=debug)

        if args.viz_cover:
            visualize_cover_only(G_full, cover, title=f"{name} cover (|C|={len(cover)})")
        if args.viz_merged:
            visualize_exec_with_cover(G_full, hist, cover, m, n, pause=args.pause,
                                      title=f"{name} merged", show_faces=args.show_faces)

if __name__ == "__main__":
    main()

