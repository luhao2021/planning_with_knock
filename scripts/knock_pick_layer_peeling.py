#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Set, Optional
import random
import collections

import networkx as nx
import matplotlib.pyplot as plt


# =============================================================================
# Build graphs (rectangle or arbitrary polyomino in a bounding box)
# =============================================================================

def build_polyomino_graph(cells: Set[Tuple[int, int]]) -> nx.Graph:
    """Build adjacency graph for an occupied cell set."""
    G = nx.Graph()
    for (i, j) in cells:
        G.add_node((i, j))
        # visualization position
        G.nodes[(i, j)]["pos"] = (float(j), float(-i))
    for (i, j) in cells:
        for v in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
            if v in cells:
                G.add_edge((i, j), v)
    return G


def build_grid_graph(m: int, n: int) -> nx.Graph:
    """Full m-by-n rectangle."""
    cells = {(i, j) for i in range(m) for j in range(n)}
    return build_polyomino_graph(cells)


# =============================================================================
# Pick / knock predicates
# =============================================================================

def is_I_shape_degree_two(G: nx.Graph, u: Any) -> bool:
    if G.degree(u) != 2:
        return False
    v1, v2 = list(G.neighbors(u))
    x1, y1 = G.nodes[v1]["pos"]
    x2, y2 = G.nodes[v2]["pos"]
    return (x1 == x2) or (y1 == y2)


def is_pickable(G: nx.Graph, u: Any) -> bool:
    d = G.degree(u)
    if d <= 1:
        return True
    if d == 2 and is_I_shape_degree_two(G, u):
        return True
    return False


def is_knockable(G: nx.Graph, u: Any) -> bool:
    return G.degree(u) <= 3


def has_open_side(u: Tuple[int, int], occ: Set[Tuple[int, int]]) -> bool:
    i, j = u
    return ((i - 1, j) not in occ) or ((i + 1, j) not in occ) or ((i, j - 1) not in occ) or ((i, j + 1) not in occ)


# =============================================================================
# Layer assignment by boundary peeling (erosion)
# =============================================================================
def assign_layers_by_hull_padding(cells: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """
    Layer index computed w.r.t. the convex hull rectangle (axis-aligned bbox):
      layer(u) = 1 + min(dist to top, left, bottom, right hull boundary)

    This corresponds to "pad to full grid (convex hull) first, then assign layer to original subgraph".
    Concave boundary cells that are far from the hull boundary become inner layers.
    """
    if not cells:
        return {}

    min_i = min(i for i, _ in cells)
    max_i = max(i for i, _ in cells)
    min_j = min(j for _, j in cells)
    max_j = max(j for _, j in cells)

    layer: Dict[Tuple[int, int], int] = {}
    for (i, j) in cells:
        di = i - min_i
        dj = j - min_j
        ui = max_i - i
        uj = max_j - j
        layer[(i, j)] = 1 + min(di, dj, ui, uj)
    return layer


def assign_layers_by_peeling(cells: Set[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
    """
    Layer 1 = boundary cells (has_open_side).
    Remove them; Layer 2 = boundary of remaining; ...
    Works for concave/convex hole-free polyominoes.
    """
    remaining = set(cells)
    layer: Dict[Tuple[int, int], int] = {}
    k = 1
    while remaining:
        boundary = {u for u in remaining if has_open_side(u, remaining)}
        # If somehow nothing is boundary (shouldn't happen in finite grid), break.
        if not boundary:
            # assign rest same layer to avoid infinite loop
            for u in remaining:
                layer[u] = k
            break
        for u in boundary:
            layer[u] = k
        remaining -= boundary
        k += 1
    return layer


# =============================================================================
# Outer boundary cycle tracing (grid-edge contour), and boundary cell order
# =============================================================================

def _boundary_edges_and_map(occ: Set[Tuple[int, int]]):
    """
    Build:
      - adjacency of lattice points along boundary edges
      - map from an undirected boundary edge to its inside cell (unique)
    Lattice points are (r,c) on the grid of corners.
    """
    edge_to_cell: Dict[frozenset, Tuple[int, int]] = {}
    adj = collections.defaultdict(set)

    for (i, j) in occ:
        # North side (between (i,j) and (i,j+1)) if neighbor above missing
        if (i - 1, j) not in occ:
            p, q = (i, j), (i, j + 1)
            edge_to_cell[frozenset((p, q))] = (i, j)
            adj[p].add(q); adj[q].add(p)

        # South side (between (i+1,j) and (i+1,j+1)) if neighbor below missing
        if (i + 1, j) not in occ:
            p, q = (i + 1, j), (i + 1, j + 1)
            edge_to_cell[frozenset((p, q))] = (i, j)
            adj[p].add(q); adj[q].add(p)

        # West side (between (i,j) and (i+1,j)) if neighbor left missing
        if (i, j - 1) not in occ:
            p, q = (i, j), (i + 1, j)
            edge_to_cell[frozenset((p, q))] = (i, j)
            adj[p].add(q); adj[q].add(p)

        # East side (between (i,j+1) and (i+1,j+1)) if neighbor right missing
        if (i, j + 1) not in occ:
            p, q = (i, j + 1), (i + 1, j + 1)
            edge_to_cell[frozenset((p, q))] = (i, j)
            adj[p].add(q); adj[q].add(p)

    return adj, edge_to_cell


def _walk_cycle(adj: Dict[Tuple[int, int], Set[Tuple[int, int]]], comp: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Walk a degree-2 component into a lattice-point cycle (unordered start/orientation)."""
    start = min(comp)
    nbrs = sorted(adj[start])
    if not nbrs:
        return []
    prev = start
    curr = nbrs[0]
    cycle = [start, curr]
    while curr != start:
        n1, n2 = tuple(adj[curr])
        nxt = n1 if n1 != prev else n2
        prev, curr = curr, nxt
        if curr == start:
            break
        cycle.append(curr)
        if len(cycle) > 10_000_000:
            raise RuntimeError("Boundary cycle walk too long; malformed boundary graph?")
    return cycle


def _polygon_area(cycle: List[Tuple[int, int]]) -> float:
    """Shoelace on lattice points (x=c, y=r)."""
    if len(cycle) < 3:
        return 0.0
    area2 = 0
    pts = cycle[:] + [cycle[0]]
    for (r1, c1), (r2, c2) in zip(pts[:-1], pts[1:]):
        x1, y1 = c1, r1
        x2, y2 = c2, r2
        area2 += x1 * y2 - x2 * y1
    return 0.5 * area2


def boundary_cell_order_outer(occ: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Return boundary cells in cyclic order along the OUTER boundary.
    Works for simply-connected (hole-free) polyominoes; if multiple components exist,
    picks the one with largest |area|.
    """
    adj, edge_to_cell = _boundary_edges_and_map(occ)
    if not adj:
        return []

    # Connected components of boundary lattice graph
    verts = list(adj.keys())
    seen = set()
    best_cycle: List[Tuple[int, int]] = []
    best_area = -1.0

    for s in verts:
        if s in seen:
            continue
        stack = [s]
        comp = set([s])
        seen.add(s)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)

        # Prefer degree-2 components (cycles)
        if not all(len(adj[v]) == 2 for v in comp):
            continue

        cyc = _walk_cycle(adj, comp)
        if len(cyc) < 3:
            continue
        area = abs(_polygon_area(cyc))
        if area > best_area:
            best_area = area
            best_cycle = cyc

    # Fallback: if no perfect degree-2 component found, return empty
    if not best_cycle:
        return []

    # Map cycle edges -> inside cells, then compress duplicates into a cell order
    edge_cells: List[Tuple[int, int]] = []
    for t in range(len(best_cycle) - 1):
        p, q = best_cycle[t], best_cycle[t + 1]
        cell = edge_to_cell.get(frozenset((p, q)))
        if cell is not None:
            edge_cells.append(cell)
    # close
    cell = edge_to_cell.get(frozenset((best_cycle[-1], best_cycle[0])))
    if cell is not None:
        edge_cells.append(cell)

    ordered: List[Tuple[int, int]] = []
    for c in edge_cells:
        if not ordered or ordered[-1] != c:
            ordered.append(c)
    if len(ordered) > 1 and ordered[0] == ordered[-1]:
        ordered.pop()

    return ordered


# =============================================================================
# Visualization (kept from your code)
# =============================================================================

@dataclass
class ExecStep:
    remaining: Set[Any]
    picked: Set[Any]
    knocked: Set[Any]
    last_action: Optional[Tuple[Any, str]]
    remaining_K: Set[Any]


def visualize_exec_history(G_full: nx.Graph, hist: List[ExecStep], pause: float = 0.25, title: str = "") -> None:
    pos = nx.get_node_attributes(G_full, "pos")
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 12))

    for t, st in enumerate(hist):
        ax.clear()
        nx.draw_networkx_edges(G_full, pos=pos, ax=ax, alpha=0.10)

        rem = list(st.remaining)
        pk = list(st.picked)
        kn = list(st.knocked)
        remK = list(st.remaining_K.intersection(st.remaining))

        if rem:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=rem, ax=ax, node_size=360, alpha=0.75)
        if pk:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=pk, ax=ax, node_size=220, node_color="green", alpha=0.85)
        if kn:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=kn, ax=ax, node_size=220, node_color="red", alpha=0.85, node_shape="s")
        if remK:
            nx.draw_networkx_nodes(G_full, pos=pos, nodelist=remK, ax=ax, node_size=430, node_color="orange", alpha=0.25)

        if st.last_action is None:
            ax.set_title(f"{title} step {t}")
        else:
            u, act = st.last_action
            ax.set_title(f"{title} step {t}: {act} {u}")

        # --- draw step indices for ALL past actions ---
        for tt in range(t + 1):
            past_step = hist[tt]
            if past_step.last_action is None:
                continue

            u, act = past_step.last_action
            if u not in pos:
                continue

            x, y = pos[u]
            dx, dy = (0.08, 0.08)

            ax.text(
                x + dx,
                y + dy,
                str(tt),
                fontsize=14,
                fontweight="bold",
                color="green" if act == "pick" else "red",
                ha="center",
                va="center",
                zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        ax.set_axis_off()
        plt.pause(pause)

    plt.ioff()
    plt.show()


# =============================================================================
# Layer-peeling knock+pick with “日” pattern selection on boundary order
# =============================================================================

def layer_peeling_knock_pick_sequence(
    G_full: nx.Graph,
    debug: bool = True,
    layer_mode: str = "hull",  # "hull" or "erode"
) -> Tuple[List[Tuple[Any, str]], List[ExecStep]]:
    """
    Core algorithm you asked for:

    - Preassign layer index by boundary peeling (outer layer = 1, ...).
    - Process k=1..:
        * only pick/knock vertices whose layer == current k
        * greedy pick (deg<=1 or deg2 I-shape)
        * if stuck, trace outer boundary cycle and take the degree-3 boundary vertices
          in boundary order (these correspond to the “日 centers”),
          then knock a maximum non-overlapping set: every other one along that cycle.
        * repeat until current layer is empty; then proceed inward.
    Handles concave/convex hole-free subgraphs (simply connected polyominoes).
    """
    G = G_full.copy()
    cells0 = set(G.nodes())
    if layer_mode == "hull":
        layer = assign_layers_by_hull_padding(cells0)
    elif layer_mode == "erode":
        layer = assign_layers_by_peeling(cells0)
    else:
        raise ValueError("layer_mode must be 'hull' or 'erode'")

    picked: Set[Any] = set()
    knocked: Set[Any] = set()
    actions: List[Tuple[Any, str]] = []
    hist: List[ExecStep] = []

    planned_K: List[Any] = []

    def record(last_action: Optional[Tuple[Any, str]]):
        hist.append(ExecStep(
            remaining=set(G.nodes()),
            picked=set(picked),
            knocked=set(knocked),
            last_action=last_action,
            remaining_K=set(planned_K),
        ))

    record(None)

    while G.number_of_nodes() > 0:
        remaining = set(G.nodes())
        k = min(layer[u] for u in remaining)  # current layer to peel
        Lk = {u for u in remaining if layer[u] == k}

        if debug:
            print(f"\n[LAYER {k}] start: |remaining|={len(remaining)} |Lk|={len(Lk)}")

        # peel this layer completely
        while Lk:
            progressed = True
            while progressed and Lk:
                progressed = False
                # pick only within current layer
                pickables = [u for u in Lk if is_pickable(G, u)]
                if pickables:
                    pickables.sort()  # deterministic
                    u = pickables[0]
                    actions.append((u, "pick"))
                    picked.add(u)
                    G.remove_node(u)
                    record((u, "pick"))
                    progressed = True
                    remaining = set(G.nodes())
                    Lk = {v for v in remaining if layer[v] == k}

            if not Lk:
                break

            # stuck: need to knock within layer k.
            remaining = set(G.nodes())
            occ = remaining  # polyomino occupied set

            boundary_order = boundary_cell_order_outer(occ)
            # Filter boundary order to current layer nodes still present
            boundary_order = [u for u in boundary_order if u in Lk]

            # “日 centers”: degree-3 boundary vertices (most beneficial knocks)
            cand3 = [u for u in boundary_order if G.has_node(u) and G.degree(u) == 3 and is_knockable(G, u)]

            # If we have a boundary cycle list of cand3, the maximum non-overlapping “adjacent 日 patterns”
            # corresponds to taking every other candidate along this cyclic order (max independent set on the cycle).
            plan_even = cand3[0::2]
            plan_odd = cand3[1::2]
            planned_K = plan_even if len(plan_even) >= len(plan_odd) else plan_odd

            if debug:
                print(f"[LAYER {k}] stuck. boundary_len={len(boundary_order)} cand3={len(cand3)} plan={len(planned_K)}")

            # If cand3 is empty (common on thin concave boundaries), fall back to any knockable in layer (prefer higher degree)
            if not planned_K:
                candidates = [u for u in Lk if is_knockable(G, u)]
                if not candidates:
                    raise RuntimeError(f"Deadlock: no knockable in current layer {k}. Remaining layer nodes: {sorted(Lk)[:20]} ...")
                # prefer higher degree (3 > 2 > 1), tie by boundary order if available else lexicographic
                order_index = {u: idx for idx, u in enumerate(boundary_order)}
                candidates.sort(key=lambda u: (-G.degree(u), order_index.get(u, 10**9), u))
                planned_K = [candidates[0]]
                if debug:
                    print(f"[LAYER {k}] fallback knock plan: {planned_K[0]} deg={G.degree(planned_K[0])}")

            # Execute the plan sequentially (knock, then greedy pick again)
            while planned_K and Lk:
                u = planned_K.pop(0)
                if (not G.has_node(u)) or (u not in Lk):
                    continue
                actions.append((u, "knock"))
                knocked.add(u)
                G.remove_node(u)
                record((u, "knock"))

                remaining = set(G.nodes())
                Lk = {v for v in remaining if layer[v] == k}

                # after each knock, greedily pick what becomes pickable in this layer
                progressed2 = True
                while progressed2 and Lk:
                    progressed2 = False
                    pickables2 = [x for x in Lk if is_pickable(G, x)]
                    if pickables2:
                        pickables2.sort()
                        x = pickables2[0]
                        actions.append((x, "pick"))
                        picked.add(x)
                        G.remove_node(x)
                        record((x, "pick"))
                        progressed2 = True
                        remaining = set(G.nodes())
                        Lk = {v for v in remaining if layer[v] == k}

            # recompute Lk (loop continues; if still stuck it will plan again)
            remaining = set(G.nodes())
            Lk = {v for v in remaining if layer[v] == k}

        planned_K = []

        if debug:
            print(f"[LAYER {k}] done.")

    return actions, hist


# =============================================================================
# FEEDBACK (paper) - keep as your unweighted standard FVS
# =============================================================================

def cleanup_deg_le_1(G: nx.Graph) -> None:
    changed = True
    while changed:
        changed = False
        for u in list(G.nodes()):
            if G.degree(u) <= 1:
                G.remove_node(u)
                changed = True


def find_semidisjoint_cycle(G: nx.Graph) -> List[Any]:
    if G.number_of_edges() == 0:
        return []
    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNoCycle:
        return []
    for cyc in cycles:
        non_deg2 = sum(1 for u in cyc if G.degree(u) != 2)
        if non_deg2 <= 1:
            return cyc
    return []


def is_fvs_for_original(G_original: nx.Graph, F: Set[Any]) -> bool:
    H = G_original.copy()
    H.remove_nodes_from(F)
    return nx.is_forest(H)


def generate_uniform_weights(G: nx.Graph, det_eps: float = 1e-6) -> Dict[Any, float]:
    nodes_sorted = sorted(G.nodes())
    denom = max(1, len(nodes_sorted) - 1)
    w = {}
    for k, u in enumerate(nodes_sorted):
        w[u] = 1.0 + det_eps * (k / denom)
    return w


@dataclass
class FeedbackTrace:
    F_all: Set[Any]
    F_final: Set[Any]
    push_order: List[Any]


def feedback_with_trace(G_in: nx.Graph, w_in: Dict[Any, float], debug: bool = False) -> FeedbackTrace:
    G_original = G_in.copy()
    G = G_in.copy()
    w = {u: float(w_in[u]) for u in G.nodes()}

    F: Set[Any] = {u for u in G.nodes() if abs(w[u]) <= 1e-12}
    G.remove_nodes_from(F)

    stack: List[Any] = []
    push_order: List[Any] = []
    for u in F:
        stack.append(u)
        push_order.append(u)

    F_all: Set[Any] = set(F)

    cleanup_deg_le_1(G)

    while G.number_of_nodes() > 0:
        C = find_semidisjoint_cycle(G)
        if C:
            gamma = min(w[u] for u in C)
            for u in C:
                w[u] -= gamma
        else:
            ratios = [w[u] / (G.degree(u) - 1) for u in G.nodes() if G.degree(u) > 1]
            gamma = min(ratios) if ratios else 0.0
            for u in list(G.nodes()):
                d = G.degree(u)
                if d > 1:
                    w[u] -= gamma * (d - 1)

        for u in list(w.keys()):
            if w[u] < 0 and w[u] > -1e-9:
                w[u] = 0.0

        zeros = [u for u in G.nodes() if abs(w[u]) <= 1e-12]
        for u in zeros:
            G.remove_node(u)
            F.add(u)
            F_all.add(u)
            stack.append(u)
            push_order.append(u)

        cleanup_deg_le_1(G)

    # inverse pass
    while stack:
        u = stack.pop()
        if u in F and is_fvs_for_original(G_original, F - {u}):
            F.remove(u)

    if debug:
        print(f"[FEEDBACK] |F_all|={len(F_all)} |F_final|={len(F)}")

    return FeedbackTrace(F_all=F_all, F_final=F, push_order=push_order)


# =============================================================================
# Demo / tests
# =============================================================================

def run_case_rectangle(m: int, n: int, visualize: bool = True) -> None:
    print(f"\n=== Rectangle {m}x{n} ===")
    G = build_grid_graph(m, n)

    # (A) Layer-peeling knock+pick
    actionsA, histA = layer_peeling_knock_pick_sequence(G, debug=True)
    knocksA = sum(1 for _, a in actionsA if a == "knock")
    picksA = sum(1 for _, a in actionsA if a == "pick")
    print(f"[LAYER-PEEL] knocks={knocksA}, picks={picksA}, total_steps={len(actionsA)}")
    print("First 30 actions:")
    for u, a in actionsA[:30]:
        print(f"  {a:5s} {u}")
    if len(actionsA) > 30:
        print("  ...")

    if visualize:
        visualize_exec_history(G, histA, pause=0.20, title=f"{m}x{n} layer-peel knock+pick")

    # (B) Standard FVS (unweighted) via FEEDBACK
    wU = generate_uniform_weights(G)
    trace = feedback_with_trace(G, wU, debug=True)
    print(f"[UNWEIGHTED FVS] |F_final|={len(trace.F_final)}")

    # Optional: visualize FVS nodes (static overlay) by running a fake history of one frame
    if visualize:
        # One-frame history showing F_final as "remaining_K" halo
        st = ExecStep(
            remaining=set(G.nodes()),
            picked=set(),
            knocked=set(),
            last_action=None,
            remaining_K=set(trace.F_final),
        )
        visualize_exec_history(G, [st], pause=0.5, title=f"{m}x{n} FEEDBACK FVS (orange halo)")


def run_case_concave_example(visualize: bool = True) -> None:
    """
    A simple hole-free concave polyomino inside a 7x7 hull: an L-shape / U-shape style.
    """
    hull_m, hull_n = 7, 7
    # Start with full hull, carve a concavity (still hole-free)
    cells = {(i, j) for i in range(hull_m) for j in range(hull_n)}
    carve = {(i, j) for i in range(2, 6) for j in range(2, 6)}  # remove a 4x4 block
    # Put back a corridor so it remains hole-free (no enclosed cavity)
    carve -= {(2, 2), (2, 3), (2, 4)}  # open to boundary
    cells -= carve

    G = build_polyomino_graph(cells)
    print(f"\n=== Concave hole-free example in {hull_m}x{hull_n} hull: |V|={G.number_of_nodes()} ===")

    actions, hist = layer_peeling_knock_pick_sequence(G, debug=True)
    knocks = sum(1 for _, a in actions if a == "knock")
    picks = sum(1 for _, a in actions if a == "pick")
    print(f"[LAYER-PEEL concave] knocks={knocks}, picks={picks}, total_steps={len(actions)}")

    if visualize:
        visualize_exec_history(G, hist, pause=0.05, title="Concave hole-free layer-peel")


if __name__ == "__main__":
    run_case_rectangle(3, 3, visualize=True)
    run_case_rectangle(5, 5, visualize=True)
    run_case_rectangle(6, 4, visualize=True)
    run_case_rectangle(9, 9, visualize=True)

    # Extra (optional) concave hole-free test:
    run_case_concave_example(visualize=True)

