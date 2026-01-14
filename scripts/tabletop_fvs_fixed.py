#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Set, Optional
import random

import networkx as nx
import matplotlib.pyplot as plt


# =============================================================================
# Basic graph helpers
# =============================================================================

def build_grid_graph(m: int, n: int) -> nx.Graph:
    G = nx.grid_2d_graph(m, n)
    for (i, j) in G.nodes():
        G.nodes[(i, j)]["pos"] = (float(j), float(-i))
    return G


def cleanup_deg_le_1(G: nx.Graph) -> None:
    """Paper cleanup: iteratively remove degree<=1 vertices."""
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


# =============================================================================
# Pick/knock constraints for tabletop
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


# =============================================================================
# Weights (layered, deterministic tie-break)
# =============================================================================

def layer_index(i: int, j: int, m: int, n: int) -> int:
    return 1 + min(i, j, m - 1 - i, n - 1 - j)


def generate_layer_weights(
    m: int,
    n: int,
    corner_bias: float = 0.05,
    det_eps: float = 1e-3,
    rand_eps: float = 0.0,
    seed: int = 0,
) -> Dict[Any, float]:
    """
    Lower weight = cheaper to knock (preferred by FEEDBACK).
    Outer layer has smaller layer index -> cheaper.
    """
    assert corner_bias + det_eps + rand_eps < 1.0
    rng = random.Random(seed)
    denom = max(1, m * n - 1)

    w: Dict[Any, float] = {}
    '''
    for i in range(m):
        for j in range(n):
            L = float(layer_index(i, j, m, n))
            is_corner = (i in (0, m - 1)) and (j in (0, n - 1))
            det = det_eps * ((i * n + j) / denom)
            jit = rng.random() * rand_eps
            w[(i, j)] = L + (corner_bias if is_corner else 0.0) + det + jit
    '''

    B = 3 * m * n + 4
    det_eps = 1e-3
    for i in range(m):
        for j in range(n):
            L = float(B**(layer_index(i, j, m, n))) + det_eps * (i * n + j)
            w[(i, j)] = L
    return w


# =============================================================================
# FEEDBACK (paper) but returning both:
#   - F_final (after redundancy removal)
#   - F_all   (all zero-weight vertices ever pushed)
#   - push_order (stack push order)
# =============================================================================

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

        # numeric guard
        for u in w:
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

    # redundancy removal (paper)
    while stack:
        u = stack.pop()
        if u in F and is_fvs_for_original(G_original, F - {u}):
            F.remove(u)

    if debug:
        print(f"[FEEDBACK] |F_all|={len(F_all)} |F_final|={len(F)}")

    return FeedbackTrace(F_all=F_all, F_final=F, push_order=push_order)


# =============================================================================
# Execution simulator: remove all blocks using picks + knocks
#   - If allow_outside_knock=False, we MUST knock only from knockset K.
# =============================================================================

@dataclass
class ExecStep:
    remaining: Set[Any]
    picked: Set[Any]
    knocked: Set[Any]
    last_action: Optional[Tuple[Any, str]]
    remaining_K: Set[Any]


def execute_with_knockset(
    G_full: nx.Graph,
    w_order: Dict[Any, float],
    K: Set[Any],
    allow_outside_knock: bool = False,
    debug_deadlock: bool = False,
) -> Tuple[bool, List[Tuple[Any, str]], List[ExecStep]]:
    G = G_full.copy()
    picked: Set[Any] = set()
    knocked: Set[Any] = set()
    actions: List[Tuple[Any, str]] = []
    hist: List[ExecStep] = []

    def record(last_action: Optional[Tuple[Any, str]]):
        hist.append(ExecStep(
            remaining=set(G.nodes()),
            picked=set(picked),
            knocked=set(knocked),
            last_action=last_action,
            remaining_K=set(u for u in K if G.has_node(u))
        ))

    record(None)

    while G.number_of_nodes() > 0:
        # greedily pick everything pickable
        progressed = True
        while progressed and G.number_of_nodes() > 0:
            progressed = False
            pickables = [u for u in G.nodes() if is_pickable(G, u)]
            if pickables:
                pickables.sort(key=lambda u: (w_order.get(u, 0.0), u[0], u[1]))
                u = pickables[0]
                actions.append((u, "pick"))
                picked.add(u)
                G.remove_node(u)
                record((u, "pick"))
                progressed = True

        if G.number_of_nodes() == 0:
            break

        # need a knock
        candidates = [u for u in K if G.has_node(u) and is_knockable(G, u)]
        if not candidates:
            if debug_deadlock:
                remK = [u for u in K if G.has_node(u)]
                print("[DEADLOCK] No knockable in K.")
                print("  remaining_K:", remK)
                for u in remK:
                    print("   ", u, "deg=", G.degree(u))
            if not allow_outside_knock:
                return False, actions, hist

            # fallback outside K
            outside = [u for u in G.nodes() if is_knockable(G, u)]
            if not outside:
                return False, actions, hist
            outside.sort(key=lambda u: (w_order.get(u, 0.0), u[0], u[1]))
            u = outside[0]
            actions.append((u, "knock_outside"))
            knocked.add(u)
            G.remove_node(u)
            record((u, "knock_outside"))
            continue

        candidates.sort(key=lambda u: (w_order.get(u, 0.0), u[0], u[1]))
        u = candidates[0]
        actions.append((u, "knock"))
        knocked.add(u)
        G.remove_node(u)
        record((u, "knock"))

    return True, actions, hist


# =============================================================================
# Feasibility-aware inverse pruning on K (key fix)
#   Remove u from K if we can still finish without knocking outside K\{u}.
# =============================================================================

def prune_knockset_feasibility(
    G_full: nx.Graph,
    w_order: Dict[Any, float],
    K_init: Set[Any],
    push_order: List[Any],
    debug: bool = False,
) -> Set[Any]:
    K = set(K_init)

    # iterate in reverse push order (mirrors paper's inverse pass structure)
    for u in reversed(push_order):
        if u not in K:
            continue
        K_try = set(K)
        K_try.remove(u)
        ok, _, _ = execute_with_knockset(G_full, w_order, K_try, allow_outside_knock=False)
        if ok:
            K = K_try
            if debug:
                print(f"[PRUNE] removed {u}, |K|={len(K)}")
    return K


# =============================================================================
# Visualization
# =============================================================================

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
            # orange halo = still in knockset and still remaining
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

            # small offset to avoid overlap with marker
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
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )

        ax.set_axis_off()
        plt.pause(pause)

    plt.ioff()
    plt.show()


# =============================================================================
# Demo / tests
# =============================================================================

def run_case(m: int, n: int, visualize: bool = True) -> None:
    print(f"\n=== Case {m}x{n} ===")
    G = build_grid_graph(m, n)
    w = generate_layer_weights(m, n, corner_bias=0.05, det_eps=1e-3, rand_eps=0.0, seed=0)

    trace = feedback_with_trace(G, w, debug=True)

    # (A) Using F_final only (paper-bounded FVS) — may deadlock under knockability constraint
    okA, actionsA, histA = execute_with_knockset(G, w, trace.F_final, allow_outside_knock=False, debug_deadlock=True)
    outsideA = sum(1 for _, a in actionsA if a == "knock_outside")
    knocksA = sum(1 for _, a in actionsA if a == "knock")
    print(f"[EXEC using F_final] knocks_in_F={knocksA}, knocks_outside={outsideA}, total_steps={len(actionsA)}")

    # (B) Use K_init = F_all (pre-redundancy stack zeros) then prune by feasibility
    K_init = set(trace.F_all)
    K = prune_knockset_feasibility(G, w, K_init, trace.push_order, debug=True)
    print(f"[K] |F_final|={len(trace.F_final)} |F_all|={len(trace.F_all)} |K_pruned|={len(K)}")

    print("|F_all|=", len(trace.F_all), ",", trace.F_all)
    print("|F_final|=", len(trace.F_final), ",", trace.F_final)
    print("|K|=", len(K), ",", K)
    okB, actionsB, histB = execute_with_knockset(G, w, K, allow_outside_knock=False, debug_deadlock=True)
    if not okB:
        print("[ERROR] Still deadlocked even with pruned K (should be rare). Try disabling pruning or adjusting weights.")
        return

    knocksB = sum(1 for _, a in actionsB if a == "knock")
    picksB = sum(1 for _, a in actionsB if a == "pick")
    print(f"[EXEC using K_pruned] knocks={knocksB}, picks={picksB}, total_steps={len(actionsB)}")
    print("First 30 actions:")
    for u, a in actionsB[:30]:
        print(f"  {a:5s} {u}")
    if len(actionsB) > 30:
        print("  ...")

    if visualize:
        visualize_exec_history(G, histA, pause=0.20, title=f"{m}x{n} tabletop")


if __name__ == "__main__":
    run_case(3, 3, visualize=True)
    run_case(5, 5, visualize=True)
    run_case(6, 6, visualize=True)
    run_case(6, 8, visualize=True)
    run_case(7, 7, visualize=True)
    run_case(9, 9, visualize=True)
    run_case(20, 20, visualize=True)

