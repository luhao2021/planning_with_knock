#!/usr/bin/env python3
"""
Generate block manipulation config YAML (inline single-line entries).

Inputs:
  --m, --n         : grid rows/cols (k = m*n blocks)
  --size           : cube edge length (m)
  --gap_s          : initial edge-to-edge gap (m)
  --gap_d          : minimal edge-to-edge gap between goal blocks (m)
  --center_x, --center_y : initial grid center
  --seed           : RNG seed (default: 0)
  --output         : output YAML path (default: config.yaml)
  --precision      : decimals for floats in YAML (default: 3)

Rules:
- First k pads are the initial grid (1..k).
- Next  k pads are the goal pads (k+1..2k).
- Goal pads must:
  * fit entirely inside box [(0,-0.8),(0.8,0.8)]
  * not overlap any initial pad
  * be at least gap_d edge-to-edge apart from each other
- Names cycle: Red1, Green1, Blue1, Red2, ...
- Heights = 0
- z_base = size/2 ; z_increment = size
- Exits with error if valid goal layout is impossible.
"""

import argparse
import math
import random
import sys
from typing import List, Tuple

GOAL_X_MIN, GOAL_Y_MIN = 0.4, -0.6
GOAL_X_MAX, GOAL_Y_MAX = 0.6,  0.6


def edge_distance(a: Tuple[float, float], b: Tuple[float, float], s: float) -> float:
    """Edge-to-edge distance between two same-size squares (side s), centered at a and b."""
    dx = max(abs(a[0] - b[0]) - s, 0.0)
    dy = max(abs(a[1] - b[1]) - s, 0.0)
    return math.hypot(dx, dy)


def compute_initial_centers(m: int, n: int, size: float, gap_s: float,
                            cx: float, cy: float) -> List[Tuple[float, float]]:
    """m×n grid around (cx,cy) with edge gap gap_s (works for even/odd m,n)."""
    pitch = size + gap_s
    xs = [cx + (j - (n - 1)/2.0) * pitch for j in range(n)]
    ys = [cy + ((m - 1)/2.0 - i) * pitch for i in range(m)]
    return [(xs[j], ys[i]) for i in range(m) for j in range(n)]


def capacity_estimate(size: float, gap_d: float) -> int:
    """Rough capacity if packed on a grid with pitch=(size+gap_d), fully inside region."""
    usable_w = (GOAL_X_MAX - GOAL_X_MIN) - size
    usable_h = (GOAL_Y_MAX - GOAL_Y_MIN) - size
    if usable_w < 0 or usable_h < 0:
        return 0
    pitch = size + gap_d
    nx = int(usable_w // pitch) + 1
    ny = int(usable_h // pitch) + 1
    return max(nx, 0) * max(ny, 0)


def generate_goal_centers(k: int, size: float, gap_d: float,
                          initial: List[Tuple[float, float]],
                          seed: int) -> List[Tuple[float, float]]:
    """Rejection sampling of goal centers inside region under constraints."""
    rng = random.Random(seed)

    cap = capacity_estimate(size, gap_d)
    if k > cap:
        raise RuntimeError(
            f"Goal area too tight: capacity≈{cap} with size={size} and gap_d={gap_d}, need {k}."
        )

    half = size / 2.0
    xmin, xmax = GOAL_X_MIN + half, GOAL_X_MAX - half
    ymin, ymax = GOAL_Y_MIN + half, GOAL_Y_MAX - half

    goals: List[Tuple[float, float]] = []
    max_tries = max(100000, 4000 * k)
    tries = 0
    while len(goals) < k and tries < max_tries:
        tries += 1
        c = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))

        # 1) No overlap with any initial square
        if any(edge_distance(c, ci, size) <= 0.1 for ci in initial):
            continue

        # 2) Pairwise >= gap_d (edge-to-edge) to already placed goals
        if any(edge_distance(c, g, size) < gap_d for g in goals):
            continue

        goals.append(c)

    if len(goals) < k:
        raise RuntimeError(
            "Failed to place all goal pads with constraints. "
            "Try reducing --gap_d/--size or use a different seed."
        )

    return goals


def make_names(k: int) -> List[str]:
    colors = ["Red", "Green", "Blue"]
    return [f"{colors[i % 3]}{i // 3 + 1}" for i in range(k)]


def fmt(x: float, prec: int) -> str:
    return f"{x:.{prec}f}"


def write_yaml_inline(path: str,
                      initial_centers: List[Tuple[float, float]],
                      goal_centers: List[Tuple[float, float]],
                      size: float,
                      prec: int) -> None:
    k = len(initial_centers)
    zb = size / 2.0
    zi = size

    lines: List[str] = []
    lines.append("pads:")
    # Initial pads: 1..k
    for i, (x, y) in enumerate(initial_centers, start=1):
        lines.append(f"  {i}: {{x: {fmt(x,prec)}, y: {fmt(y,prec)}}}")
    # Goal pads: k+1..2k
    for i, (x, y) in enumerate(goal_centers, start=k+1):
        lines.append(f"  {i}: {{x: {fmt(x,prec)}, y: {fmt(y,prec)}}}")

    # z values
    lines.append("")
    # Keep a little spacing flair like the sample (not required)
    zbase_str = fmt(zb, prec)
    zinc_str  = fmt(zi, prec)
    pad_space = " " * max(0, 12 - len(zbase_str))  # cosmetic spacing
    lines.append(f"z_base:      {zbase_str}{pad_space}# table + 1/2 block")
    lines.append(f"z_increment: {zinc_str}     # full block height")
    lines.append("# ----------------------------------------------------------")
    lines.append("")

    # initial mapping list
    names = make_names(k)
    lines.append("initial:")
    for i in range(k):
        lines.append(f"  - {{name: {names[i]}, pad: {i+1}, height: 0}}")

    lines.append("")
    # goal mapping list (pads k+1..2k)
    lines.append("goal:")
    for i in range(k):
        lines.append(f"  - {{name: {names[i]}, pad: {k+i+1}, height: 0}}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--size", type=float, required=True)
    ap.add_argument("--gap_s", type=float, required=True)
    ap.add_argument("--gap_d", type=float, required=True)
    ap.add_argument("--center_x", type=float, required=True)
    ap.add_argument("--center_y", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="config.yaml")
    ap.add_argument("--precision", type=int, default=3,
                    help="decimal places for floats in YAML (default: 3)")
    args = ap.parse_args()

    if args.m <= 0 or args.n <= 0:
        sys.exit("m and n must be positive.")

    k = args.m * args.n
    size = args.size

    initial_centers = compute_initial_centers(args.m, args.n, size, args.gap_s,
                                              args.center_x, args.center_y)
    try:
        goal_centers = generate_goal_centers(k, size, args.gap_d,
                                             initial_centers, args.seed)
    except RuntimeError as e:
        sys.exit(f"ERROR: {e}")

    write_yaml_inline(args.output, initial_centers, goal_centers, size, args.precision)
    print(f"Wrote {args.output} with {k} initial pads and {k} goal pads (total {2*k}).")


if __name__ == "__main__":
    main()

