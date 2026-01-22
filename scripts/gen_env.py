#!/usr/bin/env python3
"""
Generate block manipulation config YAML (inline single-line entries).

Inputs:
  --m, --n         : grid rows/cols (k = m*n blocks)
  --size           : cube edge length (m)
  --gap_s          : initial edge-to-edge gap (m)
  --center_x, --center_y : initial grid center
  --seed           : RNG seed (default: 0)
  --output         : output YAML path (default: config.yaml)
  --precision      : decimals for floats in YAML (default: 3)

Rules:
- First k pads are the initial grid (1..k).
- Next  1 pad are the fixed goal pads.
- Goal pads is fixed out of table pad
- Names cycle: Red1, Green1, Blue1, Red2, ...
- Heights = 0
- z_base = size/2 ; z_increment = size
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


def make_names(k: int) -> List[str]:
    colors = ["Red", "Green", "Blue"]
    return [f"{colors[i % 3]}{i // 3 + 1}" for i in range(k)]


def fmt(x: float, prec: int) -> str:
    return f"{x:.{prec}f}"


def write_yaml_inline(path: str,
                      initial_centers: List[Tuple[float, float]],
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
    # Goal pad:
    lines.append(f"  {i+1}: {{x: -0.253, y: 0.457}}")

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
    lines.append("goal:")
    for i in range(k):
        lines.append(f"  - {{name: {names[i]}, pad: {k+1}, height: 0}}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--size", type=float, required=True)
    ap.add_argument("--gap_s", type=float, required=True)
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

    write_yaml_inline(args.output, initial_centers, size, args.precision)
    print(f"Wrote {args.output} with {k} initial pads and 1 goal pad (total {k+1}).")


if __name__ == "__main__":
    main()

