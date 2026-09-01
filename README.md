<div align="center">

# Optimal Knock-Pick Planning

**for Tightly Packed Tabletop Blocks With Parallel Grippers**

[![Paper](https://img.shields.io/badge/arXiv-2605.17800-b31b1b.svg)](https://arxiv.org/abs/2605.17800)
![WAFR 2026](https://img.shields.io/badge/WAFR_2026-Accepted-2E8B57.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Research code for<br>
**[Optimal Knock-Pick Planning for Tightly Packed Tabletop Blocks With Parallel Grippers](https://arxiv.org/abs/2605.17800)**<br>
Hao Lu and Rahul Shome<br>
Accepted to the **The 17th World Symposium on the Algorithmic Foundations of Robotics (WAFR 2026)**.

</div>

## Overview

Tightly packed tabletop blocks expose a fundamental limitation of parallel grippers: without lateral clearance, a direct top-down pick may be impossible. Even a fully occupied 2 × 2 group can lock every block against picking. This work treats a controlled **knock** as a deliberate non-prehensile action that creates clearance, enabling ordinary pick-and-place manipulation to resume.

The key result is stronger than simply finding a feasible sequence:

> **Under the paper's grid and manipulation assumptions, the planner computes a globally optimal knock-pick plan in polynomial time.**

At first glance, choosing which blocks to knock—and in what order—appears to require a combinatorial search over action sequences. The algorithm avoids that search through a compact graph-theoretic abstraction:

1. Exhaustively remove every immediately pickable block to obtain the cleaned object-location graph.
2. Represent the remaining local picking constraints as occupied 2 × 2 faces and three minimal gadget types.
3. Reduce the minimum-knock problem to an optimal exact face cover.
4. Exploit the special singleton-or-pair structure of that cover to solve it with **maximum-weight perfect matching**.
5. Convert the optimal cover into a ray-feasible execution by interleaving each selected knock with the picks it enables.

This reduction is the conceptual elegance of the method: a manipulation problem that appears to demand exponential sequence search becomes a polynomial-time matching problem, while retaining a proof of global optimality. The resulting plan uses the minimum possible number of knocks for the cleaned instance. Since every block must eventually be picked and placed, this also minimizes the total number of actions up to the unavoidable one-pick-per-block constant.

Knocks are therefore used only when necessary and as economically as possible. This matters because a knock is more disruptive and less predictable than a pick; minimizing knocks improves the high-level plan without pretending that physical execution and contact dynamics are free of uncertainty.

## Results

The experiments support both parts of the result: **optimality** and **practical polynomial-time scalability**. Every reported knock count is globally optimal for the cleaned instance because it is obtained from the matching reduction—not from a greedy approximation or bounded search.

| Synthetic instance | Blocks | Globally optimal knocks | Total solver time |
| --- | ---: | ---: | ---: |
| Full 3 × 3 grid | 9 | 2 | 0.55 ms |
| Full 5 × 5 grid | 25 | 8 | 1.47 ms |
| Full 10 × 10 grid | 100 | 41 | 13.05 ms |
| Full 20 × 10 grid | 200 | 86 | 61.62 ms |
| Full 20 × 20 grid | 400 | 181 | 189.79 ms |

The largest reported full-grid problem contains 400 blocks and 361 constraining faces. Its globally optimal knock-pick sequence is computed in under 0.2 seconds on the paper's benchmark machine. Runtime grows smoothly with instance size, consistent with the polynomial-time maximum-weight matching step followed by efficient sequence construction.

For a full m × n grid, there are (m − 1)(n − 1) unit faces. The matching pairs adjacent faces whenever possible, producing the closed-form optimum of ⌈(m − 1)(n − 1)/2⌉ knocks. For irregular subgraphs, the same algorithm computes the global optimum rather than relying on this full-grid formula.

The paper's reported physical benchmark used ROS 2 and MoveIt. This repository's current execution path instead connects the same optimal high-level plans directly to Isaac Sim's native Franka controllers; the historical MoveIt implementation remains in the tree for paper traceability but is excluded from builds.

## Repository layout

```text
.
├── scripts/
│   ├── kp_planner_opt.py       # main optimal planner and benchmark
│   ├── run_bench.py            # repeated Isaac Sim orchestration
│   ├── summarize.py            # aggregate Isaac run results
│   └── gen_env.py              # standalone block-grid YAML helper
├── problems/                   # example/generated problem YAML
├── src/
│   ├── isaac_executor/         # native Isaac Sim pick/knock executor
│   ├── blocksworld_executor/   # historical MTC executor (COLCON_IGNORE)
│   ├── blocksworld_scene/      # historical planning-scene service (ignored)
│   └── isaac_moveit/           # historical ROS bridge (ignored)
├── tests/                      # planner/serialization regressions
├── requirements.txt
└── LICENSE
```

`scripts/kp_planner_opt.py` is the supported entry point. Other planner, layer-peeling, and feedback-vertex-set scripts are research-development variants retained for traceability.

## Installation

### Synthetic benchmark

Developed on Ubuntu 22.04 with Python 3.10:

```bash
git clone git@github.com:luhao2021/planning_with_knock.git
cd planning_with_knock
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Isaac Sim

The native executor targets Isaac Sim 5.0.0. It does not require ROS 2, MoveIt, MTC, a planning scene, or `ros2_control`. No build is required. A plain CMake install is optional if you want to install the launcher script:

```bash
cmake -S src/isaac_executor -B build/isaac_executor
cmake --install build/isaac_executor --prefix install/isaac_executor
```

## Quick start and synthetic benchmark

Run one deterministic case:

```bash
python3 scripts/kp_planner_opt.py --one full_3_3
```

Run the mixed-height suite. It covers every square hull from 3 × 3 through
20 × 20, assigns each block a reproducible random height from `{1, 2, 3, 4}`
side lengths, and generates 20 connected, hole-free subgraphs per size. Half
of the subgraphs protect an equal-height 2 × 2, 2 × 3, or 3 × 2 gadget so the
suite includes knock-required cases; unbiased cases are allowed to require
zero knocks.

```bash
python3 scripts/kp_planner_opt.py --bench synthetic --seed 0
```

Use `--visualize` to save two complete views per problem. Aligned heatmaps for
the height-agnostic knock degree and height-aware picking degree are written to
`results/degree_sequences/`. Compressed graph sequences are written to
`results/graph_sequences/`: every node shows its height, consecutive picks are
merged into one updated graph, and every knock receives its own graph with a
direction arrow. Change the count or output directories with
`--synthetic-problems-per-size`, `--degree-plot-dir`, and `--graph-plot-dir`.
Use `--debug` for planner diagnostics. `t_cover` measures cleaned/face graph
and matching work, `t_exec` measures cover-to-sequence conversion, and
`replay_sanity` independently replays action feasibility.

## Isaac Sim benchmark

```text
Problem generation → optimal planner → knock/pick sequence + cuboid geometry
        → problems/*.yaml → native Isaac Sim controllers → execution
```

Generate the full 3 × 3 and 5 × 5 problems plus five random subgraphs of each size:

```bash
python3 scripts/kp_planner_opt.py --bench isaac --outdir problems --seed 0
```

Regenerate existing problem files after changing the side length or height multipliers; the executor rejects inconsistent manually authored geometry.

Each YAML stores pads, per-object dimensions, initial/goal assignments, and ordered actions. A planned knock is `Knock` followed by the `Move` that picks and places the displaced block. The executor also reads the historical `Smash` token for backward compatibility. Cuboid heights are integer multiples of their side length. Generated objects default to 1×, 2×, and 3× heights; customize this with `--height-multipliers`, for example `--height-multipliers 1,2,4`.

Execute the complete plan in Isaac Sim:

```bash
export ISAACSIM_PATH=/absolute/path/to/isaacsim-5.0.0
export ISAACSIM_PYTHON_EXE="$ISAACSIM_PATH/python.sh"
"$ISAACSIM_PYTHON_EXE" src/isaac_executor/scripts/isaac_executor.py \
  --yaml problems/full_3x3.yaml
```

After the scene settles, execution waits for one terminal key press while continuing to step physics and render the Isaac UI normally; pass `--no-wait` for unattended runs. Add `--headless` for non-visual execution. `Move` uses convergence-gated RMPflow stages and the Franka gripper from the object's live pose, verifies capture and lift, and maintains a dimension-aware stack height at shared goal pads. This is important after a knock because a lying cuboid's measured center is lower than its nominal upright center. `Knock` closes the gripper, applies a direction-dependent wrist orientation in RMPflow's `right_gripper` frame, approaches above the outward region of the selected cuboid's top face, pauses near the surface, and tracks a slow diagonal downward-and-outward stroke. The default stroke is 15° below horizontal and can be changed with `--knock-angle-deg`. This top-edge contact creates the intended flipping moment. Approach, stroke, grasp, and goal-stack heights scale with the cuboid dimensions.

For contact stability the executor uses a 120 Hz physics step, the TGS solver, continuous collision detection, increased Franka articulation solver iterations, the original project's Franka joint armature/friction overrides, softened arm drives, and increased drive damping. The contact stroke is issued as small bounded increments so RMPflow never accumulates a large unreachable target error against the block. Advanced tuning flags include `--knock-contact-steps`, `--knock-contact-frames`, `--knock-precontact-clearance`, `--drive-stiffness-scale`, and `--drive-damping-scale`.

## Batch execution and result analysis

`scripts/run_bench.py` starts a fresh headless Isaac process for each run and records its exit status, action counts, execution wall time, YAML, and log. For example:

```bash
export ISAACSIM_PYTHON_EXE=/absolute/path/to/isaacsim-5.0.0/python.sh
python3 scripts/run_bench.py --problems 'problems/full_*.yaml' --runs 10
```

Results use this layout:

```text
runs/<YYYYMMDD-HHMMSS>/
├── overall_summary.json
└── <problem>__runNN/
    ├── summary.json
    ├── <problem>.yaml
    └── isaac_executor.log
```

Analyze the batch summary with:

```bash
python3 scripts/summarize.py runs/<batch>/overall_summary.json
```

## Problem format

The YAML fields are `pads`, `objects`, `initial`, `goal`, and `actions`. Every `objects.<name>.dimensions` value is `[x, y, z]` in metres and is the single source of truth for both spawned collision geometry and height-aware manipulation. `Move` means pick-and-place. `Knock` names affected cuboids and a ROS-frame direction (`+x`, `-x`, `+y`, or `-y`). The executor applies the original scene's `Rz(+90°)` transform: ROS `+x/-x/+y/-y` become Isaac `+y/-y/-x/+x`, respectively. `z_base` and `z_increment` remain only for backward compatibility with historical problem files; files without `objects` are interpreted as cubes.

## Validation

```bash
python3 -m unittest discover -s tests -v
```

A colcon build additionally checks ROS manifests, launch installation, C++ compilation, and interface generation. Isaac Sim itself is required for physical execution validation.

## Citation

```bibtex
@misc{lu2026optimalKnockPick,
  title         = {Optimal Knock-Pick Planning for Tightly Packed Tabletop Blocks With Parallel Grippers},
  author        = {Lu, Hao and Shome, Rahul},
  year          = {2026},
  eprint        = {2605.17800},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  note          = {Accepted by WAFR 2026},
  doi           = {10.48550/arXiv.2605.17800}
}
```

## License and contact

Released under the [Apache License 2.0](LICENSE). Third-party-derived files may carry their own notices. Contact Hao Lu at `Hao.Lu@anu.edu.au`.
