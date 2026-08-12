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

The Isaac Sim benchmark then tests whether these combinatorially optimal plans can drive a physical execution pipeline. With a Franka Panda, ROS 2, MoveIt 2, and Isaac Sim 5.0.0, the paper reports a run success rate of 1.0 for the full 3 × 3 case and 0.8 for the full 5 × 5 case over 10 runs each, using the optimal knock counts of 2 and 8. See [Tables 1 and 2 in the paper](https://arxiv.org/pdf/2605.17800) for the complete full-grid and random-subgraph results, timing definitions, retry statistics, and hardware details.

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
│   ├── blocksworld_executor/   # ROS 2 / MoveIt Task Constructor executor
│   ├── blocksworld_scene/      # planning-scene interfaces
│   └── isaac_moveit/           # Isaac Sim scene and Panda MoveIt launch
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

### Isaac Sim / ROS 2

The paper workflow uses Ubuntu 22.04, ROS 2 Humble, MoveIt 2 Humble with MoveIt Task Constructor and Panda resources, and Isaac Sim 5.0.0 with its ROS 2 bridge. Install dependencies declared in `src/*/package.xml`, then build:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Quick start and synthetic benchmark

Run one deterministic case:

```bash
python3 scripts/kp_planner_opt.py --one full_3_3
```

Run the paper suite (six full grids and 20 seeded random, connected, hole-free subgraphs per size):

```bash
python3 scripts/kp_planner_opt.py --bench synthetic --seed 0
```

Use `--visualize` for the graph, gadget cover, and sequence; use `--debug` for diagnostics. The script prints full-grid results and subgraph averages. `t_cover` measures cleaned/face graph and matching work, `t_exec` measures cover-to-sequence conversion, and `replay_sanity` independently replays action feasibility.

## Isaac Sim benchmark

```text
Problem generation → optimal planner → knock/pick sequence → problems/*.yaml
        → Isaac Sim → ROS 2 / MoveIt 2 → execution
```

Generate the full 3 × 3 and 5 × 5 problems plus five random subgraphs of each size:

```bash
python3 scripts/kp_planner_opt.py --bench isaac --outdir problems --seed 0
```

Each YAML stores pads, initial/goal assignments, and ordered actions. A planned knock is `Knock` followed by the `Move` that picks and places the displaced block. The executor also reads the historical `Smash` token for backward compatibility.

Start Isaac Sim (terminal 1):

```bash
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ISAACSIM_PATH=/absolute/path/to/isaacsim-5.0.0
export ISAACSIM_PYTHON_EXE="$ISAACSIM_PATH/python.sh"
source /absolute/path/to/isaac_sim_moveit_ws/install/setup.bash
"$ISAACSIM_PYTHON_EXE" src/isaac_moveit/scripts/scene_isaacsim.py \
  --yaml problems/full_3x3.yaml
```

Add `--headless` when appropriate and follow Isaac Sim's ROS bridge setup if its library directory must be added to `LD_LIBRARY_PATH`.

Start MoveIt (terminal 2):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch isaac_moveit isaac_moveit.launch.py use_rviz:=false
```

Execute the sequence (terminal 3):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch blocksworld_executor knock_pick_demo.launch.py \
  problem:=problems/full_3x3.yaml
```

The executor obtains live block poses, plans each primitive with MoveIt Task Constructor/OMPL, executes it, checks knock clearance, and retries if required. The following section documents repeated batch execution and analysis of the generated case and overall summaries.

## Batch execution and result analysis

`scripts/run_bench.py` runs the full Isaac Sim → MoveIt → executor pipeline repeatedly and records both failures and successful timing metrics. Before starting it:

1. Generate the problem YAML files with `kp_planner_opt.py --bench isaac`.
2. Build and source the ROS workspace.
3. Set the installation-specific environment variables:
   - `ISAACSIM_PATH`: Isaac Sim 5.0.0 root containing `python.sh`.
   - `ISAAC_MOVEIT_WS_SETUP`: Isaac Sim ROS workspace `setup.bash`.
   - `MOVEIT_WS_SETUP`: MoveIt overlay workspace `setup.bash`.
4. Select `CASE_SPECS` near the top of `run_bench.py`. Each entry gives a result label and a glob relative to `problems/`.
5. Set `N_RUNS_PER_PROBLEM` (the paper uses 10).

Example configuration:

```python
CASE_SPECS = [
    {"name": "full_3x3", "glob": "full_3x3.yaml"},
    {"name": "full_5x5", "glob": "full_5x5.yaml"},
    {"name": "sub_3x3_k5", "glob": "sub_3x3_k5_*.yaml"},
    {"name": "sub_5x5_k16", "glob": "sub_5x5_k16_*.yaml"},
]
N_RUNS_PER_PROBLEM = 10
```

Run a batch from the repository root:

```bash
export ISAACSIM_PATH=/absolute/path/to/isaacsim-5.0.0
export ISAAC_MOVEIT_WS_SETUP=/absolute/path/to/isaac_sim_moveit_ws/install/setup.bash
export MOVEIT_WS_SETUP=/absolute/path/to/moveit_overlay_ws/install/setup.bash
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/run_bench.py
```

For every repetition, the runner starts a fresh headless Isaac scene, waits for the planning scene publication, starts MoveIt and waits for all three Panda controllers, then launches `knock_pick_demo.launch.py`. An attempt is successful only when the executor log contains both the all-tasks-success line and its planning/execution timing line. Failed attempts are retried up to `EXEC_MAX_ATTEMPTS`; scene, MoveIt, and executor timeouts are configured near the top of the script.

Results use this layout:

```text
runs/<YYYYMMDD-HHMMSS>/
├── overall_summary.json
└── <case>/
    ├── case_summary.json
    └── <problem>__runNN/
        ├── summary.json
        ├── <problem>.yaml
        ├── isaac_scene_attemptN.log
        ├── isaac_moveit_attemptN.log
        └── executor_attemptN.log
```

`summary.json` records success/failure, failure reason, action and knock counts, executor attempts, knock retries, TAMP time, physics execution time, and measured executor wall time. `case_summary.json` contains all run records and successful-run averages. `overall_summary.json` contains every case in the batch.

Analyze one case:

```bash
python3 scripts/summarize.py runs/<batch>/<case>/case_summary.json
```

Analyze the entire batch:

```bash
python3 scripts/summarize.py runs/<batch>/overall_summary.json
```

The analyzer prints one row per case followed by aggregate sums, counts, and means. Metric averages include only runs marked `success: true`; the displayed success rate still uses all attempted runs. This prevents failed or incomplete executor logs from biasing timing and retry averages.


## Problem format

The YAML fields are `pads`, `z_base`, `z_increment`, `initial`, `goal`, and `actions`. `Move` means pick-and-place. `Knock` includes affected blocks, its start pad/height, and a direction (`+x`, `-x`, `+y`, or `-y`). Isaac Sim consumes geometry and initial state; the ROS executor consumes the full sequence.

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
