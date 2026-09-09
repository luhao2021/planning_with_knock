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

Tightly packed tabletop blocks expose a fundamental limitation of parallel grippers: without lateral clearance, a direct top-down pick may be impossible. Even a fully occupied 2 × 2 group can lock every block against picking. This work treats a controlled **clearance action** (called a *knock* in the original paper) as a deliberate non-prehensile action that creates finger clearance, enabling ordinary pick-and-place manipulation to resume.

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
├── cuboid_knock_pick_solvers.py # heterogeneous-footprint MILP/SAT planner
├── src/
│   ├── isaac_executor/         # native Isaac Sim pick/knock executor
│   ├── blocksworld_executor/   # historical MTC executor (COLCON_IGNORE)
│   ├── blocksworld_scene/      # historical planning-scene service (ignored)
│   └── isaac_moveit/           # historical ROS bridge (ignored)
├── tests/                      # planner/serialization regressions
├── requirements.txt
└── LICENSE
```

`scripts/kp_planner_opt.py` is the supported entry point for the original grid
model and its variable-height extension. `cuboid_knock_pick_solvers.py` handles
equal-height, axis-aligned cuboids with heterogeneous rectangular footprints,
partial side contacts, and full directional escape corridors. Both are
high-level planners; Isaac Sim is only the low-level executor. Other planner,
layer-peeling, and feedback-vertex-set scripts are research-development variants
retained for traceability.

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

## Heterogeneous rectangular-cuboid solver

The cuboid solver preprocesses continuous parallel-jaw grasps of oriented
rectangles into finite Pick
rules. A rule records the closing axis, an exact finger-placement witness, and
the objects blocking that placement. Each directional Clearance rule records
every object in the target's full outward footprint corridor. It then selects
one rule per object while enforcing an acyclic blocker-precedence graph.

- The MILP uses SciPy/HiGHS and continuous topological ranks.
- The SAT solver uses PySAT when installed, otherwise its bundled validation
  DPLL implementation.
- A Pick-closure BFS independently checks small instances.

Run the complete 34-instance suite:

```bash
python3 cuboid_knock_pick_solvers.py --output-dir results
```

The suite includes exact gap grasps, multiple proper partial-contact layouts,
nonuniform packed tilings, long-spine and double-rail cases, four separated
two-clump cases, deterministic random guillotine partitions, and twelve yaw cases:
three rigidly rotated common-yaw clumps, three multi-clump scenes with a distinct
yaw per clump, and six independently oriented scenes. Two of the independent-yaw
scenes contain finite-clearance dependency cycles and require at least one
Knock; the other four are zero-Knock closure cases. The last family uses the
paper's approved finite-clearance model (`eta_F > 0`): polygon intersections in
each target's local side strip are projected into exact scalar blocking
intervals. Knock direction labels are object-local: preprocessing rotates the
selected local axis by the planned yaw, and Isaac freezes the corresponding live
world direction at the start of the Clearance action. Knock rules use the full
oriented-footprint swept polygon. The MILP and SAT layers are unchanged. Output includes
CSV and Markdown summaries, readable plans, layouts, solution overlays,
step-by-step figures, and executable YAML files under
`results/isaac_problems/`. Use `--no-plots` or
`--no-isaac-export` to disable those outputs.

Isaac export performs a separate physical-feasibility audit. A uniform scale
must keep the scene within 25 cm, every closing span within the Panda's 75 mm
safe jaw limit, the complete 20 mm tangential finger footprint inside the
planner witness, and every cuboid side at least 20 mm after the 5 mm contact-gap
shrink. A post-Clearance exposed-end grasp must also fit the jaw. Instances that
cannot satisfy all constraints remain in the exact solver tables with
`isaac_export=unsuitable:...`; no YAML is emitted (and any stale YAML is removed).

The contracted objective minimizes Clearances. Every selected Clearance is
followed by its mandatory Pick/Place, so `n` objects and optimum `k` require
`n + k` physical high-level actions.

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

Each YAML stores pads, per-object dimensions, initial/goal assignments, and
ordered actions. The preferred non-prehensile token is `Clearance`; historical
`Knock` and `Smash` remain accepted. A Clearance is followed by the `Move` that
picks and places the displaced object. Grid-planner cuboid heights are integer
multiples of their side length. The heterogeneous-footprint solver instead uses
equal-height rectangular cuboids.

Execute the complete plan in Isaac Sim:

```bash
export ISAACSIM_PATH=/absolute/path/to/isaacsim-5.0.0
export ISAACSIM_PYTHON_EXE="$ISAACSIM_PATH/python.sh"
"$ISAACSIM_PYTHON_EXE" src/isaac_executor/scripts/isaac_executor.py \
  --yaml problems/full_3x3.yaml
```

After the scene settles, execution waits for one terminal key press while
continuing to step physics and render normally; pass `--no-wait` for unattended
runs. `Move` reads its source pose from live Isaac state. A direct heterogeneous
Pick also applies the solver's closing axis and offset finger window. A Move
after Clearance uses the displaced live pose and the end-window grasp exposed
by the planner-selected clearance direction; it never substitutes a centered
grasp. The exporter stores this derived witness explicitly in the Move action.
The post-clearance physical finger window uses an explicit 40 mm side-contact
envelope. Following the planner's start-coordinate convention, offset `S`
means the interval `[S,S+W]`; Isaac commands its center at `S+W/2`. The exporter
computes this interval against the physically shrunken cuboid and keeps its
whole width at least 5 mm inside the end edge. Clearance displacement is also
required to reach at least `W+5 mm`, so the fuller contact does not consume an
uncreated corridor.

Clearance closes the gripper, reaches the target from a collision-safe overhead
pose, and descends vertically only until the physical finger band contacts the
live cuboid top. Contact immediately starts a shallow, outward-dominant top-edge
stroke; its small downward component is delayed until lateral motion is under
way and capped at 4 mm to favor sliding over tipping. It repeats from the object's updated
live pose until cumulative outward displacement reaches the largest of
`F + 5 mm`, the plan's requested displacement, and the corridor required by
the fully open 80 mm Panda gripper. The last term includes half the difference
between aperture and live projected cuboid width, physical finger allowance,
and the 5 mm robustness margin. The default cap is 12 attempts
and never less than 10; override it with `--max-clearance-attempts`. If the cap
is exhausted, metrics are written and an interactive scene remains open for
inspection. The default stroke is 8° below horizontal and can be changed with
`--knock-angle-deg` (the historical flag name is retained).

Planner contacts remain exact. Isaac exports uniformly fit the complete layout
inside a 25 cm square, apply the repository's original `Rz(+90°)` convention,
shift both initial and goal pads by 5 cm along ROS +X (Isaac +Y, away from the
robot), and shrink each physical footprint symmetrically to create a 5 mm nominal gap.
`F`, `D`, grasp witnesses, and clearance thresholds are scaled consistently.

For contact stability the executor uses a 120 Hz physics step, the TGS solver,
continuous collision detection, increased Franka articulation solver iterations,
the original project's Franka joint armature/friction overrides, moderate
cuboid/table sliding friction, and a moderately high-friction material on the
physical finger links. Free-space drives use `0.85×` nominal stiffness and
`1.50×` damping to avoid excessively overdamped ordinary motion. During loaded
clearance contact stiffness is further scaled by `0.80×` and damping by
`1.50×`, then both are restored exactly afterward. Cuboid damping and a bounded depenetration
velocity absorb impact spikes without attaching or constraining the objects.
Picking uses no pose-follow attachment: it measures the actual finger
link aperture and pad pose, recomputes wrist yaw from the cuboid's latest pose,
including selecting the two horizontal body axes of its current supporting face
after a 90-degree roll,
keeps the fingers fully open during approach/descent, descends to a
height-dependent fingertip overlap while aligning the measured finger-pad band
to the live cuboid height center in one continuous descent, and accepts a grasp only when opposing
finger contact physically lifts the cuboid. Excess descent/closure displacement
or a failed physical lift is reported as an execution failure.
Pick approach is decomposed into vertical clearance, overhead translation, and
vertical descent. Loaded lift and transport, and the post-release retreat, use
bounded smooth Cartesian trajectories. The executor never swings a grasped
object to the home joint posture as a recovery maneuver.

Clearance creation is a measured,
continuous-contact slide: the arm follows small bounded trajectory increments,
and success is determined only from the selected cuboid's measured PhysX
displacement in the requested outward direction. Motion in an arbitrary planar
direction no longer counts as success; insufficient directed clearance causes
another attempt from the cuboid's latest live pose. The executor first lifts vertically above the live scene, moves
horizontally at that safe height, aligns the palm, and then descends vertically
to contact. Neither the target nor its neighbors are projected, attached,
frozen, or restored. Unintended disturbances persist into later actions, whose
grasps are recomputed from current poses. A failed Move is retried from live
state up to `--max-move-attempts` (default 5). An exhausted Move is recorded and
skipped, and execution continues with the remaining high-level actions from the
unchanged live scene. A failed Clearance is handled the same way after its
configured physical retries. Metrics distinguish full and partial sequence
completion, count failed picks and clearances, and include action/object
completion percentages and all final cuboid poses. After a clearance stroke,
Each failed Move attempt also records a failure category and the current arm
joint positions, minimum joint-limit margin, and any near-limit joints. A
failed physical lift is therefore reported as grasp slip rather than being
mislabelled as a joint-limit failure.
Planar motion during lift is not itself a failure: if the cuboid rises at least
25 mm with the fingers, transport continues and the measured slip is logged. If
it stays on its support surface, the executor re-observes its latest Isaac pose
and retries. Each post-clearance retry shifts the same exposed-end witness 10 mm
inward, capped at 60% of its original offset, so it gains contact without ever
silently becoming a centered grasp.
After a clearance stroke,
the arm retains compliant contact gains and slowly unloads through a purely
vertical retreat before restoring free-space gains; the hand is then explicitly
closed and reopened at safe height so contact cannot leave its state ambiguous.
The Panda's native finger drives remain authoritative (no passive joint-friction
override); high-friction contact material is applied only to the physical finger
links, avoiding a setting that can prevent commanded closure.
Both the pre-contact descent and its preceding vertical approach use bounded,
incremental Cartesian trajectories; they do not wait on the generic long
Cartesian timeout or permit a low-height lateral correction through clutter.
For the two equivalent signs of a palm-parallel axis, the executor selects the
direction-specific wrist branch reachable at the shifted far-side workspace;
this does not change the planned clearance direction.
Disposal placement is best-effort: its descent is commanded
for a bounded interval without a Cartesian convergence check, and the released
object is retired regardless of the achieved disposal waypoint. This is the
sole deliberate scene retirement. Its XY waypoint is selected at execution
time from ROS `+x`, `+y`, and `-y`: the executor evaluates the live unresolved
objects and chooses the reachable candidate with the largest separation. A
disposed object is retired as soon as its fingers separate, before it can fall
or bounce back into the unresolved clump. Advanced tuning flags include `--knock-contact-steps`,
`--knock-contact-frames`, `--knock-precontact-clearance`,
`--drive-stiffness-scale`, and `--drive-damping-scale`.

## Batch execution and result analysis

`scripts/run_bench.py` starts a fresh headless Isaac process for each run and records its exit status, action counts, execution wall time, YAML, and log. For example:

```bash
export ISAACSIM_PYTHON_EXE=/absolute/path/to/isaacsim-5.0.0/python.sh
python3 scripts/run_bench.py \
  --problems 'results/isaac_problems/*.yaml' --runs 10 --workers 4
```

Start with one worker on a new machine to validate physical behavior.
`--workers N` runs `N`
isolated Isaac processes concurrently, so robots cannot collide or share physics
state. Parallel GPU processes can consume substantial VRAM and reduce
determinism. On the tested RTX 5090 (32 GB), four concurrent workers completed
the representative clearance workload successfully; eight workers all exceeded
the 300-second timeout. Use `--workers 4` as the tested maximum on that machine.

Results use this layout:

```text
runs/<YYYYMMDD-HHMMSS>/
├── overall_summary.json
├── aggregate.csv
├── aggregate.md
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

The YAML fields are `pads`, `objects`, `initial`, `goal`, and `actions`.
`objects.<name>.dimensions` contains Isaac-frame `[x, y, z]` collision geometry
in metres. A `Move.grasp` specifies the solver witness `g=(o,t)`: `closing_axis`
stores `o`, while planar `offset` stores the finger-window center `t + F/2`
relative to the cuboid footprint center. Thus only grasp Z defaults to the live
cuboid height center; grasp X/Y need not be at the footprint center. If the
dynamic body yaws, the executor rotates this planar witness with the body.
`initial.<object>.yaw_rad` stores the planned in-plane yaw and initializes the
dynamic Isaac body with that orientation; subsequent actions always use its live
pose rather than restoring the planned pose.
A post-Clearance Move likewise contains a `grasp` payload derived from its
planner-selected directional clearance witness. For a `+y` clearance, for
example, the window is placed at the exposed +Y end and closes across X.
`Clearance` names affected cuboids and an object-local ROS-plan direction (`+x`,
`-x`, `+y`, or `-y`). The direction is first rotated by the cuboid yaw and then
mapped through the original scene's `Rz(+90°)` ROS-to-Isaac transform. Initial source poses are transformed
once; all poses after physics starts are read directly from Isaac. `z_base` and
`z_increment` remain only for historical files.

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
