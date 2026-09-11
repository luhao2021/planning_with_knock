# Experimental Evaluation Protocol

## 1. Experimental objectives

The experiments evaluate three questions:

1. **Planning optimality and efficiency:** Can the proposed Minimum Knock-Key formulation compute plans with the minimum possible number of clearance-creating Knock actions?
2. **Physical executability:** Can plans produced by the high-level solver be executed by a Franka Panda in Isaac Sim using physical grasping and contact-based clearance creation?
3. **Baseline comparison:** How does the proposed method compare with random feasible-Knock planning and physical execution, and with an online random-Push baseline operating on the evolving physical scene?

The evaluation separates high-level combinatorial planning from low-level physical execution. The solver chooses Pick and Knock actions and their geometric witnesses. Isaac Sim executes the plan using robot motion, gripper control, collision, friction, and contact dynamics.

## 2. Benchmark problems

The restructured primary benchmark contains 40 heterogeneous cuboid arrangements: 20 axis-aligned cases and 20 non-axis-aligned cases. Legacy cases remain available as an `others` backup category but are excluded from the primary reported averages.

| Family | Cases | Main purpose |
|---|---:|---|
| Axis-aligned | 20 | Double rail, gap grasp, long spine, partial-contact/window, and compact random-guillotine layouts |
| Non-axis-aligned | 20 | Dense independently yawed locked clutter with randomized count, dimensions, position, and yaw |

The former two-clump, common-yaw, and multi-clump scenes are retained in the source catalog as `others`, but they are not included in the primary 40-case evaluation.

The axis-aligned set retains seven requested hand-designed cases, three
existing compact guillotine cases, and ten additional seeded compact
guillotine partitions. The non-axis-aligned cases use the reproducible naming
scheme `yaw_independent_locked_dense_<index>_n<count>_s<seed>`.

Only cases satisfying the Franka Panda's physical constraints are included in Isaac evaluation. The jaw opening must not exceed the usable 80 mm aperture; finger-contact regions must accommodate the physical finger pad; scaled dimensions must remain physically meaningful; each layout must fit in the tabletop workspace; and planned grasp and clearance witnesses must remain valid after scaling.

## 3. Geometric preprocessing

Each cuboid is represented by a planar oriented rectangle with a center, width, length, and yaw. Preprocessing constructs finite Pick witnesses, their blocker sets, directional clearance rules, and the blocker set associated with every clearance corridor. Continuous finger placement is reduced to a finite exact witness set using the paper's critical-event discretization. Inclusion-dominated rules are removed before optimization.

For yawed objects, Pick geometry is evaluated in the target object's local coordinate frame. A common-yaw clump can be transformed into one shared axis-aligned frame. For independently oriented objects, neighboring footprints are projected into each target's local frame. Planned cuboid-local clearance directions are transformed into the common workspace frame for execution.

## 4. Proposed optimal planner

The proposed method solves the monotone rule system as a Minimum Knock-Key problem. Pick rules have zero cost in the contracted optimization, whereas Knock rules have unit cost. Every object subsequently incurs one mandatory physical Pick/Place action. For (n) objects and an optimum of (k) Knocks, the nominal physical action count is

\[
n+k.
\]

### Exact solvers

The primary optimizer is a MILP implemented with `scipy.optimize.milp` and the HiGHS backend. It selects one action per object and uses topological-rank variables to prohibit cyclic dependencies.

The SAT formulation independently selects one action per object, imposes a Knock budget, and adds lazy dependency-cycle clauses. The current environment uses the bundled `SimpleDPLL` implementation; PySAT/Glucose4 can be used when installed. Small instances may also be checked using Pick-closed-state breadth-first search as an independent optimality oracle.

Both exact solvers have a default 60-second limit per problem. The high-level sequence exported to Isaac is generated from the MILP solution. SAT is used for validation rather than being included in the default execution-planning pipeline.

The paper-facing planning time is

\[
T_{\mathrm{plan}}=T_{\mathrm{preprocess}}+T_{\mathrm{MILP}}.
\]

This includes geometric rule construction, dominance reduction, and exact MILP optimization. SAT time remains in the full results but is excluded from the proposed method's planning time.

## 5. Isaac Sim scene construction

Problems are uniformly scaled so that the complete arrangement fits within an approximately (25\,\mathrm{cm}\times25\,\mathrm{cm}) workspace. Exports use equal-height physical cuboids, a nominal 5 mm gap between geometrically contacting cuboids, and consistently scaled values of (F), (D), grasp witnesses, and clearance thresholds. The scene uses the repository's ROS-to-Isaac transformation and an additional 5 cm shift along ROS (+x), corresponding to Isaac (+y), to move the workspace farther from the robot.

All cuboids have colliders, density-based mass and inertia, friction, damping, and rigid-body dynamics enabled.

## 6. Robot and physics configuration

Execution uses a Franka Panda in Isaac Sim 5.0 with:

- a 120 Hz physics rate;
- the TGS solver;
- continuous collision detection;
- increased articulation solver iterations;
- Franka joint armature and friction tuning;
- moderate cuboid/table sliding friction;
- higher-friction material on physical finger links;
- bounded depenetration velocity; and
- cuboid linear and angular damping.

Free-space drive gains use a stiffness multiplier of 0.85 and damping multiplier of 1.50. During clearance contact, stiffness is scaled by 0.80 and damping by 1.50, after which free-space gains are restored.

No kinematic attachment, pose-following constraint, teleportation, object restoration, or artificial grasp constraint is used. Objects move through simulated contact physics.

## 7. Pick execution

Before every Pick, the executor reads the cuboid's current physical pose. It does not assume that the object remains at its initial pose or stays axis aligned. The planned witness specifies a gripper closing axis and tangential finger-contact interval. This witness is transformed using the latest measured cuboid orientation.

A Pick consists of:

1. fully opening the gripper;
2. moving vertically to a safe height;
3. translating above the live grasp witness;
4. aligning with the cuboid's measured supporting-face orientation;
5. descending while aligning the finger-pad midpoint with the cuboid's height center;
6. closing the gripper;
7. checking opposing physical contact;
8. lifting and verifying that the cuboid rises with the fingers;
9. transporting the object outside the unresolved clump; and
10. opening the gripper and retiring the disposed object.

The tangential position respects the planner's witness and is not replaced by a center grasp. A small physical inward offset positions the complete finger-contact region within the planned feasible interval.

Disposal directions are selected online from ROS (+x,+y,-y). The candidate having the greatest separation from unresolved objects is preferred. Since exact disposal coordinates are outside the theoretical objective, the object is retired after release without requiring exact waypoint convergence.

A failed Pick is retried from the latest live pose up to five times. Failure records distinguish reachability or alignment failure, insufficient clearance, missing opposing contact, grasp slip, failure to lift, Cartesian convergence, joint-limit proximity, and other physical errors. After all retries are exhausted, the Pick is recorded as skipped and execution continues without restoring the scene.

## 8. Knock/clearance execution

Although called a Knock in the theoretical model, the physical primitive is implemented as a compliant clearance-creating slide:

1. close the gripper;
2. retreat vertically above the live scene;
3. translate horizontally at a safe height;
4. align the palm parallel to the planned clearance direction;
5. descend vertically above the selected edge;
6. detect initial finger contact;
7. apply a mainly outward and slightly downward sliding stroke;
8. measure directed cuboid displacement;
9. unload contact through a slow vertical retreat; and
10. re-establish a consistent gripper state.

Success is determined from displacement along the planned outward direction. Arbitrary planar motion does not count. If one stroke does not provide sufficient grasp clearance, another attempt begins from the cuboid's latest pose. Up to 24 physical attempts are currently permitted. The target and neighboring cuboids are never restored after unintended motion.

## 9. Compared methods

### 9.1 Proposed optimal Knock--Pick

The proposed method performs exact geometric preprocessing, computes a minimum-Knock sequence using MILP, executes its planned Pick witnesses and clearance directions, and adapts execution to live cuboid poses.

### 9.2 Random feasible-Knock baseline

This baseline repeatedly exhausts feasible Picks. At a deadlock, it uniformly selects an object having at least one physically admissible directional Knock and then uniformly selects one of its admissible directions. It uses the same geometric rule system and Panda post-clearance grasp constraints as the proposed planner, thereby isolating high-level action selection. Each case is evaluated with 1,000 seeded planner trials. In addition, ten independently sampled plans per case are exported and physically executed in Isaac.

The principal comparison is Knock regret:

\[
K_{\mathrm{random}}-K_{\mathrm{optimal}}.
\]

Because every method ultimately Picks all (n) objects, every excess Knock adds one physical action.

### 9.3 Online random-Push baseline

This baseline is evaluated in Isaac Sim and does not follow the proposed clearance sequence. It observes unresolved cuboids, reconstructs their oriented footprints, exhausts feasible Picks, and detects deadlocks. At a deadlock, it enumerates side faces with clear approach and sweep corridors, randomly selects a feasible object-face pair, physically pushes it by approximately 5--10 cm, re-observes the scene, and recomputes Pick feasibility. The procedure repeats until all objects are removed or an execution limit is reached.

## 10. Repetitions and reproducibility

The planned final evaluation uses:

- 40 primary problem instances, balanced as 20 axis-aligned and 20 non-axis-aligned;
- 10 Isaac repetitions per problem for the proposed method;
- 10 independently planned and executed random-Knock repetitions per problem;
- 10 Isaac repetitions per problem for random Push; and
- 1,000 planner-only random-Knock trials per problem.

This produces 1,200 independent Isaac executions and 40,000 additional planner-only random-Knock trials. Deterministic derived seeds are used throughout. Each ordinary Isaac repetition starts in a fresh simulator process so physics state is not shared between trials.

Up to four Isaac processes run concurrently. Four workers are the tested maximum on the RTX 5090 with 32 GB VRAM; eight workers previously exceeded the timeout. The default per-execution timeout is 900 seconds.

## 11. Metrics

### Planning metrics

The full results retain object and rule counts, maximum rule ranks, preprocessing time, MILP and SAT statuses, optima and runtimes, SAT calls, cycle cuts, DPLL nodes, BFS verification, and nominal physical action count.

The compact paper results retain total planning time, optimal Knocks, physical actions, random-Knock mean and standard deviation, Knock regret, and planning completion rate.

### Physical-execution metrics

Each Isaac run records:

- execution status and full or partial sequence completion;
- simulator execution time and process wall time;
- planned, attempted, completed, failed, and skipped actions;
- action-completion percentage;
- objects removed, objects remaining, and object-completion percentage;
- Pick, clearance, and Push attempts and successes;
- clearance retries and directed displacement;
- online replanning count and time;
- failure categories and stages;
- joint positions and joint-limit margins at failure; and
- final poses of all cuboids.

Repeated-run scalar metrics are reported using their mean and standard deviation. Simulator execution time excludes application startup overhead, while wall time is retained in the full results.

## 12. Planned paper tables

Six logical tables are generated.

### Full supplementary tables

1. **Proposed method: complete planning and Isaac metrics.**
2. **Optimal planner versus random feasible Knock.**
3. **Proposed method versus online random Push.**

These preserve all collected scalar metrics for supplementary material and artifact evaluation.

### Compact main-paper tables

4. **Proposed method performance:** problem family, object count, yaw model, planning time, action counts, physical Knock attempts, execution time, success, and object completion.
5. **Optimal planner versus random feasible Knock:** two rows per case containing planning time, actions, Picks, Knocks, random Knocks, regret, and planning success.
6. **Proposed method versus online random Push:** two rows per case containing planning or replanning time, action attempts, Picks, Knocks, Pushes, execution time, success, object completion, and failed actions.

Fixed parameters such as (F), (D), friction, stiffness, and controller gains should appear once in the experimental-setup text rather than in every table row.

## 13. Preliminary results

The latest compact batch, `runs/all-methods/20260910-170059`, contains five Isaac repetitions per problem and should be described as pilot data. Across its 20 cases:

- the proposed method achieved 100% success and 100% object completion on every case;
- proposed plans used an average of 1.30 Knocks per problem;
- random feasible Knock used an average of 1.7555 Knocks;
- random Knock had positive average regret on 9 of 20 cases;
- average case-level Knock regret was 0.4555 and the maximum was 1.88;
- proposed execution averaged approximately 47.35 seconds per case;
- random-Push execution averaged approximately 143.50 seconds per case;
- random Push had a mean case-level success rate of 78%; and
- random Push achieved approximately 95.29% mean object completion, showing substantial partial progress in some failed trials.

These preliminary results suggest that exact action selection reduces paid clearance actions and that planned Knock--Pick execution is more reliable and faster than repeated online random pushing. The final paper should replace these pilot values with the planned ten-repetition results before making definitive claims.

## 14. Threats to validity

- Isaac Sim is physically meaningful but does not replace real-robot validation.
- A geometrically feasible Knock additionally depends on friction, compliance, controller behavior, and contact location during execution.
- Twenty physically exportable cases provide representative validation rather than large-scale statistical coverage.
- Random Knock shares the proposed geometric preprocessing and isolates only high-level action selection.
- Random Push includes online, physics-dependent replanning, so its runtime is not directly equivalent to offline planning time.
- Concurrent simulator processes may introduce GPU-contention variability; execution time should therefore include standard deviation.
- Failed actions are skipped rather than immediately terminating execution, making object completion an important complement to binary success.
- Objects are retired following disposal release, abstracting away precise target-placement accuracy.
- The current theoretical model assumes equal-height cuboids and irreversible removal from the packed scene.

## 15. Reproduction command

The complete final experiment can be launched with:

```bash
export ISAACSIM_PATH="/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

python3 scripts/all_methods_benchmark.py \
  --runs 10 --random-knock-trials 1000 \
  --workers 4 --timeout 900
```

The generated full and compact results are saved under `runs/all-methods/<timestamp>/`.
