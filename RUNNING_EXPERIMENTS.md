# Running the Primary Experiments

This guide contains the commands for the balanced 40-case experiment: 20
axis-aligned and 20 independently yawed cases. Run all commands from the
repository root.

## Environment

```bash
cd /home/luhao/Workspace/manipulation/planning_with_knock

export ISAACSIM_PATH="/home/luhao/Tools/isaac/isaacsim-5.0.0/_build/linux-x86_64/release"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

Four independent Isaac processes are the tested maximum on the current RTX
5090 workstation.

## Generate and solve the 40 problems

This runs geometric preprocessing, MILP, SAT, validation, visualization, and
Isaac YAML export, but does not launch Isaac execution:

```bash
"$ISAACSIM_PYTHON_EXE" cuboid_knock_pick_solvers.py \
  --output-dir results/experiment_40_planning \
  --primary-experiment-only --isaac-exportable-only
```

## Proposed optimal Knock--Pick method

Run every problem ten times in Isaac:

```bash
python3 scripts/run_bench.py \
  --problems 'results/experiment_40_planning/isaac_problems/*.yaml' \
  --runs 10 --policy planned --workers 4 --timeout 900 \
  --seed 20260910 --isaac-python "$ISAACSIM_PYTHON_EXE" \
  --output runs/proposed-primary40
```

This performs 400 physical executions.

## Random feasible-Knock baseline

### Planner distribution

Run 1,000 abstract random plans per case to estimate Knock count, regret, and
planning-time distributions:

```bash
"$ISAACSIM_PYTHON_EXE" scripts/baseline_compare.py \
  --problems 'results/experiment_40_planning/isaac_problems/*.yaml' \
  --trials 1000 --seed 20260910 \
  --output runs/random-knock-planning-primary40
```

This performs 40,000 planner-only trials and launches no simulator.

### Planning and physical execution

Sample ten independent random feasible-Knock plans per case and execute each
sampled plan in Isaac:

```bash
"$ISAACSIM_PYTHON_EXE" scripts/random_knock_benchmark.py \
  --problems 'results/experiment_40_planning/isaac_problems/*.yaml' \
  --runs 10 --workers 4 --timeout 900 --seed 20260910 \
  --isaac-python "$ISAACSIM_PYTHON_EXE" \
  --output runs/random-knock-isaac-primary40
```

This creates 400 independently sampled plans and 400 physical executions. The
sampled YAML, plan metrics, Isaac logs, per-run metrics, aggregate CSV/Markdown,
and `overall_summary.json` are retained together.

## Online random-Push baseline

```bash
python3 scripts/run_bench.py \
  --problems 'results/experiment_40_planning/isaac_problems/*.yaml' \
  --runs 10 --policy random-push --workers 4 --timeout 900 \
  --seed 20260910 --isaac-python "$ISAACSIM_PYTHON_EXE" \
  --output runs/random-push-primary40
```

This performs 400 physical executions. Pick feasibility and random Push choices
are recomputed from the live Isaac state.

## Complete flow: proposed method and both baselines

The following command regenerates the primary problems, executes all three
physical policies, runs the 1,000-trial random-Knock planning study, and writes
the combined result tables:

```bash
python3 scripts/all_methods_benchmark.py \
  --runs 10 --random-knock-trials 1000 \
  --workers 4 --timeout 900 --seed 20260910 \
  --isaac-python "$ISAACSIM_PYTHON_EXE" \
  --planner-output results/experiment_40_planning \
  --problems 'results/experiment_40_planning/isaac_problems/*.yaml' \
  --output runs/all-methods-primary40
```

This complete flow performs:

- 400 proposed-method Isaac runs;
- 400 random-Knock Isaac runs using independently sampled plans;
- 400 online random-Push Isaac runs; and
- 40,000 additional planner-only random-Knock trials.

To reuse the already generated problem set, add `--no-regenerate-plans`.

Results are stored under `runs/all-methods-primary40/<timestamp>/`. The
`proposed`, `random_knock_isaac`, `random_push`, and `random_knock` directories
preserve raw outputs. The root contains combined full and paper-facing tables.

## Short smoke test

Before launching the full experiment, test one problem and one repetition:

```bash
"$ISAACSIM_PYTHON_EXE" scripts/random_knock_benchmark.py \
  --problems 'results/experiment_40_planning/isaac_problems/gap_grasp_5.yaml' \
  --runs 1 --workers 1 --timeout 900 --seed 20260910 \
  --isaac-python "$ISAACSIM_PYTHON_EXE" \
  --output runs/random-knock-smoke
```
