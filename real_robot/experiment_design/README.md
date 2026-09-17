# Thirteen physical experiment scene templates

Use [overview.png](overview.png) to arrange the table approximately, or print
[reference_layouts.pdf](reference_layouts.pdf) for one scene per page with centre
coordinates and yaw. [reference_positions.csv](reference_positions.csv) and
[reference_scenes.json](reference_scenes.json) contain the same reference geometry.
These coordinates are **not fed to perception or the solver**. Every actual run
uses its initial RGB-D detection to generate its own fixed plan.

All dimensions are nominal: Protex **90 × 60 × 28 mm**, BEACH **110 × 65 × 25 mm**.
Boxes lie flat. Drawings use link_base coordinates, +X right and +Y up, rather than
camera-image directions. Yaw is counterclockwise from +X along the long side.
The dashed source area is X=300–570 mm, Y=−250–250 mm. All example footprints fit
inside it without overlap. Close-grid nominal gaps are about 6–12 mm; approximate
placement is sufficient. Reference geometry has not been physically validated.

| Template | Inventory | Example arrangement | Seed setup |
|---|---|---|---|
| S01 | 2 Protex | Different individual orientations | Automatic |
| S02 | 2 BEACH | Different individual orientations | Click each box |
| S03 | 1 Protex + 2 BEACH | Close, axis-aligned group | Click each box |
| S04 | 1 BEACH + 2 Protex | Separated, different orientations | Click each box |
| S05 | 3 Protex | Close, axis-aligned group | Automatic |
| S06 | 3 BEACH | Different individual orientations | Click each box |
| S07 | 2 BEACH + 2 Protex | Close grid | Click each box |
| S08 | 1 BEACH + 3 Protex | Staggered grid | Click each box |
| S09 | 3 BEACH + 1 Protex | Close grid | Click each box |
| S10 | 4 BEACH | Close grid | Click each box |
| S11 | 4 Protex | Close grid | Automatic |
| S12 | 3 Protex + 2 BEACH | Five-box, staggered group | Click each box |
| S13 | 4 Protex + 2 BEACH | Staggered group plus rotated box | Click each box |

“Bench” in the requested list is the BEACH box (`beach` in configuration).
S03 contains one Protex and two BEACH; S04 contains one BEACH and two Protex.
None of these templates contains a one-box scene.

**Each template supports different positions and orientations for every box.**
The config fixes only the inventory and perception/execution settings; it contains
no target reference X/Y/yaw. You can independently translate and rotate each box
on the table, use a tight or loose grouping, and reuse the same inventory template.
Fresh detection supplies all actual poses. Keep boxes flat and visible inside the
source area; this does not add support for stacking or standing boxes on an edge.
A valid detection and feasible initial plan are still needed for the actual scene.

For example, use S07 for both an aligned 2+2 group and a diagonally arranged 2+2
group with four different yaws. Record them as `layout_variant=A` and `B` in the
results sheet. Rerun manual seed setup after rearranging either group. For the
Protex-only S01/S05/S11 templates, new seeds are automatic on each run.

## Current focus: S07, S12 and S13

For the complete preparation → preview → execution commands, see the
[real-robot quick start](../README.md#prepare-configuration-and-execute-s07-s12-or-s13).

Use [focus_S07_S12_S13.png](focus_S07_S12_S13.png) for the side-by-side references.
S07 has **2 Protex + 2 BEACH**, S12 has **3 Protex + 2 BEACH**, and S13 has
**4 Protex + 2 BEACH**. Each box can have
its own detected position and yaw; the drawings do not constrain your real scene.
All three templates use named manual clicks and the current class-specific model/grip/
height settings. BEACH remains 110 × 65 × 25 mm; Protex 90 × 60 × 28 mm.

For S07:

```bash
python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S07.json
```

For S12:

```bash
python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S12.json
```

For S13:

```bash
python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S13.json
```

Each command resets the camera once and opens the seed-selection image. Click
inside each requested physical box and press Enter (four clicks for S07, five for
S12, six for S13). Then use the printed preview or execute command with the newly prepared
configuration. Reset the full scene and prepare fresh seeds for the next trial.

The manual setup helper now accepts up to six boxes. Automatic Protex startup
retains its four-box limit; these mixed templates use manual startup. The per-object
limits remain 25 Knocks and one Pick. S12's overall action budget is 130 and
S13's is 156, accommodating those per-object limits for five and six boxes
respectively. S07 retains its 120-action budget. The shared solver and Isaac pipeline are unchanged.

## What needs to change when the scene changes?

The existing pose/mesh, planning, and execution code understands both classes.
**Automatic startup currently supports Protex only.** It proposes and filters
Protex masks; setting its `kind` to BEACH or supplying a mixed inventory does not
make it a mixed-class detector.

- **Only Protex, same count, different positions/yaws:** reuse the same config.
  Seeds are regenerated automatically on every new run.
- **Only Protex, different count:** set
  `perception.automatic_scene.expected_instances` to the actual count (2–4 for this experiment set).
  The S01, S05 and S11 templates specify two, three and four respectively. When
  automatic mode is enabled, the manual `objects` and `points` entries are not
  used to assign initial objects.
- **Any BEACH:** use manual mode (`automatic_scene.enabled=false`) with an
  accurate `perception.objects` inventory. The supplied templates already contain
  the right class, instance ID and mesh for their advertised mix. Run the helper
  below to select one point inside each actual box; it writes the points and the
  prepared config for you. No pixel-coordinate editing is necessary.
- **Different manual-mode positions, orientations, count or types:** run the
  helper again; first choose/edit a template with the right inventory. Do not
  reuse seeds from a previous physical arrangement. During a run, tracking and
  identity retirement are handled by the existing pipeline.

A SAM2 seed is a pixel inside the intended box. SAM2 produces that box's image
mask; FoundationPose uses the mask, depth and class-specific box mesh to estimate
its 3D pose. For mixed scenes, your named clicks assign the classes—SAM2 does not
infer Protex versus BEACH here. Choose the visible top-face interior, away from
edges and neighbouring boxes. Boxes of the same class may be named in any order;
those names then remain bound within that run. Drawing labels are only references,
so automatically assigned names may differ from the labels shown.

For a custom inventory, each object entry must follow this format, with no entries
for boxes absent from the table:

```json
"beach_1": {
  "class": "beach",
  "instance_id": "1",
  "mesh": "real_robot/models/beach_110x65x25mm.obj"
}
```

The nominal meshes are already present on this machine. On another checkout:

```bash
.venv-real/bin/python real_robot/make_box_mesh.py --kind protex --out real_robot/models/protex_90x60x28mm.obj
.venv-real/bin/python real_robot/make_box_mesh.py --kind beach --out real_robot/models/beach_110x65x25mm.obj
```

## Run a scene

Run commands from `/home/hao/Workspace/Code/planning_with_knock`.
First arrange the boxes flat and visible, with space to slide outward inside the
source area. Keep the fixed camera and table unchanged. Empty/reset the bowls
between trials; current release TCP points are **(600, −200, 60)** and
**(600, +200, 60) mm**.

For **S11, four Protex**, the config is ready to use:

```bash
# Fresh perception, initial plan and first-action waypoint preview; no robot connection.
.venv-real/bin/python real_robot/run_pipeline.py --config real_robot/experiment_design/configs/S11.json
# Physical execution; the program announces motion before issuing it.
.venv-real/bin/python real_robot/run_pipeline.py --config real_robot/experiment_design/configs/S11.json --execute
```

For **S07, two Protex and two BEACH**, prepare its seeds first:

```bash
.venv-real/bin/python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S07.json
```

This resets the RealSense, waits for reconnection, captures one image and opens a
window. The live preview/execute command also resets once before its initial
capture; subsequent post-action observations do not reset the camera. For each displayed name, left-click
inside the corresponding actual box, then press Enter. Right-click removes a
mistaken point before accepting; closing the window cancels setup. The helper
saves the image, seeds, annotated selection and ready-to-use `pipeline_config.json`
under `real_robot/experiments/scene_setup/<timestamp>/`. It prints the exact preview
and execute commands using that prepared config. Keep the scene unchanged between
seed selection and pipeline startup. The helper does not connect to the robot.
The manual templates deliberately have no fabricated seed points: pass them to
the helper, not directly to `--execute`.

A preview and an execute invocation create **separate runs**, each with a fresh
initial observation and one initial solver plan. The execution run then preserves
its plan throughout its actions. A preview checks perception/planning and the
first action's generated waypoints; it is not a physical reachability test for
every later action. For an initial action trial, `--execute --single-action`
executes one complete Pick or Knock batch, retreats and observes, then stops. A
Knock plus its following Pick are two batches. A later invocation starts a new run,
not a resume of that stopped plan; refresh manual seeds if the scene has moved.

## Repeat a template

1. Return the full inventory for that template to the source area and empty/reset
   the bowls. You may approximately reproduce the previous layout or choose new
   positions and yaws. Keep the boxes flat and visible and leave the camera fixed.
2. For **Protex-only S01, S05 or S11**, rerun its execution command. Automatic
   startup regenerates the seeds from the current scene. For example, two Protex:

   ```bash
   .venv-real/bin/python real_robot/run_pipeline.py --config real_robot/experiment_design/configs/S01.json --execute
   ```

3. For **any template containing BEACH**, run seed setup again after resetting or
   rearranging the boxes. For S03 (one Protex and two BEACH):

   ```bash
   .venv-real/bin/python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S03.json
   ```

   Click each requested box and press Enter. Run the **new execution command
   printed by this helper**, which points to its newly prepared configuration.
   Replace S03 with the desired template ID. Do not run the manual template
   directly with `--execute`, or reuse the prepared seeds from before a physical
   run/reset. If you only ran a preview and nothing moved, you may reuse that
   prepared configuration for execution.
4. Record the new run directory and increment `trial` in a copy of the results
   sheet. If deliberately testing another arrangement, also change `layout_variant`
   (for example A to B). Each invocation saves a separate timestamped run with a
   fresh initial detection and one fixed plan; previous run data is retained.

A new invocation starts from the current scene; it does not resume an interrupted
plan. For an independent repeat after a partial run, first restore all boxes for
the selected template. No editing of reference coordinates or manual 3D poses is
needed for any repeat.

## BEACH height model

BEACH uses a **110 × 65 × 25 mm** nominal mesh and catalog entry; Protex stays
90 × 60 × 30 mm. With the existing table-at-base-Z=0 Knock rule, BEACH now uses 4 mm top indentation
and 1 mm slide descent: contact TCP Z is **21 mm**, ending at **20 mm**. This lowers
the previous contact command by 2 mm without changing the 25 mm model.
Protex remains 28 → 27 mm. Pick retains the median table estimate and uses
2 mm modeled fingertip clearance for BEACH (another 4 mm lower than its previous
6 mm setting; 8 mm lower than the original 10 mm formula),
with 10 mm for other classes. Release remains absolute TCP Z=60 mm.
All classes now slide 80 mm per Knock at 4 mm/s, about 20 seconds per stroke.

Prepared configs made before this correction may still reference the old
`beach_110x65x30mm.obj`, `beach_110x65x28mm.obj` or `beach_110x65x26mm.obj`. After restoring the scene, rerun the setup helper with the
updated template and use the newly printed command. Previous run snapshots and
the old mesh are retained as historical records.

## BEACH grasp baseline before mixed trials

Start with S01 and S02, then the close grids/mixed scenes. This checks each box
type's actual grip before interpreting dense-scene failures as planning failures.
Protex has a **nominal 50 mm closing-aperture override**. BEACH now has a
**50 mm override**, matching Protex, for its nominal 65 mm short side. This
commands **495 pulses**, replacing the previous 60 mm / 594-pulse target. Opening for approach and release remains 842 pulses;
Knock still closes fully to 0 pulses. These are SDK pulse-mapping estimates, not
measured jaw openings or grasp-force measurements.

The settings are `pick_aperture_mm_by_class.protex = 50` and
`pick_aperture_mm_by_class.beach = 50` in
`real_robot/execution_config.batch.local.json`. A successful Protex grip does not
establish that the BEACH command will hold the box. Observe the actual BEACH grip
and record any later adjustments; do not alter the box width to tune closing.
Keep the selected settings constant across comparison trials. Speeds, drop points and planning shrink retain their existing settings.
The recorded latest BEACH Pick becomes TCP Z=10.190 mm. Another recorded scene
estimate becomes 9.969 mm, so the controller minimum TCP Z was lowered to 9 mm,
verified by readback and acknowledged save. Other controller boundary values and
protections remain unchanged. Actual future heights still depend on the newly
estimated table Z.

## Inspect plans and record outcomes

Each run prints `RUN DIRECTORY`. Under that directory:

| Artifact | Relative path |
|---|---|
| Initial detected layout image | `observations/0000/planning/layouts/learned_pose_scene.png` |
| Initial plan overlay | `observations/0000/planning/solution_overlays/learned_pose_scene_ilp_highs.png` |
| Initial solver layout/plan data | `observations/0000/planning/layout_and_plan.json` |
| Fixed execution plan | `observations/0000/planning/fixed_execution_plan.json` |
| Plan sequence images | `observations/0000/planning/solution_sequences/` |
| Actual observations/pose data | `observations/NNNN/` |
| All precomputed stages for an action | `actions/NNNN/action.json` |
| SDK events and timestamps | `actions/NNNN/execution.jsonl` |
| Post-action assessment | `actions/NNNN/outcome.json` |
| Completion/attempt counts | `state.json` |
| Failure description, if stopped | `failure.json` |
| Configuration snapshots | `pipeline_config.json`, `execution_config.json` |

Copy [results_template.csv](results_template.csv) for your results. Record scene,
layout variant, trial, actual box counts, run directory, commit, number placed, Knock/Pick batch
counts, elapsed time, failure stage, and any manual intervention. Use the initial
capture/overlay to document the actual arrangement, rather than claiming it
exactly matches the reference. A visually confirmed placement in the bowl is the
physical success measure: source removal alone does not establish correct bowl
placement. Count interventions and stopped runs as such, rather than silently
omitting them. If practical, repeat each scene three times; thirteen templates with one layout each and
three repeats means thirty-nine runs. Additional layout variants add more runs. One run per scene is a useful exploratory pass.

Reset the complete scene between independent trials. After a failure, preserve
its run directory and record the stage before changing settings. A new execution
invocation always detects and plans from the scene present at its start.

## Regenerate the references

```bash
.venv-real/bin/python real_robot/experiment_design/generate_examples.py
```

This uses current local pipeline settings as the base for separate templates. It
validates nominal footprints and regenerates the drawings/configs, preserving
existing results rows and appending missing template rows. It does not change the working default configuration
or the Isaac pipeline, capture an image, or connect to the robot. Treat these
files as disposable experiment aids; actual run data remains under
`real_robot/experiments/pipeline/`.

## Class destinations and release heights for S07, S12 and S13

These three templates group Protex into `bowl_1` at link_base XY **(600, −200) mm**
and BEACH into `bowl_2` at **(600, +200) mm**, independently of the fixed plan order.
Their absolute placing TCP heights are:

| Template | Protex release TCP Z | BEACH release TCP Z |
|---|---:|---:|
| S07 | 60 mm | 60 mm |
| S12 | 80 mm | 60 mm |
| S13 | 110 mm | 60 mm |

The increased heights apply to every Protex placement in that template. They are
absolute TCP heights, not box-bottom clearances or per-object stacking increments.
Pick/Knock contact heights, slide distance, speeds and the 300 mm transit remain
controlled by the shared execution configuration.

Template `execution_overrides.class_to_bowl` and
`execution_overrides.release_tcp_z_mm_by_class` overlay the shared execution file
when the pipeline starts. Class routing takes precedence over individual-instance
routing in these templates. Templates without these settings retain their existing
routing and release-height behavior. The effective settings are saved in each
run's `execution_config.json`.

**Prepare a new scene after updating the templates.** Old prepared configurations
are snapshots and retain their previous placement settings. For example:

```bash
python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S12.json
```

Then use the newly printed preview/execute command. Substitute S07 or S13 as needed.

## Automatic statistics for every pipeline trial

Each new run saves **`summary.md`** for reading and **`summary.json`** for analysis
at the root of its printed run directory. They include:

- Initial fixed-plan order, object count, planned Picks, planned Knocks and total
  planned action batches. A planned Knock requires a subsequent Pick: one Knock
  plus one Pick counts as two planned actions.
- Actual Pick/Knock attempts, Knock retries, completed batches, per-object attempts,
  source removals, remaining targets, run status and failure reason.
- SDK Knock slides started/completed, separately from attempts that can fail during
  controller initialization before any action motion.
- Total elapsed time, observation, planning, candidate selection, waypoint generation,
  batch execution and outcome assessment. Nested details include camera reset/capture,
  SAM2/FoundationPose subprocesses (including recovery), SDK home/motion/gripper stages,
  and robot connection/setup/teardown. Per-action timings and placement poses are in JSON.

Durations use a monotonic clock. Total timing starts at argument parsing and ends
before summary writing; imports and summary serialization are excluded. Nested
perception/robot details are already included in the high-level phase totals.
`run_timings.jsonl` records completed and failed timed calls; SDK timestamps remain
in `actions/NNNN/execution.jsonl`. Preview, configuration-check, ordinary failure and
Ctrl+C exits also produce summaries. A process killed with SIGKILL or power loss
cannot finalize its summary; the existing event logs remain available.

An attempt means an executor invocation; the slide counters identify commands that
actually reached the Knock stage. Source removal does not verify successful bowl
placement. Simulations are explicitly labeled and their timing does not predict
physical execution time. Existing historical runs are not modified.

A successful full execution now returns the robot to its configured joint home
after the last Pick retreat and removal observation. No command changes are
needed. `summary.md` / `summary.json` include the final-home status and time;
`final_home/execution.jsonl` contains its SDK log. Failed plans and
`--single-action` runs do not trigger this additional motion.
