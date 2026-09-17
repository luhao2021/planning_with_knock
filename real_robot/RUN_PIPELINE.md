# One-command xArm pipeline

```bash
python real_robot/run_pipeline.py --execute
```

Run inside `.venv-real` from the repository root. The local settings are in [`execution_config.batch.local.json`](execution_config.batch.local.json); [`pipeline_config.local.json`](pipeline_config.local.json) contains the model paths and retry limits. The fallback examples are included for another checkout.

The logic is:

```text
Detect the current boxes with SAM 2 + FoundationPose
Generate ONE initial layout, solver plan and overlay
Keep that target order, Pick witnesses and local Knock directions fixed
Use the latest poses to generate every waypoint for the current action
Execute the entire action and retreat
Detect again; check clearance for the fixed grasp, retry or advance in that plan
```

There are no images, pose queries, contact checks, or replanning between stages. The post-action observation is also the next action's starting observation. SDK `wait=True` sequences the commands, and an SDK error stops the batch. The code does not disable the robot controller's collision detection or emergency stop.

## Camera startup reset

A live pipeline invocation resets the connected RealSense through its official
SDK before observation 0, waits five seconds, then allows up to twenty more
seconds for the same serial to reconnect. The setup helper and standalone
`capture_rgbd.py` command also reset before their capture. Reset/reconnection
messages are printed. This recovers the observed depth-stream startup stall.
Post-action observations capture normally without resetting. Recorded-input
previews, simulation, imports and configuration checks never reset the camera.
The camera reset does not connect to or move the robot.

## Manual pose translation correction

The current `perception.base_pose_offset_mm` is `[15, -10, 0]`. For every detected
instance, after the supplied camera-to-link_base transform, add 0.015 m to base X
and subtract 0.010 m from base Y. Base Z, orientation and nominal dimensions are
unchanged. This is a user-requested empirical correction, not a newly measured
camera calibration. The supplied camera transform itself remains recorded intact.

Each observation saves `observed_objects_uncorrected.json` and corrected
`observed_objects.json`. Corrected records include the original centre/transform
and the applied offset. Layouts, source-area checks, action waypoints and
post-action assessments use corrected observations. FoundationPose mask/depth
validation still uses its original camera-frame poses. The offset is applied once
per fresh conversion, including every observation after an action; reapplying it
to an already corrected record is rejected. Recorded RGB-D previews use the same
correction. Synthetic world coordinates are already in the robot frame and are
not camera detections, so this camera correction is not applied to them.

Drop points are absolute robot coordinates `[600,-200]` and `[600,200]` mm. X remains 600 mm and Y was widened to ±200 mm at the user's request. Pick transfer and placement remain Cartesian SDK segments.
The correction translates the entire detected scene and preserves relative gaps.
Tiny planning-only overlap corrections still run afterward as a separate step.

## Contact heights

The local and example configurations now express the user's requested contact geometry:

| Setting | Current value and meaning |
|---|---|
| `pick_fingertip_clearance_mm` | Default 10 mm minimum modeled clearance above the estimated tabletop throughout finger closing. |
| `pick_fingertip_clearance_mm_by_class.beach` | 2 mm: lowers BEACH Pick TCP Z by another 4 mm from the previous 6 mm setting (8 mm below the original 10 mm formula), for the same detected poses. |
| `knock.nominal_height_tcp_classes` | `protex`, `beach`: use the class-specific nominal top at base table Z=0: Protex 28 mm, BEACH 25 mm; command TCP heights directly. |
| `knock.top_indent_mm` | Default 2 mm below the selected top reference: Protex TCP Z = 28 − 2 = 26 mm. |
| `knock.top_indent_mm_by_class.beach` | 4 mm below the 25 mm BEACH top: contact TCP Z = 21 mm, 2 mm lower than its previous command. |
| `knock.slide_descent_mm` | Descend a further 1 mm while sliding outward 80 mm: Protex slide ends at TCP Z = 25 mm; BEACH at 20 mm. |
| `release_tcp_z_mm` | Open at absolute TCP Z = 60 mm in `link_base`, as requested. This is not box-bottom clearance. |

`pick_tcp_z_mm` and `knock.contact_tcp_z_mm` remain optional absolute overrides. They are null in the current configuration because their values are calculated for each action; they are no longer missing inputs. Explicit overrides take precedence. No robot motion is needed to calculate these waypoints.

The executor sets the TCP offset to `[0,0,172,0,0,0]`. A read-only official-SDK report on 2026-09-15 confirmed that exact controller setting and a zero world offset; see `experiments/tcp_readback/20260915_010352/controller.json`. Reading the offset does not measure fingertip geometry or the table.

Pick and the fallback Knock calculation use the nominal standard G1 gripper model in `standard_gripper_geometry.py`. Source files downloaded from UFACTORY are retained in `models/xarm_gripper_geometry/`. With the standard downward orientation, the modeled closed fingertips extend about 2.4904 mm below the configured TCP. Maximum extension during the full finger travel is about 3.1 mm; Pick uses this maximum so an approximate pulse/aperture mapping cannot reduce modeled clearance below the selected 10 mm default or 2 mm BEACH value.

For each action, using the existing pre-action detection:

- Estimate table Z as the median of active source-box centre Z minus half nominal box height. This assumes flat boxes on the same table; it is an estimate, not a newly measured table plane.
- Pick TCP Z = estimated table Z + class-specific fingertip clearance + maximum fingertip extension. Default clearance is 10 mm; BEACH uses 2 mm, another 4 mm lower than its previous 6 mm setting for unchanged poses.
- Protex/BEACH Knock: use each class’s nominal flat-box height at base table Z=0. Protex centre/top Z = 14/28 mm gives TCP Z = 26 mm, ending at 25 mm. BEACH centre/top Z = 12.5/25 mm with its 4 mm indentation gives TCP Z = 21 mm, ending at 20 mm. No fingertip-extension compensation is added. Detected XY/yaw still determine the stroke.
- Toothpaste and classes outside `nominal_height_tcp_classes` retain the fallback Knock rule: target centre Z + half nominal box height − 2 mm + closed-fingertip extension. The single straight sliding segment ends 1 mm lower.
- Release uses TCP Z = 60 mm exactly.

The Protex/BEACH rule uses direct absolute TCP heights: **Protex 28 → 27 mm;
BEACH 21 → 20 mm**. BEACH retains its 25 mm model height; the latest requested
2 mm contact reduction is a per-class indentation setting, not another model
change. BEACH permits the explicitly requested 4 mm initial/5 mm final commanded
indentation, while other classes retain the 3 mm total limit. No retry increases
these settings. These
commands are not measured fingertip indentation. If the nominal gripper model
were exact, closed fingertips would be about 2.49 mm below these TCP positions;
the model has not been physically calibrated. The height rule assumes boxes lie
flat on the base Z=0 tabletop. Previously prepared configs referencing the 30 mm, 28 mm or 26 mm
BEACH mesh should be regenerated using the updated scene template before the next
trial; historical run files and the original mesh are retained.

Raw detected poses remain in the observation and action records. Only the Knock height calculation uses the constrained nominal centre/top, recorded alongside the detected top in `height_geometry`. Pick keeps its existing median table estimate and maximum-extension clearance calculation. Release stays at TCP Z=60 mm. No retry increases the top offset or slide descent.

The controller minimum TCP Z was lowered from 10 to **9 mm** under the user's
instruction to lower the boundary if the new Pick heights need it. In the latest
recorded BEACH run (`20260915_211038_aebcd7`), the new 2 mm clearance gives Pick TCP
Z estimates of **9.969–10.715 mm**, including a last Pick at **10.190 mm** (previously
14.190 mm with 6 mm clearance). The minimum estimate would cross the former 10 mm
floor. The SDK acknowledged both the boundary update and save, and readback
confirmed 9 mm with the other boundary values, fence mode and collision rebound
unchanged. The arm stayed idle with no faults or motion commands. See
`experiments/controller_settings/20260915_214053/boundary_update.json`.

These are recorded-input calculations, not fresh detection or physical validation.
Future absolute Pick Z still depends on the estimated table Z; at table Z=0 the
formula gives about 5.10 mm, below the current controller floor. The program does
not automatically lower the controller boundary in response to a new pose.

The example contains no `*_calibrated`, `*_reviewed`, `provisional`, `approved_plan_id`, or bowl-check flags. The program checks only values needed to form a motion command. Old approval flags, if present in an older configuration, are ignored.

To inspect configuration completeness without hardware access:

```bash
python real_robot/run_pipeline.py --check-config
```

The following settings are already supplied:

- IP: `192.168.1.231`; standard controller TCP: `[0,0,172,0,0,0]`.
- Drop-point XY: `(600,-200)` and `(600,200)` mm. Bowls alternate, or `instance_to_bowl` supplies a manual assignment.
- Initial aperture estimate: 842 pulses at approximately 85 mm; full close: 0 pulses. This uses the user's approximate opening measurement rather than claiming a new aperture calibration.
- Standard gripper closing axis: tool Y, represented by `tool_closing_axis_yaw_offset_deg=90`. UFACTORY's [gripper model](https://github.com/xArm-Developer/xarm_ros2/blob/humble/xarm_description/urdf/gripper/xarm_gripper.urdf.xacro) places the opposed fingers on ±Y and its TCP at Z=172 mm. This assumes the standard mounting orientation.
- Working footprint allowance: 2 mm total reduction per XY dimension, as previously requested. Raw measured model dimensions and estimated poses are preserved.
- Planning finger thickness: 20 mm (`workflow.finger_thickness_mm`), increased from 15 mm at the user's request. Initial real-robot Pick planning reserves the full 85 mm open aperture plus both 20 mm fingers and a 3 mm margin on every side of the 30 mm contact window. It uses a solid envelope covering both finger sweeps. `workflow.planning_grasp_margin_mm` controls that additional planning margin. The setting is also saved in each layout JSON.
- Maximum planning centre correction: 5 mm in XY per object (`workflow.max_pose_correction_mm`), approved to accommodate packaging dimensions and detection noise. This is a bound on the correction vector length; the optimizer enforces that bound while choosing the smallest corrections, rather than rejecting an unconstrained solution afterward. It does not change box dimensions or contact heights.

If the robot is fitted with a rotated gripper mount or a different TCP, those numerical settings must reflect that installation. These calculations assume flat boxes on a common tabletop and the standard gripper with the configured downward orientation. Object dimensions are not changed by contact-height calculation.

## Actions

**Knock:** close to 0 pulses → approach above the target → descend to the calculated contact Z → slide outward while descending the configured 1 mm → retreat vertically. The SDK executes the slide as one straight Cartesian segment. The current per-attempt travel is 80 mm at 4 mm/s, approximately 20 seconds per slide. This distance applies to all classes. With `approach_speed_scale=2.5`, pre-knock/approach speeds are 187.5/112.5 mm/s; near-contact runs at 37.5 mm/s and final contact approach at 5 mm/s. Vertical release and retreat retain 22.5/45 mm/s. Failed attempts never automatically lower the contact height or increase the stroke.

The `−x`/`+x` solver labels are **object-local**. They are rotated using the detected yaw to obtain `link_base` motion. For the approximately 90°-rotated Protex boxes, these become approximately robot `−y`/`+y`. The saved action JSON includes both directions.

**Pick:** pre-pick → open → approach/grasp → close → lift → transfer → place/open → retreat. The solver chooses the closing axis and the 30 mm finger-contact window. Protex uses `pick_aperture_mm_by_class.protex = 50`, commanding 495 pulses rather than the previous 584 pulses. This replaces the former nominal 60 mm width minus 1 mm target (59 mm). The mapping remains an uncalibrated linear estimate, `pulse = round(842 × target_mm / 85)`: neither 50 mm nor 59 mm is a measured physical opening. BEACH uses `pick_aperture_mm_by_class.beach = 50`, commanding 495 pulses, the same as Protex. This replaces its previous 60 mm / 594-pulse target. Its physical model width remains 65 mm. This is a nominal aperture command, not measured finger spacing or force control. Other classes retain nominal closing width minus 1 mm unless given their own explicit override. The planning footprint shrink does not reduce the physical jaw command. This remains position control, not force control. Pick pre-pick runs at 225 mm/s, approach at 150 mm/s, and grasp descent at 20 mm/s. Transfer retains 135 mm/s, retreat 90 mm/s, lift 67.5 mm/s, and placement descent 12 mm/s. The gripper speed setting is 1500 (previously 1000).

For the failed Pick in `experiments/pipeline/20260915_035152_c4d930`, the descent, 584-pulse close and lift returned SDK success; failure was in the elevated transfer to `[700,-100,300]` with RPY `[180,0,79.71895]` degrees. This does not establish that the jaws physically held the box. A read-only controller IK check found no solution for that elevated target (code 10), but found solutions with FK agreement at `[600,±100,300]` and `[600,±100,60]`. Records are under that run's `diagnosis/20260915_040439/`. The checks apply to this recorded orientation and are endpoint kinematics, not collision-free trajectory validation. Place bowls at the new 600 mm drop locations. Release remains absolute TCP Z=60 mm.

RRT has not been introduced: the current pipeline still uses Cartesian SDK segments with the configured downward orientation. RRT cannot make an infeasible fixed endpoint pose reachable. Reaching farther with a tilted gripper would require a different goal orientation and validated joint/whole-gripper/scene collision planning. The requested closer-drop fallback is used here; it has not yet been physically validated. The legacy standalone executor retains its historical settings.

The `motion_speed_scale=1.5` applies to lift, transfer, place and retreat. `approach_speed_scale=2.5` overrides it for pre-pick/pre-knock, approach, near-contact, contact, and grasp descent; this is 2.5× the original base speeds, about 1.67× their previous 1.5× settings. If omitted, it defaults to `motion_speed_scale`. The Knock stroke uses the unchanged absolute `knock.speed_mm_s=4`. The gripper has its own `gripper_speed_setting`. Cartesian acceleration remains 50 mm/s², so short movements may not achieve the full requested speed increase in elapsed time.

Before each batch the program connects and checks controller state/errors. States 4/5 with no active error or warning are recoverable on a new invocation: send state 4 to discard stopped buffered commands, set the TCP offset, check the initial pose, then call `motion_enable(True)`, `set_mode(0)` and `set_state(0)`. Setting TCP offset can itself enter state 5, so readiness is restored afterward. Startup reads verify state 0/2 and no faults, with at most 3 seconds of settling; enable/reset is attempted only once. Moving or paused controllers are rejected. Active faults are reported rather than automatically cleared.

At the start of each invocation, the executor checks initial TCP Z before enabling. **Z must be strictly above 100 mm**; at or below that threshold it stops for manual clearance/repositioning. It then moves all seven joints to `startup_home.joint_angles_deg = [0,-30,0,60,0,30,0]` degrees at 20 deg/s with acceleration 50 deg/s². This happens even when the initial TCP is already above 300 mm. The previous vertical-only startup lift has been replaced. Homing runs once per executor/invocation, before the first action, and is not repeated between actions.

These joints come from the working `test_xarm.py` example. A read-only controller query with TCP offset `[0,0,172,0,0,0]` and zero world offset returned home TCP Z=553.176 mm and no joint-limit violation; see `experiments/home_readback/20260915_031441/controller.json`. Startup checks controller joint limits and forward kinematics again before enabling, requiring the predicted home Z above both 300 mm and transit height. The home move is announced and logged in `execution.jsonl`, waits for completion, then verifies all seven joints within 1 degree, measured TCP above 300 mm and at least transit height, and ready state/no faults before beginning action stages. No gripper aperture change is commanded during homing. A failed home aborts the run.

The 100 mm starting-height guard and valid home endpoint do not establish that the intervening joint-motion path is clear of objects: the arm can sweep sideways or down during a joint move. This startup path still needs physical validation. Motion is announced before enable and home/batch commands. Stops/errors during a batch terminate it, with no automatic re-enable or partial-action resume. No state/pose queries are added between action stages. This follows the [official enable/mode/state initialization](https://docs.api.ufactory.cc/xarm_python_sdk_docs/2.-linear-motion.html) and [TCP-offset state behavior](https://docs.api.ufactory.cc/xarm_python_sdk_docs/4.-Motion%20Parameters.html).

## After each complete batch

- After Knock: check the same fixed Pick witness at the newly detected poses. Once its full finger sweep is clear, execute the mandatory Pick. Otherwise retry the original object-local Knock direction, rotated by the latest symmetry-aligned execution yaw. The initial plan supplies that outward direction; the loop does not repeat the abstract Knock-corridor test on later noisy poses. No solver call or new grasp search occurs.
- After Pick: immediately retire the cuboid and tracking label once the SDK completes the drop, opening, and retreat. Advance without source-removal or bowl detection. Subsequent perception includes only remaining identities. SDK failure stops execution without retirement.
- Only requested action targets must pass fresh pose validation. Unrelated pose failures do not gate the current action. A requested target with unrecoverable detection failure, a tipped current target, or an SDK failure still stops with the reason. A missing detection is not proof that a box was placed.


Limits are **25 Knock attempts per object, one Pick attempt per object, and 120 total actions**. The attempt limit counts total executions, including the first attempt, within one invocation. New runs start new counters. Measured outward progress and no-progress streaks are logged, but do not stop retries before the 25-attempt limit. No retry increases indentation, stroke length, speed or grip compression. The terminal prints the selected action and attempt number/limit; an exhausted limit reports both numbers. These prevent an unattended retry loop. The initial layout is solved once; subsequent observations only update poses and geometric execution checks. Mask/depth validation applies to requested targets after observation, with one recovery attempt when needed, never during motion.

Before selecting a real Pick, the pipeline checks both fingers over their full travel from the commanded 85 mm opening to the configured closing target (Protex: estimated 50 mm; BEACH: estimated 50 mm; otherwise nominal target width minus gentle compression). Each finger uses the existing 30 mm contact window and 20 mm planning thickness. Collision checks use the nominal, unshrunk box dimensions and observed base poses after the manual XY offset, before any planning-only overlap correction. Both the approach footprint with open fingers and the intermediate closing positions are checked. The local `pick_overlap_tolerance_mm=3` accepts up to 3 mm minimum separating translation between the conservative finger rectangle and a neighboring nominal box. Tolerated overlaps are logged in `overlap_details`/`tolerated_neighbors` and announced when the Pick is selected; larger overlaps remain blockers. This is a tolerance on estimated XY geometry, not permission to ignore a controller collision stop. The 20 mm finger model and box dimensions are unchanged. Set this option to 0 for strict checking; allowed configuration range is 0..5 mm. For aligned Protex boxes, full opening places each finger 12.5 mm farther outward than the 60 mm target surface, so a parallel neighbor overlapping the grasp window needs more than 32.5 mm clearance (12.5 + 20) with strict checking. With the current 3 mm overlap tolerance, an estimated aligned gap of at least 29.5 mm passes; a 5 mm gap still fails. This is the commanded opening geometry, not a new fixed planning margin or dimension change.

The shared solver and Isaac pipeline are unchanged. Real-robot initial Pick rules use an independent conservative open-finger envelope in `conservative_pick_planning.py`. Its full transverse span is 85 + 2 × 20 + 2 × 3 = 131 mm, and its tangential contact window is 30 + 2 × 3 = 36 mm. Neighbor obstacles use nominal dimensions and the original observed XY/yaw, so the approved footprint shrink and overlap correction cannot create artificial grasp clearance. Those planning-only corrections remain in the abstract layout and Knock geometry. Box dimensions and the approved shrink are unchanged.

This planner envelope contains the execution finger sweeps. For the same poses and grasp, every execution blocker must therefore also be a planning blocker; planning can be stricter. For aligned Protex boxes sharing a grasp window, planning needs more than 35.5 mm gap, while execution with its 3 mm noise tolerance accepts at least 29.5 mm. Planning does not call the execution acceptance function or filter a completed plan afterward. The solver runs once on these conservative rules, and the saved overlay and fixed execution plan describe that result.

Execution uses the latest poses for the fixed grasp. A planned Knock/Pick pair repeats its fixed Knock until that grasp passes execution clearance. The same-pose model relationship does not guarantee feasibility after an unmodeled box movement, slip, or arbitrary pose error, nor full gripper-body/arm clearance. `build_batch` retains its execution check, and controller faults stop motion. No extra observations occur inside an action.

### Target-scoped observations

Automatic startup generates the four-Protex prompts and inventory in each run (manual mode copies the configured prompts). Observation
0 binds each configured name to its initial validated pose. `identity_registry.json`
stores initial pose, last valid pose, observation number, and active/removed status.
The initial solver plan, order, grasp offsets and Knock directions remain fixed.

After each complete batch, SAM2 initializes **only the active names**, excluding
previously removed names. Its output labels are treated as proposals: depth-backed
source masks are matched one-to-one to the remaining physical identities by XY
position. Bowl masks, table surfaces and duplicate masks are excluded. An untouched
box uses its previous matched mask centroid; a visible-surface centroid is not
compared against the different FoundationPose geometric box centre. The registry
stores these mask references separately from validated motion poses and updates
all matched active mask references after each observation. A just-knocked target
uses the segment from its last mask centroid along the commanded local-to-base direction,
up to the configured 80 mm stroke. Matching allows 50 mm around that point/segment (`workflow.identity_match_tolerance_mm`)
for movement and detection noise; it is not a
planning clearance, box-size change, or commanded-motion allowance.

After Knock, FoundationPose still estimates only the current target. After Pick,
a successful complete SDK batch (including drop, gripper opening, and retreat)
immediately retires the target and its label, before the next capture. The plan
advances without checking the picked box's mask, source footprint, or bowl pose.
Only remaining identities are passed to SAM2; FoundationPose refreshes the next
fixed-plan target. After the final Pick there is no further capture or pose call;
the arm returns home. Single-action Pick also retires its target without another
capture. SDK errors do not retire the target.

This completion policy trusts successful motion/gripper commands; it does not
visually verify grasp or placement success. `actions/<step>/outcome.json` and
`identity_retirement.json` record that basis, and `identity_registry.json` retains
the historical pose while marking the label removed. The completed-target state
is saved before detecting the next object, so a later perception failure cannot
undo a completed Pick. The legacy depth-removal helper remains available for
recorded-data analysis; it no longer gates the execution loop.

The execution scene retains last validated neighbor poses for grasp clearance and
table-height estimation. They are not silently used as a fresh target pose when
identity association fails. `execution_scene.json` records their observation
indices. `identity_association.json` records mask/name matches, rejected candidates
and association evidence. `comparisons` includes every candidate distance, reference type and threshold.
`perception_timings.jsonl` records elapsed time and exit status of each SAM2/
FoundationPose subprocess. The mask preview and `instance_masks.npz` show associated
active source identities; original SAM output remains in `tracker_masks.*` and
`tracker_masks_preview.png`. `observed_objects.json` contains fresh poses of remaining targets. A retired
box's old position remains historical evidence in the registry, never a new
detected pose or motion target.

This assumes unrelated boxes remain near their last known positions. Ambiguous or
unmatched action targets still require resolution rather than moving toward an
unverified stale target. All identity work occurs after retreat, without an extra
capture, intermediate motion queries or replanning.

If registration fails the pose/mask/depth fit check, perception first checks the
last accepted pose for each failed identity against the **current** capture and
mask. A prior pose is reused only when it passes all the unchanged validation
thresholds. It is labeled `previous_accepted_pose_revalidated`; the cache retains
separate records across target-only observations and removes retired identities.
Evidence and the prior observation index are saved under
`foundationpose/recovery/previous_accepted_pose/`. No extra capture or model call
is needed for this fallback.

For targets without a currently valid cached pose, perception retains **one model
recovery attempt**: after observation 0, register the previous saved RGB-D frame,
then track/refine on the current frame (at least 10 refinement iterations). At
initial observation, retry global registration with at least 256 hypotheses.
The same validation thresholds apply. If both revalidation and the bounded model
retry fail, the active target is not accepted. Already retired labels are not
requested or passed through pose recovery.

Each observation retains the original `foundationpose_instances.json` and
`validation.json`. `accepted_instances.json` and `accepted_validation.json` record
the poses actually used, including their source. Recovery inputs, logs, estimates
and status live in `foundationpose/recovery/`. This adds no camera capture, robot
query, plan generation or mid-action processing.

## Results and tests

Every run creates:

```text
real_robot/experiments/pipeline/YYYYMMDD_HHMMSS_unique/
  state.json
  pipeline_config.json
  execution_config.json
  initial_points.json, object_specs.json
  observations/0000/
    capture.npz, capture_rgb.png
    masks/, foundationpose/, observed_objects_uncorrected.json, observed_objects.json
    planning/layout_and_plan.json
    planning/fixed_execution_plan.json
    execution_selection.json
    planning/plan.txt
    planning/layouts/
    planning/solution_overlays/
    planning/solution_sequences/
  observations/0001/  # capture, masks and poses only; no new planning/ folder
    execution_selection.json  # current fixed-plan step and geometric checks
  actions/0000/
    action.json
    execution.jsonl
    outcome.json
```

All waypoints are saved before sending the batch. There is no automatic resume from a partially executed action. The previous verbose local configurations were preserved in `real_robot/config_backups/`.

```bash
# Full decision loop, no physical hardware or contact physics:
python real_robot/run_pipeline.py --simulate

# Capture + learned perception + solver preview, no robot connection:
python real_robot/run_pipeline.py

# Same preview on an existing RGB-D file, without taking a camera frame:
python real_robot/run_pipeline.py --recorded-capture real_robot/experiments/protex_4_20260914_183700/capture.npz

# Offline regression tests, including a fake SDK:
python -m unittest discover -s real_robot -p 'test_*.py' -v
```

If neither a TCP override nor the corresponding geometry setting is supplied, a preview still saves the layout and solver plan, plus `action_request.json` identifying the missing height. It does not fabricate a waypoint. Simulation uses the configured height geometry, supplying synthetic fallback heights only when neither a TCP override nor a geometry setting is supplied. The earlier standalone `xarm_executor.py` and its configuration are retained for historical compatibility; the current pipeline does not call that executor or its validation logic.

## Next physical validation

The updated contact motion has not been physically validated. After a fresh preview
(`python real_robot/run_pipeline.py`), a bounded trial is available:

```bash
python real_robot/run_pipeline.py --execute --single-action
```

This executes exactly one solver-selected Pick or Knock through its final retreat,
then captures again, assesses the outcome and saves the updated execution cursor. It does not regenerate a plan or overlay. A run with
remaining boxes reports `single_action_complete`; this does not mean all boxes are
placed. Detection or SDK failures still stop with a reason. The usual `--execute`
command continues through the full loop. No calibration or approval flags are added.

## Scene inventory

Automatic startup seeds are enabled in the local and example configurations:

```json
"automatic_scene": {
  "enabled": true,
  "kind": "protex",
  "max_instances": 4,
  "expected_instances": 4,
  "mesh": "real_robot/models/protex_90x60x30mm.obj"
}
```

Place the four visible, flat Protex boxes anywhere within the camera-visible
source workspace (base X=300..570 mm, Y=-250..250 mm). Their initial arrangement
may change between runs; no manual pixel edits are needed. The one startup
RGB-D capture supplies depth-directed SAM2 automatic proposals. The selector
excludes bowls/background using source position, visible height and nominal
Protex footprint ranges, removes duplicate masks, then chooses an interior seed
for each selected box. The nominal motion/planning box dimensions are unchanged.
Names are assigned deterministically by initial image position and are bound for
that run. They need not identify the same physical packaging across separate runs.

The configured count is four. If the first selection cannot find four distinct
boxes, it retries once with a denser prompt grid and less restrictive SAM proposal
scores on the **same capture**; physical mask selection criteria still apply.
Once four valid masks are found, unrelated raised background pixels do not veto
that complete inventory. If recovery cannot obtain the configured count, startup
reports the saved diagnostic rather than generating an incomplete plan.

Every run saves `initial_points.json`, `object_specs.json`, and
`observations/0000/automatic_scene/automatic_seeds.png`. `initialization.json`
records whether the first attempt or recovery was used; recovery artifacts are
under `automatic_scene/recovery/`. Generated seeds initialize the existing tracker,
FoundationPose and the single fixed plan. Automatic seed generation runs only at
startup, never during action stages or after every Knock.

`perception.points` and `perception.objects` are retained only as a manual fallback
when `automatic_scene.enabled` is false. They are not read in automatic mode. The
current configuration expects four Protex boxes; changing the inventory requires
matching the automatic count/class configuration rather than editing pixel seeds.

Nominal supported classes are `protex` (90 × 60 × 30 mm), `beach` (110 × 65 × 25 mm),
and `toothpaste` (210 × 50 × 40 mm), shared by mesh generation, pose conversion and
depth validation. For a changed scene, list only the present objects in the points
JSON and `perception.objects`, using `class_instance_id` names and matching meshes.
The available inventory is up to four Protex, four BEACH and two toothpaste boxes.
Meshes can be reproduced with `make_box_mesh.py --kind beach` or `--kind toothpaste`
and `--out PATH`; local meshes are under `real_robot/models/`.

The existing solver still plans XY footprints. Per-object nominal heights enter the
Pick table estimate and Knock contact height; mixed-height contact mechanics and
full gripper-body collisions are not validated by its abstract plan. Neither the
shared Isaac code nor the approved 2 mm local planning shrink is changed here.

## Inspect a run's plan

Within `experiments/pipeline/<run>/observations/0000/planning/`, inspect:

- `layouts/learned_pose_scene.png`: corrected, shrunk planning footprints.
- `solution_overlays/learned_pose_scene_ilp_highs.png`: solver plan overlay.
- `solution_sequences/learned_pose_scene_ilp_highs.png`: planned sequence.
- `layout_and_plan.json`: observed and planning poses, corrections, finger thickness, solver witnesses, conservative planning actions/dependencies and `planning_gripper_envelope`. Per-action finger sweep and tolerated overlaps are in `actions/<step>/action.json`.
- `fixed_execution_plan.json`: the immutable target order, initial action types, fixed Pick witnesses and local Knock directions. For a Knock/Pick pair, the 30 mm Pick window is fixed at the end exposed by the local Knock direction, using the initial planning footprint and usable jaw opening.

`observations/<step>/execution_selection.json` records the current fixed-plan step,
attempt count and current fixed-grasp clearance checks. `actions/<step>/action.json`
contains the latest-pose waypoints for that complete action.

`observations/0000/masks/tracked_masks_preview.png` shows the camera image with
instance labels. Later observations contain updated captures, masks and poses,
with no new plan or overlay. The single solver overlay shows the initial abstract
footprint plan. Execution checks the full open-to-closed finger sweep on the latest
nominal box poses; full gripper-body and arm collision clearance is not established
by that check.

### Experiment placement and summaries

S07/S12/S13 now route Protex to XY (600, −200) mm and BEACH to (600, +200) mm.
Protex release TCP Z is respectively 60/80/110 mm; BEACH remains 60 mm. Re-run
`prepare_experiment_scene.py` with the updated template to create a new prepared
configuration. Old prepared configs are unchanged snapshots. See
[the experiment guide](experiment_design/README.md) for the exact override fields.

Every new pipeline invocation writes `summary.md` and `summary.json` in its run
folder, including stopped runs and previews. These report planned actions, actual
attempts/retries, source removals, destinations, status and elapsed time by phase.
A planned Knock includes a subsequent Pick (two batches); an executor attempt that
fails at startup is distinguished from an SDK slide command. Perception and robot
stage timing details are nested inside the total, not additional time.

### Return home after completing the plan

After the final Pick drop and retreat, a full execution
returns to the configured `startup_home.joint_angles_deg` using the same joint
speed and acceleration (currently [0, −30, 0, 60, 0, 30, 0] degrees at 20 deg/s).
The program announces this motion, checks the controller and current transit
height, and verifies the home joints and TCP height after the move. It does not
reset a controller fault or resume a stopped arm to perform final homing.

No extra flag is required. Preview, incomplete `--single-action` runs, and failed
or interrupted plans do not trigger final homing. The `--single-action` mode
always leaves the arm at its action retreat. Simulation records a synthetic home
return without connecting to hardware.

Home logs are saved in `final_home/execution.jsonl`. The summary includes a
separate `final_home` timing phase and completion status; home is not counted as
a Pick/Knock attempt. If returning home fails, the run reports stopped while
preserving the completed object-removal counts.

### Post-Knock exposed-end grasp

For a planned local +X/−X Knock, the mandatory Pick closes along local Y with its
30 mm contact window at the corresponding +X/−X end. Local +Y/−Y uses the
corresponding Y end and an X-closing grasp when that width fits the gripper.
If a transverse grasp cannot fit the usable jaw opening (for example, across
BEACH's 110 mm length), the original usable Pick witness is retained.

The follow-up grasp is fixed once at initial planning, with the same approved
footprint shrink; object order, Knock directions, dimensions, and motion settings
are unchanged. After each complete Knock and retreat, the existing finger-sweep
check determines whether to Pick or retry. There is no blanket 30 mm displacement
threshold: the finger window is 30 mm long and finger thickness is configured
separately (currently 20 mm). Rotation and neighboring footprints affect clearance.

Regression: in S07 run `20260915_230300_148c89`, the first Knock moved BEACH_2
72.4 mm outward. The former trailing-end Pick (`pick-y@-0.054`) still overlapped
BEACH_1 by 28.7 mm in the finger model. The exposed-end Pick (`pick-y@0.024`)
passes on the same saved scene with no neighboring overlap and avoids the second
Knock decision. This is recorded-data verification, not physical validation.

### Source boundary tolerance after motion

`workflow.source_boundary_tolerance_mm = 100` permits active tracked boxes up to
100 mm beyond each original source boundary. With the current nominal rectangle,
post-action mask centroids and active pose centres can occupy X = 200–670 mm and
Y = −350–350 mm. Initial inventory/automatic seed generation still uses the
original X = 300–570 mm, Y = −250–250 mm scene area.

Boundary masks must still match an active identity using the existing motion-aware
matching tolerance. Unmatched bowl masks do not become source objects. Pick
retirement uses successful SDK batch completion, not a boundary check.
These are perception/scene bounds; controller
limits and robot protections are unchanged.

The setting is included in default configs, all experiment templates, and prepared
S07 config `experiments/scene_setup/20260915_230211_ec8a35/pipeline_config.json`.

### Placement approach speed

`execution_config.batch.local.json` sets `placement_speed_multiplier = 3`.
Only the `place` descent from transit height to release height gets this additional
multiplier: 8 mm/s base × 1.5 motion scale × 3 = **36 mm/s** (previously 12 mm/s).
Elevated transfer, Pick approach/lift, retreat, Knock motion, gripper settings,
waypoints and acceleration are unchanged. Omitted multiplier defaults to 1 for
compatibility with older saved execution configurations. All templates referencing
the shared local execution file use this setting on their next invocation.

### Stable local axes across cuboid pose symmetry

Raw detection represents the cuboid long-axis yaw modulo 180 degrees. That is
sufficient for its rectangular footprint but not for signed local Knock directions
or an offset end grasp: crossing the 0/180-degree boundary can reverse both.
After every post-action observation, execution selects the equivalent yaw nearest
the same object's previous execution yaw. Small observed rotations are retained;
arbitrary 180-degree sign flips are removed before assessing/selecting the next
action. The initial plan, witnesses and object order remain fixed.

`observed_objects.json` and the camera/base pose transforms retain raw detections.
`yaw_alignment.json`, `execution_scene.json` and the next `action.json` record the
aligned execution yaw and its reference. Positions, dimensions, contact heights,
speeds and controller protections are unchanged. No camera capture or query is
added between action stages. This temporal disambiguation assumes less than a
90-degree physical yaw change between observations, as expected for the short
straight Knock motions; box appearance alone cannot identify a physical half-turn.

Recorded regression `20260916_001038_e295d8`: BEACH_2 yaw 0.79 → 176.99 →
178.21 → 0.09 degrees previously reversed local −X into robot +X. Execution now
uses 0.79 → −3.01 → −1.79 → 0.09 degrees. On the first saved post-Knock scene,
the fixed exposed-end Pick passes clearance, so another Knock is not selected.
This replay verifies software decisions, not physical contact or reachability.
