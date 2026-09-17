# Single-command pipeline

For a new installation, start with [virtual environments, libraries, GPU runtime, and model setup](SETUP.md).

See [RUN_PIPELINE.md](RUN_PIPELINE.md) for the initial perception → one fixed plan → Pick/Knock batch → pose-update loop, commands, configuration, and offline validation. `--execute` is the only pipeline mode that connects to the xArm. The historical snapshot descriptions below refer to the earlier perception-only work.

For experiment scenes, use the matching template and prepare fresh seeds with the commands below. S07, S12, and S13 use one click per box; the helper writes the configuration automatically. Automatic Protex-only templates use a separate startup seed-generation path. See [all scene templates](experiment_design/README.md).

The initial real-robot plan uses a full open-finger envelope plus a 3 mm planning margin, which is stricter than execution clearance for the same poses/grasp. Execution keeps that initial plan fixed. Small estimated Pick-envelope overlaps up to 3 mm are logged and tolerated; Knocks may use the full 25-attempt budget without a separate no-progress cutoff. Controller faults still stop execution.

## Prepare configuration and execute S07, S12 or S13

Run from the repository root:

```bash
cd /home/hao/Workspace/Code/planning_with_knock
```

Choose **one** preparation command for the scene on the table:

```bash
# S07: 2 Protex + 2 BEACH
.venv-real/bin/python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S07.json

# S12: 3 Protex + 2 BEACH
.venv-real/bin/python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S12.json

# S13: 4 Protex + 2 BEACH
.venv-real/bin/python real_robot/prepare_experiment_scene.py --config real_robot/experiment_design/configs/S13.json
```

Preparation resets the camera, captures one RGB-D frame, and opens the seed-selection
window. Click inside each requested box (near its visible centre is fine), following
the displayed names, pressing Enter after each box. It does not connect to or move the robot.
Keep the boxes and camera in place until execution starts.

The helper prints `Prepared configuration:` followed by a timestamped
`pipeline_config.json` path, plus exact `Preview:` and `Execute:` commands. Use that
**newly prepared configuration**, not the template JSON, for the following commands.
Replace `YOUR_SETUP_TIMESTAMP` below with the folder printed by the helper:

```bash
SCENE_CONFIG="/home/hao/Workspace/Code/planning_with_knock/real_robot/experiments/scene_setup/YOUR_SETUP_TIMESTAMP/pipeline_config.json"

# Optional preview: perception and planning; no robot connection.
.venv-real/bin/python real_robot/run_pipeline.py --config "$SCENE_CONFIG"

# Execute the complete fixed plan; this moves the robot.
.venv-real/bin/python real_robot/run_pipeline.py --config "$SCENE_CONFIG" --execute
```

For a one-action trial, use this **instead of** the complete-plan execution command:

```bash
.venv-real/bin/python real_robot/run_pipeline.py --config "$SCENE_CONFIG" --execute --single-action
```

A single action is one complete Knock or Pick batch, not one motion stage. It does
not resume automatically when rerun. After rearranging or replenishing the scene,
run the preparation command again, even when repeating the same template. An
unchanged scene can use its prepared configuration for preview followed by execution.

Protex goes to `(600, -200)` mm; BEACH goes to `(600, +200)` mm. Protex release
TCP Z is 60/80/110 mm for S07/S12/S13 respectively; BEACH release TCP Z is 60 mm.
After a successful Pick drop and retreat, its cuboid and label are retired before
further detection. After the full plan finishes, the arm returns home. This trusts
successful SDK commands without visually verifying grasp or placement.

Every invocation prints its `RUN DIRECTORY` under `real_robot/experiments/pipeline/`.
The initial plan and overlay are in `observations/0000/planning/`; run statistics
are in `summary.md` and `summary.json`.

# Learned four-Protex perception and abstract planning

The historical perception-only path keeps the Isaac pipeline intact and sends **no commands to the xArm**. The read-only path is D435i aligned RGB-D capture → official pretrained SAM 2.1 prompted instance masks → official NVIDIA FoundationPose multi-object 6D registration → supplied `link_base → camera_color_optical_frame` transform → existing cuboid MILP solver.

The documented four-box run and review figures are in [experiments/protex_4_20260914/README.md](experiments/protex_4_20260914/README.md). The rearranged 2 × 2 scene is recorded separately in [experiments/protex_4_20260914_183700/README.md](experiments/protex_4_20260914_183700/README.md).

## Completed snapshot

- `captures/protex_4_complete_20260914.npz`: aligned color/depth, camera intrinsics and depth scale.
- `protex_4_points.json`: one visible point per physical instance. SAM 2.1 large produced four distinct masks in `output/protex_4_prompted/`.
- `models/protex_90x60x28mm.obj`: centred, measured-size cuboid CAD mesh (metres); `models/foundationpose/`: official public RefineNet and ScoreNet ONNX weights.
- `output/protex_4_foundationpose_refined/foundationpose_instances.json`: four pretrained 6D camera-frame poses, registered together with 128 hypotheses and 5 refinement iterations. `output/protex_4_foundationpose/model_overlay.png` shows the first registration projected on RGB; the refined base-frame poses are within a few millimetres of those shown.
- `output/protex_4_plan/layout.png` and `layout_and_plan.json`: base-frame top-down rectangles and a verified abstract plan. The MILP status is optimal with zero Clearances.

The full 90 × 60 mm estimated footprints overlap at two near contacts. The saved run keeps all physical dimensions unchanged and uses bounded XY centre-pose corrections (maximum 1.15 mm) to reconcile the independently estimated poses before solving. Both raw and corrected poses are saved. See the [experiment record](experiments/protex_4_20260914/README.md) for the exact corrections and plots. The weakest FoundationPose raw score is `-0.116` for `protex_3`; scores are logits, not calibrated probabilities. Its projected silhouette has IoU 0.875 with its SAM mask. Mask IoUs for instances 1–4 are 0.732, 0.828, 0.875, 0.873.

The static transform is copied from the provided ROS `static_transform_publisher` quaternion in XYZW order and tested in `pose_to_plan.py`. The optional optical-to-`camera_link` transform is unnecessary for pose conversion because FoundationPose uses the color optical frame. The supplied static transform has been **applied, not independently recalibrated or physically checked**. Names `protex_1`–`protex_4` correspond to the four prompted masks in this snapshot; persistent names after boxes move need multi-frame instance tracking.

## Reproduce

The project-local `real_robot/.venv-pose` contains CUDA 12.8 PyTorch, official SAM 2 and NVIDIA Python bindings. The official FoundationPose runtime was built through Docker at `external/foundation-pose-inference-library/build/libfoundation_pose_nvidia.so`. Model assets, source checkouts, captures, environments and generated results are ignored by Git.

Run learned masks again if a new scene is captured. Points in `protex_4_points.json` must be updated to one visible pixel inside each box:

```bash
.venv-real/bin/python real_robot/capture_rgbd.py --out real_robot/captures/new_scene.npz
real_robot/.venv-pose/bin/python real_robot/run_sam2_prompted.py \
  --capture real_robot/captures/new_scene.npz \
  --points real_robot/protex_4_points.json \
  --checkpoint real_robot/models/sam2.1_hiera_large.pt \
  --out real_robot/output/new_masks
```

The FoundationPose `.so` is linked against CUDA 13/TensorRT in NVIDIA's container. Run registration inside that image, then transform and solve on the host:

```bash
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/project/real_robot/external/foundation-pose-inference-library/python/src \
  -e LD_LIBRARY_PATH=/project/real_robot/external/foundation-pose-inference-library/build:/usr/lib/x86_64-linux-gnu \
  -v "$PWD:/project" -w /project nvcr.io/nvidia/pytorch:26.05-py3 \
  python real_robot/run_foundationpose.py \
  --capture real_robot/captures/new_scene.npz \
  --masks real_robot/output/new_masks/instance_masks.npz \
  --mesh real_robot/models/protex_90x60x28mm.obj \
  --library real_robot/external/foundation-pose-inference-library/build/libfoundation_pose_nvidia.so \
  --refiner real_robot/models/foundationpose/refiner_net.onnx \
  --scorer real_robot/models/foundationpose/score_net.onnx \
  --n-hypotheses 128 --n-refine 5 \
  --out real_robot/output/new_foundationpose
.venv-real/bin/python real_robot/pose_to_plan.py \
  --poses real_robot/output/new_foundationpose/foundationpose_instances.json \
  --expected-protex 4 --max-pose-correction-mm 2 \
  --out real_robot/output/new_plan
```

The planner's Pick witnesses are abstract. Gripper command mapping, reachability, collision checks, TCP calibration, and the safety review still precede physical execution.

## Real xArm execution preparation

Startup automatically generates fresh SAM2 seeds from the initial RGB-D capture for the four visible Protex boxes, so rearranging their initial layout does not require manual pixel edits. It estimates all four poses for one fixed plan. After Knock, observations estimate only its target. A successful Pick drop and retreat immediately retires the cuboid and tracking label; the next capture estimates only the next planned target. This trusts SDK completion without visual placement confirmation. After the final Pick, the arm returns home without another capture. Remaining source masks are matched one-to-one against initial/last valid positions, allowing the commanded motion for a Knock target. Other boxes retain their last accepted poses for grasp clearance. Target pose failures retain one recovery attempt. Knock travel is 80 mm at 4 mm/s with 1 mm total descent. See [target-scoped observations](RUN_PIPELINE.md#target-scoped-observations) for the cached-pose assumption and artifacts.

Use [RUN_PIPELINE.md](RUN_PIPELINE.md) for the current single-command physical executor, calculated fingertip contact heights, and configured TCP release height. The previous standalone executor and [EXECUTION_SAFETY_PLAN.md](EXECUTION_SAFETY_PLAN.md) are historical records; their approval/calibration flags are not used by the current pipeline.
