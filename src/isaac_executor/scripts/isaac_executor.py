#!/usr/bin/env python3
"""Execute a kp_planner_opt YAML plan entirely inside Isaac Sim.

The planner owns action selection and ordering. This program only realizes each
high-level Move or Knock with Isaac's Franka controllers and simulated contact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import sys
import threading
import time
import random

import numpy as np

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", help="High-level plan YAML")
    parser.add_argument("--batch-manifest",
                        help="JSON task manifest for synchronized in-process execution")
    parser.add_argument("--parallel-workspaces", type=int, default=0,
                        help="Opt-in number of Panda workspaces in one Isaac World")
    parser.add_argument("--workspace-spacing", type=float, default=2.5,
                        help="Center spacing in metres for in-process workspaces")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--physics-dt", type=float, default=1.0 / 120.0)
    parser.add_argument("--settle-steps", type=int, default=45)
    parser.add_argument("--action-timeout", type=int, default=2400)
    parser.add_argument("--position-tolerance", type=float, default=0.012)
    parser.add_argument("--no-wait", action="store_true",
                        help="Start immediately instead of waiting for a key press")
    parser.add_argument("--knock-contact-steps", type=int, default=96,
                        help="Number of waypoints in the contact stroke")
    parser.add_argument("--knock-contact-frames", type=int, default=5,
                        help="Physics frames spent at each contact waypoint")
    parser.add_argument("--knock-precontact-clearance", type=float, default=0.0,
                        help="Tool-frame height above the cuboid top before the stroke")
    parser.add_argument("--knock-angle-deg", type=float, default=8.0,
                        help="Downward angle of the outward top-edge stroke")
    parser.add_argument("--drive-stiffness-scale", type=float, default=0.85)
    parser.add_argument("--drive-damping-scale", type=float, default=1.50)
    parser.add_argument("--contact-stiffness-scale", type=float, default=0.80,
                        help="Arm-drive stiffness multiplier during loaded sliding")
    parser.add_argument("--contact-damping-scale", type=float, default=1.50,
                        help="Arm-drive damping multiplier during loaded sliding")
    parser.add_argument("--max-clearance-attempts", type=int, default=None,
                        help="Override the problem's retry limit (default: at least 10)")
    parser.add_argument("--max-move-attempts", type=int, default=5,
                        help="Maximum physical retries for a failed Move action")
    parser.add_argument("--panda-finger-clearance", type=float, default=0.022,
                        help="Physical finger thickness/corridor allowance used when fully open")
    parser.add_argument("--metrics-json", help="Write detailed execution metrics to this path")
    parser.add_argument("--policy", choices=("planned", "random-push"),
                        default="planned",
                        help="Execute the YAML plan or the closed-loop random-push baseline")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the random-push policy")
    parser.add_argument("--push-min-distance", type=float, default=0.04)
    parser.add_argument("--push-max-distance", type=float, default=0.08)
    parser.add_argument("--max-random-pushes", type=int, default=50)
    return parser.parse_args()


ARGS = parse_args()
if bool(ARGS.yaml) == bool(ARGS.batch_manifest):
    raise SystemExit("provide exactly one of --yaml or --batch-manifest")
if ARGS.parallel_workspaces < 0:
    raise SystemExit("--parallel-workspaces cannot be negative")
if ARGS.batch_manifest and ARGS.parallel_workspaces < 1:
    raise SystemExit("--batch-manifest requires --parallel-workspaces >= 1")
if not 0.0 < ARGS.knock_angle_deg < 60.0:
    raise SystemExit("--knock-angle-deg must be between 0 and 60 degrees")
if ARGS.max_move_attempts < 1:
    raise SystemExit("--max-move-attempts must be positive")
if not 0.0 < ARGS.push_min_distance <= ARGS.push_max_distance:
    raise SystemExit("push distances must satisfy 0 < min <= max")
APP = SimulationApp({"headless": ARGS.headless, "renderer": "RaytracedLighting"})

# Imports below this point require a live SimulationApp.
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.utils import stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.rotations import (
    euler_angles_to_quat, quat_to_rot_matrix, rot_matrix_to_quat,
)
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import (
    RMPFlowController,
)
import omni.usd
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problem import knock_direction_to_isaac, load_problem


SCENE_OFFSET = np.array([0.0, 0.1, 0.01459])
ROOM_USD = "/Isaac/Environments/Simple_Room/simple_room.usd"
TOP_DOWN_QUAT_WXYZ = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))


def wait_for_keypress(world):
    """Wait for a key while continuing to pump Isaac physics and UI events."""
    print("\n[READY] Isaac scene is settled. Press any key to execute the plan...", flush=True)
    if sys.stdin.isatty():
        import termios
        import tty

        fd = sys.stdin.fileno()
        previous = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while APP.is_running():
                readable, _, _ = select.select([sys.stdin], [], [], 0.0)
                if readable:
                    os.read(fd, 1)
                    break
                world.step(render=True)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, previous)
        print(flush=True)
    else:
        # Pipes do not support single-key reads. Read a line on a helper thread
        # while the main thread remains responsible for Isaac stepping.
        pressed = threading.Event()

        def read_line():
            try:
                sys.stdin.readline()
            finally:
                pressed.set()

        threading.Thread(target=read_line, daemon=True).start()
        while APP.is_running() and not pressed.is_set():
            world.step(render=True)


def tune_franka_drives(root="/World/Franka"):
    """Add passive damping/armature and soften drives for stable contact."""
    usd_stage = omni.usd.get_context().get_stage()
    joints = {
        "panda_joint1": ("panda_link0", 0.10, 0.50),
        "panda_joint2": ("panda_link1", 0.10, 0.50),
        "panda_joint3": ("panda_link2", 0.10, 0.50),
        "panda_joint4": ("panda_link3", 0.10, 0.50),
        "panda_joint5": ("panda_link4", 0.05, 0.20),
        "panda_joint6": ("panda_link5", 0.05, 0.20),
        "panda_joint7": ("panda_link6", 0.05, 0.20),
    }
    for joint_name, (parent, armature, friction) in joints.items():
        prim = usd_stage.GetPrimAtPath(f"{root}/{parent}/{joint_name}")
        if not prim.IsValid():
            print(f"[TUNE] joint prim not found: {joint_name}")
            continue
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)
        physx_joint.CreateArmatureAttr(float(armature))
        physx_joint.CreateJointFrictionAttr(float(friction))
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            stiffness_attr = drive.GetStiffnessAttr()
            damping_attr = drive.GetDampingAttr()
            stiffness = stiffness_attr.Get()
            damping = damping_attr.Get()
            if stiffness is not None:
                stiffness_attr.Set(float(stiffness) * ARGS.drive_stiffness_scale)
            if damping is not None:
                damping_attr.Set(float(damping) * ARGS.drive_damping_scale)

def bind_finger_physics_material(material, root="/World/Franka"):
    """Bind a rubber-like contact material to both physical finger links."""
    usd_stage = omni.usd.get_context().get_stage()
    bound = []
    for prim in usd_stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root):
            continue
        if "panda_leftfinger" not in path and "panda_rightfinger" not in path:
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material.material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
        bound.append(path)
    print(
        f"[TUNE] high-friction finger material bound to {len(bound)} prims",
        flush=True,
    )


def scale_franka_arm_drives(stiffness_factor=1.0, damping_factor=1.0,
                            root="/World/Franka"):
    """Scale live arm drives and return values for exact restoration."""
    usd_stage = omni.usd.get_context().get_stage()
    saved = []
    for joint_index in range(1, 8):
        parent_index = joint_index - 1
        prim = usd_stage.GetPrimAtPath(
            f"{root}/panda_link{parent_index}/panda_joint{joint_index}"
        )
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            continue
        stiffness_attr = drive.GetStiffnessAttr()
        damping_attr = drive.GetDampingAttr()
        stiffness = stiffness_attr.Get()
        damping = damping_attr.Get()
        saved.append((stiffness_attr, stiffness, damping_attr, damping))
        if stiffness is not None:
            stiffness_attr.Set(float(stiffness) * float(stiffness_factor))
        if damping is not None:
            damping_attr.Set(float(damping) * float(damping_factor))
    return saved


def restore_franka_arm_drives(saved):
    for stiffness_attr, stiffness, damping_attr, damping in saved:
        if stiffness is not None:
            stiffness_attr.Set(stiffness)
        if damping is not None:
            damping_attr.Set(damping)


def tune_cuboid_rigid_body(prim_path):
    """Dissipate impact energy without moving or constraining the body."""
    prim = omni.usd.get_context().get_stage().GetPrimAtPath(prim_path)
    rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid.CreateLinearDampingAttr(1.5)
    rigid.CreateAngularDampingAttr(3.0)
    rigid.CreateMaxDepenetrationVelocityAttr(0.20)


def plan_xy_to_isaac(xy, scene_offset=SCENE_OFFSET):
    """Apply the repository's ROS-plan to Isaac stage transform."""
    x, y = xy
    scene_offset = np.asarray(scene_offset, dtype=float)
    return np.array([-y + scene_offset[0], x + scene_offset[1]], dtype=float)


def plan_dir_to_isaac(direction):
    """Convert the YAML's ROS direction using the scene's Rz(+90) rotation."""
    return np.asarray(knock_direction_to_isaac(direction), dtype=float)


def oriented_box_extent(dimensions, orientation, world_axis):
    """Full projection of an oriented cuboid on a normalized world axis."""
    rotation = quat_to_rot_matrix(np.asarray(orientation, dtype=float))
    axis = np.asarray(world_axis, dtype=float)
    axis /= max(np.linalg.norm(axis), 1e-9)
    return float(np.dot(
        np.asarray(dimensions, dtype=float),
        np.abs(rotation.T @ axis),
    ))


def live_horizontal_body_axes(orientation, minimum_planar_norm=0.80):
    """Return reliable horizontal axes of the cuboid's current supporting face.

    After a clearance action a cuboid may have rolled 90 degrees. Its original
    X or Y axis can then point vertically; normalizing that axis's tiny XY
    numerical residue produces an arbitrary grasp yaw. Consider all three live
    body axes and discard axes that are currently near vertical.
    """
    rotation = quat_to_rot_matrix(np.asarray(orientation, dtype=float))
    axes = []
    for local_index in range(3):
        world_axis = np.asarray(rotation[:, local_index], dtype=float)
        planar_norm = float(np.linalg.norm(world_axis[:2]))
        if planar_norm < minimum_planar_norm:
            continue
        horizontal = np.array([
            world_axis[0] / planar_norm,
            world_axis[1] / planar_norm,
            0.0,
        ])
        axes.append((local_index, horizontal, planar_norm))
    if len(axes) < 2:
        raise RuntimeError(
            "cuboid has no stable horizontal supporting-face axes: "
            f"orientation={np.asarray(orientation).tolist()}"
        )
    return axes


def knock_orientation(direction):
    """Make the tool point down with the visible palm parallel to the stroke."""
    palm_y = np.asarray(direction, dtype=float)
    palm_y /= np.linalg.norm(palm_y)
    # A palm axis is a line, not an arrow: +axis and -axis are equally
    # parallel to the stroke. Choose the branch that has proved reachable in
    # this shifted Panda workspace: -X for X strokes and +Y for Y strokes.
    # In particular, +Y avoids the equivalent 180-about-X wrist branch that
    # RMPflow cannot lower reliably at the far side of the table.
    if abs(palm_y[0]) >= abs(palm_y[1]):
        if palm_y[0] > 0.0:
            palm_y = -palm_y
    elif palm_y[1] < 0.0:
        palm_y = -palm_y
    tool_z = np.array([0.0, 0.0, -1.0])
    palm_x = np.cross(palm_y, tool_z)
    # Matrix columns are right_gripper's local axes expressed in world space.
    rotation = np.column_stack((palm_x, palm_y, tool_z))
    return rot_matrix_to_quat(rotation)


def grasp_orientation(closing_axis):
    """Top-down wrist pose whose finger-closing line matches a ROS plan axis."""
    vector = plan_dir_to_isaac("+" + closing_axis)
    return knock_orientation(vector)


def equivalent_grasp_orientation(orientation):
    """Swap fingers with a 180-degree tool-Z turn, preserving the grasp line."""
    rotation = quat_to_rot_matrix(np.asarray(orientation, dtype=float))
    rotation[:, 0] *= -1.0
    rotation[:, 1] *= -1.0
    return rot_matrix_to_quat(rotation)


class IsaacPlanExecutor:
    def __init__(self, world, robot, problem, *, scene_offset=SCENE_OFFSET,
                 object_name_prefix="", robot_root="/World/Franka",
                 policy=None, seed=None):
        self.world = world
        self.robot = robot
        self.problem = problem
        self.scene_offset = np.asarray(scene_offset, dtype=float)
        self.object_name_prefix = str(object_name_prefix)
        self.robot_root = str(robot_root)
        self.policy = policy or ARGS.policy
        self.seed = ARGS.seed if seed is None else int(seed)
        self.cartesian_controller = RMPFlowController(
            name="native_knock_" + self.robot_root.replace("/", "_"),
            robot_articulation=robot,
        )
        self.knock_active_joints = (
            self.cartesian_controller.articulation_rmp.get_active_joints_subset()
        )
        self.goal_stack_heights = {}
        self.last_clearance_orientation = None
        self.clearance_directions = {}
        self.home_joint_positions = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        usd_stage = omni.usd.get_context().get_stage()
        self.left_finger_prim = next(
            (prim for prim in usd_stage.Traverse()
             if str(prim.GetPath()).startswith(self.robot_root)
             and prim.GetName() == "panda_leftfinger"), None
        )
        self.right_finger_prim = next(
            (prim for prim in usd_stage.Traverse()
            if str(prim.GetPath()).startswith(self.robot_root)
            and prim.GetName() == "panda_rightfinger"), None
        )
        if self.left_finger_prim is None or self.right_finger_prim is None:
            raise RuntimeError("could not locate Panda finger link prims")
        self.metrics = {
            "status": "running", "actions_planned": len(problem.actions),
            "actions_completed": 0, "objects_removed": 0,
            "clearance_actions": 0, "clearance_attempts": 0,
            "clearance_retries": 0, "failed_move_actions": 0,
            "failed_clearance_actions": 0, "skipped_moves": 0,
            "failed_picks": 0, "failed_knocks": 0,
            "policy": self.policy, "seed": self.seed,
            "push_actions": 0, "push_attempts": 0,
            "successful_pushes": 0, "failed_pushes": 0,
            "pick_attempts": 0, "successful_picks": 0,
            "deadlocks": 0, "live_replans": 0,
            "live_replan_time_sec": 0.0,
            "action_records": [],
        }

    def get_object(self, name):
        return self.world.scene.get_object(self.object_name_prefix + str(name))

    def plan_xy(self, xy):
        return plan_xy_to_isaac(xy, self.scene_offset)

    def clearance_direction(self, action, name):
        """Resolve a legacy world or new object-local ROS direction in Isaac."""
        base = plan_dir_to_isaac(str(action["dir"]))
        if action.get("direction_frame", "world") != "object_local":
            return base
        _, orientation = self.get_object(str(name)).get_world_pose()
        direction = quat_to_rot_matrix(np.asarray(orientation, dtype=float)) @ base
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 0.50:
            raise RuntimeError(
                f"{name}: selected local clearance axis is no longer horizontal"
            )
        return direction / norm
    def step_world(self, render=True):
        self.world.step(render=render)

    def choose_disposal_destination(self, name, source, height):
        """Choose a live obstacle-away disposal point in an allowed ROS direction."""
        allowed = ("+x", "+y", "-y")
        others = []
        for other_name in self.problem.cuboids:
            if str(other_name) == str(name):
                continue
            position, _ = self.get_object(str(other_name)).get_world_pose()
            position = np.asarray(position, dtype=float)
            if position[2] > -0.5 and position[0] < 4.0:
                others.append(position[:2])
        source_xy = np.asarray(source[:2], dtype=float)
        workspace_delta = self.scene_offset[:2] - SCENE_OFFSET[:2]
        candidates = []
        for ros_direction in allowed:
            direction = plan_dir_to_isaac(ros_direction)[:2]
            if others:
                furthest = max(float(np.dot(p - source_xy, direction))
                               for p in others)
            else:
                furthest = 0.0
            travel = max(0.10, furthest + 0.10)
            candidate = source_xy + travel * direction
            # Stay on the usable tabletop/reachable band while remaining well
            # outside the compact 25 cm source clump.
            candidate[0] = np.clip(candidate[0], -0.22 + workspace_delta[0],
                                   0.22 + workspace_delta[0])
            candidate[1] = np.clip(candidate[1], 0.38 + workspace_delta[1],
                                   0.76 + workspace_delta[1])
            separation = (min(np.linalg.norm(candidate - p) for p in others)
                          if others else 1.0)
            achieved_outward = float(np.dot(candidate - source_xy, direction))
            candidates.append((separation, achieved_outward, ros_direction,
                               candidate))
        _, _, selected_direction, selected = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        destination = np.array([selected[0], selected[1], height], dtype=float)
        print(
            f"[DISPOSAL] {name}: selected_ros_direction={selected_direction}; "
            f"source={source_xy.round(5).tolist()}; "
            f"destination={selected.round(5).tolist()}; candidates="
            f"{[(d, round(s, 4), c.round(4).tolist()) for s, _, d, c in candidates]}",
            flush=True,
        )
        return destination

    def joint_limit_diagnostics(self):
        """Capture joint-limit proximity without claiming it caused failure."""
        try:
            positions = np.asarray(self.robot.get_joint_positions(), dtype=float)
            limits = np.asarray(self.robot.get_dof_limits(), dtype=float)
            margins = np.minimum(positions - limits[:, 0], limits[:, 1] - positions)
            names = list(self.robot.dof_names)
            near = [names[i] for i, margin in enumerate(margins[:7])
                    if margin < 0.05]
            return {
                "minimum_arm_joint_margin": float(np.min(margins[:7])),
                "near_limit_arm_joints": near,
                "arm_joint_positions": positions[:7].tolist(),
            }
        except Exception as exc:
            return {"diagnostic_error": str(exc)}

    @staticmethod
    def classify_move_failure(error):
        message = str(error)
        if ("physical grasp failed" in message
                or "physical grasp slip" in message
                or "pushed out of the grasp" in message):
            return "physical_grasp_slip"
        if "planned grasp is wider" in message or "opening is insufficient" in message:
            return "grasp_aperture_infeasible"
        if ("Cartesian trajectory missed" in message
                or "Cartesian target timed out" in message):
            return "kinematic_or_controller_reachability"
        if "without opposing contact" in message or "without enclosing" in message:
            return "grasp_contact_failure"
        if "open-finger descent displaced" in message:
            return "pregrasp_object_disturbance"
        return "other_move_failure"

    def settle(self):
        for _ in range(ARGS.settle_steps):
            self.step_world(render=not ARGS.headless)

    @staticmethod
    def prim_world_position(prim):
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        return np.asarray(transform.ExtractTranslation(), dtype=float)

    def finger_pad_geometry(self, orientation):
        """Return pad-center midpoint, closing axis, and physical aperture.

        Panda finger link frames are at the pad bases.  The usable lower
        contact band is approximately 48 mm down the 58 mm fingers, along the
        tool's local +Z direction (world-down for every top grasp used here).
        """
        left = self.prim_world_position(self.left_finger_prim)
        right = self.prim_world_position(self.right_finger_prim)
        separation = left - right
        aperture = float(np.linalg.norm(separation))
        closing_axis = separation / max(aperture, 1e-9)
        tool_z = quat_to_rot_matrix(np.asarray(orientation, dtype=float))[:, 2]
        midpoint = 0.5 * (left + right) + 0.048 * tool_z
        return midpoint, closing_axis, aperture, left, right

    def recover_home(self, frames=240):
        """Smoothly return to the initialized joint posture between actions."""
        start = np.asarray(self.robot.get_joint_positions(), dtype=float)
        goal = self.home_joint_positions.copy()
        # Preserve the current gripper command; it is opened explicitly after
        # the arm has cleared the contact scene.
        goal[-2:] = start[-2:]
        for alpha in np.linspace(0.0, 1.0, frames):
            progress = alpha * alpha * (3.0 - 2.0 * alpha)
            target = (1.0 - progress) * start + progress * goal
            self.robot.apply_action(ArticulationAction(joint_positions=target))
            self.step_world(render=not ARGS.headless)
        self.cartesian_controller.reset()

    def move_ee(self, target, tolerance=None, orientation=TOP_DOWN_QUAT_WXYZ,
                max_frames=None):
        """Move RMPflow's controlled ``right_gripper`` frame to a waypoint."""
        tolerance = tolerance or ARGS.position_tolerance
        frame_limit = ARGS.action_timeout if max_frames is None else int(max_frames)
        for _ in range(frame_limit):
            action = self.cartesian_controller.forward(
                target_end_effector_position=np.asarray(target),
                target_end_effector_orientation=orientation,
            )
            self.robot.apply_action(action)
            self.step_world(render=not ARGS.headless)
            # Franka.end_effector is the panda_rightfinger USD prim, whereas
            # RMPflow controls the URDF frame named right_gripper. They are
            # separated by a fixed offset, so checking the finger prim can
            # never converge to the RMPflow target.
            position, _ = self.cartesian_controller.rmp_flow.get_end_effector_pose(
                self.knock_active_joints.get_joint_positions()
            )
            if np.linalg.norm(np.asarray(position) - target) <= tolerance:
                return
        finger_position, _ = self.robot.end_effector.get_world_pose()
        raise RuntimeError(
            "Cartesian target timed out: "
            f"frames={frame_limit}, "
            f"target={target.tolist()}, controlled_frame={np.asarray(position).tolist()}, "
            f"right_finger={np.asarray(finger_position).tolist()}"
        )

    def command_ee_unchecked(self, target, orientation, frames=180):
        """Command a bounded trajectory segment without a convergence gate."""
        target = np.asarray(target, dtype=float)
        for _ in range(frames):
            action = self.cartesian_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation,
            )
            self.robot.apply_action(action)
            self.step_world(render=not ARGS.headless)

    def command_linear_ee(self, target, orientation, steps=120,
                          frames_per_step=2):
        """Track a smooth bounded Cartesian line without a long timeout."""
        start = self.controlled_ee_position().copy()
        target = np.asarray(target, dtype=float)
        for alpha in np.linspace(0.0, 1.0, steps):
            progress = alpha * alpha * (3.0 - 2.0 * alpha)
            waypoint = (1.0 - progress) * start + progress * target
            for _ in range(frames_per_step):
                action = self.cartesian_controller.forward(
                    target_end_effector_position=waypoint,
                    target_end_effector_orientation=orientation,
                )
                self.robot.apply_action(action)
                self.step_world(render=not ARGS.headless)
        return self.controlled_ee_position().copy()

    def command_linear_ee_checked(self, target, orientation, steps=120,
                                  frames_per_step=2, tolerance=0.030):
        achieved = self.command_linear_ee(
            target, orientation, steps, frames_per_step
        )
        error = float(np.linalg.norm(achieved - np.asarray(target, dtype=float)))
        if error > tolerance:
            raise RuntimeError(
                "bounded Cartesian trajectory missed its target: "
                f"target={np.asarray(target).tolist()}, "
                f"achieved={achieved.tolist()}, error={error:.4f} m"
            )
        return achieved

    def slow_vertical_contact_retreat(self, orientation, distance=0.10,
                                      steps=120, frames_per_step=2):
        """Unload contact through a slow vertical path before moving laterally."""
        end = self.controlled_ee_position().copy()
        end[2] += distance
        self.robot.gripper.close()
        self.command_linear_ee(end, orientation, steps, frames_per_step)

    def descend_to_clearance_contact(self, target, orientation, object_prim,
                                     cuboid, tolerance=0.006,
                                     max_frames=180):
        """Move vertically until the real finger band reaches the live top."""
        target = np.asarray(target, dtype=float)
        vertical_history = []
        for frame in range(max_frames):
            object_position, object_orientation = object_prim.get_world_pose()
            half_extents = np.abs(quat_to_rot_matrix(
                np.asarray(object_orientation, dtype=float)
            )) @ (0.5 * np.asarray(cuboid.dimensions, dtype=float))
            live_top_z = float(np.asarray(object_position)[2] + half_extents[2])
            pad_midpoint, _, _, _, _ = self.finger_pad_geometry(orientation)
            pad_clearance = float(pad_midpoint[2] - live_top_z)
            if pad_clearance <= 0.0015:
                return True, pad_clearance, frame

            action = self.cartesian_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation,
            )
            self.robot.apply_action(action)
            self.step_world(render=not ARGS.headless)
            position = self.controlled_ee_position()
            vertical_history.append(float(position[2]))
            # The mesh's usable contact band is several millimetres above the
            # nominal pad sample for this rotated palm branch. Detect the
            # resulting loaded-controller plateau as contact as well: once a
            # low tool has made <0.5 mm progress for 45 physics frames, further
            # downward command only preloads the cuboid and cannot be an IK
            # approach failure.
            if (len(vertical_history) >= 45
                    and position[2] <= live_top_z + 0.025
                    and pad_clearance <= 0.015
                    and abs(vertical_history[-1] - vertical_history[-45])
                    <= 0.0005):
                return True, pad_clearance, frame + 1
            if np.linalg.norm(position - target) <= tolerance:
                return False, pad_clearance, frame + 1
        position = self.controlled_ee_position()
        raise RuntimeError(
            "bounded vertical pre-contact motion did not converge or touch: "
            f"target={target.tolist()}, controlled_frame={position.tolist()}"
        )

    def follow_contact_stroke(self, name, object_prim, start, end, orientation,
                              object_start=None, direction=None,
                              stop_displacement=None):
        """Track a slow, smooth diagonal stroke and report its progress."""
        report_indices = {
            0,
            ARGS.knock_contact_steps // 4,
            ARGS.knock_contact_steps // 2,
            3 * ARGS.knock_contact_steps // 4,
            ARGS.knock_contact_steps - 1,
        }
        for index, alpha in enumerate(np.linspace(0.0, 1.0, ARGS.knock_contact_steps)):
            progress = alpha * alpha * (3.0 - 2.0 * alpha)
            target = start.copy()
            target[:2] = (1.0 - progress) * start[:2] + progress * end[:2]
            # Contact was established by the vertical approach. Interpolate
            # the shallow drop with the same smooth progress as XY: the motion
            # remains outward-dominant without postponing all contact loading
            # until the tool has already tried to leave the top surface.
            vertical_progress = progress
            target[2] = ((1.0 - vertical_progress) * start[2]
                         + vertical_progress * end[2])
            for _ in range(ARGS.knock_contact_frames):
                controller_action = self.cartesian_controller.forward(
                    target_end_effector_position=target,
                    target_end_effector_orientation=orientation,
                )
                self.robot.apply_action(controller_action)
                self.step_world(render=not ARGS.headless)
            if stop_displacement is not None:
                object_now, _ = object_prim.get_world_pose()
                achieved_displacement = float(np.dot(
                    np.asarray(object_now) - np.asarray(object_start), direction
                ))
                if achieved_displacement >= stop_displacement:
                    print(
                        f"[KNOCK EARLY STOP] {name}: "
                        f"measured_displacement={achieved_displacement:.4f}; "
                        f"remaining_required={stop_displacement:.4f}; "
                        f"stroke_progress={100.0 * alpha:.1f}%",
                        flush=True,
                    )
                    return
            if index in report_indices:
                self.log_knock_stage(
                    name, f"contact stroke {100.0 * alpha:.0f}%", object_prim,
                    target,
                )

    def orient_ee_at_clearance(self, target, orientation, frames=120):
        """Rotate the wrist at a safe height before descending near objects."""
        for _ in range(frames):
            controller_action = self.cartesian_controller.forward(
                target_end_effector_position=target,
                target_end_effector_orientation=orientation,
            )
            self.robot.apply_action(controller_action)
            self.step_world(render=not ARGS.headless)
        achieved = self.controlled_ee_position()
        error = float(np.linalg.norm(achieved - np.asarray(target)))
        if error > 0.040:
            raise RuntimeError(
                "Palm alignment left the clearance waypoint: "
                f"target={np.asarray(target).tolist()}, "
                f"controlled_frame={achieved.tolist()}, error={error:.4f} m"
            )

    def controlled_ee_position(self):
        position, _ = self.cartesian_controller.rmp_flow.get_end_effector_pose(
            self.knock_active_joints.get_joint_positions()
        )
        return np.asarray(position, dtype=float)

    def log_knock_stage(self, name, stage_name, object_prim, target=None):
        """Print commanded and measured state at a knock stage boundary."""
        ee_position = self.controlled_ee_position()
        object_position, object_orientation = object_prim.get_world_pose()
        target_text = "none" if target is None else np.asarray(target).round(5).tolist()
        print(
            f"[KNOCK STAGE] {name}: {stage_name}; target={target_text}; "
            f"controlled_frame={ee_position.round(5).tolist()}; "
            f"object_position={np.asarray(object_position).round(5).tolist()}; "
            f"object_quaternion={np.asarray(object_orientation).round(5).tolist()}",
            flush=True,
        )

    def log_move_stage(self, name, stage_name, object_prim, target=None):
        """Report the live object, controller frame, and gripper state."""
        ee_position = self.controlled_ee_position()
        object_position, object_orientation = object_prim.get_world_pose()
        target_text = "none" if target is None else np.asarray(target).round(5).tolist()
        try:
            finger_joints = np.asarray(
                self.robot.gripper.get_joint_positions(), dtype=float
            ).round(5).tolist()
        except Exception:
            finger_joints = "unavailable"
        print(
            f"[MOVE STAGE] {name}: {stage_name}; target={target_text}; "
            f"controlled_frame={ee_position.round(5).tolist()}; "
            f"object_position={np.asarray(object_position).round(5).tolist()}; "
            f"object_quaternion={np.asarray(object_orientation).round(5).tolist()}; "
            f"finger_joints={finger_joints}",
            flush=True,
        )

    def execute_move(self, action):
        name = str(action["block"])
        cuboid = self.problem.cuboids[name]
        object_prim = self.get_object(name)
        source, source_orientation = object_prim.get_world_pose()
        source = np.asarray(source, dtype=float)
        # The live rigid-body center supplies translation and, specifically,
        # the height center. Direct Picks may add a noncentral planar witness.
        destination_xy = self.plan_xy(self.problem.pads[int(action["to"]["pad"])])
        destination_pad = int(action["to"]["pad"])
        support_height = self.goal_stack_heights.get(destination_pad, 0.0)
        destination = np.array([
            destination_xy[0], destination_xy[1],
            self.scene_offset[2] + support_height + 0.5 * cuboid.height,
        ])
        if action.get("remove_after_place", False):
            destination = self.choose_disposal_destination(
                name, source, destination[2]
            )

        grasp_spec = action.get("grasp")
        grasp_offset = np.zeros(3)
        move_orientation = TOP_DOWN_QUAT_WXYZ
        source_rotation = quat_to_rot_matrix(
            np.asarray(source_orientation, dtype=float)
        )
        if grasp_spec:
            ros_offset = np.asarray(grasp_spec["offset"], dtype=float)
            if "finger_window" not in grasp_spec:
                raise RuntimeError(
                    f"Move {name} uses a legacy grasp without a physical "
                    "finger window; regenerate the Isaac YAML so a boundary "
                    "witness cannot leave part of the finger in air"
                )
            window = np.asarray(grasp_spec["finger_window"], dtype=float)
            window_width = float(window[1] - window[0])
            requested_width = float(
                grasp_spec.get("finger_side_contact_width", window_width)
            )
            tangent_index = 1 if grasp_spec["closing_axis"] == "x" else 0
            window_center = 0.5 * float(window[0] + window[1])
            if (abs(window_width - requested_width) > 1e-6
                    or abs(float(ros_offset[tangent_index]) - window_center) > 1e-6):
                raise RuntimeError(
                    f"Move {name} has an inconsistent physical grasp: "
                    f"offset={ros_offset}, finger_window={window.tolist()}, "
                    f"finger_width={requested_width:.6f}; regenerate the YAML"
                )
            effective_ros_offset = ros_offset.copy()
            inward_nudge = max(0.0, float(grasp_spec.get("inward_nudge", 0.0)))
            base_tangent = float(effective_ros_offset[tangent_index])
            if base_tangent < -1e-9:
                effective_ros_offset[tangent_index] += inward_nudge
            elif base_tangent > 1e-9:
                effective_ros_offset[tangent_index] -= inward_nudge
            # Keep the physical finger interval on the cuboid even if a YAML
            # is manually given a nudge larger than the available side length.
            tangent_dimension_index = 0 if tangent_index == 1 else 1
            tangent_span = float(cuboid.dimensions[tangent_dimension_index])
            center_limit = max(
                0.0,
                0.5 * tangent_span
                - 0.5 * requested_width
                - float(grasp_spec.get("physical_edge_margin", 0.0)),
            )
            effective_ros_offset[tangent_index] = float(np.clip(
                effective_ros_offset[tangent_index], -center_limit, center_limit
            ))
            grasp_offset[:2] = (
                plan_dir_to_isaac("+x")[:2] * float(effective_ros_offset[0])
            )
            grasp_offset[:2] += (
                plan_dir_to_isaac("+y")[:2] * float(effective_ros_offset[1])
            )
            # ``offset`` is the center of the solver's finger window,
            # t + F/2, relative to the footprint center. Preserve that
            # noncentral witness if the dynamic body has yawed in Isaac.
            grasp_offset = source_rotation @ grasp_offset
            grasp_offset[2] = 0.0
            attempt_index = int(action.get("_attempt", 1))
            nominal_closing_world = plan_dir_to_isaac(
                "+" + str(grasp_spec["closing_axis"])
            )
            closing_3d = source_rotation @ nominal_closing_world
            closing_world = closing_3d.copy()
            closing_world[2] = 0.0
            closing_world /= max(np.linalg.norm(closing_world), 1e-9)
            if action.get("after_clearance"):
                # A clearance can roll a thin cuboid onto another face. Keep
                # the exposed-end witness on the actual outgoing corridor and
                # choose a currently horizontal, jaw-feasible body axis for
                # closing instead of projecting a now-vertical planned axis.
                corridor = self.clearance_directions.get(name)
                if corridor is not None and np.linalg.norm(grasp_offset[:2]) > 1e-9:
                    grasp_offset[:2] = (
                        corridor[:2] / np.linalg.norm(corridor[:2])
                        * np.linalg.norm(grasp_offset[:2])
                    )
                axes = live_horizontal_body_axes(source_orientation)
                feasible_axes = [entry for entry in axes
                                 if cuboid.dimensions[entry[0]] <= 0.075]
                if not feasible_axes:
                    raise RuntimeError(
                        "no horizontal supporting-face axis fits the Panda jaw "
                        f"after clearance; dimensions={cuboid.dimensions}"
                    )
                tangent = grasp_offset[:2].copy()
                tangent /= max(np.linalg.norm(tangent), 1e-9)
                _, closing_world, _ = min(
                    feasible_axes,
                    key=lambda entry: abs(float(np.dot(
                        entry[1][:2], tangent
                    ))),
                )
            move_orientation = knock_orientation(closing_world)
            # A parallel-jaw grasp is unchanged when the hand rotates 180
            # degrees about its tool Z axis.  For oblique grasps, one of these
            # two wrist branches can enter an RMPflow/contact local minimum.
            # Alternate only on retries; nominal successful motions and all
            # clearance motions retain their existing controller behavior.
            if attempt_index % 2 == 0:
                move_orientation = equivalent_grasp_orientation(
                    move_orientation
                )
                print(
                    f"[GRASP WRIST BRANCH] {name}: retry={attempt_index}; "
                    "using equivalent finger-swapped orientation",
                    flush=True,
                )
            print(
                f"[GRASP CONTACT WINDOW] {name}: local_interval="
                f"{grasp_spec['finger_window']}; center={window_center:.5f}; "
                f"inward_nudge={inward_nudge:.5f}; effective_offset="
                f"{effective_ros_offset.round(5).tolist()}; "
                f"actual_mesh_width=0.021025; reserved_width="
                f"{requested_width:.5f}; edge_margin="
                f"{grasp_spec.get('physical_edge_margin')}",
                flush=True,
            )
        elif action.get("after_clearance"):
            # The clearance action opens an approach corridor along its travel
            # direction. Use that same world axis as the parallel-jaw closing
            # line. Choosing merely the cuboid's shortest side can put the
            # open fingers across untouched neighbors and sweep the target.
            corridor_world = self.clearance_directions.get(name)
            if corridor_world is None:
                raise RuntimeError(
                    f"Move {name} is marked after_clearance but has no "
                    "preceding clearance direction"
                )
            raise RuntimeError(
                f"post-clearance Move {name} has no planner-derived grasp "
                "witness; regenerate this Isaac problem instead of using a "
                "centered or executor-invented grasp"
            )

        print(
            f"[MOVE] {name}: measured_source={source.round(5).tolist()}; "
            f"source_quaternion={np.asarray(source_orientation).round(5).tolist()}; "
            f"dimensions={np.asarray(cuboid.dimensions).tolist()}; "
            f"destination={destination.round(5).tolist()}; "
            f"scene_offset={self.scene_offset.tolist()}",
            f"grasp={grasp_spec}; grasp_offset_world={grasp_offset.tolist()}",
            flush=True,
        )
        stage_name = "initialization"
        try:
            self.cartesian_controller.reset()
            stage_name = "open gripper"
            self.robot.gripper.open()
            for _ in range(45):
                self.step_world(render=not ARGS.headless)
            self.log_move_stage(name, "gripper open", object_prim)

            # Z is the live cuboid height center. XY is the retained geometric
            # witness g=(o,t), and is generally not the footprint center. A
            # post-clearance Move has no Pick witness and intentionally uses
            # the center of the newly created grasp corridor.
            grasp_target = source.copy() + grasp_offset
            world_half_extents = np.abs(
                quat_to_rot_matrix(np.asarray(source_orientation, dtype=float))
            ) @ (0.5 * np.asarray(cuboid.dimensions, dtype=float))
            closing_world = quat_to_rot_matrix(move_orientation)[:, 1]
            closing_width = oriented_box_extent(
                cuboid.dimensions, source_orientation, closing_world
            )
            if closing_width > 0.075 + 0.0005:
                raise RuntimeError(
                    "planned grasp is wider than the Panda's usable 80 mm "
                    f"opening: closing_width={closing_width:.4f} m"
                )
            vertical_half_extent = float(world_half_extents[2])
            desired_grasp_center = grasp_target.copy()
            # Initial descent target; measured finger-link feedback below
            # performs the final pad-center alignment.
            grasp_target[2] -= min(0.010, 0.40 * vertical_half_extent)
            approach = grasp_target.copy()
            approach[2] = max(0.25, grasp_target[2] + 0.20)

            stage_name = "approach"
            # This waypoint is a collision-free staging pose, not a contact
            # target.  RMPflow can settle 12--15 mm away near workspace-edge
            # wrist branches; the subsequent descent and proximity gate are
            # the checks that determine whether the grasp is valid.
            safe_approach = self.controlled_ee_position().copy()
            safe_approach[2] = max(0.32, safe_approach[2])
            self.command_linear_ee_checked(
                safe_approach, move_orientation, steps=90, tolerance=0.035
            )
            overhead_approach = approach.copy()
            overhead_approach[2] = safe_approach[2]
            self.command_linear_ee_checked(
                overhead_approach, move_orientation, steps=150,
                tolerance=0.035,
            )
            self.command_linear_ee_checked(
                approach, move_orientation, steps=120, tolerance=0.035
            )
            self.log_move_stage(name, "approach complete", object_prim, approach)

            # Keep the Panda fully open during alignment/descent. Clearance
            # actions are sized below for this full 80 mm swept envelope.
            self.robot.gripper.open()
            for _ in range(45):
                self.step_world(render=not ARGS.headless)
            measured_opening = self.finger_pad_geometry(move_orientation)[2]
            print(
                f"[GRASP FULL OPEN] {name}: object_width={closing_width:.4f}; "
                f"measured_aperture={measured_opening:.4f}", flush=True,
            )
            if measured_opening < 0.076:
                raise RuntimeError(
                    "Panda gripper did not reach its full opening before "
                    f"descent: aperture={measured_opening:.4f} m"
                )

            stage_name = "grasp descent"
            alignment_ok = False
            # Do not command a blind center-height descent. Center the open
            # fingers from the collision-free approach pose first, then lower
            # them only by the measured fingertip-overlap deficit.
            for correction_index in range(5):
                # The open fingers may make light contact while descending.
                # Always align to the current rigid-body pose rather than the
                # pose sampled before the approach.
                live_object_position, live_object_orientation = object_prim.get_world_pose()
                live_object_position = np.asarray(live_object_position, dtype=float)
                world_half_extents = np.abs(quat_to_rot_matrix(
                    np.asarray(live_object_orientation, dtype=float)
                )) @ (0.5 * np.asarray(cuboid.dimensions, dtype=float))
                vertical_half_extent = float(world_half_extents[2])
                desired_grasp_center = live_object_position + grasp_offset
                pad_midpoint, physical_closing_axis, open_aperture, left, right = (
                    self.finger_pad_geometry(move_orientation)
                )
                # Use the achieved physical finger axis. Near contact the arm
                # may not exactly attain the requested wrist quaternion.
                closing_width = oriented_box_extent(
                    cuboid.dimensions, live_object_orientation,
                    physical_closing_axis,
                )
                pad_error = desired_grasp_center - pad_midpoint
                tool_z = quat_to_rot_matrix(
                    np.asarray(move_orientation, dtype=float)
                )[:, 2]
                finger_base_midpoint = 0.5 * (left + right)
                fingertip_midpoint = finger_base_midpoint + 0.058 * tool_z
                live_top_z = float(
                    live_object_position[2] + world_half_extents[2]
                )
                top_overlap = live_top_z - float(fingertip_midpoint[2])
                # The usable pad band, rather than merely the fingertip, must
                # be centered at the live cuboid height center.
                required_overlap = min(0.012, 0.30 * (2.0 * vertical_half_extent))
                height_center_error = abs(float(pad_error[2]))
                closing_center_error = abs(float(np.dot(
                    pad_error[:2], physical_closing_axis[:2]
                )))
                closing_component = (
                    np.dot(pad_error[:2], physical_closing_axis[:2])
                    * physical_closing_axis[:2]
                )
                transverse_center_error = float(np.linalg.norm(
                    pad_error[:2] - closing_component
                ))
                closing_margin = max(
                    0.0, 0.5 * (open_aperture - closing_width)
                )
                print(
                    f"[GRASP PAD ALIGN] {name}: pass={correction_index + 1}; "
                    f"desired_center={desired_grasp_center.round(5).tolist()}; "
                    f"pad_midpoint={pad_midpoint.round(5).tolist()}; "
                    f"error={pad_error.round(5).tolist()}; "
                    f"open_aperture={open_aperture:.4f}; "
                    f"closing_center_error={closing_center_error:.4f}; "
                    f"transverse_center_error={transverse_center_error:.4f}; "
                    f"closing_margin={closing_margin:.4f}; "
                    f"fingertip_z={fingertip_midpoint[2]:.4f}; "
                    f"top_overlap={top_overlap:.4f}; "
                    f"required_overlap={required_overlap:.4f}; "
                    f"height_center_error={height_center_error:.4f}; "
                    f"physical_closing_axis="
                    f"{physical_closing_axis.round(5).tolist()}",
                    flush=True,
                )
                if (correction_index > 0 and height_center_error > 0.040
                        and transverse_center_error > 0.015):
                    raise RuntimeError(
                        "grasp wrist branch became unstable during descent: "
                        f"pad_error={pad_error.tolist()}; retrying from home "
                        "with the equivalent parallel-jaw orientation"
                    )
                # When both open-commanded fingers are already touching the
                # opposed faces, PhysX compliance can report 1--3 mm less than
                # the rigid geometric width. This is enclosing contact, not an
                # infeasible jaw opening; larger deficits remain invalid.
                if open_aperture + 0.006 < closing_width:
                    raise RuntimeError(
                        "gripper opening is insufficient for the selected "
                        f"cuboid width: aperture={open_aperture:.4f}, "
                        f"width={closing_width:.4f}"
                    )
                lateral_ok = (
                    closing_center_error <= closing_margin + 0.005
                    and transverse_center_error <= 0.008
                )
                # PhysX/RMPflow settles with sub-millimetre variation at
                # contact. A 1 mm numerical band preserves meaningful finger
                # overlap without repeatedly driving into the tabletop.
                height_ok = (
                    height_center_error <= 0.006
                    and top_overlap + 0.001 >= required_overlap
                )
                if lateral_ok and height_ok:
                    alignment_ok = True
                    break
                # Correct from the measured controller pose, not from the
                # mathematically inaccessible below-table target. First reach
                # the contact-limited height; once there, remove any lateral
                # collision-response drift while holding that height.
                correction_target = self.controlled_ee_position().copy()
                if not lateral_ok:
                    correction_target[:2] += pad_error[:2]
                elif not height_ok:
                    # The controller already generates a smooth trajectory to
                    # a Cartesian waypoint. Command the complete measured
                    # pad-center correction at once; splitting it into 40 mm
                    # waypoints made the arm settle and visibly pause between
                    # every segment of an otherwise straight descent.
                    correction_target[2] += float(pad_error[2])
                try:
                    # Millimetre-scale fingertip overlap corrections must not
                    # be swallowed by the normal Cartesian waypoint tolerance.
                    correction_distance = float(np.linalg.norm(
                        correction_target - self.controlled_ee_position()
                    ))
                    correction_steps = int(np.clip(
                        np.ceil(correction_distance / 0.0015), 20, 120
                    ))
                    self.command_linear_ee_checked(
                        correction_target, move_orientation,
                        steps=correction_steps, tolerance=0.0015,
                    )
                except RuntimeError as alignment_error:
                    measured_midpoint, _, _, _, _ = self.finger_pad_geometry(
                        move_orientation
                    )
                    measured_error = desired_grasp_center - measured_midpoint
                    print(
                        f"[GRASP CONTACT-LIMITED ALIGN] {name}: controller "
                        f"stopped at physical contact; measured pad error="
                        f"{measured_error.round(5).tolist()}; "
                        f"controller_error={alignment_error}",
                        flush=True,
                    )
            if not alignment_ok:
                live_object_position, _ = object_prim.get_world_pose()
                live_object_position = np.asarray(live_object_position, dtype=float)
                desired_grasp_center = live_object_position + grasp_offset
                pad_midpoint, physical_closing_axis, open_aperture, _, _ = (
                    self.finger_pad_geometry(move_orientation)
                )
                pad_error = desired_grasp_center - pad_midpoint
                closing_center_error = abs(float(np.dot(
                    pad_error[:2], physical_closing_axis[:2]
                )))
                closing_component = (
                    np.dot(pad_error[:2], physical_closing_axis[:2])
                    * physical_closing_axis[:2]
                )
                transverse_center_error = float(np.linalg.norm(
                    pad_error[:2] - closing_component
                ))
                closing_margin = max(
                    0.0, 0.5 * (open_aperture - closing_width)
                )
                left = self.prim_world_position(self.left_finger_prim)
                right = self.prim_world_position(self.right_finger_prim)
                fingertip_midpoint = (
                    0.5 * (left + right)
                    + 0.058 * quat_to_rot_matrix(
                        np.asarray(move_orientation, dtype=float)
                    )[:, 2]
                )
                live_top_z = float(
                    live_object_position[2] + world_half_extents[2]
                )
                top_overlap = live_top_z - float(fingertip_midpoint[2])
                height_center_error = abs(float(pad_error[2]))
                if (top_overlap + 0.001 < required_overlap
                        or height_center_error > 0.008
                        or closing_center_error > closing_margin + 0.005
                        or transverse_center_error > 0.010):
                    raise RuntimeError(
                        "could not align open fingers around the physical "
                        f"grasp region: pad_error={pad_error.tolist()}, "
                        f"closing_margin={closing_margin:.4f}, "
                        f"top_overlap={top_overlap:.4f}, "
                        f"height_center_error={height_center_error:.4f}"
                    )
            self.log_move_stage(name, "grasp descent complete", object_prim, grasp_target)

            object_at_grasp, _ = object_prim.get_world_pose()
            object_at_grasp = np.asarray(object_at_grasp, dtype=float)
            preclose_displacement = float(np.linalg.norm(
                object_at_grasp[:2] - source[:2]
            ))
            if preclose_displacement > 0.015:
                raise RuntimeError(
                    "open-finger descent displaced the cuboid too far for a "
                    "realistic grasp; refusing to close on a chased object: "
                    f"start={source.tolist()}, before_close="
                    f"{object_at_grasp.tolist()}, displacement="
                    f"{preclose_displacement:.4f} m"
                )

            stage_name = "close gripper"
            self.robot.gripper.close()
            for _ in range(120):
                self.step_world(render=not ARGS.headless)
            self.log_move_stage(name, "gripper closed", object_prim, grasp_target)
            finger_positions = np.asarray(
                self.robot.gripper.get_joint_positions(), dtype=float
            )
            pad_midpoint, physical_closing_axis, finger_aperture, _, _ = (
                self.finger_pad_geometry(move_orientation)
            )
            closed_position, _ = object_prim.get_world_pose()
            closed_position = np.asarray(closed_position, dtype=float)
            print(
                f"[PHYSICAL GRASP CHECK] {name}: closing_width="
                f"{closing_width:.4f}; finger_aperture={finger_aperture:.4f}; "
                f"object_drift={np.linalg.norm(closed_position - object_at_grasp):.4f}; "
                f"desired_grasp_height={desired_grasp_center[2]:.4f}; "
                f"pad_height={pad_midpoint[2]:.4f}; "
                f"finger_joints={finger_positions.round(5).tolist()}",
                flush=True,
            )
            if finger_aperture < max(0.004, 0.45 * closing_width):
                raise RuntimeError(
                    "fingers closed without enclosing the cuboid: "
                    f"aperture={finger_aperture:.4f}, "
                    f"expected_width={closing_width:.4f}"
                )
            if finger_aperture > closing_width + 0.012:
                raise RuntimeError(
                    "fingers closed without opposing contact on the cuboid: "
                    f"aperture={finger_aperture:.4f}, "
                    f"expected_width={closing_width:.4f}"
                )
            if np.linalg.norm(closed_position[:2] - object_at_grasp[:2]) > 0.015:
                raise RuntimeError(
                    "cuboid was pushed out of the grasp during closure: "
                    f"before={object_at_grasp.tolist()}, "
                    f"after={closed_position.tolist()}"
                )

            # With a real friction grasp, lift vertically from the measured
            # closed pose.  Returning laterally to the nominal pre-contact
            # command while rising applies a shear load that can pull an
            # otherwise valid cuboid out of the fingers.
            lift = self.controlled_ee_position().copy()
            lift[2] = max(0.25, lift[2] + 0.18)

            stage_name = "lift"
            self.command_linear_ee_checked(
                lift, move_orientation, steps=180, tolerance=0.035
            )
            self.log_move_stage(name, "lift complete", object_prim, lift)
            lifted_position, _ = object_prim.get_world_pose()
            lifted_position = np.asarray(lifted_position, dtype=float)
            lift_height = float(lifted_position[2] - closed_position[2])
            lift_planar_slip = float(np.linalg.norm(
                lifted_position[:2] - closed_position[:2]
            ))
            print(
                f"[GRASP LIFT RESULT] {name}: vertical_rise={lift_height:.4f}; "
                f"planar_slip={lift_planar_slip:.4f}", flush=True,
            )
            # Planar slip is acceptable if the object is still physically
            # carried. Only a cuboid that remains essentially on its support
            # surface requires a re-pick from the latest live pose.
            if lift_height < 0.025:
                raise RuntimeError(
                    "physical grasp slip requires re-pick from latest pose: "
                    f"before={closed_position.tolist()}, "
                    f"after={lifted_position.tolist()}, "
                    f"vertical_rise={lift_height:.4f}, "
                    f"planar_slip={lift_planar_slip:.4f}"
                )

            carried_offset = self.controlled_ee_position() - lifted_position

            above_destination = destination + carried_offset
            above_destination[2] = lift[2]
            stage_name = "transport"
            self.command_linear_ee_checked(
                above_destination, move_orientation, steps=240,
                tolerance=0.030,
            )
            self.log_move_stage(
                name, "transport complete", object_prim, above_destination
            )

            # Re-observe both pose and gripper/object transform after transport.
            # Long end-grasped objects can rotate under gravity; placement
            # height must use the current vertical projection, not the original
            # tabletop height or orientation.
            carried_position, carried_orientation = object_prim.get_world_pose()
            carried_position = np.asarray(carried_position, dtype=float)
            carried_offset = self.controlled_ee_position() - carried_position
            carried_half_extents = np.abs(quat_to_rot_matrix(
                np.asarray(carried_orientation, dtype=float)
            )) @ (0.5 * np.asarray(cuboid.dimensions, dtype=float))
            place_target = destination + carried_offset
            if action.get("remove_after_place", False):
                # Disposal pads are outside the table and the object is meant
                # to fall away.  Release from a safe height instead of asking
                # the controlled frame to converge through the floor/object.
                collision_free_object_center_z = (
                    self.scene_offset[2] + float(carried_half_extents[2]) + 0.030
                )
                place_target[2] = max(
                    0.12, collision_free_object_center_z + carried_offset[2]
                )
                print(
                    f"[PLACE LIVE ORIENTATION] {name}: orientation="
                    f"{np.asarray(carried_orientation).round(5).tolist()}; "
                    f"vertical_half_extent={carried_half_extents[2]:.4f}; "
                    f"release_controlled_z={place_target[2]:.4f}",
                    flush=True,
                )
            stage_name = "place descent"
            if action.get("remove_after_place", False):
                # Disposal placement is intentionally best-effort. Command a
                # bounded descent but do not test Cartesian convergence: the
                # object is released and retired regardless of the achieved
                # waypoint, as required by the high-level removal action.
                self.command_ee_unchecked(
                    place_target, move_orientation, frames=180
                )
                self.log_move_stage(
                    name, "place descent command complete (unchecked)",
                    object_prim, place_target,
                )
            else:
                self.move_ee(place_target, tolerance=0.018,
                             orientation=move_orientation)
                self.log_move_stage(
                    name, "place descent complete", object_prim, place_target
                )

            stage_name = "release"
            self.robot.gripper.open()
            # The high-level removal destination is outside the live clump.
            # Retire immediately after separation begins so a falling disposal
            # object cannot bounce back into unresolved objects.
            release_frames = 3 if action.get("remove_after_place", False) else 45
            for _ in range(release_frames):
                self.step_world(render=not ARGS.headless)
            self.log_move_stage(name, "object released", object_prim, destination)

            if action.get("remove_after_place", False):
                # The high-level model permanently clears placed objects.
                # Retire only after a real transport and release so disposal
                # clutter cannot violate that assumption in later actions.
                object_prim.set_world_pose(
                    position=np.array([5.0, 5.0, -2.0], dtype=float)
                )
                object_prim.set_visibility(False)
                for _ in range(10):
                    self.step_world(render=not ARGS.headless)

            retreat = place_target.copy()
            retreat[2] = max(0.25, place_target[2] + 0.18)
            stage_name = "retreat"
            self.command_linear_ee_checked(
                retreat, move_orientation, steps=120, tolerance=0.035
            )
            self.settle()
            self.log_move_stage(name, "move settled", object_prim, destination)
            if not action.get("remove_after_place", False):
                self.goal_stack_heights[destination_pad] = support_height + cuboid.height
        except Exception as exc:
            raise RuntimeError(
                f"Move {name} failed during stage '{stage_name}': {exc}"
            ) from exc

    def execute_clearance_once(self, action, retry=False,
                               stop_displacement=None, direction_override=None):
        # The optimal planner currently emits one displaced cuboid per knock.
        # Iterate for backward-compatible files that list a short stack.
        ros_direction = str(action["dir"])
        direction = (np.asarray(direction_override, dtype=float)
                     if direction_override is not None else
                     self.clearance_direction(action, action["blocks"][0]))
        orientation = knock_orientation(direction)
        print(
            f"[KNOCK] direction ROS-local {ros_direction} -> "
            f"Isaac vector {direction.tolist()}, "
            f"tool quaternion WXYZ {orientation.tolist()}",
            flush=True,
        )
        displacements = []
        for name in action["blocks"]:
            stage_name = "initialization"
            cuboid = self.problem.cuboids[str(name)]
            object_prim = self.get_object(str(name))
            initial_position, initial_orientation = object_prim.get_world_pose()
            initial_position = np.asarray(initial_position, dtype=float)

            # Reproduce createTopKnockTask's physical intent: place the closed
            # fingers over the outward region of the top face, descend to a
            # small clearance, then move diagonally down and outward. Contact
            # at the top edge creates a tipping moment rather than a side push.
            world_half_extents = np.abs(
                quat_to_rot_matrix(np.asarray(initial_orientation, dtype=float))
            ) @ (0.5 * np.asarray(cuboid.dimensions, dtype=float))
            planar_size = 2.0 * max(world_half_extents[0:2])
            normal_size = 2.0 * float(np.dot(world_half_extents, np.abs(direction)))
            top_z = initial_position[2] + world_half_extents[2]
            # Begin near the center on the outward half. Starting at 35% of
            # the full width left almost no top surface beneath the fingertip,
            # so a physical (unattached) tool immediately slipped off. A 5%
            # offset preserves the requested outward-side contact while giving
            # the downward/outward stroke real top-surface travel.
            edge_xy = initial_position + direction * (0.05 * normal_size)
            above_edge = edge_xy.copy()
            above_edge[2] = top_z + max(0.10, cuboid.height)
            near_top = edge_xy.copy()
            # This must remain a reachable free-space waypoint. Asking the
            # position controller to converge below a rigid top surface makes
            # contact look like an IK timeout and can preload/tip the cuboid
            # before the intentional diagonal stroke begins.
            # The controlled right_gripper frame remains roughly 15--25 mm
            # above the actual finger contact surface.  Bias the command into
            # the top plane so RMPflow's collision response settles the finger
            # onto the surface before the outward stroke instead of allowing
            # the whole trajectory to pass just above short cuboids.
            # The two palm branches have different fixed offsets between the
            # controlled frame and the lowest finger surface.  World-X strokes
            # need a small command bias to establish contact; world-Y strokes
            # already contact at the top plane and that bias would preload and
            # disturb the object before the intentional stroke.
            contact_bias = -0.027
            near_top[2] = top_z + contact_bias + ARGS.knock_precontact_clearance

            print(
                f"[KNOCK GEOMETRY] {name}: center={initial_position.round(5).tolist()}; "
                f"outward_world={direction.tolist()}; "
                f"edge_offset={(edge_xy - initial_position).round(5).tolist()}; "
                f"above_edge={above_edge.round(5).tolist()}; "
                f"near_top={near_top.round(5).tolist()}",
                flush=True,
            )

            try:
                self.log_knock_stage(str(name), "start", object_prim)
                self.cartesian_controller.reset()
                stage_name = "closing gripper"
                self.robot.gripper.close()
                for _ in range(20):
                    self.step_world(render=not ARGS.headless)
                self.log_knock_stage(str(name), "gripper closed", object_prim)

                # Always leave the clutter vertically before translating. A
                # direct lateral path to above_edge can sweep the closed palm
                # through taller or already disturbed neighbors.
                live_scene_top = top_z
                for other_name, other_cuboid in self.problem.cuboids.items():
                    other = self.get_object(str(other_name))
                    other_position, other_orientation = other.get_world_pose()
                    other_half_extents = np.abs(quat_to_rot_matrix(
                        np.asarray(other_orientation, dtype=float)
                    )) @ (0.5 * np.asarray(other_cuboid.dimensions, dtype=float))
                    live_scene_top = max(
                        live_scene_top,
                        float(np.asarray(other_position)[2] + other_half_extents[2]),
                    )
                safe_z = max(0.32, live_scene_top + 0.12)
                stage_name = "vertical overhead recovery"
                overhead_current = self.controlled_ee_position().copy()
                overhead_current[2] = safe_z
                self.move_ee(overhead_current, tolerance=0.020,
                             orientation=TOP_DOWN_QUAT_WXYZ)
                self.log_knock_stage(
                    str(name), "vertical overhead complete", object_prim,
                    overhead_current,
                )

                stage_name = "overhead translation"
                overhead_target = above_edge.copy()
                overhead_target[2] = safe_z
                self.move_ee(overhead_target, tolerance=0.025,
                             orientation=TOP_DOWN_QUAT_WXYZ)
                self.log_knock_stage(
                    str(name), "overhead translation complete", object_prim,
                    overhead_target,
                )

                stage_name = "palm alignment"
                self.orient_ee_at_clearance(overhead_target, orientation)
                self.log_knock_stage(
                    str(name), "palm alignment complete", object_prim,
                    overhead_target,
                )

                stage_name = "vertical approach"
                achieved_approach = self.command_linear_ee(
                    above_edge, orientation, steps=120, frames_per_step=2,
                )
                approach_error = achieved_approach - above_edge
                if (np.linalg.norm(approach_error[:2]) > 0.020
                        or abs(float(approach_error[2])) > 0.030):
                    raise RuntimeError(
                        "bounded vertical approach missed its corridor: "
                        f"target={above_edge.tolist()}, "
                        f"achieved={achieved_approach.tolist()}, "
                        f"error={approach_error.tolist()}"
                    )
                self.log_knock_stage(
                    str(name), "vertical approach complete", object_prim,
                    above_edge,
                )

                stage_name = "pre-knock descent"
                # Contact intentionally prevents exact convergence of the
                # controlled frame to this biased waypoint.  The following
                # measured stroke/object displacement checks determine actual
                # success; this gate only rejects gross IK misses.
                descent_start = self.controlled_ee_position().copy()
                direction_xy = direction[:2]
                for descent_index, alpha in enumerate(
                        np.linspace(0.0, 1.0, 13)[1:], start=1):
                    descent_target = near_top.copy()
                    descent_target[2] = (
                        (1.0 - alpha) * descent_start[2]
                        + alpha * near_top[2]
                    )
                    contact_limited, pad_clearance, contact_frames = (
                        self.descend_to_clearance_contact(
                            descent_target, orientation, object_prim, cuboid,
                        )
                    )
                    achieved_step = self.controlled_ee_position()
                    lateral_error = achieved_step[:2] - near_top[:2]
                    lateral_error -= direction_xy * np.dot(
                        lateral_error, direction_xy
                    )
                    print(
                        f"[KNOCK VERTICAL DESCENT] {name}: "
                        f"step={descent_index}/12; z={achieved_step[2]:.4f}; "
                        f"transverse_error={np.linalg.norm(lateral_error):.4f}; "
                        f"pad_clearance={pad_clearance:.4f}",
                        flush=True,
                    )
                    if np.linalg.norm(lateral_error) > 0.012:
                        raise RuntimeError(
                            "vertical pre-contact approach drifted laterally; "
                            "refusing a low-height correction through clutter: "
                            f"error={np.linalg.norm(lateral_error):.4f} m"
                        )
                    if contact_limited:
                        print(
                            f"[KNOCK VERTICAL CONTACT] {name}: finger contact "
                            f"detected after {contact_frames} frames at "
                            f"controlled_z={achieved_step[2]:.4f}, "
                            f"pad_clearance={pad_clearance:.4f}; switching "
                            f"immediately to outward stroke",
                            flush=True,
                        )
                        break
                achieved_contact = self.controlled_ee_position()
                self.log_knock_stage(
                    str(name), "pre-knock complete", object_prim, near_top
                )
                pre_position, _ = object_prim.get_world_pose()
                raw_pre_displacement = float(np.dot(
                    np.asarray(pre_position) - initial_position, direction
                ))
                pre_displacement = max(0.0, raw_pre_displacement)
                transverse_pre_displacement = np.asarray(pre_position) - initial_position
                transverse_pre_displacement -= (
                    raw_pre_displacement * direction
                )
                print(
                    f"[KNOCK PRECONTACT DISTURBANCE] {name}: "
                    f"outward={raw_pre_displacement:.4f}; "
                    f"transverse={np.linalg.norm(transverse_pre_displacement[:2]):.4f}; "
                    f"vertical={transverse_pre_displacement[2]:.4f}",
                    flush=True,
                )
                pre_sufficient = (
                    stop_displacement is not None
                    and pre_displacement >= stop_displacement
                )
                if pre_sufficient:
                    print(
                        f"[KNOCK EARLY STOP] {name}: pre-contact descent "
                        f"created {pre_displacement:.4f} m; remaining_required="
                        f"{stop_displacement:.4f} m; skipping contact stroke",
                        flush=True,
                    )

                # Long cuboids must not receive a proportionally huge stroke:
                # that can launch the object outside the Panda workspace even
                # though only F+margin clearance is required.  Repeat bounded
                # strokes and stop on measured displacement instead.
                outward_travel = 2.0 * normal_size
                stroke_end = achieved_contact + direction * outward_travel
                # This is a shallow clearance slide, not another descent.
                requested_drop = outward_travel * np.tan(
                    np.deg2rad(ARGS.knock_angle_deg)
                )
                downward_drop = min(requested_drop, 0.004,
                                    0.08 * cuboid.height)
                stroke_end[2] = achieved_contact[2] - downward_drop
                print(
                    f"[KNOCK] {name} center={initial_position.tolist()} "
                    f"top_z={top_z:.4f} contact={achieved_contact.tolist()} "
                    f"stroke_end={stroke_end.tolist()} "
                    f"outward_travel={outward_travel:.4f} "
                    f"downward_drop={downward_drop:.4f} "
                    f"angle_deg={ARGS.knock_angle_deg:.1f}",
                    flush=True,
                )

                if not pre_sufficient:
                    stage_name = "pre-contact hold"
                    self.log_knock_stage(
                        str(name), "contact stroke starting", object_prim,
                        achieved_contact,
                    )

                stage_name = "contact stroke"
                contact_object_position, contact_object_orientation = (
                    object_prim.get_world_pose()
                )
                planar_contact_error = np.linalg.norm(
                    achieved_contact[:2] - np.asarray(contact_object_position)[:2]
                )
                contact_delta_xy = (
                    achieved_contact[:2]
                    - np.asarray(contact_object_position)[:2]
                )
                contact_longitudinal = float(np.dot(
                    contact_delta_xy, direction[:2]
                ))
                contact_transverse = float(np.linalg.norm(
                    contact_delta_xy
                    - contact_longitudinal * direction[:2]
                ))
                print(
                    f"[CLEARANCE CONTACT CHECK] {name}: "
                    f"planar={planar_contact_error:.4f}; "
                    f"longitudinal={contact_longitudinal:.4f}; "
                    f"transverse={contact_transverse:.4f}; "
                    f"height_above_top={achieved_contact[2] - top_z:.4f}; "
                    f"planar_limit={0.5 * planar_size + 0.040:.4f}",
                    flush=True,
                )
                if (not pre_sufficient
                        and (planar_contact_error > 0.5 * planar_size + 0.040
                             or achieved_contact[2] > top_z + 0.060)):
                    raise RuntimeError(
                        "tool is not close enough to establish continuous top contact: "
                        f"planar_error={planar_contact_error:.4f}, "
                        f"height_above_top={achieved_contact[2] - top_z:.4f}"
                    )
                # Keep compliant, overdamped gains active until the fingers
                # have unloaded contact vertically. Restoring stiff gains and
                # immediately taking a joint-space retreat released stored
                # contact force into the cuboid and caused random disturbance.
                saved_drives = scale_franka_arm_drives(
                    ARGS.contact_stiffness_scale,
                    ARGS.contact_damping_scale,
                )
                try:
                    if not pre_sufficient:
                        self.follow_contact_stroke(
                            str(name), object_prim, achieved_contact, stroke_end,
                            orientation, object_start=initial_position,
                            direction=direction,
                            stop_displacement=stop_displacement,
                        )
                    self.log_knock_stage(
                        str(name), "contact stroke complete", object_prim,
                        stroke_end,
                    )
                    stage_name = "slow vertical contact release"
                    self.slow_vertical_contact_retreat(orientation)
                    self.log_knock_stage(
                        str(name), "slow vertical contact release complete",
                        object_prim, self.controlled_ee_position(),
                    )
                finally:
                    restore_franka_arm_drives(saved_drives)
                # RMPflow can remain trapped in the direction-dependent wrist
                # branch after sustained contact.  Use the articulation's
                # joint drives for a smooth, physical return to the known safe
                # home posture before the next Cartesian action.
                stage_name = "retreat recovery"
                self.recover_home()
                self.log_knock_stage(
                    str(name), "retreat recovery complete", object_prim,
                    self.controlled_ee_position(),
                )

                # Reassert a deterministic closed state after loaded contact;
                # open only once the hand is safely clear of the cuboid.
                self.robot.gripper.close()
                for _ in range(20):
                    self.step_world(render=not ARGS.headless)
                self.robot.gripper.open()
                self.settle()
                self.log_knock_stage(str(name), "settling complete", object_prim)
            except Exception as exc:
                raise RuntimeError(
                    f"Knock {name} failed during stage '{stage_name}': {exc}"
                ) from exc

            final, _ = object_prim.get_world_pose()
            displacement = float(np.dot(np.asarray(final) - initial_position, direction))
            print(
                f"[KNOCK] {name} final={np.asarray(final).tolist()} "
                f"outward_displacement={displacement:.4f}",
                flush=True,
            )
            displacements.append(displacement)
        return min(displacements) if displacements else 0.0

    def execute_clearance(self, action):
        """Repeat the contact primitive until cumulative finger clearance is met."""
        clearance_direction = self.clearance_direction(
            action, action["blocks"][0]
        )
        full_open_required = 0.0
        clearance_widths = {}
        for name in action["blocks"]:
            object_prim = self.get_object(str(name))
            _, orientation = object_prim.get_world_pose()
            cuboid = self.problem.cuboids[str(name)]
            width = oriented_box_extent(
                cuboid.dimensions, orientation, clearance_direction
            )
            clearance_widths[str(name)] = width
            # At full opening, the inner finger begins beyond the cuboid side
            # by half of (80 mm - object width). Add the configured physical
            # finger/corridor allowance and robustness margin.
            full_open_required = max(
                full_open_required,
                0.5 * max(0.0, 0.080 - width)
                + ARGS.panda_finger_clearance
                + self.problem.clearance_margin,
            )
        required = max(
            self.problem.finger_contact_length + self.problem.clearance_margin,
            self.problem.finger_side_contact_width
            + self.problem.clearance_margin,
            float(action.get("required_displacement", 0.0)),
            full_open_required,
        )
        max_attempts = ARGS.max_clearance_attempts
        if max_attempts is None:
            max_attempts = self.problem.max_clearance_attempts
        max_attempts = max(10, int(max_attempts))
        cumulative = 0.0
        attempts = []
        attempt_errors = []
        action_start_positions = {}
        for name in action["blocks"]:
            position, _ = self.get_object(str(name)).get_world_pose()
            action_start_positions[str(name)] = np.asarray(position, dtype=float)
        print(
            f"[CLEARANCE] required={required:.4f} m "
            f"(F={self.problem.finger_contact_length:.4f}, "
            f"finger_side_width={self.problem.finger_side_contact_width:.4f}, "
            f"full_open=0.0800, finger={ARGS.panda_finger_clearance:.4f}, "
            f"projected_widths={clearance_widths}, "
            f"margin={self.problem.clearance_margin:.4f}); max_attempts={max_attempts}",
            flush=True,
        )
        self.last_clearance_orientation = knock_orientation(
            clearance_direction
        )
        for name in action["blocks"]:
            self.clearance_directions[str(name)] = clearance_direction.copy()
        for attempt in range(1, max_attempts + 1):
            before_positions = {}
            for name in action["blocks"]:
                position, _ = self.get_object(str(name)).get_world_pose()
                before_positions[str(name)] = np.asarray(position, dtype=float)
            try:
                displacement = self.execute_clearance_once(
                    action, retry=attempt > 1,
                    stop_displacement=max(0.0, required - cumulative),
                    direction_override=clearance_direction,
                )
                attempt_errors.append(None)
            except Exception as attempt_error:
                # Preserve every body's resulting pose. Credit only measured
                # outward progress and retry from the new physical state.
                measured = []
                for name in action["blocks"]:
                    position, _ = self.get_object(str(name)).get_world_pose()
                    measured.append(float(np.dot(
                        np.asarray(position) - before_positions[str(name)],
                        clearance_direction,
                    )))
                displacement = max(0.0, min(measured) if measured else 0.0)
                attempt_errors.append(str(attempt_error))
                print(
                    f"[CLEARANCE ATTEMPT ERROR] attempt={attempt}; "
                    f"measured_increment={displacement:.4f}; "
                    f"error={attempt_error}", flush=True,
                )
            cumulative += max(0.0, displacement)
            planar_displacements = []
            outward_displacements = []
            for name in action["blocks"]:
                position, _ = self.get_object(str(name)).get_world_pose()
                delta = np.asarray(position) - action_start_positions[str(name)]
                planar_displacements.append(float(np.linalg.norm(delta[:2])))
                outward_displacements.append(float(np.dot(
                    delta, clearance_direction
                )))
            net_planar = min(planar_displacements) if planar_displacements else 0.0
            net_outward = min(outward_displacements) if outward_displacements else 0.0
            # Arbitrary planar motion does not prove that the planned grasp
            # corridor opened. Retry from the live pose until directed outward
            # clearance is sufficient.
            effective_clearance = max(cumulative, net_outward)
            attempts.append(displacement)
            self.metrics["clearance_attempts"] += 1
            print(
                f"[CLEARANCE] attempt={attempt}/{max_attempts}; "
                f"increment={displacement:.4f}; cumulative={cumulative:.4f}; "
                f"net_outward={net_outward:.4f}; net_planar={net_planar:.4f}; "
                f"effective={effective_clearance:.4f}; "
                f"required={required:.4f}", flush=True,
            )
            # PhysX settled-pose noise and YAML rounding are sub-millimetre.
            # Do not perform another loaded contact cycle for a residual such
            # as 0.2 mm; it cannot materially change gripper accessibility and
            # can disturb an already cleared cuboid during release.
            acceptance_tolerance = 0.0020
            if effective_clearance + acceptance_tolerance >= required:
                self.metrics["clearance_retries"] += attempt - 1
                return {"attempts": attempt, "increments": attempts,
                        "attempt_errors": attempt_errors,
                        "cumulative_outward_displacement": cumulative,
                        "net_outward_displacement": net_outward,
                        "net_planar_displacement": net_planar,
                        "effective_clearance": effective_clearance,
                        "required": required,
                        "acceptance_tolerance": acceptance_tolerance}
            if attempt_errors[-1] is not None and attempt < max_attempts:
                self.recover_for_retry(
                    "CLEARANCE", ",".join(map(str, action["blocks"])),
                    attempt, max_attempts, attempt_errors[-1],
                )
        self.metrics["clearance_retries"] += max_attempts - 1
        raise RuntimeError(
            f"Clearance failed after {max_attempts} attempts: "
            f"cumulative={cumulative:.4f} m, required={required:.4f} m; "
            f"attempt_errors={attempt_errors}"
        )

    def scene_state_metrics(self):
        states = {}
        active = 0
        for name in self.problem.cuboids:
            prim = self.get_object(str(name))
            position, orientation = prim.get_world_pose()
            position = np.asarray(position, dtype=float)
            removed = bool(position[2] < -0.5 or position[0] > 4.0)
            active += int(not removed)
            states[str(name)] = {
                "position": position.tolist(),
                "orientation_wxyz": np.asarray(orientation, dtype=float).tolist(),
                "removed": removed,
            }
        total_actions = max(
            1, int(self.metrics.get("actions_planned") or len(self.problem.actions))
        )
        total_objects = max(1, len(self.problem.cuboids))
        self.metrics["action_completion_percentage"] = (
            100.0 * self.metrics["actions_completed"] / total_actions
        )
        self.metrics["object_completion_percentage"] = (
            100.0 * self.metrics["objects_removed"] / total_objects
        )
        self.metrics["objects_remaining"] = active
        self.metrics["final_object_states"] = states

    def recover_for_retry(self, kind, name, attempt, maximum, error):
        print(
            f"[{kind} RETRY] {name}: attempt={attempt}/{maximum} "
            f"failed: {error}", flush=True,
        )
        self.robot.gripper.open()
        for _ in range(30):
            self.step_world(render=not ARGS.headless)
        try:
            vertical = self.controlled_ee_position().copy()
            vertical[2] = max(0.32, vertical[2])
            self.move_ee(vertical, tolerance=0.025,
                         orientation=TOP_DOWN_QUAT_WXYZ, max_frames=360)
        except Exception as recovery_error:
            print(
                f"[{kind} RETRY RECOVERY WARNING] {name}: {recovery_error}",
                flush=True,
            )
        # Joint-space recovery is deterministic even when RMPflow remains in a
        # contact local minimum. It does not alter any cuboid pose.
        self.recover_home(frames=180)
        self.settle()

    @staticmethod
    def rectangles_overlap(corners_a, corners_b, clearance=0.0):
        """Strict SAT overlap for two planar oriented rectangles."""
        a = np.asarray(corners_a, dtype=float)
        b = np.asarray(corners_b, dtype=float)
        axes = []
        for poly in (a, b):
            for edge in (poly[1] - poly[0], poly[3] - poly[0]):
                axis = np.array([-edge[1], edge[0]], dtype=float)
                axis /= max(float(np.linalg.norm(axis)), 1e-9)
                axes.append(axis)
        for axis in axes:
            pa, pb = a @ axis, b @ axis
            if pa.max() <= pb.min() + clearance or pb.max() <= pa.min() + clearance:
                return False
        return True

    def live_planar_geometry(self, name):
        """Return center, local plan axes, and half extents from live PhysX pose."""
        cuboid = self.problem.cuboids[name]
        obj = self.get_object(name)
        position, orientation = obj.get_world_pose()
        position = np.asarray(position, dtype=float)
        rotation = quat_to_rot_matrix(np.asarray(orientation, dtype=float))
        # Export swaps planning X/Y into Isaac body Y/X.
        axis_x = rotation @ np.array([0.0, 1.0, 0.0])
        axis_y = rotation @ np.array([-1.0, 0.0, 0.0])
        for axis in (axis_x, axis_y):
            axis[2] = 0.0
            axis /= max(float(np.linalg.norm(axis[:2])), 1e-9)
        half_x = 0.5 * float(cuboid.dimensions[1])
        half_y = 0.5 * float(cuboid.dimensions[0])
        corners = np.asarray([
            position[:2] + sx*half_x*axis_x[:2] + sy*half_y*axis_y[:2]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ])
        return position, orientation, axis_x, axis_y, half_x, half_y, corners

    @staticmethod
    def rectangle_corners(center, axis_t, axis_n, half_t, half_n):
        return np.asarray([
            center + st*half_t*axis_t + sn*half_n*axis_n
            for st, sn in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ])

    def active_names(self, removed):
        return [name for name in self.problem.cuboids if name not in removed]

    def live_pick_candidates(self, removed):
        """Recompute collision-free top-grasp windows from current rigid poses."""
        active = self.active_names(removed)
        geometry = {name: self.live_planar_geometry(name) for name in active}
        candidates = []
        finger_width = float(self.problem.finger_side_contact_width)
        normal_depth = max(0.006, float(ARGS.panda_finger_clearance))
        for name in active:
            _, _, axis_x, axis_y, half_x, half_y, _ = geometry[name]
            for closing_axis, normal, tangent, half_normal, half_tangent in (
                ("x", axis_x, axis_y, half_x, half_y),
                ("y", axis_y, axis_x, half_y, half_x),
            ):
                closing_width = 2.0 * half_normal
                if closing_width > 0.075 or 2.0*half_tangent + 1e-9 < finger_width:
                    continue
                lo = -half_tangent + 0.5*finger_width
                hi = half_tangent - 0.5*finger_width
                placements = np.linspace(lo, hi, max(3, int((hi-lo)/0.005)+2))
                for center_t in placements:
                    blocked = False
                    target_center = geometry[name][0][:2]
                    for side in (-1.0, 1.0):
                        finger_center = (
                            target_center + center_t*tangent[:2]
                            + side*(half_normal + 0.5*normal_depth)*normal[:2]
                        )
                        finger_rect = self.rectangle_corners(
                            finger_center, tangent[:2], normal[:2],
                            0.5*finger_width, 0.5*normal_depth,
                        )
                        for other in active:
                            if other == name:
                                continue
                            if self.rectangles_overlap(finger_rect, geometry[other][6]):
                                blocked = True
                                break
                        if blocked:
                            break
                    if not blocked:
                        offset = ([0.0, float(center_t)] if closing_axis == "x"
                                  else [float(center_t), 0.0])
                        candidates.append({
                            "name": name, "closing_axis": closing_axis,
                            "offset": offset,
                            "finger_window": [float(center_t-0.5*finger_width),
                                              float(center_t+0.5*finger_width)],
                        })
                        break
        return candidates

    def live_push_candidates(self, removed, distance):
        """Return object/local-face pairs with an empty sampled sweep corridor."""
        active = self.active_names(removed)
        geometry = {name: self.live_planar_geometry(name) for name in active}
        candidates = []
        for name in active:
            _, _, axis_x, axis_y, half_x, half_y, corners = geometry[name]
            for label, direction in (("+x", axis_x), ("-x", -axis_x),
                                     ("+y", axis_y), ("-y", -axis_y)):
                free = True
                for alpha in np.linspace(0.05, 1.0, 20):
                    moved = corners + alpha*distance*direction[:2]
                    if any(self.rectangles_overlap(moved, geometry[o][6])
                           for o in active if o != name):
                        free = False
                        break
                if free:
                    candidates.append((name, label, direction.copy()))
        return candidates

    def execute_side_push(self, name, direction, distance):
        """Approach from overhead, contact an unblocked side, and push 5--10 cm."""
        obj = self.get_object(name)
        before, orientation = obj.get_world_pose()
        before = np.asarray(before, dtype=float)
        _, _, _, _, _, _, corners = self.live_planar_geometry(name)
        projection = corners @ direction[:2]
        rear_support = float(before[:2] @ direction[:2] - projection.min())
        contact_allowance = 0.012
        contact_xy = before[:2] - direction[:2]*(rear_support + contact_allowance)
        # Descending at the face itself lets the fingers clip the packed
        # boundary.  Descend outside the clump, then make the intentional
        # horizontal side-face approach.
        staging_xy = contact_xy - 0.060 * direction[:2]
        push_orientation = knock_orientation(direction)
        controlled = self.controlled_ee_position().copy()
        overhead_z = max(0.32, controlled[2])
        self.robot.gripper.close()
        for _ in range(45):
            self.step_world(render=not ARGS.headless)
        self.move_ee(np.array([controlled[0], controlled[1], overhead_z]),
                     tolerance=0.025, orientation=push_orientation)
        self.move_ee(np.array([staging_xy[0], staging_xy[1], overhead_z]),
                     tolerance=0.020, orientation=push_orientation)
        # The controlled frame sits about 6.4 mm above the measured pad
        # midpoint in the top-down configurations used by the executor.
        side_height = float(before[2] + 0.0064)
        staging = np.array([staging_xy[0], staging_xy[1], side_height])
        self.command_linear_ee_checked(staging, push_orientation, steps=600,
                                       tolerance=0.030)
        start = np.array([contact_xy[0], contact_xy[1], side_height])
        self.command_linear_ee_checked(start, push_orientation, steps=240,
                                       tolerance=0.030)
        end = start + distance*direction
        self.command_linear_ee_checked(end, push_orientation, steps=300,
                                       tolerance=0.025)
        release = self.controlled_ee_position().copy()
        release[2] = max(0.25, release[2] + 0.12)
        # Once the push stroke has completed, exact retreat convergence is not
        # part of push success.  Command a smooth unload and let deterministic
        # home recovery finish it instead of rejecting a 1 mm tolerance miss.
        self.command_ee_unchecked(release, push_orientation, frames=180)
        self.recover_home(frames=180)
        self.settle()
        after, _ = obj.get_world_pose()
        after = np.asarray(after, dtype=float)
        displacement = float(np.linalg.norm(after[:2] - before[:2]))
        return before, after, displacement

    def run_random_push(self):
        """Closed-loop baseline: exhaustive live Picks, then one random side Push."""
        started = time.perf_counter()
        rng = random.Random(self.seed)
        removed = set()
        destination = {str(a.get("block")): a.get("to")
                       for a in self.problem.actions if a.get("type") == "Move"}
        action_index = 0
        while len(removed) < len(self.problem.cuboids):
            self.metrics["live_replans"] += 1
            picked_during_closure = False
            failed_this_closure = set()
            while True:
                replan_started = time.perf_counter()
                candidates = [c for c in self.live_pick_candidates(removed)
                              if c["name"] not in failed_this_closure]
                self.metrics["live_replan_time_sec"] += (
                    time.perf_counter() - replan_started
                )
                if not candidates:
                    break
                candidate = candidates[0]
                name = candidate["name"]
                if name not in destination:
                    failed_this_closure.add(name)
                    continue
                action_index += 1
                self.metrics["actions_planned"] = action_index
                self.metrics["pick_attempts"] += 1
                action_started = time.perf_counter()
                record = {"index": action_index, "type": "RandomBaselinePick",
                          "target": name, "live_grasp": candidate}
                action = {
                    "type": "Move", "block": name,
                    "to": destination[name], "remove_after_place": True,
                    "grasp": {
                        "closing_axis": candidate["closing_axis"],
                        "offset": candidate["offset"],
                        "finger_window": candidate["finger_window"],
                        "finger_side_contact_width": self.problem.finger_side_contact_width,
                        "physical_edge_margin": 0.0,
                        "inward_nudge": 0.0,
                    },
                }
                try:
                    self.execute_move(action)
                    removed.add(name)
                    picked_during_closure = True
                    self.metrics["successful_picks"] += 1
                    self.metrics["objects_removed"] += 1
                    self.metrics["actions_completed"] += 1
                    record["status"] = "success"
                except Exception as exc:
                    failed_this_closure.add(name)
                    self.metrics["failed_picks"] += 1
                    record.update(status="failed", error=str(exc),
                                  failure_category=self.classify_move_failure(exc))
                    try:
                        self.recover_for_retry("RANDOM PICK", name, 1, 1, exc)
                    except Exception:
                        pass
                record["duration_sec"] = time.perf_counter() - action_started
                self.metrics["action_records"].append(record)
                self.scene_state_metrics()
            if len(removed) == len(self.problem.cuboids):
                break
            self.metrics["deadlocks"] += 1
            if self.metrics["push_actions"] >= ARGS.max_random_pushes:
                self.metrics["failure"] = "maximum_random_pushes_reached"
                break
            distance = rng.uniform(ARGS.push_min_distance, ARGS.push_max_distance)
            replan_started = time.perf_counter()
            push_candidates = self.live_push_candidates(removed, distance)
            self.metrics["live_replan_time_sec"] += (
                time.perf_counter() - replan_started
            )
            if not push_candidates:
                self.metrics["failure"] = "no_collision_free_push_face"
                break
            name, face, direction = rng.choice(push_candidates)
            action_index += 1
            self.metrics["actions_planned"] = action_index
            self.metrics["push_actions"] += 1
            self.metrics["push_attempts"] += 1
            action_started = time.perf_counter()
            record = {"index": action_index, "type": "RandomPush",
                      "target": name, "face": face,
                      "direction_isaac": direction.tolist(),
                      "commanded_distance": distance}
            print(f"[RANDOM PUSH] {name} face={face} distance={distance:.4f}",
                  flush=True)
            try:
                before, after, displacement = self.execute_side_push(
                    name, direction, distance)
                record.update(before=before.tolist(), after=after.tolist(),
                              displacement=displacement)
                if displacement >= 0.005:
                    record["status"] = "success"
                    self.metrics["successful_pushes"] += 1
                    self.metrics["actions_completed"] += 1
                else:
                    record.update(status="failed",
                                  error="physical displacement below 5 mm")
                    self.metrics["failed_pushes"] += 1
            except Exception as exc:
                record.update(status="failed", error=str(exc))
                self.metrics["failed_pushes"] += 1
                try:
                    self.recover_for_retry("RANDOM PUSH", name, 1, 1, exc)
                except Exception:
                    pass
            record["duration_sec"] = time.perf_counter() - action_started
            self.metrics["action_records"].append(record)
            self.metrics["execution_time_sec"] = time.perf_counter() - started
            self.scene_state_metrics()
        self.metrics["execution_time_sec"] = time.perf_counter() - started
        self.metrics["status"] = (
            "success" if len(removed) == len(self.problem.cuboids)
            else "partial_success" if removed else "failed"
        )
        self.metrics["objects_removed"] = len(removed)
        # Baseline action count is adaptive rather than YAML-plan length.
        self.metrics["actions_planned"] = action_index
        self.scene_state_metrics()
        return self.metrics

    def run(self):
        started = time.perf_counter()
        for index, action in enumerate(self.problem.actions, start=1):
            kind = action["type"]
            label = action.get("block", ",".join(action.get("blocks", [])))
            print(f"[EXEC] {index}/{len(self.problem.actions)} {kind} {label}", flush=True)
            action_started = time.perf_counter()
            record = {"index": index, "type": kind, "target": label}
            try:
                if kind == "Move":
                    move_errors = []
                    move_succeeded = False
                    for attempt in range(1, ARGS.max_move_attempts + 1):
                        try:
                            attempt_action = dict(action)
                            attempt_action["_attempt"] = attempt
                            self.execute_move(attempt_action)
                            record["attempts"] = attempt
                            record["retries"] = attempt - 1
                            move_succeeded = True
                            break
                        except Exception as move_error:
                            move_errors.append(str(move_error))
                            record.setdefault("attempt_diagnostics", []).append({
                                "attempt": attempt,
                                "error": str(move_error),
                                "failure_category": self.classify_move_failure(
                                    move_error
                                ),
                                "joint_limits": self.joint_limit_diagnostics(),
                            })
                            if attempt >= ARGS.max_move_attempts:
                                record["attempts"] = attempt
                                record["retries"] = attempt - 1
                                record["attempt_errors"] = move_errors
                                break
                            self.recover_for_retry(
                                "MOVE", str(label), attempt,
                                ARGS.max_move_attempts, move_error
                            )
                    if not move_succeeded:
                        self.metrics["failed_move_actions"] += 1
                        self.metrics["failed_picks"] += 1
                        self.metrics["skipped_moves"] += 1
                        record["status"] = "skipped"
                        record["error"] = move_errors[-1]
                        record["failure_category"] = self.classify_move_failure(
                            move_errors[-1]
                        )
                        print(
                            f"[MOVE SKIPPED] {label}: failed after "
                            f"{ARGS.max_move_attempts} attempts; preserving "
                            "the live scene and continuing the plan",
                            flush=True,
                        )
                        continue
                    self.metrics["objects_removed"] += 1
                else:
                    self.metrics["clearance_actions"] += 1
                    record.update(self.execute_clearance(action))
                record["status"] = "success"
                self.metrics["actions_completed"] += 1
            except Exception as exc:
                if kind != "Move":
                    self.metrics["failed_clearance_actions"] += 1
                    self.metrics["failed_knocks"] += 1
                record["status"] = "failed"
                record["error"] = str(exc)
                print(
                    f"[CLEARANCE SKIPPED] {label}: {exc}; preserving the "
                    "live scene and continuing the plan",
                    flush=True,
                )
                try:
                    self.recover_for_retry(
                        "CLEARANCE", str(label), 1, 1, exc
                    )
                except Exception as recovery_error:
                    print(
                        f"[CLEARANCE SKIP RECOVERY WARNING] {label}: "
                        f"{recovery_error}", flush=True,
                    )
            finally:
                record["duration_sec"] = time.perf_counter() - action_started
                self.metrics["action_records"].append(record)
                self.metrics["execution_time_sec"] = time.perf_counter() - started
                self.scene_state_metrics()
        if (self.metrics["failed_move_actions"]
                or self.metrics["failed_clearance_actions"]):
            print(
                f"[EXEC DONE] plan exhausted with "
                f"{self.metrics['failed_move_actions']} skipped Move actions "
                f"and {self.metrics['failed_clearance_actions']} failed "
                "Clearance actions"
            )
            self.metrics["status"] = "partial_success"
        else:
            print(f"[EXEC DONE] all {len(self.problem.actions)} high-level actions succeeded")
            self.metrics["status"] = "success"
        self.scene_state_metrics()
        return self.metrics


def configure_world():
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=ARGS.physics_dt,
        rendering_dt=ARGS.physics_dt,
    )
    physics = world.get_physics_context()
    physics.set_solver_type("TGS")
    physics.enable_ccd(True)
    if ARGS.batch_manifest:
        # GPU dynamics/broadphase amortize one simulation step across the
        # separated environments. The legacy single-scene path retains its
        # established CPU physics settings.
        physics.enable_gpu_dynamics(True)
        physics.set_broadphase_type("GPU")
    if ARGS.batch_manifest:
        world.scene.add_default_ground_plane(z_position=SCENE_OFFSET[2])
    else:
        assets = get_assets_root_path()
        if assets:
            stage.add_reference_to_stage(assets + ROOM_USD, "/World/Room")
        else:
            world.scene.add_default_ground_plane(z_position=SCENE_OFFSET[2])
    return world


def create_materials():
    material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/CuboidMaterial",
        # Prefer controlled sliding over tipping when the closed fingers drag
        # across the top. Static friction still holds the settled layout, while
        # the lower dynamic coefficient avoids converting lateral force into a
        # large overturning moment.
        static_friction=0.45,
        dynamic_friction=0.30,
        restitution=0.0,
    )
    finger_material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/FingerMaterial",
        static_friction=1.20,
        dynamic_friction=1.00,
        restitution=0.0,
    )
    return material, finger_material


def create_robot(world, namespace, scene_offset):
    root = f"/World/{namespace}/Franka" if namespace else "/World/Franka"
    name = f"{namespace.replace('/', '__')}__franka" if namespace else "franka"
    robot = world.scene.add(Franka(
        prim_path=root,
        name=name,
        position=np.asarray(scene_offset, dtype=float),
        orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.pi / 2.0])),
    ))
    tune_franka_drives(root)
    return robot, root


def create_problem_cuboids(world, problem, namespace, scene_offset, material):
    prefix = f"{namespace}__" if namespace else ""
    root = f"/World/{namespace}/Cuboids" if namespace else "/World/Cuboids"
    names = []
    for cuboid in problem.cuboids.values():
        xy = plan_xy_to_isaac(problem.pads[cuboid.initial_pad], scene_offset)
        z = scene_offset[2] + cuboid.level * cuboid.height + 0.5 * cuboid.height
        color = (
            np.array([0.85, 0.15, 0.15]) if cuboid.name.lower().startswith("red")
            else np.array([0.15, 0.75, 0.20]) if cuboid.name.lower().startswith("green")
            else np.array([0.15, 0.30, 0.85])
        )
        cuboid_path = f"{root}/{cuboid.name}"
        object_name = prefix + cuboid.name
        world.scene.add(DynamicCuboid(
            prim_path=cuboid_path,
            name=object_name,
            position=np.array([xy[0], xy[1], z]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, cuboid.yaw_rad])),
            scale=np.asarray(cuboid.dimensions),
            size=1.0,
            color=color,
            mass=500.0 * np.prod(cuboid.dimensions),
            physics_material=material,
        ))
        tune_cuboid_rigid_body(cuboid_path)
        names.append(object_name)
    return prefix, names


def set_problem_active(world, task_info, active, scene_offset):
    """Activate a pre-authored task at its lane or park it outside physics."""
    problem = task_info["problem"]
    prefix = task_info["prefix"]
    target_offset = (np.asarray(scene_offset, dtype=float) if active else
                     np.asarray(task_info["storage_offset"], dtype=float))
    for cuboid in problem.cuboids.values():
        obj = world.scene.get_object(prefix + cuboid.name)
        xy = plan_xy_to_isaac(problem.pads[cuboid.initial_pad], target_offset)
        z = target_offset[2] + cuboid.level*cuboid.height + 0.5*cuboid.height
        obj.set_world_pose(
            np.array([xy[0], xy[1], z]),
            euler_angles_to_quat(np.array([0.0, 0.0, cuboid.yaw_rad])),
        )
        obj.set_linear_velocity(np.zeros(3))
        obj.set_angular_velocity(np.zeros(3))


def tune_initialized_robot(robot, finger_material=None, robot_root=None):
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(16)
    robot.set_stabilization_threshold(0.0)
    robot.set_sleep_threshold(0.0)
    robot.gripper.open()


def run_single():
    problem = load_problem(ARGS.yaml)
    world = configure_world()
    material, finger_material = create_materials()
    robot, robot_root = create_robot(world, "", SCENE_OFFSET)
    bind_finger_physics_material(finger_material, robot_root)
    create_problem_cuboids(world, problem, "", SCENE_OFFSET, material)
    world.reset()
    tune_initialized_robot(robot, finger_material, robot_root)
    executor = IsaacPlanExecutor(world, robot, problem)
    executor.settle()
    if not ARGS.no_wait:
        wait_for_keypress(world)
    failure = None
    try:
        if ARGS.policy == "random-push":
            executor.run_random_push()
        else:
            executor.run()
    except Exception as exc:
        failure = exc
        print(f"[EXEC ERROR] {exc}", file=sys.stderr, flush=True)
    finally:
        if ARGS.metrics_json:
            metrics_path = Path(ARGS.metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(executor.metrics, indent=2), encoding="utf-8"
            )
            print(f"[METRICS] {metrics_path}", flush=True)
    if not ARGS.headless:
        if failure is not None:
            print("[EXEC PAUSED] Failure scene remains open for inspection.", flush=True)
        while APP.is_running():
            world.step(render=True)
    if failure is not None:
        raise failure


class SynchronizedWorld:
    """World proxy that advances one shared frame after every lane is ready."""

    def __init__(self, world, coordinator, lane):
        self._world = world
        self._coordinator = coordinator
        self._lane = lane
        self.scene = world.scene

    def step(self, render=False):
        self._coordinator.request_step(self._lane)


class ParallelStepCoordinator:
    """Keep Isaac stepping on its main thread while policies run as clients."""

    def __init__(self, world, lanes):
        self.world = world
        self.condition = threading.Condition()
        self.active = set(range(lanes))
        self.waiting = set()
        self.generation = 0
        self.started = time.perf_counter()

    def request_step(self, lane):
        with self.condition:
            generation = self.generation
            self.waiting.add(lane)
            self.condition.notify_all()
            self.condition.wait_for(
                lambda: self.generation != generation or lane not in self.active
            )

    def finish(self, lane):
        with self.condition:
            self.active.discard(lane)
            self.waiting.discard(lane)
            self.condition.notify_all()

    def run(self):
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: not self.active or self.active.issubset(self.waiting)
                )
                if not self.active:
                    elapsed = time.perf_counter() - self.started
                    print(f"[PARALLEL PHYSICS] frames={self.generation}; "
                          f"wall_time_sec={elapsed:.3f}; "
                          f"frames_per_sec={self.generation/max(elapsed, 1e-9):.1f}",
                          flush=True)
                    return
            # Isaac's update loop and tensor backend must remain on the thread
            # that created SimulationApp.
            self.world.step(render=not ARGS.headless)
            with self.condition:
                self.waiting.difference_update(self.active)
                self.generation += 1
                self.condition.notify_all()


def workspace_offset(index, count, spacing):
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    row, column = divmod(index, columns)
    return SCENE_OFFSET + np.array([
        (column - 0.5 * (columns - 1)) * spacing,
        (row - 0.5 * (rows - 1)) * spacing,
        0.0,
    ])


def run_parallel_round(world, robots, roots, homes, task_infos, round_index):
    executors = []
    for lane, task_info in enumerate(task_infos):
        offset = workspace_offset(lane, len(robots), ARGS.workspace_spacing)
        set_problem_active(world, task_info, True, offset)
        robots[lane].set_joint_positions(homes[lane])
        robots[lane].set_joint_velocities(np.zeros_like(homes[lane]))
        executors.append((task_info["task"], task_info["problem"],
                          task_info["prefix"], offset))
    for lane in range(len(task_infos)):
        # Physics/material tuning is authored once after the sole World reset.
        # Rebinding collision materials here invalidates Isaac tensor views.
        robots[lane].gripper.open()

    # Policies submit controller commands from independent clients. Isaac's
    # shared physics world itself is advanced only by this main-thread driver.
    coordinator = ParallelStepCoordinator(world, len(task_infos))
    instances = []
    for lane, (task, problem, prefix, offset) in enumerate(executors):
        instances.append(IsaacPlanExecutor(
            SynchronizedWorld(world, coordinator, lane), robots[lane], problem,
            scene_offset=offset, object_name_prefix=prefix,
            robot_root=roots[lane], policy=task.get("policy", ARGS.policy),
            seed=task.get("seed", ARGS.seed),
        ))

    failures = [None] * len(task_infos)

    def worker(index):
        executor = instances[index]
        try:
            executor.settle()
            if executor.policy == "random-push":
                executor.run_random_push()
            else:
                executor.run()
        except Exception as exc:
            failures[index] = str(exc)
            executor.metrics["status"] = "failed"
            executor.metrics["failure"] = str(exc)
            print(f"[PARALLEL ERROR] lane={index}: {exc}", file=sys.stderr,
                  flush=True)
        finally:
            coordinator.finish(index)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(len(task_infos))]
    for thread in threads:
        thread.start()
    coordinator.run()
    for thread in threads:
        thread.join()

    for task_info, executor in zip(task_infos, instances):
        task = task_info["task"]
        metrics_path = Path(task["metrics_json"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(executor.metrics, indent=2),
                                encoding="utf-8")
        print(f"[PARALLEL RESULT] {task.get('label', task['yaml'])}: "
              f"{executor.metrics.get('status')} -> {metrics_path}", flush=True)

    for task_info in task_infos:
        set_problem_active(world, task_info, False, SCENE_OFFSET)
    return failures


def run_parallel_batch():
    manifest_path = Path(ARGS.batch_manifest)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = document.get("tasks", document) if isinstance(document, dict) else document
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("batch manifest must contain a nonempty tasks list")
    # The scheduler hands off at every physics frame. Python's default 5 ms
    # thread switch interval otherwise dominates small vectorized simulations.
    sys.setswitchinterval(0.0002)
    lanes = min(ARGS.parallel_workspaces, len(tasks))
    if ARGS.workspace_spacing < 2.0:
        raise ValueError("parallel workspace spacing must be at least 2.0 m")
    world = configure_world()
    material, finger_material = create_materials()
    robots, roots = [], []
    for lane in range(lanes):
        offset = workspace_offset(lane, lanes, ARGS.workspace_spacing)
        robot, root = create_robot(world, f"ParallelRobots/env_{lane:03d}", offset)
        robots.append(robot)
        roots.append(root)
    task_infos = []
    for task_index, task in enumerate(tasks):
        problem = load_problem(task["yaml"])
        namespace = f"ParallelTasks/task_{task_index:04d}"
        # Park inactive scenes as a dense but mutually isolated grid on the
        # same ground plane. No USD physics properties are edited after reset,
        # because doing so invalidates Isaac articulation tensor views.
        storage_columns = int(np.ceil(np.sqrt(len(tasks))))
        storage_row, storage_column = divmod(task_index, storage_columns)
        storage_offset = np.array([
            14.0 + 0.60*storage_column,
            -4.0 + 0.60*storage_row,
            SCENE_OFFSET[2],
        ])
        prefix, object_names = create_problem_cuboids(
            world, problem, namespace, storage_offset, material
        )
        task_infos.append({
            "task": task, "problem": problem, "prefix": prefix,
            "object_names": object_names,
            "storage_offset": storage_offset,
        })
    for root in roots:
        bind_finger_physics_material(finger_material, root)
    world.reset()
    for robot, root in zip(robots, roots):
        tune_initialized_robot(robot, finger_material, root)
    homes = [np.asarray(robot.get_joint_positions(), dtype=float).copy()
             for robot in robots]
    for task_info in task_infos:
        set_problem_active(world, task_info, False, SCENE_OFFSET)

    all_failures = []
    for round_index, start in enumerate(range(0, len(tasks), lanes)):
        round_tasks = task_infos[start:start + lanes]
        print(f"[PARALLEL ROUND] {round_index + 1}/"
              f"{int(np.ceil(len(tasks) / lanes))}: {len(round_tasks)} tasks",
              flush=True)
        all_failures.extend(run_parallel_round(
            world, robots, roots, homes, round_tasks, round_index,
        ))
    summary_path = document.get("summary_json") if isinstance(document, dict) else None
    if summary_path:
        Path(summary_path).write_text(json.dumps({
            "tasks": len(tasks), "parallel_workspaces": lanes,
            "failed": sum(failure is not None for failure in all_failures),
            "failures": all_failures,
        }, indent=2), encoding="utf-8")


def main():
    if ARGS.batch_manifest:
        run_parallel_batch()
    else:
        run_single()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[EXEC TERMINATED] {exc}", file=sys.stderr)
        raise
    finally:
        APP.close()
