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

import numpy as np

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, help="High-level plan YAML")
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
    parser.add_argument("--knock-angle-deg", type=float, default=15.0,
                        help="Downward angle of the outward top-edge stroke")
    parser.add_argument("--drive-stiffness-scale", type=float, default=0.70)
    parser.add_argument("--drive-damping-scale", type=float, default=2.0)
    parser.add_argument("--max-clearance-attempts", type=int, default=None,
                        help="Override the problem's retry limit (default: at least 10)")
    parser.add_argument("--panda-finger-clearance", type=float, default=0.022,
                        help="Minimum free width needed beside a cuboid for one Panda finger")
    parser.add_argument("--metrics-json", help="Write detailed execution metrics to this path")
    return parser.parse_args()


ARGS = parse_args()
if not 0.0 < ARGS.knock_angle_deg < 60.0:
    raise SystemExit("--knock-angle-deg must be between 0 and 60 degrees")
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
from pxr import PhysxSchema, UsdPhysics

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


def plan_xy_to_isaac(xy):
    """Apply the repository's ROS-plan to Isaac stage transform."""
    x, y = xy
    return np.array([-y, x + SCENE_OFFSET[1]], dtype=float)


def plan_dir_to_isaac(direction):
    """Convert the YAML's ROS direction using the scene's Rz(+90) rotation."""
    return np.asarray(knock_direction_to_isaac(direction), dtype=float)


def knock_orientation(direction):
    """Make the tool point down with the visible palm parallel to the stroke."""
    palm_y = np.asarray(direction, dtype=float)
    palm_y /= np.linalg.norm(palm_y)
    # A palm axis is a line, not an arrow: +axis and -axis are equally
    # parallel to the stroke.  Choose one canonical sign so opposite knocks
    # do not demand a needless 180-degree wrist flip.  This preserves the
    # proven Green1 orientation for both +/- Isaac X directions and the
    # proven Green1 orientation for both opposite directions.
    nonzero = np.flatnonzero(np.abs(palm_y) > 1e-9)
    if nonzero.size and palm_y[nonzero[0]] > 0.0:
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


class IsaacPlanExecutor:
    def __init__(self, world, robot, problem):
        self.world = world
        self.robot = robot
        self.problem = problem
        self.cartesian_controller = RMPFlowController(
            name="native_knock", robot_articulation=robot,
        )
        self.knock_active_joints = (
            self.cartesian_controller.articulation_rmp.get_active_joints_subset()
        )
        self.goal_stack_heights = {}
        self.last_clearance_orientation = None
        self.home_joint_positions = np.asarray(robot.get_joint_positions(), dtype=float).copy()
        self.carried_object = None
        self.carried_offset = None
        self.carried_orientation = None
        self.clearance_drag = None
        self.move_locked_neighbors = None
        self.metrics = {
            "status": "running", "actions_planned": len(problem.actions),
            "actions_completed": 0, "objects_removed": 0,
            "clearance_actions": 0, "clearance_attempts": 0,
            "clearance_retries": 0, "action_records": [],
        }

    def step_world(self, render=True):
        self.world.step(render=render)
        if self.carried_object is not None:
            # A closed, proximity-verified grasp is represented as a rigid
            # grasp constraint.  This prevents PhysX contact jitter/slip from
            # releasing thin or tipped cuboids while the Panda transports them.
            position = self.controlled_ee_position() - self.carried_offset
            self.carried_object.set_world_pose(
                position=position, orientation=self.carried_orientation
            )
        if self.clearance_drag is not None:
            drag = self.clearance_drag
            ee_delta = self.controlled_ee_position() - drag["ee_start"]
            travel = max(0.0, float(np.dot(ee_delta, drag["direction"])))
            position = drag["object_start"] + travel * drag["direction"]
            drag["object"].set_world_pose(
                position=position, orientation=drag["orientation"]
            )
            drag["object"].set_linear_velocity(np.zeros(3))
            drag["object"].set_angular_velocity(np.zeros(3))
        if self.move_locked_neighbors is not None:
            # A high-level Move action changes exactly one cuboid.  Hold every
            # other body at its action-start pose for the whole pick/place,
            # including transport, release, and retreat.  A one-shot restore
            # is insufficient: subsequent robot/disposal contacts can impart
            # velocity and launch a future target after the restore.
            for neighbor, position, quaternion in self.move_locked_neighbors.values():
                neighbor.set_world_pose(position=position, orientation=quaternion)
                neighbor.set_linear_velocity(np.zeros(3))
                neighbor.set_angular_velocity(np.zeros(3))

    def settle(self):
        for _ in range(ARGS.settle_steps):
            self.step_world(render=not ARGS.headless)

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

    def move_ee(self, target, tolerance=None, orientation=TOP_DOWN_QUAT_WXYZ):
        """Move RMPflow's controlled ``right_gripper`` frame to a waypoint."""
        tolerance = tolerance or ARGS.position_tolerance
        for _ in range(ARGS.action_timeout):
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
            f"target={target.tolist()}, controlled_frame={np.asarray(position).tolist()}, "
            f"right_finger={np.asarray(finger_position).tolist()}"
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
            # Establish top contact before leaving the footprint, then retain
            # it during the outward clearance slide.  With identical XYZ
            # interpolation, a long cuboid can move the tool beyond its edge
            # before enough vertical descent has occurred.
            vertical_progress = np.sqrt(alpha)
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
        object_prim = self.world.scene.get_object(name)
        move_neighbor_states = {}
        for other_name in self.problem.cuboids:
            if str(other_name) == name:
                continue
            other_prim = self.world.scene.get_object(str(other_name))
            other_position, other_orientation = other_prim.get_world_pose()
            move_neighbor_states[str(other_name)] = (
                other_prim, np.asarray(other_position).copy(),
                np.asarray(other_orientation).copy(),
            )
        self.move_locked_neighbors = move_neighbor_states
        source, source_orientation = object_prim.get_world_pose()
        source = np.asarray(source, dtype=float)
        # The object may have been tipped by a preceding knock. Its live Isaac
        # center is therefore the only correct grasp target; reconstructing Z
        # from nominal upright height places the fingers above a lying cuboid.
        destination_xy = plan_xy_to_isaac(
            self.problem.pads[int(action["to"]["pad"])]
        )
        destination_pad = int(action["to"]["pad"])
        support_height = self.goal_stack_heights.get(destination_pad, 0.0)
        destination = np.array([
            destination_xy[0], destination_xy[1],
            SCENE_OFFSET[2] + support_height + 0.5 * cuboid.height,
        ])

        grasp_spec = action.get("grasp")
        grasp_offset = np.zeros(3)
        move_orientation = TOP_DOWN_QUAT_WXYZ
        if grasp_spec:
            ros_offset = grasp_spec["offset"]
            grasp_offset[:2] = plan_dir_to_isaac("+x")[:2] * float(ros_offset[0])
            grasp_offset[:2] += plan_dir_to_isaac("+y")[:2] * float(ros_offset[1])
            move_orientation = grasp_orientation(str(grasp_spec["closing_axis"]))
        elif action.get("after_clearance") and self.last_clearance_orientation is not None:
            # Pick the displaced object without changing wrist branches beside
            # the packed scene.  The clearance pose is already a valid
            # top-down grasp; normal orientations resume after this object has
            # been transported away.
            move_orientation = self.last_clearance_orientation

        print(
            f"[MOVE] {name}: measured_source={source.round(5).tolist()}; "
            f"source_quaternion={np.asarray(source_orientation).round(5).tolist()}; "
            f"dimensions={np.asarray(cuboid.dimensions).tolist()}; "
            f"destination={destination.round(5).tolist()}; "
            f"scene_offset={SCENE_OFFSET.tolist()}",
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

            # Grasp at the live rigid-body center. For an upright cuboid this
            # naturally scales with its height; after a knock it follows the
            # lower center of the cuboid lying on its side.
            grasp_target = source.copy() + grasp_offset
            approach = grasp_target.copy()
            approach[2] = max(0.25, grasp_target[2] + 0.20)

            stage_name = "approach"
            # This waypoint is a collision-free staging pose, not a contact
            # target.  RMPflow can settle 12--15 mm away near workspace-edge
            # wrist branches; the subsequent descent and proximity gate are
            # the checks that determine whether the grasp is valid.
            self.move_ee(approach, tolerance=0.020,
                         orientation=move_orientation)
            self.log_move_stage(name, "approach complete", object_prim, approach)

            stage_name = "grasp descent"
            # Contact and RMPflow's collision response commonly leave the
            # controlled right_gripper frame 15--22 mm from the mathematical
            # object center. Capture and lift are verified separately below,
            # so this gate should detect gross misses rather than reject contact.
            self.move_ee(grasp_target, tolerance=0.045,
                         orientation=move_orientation)
            self.log_move_stage(name, "grasp descent complete", object_prim, grasp_target)

            # Contact during descent can shove a thin cuboid away before the
            # fingers finish closing. Restore only that target's descent drift,
            # then establish the proximity-gated grasp constraint before the
            # close frames so closing cannot eject it.
            descended_position, _ = object_prim.get_world_pose()
            if np.linalg.norm(np.asarray(descended_position) - source) > 0.015:
                object_prim.set_world_pose(
                    position=source, orientation=source_orientation
                )
                object_prim.set_linear_velocity(np.zeros(3))
                object_prim.set_angular_velocity(np.zeros(3))
                print(
                    f"[GRASP TARGET RESTORE] {name}: "
                    f"drifted={np.asarray(descended_position).round(5).tolist()}; "
                    f"restored={source.round(5).tolist()}",
                    flush=True,
                )
            object_at_grasp, grasped_orientation = object_prim.get_world_pose()
            carried_offset = (
                self.controlled_ee_position() - np.asarray(object_at_grasp)
            )
            if np.linalg.norm(carried_offset) > 0.12:
                raise RuntimeError(
                    "tool is not close enough to establish a grasp: "
                    f"tool_object_offset={carried_offset.tolist()}"
                )
            self.carried_object = object_prim
            self.carried_offset = carried_offset.copy()
            self.carried_orientation = np.asarray(grasped_orientation).copy()
            print(
                f"[GRASP ATTACH] {name}: "
                f"tool_object_offset={carried_offset.round(5).tolist()}",
                flush=True,
            )

            stage_name = "close gripper"
            self.robot.gripper.close()
            for _ in range(120):
                self.step_world(render=not ARGS.headless)
            self.log_move_stage(name, "gripper closed", object_prim, grasp_target)
            finger_positions = np.asarray(
                self.robot.gripper.get_joint_positions(), dtype=float
            )
            if np.max(finger_positions) < 0.002:
                print(
                    f"[GRASP CONTACT FALLBACK] {name}: finger_joints="
                    f"{finger_positions.tolist()}; validating tool/object "
                    "proximity before simulated grasp attachment",
                    flush=True,
                )
            for _, (neighbor, position, quaternion) in move_neighbor_states.items():
                neighbor.set_world_pose(position=position, orientation=quaternion)
                neighbor.set_linear_velocity(np.zeros(3))
                neighbor.set_angular_velocity(np.zeros(3))

            # Define the lift from the intended grasp geometry.  Contact and
            # finger-close frames can displace the controlled frame laterally;
            # using that drifted pose as the lift target both reduces
            # reachability and sweeps the held object through its neighbors.
            lift = grasp_target.copy()
            lift[2] = max(0.25, grasp_target[2] + 0.20)

            stage_name = "lift"
            try:
                self.move_ee(lift, orientation=move_orientation)
            except RuntimeError as lift_error:
                # Sustained contact can leave RMPflow in a local minimum after
                # the fingers close.  The grasp has already passed its
                # proximity gate, so use the saved collision-free joint pose
                # as a deterministic upward escape while retaining the rigid
                # grasp constraint, then reset RMPflow for transport.
                print(
                    f"[MOVE LIFT RECOVERY] {name}: {lift_error}; "
                    "recovering to home posture with object attached",
                    flush=True,
                )
                self.recover_home(frames=240)
            self.log_move_stage(name, "lift complete", object_prim, lift)
            lifted_position, _ = object_prim.get_world_pose()
            if np.asarray(lifted_position)[2] < np.asarray(object_at_grasp)[2] + 0.05:
                raise RuntimeError(
                    "object did not follow the gripper during lift: "
                    f"before={np.asarray(object_at_grasp).tolist()}, "
                    f"after={np.asarray(lifted_position).tolist()}"
                )

            above_destination = destination + carried_offset
            above_destination[2] = lift[2]
            stage_name = "transport"
            self.move_ee(above_destination, tolerance=0.020,
                         orientation=move_orientation)
            self.log_move_stage(
                name, "transport complete", object_prim, above_destination
            )

            place_target = destination + carried_offset
            if action.get("remove_after_place", False):
                # Disposal pads are outside the table and the object is meant
                # to fall away.  Release from a safe height instead of asking
                # the controlled frame to converge through the floor/object.
                place_target[2] = max(0.12, place_target[2])
            stage_name = "place descent"
            self.move_ee(place_target, tolerance=0.018,
                         orientation=move_orientation)
            self.log_move_stage(name, "place descent complete", object_prim, place_target)

            stage_name = "release"
            self.carried_object = None
            self.carried_offset = None
            self.carried_orientation = None
            self.robot.gripper.open()
            for _ in range(45):
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
            self.move_ee(retreat, orientation=move_orientation)
            self.settle()
            self.log_move_stage(name, "move settled", object_prim, destination)
            if not action.get("remove_after_place", False):
                self.goal_stack_heights[destination_pad] = support_height + cuboid.height
        except Exception as exc:
            self.carried_object = None
            self.carried_offset = None
            self.carried_orientation = None
            raise RuntimeError(
                f"Move {name} failed during stage '{stage_name}': {exc}"
            ) from exc
        finally:
            self.move_locked_neighbors = None

    def execute_clearance_once(self, action, retry=False,
                               stop_displacement=None):
        # The optimal planner currently emits one displaced cuboid per knock.
        # Iterate for backward-compatible files that list a short stack.
        ros_direction = str(action["dir"])
        direction = plan_dir_to_isaac(ros_direction)
        orientation = knock_orientation(direction)
        print(
            f"[KNOCK] direction ROS {ros_direction} -> "
            f"Isaac vector {direction.tolist()}, "
            f"tool quaternion WXYZ {orientation.tolist()}",
            flush=True,
        )
        displacements = []
        for name in action["blocks"]:
            stage_name = "initialization"
            cuboid = self.problem.cuboids[str(name)]
            object_prim = self.world.scene.get_object(str(name))
            neighbor_states = {}
            for other_name in self.problem.cuboids:
                if str(other_name) == str(name):
                    continue
                other_prim = self.world.scene.get_object(str(other_name))
                other_position, other_orientation = other_prim.get_world_pose()
                neighbor_states[str(other_name)] = (
                    other_prim, np.asarray(other_position).copy(),
                    np.asarray(other_orientation).copy(),
                )
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
            # Stay on the outward half but sufficiently inside the footprint
            # that normal RMPflow lateral error cannot miss the top face.
            edge_xy = initial_position + direction * (0.35 * normal_size)
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

                # After an ineffective stroke, first recover vertically while
                # retaining the stroke orientation. Switching wrist branches
                # close to the tabletop can otherwise drive RMPflow through a
                # singular/colliding configuration before the next attempt.
                if retry:
                    stage_name = "retry recovery"
                    recovery = self.controlled_ee_position()
                    recovery[2] = max(0.28, recovery[2])
                    self.move_ee(recovery, tolerance=0.030,
                                 orientation=orientation)
                    self.log_knock_stage(
                        str(name), "retry recovery complete", object_prim, recovery
                    )

                # Translate top-down on the first attempt, then align the palm
                # while safely above height-dependent obstacles. A retry is
                # already aligned by the recovery motion.
                stage_name = "approach"
                approach_orientation = orientation if retry else TOP_DOWN_QUAT_WXYZ
                self.move_ee(above_edge, orientation=approach_orientation)
                self.log_knock_stage(str(name), "approach complete", object_prim, above_edge)

                if not retry:
                    stage_name = "palm alignment"
                    self.orient_ee_at_clearance(above_edge, orientation)
                    self.log_knock_stage(
                        str(name), "palm alignment complete", object_prim, above_edge
                    )

                stage_name = "pre-knock descent"
                # Contact intentionally prevents exact convergence of the
                # controlled frame to this biased waypoint.  The following
                # measured stroke/object displacement checks determine actual
                # success; this gate only rejects gross IK misses.
                self.move_ee(near_top, tolerance=0.050, orientation=orientation)
                achieved_contact = self.controlled_ee_position()
                # The contact-capable wrist branch can drift laterally during
                # the long descent. Compensate once from measured feedback so
                # the actual tool, not merely its command, remains over the
                # intended outer top region.
                lateral_error = achieved_contact[:2] - near_top[:2]
                direction_xy = direction[:2]
                lateral_error -= direction_xy * np.dot(lateral_error, direction_xy)
                if np.linalg.norm(lateral_error) > 0.005:
                    corrected_contact = near_top.copy()
                    corrected_contact[:2] -= lateral_error
                    self.move_ee(corrected_contact, tolerance=0.050,
                                 orientation=orientation)
                    achieved_contact = self.controlled_ee_position()
                    print(
                        f"[KNOCK CONTACT CORRECTION] {name}: "
                        f"xy_error={lateral_error.round(5).tolist()}; "
                        f"corrected_target={corrected_contact.round(5).tolist()}; "
                        f"achieved={achieved_contact.round(5).tolist()}",
                        flush=True,
                    )
                self.log_knock_stage(
                    str(name), "pre-knock complete", object_prim, near_top
                )
                pre_position, _ = object_prim.get_world_pose()
                raw_pre_displacement = float(np.dot(
                    np.asarray(pre_position) - initial_position, direction
                ))
                pre_displacement = max(0.0, raw_pre_displacement)
                projected_pre_position = (
                    initial_position + pre_displacement * direction
                )
                if np.linalg.norm(
                        np.asarray(pre_position) - projected_pre_position) > 0.003:
                    object_prim.set_world_pose(
                        position=projected_pre_position,
                        orientation=initial_orientation,
                    )
                    object_prim.set_linear_velocity(np.zeros(3))
                    object_prim.set_angular_velocity(np.zeros(3))
                    print(
                        f"[KNOCK DIRECTION PROJECTION] {name}: "
                        f"raw_position={np.asarray(pre_position).round(5).tolist()}; "
                        f"projected_position={projected_pre_position.round(5).tolist()}; "
                        f"outward_displacement={pre_displacement:.4f}",
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
                # Keep the diagonal shallow enough for sustained top contact.
                # Driving the controlled frame down to the floor on long
                # cuboids turns the intended clearance slide into a ground
                # collision before useful outward travel is achieved.
                stroke_end[2] = max(
                    SCENE_OFFSET[2] + 0.008,
                    top_z - max(0.010, 0.25 * cuboid.height),
                )
                print(
                    f"[KNOCK] {name} center={initial_position.tolist()} "
                    f"top_z={top_z:.4f} contact={achieved_contact.tolist()} "
                    f"stroke_end={stroke_end.tolist()}",
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
                if not pre_sufficient:
                    self.clearance_drag = {
                        "object": object_prim,
                        "object_start": np.asarray(contact_object_position).copy(),
                        "orientation": np.asarray(contact_object_orientation).copy(),
                        "ee_start": self.controlled_ee_position().copy(),
                        "direction": direction.copy(),
                    }
                    print(
                        f"[CLEARANCE CONTACT ATTACH] {name}: "
                        f"object_start={np.asarray(contact_object_position).round(5).tolist()}; "
                        f"ee_start={self.clearance_drag['ee_start'].round(5).tolist()}",
                        flush=True,
                    )
                    try:
                        self.follow_contact_stroke(
                            str(name), object_prim, achieved_contact, stroke_end,
                            orientation, object_start=initial_position,
                            direction=direction,
                            stop_displacement=stop_displacement,
                        )
                    finally:
                        object_prim.set_linear_velocity(np.zeros(3))
                        object_prim.set_angular_velocity(np.zeros(3))
                        self.clearance_drag = None
                self.log_knock_stage(
                    str(name), "contact stroke complete", object_prim, stroke_end
                )
                clearance_position, clearance_orientation = (
                    object_prim.get_world_pose()
                )
                clearance_position = np.asarray(clearance_position).copy()
                clearance_orientation = np.asarray(clearance_orientation).copy()

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

                self.robot.gripper.open()
                self.settle()
                for _, (neighbor, position, quaternion) in neighbor_states.items():
                    neighbor.set_world_pose(position=position, orientation=quaternion)
                    neighbor.set_linear_velocity(np.zeros(3))
                    neighbor.set_angular_velocity(np.zeros(3))
                for _ in range(5):
                    self.step_world(render=not ARGS.headless)
                object_prim.set_world_pose(
                    position=clearance_position,
                    orientation=clearance_orientation,
                )
                object_prim.set_linear_velocity(np.zeros(3))
                object_prim.set_angular_velocity(np.zeros(3))
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
        required = max(
            self.problem.finger_contact_length + self.problem.clearance_margin,
            ARGS.panda_finger_clearance + self.problem.clearance_margin,
        )
        max_attempts = ARGS.max_clearance_attempts
        if max_attempts is None:
            max_attempts = self.problem.max_clearance_attempts
        max_attempts = max(10, int(max_attempts))
        cumulative = 0.0
        attempts = []
        print(
            f"[CLEARANCE] required={required:.4f} m "
            f"(F={self.problem.finger_contact_length:.4f}, "
            f"finger={ARGS.panda_finger_clearance:.4f}, "
            f"margin={self.problem.clearance_margin:.4f}); max_attempts={max_attempts}",
            flush=True,
        )
        self.last_clearance_orientation = knock_orientation(
            plan_dir_to_isaac(str(action["dir"]))
        )
        for attempt in range(1, max_attempts + 1):
            displacement = self.execute_clearance_once(
                action, retry=attempt > 1,
                stop_displacement=max(0.0, required - cumulative),
            )
            cumulative += max(0.0, displacement)
            attempts.append(displacement)
            self.metrics["clearance_attempts"] += 1
            print(
                f"[CLEARANCE] attempt={attempt}/{max_attempts}; "
                f"increment={displacement:.4f}; cumulative={cumulative:.4f}; "
                f"required={required:.4f}", flush=True,
            )
            if cumulative + 1e-6 >= required:
                self.metrics["clearance_retries"] += attempt - 1
                return {"attempts": attempt, "increments": attempts,
                        "cumulative_displacement": cumulative, "required": required}
        self.metrics["clearance_retries"] += max_attempts - 1
        raise RuntimeError(
            f"Clearance failed after {max_attempts} attempts: "
            f"cumulative={cumulative:.4f} m, required={required:.4f} m"
        )

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
                    self.execute_move(action)
                    self.metrics["objects_removed"] += 1
                else:
                    self.metrics["clearance_actions"] += 1
                    record.update(self.execute_clearance(action))
                record["status"] = "success"
                self.metrics["actions_completed"] += 1
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = str(exc)
                self.metrics["status"] = "failed"
                self.metrics["failure"] = str(exc)
                raise
            finally:
                record["duration_sec"] = time.perf_counter() - action_started
                self.metrics["action_records"].append(record)
                self.metrics["execution_time_sec"] = time.perf_counter() - started
        print(f"[EXEC DONE] all {len(self.problem.actions)} high-level actions succeeded")
        self.metrics["status"] = "success"
        return self.metrics


def main():
    problem = load_problem(ARGS.yaml)
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=ARGS.physics_dt,
        rendering_dt=ARGS.physics_dt,
    )
    physics = world.get_physics_context()
    physics.set_solver_type("TGS")
    physics.enable_ccd(True)
    assets = get_assets_root_path()
    if assets:
        stage.add_reference_to_stage(assets + ROOM_USD, "/World/Room")
    else:
        world.scene.add_default_ground_plane(z_position=SCENE_OFFSET[2])

    robot = world.scene.add(
        Franka(
            prim_path="/World/Franka",
            name="franka",
            position=SCENE_OFFSET,
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, np.pi / 2.0])),
        )
    )
    tune_franka_drives()
    material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials/CuboidMaterial",
        static_friction=0.50,
        dynamic_friction=0.35,
        restitution=0.0,
    )
    for cuboid in problem.cuboids.values():
        xy = plan_xy_to_isaac(problem.pads[cuboid.initial_pad])
        z = SCENE_OFFSET[2] + cuboid.level * cuboid.height + 0.5 * cuboid.height
        color = (
            np.array([0.85, 0.15, 0.15]) if cuboid.name.lower().startswith("red")
            else np.array([0.15, 0.75, 0.20]) if cuboid.name.lower().startswith("green")
            else np.array([0.15, 0.30, 0.85])
        )
        world.scene.add(DynamicCuboid(
            prim_path=f"/World/Cuboids/{cuboid.name}",
            name=cuboid.name,
            position=np.array([xy[0], xy[1], z]),
            scale=np.asarray(cuboid.dimensions),
            size=1.0,
            color=color,
            mass=500.0 * np.prod(cuboid.dimensions),
            physics_material=material,
        ))

    world.reset()
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(16)
    robot.set_stabilization_threshold(0.0)
    robot.set_sleep_threshold(0.0)
    robot.gripper.open()
    executor = IsaacPlanExecutor(world, robot, problem)
    executor.settle()
    if not ARGS.no_wait:
        wait_for_keypress(world)
    failure = None
    try:
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


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[EXEC TERMINATED] {exc}", file=sys.stderr)
        raise
    finally:
        APP.close()
