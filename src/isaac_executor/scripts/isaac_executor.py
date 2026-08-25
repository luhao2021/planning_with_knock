#!/usr/bin/env python3
"""Execute a kp_planner_opt YAML plan entirely inside Isaac Sim.

The planner owns action selection and ordering. This program only realizes each
high-level Move or Knock with Isaac's Franka controllers and simulated contact.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import select
import sys
import threading

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
from isaacsim.core.utils.rotations import euler_angles_to_quat, rot_matrix_to_quat
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
    # ordinary top-down branch for both +/- Isaac Y directions.
    nonzero = np.flatnonzero(np.abs(palm_y) > 1e-9)
    if nonzero.size and palm_y[nonzero[0]] > 0.0:
        palm_y = -palm_y
    tool_z = np.array([0.0, 0.0, -1.0])
    palm_x = np.cross(palm_y, tool_z)
    # Matrix columns are right_gripper's local axes expressed in world space.
    rotation = np.column_stack((palm_x, palm_y, tool_z))
    return rot_matrix_to_quat(rotation)


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

    def step_world(self, render=True):
        self.world.step(render=render)

    def settle(self):
        for _ in range(ARGS.settle_steps):
            self.step_world(render=not ARGS.headless)

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

    def follow_contact_stroke(self, name, object_prim, start, end, orientation):
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
            target = (1.0 - progress) * start + progress * end
            for _ in range(ARGS.knock_contact_frames):
                controller_action = self.cartesian_controller.forward(
                    target_end_effector_position=target,
                    target_end_effector_orientation=orientation,
                )
                self.robot.apply_action(controller_action)
                self.step_world(render=not ARGS.headless)
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
        if error > 0.025:
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

        print(
            f"[MOVE] {name}: measured_source={source.round(5).tolist()}; "
            f"source_quaternion={np.asarray(source_orientation).round(5).tolist()}; "
            f"dimensions={np.asarray(cuboid.dimensions).tolist()}; "
            f"destination={destination.round(5).tolist()}; "
            f"scene_offset={SCENE_OFFSET.tolist()}",
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
            grasp_target = source.copy()
            approach = grasp_target.copy()
            approach[2] = max(0.25, grasp_target[2] + 0.20)

            stage_name = "approach"
            self.move_ee(approach, orientation=TOP_DOWN_QUAT_WXYZ)
            self.log_move_stage(name, "approach complete", object_prim, approach)

            stage_name = "grasp descent"
            self.move_ee(grasp_target, tolerance=0.015,
                         orientation=TOP_DOWN_QUAT_WXYZ)
            self.log_move_stage(name, "grasp descent complete", object_prim, grasp_target)

            stage_name = "close gripper"
            self.robot.gripper.close()
            for _ in range(120):
                self.step_world(render=not ARGS.headless)
            self.log_move_stage(name, "gripper closed", object_prim, grasp_target)
            finger_positions = np.asarray(
                self.robot.gripper.get_joint_positions(), dtype=float
            )
            if np.max(finger_positions) < 0.005:
                raise RuntimeError(
                    "gripper closed fully without capturing an object: "
                    f"finger_joints={finger_positions.tolist()}"
                )

            object_at_grasp, _ = object_prim.get_world_pose()
            grasp_offset = self.controlled_ee_position() - np.asarray(object_at_grasp)
            lift = self.controlled_ee_position().copy()
            lift[2] = max(0.25, lift[2] + 0.18)

            stage_name = "lift"
            self.move_ee(lift, orientation=TOP_DOWN_QUAT_WXYZ)
            self.log_move_stage(name, "lift complete", object_prim, lift)
            lifted_position, _ = object_prim.get_world_pose()
            if np.asarray(lifted_position)[2] < np.asarray(object_at_grasp)[2] + 0.05:
                raise RuntimeError(
                    "object did not follow the gripper during lift: "
                    f"before={np.asarray(object_at_grasp).tolist()}, "
                    f"after={np.asarray(lifted_position).tolist()}"
                )

            above_destination = destination + grasp_offset
            above_destination[2] = lift[2]
            stage_name = "transport"
            self.move_ee(above_destination, orientation=TOP_DOWN_QUAT_WXYZ)
            self.log_move_stage(
                name, "transport complete", object_prim, above_destination
            )

            place_target = destination + grasp_offset
            stage_name = "place descent"
            self.move_ee(place_target, tolerance=0.018,
                         orientation=TOP_DOWN_QUAT_WXYZ)
            self.log_move_stage(name, "place descent complete", object_prim, place_target)

            stage_name = "release"
            self.robot.gripper.open()
            for _ in range(90):
                self.step_world(render=not ARGS.headless)
            self.log_move_stage(name, "object released", object_prim, destination)

            retreat = place_target.copy()
            retreat[2] = max(0.25, place_target[2] + 0.18)
            stage_name = "retreat"
            self.move_ee(retreat, orientation=TOP_DOWN_QUAT_WXYZ)
            self.settle()
            self.log_move_stage(name, "move settled", object_prim, destination)
            self.goal_stack_heights[destination_pad] = support_height + cuboid.height
        except Exception as exc:
            raise RuntimeError(
                f"Move {name} failed during stage '{stage_name}': {exc}"
            ) from exc

    def execute_knock(self, action):
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
        for name in action["blocks"]:
            stage_name = "initialization"
            cuboid = self.problem.cuboids[str(name)]
            object_prim = self.world.scene.get_object(str(name))
            initial_position, _ = object_prim.get_world_pose()
            initial_position = np.asarray(initial_position, dtype=float)

            # Reproduce createTopKnockTask's physical intent: place the closed
            # fingers over the outward region of the top face, descend to a
            # small clearance, then move diagonally down and outward. Contact
            # at the top edge creates a tipping moment rather than a side push.
            planar_size = max(cuboid.dimensions[0:2])
            top_z = initial_position[2] + 0.5 * cuboid.height
            # Stay on the outward half but sufficiently inside the footprint
            # that normal RMPflow lateral error cannot miss the top face.
            edge_xy = initial_position + direction * (0.20 * planar_size)
            above_edge = edge_xy.copy()
            above_edge[2] = top_z + max(0.10, cuboid.height)
            near_top = edge_xy.copy()
            # This must remain a reachable free-space waypoint. Asking the
            # position controller to converge below a rigid top surface makes
            # contact look like an IK timeout and can preload/tip the cuboid
            # before the intentional diagonal stroke begins.
            near_top[2] = top_z + ARGS.knock_precontact_clearance

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

                # Translate top-down, then align the palm while safely above
                # height-dependent obstacles.
                stage_name = "approach"
                self.move_ee(above_edge, orientation=TOP_DOWN_QUAT_WXYZ)
                self.log_knock_stage(str(name), "approach complete", object_prim, above_edge)

                stage_name = "palm alignment"
                self.orient_ee_at_clearance(above_edge, orientation)
                self.log_knock_stage(
                    str(name), "palm alignment complete", object_prim, above_edge
                )

                stage_name = "pre-knock descent"
                self.move_ee(near_top, tolerance=0.020, orientation=orientation)
                achieved_contact = self.controlled_ee_position()
                self.log_knock_stage(
                    str(name), "pre-knock complete", object_prim, near_top
                )

                outward_travel = 2.0 * planar_size
                stroke_end = achieved_contact + direction * outward_travel
                stroke_end[2] = max(
                    SCENE_OFFSET[2] + 0.008,
                    achieved_contact[2]
                    - outward_travel * np.tan(np.deg2rad(ARGS.knock_angle_deg)),
                )
                retreat = stroke_end.copy()
                retreat[2] = top_z + max(0.12, cuboid.height)
                print(
                    f"[KNOCK] {name} center={initial_position.tolist()} "
                    f"top_z={top_z:.4f} contact={achieved_contact.tolist()} "
                    f"stroke_end={stroke_end.tolist()}",
                    flush=True,
                )

                stage_name = "pre-contact hold"
                for _ in range(max(30, ARGS.settle_steps)):
                    controller_action = self.cartesian_controller.forward(
                        target_end_effector_position=achieved_contact,
                        target_end_effector_orientation=orientation,
                    )
                    self.robot.apply_action(controller_action)
                    self.step_world(render=not ARGS.headless)
                self.log_knock_stage(
                    str(name), "contact stroke starting", object_prim, achieved_contact
                )

                stage_name = "contact stroke"
                self.follow_contact_stroke(
                    str(name), object_prim, achieved_contact, stroke_end, orientation
                )
                self.log_knock_stage(
                    str(name), "contact stroke complete", object_prim, stroke_end
                )

                stage_name = "retreat"
                self.move_ee(retreat, orientation=orientation)
                self.log_knock_stage(str(name), "retreat complete", object_prim, retreat)
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
            minimum = 0.35 * planar_size
            if displacement < minimum:
                raise RuntimeError(
                    f"Knock failed for {name}: displacement={displacement:.3f} m"
                )

    def run(self):
        for index, action in enumerate(self.problem.actions, start=1):
            kind = action["type"]
            label = action.get("block", ",".join(action.get("blocks", [])))
            print(f"[EXEC] {index}/{len(self.problem.actions)} {kind} {label}", flush=True)
            if kind == "Move":
                self.execute_move(action)
            else:
                self.execute_knock(action)
        print(f"[EXEC DONE] all {len(self.problem.actions)} high-level actions succeeded")


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
        static_friction=1.2,
        dynamic_friction=1.0,
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
    executor.run()
    if not ARGS.headless:
        while APP.is_running():
            world.step(render=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[EXEC ERROR] {exc}", file=sys.stderr)
        raise
    finally:
        APP.close()
