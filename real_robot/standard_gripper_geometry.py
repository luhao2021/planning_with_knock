"""Nominal G1 finger geometry; no hardware access or calibration claims.

Source: xArm-Developer/xarm_ros2, humble, xarm_description/urdf/gripper/
xarm_gripper.urdf.xacro and meshes/gripper/xarm/left_finger.stl.
The mirrored fingers stay parallel: the finger joint cancels the drive rotation.
Local model copies are under models/xarm_gripper_geometry/.
"""
import math

DRIVE_Z_MM = 59.098
FINGER_JOINT_Y_MM = 35.465
FINGER_JOINT_Z_MM = 42.039
FINGER_MESH_MAX_Z_MM = 61.00320443511009
CLOSED_JOINT_RAD = 0.85


def tip_distance_from_flange_mm(q):
    return (DRIVE_Z_MM + FINGER_JOINT_Y_MM*math.sin(q)
            + FINGER_JOINT_Z_MM*math.cos(q) + FINGER_MESH_MAX_Z_MM)


def tip_extension_mm(tcp_offset_z_mm, *, closed=False):
    # Closed q is the official URDF limit. For Pick, use the maximum over
    # the entire closing travel so the tips retain at least the requested
    # clearance, even when the aperture/pulse mapping is approximate.
    q = CLOSED_JOINT_RAD if closed else math.atan2(FINGER_JOINT_Y_MM, FINGER_JOINT_Z_MM)
    return tip_distance_from_flange_mm(q) - tcp_offset_z_mm
