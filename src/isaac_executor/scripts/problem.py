#!/usr/bin/env python3
"""Validation and geometry helpers for knock-pick Isaac problems.

This module intentionally has no Isaac Sim imports, so problem files can be
validated with ordinary Python before starting the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


ROS_PLAN_DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


def ros_vector_to_isaac(vector: Tuple[float, float, float]):
    """Rotate a ROS plan vector with scene_isaacsim.py's Rz(+90) map."""
    x_ros, y_ros, z_ros = vector
    return (-y_ros, x_ros, z_ros)


def knock_direction_to_isaac(direction: str):
    if direction not in ROS_PLAN_DIRECTIONS:
        raise ValueError(f"invalid ROS knock direction: {direction!r}")
    return ros_vector_to_isaac(ROS_PLAN_DIRECTIONS[direction])


@dataclass(frozen=True)
class Cuboid:
    name: str
    dimensions: Tuple[float, float, float]
    initial_pad: int
    level: int
    plan_center: Optional[Tuple[float, float]] = None
    yaw_rad: float = 0.0
    sort_color: str = "red"

    @property
    def height(self) -> float:
        return self.dimensions[2]


@dataclass(frozen=True)
class Problem:
    pads: Dict[int, Tuple[float, float]]
    cuboids: Dict[str, Cuboid]
    actions: list
    finger_contact_length: float = 0.0
    contact_gap: float = 0.0
    clearance_margin: float = 0.005
    max_clearance_attempts: int = 10
    finger_side_contact_width: float = 0.040
    sorting_chutes: Optional[Dict[str, int]] = None
    table_bounds_isaac: Optional[Tuple[float, float, float, float]] = None
    # Optional provenance/policy marker embedded by benchmark plan exporters.
    # Keeping this in the parsed problem lets the executor enable recovery
    # behavior for a baseline without changing nominal proposed-method runs.
    policy: str = "planned"


def load_problem(path: str | Path) -> Problem:
    with Path(path).open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict):
        raise ValueError("problem document must be a mapping")
    for key in ("pads", "initial", "actions"):
        if key not in document:
            raise ValueError(f"problem is missing required key: {key}")

    pads = {
        int(key): (float(value["x"]), float(value["y"]))
        for key, value in document["pads"].items()
    }
    geometry = document.get("objects", {})
    legacy_size = float(document.get("z_increment", 0.05))
    cuboids: Dict[str, Cuboid] = {}
    for item in document["initial"]:
        name = str(item["name"])
        dimensions = geometry.get(name, {}).get(
            "dimensions", [legacy_size, legacy_size, legacy_size]
        )
        if len(dimensions) != 3:
            raise ValueError(f"{name}: dimensions must have exactly three values")
        dims = tuple(float(value) for value in dimensions)
        if any(value <= 0.0 for value in dims):
            raise ValueError(f"{name}: dimensions must be positive")
        pad = int(item["pad"])
        if pad not in pads:
            raise ValueError(f"{name}: unknown initial pad {pad}")
        center = geometry.get(name, {}).get("plan_center")
        if center is not None:
            if len(center) != 2:
                raise ValueError(f"{name}: plan_center must have two values")
            center = (float(center[0]), float(center[1]))
        cuboids[name] = Cuboid(
            name, dims, pad, int(item.get("height", 0)), center,
            float(item.get("yaw_rad", geometry.get(name, {}).get("yaw_rad", 0.0))),
            str(geometry.get(name, {}).get("sort_color", "red")),
        )

    for index, action in enumerate(document["actions"]):
        kind = str(action.get("type", ""))
        if kind in ("Smash", "Knock"):
            kind = "Clearance"
            action["type"] = kind
        if kind not in ("Move", "Clearance"):
            raise ValueError(f"action {index}: unsupported type {kind!r}")
        names = [action.get("block")] if kind == "Move" else action.get("blocks", [])
        if not names or any(name not in cuboids for name in names):
            raise ValueError(f"action {index}: references an unknown cuboid")
        if kind == "Move" and int(action["to"]["pad"]) not in pads:
            raise ValueError(f"action {index}: unknown destination pad")
        if kind == "Clearance" and action.get("dir") not in ("+x", "-x", "+y", "-y"):
            raise ValueError(f"action {index}: invalid clearance direction")
        if kind == "Move" and "grasp" in action:
            grasp = action["grasp"]
            if grasp.get("closing_axis") not in ("x", "y"):
                raise ValueError(f"action {index}: invalid grasp closing axis")
            if len(grasp.get("offset", [])) != 2:
                raise ValueError(f"action {index}: grasp offset must have two values")

    execution = document.get("execution", {})
    sorting = document.get("sorting", {})
    chute_pads = {
        str(color): int(data["pad"])
        for color, data in sorting.get("chutes", {}).items()
    }
    for color, pad in chute_pads.items():
        if color not in ("red", "blue"):
            raise ValueError(f"unsupported sorting color: {color!r}")
        if pad not in pads:
            raise ValueError(f"sorting chute {color}: unknown pad {pad}")
    table = sorting.get("table_bounds_isaac")
    table_bounds = None
    if table is not None:
        table_bounds = tuple(float(table[key]) for key in (
            "xmin", "xmax", "ymin", "ymax"
        ))
    attempts = int(execution.get("max_clearance_attempts", 10))
    if attempts < 1:
        raise ValueError("max_clearance_attempts must be positive")
    return Problem(
        pads, cuboids, list(document["actions"]),
        finger_contact_length=float(document.get("finger_contact_length", 0.0)),
        contact_gap=float(document.get("contact_gap", 0.0)),
        clearance_margin=float(execution.get("clearance_margin", 0.005)),
        max_clearance_attempts=attempts,
        finger_side_contact_width=float(
            execution.get("finger_side_contact_width", 0.040)
        ),
        sorting_chutes=chute_pads or None,
        table_bounds_isaac=table_bounds,
        policy=str(document.get("policy", "planned")),
    )


def center_height(cuboid: Cuboid, support_height: float = 0.0) -> float:
    """World Z of the cuboid center when resting on a support."""
    return support_height + 0.5 * cuboid.height


def grasp_height(cuboid: Cuboid, support_height: float = 0.0) -> float:
    """Side-grasp center height; clamped away from the tabletop."""
    return support_height + max(0.012, 0.55 * cuboid.height)


def contact_height(cuboid: Cuboid, support_height: float = 0.0) -> float:
    """Stable knock height below the center of mass for variable-height objects."""
    return support_height + max(0.010, 0.40 * cuboid.height)
