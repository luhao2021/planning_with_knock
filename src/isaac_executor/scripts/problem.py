#!/usr/bin/env python3
"""Validation and geometry helpers for knock-pick Isaac problems.

This module intentionally has no Isaac Sim imports, so problem files can be
validated with ordinary Python before starting the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

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

    @property
    def height(self) -> float:
        return self.dimensions[2]


@dataclass(frozen=True)
class Problem:
    pads: Dict[int, Tuple[float, float]]
    cuboids: Dict[str, Cuboid]
    actions: list


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
        if abs(dims[0] - dims[1]) > 1e-8:
            raise ValueError(f"{name}: cuboid base must have equal x/y side lengths")
        height_multiple = dims[2] / dims[0]
        if abs(height_multiple - round(height_multiple)) > 1e-8:
            raise ValueError(
                f"{name}: height must be an integer multiple of side length"
            )
        pad = int(item["pad"])
        if pad not in pads:
            raise ValueError(f"{name}: unknown initial pad {pad}")
        cuboids[name] = Cuboid(name, dims, pad, int(item.get("height", 0)))

    for index, action in enumerate(document["actions"]):
        kind = str(action.get("type", ""))
        if kind == "Smash":
            kind = "Knock"
            action["type"] = kind
        if kind not in ("Move", "Knock"):
            raise ValueError(f"action {index}: unsupported type {kind!r}")
        names = [action.get("block")] if kind == "Move" else action.get("blocks", [])
        if not names or any(name not in cuboids for name in names):
            raise ValueError(f"action {index}: references an unknown cuboid")
        if kind == "Move" and int(action["to"]["pad"]) not in pads:
            raise ValueError(f"action {index}: unknown destination pad")
        if kind == "Knock" and action.get("dir") not in ("+x", "-x", "+y", "-y"):
            raise ValueError(f"action {index}: invalid knock direction")
    return Problem(pads, cuboids, list(document["actions"]))


def center_height(cuboid: Cuboid, support_height: float = 0.0) -> float:
    """World Z of the cuboid center when resting on a support."""
    return support_height + 0.5 * cuboid.height


def grasp_height(cuboid: Cuboid, support_height: float = 0.0) -> float:
    """Side-grasp center height; clamped away from the tabletop."""
    return support_height + max(0.012, 0.55 * cuboid.height)


def contact_height(cuboid: Cuboid, support_height: float = 0.0) -> float:
    """Stable knock height below the center of mass for variable-height objects."""
    return support_height + max(0.010, 0.40 * cuboid.height)
