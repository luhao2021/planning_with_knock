#!/usr/bin/env python3
"""Convert pretrained model 6D instance poses to xArm-base layouts and abstract plans.

This file performs frame conversion and planner I/O only. It never estimates a
pose, connects to the xArm, or commands motion. The model JSON must contain one
independently predicted pose per physical object instance.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cuboid_knock_pick_solvers import (GeometryInstance, Rect, physical_plan,
                                        plot_instance, plot_solution_overlay,
                                        plot_solution_sequence, preprocess, solve_ilp,
                                        verify_solution, _convex_intersection, _polygon_area)

from object_catalog import DIMENSIONS
# ROS static_transform_publisher: x y z qx qy qz qw parent child.
BASE_TO_OPTICAL_XYZ = [1.0494764484901562, -0.05633311429031207, 0.6130679657159153]
BASE_TO_OPTICAL_XYZW = [-0.655593643359166, -0.6289576561434944,
                         0.28533779326690006, 0.3052729685037842]
OPTICAL_TO_CAMERA_LINK_XYZW = [0.5, -0.5, 0.5, 0.5]


def transform(xyz: list[float], xyzw: list[float]) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=float)
    quat = np.asarray(xyzw, dtype=float)
    if xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(quat).all():
        raise ValueError('invalid static transform')
    if not np.isclose(np.linalg.norm(quat), 1.0, atol=0.01):
        raise ValueError('static quaternion is not unit length')
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat / np.linalg.norm(quat)).as_matrix()
    T[:3, 3] = xyz
    return T


def base_to_optical() -> np.ndarray:
    return transform(BASE_TO_OPTICAL_XYZ, BASE_TO_OPTICAL_XYZW)


def convert(records: list[dict], *, min_score: float | None = None, require_flat: bool = True) -> list[dict]:
    if not records:
        raise ValueError('pose model returned no instances')
    T_base_camera = base_to_optical()
    seen = set()
    output = []
    for record in records:
        kind = str(record['class']).lower()
        if kind not in DIMENSIONS:
            raise ValueError(f'unsupported object class {kind}')
        uid = str(record['instance_id'])
        if not uid or (kind, uid) in seen:
            raise ValueError(f'missing or duplicate instance ID {uid!r}')
        seen.add((kind, uid))
        score = float(record['score'])
        if not np.isfinite(score) or (min_score is not None and score < min_score):
            raise ValueError(f'{uid}: low/invalid model confidence {score}')
        T_camera_object = np.asarray(record['T_camera_object'], dtype=float)
        if T_camera_object.shape != (4, 4) or not np.isfinite(T_camera_object).all():
            raise ValueError(f'{uid}: invalid 4x4 camera-frame pose')
        if not np.allclose(T_camera_object[3], [0, 0, 0, 1], atol=1e-5):
            raise ValueError(f'{uid}: invalid homogeneous bottom row')
        if not np.allclose(T_camera_object[:3, :3].T @ T_camera_object[:3, :3], np.eye(3), atol=0.03):
            raise ValueError(f'{uid}: non-rigid orientation')
        T_base_object = T_base_camera @ T_camera_object
        x_axis = T_base_object[:3, 0]
        z_axis = T_base_object[:3, 2]
        if require_flat and (abs(float(z_axis[2])) < 0.95 or np.linalg.norm(x_axis[:2]) < 0.95):
            raise ValueError(f'{uid}: object is not flat/upright for equal-height planner')
        yaw = math.atan2(float(x_axis[1]), float(x_axis[0])) % math.pi
        output.append(dict(name=f'{kind}_{uid}', instance_id=uid, kind=kind,
                           dimensions_m=DIMENSIONS[kind], score=score,
                           base_center_m=T_base_object[:3, 3].tolist(),
                           yaw_rad=yaw, T_base_object=T_base_object.tolist()))
    # Preserve the mask/prompt IDs within this snapshot. Cross-frame identity
    # requires explicit SAM 2 tracking before a real executor is enabled.
    return output


def checked_base_pose_offset_mm(offset_mm):
    try:
        offset = np.asarray(offset_mm, dtype=float)
    except (TypeError, ValueError):
        raise ValueError('base_pose_offset_mm must contain three finite numbers') from None
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('base_pose_offset_mm must contain three finite numbers')
    return offset.tolist()


def apply_base_pose_offset(objects: list[dict], offset_mm) -> list[dict]:
    """Apply one manual translation in link_base, after camera-frame conversion.

    Keep uncorrected coordinates for audit; never accumulate offsets on replay.
    No change is made to yaw, dimensions, or programmed robot/drop coordinates.
    """
    from copy import deepcopy
    offset_mm = checked_base_pose_offset_mm(offset_mm)
    delta = np.asarray(offset_mm)/1000
    output = deepcopy(objects)
    for obj in output:
        if 'base_pose_offset_mm' in obj:
            raise ValueError(f'{obj["name"]}: base-pose offset already applied; use uncorrected poses')
        pose = np.asarray(obj['T_base_object'], dtype=float).copy()
        centre = np.asarray(obj['base_center_m'], dtype=float)
        if pose.shape != (4,4) or centre.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(centre).all():
            raise ValueError(f'{obj["name"]}: invalid base pose for translation correction')
        if not np.allclose(pose[:3,3], centre, atol=1e-8, rtol=0):
            raise ValueError(f'{obj["name"]}: base centre and transform translation disagree')
        obj['uncorrected_base_center_m'] = centre.tolist()
        obj['T_base_object_uncorrected'] = pose.tolist()
        pose[:3,3] += delta
        obj['T_base_object'] = pose.tolist()
        obj['base_center_m'] = (centre+delta).tolist()
        obj['base_pose_offset_mm'] = offset_mm.copy()
    return output


def correct_pose_overlaps(objects: list[dict], max_shift_m: float) -> tuple[list[dict], list[dict]]:
    """Adjust centres, never dimensions/yaws, for tiny mutually inconsistent poses.

    Choose the minimum-penetration separating axis for each raw overlapping pair,
    then minimize total squared XY displacement subject to those separations.
    Raw model poses remain in the output alongside the adjusted planning poses.
    """
    if max_shift_m <= 0:
        return objects, []
    from copy import deepcopy
    raw = deepcopy(objects)
    rects = []
    for obj in raw:
        length, width, _ = obj['dimensions_m']
        x, y = obj['base_center_m'][:2]
        rects.append(Rect(obj['name'], x-length/2, x+length/2,
                          y-width/2, y+width/2, obj['yaw_rad']))
    constraints = []
    overlaps = []
    for i, a in enumerate(rects):
        for j in range(i+1, len(rects)):
            b = rects[j]
            area = _polygon_area(_convex_intersection(a.corners, b.corners))
            if area <= 1e-9:
                continue
            candidates = []
            for r in (a, b):
                for axis in ((math.cos(r.yaw), math.sin(r.yaw)),
                             (-math.sin(r.yaw), math.cos(r.yaw))):
                    axis = np.asarray(axis)
                    av, bv = a.corners @ axis, b.corners @ axis
                    penetration = min(av.max(), bv.max()) - max(av.min(), bv.min())
                    candidates.append((penetration, axis))
            penetration, axis = min(candidates, key=lambda item: item[0])
            if np.dot(b.center-a.center, axis) < 0:
                axis = -axis
            row = np.zeros(2*len(raw))
            row[2*j:2*j+2] = axis
            row[2*i:2*i+2] = -axis
            required = penetration + 1e-5
            constraints.append({'type': 'ineq', 'fun': lambda x, row=row, required=required: row @ x - required,
                                'jac': lambda x, row=row: row})
            overlaps.append({'pair': [a.name, b.name], 'raw_overlap_area_mm2': area*1e6,
                             'raw_penetration_mm': penetration*1000})
    if not constraints:
        return raw, []
    # Enforce each approved displacement bound while solving, rather than
    # rejecting an unconstrained optimum when a bounded solution may exist.
    for i in range(len(raw)):
        def bound_jac(x,i=i):
            gradient=np.zeros_like(x);gradient[2*i:2*i+2]=-2*x[2*i:2*i+2]
            return gradient
        constraints.append({'type':'ineq',
            'fun':lambda x,i=i:max_shift_m**2-float(x[2*i:2*i+2]@x[2*i:2*i+2]),
            'jac':bound_jac})
    result = minimize(lambda x: 0.5*float(x@x), np.zeros(2*len(raw)),
                      jac=lambda x: x, constraints=constraints, method='SLSQP',
                      options={'ftol': 1e-12, 'maxiter': 200})
    if not result.success:
        raise ValueError(f'pose-overlap correction failed: {result.message}')
    corrected = deepcopy(raw)
    for i, obj in enumerate(corrected):
        delta = result.x[2*i:2*i+2]
        if np.linalg.norm(delta) > max_shift_m + 1e-8:
            raise ValueError(f'{obj["name"]}: required pose correction exceeds limit')
        obj['base_center_m'][:2] = (np.asarray(obj['base_center_m'][:2]) + delta).tolist()
        T = np.asarray(obj['T_base_object'])
        T[:2, 3] += delta
        obj['T_base_object'] = T.tolist()
        obj['pose_correction_xy_mm'] = (delta*1000).tolist()
    return corrected, overlaps


def solve_scene(objects: list[dict], footprint_shrink_m: float = 0.0, jaw_opening_m: float = 0.080,
                *, finger_thickness_m: float = 0.015, planning_gripper=None, observed_objects=None):
    if not 0.0 <= footprint_shrink_m <= 0.005:
        raise ValueError("footprint shrink must be between 0 and 5 mm")
    if not 0.0 < jaw_opening_m <= 0.12:
        raise ValueError("invalid jaw opening")
    if not np.isfinite(finger_thickness_m) or finger_thickness_m <= 0:
        raise ValueError('finger thickness must be finite and positive')
    rects = []
    for obj in objects:
        length, width, _ = obj['dimensions_m']
        length -= footprint_shrink_m
        width -= footprint_shrink_m
        x, y = obj['base_center_m'][:2]
        rects.append(Rect(obj['name'], x-length/2, x+length/2,
                          y-width/2, y+width/2, obj['yaw_rad']))
    inst = GeometryInstance('learned_pose_scene', rects, F=0.030, D=jaw_opening_m,
                            finger_clearance=finger_thickness_m,
                            metadata={'plot_padding_m': '0.035',
                                      'hide_names_in_solution_figures': 'true'})
    if planning_gripper is None:
        rule = preprocess(inst)
        initial_picks=None
    else:
        from conservative_pick_planning import conservative_rule
        rule,initial_picks=conservative_rule(inst,objects if observed_objects is None else observed_objects,planning_gripper)
    result = solve_ilp(rule, time_limit=30)
    if initial_picks is not None:result.extra['initial_planning_pick_actions']=initial_picks
    if result.selected_actions is None or result.order is None:
        raise RuntimeError(f'planner status: {result.status}')
    ok, reason = verify_solution(rule, result.selected_actions, result.order)
    if not ok:
        raise RuntimeError(f'planner verification failed: {reason}')
    return rule, result


def plan(objects: list[dict], footprint_shrink_m: float = 0.0, jaw_opening_m: float = 0.080) -> tuple[list[str], int, str]:
    rule, result = solve_scene(objects, footprint_shrink_m, jaw_opening_m)
    return physical_plan(rule, result.selected_actions, result.order), int(result.optimum), result.status

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--poses', type=Path, required=True,
                        help='Pretrained model output: {instances:[{class,instance_id,score,T_camera_object}]}')
    parser.add_argument('--expected-protex', type=int, default=4)
    parser.add_argument('--footprint-shrink-mm', type=float, default=0.0,
                        help='Total reduction of each planning XY side; raw measured dimensions and 6D poses stay unchanged')
    parser.add_argument('--jaw-opening-mm', type=float, default=80.0,
                        help='Usable opening D in solver; default 80 mm is conservative versus approximate 85 mm hardware measurement')
    parser.add_argument('--max-pose-correction-mm', type=float, default=0.0,
                        help='Maximum per-object XY centre shift to resolve estimated overlaps; dimensions never change')
    parser.add_argument('--min-score', type=float, default=None,
                        help='Optional threshold for raw FoundationPose score logits; calibrate first')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.poses.read_text())
    objects = convert(source['instances'], min_score=args.min_score)
    if sum(obj['kind'] == 'protex' for obj in objects) != args.expected_protex:
        raise ValueError('expected Protex instance count does not match model output')
    planning_objects, raw_overlaps = correct_pose_overlaps(objects, args.max_pose_correction_mm / 1000.0)
    shrink_m = args.footprint_shrink_mm / 1000.0
    rule, result = solve_scene(planning_objects, shrink_m, args.jaw_opening_mm / 1000.0)
    lines = physical_plan(rule, result.selected_actions, result.order)
    clearances, solver_status = int(result.optimum), result.status
    args.out.mkdir(parents=True, exist_ok=True)
    figure_paths = {
        'benchmark_layout': 'layouts/learned_pose_scene.png',
        'solution_overlay': 'solution_overlays/learned_pose_scene_ilp_highs.png',
        'solution_sequence': 'solution_sequences/learned_pose_scene_ilp_highs.png',
    }
    for relative in figure_paths.values():
        (args.out / relative).parent.mkdir(parents=True, exist_ok=True)
    raw_rects = []
    for obj in objects:
        length, width, _ = obj['dimensions_m']
        x, y = obj['base_center_m'][:2]
        raw_rects.append(Rect(obj['name'], x-length/2, x+length/2,
                              y-width/2, y+width/2, obj['yaw_rad']))
    raw_inst = GeometryInstance('raw_learned_pose_scene', raw_rects, F=0.030, D=0.080,
                                metadata={'plot_padding_m': '0.035'})
    plot_instance(raw_inst, args.out / 'raw_estimated_layout.png')
    plot_instance(rule.geometry, args.out / figure_paths['benchmark_layout'])
    plot_solution_overlay(rule, result.selected_actions, result.order,
                          args.out / figure_paths['solution_overlay'], 'ILP/HiGHS')
    plot_solution_sequence(rule, result.selected_actions, result.order,
                           args.out / figure_paths['solution_sequence'], 'ILP/HiGHS')
    payload = dict(status='model-pose-to-abstract-plan-only',
                   T_base_camera_color_optical_frame=base_to_optical().tolist(),
                   objects=objects, planning_objects=planning_objects,
                   raw_pose_overlaps=raw_overlaps, max_pose_correction_mm=args.max_pose_correction_mm,
                   footprint_shrink_total_m=shrink_m, jaw_opening_m=args.jaw_opening_mm / 1000.0,
                   planning_dimensions_m={kind: [round(d[0]-shrink_m, 6), round(d[1]-shrink_m, 6), d[2]]
                                          for kind,d in DIMENSIONS.items()},
                   solver_status=solver_status, plan_verified=True,
                   optimal_clearances=clearances, plan=lines, figures=figure_paths,
                   action_details=[{'step': step, 'target': rule.names[index],
                                    'kind': result.selected_actions[index].kind,
                                    'witness': result.selected_actions[index].witness,
                                    'blockers': [rule.names[j] for j in sorted(result.selected_actions[index].blockers)]}
                                   for step, index in enumerate(result.order, 1)],
                   source_pose_file=str(args.poses))
    (args.out / 'layout_and_plan.json').write_text(json.dumps(payload, indent=2))
    (args.out / 'plan.txt').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    fig, ax = plt.subplots(figsize=(9, 7))
    all_corners = []
    for obj in planning_objects:
        length, width, _ = obj['dimensions_m']
        length -= shrink_m
        width -= shrink_m
        x, y = obj['base_center_m'][:2]
        r = Rect(obj['name'], x-length/2, x+length/2,
                 y-width/2, y+width/2, obj['yaw_rad'])
        all_corners.extend(r.corners)
        ax.add_patch(Polygon(r.corners, fill=False, linewidth=2))
        ax.text(x, y, obj['name'], ha='center', va='center', fontsize=8)
    bounds = np.asarray(all_corners)
    ax.set_xlim(float(bounds[:,0].min()-0.035), float(bounds[:,0].max()+0.035))
    ax.set_ylim(float(bounds[:,1].min()-0.035), float(bounds[:,1].max()+0.035))
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_xlabel('x in link_base (m)'); ax.set_ylabel('y in link_base (m)')
    ax.set_title('Pretrained 6D poses projected to tabletop planner')
    fig.savefig(args.out / 'layout.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
