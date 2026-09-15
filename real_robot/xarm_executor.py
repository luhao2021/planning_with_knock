#!/usr/bin/env python3
"""Read-only xArm action preview, with an explicitly gated single-action executor.

No network access or xArm import occurs in preview mode. The physical backend is
only entered with --execute, a complete measured configuration, and an explicit
review acknowledgement. A Knock stops after one bounded stroke; reobserve and
replan before the mandatory PickPlace or the next high-level action.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsafePlan(ValueError):
    pass


def finite(value: Any, label: str) -> float:
    if value is None or not math.isfinite(float(value)):
        raise UnsafePlan(f"{label} must be a calibrated finite number")
    return float(value)


def vec(value: Any, label: str, size: int = 3) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise UnsafePlan(f"{label} needs {size} calibrated values")
    return [finite(v, label) for v in value]


def add(a, b):
    return [x+y for x, y in zip(a, b)]


@dataclass(frozen=True)
class Stage:
    phase: str
    tcp_mm_deg: tuple[float, float, float, float, float, float] | None = None
    gripper_pulse: int | None = None
    speed_mm_s: float = 20.0
    note: str = ''

    def as_dict(self):
        return dict(phase=self.phase, tcp_mm_deg=self.tcp_mm_deg,
                    gripper_pulse=self.gripper_pulse, speed_mm_s=self.speed_mm_s,
                    note=self.note)


def checked_config(cfg: dict) -> dict:
    """Require measured robot-frame geometry; no guess from a photograph."""
    if cfg.get('robot_ip') != '192.168.1.231':
        raise UnsafePlan('robot_ip must match the reviewed xArm7 address')
    if cfg.get('frame') != 'link_base':
        raise UnsafePlan('all positions must be in link_base')
    out = dict(cfg)
    out['workspace_min_mm'] = vec(cfg.get('workspace_min_mm'), 'workspace_min_mm')
    out['workspace_max_mm'] = vec(cfg.get('workspace_max_mm'), 'workspace_max_mm')
    if any(a >= b for a, b in zip(out['workspace_min_mm'], out['workspace_max_mm'])):
        raise UnsafePlan('invalid workspace bounds')
    for key in ('transit_z_mm', 'approach_z_mm', 'lift_z_mm', 'table_z_mm',
                'tcp_contact_z_offset_mm', 'tool_closing_axis_yaw_offset_deg',
                'usable_jaw_opening_mm', 'gripper_speed_setting', 'knock_stroke_mm',
                'knock_precontact_clearance_mm', 'knock_max_drop_mm'):
        out[key] = finite(cfg.get(key), key)
    out['tool_roll_pitch_deg'] = vec(cfg.get('tool_roll_pitch_deg'), 'tool_roll_pitch_deg', 2)
    out['tcp_offset_mm_deg'] = vec(cfg.get('tcp_offset_mm_deg'), 'tcp_offset_mm_deg', 6)
    calibration=cfg.get('gripper_aperture_calibration')
    if not isinstance(calibration,dict):
        raise UnsafePlan('measured gripper aperture-to-pulse calibration is required')
    out['gripper_aperture_calibration']={
        key: finite(calibration.get(key), 'gripper_aperture_calibration.'+key)
        for key in ('closed_pulse','closed_aperture_mm','open_pulse','open_aperture_mm')}
    g=out['gripper_aperture_calibration']
    if not (g['closed_pulse'] < g['open_pulse'] and
            0 <= g['closed_aperture_mm'] < g['open_aperture_mm'] <= 100):
        raise UnsafePlan('invalid gripper pulse/aperture calibration')
    if out['usable_jaw_opening_mm'] > g['open_aperture_mm']:
        raise UnsafePlan('usable jaw opening exceeds measured full-open aperture')
    out['gentle_compression_mm']=finite(cfg.get('gentle_compression_mm'), 'gentle_compression_mm')
    if not 0.0 <= out['gentle_compression_mm'] <= 2.0:
        raise UnsafePlan('gentle compression must be between 0 and 2 mm')
    if not (1 <= out['gripper_speed_setting'] <= 3000):
        raise UnsafePlan('gripper speed setting must be between 1 and 3000')
    if not 0 <= out['knock_precontact_clearance_mm'] <= 0.5:
        raise UnsafePlan('knock pre-contact clearance must be 0–0.5 mm after TCP calibration')
    if out['knock_precontact_clearance_mm'] > out['knock_max_drop_mm']:
        raise UnsafePlan('knock stroke cannot reach the calibrated top-contact plane')
    if not (0 < out['knock_stroke_mm'] <= 15):
        raise UnsafePlan('single knock stroke must be at most 15 mm')
    if not (0 <= out['knock_max_drop_mm'] <= 1):
        raise UnsafePlan('knock downward drop must be at most 1 mm')
    if not (out['table_z_mm'] + 80 < out['approach_z_mm'] <= out['lift_z_mm'] <= out['transit_z_mm']):
        raise UnsafePlan('unsafe vertical staging heights')
    bowls = cfg.get('bowls')
    if not isinstance(bowls, dict) or set(bowls) != {'bowl_1', 'bowl_2'}:
        raise UnsafePlan('two measured bowl records are required')
    for name, bowl in bowls.items():
        bowl['center_xy_mm'] = vec(bowl.get('center_xy_mm'), f'{name}.center_xy_mm', 2)
        bowl['release_z_mm'] = finite(bowl.get('release_z_mm'), f'{name}.release_z_mm')
        bowl['inner_radius_mm'] = finite(bowl.get('inner_radius_mm'), f'{name}.inner_radius_mm')
        if not (30 <= bowl['inner_radius_mm'] <= 300):
            raise UnsafePlan(f'{name}: implausible inner radius')
        if not out['table_z_mm'] + 20 <= bowl['release_z_mm'] < out['approach_z_mm']:
            raise UnsafePlan(f'{name}: release height must be measured inside bowl and below approach')
    out['measured_object_height_mm']=finite(cfg.get('measured_object_height_mm'), 'measured_object_height_mm')
    if not 20 <= out['measured_object_height_mm'] <= 45:
        raise UnsafePlan('measured box height outside soap-only range')
    return out


def check_xyz(xyz: list[float], cfg: dict, label: str):
    for dim, (v, lo, hi) in enumerate(zip(xyz, cfg['workspace_min_mm'], cfg['workspace_max_mm'])):
        if not lo <= v <= hi:
            raise UnsafePlan(f'{label}: coordinate {dim}={v:.1f} mm outside configured workspace')


def yaw_for_axis(object_yaw_rad: float, local_axis: str, cfg: dict) -> float:
    axis_yaw = math.degrees(object_yaw_rad) + (90 if local_axis == 'y' else 0)
    return (axis_yaw - cfg['tool_closing_axis_yaw_offset_deg'] + 180) % 360 - 180


def tcp(xyz: list[float], yaw_deg: float, cfg: dict) -> tuple[float, ...]:
    return tuple(xyz + cfg['tool_roll_pitch_deg'] + [yaw_deg])


def object_by_name(layout: dict, name: str) -> dict:
    records = layout.get('planning_objects', layout['objects'])
    matches = [obj for obj in records if obj['name'] == name]
    if len(matches) != 1:
        raise UnsafePlan(f'exactly one current pose required for {name}')
    return matches[0]


def bowl_for(cfg: dict, name: str, step_index: int) -> str:
    manual = cfg.get('instance_to_bowl', {})
    chosen = manual.get(name, 'bowl_1' if step_index % 2 == 0 else 'bowl_2')
    if chosen not in cfg['bowls']:
        raise UnsafePlan(f'{name}: unknown bowl {chosen}')
    return chosen


def aperture_pulse(width_mm: float, cfg: dict) -> int:
    g=cfg['gripper_aperture_calibration']
    if not g['closed_aperture_mm'] <= width_mm <= g['open_aperture_mm']:
        raise UnsafePlan('target aperture outside measured gripper calibration')
    ratio=(width_mm-g['closed_aperture_mm'])/(g['open_aperture_mm']-g['closed_aperture_mm'])
    return round(g['closed_pulse']+ratio*(g['open_pulse']-g['closed_pulse']))


def pick_stages(obj: dict, orientation: str, placement: float | None,
                bowl_name: str, cfg: dict) -> list[Stage]:
    """Pre-pick, approach, gentle grasp, lift, transfer, place, release, retreat."""
    dims = [1000*d for d in obj['dimensions_m']]
    shrink = cfg.get('footprint_shrink_total_mm', 0.0)
    length, width = dims[0]-shrink, dims[1]-shrink
    closing_width = length if orientation == 'x' else width
    if closing_width + 2 > cfg['usable_jaw_opening_mm']:
        raise UnsafePlan(f'{obj["name"]}: {closing_width:.1f} mm grasp plus 2 mm margin exceeds jaw opening')
    target_aperture=closing_width-cfg['gentle_compression_mm']
    target_pulse=aperture_pulse(target_aperture,cfg)
    center = [1000*x for x in obj['base_center_m']]
    yaw = float(obj['yaw_rad'])
    along = [math.cos(yaw), math.sin(yaw)] if orientation == 'y' else [-math.sin(yaw), math.cos(yaw)]
    tangential_width = length if orientation == 'y' else width
    local_mid = (placement + 15.0) if placement is not None else 0.0
    if not -tangential_width/2 <= local_mid <= tangential_width/2:
        raise UnsafePlan(f'{obj["name"]}: Pick witness outside box edge')
    grasp_xy = [center[0]+local_mid*along[0], center[1]+local_mid*along[1]]
    grasp_z = cfg['table_z_mm'] + cfg['measured_object_height_mm']/2 + cfg['tcp_contact_z_offset_mm']
    if grasp_z <= cfg['table_z_mm']:
        raise UnsafePlan('grasp TCP would be at or below table plane')
    heading = yaw_for_axis(yaw, orientation, cfg)
    bowl = cfg['bowls'][bowl_name]
    bx, by = bowl['center_xy_mm']
    if math.hypot(length, width)/2 + 10 > bowl['inner_radius_mm']:
        raise UnsafePlan(f'{bowl_name}: box does not fit within measured bowl inner radius')
    poses = [([*grasp_xy,cfg['transit_z_mm']], 'pre-pick'),
             ([*grasp_xy,cfg['approach_z_mm']], 'approach'),
             ([*grasp_xy,grasp_z], 'pick'),
             ([*grasp_xy,cfg['lift_z_mm']], 'lift'),
             ([*grasp_xy,cfg['transit_z_mm']], 'clearance-lift'),
             ([bx,by,cfg['transit_z_mm']], 'transfer'),
             ([bx,by,bowl['release_z_mm']], 'place'),
             ([bx,by,cfg['transit_z_mm']], 'retreat')]
    for point, label in poses:
        check_xyz(point, cfg, label)
    return [Stage('pre-pick',tcp(poses[0][0],heading,cfg),speed_mm_s=30),
            Stage('open-fingers',gripper_pulse=round(cfg['gripper_aperture_calibration']['open_pulse'])),
            Stage('approach',tcp(poses[1][0],heading,cfg),speed_mm_s=20),
            Stage('pick',tcp(poses[2][0],heading,cfg),speed_mm_s=8),
            Stage('close-fingers',gripper_pulse=target_pulse,note=f'Target aperture {target_aperture:.1f} mm; standard gripper position does not limit force'),
            Stage('lift',tcp(poses[3][0],heading,cfg),speed_mm_s=15),
            Stage('clearance-lift',tcp(poses[4][0],heading,cfg),speed_mm_s=15),
            Stage('transfer',tcp(poses[5][0],heading,cfg),speed_mm_s=30),
            Stage('place',tcp(poses[6][0],heading,cfg),speed_mm_s=8),
            Stage('open-fingers',gripper_pulse=round(cfg['gripper_aperture_calibration']['open_pulse'])),
            Stage('retreat',tcp(poses[7][0],heading,cfg),speed_mm_s=20)]


def knock_stages(obj: dict, direction: str, cfg: dict) -> list[Stage]:
    """One short top-edge slide, then stop for observed displacement and replan."""
    if direction not in ('+x','-x','+y','-y'):
        raise UnsafePlan('unknown local knock direction')
    yaw=float(obj['yaw_rad']);ux=(math.cos(yaw),math.sin(yaw));uy=(-math.sin(yaw),math.cos(yaw))
    axis={'x':ux,'y':uy}[direction[-1]]
    sign=1 if direction[0]=='+' else -1
    vec2=[sign*axis[0],sign*axis[1]]
    center=[1000*x for x in obj['base_center_m']]
    length,width,height=[1000*d for d in obj['dimensions_m']]
    normal=length if direction[-1]=='x' else width
    start_xy=[center[i]+0.05*normal*vec2[i] for i in (0,1)]
    top_z=cfg['table_z_mm']+cfg['measured_object_height_mm']
    contact_z=top_z+cfg['tcp_contact_z_offset_mm']+cfg['knock_precontact_clearance_mm']
    if contact_z <= cfg['table_z_mm']+5:
        raise UnsafePlan('knock contact TCP too low')
    end_xy=[start_xy[i]+cfg['knock_stroke_mm']*vec2[i] for i in (0,1)]
    heading=(math.degrees(math.atan2(vec2[1],vec2[0]))-cfg['tool_closing_axis_yaw_offset_deg']+180)%360-180
    poses=[([*start_xy,cfg['transit_z_mm']],'pre-knock'),
           ([*start_xy,cfg['approach_z_mm']],'approach'),
           ([*start_xy,contact_z],'contact'),
           ([*end_xy,contact_z-cfg['knock_max_drop_mm']],'short-stroke'),
           ([*end_xy,cfg['approach_z_mm']],'vertical-release'),
           ([*end_xy,cfg['transit_z_mm']],'retreat')]
    for point,label in poses: check_xyz(point,cfg,label)
    return [Stage('close-fingers',gripper_pulse=round(cfg['gripper_aperture_calibration']['closed_pulse'])),
            Stage('pre-knock',tcp(poses[0][0],heading,cfg),speed_mm_s=25),
            Stage('approach',tcp(poses[1][0],heading,cfg),speed_mm_s=15),
            Stage('contact',tcp(poses[2][0],heading,cfg),speed_mm_s=5,
                  note='Top-edge contact requires calibrated TCP and observation'),
            Stage('short-stroke',tcp(poses[3][0],heading,cfg),speed_mm_s=3,
                  note='At most one bounded stroke; no blind repeat'),
            Stage('vertical-release',tcp(poses[4][0],heading,cfg),speed_mm_s=5),
            Stage('retreat',tcp(poses[5][0],heading,cfg),speed_mm_s=10),
            Stage('pause-for-reobservation',note='Capture new RGB-D, verify displacement, replan, then mandatory PickPlace')]


def build_action(layout: dict, cfg: dict, step_index: int) -> dict:
    if not layout.get('plan_verified') or layout.get('solver_status') != 'optimal':
        raise UnsafePlan('input solver plan not verified optimal')
    actions=layout.get('action_details')
    if not isinstance(actions,list) or not 0 <= step_index < len(actions):
        raise UnsafePlan('invalid action index')
    action=actions[step_index]
    name=action['target'];obj=object_by_name(layout,name)
    cfg=dict(cfg,footprint_shrink_total_mm=1000*layout.get('footprint_shrink_total_m',0))
    if action['kind']=='pick':
        witness=action['witness']
        if not witness.startswith('pick-') or '@' not in witness:
            raise UnsafePlan('unsupported Pick witness')
        orientation=witness[5];placement=1000*float(witness.split('@',1)[1])
        chosen=bowl_for(cfg,name,step_index)
        stages=pick_stages(obj,orientation,placement,chosen,cfg)
    elif action['kind']=='knock':
        witness=action['witness']
        if not witness.startswith('knock-'):
            raise UnsafePlan('unsupported Knock witness')
        chosen=None;stages=knock_stages(obj,witness[6:],cfg)
    else: raise UnsafePlan('unsupported action kind')
    return dict(step=step_index+1,target=name,kind=action['kind'],bowl=chosen,
                stages=[s.as_dict() for s in stages],
                after_knock='stop and reobserve before mandatory PickPlace' if chosen is None else None)


def execute_one(action: dict, cfg: dict) -> None:
    """Physical backend. Never called from preview. One action per invocation."""
    from xarm.wrapper import XArmAPI
    arm=XArmAPI(cfg['robot_ip'],is_radian=False,do_not_open=True)
    try:
        arm.connect()
        if not arm.connected: raise UnsafePlan('xArm connection failed')
        code,state=arm.get_state()
        if code or state in (4,5): raise UnsafePlan(f'xArm state not ready: {code},{state}')
        if arm.collision_sensitivity < 3:
            raise UnsafePlan('controller collision sensitivity is below reviewed minimum 3')
        tcp_load = arm.tcp_load
        if not isinstance(tcp_load,(list,tuple)) or float(tcp_load[0]) < 0.7:
            raise UnsafePlan('controller TCP payload does not include standard gripper mass')
        code,err=arm.get_err_warn_code()
        if code or any(err): raise UnsafePlan(f'xArm error/warning: {code},{err}')
        code=arm.set_tcp_offset(cfg['tcp_offset_mm_deg'],is_radian=False,wait=True)
        if code: raise UnsafePlan(f'cannot set reviewed TCP offset: {code}')
        code,pos=arm.get_position(is_radian=False)
        if code: raise UnsafePlan(f'cannot read xArm TCP pose: {code}')
        if pos[2] < cfg['transit_z_mm']:
            raise UnsafePlan('initial TCP below transit height; recover with a reviewed separate motion')
        for stage in action['stages']:
            if stage['tcp_mm_deg'] is not None:
                code, _ = arm.get_inverse_kinematics(stage['tcp_mm_deg'],
                                                        input_is_radian=False,
                                                        return_is_radian=False,
                                                        limited=True)
                if code: raise UnsafePlan(f"{stage['phase']}: IK failed with code {code}")
        for label, code in (('motion_enable',arm.motion_enable(True)),
                            ('mode',arm.set_mode(0)),('state',arm.set_state(0)),
                            ('gripper_mode',arm.set_gripper_mode(0)),
                            ('gripper_enable',arm.set_gripper_enable(True)),
                            ('gripper_speed',arm.set_gripper_speed(cfg['gripper_speed_setting']))):
            if code: raise UnsafePlan(f'cannot enable {label}: {code}')
        for stage in action['stages']:
            print(f"[EXECUTE] {stage['phase']}: {stage}",flush=True)
            if stage['tcp_mm_deg'] is not None:
                x,y,z,r,p,yaw=stage['tcp_mm_deg']
                code=arm.set_position(x=x,y=y,z=z,roll=r,pitch=p,yaw=yaw,
                                      speed=stage['speed_mm_s'],mvacc=50,
                                      wait=True,timeout=30,is_radian=False)
            elif stage['gripper_pulse'] is not None:
                code=arm.set_gripper_position(stage['gripper_pulse'],wait=True,timeout=15)
            else:
                break
            if code: raise UnsafePlan(f"{stage['phase']} failed with SDK code {code}; stop and inspect")
            if stage['gripper_pulse'] is not None:
                feedback_code, feedback_pulse = arm.get_gripper_position()
                if feedback_code: raise UnsafePlan(f"{stage['phase']}: no gripper position feedback")
                print(f"[GRIPPER FEEDBACK] pulse={feedback_pulse}", flush=True)
                if abs(float(feedback_pulse)-float(stage['gripper_pulse'])) > 20:
                    raise UnsafePlan(f"{stage['phase']}: gripper did not approach target pulse; command return code alone is insufficient")
            code,err=arm.get_err_warn_code()
            if code or any(err): raise UnsafePlan(f"{stage['phase']} produced xArm error/warning {code},{err}")
            if stage['tcp_mm_deg'] is not None:
                code,actual=arm.get_position(is_radian=False)
                if code or math.dist(actual[:3],stage['tcp_mm_deg'][:3])>8:
                    raise UnsafePlan(f"{stage['phase']}: TCP did not reach commanded waypoint")
    finally:
        arm.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--step',type=int,required=True,help='one-based high-level action index')
    ap.add_argument('--out',type=Path,help='save reviewable single-action stage JSON')
    ap.add_argument('--execute',action='store_true',help='connect to xArm and move one reviewed action')
    ap.add_argument('--approval',default='',help='must equal the reviewed plan identifier for execution')
    args=ap.parse_args()
    layout=json.loads(args.plan.read_text());cfg=checked_config(json.loads(args.config.read_text()))
    action=build_action(layout,cfg,args.step-1)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(action,indent=2)+'\n')
    print(json.dumps(action,indent=2))
    if args.execute:
        if cfg.get('provisional') is True:
            raise UnsafePlan('provisional waypoint preview cannot execute')
        if cfg.get('tcp_calibrated') is not True or cfg.get('gripper_calibrated') is not True or cfg.get('table_plane_verified') is not True:
            raise UnsafePlan('TCP, gripper and table-plane calibrations must be verified before motion')
        if action['kind'] == 'pick' and cfg.get('bowls_placed_and_checked') is not True:
            raise UnsafePlan('bowls must be placed and checked before releasing a box')
        if action['kind'] == 'knock' and cfg.get('knock_contact_validated') is not True:
            raise UnsafePlan('Knock TCP contact, stroke, and packaging response must be physically validated')
        if args.step > 1 and cfg.get('fresh_scene_reviewed_for_step') != args.step:
            raise UnsafePlan('a fresh scene and plan must be reviewed before this later action')
        if cfg.get('contact_execution_reviewed') is not True:
            raise UnsafePlan('contact_execution_reviewed must be true after safety review')
        if args.approval != cfg.get('approved_plan_id') or not args.approval:
            raise UnsafePlan('explicit matching approved_plan_id required before physical execution')
        print('ABOUT TO MOVE THE REAL XARM7. This invocation executes ONE action.',flush=True)
        execute_one(action,cfg)


if __name__=='__main__':
    try:
        main()
    except UnsafePlan as exc:
        raise SystemExit(f'BLOCKED: {exc}') from None
