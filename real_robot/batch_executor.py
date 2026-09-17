"""Pose -> complete Pick/Knock waypoints -> xArm SDK batch execution."""
from __future__ import annotations
import copy
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xarm_executor import UnsafePlan
from standard_gripper_geometry import tip_extension_mm
from object_catalog import DIMENSIONS

# The startup joint home is guarded by the measured TCP height.
STARTUP_HOME_MIN_Z_MM = 100.

# Explicit TCP Z overrides remain available alongside per-class height rules.
HEIGHT_FIELDS = {
    'pick_tcp_z_mm': 'TCP Z with open fingers positioned around the box sides for grasping',
    'knock.contact_tcp_z_mm': 'optional absolute TCP Z override for closed-finger contact',
    'release_tcp_z_mm': 'TCP Z at which to open the gripper over the selected drop point',
}


def get_field(cfg, key):
    value = cfg
    for part in key.split('.'):
        if not isinstance(value, dict): return None
        value = value.get(part)
    return value


def height_available(cfg, key):
    if get_field(cfg, key) is not None: return True
    alternative = {'pick_tcp_z_mm':'pick_fingertip_clearance_mm',
                   'knock.contact_tcp_z_mm':'knock.top_indent_mm'}.get(key)
    return alternative is not None and get_field(cfg, alternative) is not None


def physical_config_report(cfg):
    missing = [dict(field=key, needed=description) for key,description in HEIGHT_FIELDS.items()
               if not height_available(cfg,key)]
    return dict(ready_to_start=not missing, missing_values=missing,
                note='TCP Z overrides are in link_base mm. Pick uses fingertip clearance. Knock uses nominal height minus indentation for configured soap classes, without fingertip compensation; other classes use detected top and nominal gripper geometry.')


def number(value, name):
    try: result = float(value)
    except (TypeError, ValueError): raise UnsafePlan(f'Set {name} to a number in the local execution config') from None
    if not math.isfinite(result): raise UnsafePlan(f'{name} must be finite')
    return result


def apply_execution_overrides(base, pipeline):
    """Template placement settings overlay current shared motion settings."""
    result=copy.deepcopy(base)
    overrides=pipeline.get('execution_overrides',{})
    allowed={'class_to_bowl','release_tcp_z_mm_by_class'}
    if not isinstance(overrides,dict) or set(overrides)-allowed:
        raise UnsafePlan('execution_overrides supports class_to_bowl and release_tcp_z_mm_by_class')
    for key,value in overrides.items():
        if not isinstance(value,dict): raise UnsafePlan(f'{key} must be an object')
        result[key]={**result.get(key,{}),**copy.deepcopy(value)}
    return result


def placement_bowl(cfg, obj, completed_count):
    return cfg.get('class_to_bowl',{}).get(obj['kind'],
        cfg.get('instance_to_bowl',{}).get(obj['name'],
            'bowl_1' if completed_count%2==0 else 'bowl_2'))


def checked_batch_config(config, *, physical=False):
    """Check values actually used by motion, once. No calibration attestations."""
    cfg=copy.deepcopy(config)
    for key,n in [('tcp_offset_mm_deg',6),('tool_roll_pitch_deg',2)]:
        values=cfg.get(key)
        if not isinstance(values,list) or len(values)!=n: raise UnsafePlan(f'{key} needs {n} numbers')
        cfg[key]=[number(v,key) for v in values]
    for key in ('transit_z_mm','approach_z_mm','usable_jaw_opening_mm','gripper_speed_setting',
                'gripper_open_pulse','gripper_open_mm','gentle_compression_mm','tool_closing_axis_yaw_offset_deg',
                'placement_tolerance_mm'):
        cfg[key]=number(cfg.get(key),key)
    cfg['pick_overlap_tolerance_mm']=number(cfg.get('pick_overlap_tolerance_mm',0),'pick_overlap_tolerance_mm')
    if not 0<=cfg['pick_overlap_tolerance_mm']<=5:
        raise UnsafePlan('pick_overlap_tolerance_mm must be between 0 and 5')
    home=cfg.get('startup_home',{})
    if not isinstance(home,dict): raise UnsafePlan('startup_home must be an object')
    angles=home.get('joint_angles_deg')
    if not isinstance(angles,list) or len(angles)!=7:
        raise UnsafePlan('startup_home.joint_angles_deg needs seven joint angles')
    cfg['startup_home']=dict(joint_angles_deg=[number(v,'startup_home.joint_angles_deg') for v in angles],
                            speed_deg_s=number(home.get('speed_deg_s'),'startup_home.speed_deg_s'),
                            acceleration_deg_s2=number(home.get('acceleration_deg_s2'),'startup_home.acceleration_deg_s2'))
    if cfg['startup_home']['speed_deg_s']<=0 or cfg['startup_home']['acceleration_deg_s2']<=0:
        raise UnsafePlan('startup_home speed and acceleration must be positive')
    cfg['motion_speed_scale']=number(cfg.get('motion_speed_scale',1),'motion_speed_scale')
    if cfg['motion_speed_scale'] <= 0: raise UnsafePlan('motion_speed_scale must be positive')
    cfg['approach_speed_scale']=number(cfg.get('approach_speed_scale',cfg['motion_speed_scale']),'approach_speed_scale')
    if cfg['approach_speed_scale'] <= 0: raise UnsafePlan('approach_speed_scale must be positive')
    cfg['placement_speed_multiplier']=number(cfg.get('placement_speed_multiplier',1),'placement_speed_multiplier')
    if cfg['placement_speed_multiplier']<=0: raise UnsafePlan('placement_speed_multiplier must be positive')
    if cfg.get('frame')!='link_base': raise UnsafePlan('motion coordinates must use link_base')
    if not cfg.get('robot_ip'): raise UnsafePlan('robot_ip is required')
    if not 0 < cfg['approach_z_mm'] < cfg['transit_z_mm']: raise UnsafePlan('approach must be below transit height')
    if not 0 < cfg['gripper_open_pulse'] <= 850 or cfg['gripper_open_mm']<=0: raise UnsafePlan('invalid gripper aperture mapping')
    if not 1 <= cfg['gripper_speed_setting'] <= 5000: raise UnsafePlan('invalid gripper speed setting')
    if not 0 <= cfg['gentle_compression_mm'] <= 2: raise UnsafePlan('gentle compression must be 0..2 mm')
    overrides=cfg.get('pick_aperture_mm_by_class',{})
    if not isinstance(overrides,dict) or any(kind not in DIMENSIONS for kind in overrides):
        raise UnsafePlan('pick_aperture_mm_by_class must map supported box classes to aperture targets')
    cfg['pick_aperture_mm_by_class']={kind:number(value,'pick_aperture_mm_by_class.'+kind)
                                     for kind,value in overrides.items()}
    if any(not 0 < value <= cfg['gripper_open_mm'] for value in cfg['pick_aperture_mm_by_class'].values()):
        raise UnsafePlan('Pick aperture overrides must be positive and within the gripper opening')
    clearances=cfg.get('pick_fingertip_clearance_mm_by_class',{})
    if not isinstance(clearances,dict) or any(kind not in DIMENSIONS for kind in clearances):
        raise UnsafePlan('pick_fingertip_clearance_mm_by_class must map supported classes to clearance')
    cfg['pick_fingertip_clearance_mm_by_class']={kind:number(value,'pick_fingertip_clearance_mm_by_class.'+kind)
                                               for kind,value in clearances.items()}
    if any(not 0<=value<cfg['approach_z_mm'] for value in cfg['pick_fingertip_clearance_mm_by_class'].values()):
        raise UnsafePlan('Pick fingertip clearance overrides must be nonnegative and below approach height')
    for key in ('stroke_mm','speed_mm_s'):
        cfg['knock'][key]=number(cfg.get('knock',{}).get(key),'knock.'+key)
        if cfg['knock'][key]<=0: raise UnsafePlan('Knock distance and speed must be positive')
    for key in HEIGHT_FIELDS:
        value=get_field(cfg,key)
        if value is not None:
            z=number(value,key)
            if not 0 < z < cfg['approach_z_mm']: raise UnsafePlan(f'{key} must be above 0 and below approach height')
            if '.' in key: cfg['knock']['contact_tcp_z_mm']=z
            else: cfg[key]=z
    for key in ('pick_fingertip_clearance_mm','knock.top_indent_mm','knock.slide_descent_mm'):
        value=get_field(cfg,key)
        if value is not None:
            value=number(value,key)
            if value < 0: raise UnsafePlan(f'{key} must be nonnegative')
            if '.' in key: cfg['knock'][key.split('.')[1]]=value
            else: cfg[key]=value
    cfg['knock'].setdefault('slide_descent_mm',0)
    if cfg['knock'].get('slide_descent_mm') is None: raise UnsafePlan('knock.slide_descent_mm must be a number')
    if cfg['knock'].get('top_indent_mm',0) is not None:
        if cfg['knock'].get('top_indent_mm',0)+cfg['knock']['slide_descent_mm'] > 3:
            raise UnsafePlan('configured Knock indentation exceeds the requested 3 mm maximum')
    indent_overrides=cfg['knock'].get('top_indent_mm_by_class',{})
    if not isinstance(indent_overrides,dict) or any(kind not in DIMENSIONS for kind in indent_overrides):
        raise UnsafePlan('knock.top_indent_mm_by_class must map supported box classes to indentation')
    cfg['knock']['top_indent_mm_by_class']={kind:number(value,'knock.top_indent_mm_by_class.'+kind)
                                           for kind,value in indent_overrides.items()}
    for kind,indent in cfg['knock']['top_indent_mm_by_class'].items():
        # Explicit user request: BEACH starts 4 mm below its nominal top and ends 5 mm below.
        maximum=5 if kind=='beach' else 3
        if indent<0 or indent+cfg['knock']['slide_descent_mm']>maximum:
            raise UnsafePlan(f'{kind}: configured Knock indentation exceeds the requested {maximum} mm maximum')
    classes=cfg['knock'].get('nominal_height_tcp_classes',[])
    if not isinstance(classes,list) or any(kind not in ('protex','beach') for kind in classes):
        raise UnsafePlan('knock.nominal_height_tcp_classes must list only protex and/or beach')
    cfg['knock']['nominal_height_tcp_classes']=classes
    uses_geometry = (cfg.get('pick_tcp_z_mm') is None and cfg.get('pick_fingertip_clearance_mm') is not None
                     or cfg['knock'].get('contact_tcp_z_mm') is None and cfg['knock'].get('top_indent_mm') is not None)
    if uses_geometry:
        if not np.allclose(cfg['tool_roll_pitch_deg'],[180,0]):
            raise UnsafePlan('standard fingertip-height calculation requires top-down roll=180, pitch=0')
        if not np.allclose([cfg['tcp_offset_mm_deg'][i] for i in (0,1,3,4,5)],0):
            raise UnsafePlan('standard fingertip-height calculation requires an axial, unrotated TCP offset')
    drops=cfg.get('drop_points_xy_mm',{})
    if set(drops)!={'bowl_1','bowl_2'}: raise UnsafePlan('two drop_points_xy_mm entries are required')
    cfg['drop_points_xy_mm']={name:[number(v,name) for v in xy] for name,xy in drops.items()}
    if any(len(xy)!=2 for xy in cfg['drop_points_xy_mm'].values()): raise UnsafePlan('drop points need x,y')
    routes=cfg.get('class_to_bowl',{})
    if not isinstance(routes,dict) or any(k not in DIMENSIONS or v not in drops for k,v in routes.items()):
        raise UnsafePlan('class_to_bowl must map supported classes to configured bowls')
    cfg['class_to_bowl']=dict(routes)
    heights=cfg.get('release_tcp_z_mm_by_class',{})
    if not isinstance(heights,dict) or any(k not in DIMENSIONS for k in heights):
        raise UnsafePlan('release_tcp_z_mm_by_class must map supported classes to TCP heights')
    cfg['release_tcp_z_mm_by_class']={k:number(v,'release_tcp_z_mm_by_class.'+k) for k,v in heights.items()}
    if any(not 0<v<cfg['approach_z_mm'] for v in cfg['release_tcp_z_mm_by_class'].values()):
        raise UnsafePlan('release TCP height overrides must be above zero and below approach height')
    if physical:
        missing=physical_config_report(cfg)['missing_values']
        if missing: raise UnsafePlan('Enter these TCP heights in mm: '+', '.join(item['field'] for item in missing))
    return cfg


def direction_base(obj, local_direction):
    if local_direction not in ('+x','-x','+y','-y'): raise UnsafePlan('invalid local Knock direction')
    angle=float(obj['yaw_rad'])+(math.pi/2 if local_direction[-1]=='y' else 0)
    sign=1 if local_direction[0]=='+' else -1
    return np.array([sign*math.cos(angle),sign*math.sin(angle),0.])


def stage(phase, pose=None, *, pulse=None, speed=20):
    return dict(phase=phase,tcp_mm_deg=None if pose is None else [float(x) for x in pose],
                gripper_pulse=pulse,speed_mm_s=float(speed))


def pick_placement_reference(action):
    """Expected box centre from the commanded grasp/release TCP poses.

    Assumes the box retains its grasp-relative position until release. This is
    an XY verification reference, not a prediction of settling height in a bowl.
    """
    from scipy.spatial.transform import Rotation
    try:
        if action['kind']!='pick':raise ValueError('requires a Pick action')
        pick,place=[np.asarray(next(s['tcp_mm_deg'] for s in action['stages'] if s['phase']==phase),float)
                    for phase in ('pick','place')]
        center=np.asarray(action['source_pose']['base_center_m'],float)*1000
        if pick.shape!=(6,) or place.shape!=(6,) or center.shape!=(3,):raise ValueError('invalid pose shape')
        if not all(np.isfinite(v).all() for v in (pick,place,center)):raise ValueError('nonfinite pose')
    except (KeyError,StopIteration,TypeError,ValueError) as exc:
        raise UnsafePlan(f'invalid Pick placement reference: {exc}') from exc
    rotation_pick=Rotation.from_euler('xyz',pick[3:],degrees=True).as_matrix()
    rotation_place=Rotation.from_euler('xyz',place[3:],degrees=True).as_matrix()
    offset=rotation_place@rotation_pick.T@(center-pick[:3])
    return dict(model='box_retains_grasp_relative_offset_until_release',
                release_tcp_xy_mm=place[:2].tolist(),
                center_offset_at_release_base_mm=offset.tolist(),
                expected_box_center_xy_mm=(place[:2]+offset[:2]).tolist())


def build_batch(layout,candidate,cfg,*,bowl=None):
    obj=next(o for o in layout['objects'] if o['name']==candidate['target'])
    if candidate.get('blockers'): raise UnsafePlan('selected action has current blockers')
    centre=np.asarray(obj['base_center_m'])*1000
    dims=np.asarray(obj['dimensions_m'])*1000
    yaw=float(obj['yaw_rad'])
    result=dict(target=obj['name'],kind=candidate['kind'],witness=candidate['witness'],
                source_pose=copy.deepcopy(obj),
                pose_source=('detected_pose_with_manual_base_offset' if any(obj.get('base_pose_offset_mm',[]))
                             else 'raw_detected_pose'),bowl=bowl)
    if candidate['kind']=='knock':
        local=candidate['witness'][6:]; direction=direction_base(obj,local)
        normal=dims[0 if local[-1]=='x' else 1]
        start=centre+.05*normal*direction
        contact=cfg['knock'].get('contact_tcp_z_mm')
        height_geometry={'mode':'explicit_tcp_z'}
        if contact is None:
            indent=number(cfg['knock'].get('top_indent_mm_by_class',{}).get(obj['kind'],
                          cfg['knock'].get('top_indent_mm')),'effective Knock top indentation')
            detected_top=centre[2]+dims[2]/2
            if obj['kind'] in cfg['knock'].get('nominal_height_tcp_classes',[]):
                # User-requested flat soap constraint: base table Z=0, per-class nominal top.
                # Direct TCP targets: Protex 28 -> 27 mm; BEACH 21 -> 20 mm.
                top=DIMENSIONS[obj['kind']][2]*1000
                contact=top-indent
                height_geometry=dict(mode='nominal_box_height_direct_tcp',
                                     reference_table_base_z_mm=0.,
                                     fitted_center_base_z_mm=top/2,
                                     object_top_base_z_mm=top,
                                     detected_object_top_base_z_mm=float(detected_top),
                                     fingertip_compensation_applied_mm=0.,
                                     tcp_below_nominal_top_mm=indent)
            else:
                extension=tip_extension_mm(cfg['tcp_offset_mm_deg'][2],closed=True)
                top=detected_top
                contact=top-indent+extension
                height_geometry=dict(mode='nominal_standard_gripper',object_top_base_z_mm=float(top),
                                     closed_tip_extension_mm=extension,initial_indent_mm=indent)
        start[2]=number(contact,'knock.contact_tcp_z_mm')
        end=start+cfg['knock']['stroke_mm']*direction
        end[2]-=cfg['knock'].get('slide_descent_mm',0)
        if not 0 < end[2] <= start[2] < cfg['approach_z_mm']:
            raise UnsafePlan('computed Knock heights must be above base Z=0 and below approach height')
        result['height_geometry']=dict(height_geometry,contact_tcp_z_mm=float(start[2]),
                                      end_tcp_z_mm=float(end[2]),slide_descent_mm=float(start[2]-end[2]))
        heading=math.degrees(math.atan2(direction[1],direction[0]))-cfg['tool_closing_axis_yaw_offset_deg']
        angles=[*cfg['tool_roll_pitch_deg'],(heading+180)%360-180]
        def pose(p,z=None): return [p[0],p[1],p[2] if z is None else z]+angles
        result.update(local_direction=local,base_direction=direction.tolist(),
                      intended_slide_mm=cfg['knock']['stroke_mm'])
        result['stages']=[stage('close-fingers',pulse=0),
            stage('pre-knock',pose(start,cfg['transit_z_mm']),speed=75),
            stage('approach',pose(start,cfg['approach_z_mm']),speed=45),
            stage('near-contact',pose(start,min(start[2]+20,cfg['approach_z_mm'])),speed=15),
            stage('contact',pose(start),speed=2),
            stage('horizontal-knock',pose(end),speed=cfg['knock']['speed_mm_s']),
            stage('vertical-release',pose(end,cfg['approach_z_mm']),speed=15),
            stage('retreat',pose(end,cfg['transit_z_mm']),speed=30)]
    elif candidate['kind']=='pick':
        from pick_clearance import pick_clearance
        clearance_check=pick_clearance(layout,candidate,cfg)
        if not clearance_check['passed']:
            raise UnsafePlan(f'{obj["name"]}: Pick clearance failed: {clearance_check["reasons"]}; '
                             f'blockers={clearance_check["blockers"]}')
        result['pick_clearance']=clearance_check
        orientation=candidate['witness'][5]
        placement=float(candidate['witness'].split('@')[1])*1000
        # The footprint shrink belongs to collision planning, not jaw compression.
        closing_width=dims[0 if orientation=='x' else 1]
        if closing_width+2>cfg['usable_jaw_opening_mm']: raise UnsafePlan('selected grasp exceeds gripper opening')
        from gripper_command import pick_aperture_mm
        aperture=pick_aperture_mm(cfg,obj['kind'],closing_width)
        pulse=round(cfg['gripper_open_pulse']*aperture/cfg['gripper_open_mm'])
        if not 0<=pulse<=cfg['gripper_open_pulse']: raise UnsafePlan('grasp pulse is outside gripper range')
        tangent=np.array([math.cos(yaw),math.sin(yaw)]) if orientation=='y' else np.array([-math.sin(yaw),math.cos(yaw)])
        grasp_xy=centre[:2]+(placement+15)*tangent  # midpoint of the solver's 30 mm finger window
        heading=math.degrees(yaw)+(90 if orientation=='y' else 0)-cfg['tool_closing_axis_yaw_offset_deg']
        angles=[*cfg['tool_roll_pitch_deg'],(heading+180)%360-180]
        z=cfg.get('pick_tcp_z_mm')
        height_geometry={'mode':'explicit_tcp_z'}
        if z is None:
            clearance=number(cfg.get('pick_fingertip_clearance_mm_by_class',{}).get(obj['kind'],
                             cfg.get('pick_fingertip_clearance_mm')),'effective Pick fingertip clearance')
            # Layout contains only active source boxes, never placed bowl objects.
            table_z=float(np.median([o['base_center_m'][2]*1000-o['dimensions_m'][2]*500
                                    for o in layout['objects']]))
            extension=tip_extension_mm(cfg['tcp_offset_mm_deg'][2])
            z=table_z+clearance+extension
            height_geometry=dict(mode='nominal_standard_gripper',table_base_z_estimate_mm=table_z,
                                 table_estimate_source='median_active_box_center_z_minus_half_height',
                                 min_fingertip_clearance_mm=clearance,max_tip_extension_mm=extension)
        z=number(z,'pick_tcp_z_mm')
        if not 0 < z < cfg['approach_z_mm']: raise UnsafePlan('computed Pick height is outside approach range')
        result['height_geometry']=dict(height_geometry,pick_tcp_z_mm=z)
        release_z=number(cfg.get('release_tcp_z_mm_by_class',{}).get(obj['kind'],
                         cfg.get('release_tcp_z_mm')),'effective release_tcp_z_mm')
        result['release_tcp_z_mm']=release_z
        drop=cfg['drop_points_xy_mm'][bowl]
        def pose(xy,height): return [*xy,height]+angles
        result['target_aperture_mm']=aperture
        result['aperture_mapping']='uncalibrated_linear_pulse_estimate'
        result['grasp_aperture_mode']=('class_override' if obj['kind'] in cfg['pick_aperture_mm_by_class']
                                       else 'nominal_width_minus_gentle_compression')
        result['nominal_closing_width_mm']=float(closing_width)
        result['stages']=[stage('pre-pick',pose(grasp_xy,cfg['transit_z_mm']),speed=90),
            stage('open-fingers',pulse=round(cfg['gripper_open_pulse'])),
            stage('approach',pose(grasp_xy,cfg['approach_z_mm']),speed=60),
            stage('pick',pose(grasp_xy,z),speed=8),stage('close-fingers',pulse=pulse),
            stage('lift',pose(grasp_xy,cfg['transit_z_mm']),speed=45),
            stage('transfer',pose(drop,cfg['transit_z_mm']),speed=90),
            stage('place',pose(drop,release_z),speed=8),
            stage('open-fingers',pulse=round(cfg['gripper_open_pulse'])),
            stage('retreat',pose(drop,cfg['transit_z_mm']),speed=60)]
    else: raise UnsafePlan('unknown action kind')
    # Knock stroke speed is an absolute setting, independent of other arm speeds.
    for motion_stage in result['stages']:
        if motion_stage['tcp_mm_deg'] is not None and motion_stage['phase'] != 'horizontal-knock':
            scale=cfg.get('motion_speed_scale',1.)
            if motion_stage['phase'] in ('pre-pick','pre-knock','approach','near-contact','contact','pick'):
                scale=cfg.get('approach_speed_scale',scale)
            if motion_stage['phase']=='place':
                scale*=cfg.get('placement_speed_multiplier',1.)
            motion_stage['speed_mm_s'] *= scale
    result['observation_policy']='before_action_and_after_complete_retreat_only'
    return result


class XArmBatchExecutor:
    def __init__(self,cfg,api_factory=None):
        self.cfg=checked_batch_config(cfg,physical=True);self.api_factory=api_factory
        self._startup_home_done=False

    def return_home(self,folder):
        """Final joint motion after successful plan completion; never recover a stop."""
        if not self._startup_home_done:
            raise UnsafePlan('final home requires an initialized execution session')
        factory=self.api_factory
        if factory is None:
            from xarm.wrapper import XArmAPI
            factory=XArmAPI
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        home=self.cfg['startup_home']
        (folder/'request.json').write_text(json.dumps(home,indent=2)+'\n')
        arm=factory(self.cfg['robot_ip'],is_radian=False,do_not_open=True)
        clock=time.monotonic();started=False
        with (folder/'execution.jsonl').open('a',buffering=1) as log:
            def record(event,**data):
                log.write(json.dumps(dict(time=time.time(),elapsed_seconds=time.monotonic()-clock,
                                          event=event,**data))+'\n')
            def ok(label,code):
                record('sdk_return',operation=label,code=code)
                if code:raise UnsafePlan(f'{label}: SDK code {code}')
            try:
                arm.connect()
                if not arm.connected:raise UnsafePlan('cannot connect to xArm for final home')
                code,state=arm.get_state();ok('final home state',code)
                code,errors=arm.get_err_warn_code();ok('final home errors',code)
                record('initial_controller_status',state=state,errors=errors)
                if state not in (0,2) or any(errors):
                    raise UnsafePlan(f'final home requires ready controller: state {state}, errors {errors}')
                code,before=arm.get_position(is_radian=False);ok('final home initial pose',code)
                record('initial_pose',tcp_mm_deg=before)
                if (np.asarray(before).shape!=(6,) or not np.isfinite(before).all()
                        or before[2]<=STARTUP_HOME_MIN_Z_MM or before[2]<self.cfg['transit_z_mm']-1):
                    raise UnsafePlan('final home requires the completed retreat at transit height')
                code,limited=arm.is_joint_limit(home['joint_angles_deg'],is_radian=False);ok('final home joint limits',code)
                if limited!=False:raise UnsafePlan('final home joints exceed controller limits')
                code,expected=arm.get_forward_kinematics(home['joint_angles_deg'],
                    input_is_radian=False,return_is_radian=False);ok('final home forward kinematics',code)
                if (np.asarray(expected).shape!=(6,) or not np.isfinite(expected).all()
                        or expected[2]<=max(300.,self.cfg['transit_z_mm'])):
                    raise UnsafePlan('final home TCP must be above 300 mm and transit height')
                label='SIMULATED SDK HOME' if self.api_factory else 'ABOUT TO MOVE XARM7'
                print(f'{label}: entire plan completed; return to joint home {home["joint_angles_deg"]} '
                      f'degrees at {home["speed_deg_s"]:g} deg/s; expected TCP Z {expected[2]:.3f} mm.',flush=True)
                record('final_home_start',home=home,expected_tcp_mm_deg=expected)
                started=True
                ok('final joint home',arm.set_servo_angle(angle=home['joint_angles_deg'],
                    speed=home['speed_deg_s'],mvacc=home['acceleration_deg_s2'],
                    wait=True,timeout=180,is_radian=False))
                code,after=arm.get_position(is_radian=False);ok('final home pose',code)
                code,joints=arm.get_servo_angle(is_radian=False);ok('final home joints',code)
                record('final_home_readback',tcp_mm_deg=after,joint_angles_deg=joints)
                if (np.asarray(after).shape!=(6,) or not np.isfinite(after).all()
                        or after[2]<=300 or after[2]<self.cfg['transit_z_mm']
                        or np.asarray(joints).shape!=(7,) or not np.isfinite(joints).all()
                        or not np.allclose(joints,home['joint_angles_deg'],atol=1,rtol=0)):
                    raise UnsafePlan('final home did not reach its joint target above transit height')
                code,state=arm.get_state();ok('final home completed state',code)
                code,errors=arm.get_err_warn_code();ok('final home completed errors',code)
                if state not in (0,2) or any(errors):
                    raise UnsafePlan(f'xArm not ready after final home: state {state}, errors {errors}')
                record('final_home_complete',state=state)
            except BaseException as exc:
                record('aborted',reason=str(exc) or type(exc).__name__)
                if started:
                    try:record('stop',code=arm.set_state(4))
                    except Exception as error:record('stop_failed',reason=str(error))
                raise
            finally:
                arm.disconnect()

    def execute(self,action,folder):
        factory=self.api_factory
        if factory is None:
            from xarm.wrapper import XArmAPI
            factory=XArmAPI
        arm=factory(self.cfg['robot_ip'],is_radian=False,do_not_open=True)
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        log=(folder/'execution.jsonl').open('a',buffering=1)
        batch_clock=time.monotonic()
        def record(event,**data):
            log.write(json.dumps(dict(time=time.time(),elapsed_seconds=time.monotonic()-batch_clock,
                                      event=event,**data))+'\n')
        def ok(label,code):
            record('sdk_return',operation=label,code=code)
            if code: raise UnsafePlan(f'{label}: SDK code {code}')
        started=False
        try:
            arm.connect()
            if not arm.connected: raise UnsafePlan('cannot connect to xArm')
            code,state=arm.get_state();ok('state',code)
            if state not in (0,2,4,5):
                raise UnsafePlan(f'xArm is moving, paused, or in an unsupported state ({state}); refusing to take over')
            code,errors=arm.get_err_warn_code();ok('errors',code)
            record('initial_controller_status',state=state,errors=errors)
            if any(errors): raise UnsafePlan(f'xArm reports error/warning {errors}; resolve the fault before restarting')
            if state in (4,5):
                # A new invocation starts a fresh batch, never resumes old commands.
                record('startup_recovery',initial_state=state)
                ok('discard stopped commands',arm.set_state(4))
            # Setting TCP offset can itself enter state 5. Enable/reset follows it.
            ok('TCP offset',arm.set_tcp_offset(self.cfg['tcp_offset_mm_deg'],is_radian=False,wait=True))
            code,before=arm.get_position(is_radian=False);ok('initial pose',code)
            record('initial_pose',tcp_mm_deg=before)
            if np.asarray(before).shape != (6,) or not np.isfinite(before).all():
                raise UnsafePlan('invalid initial TCP pose; refusing startup motion')
            if before[2] <= STARTUP_HOME_MIN_Z_MM:
                raise UnsafePlan(f'initial TCP Z is {before[2]} mm; automatic startup home requires Z > '
                                 f'{STARTUP_HOME_MIN_Z_MM} mm; manually clear the obstruction and raise the arm')
            home = None
            if not self._startup_home_done:
                home=self.cfg['startup_home']
                code,limited=arm.is_joint_limit(home['joint_angles_deg'],is_radian=False);ok('home joint limits',code)
                if limited != False: raise UnsafePlan('startup home joints exceed controller limits')
                code,home_pose=arm.get_forward_kinematics(home['joint_angles_deg'],
                    input_is_radian=False,return_is_radian=False);ok('home forward kinematics',code)
                if (np.asarray(home_pose).shape != (6,) or not np.isfinite(home_pose).all()
                        or home_pose[2] <= max(300.,self.cfg['transit_z_mm'])):
                    raise UnsafePlan('startup home TCP must be above 300 mm and transit height')
                record('startup_home_planned',home=home,expected_tcp_mm_deg=home_pose,
                       minimum_initial_z_mm=STARTUP_HOME_MIN_Z_MM)
            elif before[2] < self.cfg['transit_z_mm']-1:
                raise UnsafePlan('TCP below transit height after a previous batch; refusing automatic mid-run homing')
            label='SIMULATED SDK BATCH' if self.api_factory else 'ABOUT TO MOVE XARM7'
            print(f'{label}: enable arm, mode 0, state 0; {action["kind"]} {action["target"]}; complete batch, then detection.',flush=True)
            started=True
            for label,call in [('enable',lambda:arm.motion_enable(True)),('mode',lambda:arm.set_mode(0)),
                               ('state',lambda:arm.set_state(0))]:
                ok(label,call())
            # Bounded startup-only settling; never re-enable after a batch stop.
            for attempt in range(30):
                code,ready_state=arm.get_state();ok('ready state',code)
                if ready_state in (0,2): break
                if ready_state not in (4,5):
                    raise UnsafePlan(f'unexpected xArm state after initialization ({ready_state})')
                if attempt < 29: time.sleep(.1)
            else:
                raise UnsafePlan(f'xArm did not become ready after enable/mode/state initialization '
                                 f'(state {ready_state}); check the stop button and controller status')
            code,errors=arm.get_err_warn_code();ok('ready errors',code)
            record('ready_controller_status',state=ready_state,errors=errors)
            if any(errors): raise UnsafePlan(f'xArm reports error/warning after initialization: {errors}')
            if home is not None:
                label='SIMULATED SDK HOME' if self.api_factory else 'ABOUT TO MOVE XARM7'
                print(f'{label}: startup joint home {home["joint_angles_deg"]} degrees at '
                      f'{home["speed_deg_s"]:g} deg/s; expected TCP Z {home_pose[2]:.3f} mm, '
                      'then execute the planned batch.',flush=True)
                record('startup_home_start',home=home)
                ok('startup joint home',arm.set_servo_angle(angle=home['joint_angles_deg'],
                    speed=home['speed_deg_s'],mvacc=home['acceleration_deg_s2'],
                    wait=True,timeout=180,is_radian=False))
                # Startup-only verification, before any action or gripper stage.
                code,after=arm.get_position(is_radian=False);ok('startup home pose',code)
                code,joints=arm.get_servo_angle(is_radian=False);ok('startup home joints',code)
                record('startup_home_readback',tcp_mm_deg=after,joint_angles_deg=joints)
                if (np.asarray(after).shape != (6,) or not np.isfinite(after).all()
                        or after[2] <= 300 or after[2] < self.cfg['transit_z_mm']
                        or np.asarray(joints).shape != (7,) or not np.isfinite(joints).all()
                        or not np.allclose(joints,home['joint_angles_deg'],atol=1,rtol=0)):
                    raise UnsafePlan('startup home did not reach its joint target above transit height; refusing action')
                code,home_state=arm.get_state();ok('startup home state',code)
                code,errors=arm.get_err_warn_code();ok('startup home errors',code)
                if home_state not in (0,2) or any(errors):
                    raise UnsafePlan(f'xArm not ready after startup home: state {home_state}, errors {errors}')
                record('startup_home_complete',state=home_state)
                self._startup_home_done=True
            for label,call in [('gripper mode',lambda:arm.set_gripper_mode(0)),
                ('gripper enable',lambda:arm.set_gripper_enable(True)),
                ('gripper speed',lambda:arm.set_gripper_speed(round(self.cfg['gripper_speed_setting'])))]:
                ok(label,call())
            for stage_index,s in enumerate(action['stages']):
                record('stage_start',stage=s,stage_index=stage_index)
                if s['tcp_mm_deg'] is not None:
                    x,y,z,r,p,yaw=s['tcp_mm_deg']
                    code=arm.set_position(x=x,y=y,z=z,roll=r,pitch=p,yaw=yaw,speed=s['speed_mm_s'],
                                          mvacc=50,wait=True,timeout=180,is_radian=False)
                else: code=arm.set_gripper_position(s['gripper_pulse'],wait=True,timeout=30)
                ok(s['phase'],code);record('stage_complete',phase=s['phase'],stage_index=stage_index)
            record('batch_complete')
        except BaseException as exc:
            record('aborted',reason=str(exc))
            if started:
                try: record('stop',code=arm.set_state(4))
                except Exception as error: record('stop_failed',reason=str(error))
            raise
        finally:
            arm.disconnect();log.close()
