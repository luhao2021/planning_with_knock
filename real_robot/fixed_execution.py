"""Freeze target order, object-local Knock directions and follow-up grasps once.

A Knock exposes its leading end for the mandatory Pick. Fresh poses subsequently
update waypoints and clearance for that fixed grasp without rerunning the solver.
"""
import copy
import numpy as np
from pick_clearance import pick_clearance
from xarm_executor import UnsafePlan


def exposed_end_pick(initial,selected,cfg):
    """Place the 30 mm contact window at the end exposed by the planned Knock.

    Coordinates are object-local, with the same footprint shrink as the initial
    planner. No execution clearance rule is used to construct this grasp.
    """
    direction=selected['witness'][6:]
    if direction not in ('+x','-x','+y','-y'):raise UnsafePlan('invalid local Knock direction')
    obj=next(o for o in initial['objects'] if o['name']==selected['target'])
    tangent_axis=0 if direction[1]=='x' else 1
    closing_axis=1-tangent_axis
    width=obj['dimensions_m'][closing_axis]*1000
    span=obj['dimensions_m'][tangent_axis]-initial.get('footprint_shrink_total_m',0.)
    # Some short-axis Knocks cannot support a transverse grasp with this jaw.
    # Keep their original usable grasp instead of commanding an oversized opening.
    if width+2>min(cfg['gripper_open_mm'],cfg['usable_jaw_opening_mm']) or span<.030:
        return None
    placement=span/2-.030 if direction[0]=='+' else -span/2
    return dict(target=selected['target'],kind='pick',
                witness=f'pick-{"xy"[closing_axis]}@{placement:.9f}',blockers=[])


def freeze_plan(initial,cfg):
    steps=[]
    for selected in initial['action_details']:
        target=selected['target']
        if selected['kind']=='pick':
            pick=copy.deepcopy(selected)
        else:
            # Freeze a leading-end grasp in the same local direction as the
            # Knock, instead of arbitrarily taking the first (negative-end) option.
            pick=exposed_end_pick(initial,selected,cfg)
            if pick is None:
                options=initial.get('initial_pick_options',{}).get(target,[])
                pick=copy.deepcopy(options[0]) if options else None
            if pick is None:raise UnsafePlan(f'{target}: initial plan has no usable mandatory Pick grasp')
        pick.pop('step',None);pick['blockers']=[]
        steps.append(dict(target=target,kind=selected['kind'],pick=pick,
                          knock=(dict(target=target,kind='knock',witness=selected['witness'],blockers=[])
                                 if selected['kind']=='knock' else None)))
    return steps


def latest_layout(objects,workflow):
    for obj in objects:
        t=np.asarray(obj['T_base_object'],float)
        if t.shape!=(4,4) or not np.isfinite(t).all() or abs(t[2,2])<.95:
            raise UnsafePlan(f'{obj["name"]}: invalid or tipped pose for fixed-plan execution')
    return dict(objects=copy.deepcopy(objects),finger_thickness_mm=workflow.get('finger_thickness_mm',20))


def select_fixed_candidate(entry,layout,cfg,*,has_knocked):
    # The initial solver already chose the outward Knock. Execute that fixed
    # direction; do not rerun its abstract corridor test on noisy fresh poses.
    if entry['kind']=='knock' and not has_knocked:
        return copy.deepcopy(entry['knock']),dict(decision='initial_planned_knock')
    check=pick_clearance(layout,entry['pick'],cfg)
    if check['passed']:
        return copy.deepcopy(entry['pick']),dict(pick_clearance=check)
    if entry['kind']=='knock':
        return copy.deepcopy(entry['knock']),dict(pick_clearance=check)
    raise UnsafePlan(f'{entry["target"]}: fixed planned Pick is blocked: {check["reasons"]}; '
                     f'blockers={check["blockers"]}; plan will not be changed')
