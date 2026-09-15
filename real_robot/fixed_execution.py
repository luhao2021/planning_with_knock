"""Execute a fixed target order, grasp witnesses and object-local Knock directions.

Fresh poses update waypoints and the clearance check for the fixed Pick. No solver, action search,
new grasp generation, overlap correction, or rendering is performed here.
"""
import copy
import numpy as np
from pick_clearance import pick_clearance
from xarm_executor import UnsafePlan


def freeze_plan(initial,cfg):
    steps=[]
    for selected in initial['action_details']:
        target=selected['target']
        if selected['kind']=='pick':
            pick=copy.deepcopy(selected)
        else:
            # Mandatory Pick after Knock: select its witness once from the
            # initial solver geometry. Neighbor clearance is checked after each
            # stroke; a blocked fixed witness requests another same-direction stroke.
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
