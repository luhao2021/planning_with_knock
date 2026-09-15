"""Offline planner/decision logic shared by physical and simulated runs."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from pose_to_plan import (base_to_optical, correct_pose_overlaps, solve_scene,
                          physical_plan, plot_instance, plot_solution_overlay,
                          plot_solution_sequence)
from cuboid_knock_pick_solvers import knock_blockers, pick_actions_for_object
from xarm_executor import UnsafePlan


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n'); temp.replace(path)


def snapshot_plan(objects, folder, workflow, *, render=True, execution_cfg=None):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        r = np.asarray(obj['T_base_object'])[:3, :3]
        if abs(r[2, 2]) < .95:
            raise UnsafePlan(f'{obj["name"]} tipped; the equal-height planner cannot execute its old grasp')
    corrected, overlaps = correct_pose_overlaps(objects, workflow['max_pose_correction_mm']/1000)
    shrink = workflow['footprint_shrink_mm']/1000
    finger_thickness = float(workflow.get('finger_thickness_mm',20))/1000
    if not np.isfinite(finger_thickness) or finger_thickness <= 0:
        raise UnsafePlan('finger_thickness_mm must be finite and positive')
    payload = dict(objects=copy.deepcopy(objects), planning_objects=corrected,
                   T_base_camera_color_optical_frame=base_to_optical().tolist(),
                   finger_thickness_mm=finger_thickness*1000,
                   footprint_shrink_total_m=shrink, jaw_opening_m=workflow['jaw_opening_mm']/1000,
                   raw_pose_overlaps=overlaps, solver_status='optimal', plan_verified=True,
                   optimal_clearances=0, plan=[], action_details=[], immediately_feasible_actions=[])
    planning_gripper=None
    if execution_cfg is not None:
        planning_gripper=dict(open_aperture_mm=execution_cfg['gripper_open_mm'],
                              finger_thickness_mm=finger_thickness*1000,
                              margin_mm=float(workflow.get('planning_grasp_margin_mm',3)),
                              usable_jaw_opening_mm=min(workflow['jaw_opening_mm'],execution_cfg['usable_jaw_opening_mm']),
                              footprint_shrink_mm=workflow['footprint_shrink_mm'])
        payload['planning_gripper_envelope']=planning_gripper
        payload['planning_pick_model']='solid_open_gripper_envelope_plus_margin_on_nominal_observed_boxes'
    if objects:
        rule, result = solve_scene(corrected, shrink, workflow['jaw_opening_mm']/1000,
                                   finger_thickness_m=finger_thickness,planning_gripper=planning_gripper,observed_objects=objects)
        if result.status != 'optimal':
            raise UnsafePlan(f'solver did not prove an optimum: {result.status}')
        def serial(index, action):
            return dict(target=rule.names[index], kind=action.kind, witness=action.witness,
                        blockers=[rule.names[j] for j in sorted(action.blockers)])
        payload.update(solver_status=result.status, optimal_clearances=int(result.optimum),
                       plan=physical_plan(rule, result.selected_actions, result.order),
                       action_details=[dict(step=s, **serial(i, result.selected_actions[i]))
                                       for s,i in enumerate(result.order, 1)])
        payload['initial_pick_options'] = {rule.names[i]: [serial(i,a) for a in
            (result.extra['initial_planning_pick_actions'][i] if planning_gripper is not None else pick_actions_for_object(rule.geometry,i))]
            for i in range(len(rule.geometry.rects))}
        feasible = []
        for i, actions in rule.actions.items():
            feasible.extend(serial(i, a) for a in actions if a.kind == 'pick' and not a.blockers)
            # Preserve every feasible world-direction option for same-target retries.
            for d in ('+x', '-x', '+y', '-y'):
                if not knock_blockers(rule.geometry, i, d):
                    feasible.append(dict(target=rule.names[i], kind='knock', witness='knock-'+d, blockers=[]))
        payload['immediately_feasible_actions'] = feasible
        if render:
            for d in ('layouts', 'solution_overlays', 'solution_sequences'): (folder/d).mkdir(exist_ok=True)
            plot_instance(rule.geometry, folder/'layouts/learned_pose_scene.png')
            plot_solution_overlay(rule, result.selected_actions, result.order,
                                  folder/'solution_overlays/learned_pose_scene_ilp_highs.png', 'ILP/HiGHS')
            plot_solution_sequence(rule, result.selected_actions, result.order,
                                   folder/'solution_sequences/learned_pose_scene_ilp_highs.png', 'ILP/HiGHS')
    write_json(folder/'layout_and_plan.json', payload)
    (folder/'plan.txt').write_text('\n'.join(payload['plan'])+'\n')
    return payload


def inside_source(obj, workflow):
    xy = np.asarray(obj['base_center_m'][:2])*1000
    lo, hi = workflow['source_xy_min_mm'], workflow['source_xy_max_mm']
    return bool(np.all(xy >= lo) and np.all(xy <= hi))


def assess_action(before, after, action, cfg, workflow):
    old={o['name']:o for o in before}; new={o['name']:o for o in after}
    target=action['target']
    if target not in new:
        raise UnsafePlan(f'{target} was not detected after the action; absence alone does not prove removal')
    if action['kind']=='pick' and new[target].get('perception_status')=='source_removed':
        return dict(target=target,completed=True,decision='source_removal_confirmed_by_depth',
                    source_removal_evidence=new[target]['source_removal_evidence'])
    center=np.asarray(new[target]['base_center_m'],float)
    if center.shape!=(3,) or not np.isfinite(center).all():
        raise UnsafePlan(f'{target}: invalid detected centre after action')
    delta=(np.asarray(new[target]['base_center_m'])-old[target]['base_center_m'])*1000
    result=dict(target=target,delta_base_mm=delta.tolist(),completed=False)
    if action['kind']=='knock':
        outward=float(delta @ np.asarray(action['base_direction']))
        result.update(outward_mm=outward,progress=outward>=workflow['min_knock_progress_mm'],
                      decision='check_fixed_grasp_then_pick_or_retry_fixed_knock')
    else:
        # A completed Pick is removal from the source scene. Bowl accuracy is
        # not an execution gate and is not compared with the TCP release point.
        in_source=inside_source(new[target],workflow)
        result.update(completed=not in_source,detected_in_source=in_source,
                      decision='target_removed_from_source' if not in_source else 'pick_target_still_in_source')
    return result


def validate_workflow(workflow, cfg):
    for key in ('max_actions', 'max_knock_attempts', 'max_pick_attempts'):
        value = workflow[key]
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= (1000 if key=='max_actions' else 100):
            raise UnsafePlan(f'{key} must be an integer between 1 and {1000 if key=="max_actions" else 100}')
    for key in ('min_knock_progress_mm',):
        if not np.isfinite(workflow[key]) or not 0 < workflow[key] <= 50:
            raise UnsafePlan(f'invalid workflow tolerance {key}')
    if not 0 < workflow['jaw_opening_mm'] <= cfg['usable_jaw_opening_mm']:
        raise UnsafePlan('solver opening exceeds configured physical gripper opening')
    lo,hi = np.asarray(workflow['source_xy_min_mm']), np.asarray(workflow['source_xy_max_mm'])
    if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo,hi]).all() or not np.all(lo < hi):
        raise UnsafePlan('invalid source workspace rectangle')


def run_loop(perception, executor, cfg, workflow, run_dir, *, preview=False, render=True,
             single_action=False):
    validate_workflow(workflow, cfg)
    run_dir = Path(run_dir)
    from fixed_execution import freeze_plan, latest_layout, select_fixed_candidate
    state = dict(status='running', completed=[], attempts={}, no_progress={}, pending=None,
                 actions_executed=0,plan_cursor=0,planning_policy='one_initial_plan')
    def save(): write_json(run_dir/'state.json', state)
    try:
        observed = perception.observe(0, run_dir/'observations/0000')
        names = [o['name'] for o in observed if inside_source(o, workflow)]
        expected = set(perception.initial_names)
        if set(names) != expected or len(names) != len(set(names)):
            raise UnsafePlan(f'initial source identities differ from configuration: expected {sorted(expected)}, got {names}')
        active = expected.copy()
        pose_observations={o['name']:0 for o in observed}
        initial=snapshot_plan([o for o in observed if o['name'] in active],
                              run_dir/'observations/0000/planning',workflow,render=render,execution_cfg=cfg)
        fixed_plan=freeze_plan(initial,cfg)
        if len(fixed_plan)!=len(active) or {e['target'] for e in fixed_plan}!=active:
            raise UnsafePlan('initial plan does not cover each configured target exactly once')
        state['fixed_plan_file']='observations/0000/planning/fixed_execution_plan.json'
        write_json(run_dir/state['fixed_plan_file'],dict(policy='one_initial_plan',steps=fixed_plan))
        print('[PLAN] Initial plan fixed: '+' -> '.join(e['target'] for e in fixed_plan),flush=True)
        save()
        for step in range(workflow['max_actions']+1):
            by_name = {o['name']: o for o in observed}
            if not active <= by_name.keys():
                raise UnsafePlan('active object missing from perception; refusing to continue')
            current = [by_name[name] for name in sorted(active)]
            if any(not inside_source(o, workflow) for o in current):
                raise UnsafePlan('active box left the configured source workspace')
            layout = latest_layout(current,workflow)
            layout['pose_observation_indices']={o['name']:pose_observations[o['name']] for o in current}
            write_json(run_dir/f'observations/{step:04d}/execution_scene.json',layout)
            if not active:
                state['status'] = 'complete'; state.pop('active_action', None); save(); return state
            if single_action and state['actions_executed'] == 1:
                state['status'] = 'single_action_complete'
                state.pop('active_action', None)
                save(); return state
            if step >= workflow['max_actions']:
                raise UnsafePlan('maximum total action count reached')
            entry=fixed_plan[state['plan_cursor']]
            candidate,checks=select_fixed_candidate(entry,layout,cfg,
                has_knocked=state['attempts'].get(entry['target']+':knock',0)>0)
            key = candidate['target']+':'+candidate['kind']
            limit = workflow['max_knock_attempts'] if candidate['kind'] == 'knock' else workflow['max_pick_attempts']
            if state['attempts'].get(key, 0) >= limit:
                raise UnsafePlan(f'{key}: attempt limit reached ({state["attempts"][key]}/{limit})')
            selection=dict(selected=candidate,fixed_plan_step=state['plan_cursor']+1,
                           fixed_plan_entry=entry,checks=checks,
                           attempt_number=state['attempts'].get(key,0)+1,attempt_limit=limit)
            write_json(run_dir/f'observations/{step:04d}/execution_selection.json',selection)
            print(f'[ACTION {step}] fixed plan step {state["plan_cursor"]+1}/{len(fixed_plan)}: '
                  f'{candidate["kind"]} {candidate["target"]}; attempt '
                  f'{selection["attempt_number"]}/{limit}',flush=True)
            tolerated=checks.get('pick_clearance',{}).get('tolerated_neighbors',[])
            if tolerated:
                print(f'[POSE NOISE] Continuing fixed Pick: estimated overlap within '
                      f'{cfg["pick_overlap_tolerance_mm"]:g} mm tolerance for {tolerated}',flush=True)
            from batch_executor import build_batch
            bowl = cfg.get('instance_to_bowl', {}).get(candidate['target'],
                        'bowl_1' if len(state['completed']) % 2 == 0 else 'bowl_2')
            action_dir = run_dir/f'actions/{step:04d}'
            if preview:
                from batch_executor import height_available
                fields=['knock.contact_tcp_z_mm'] if candidate['kind']=='knock' else ['pick_tcp_z_mm','release_tcp_z_mm']
                missing=[key for key in fields if not height_available(cfg,key)]
                if missing:
                    write_json(action_dir/'action_request.json',dict(candidate=candidate,missing_values=missing))
                    state.update(status='preview',missing_motion_values=missing)
                    save();return state
            action = build_batch(layout, candidate, cfg, bowl=bowl if candidate['kind']=='pick' else None)
            write_json(action_dir/'action.json', action)
            if preview:
                state['status'] = 'preview'; state['next_action'] = str(action_dir/'action.json'); save(); return state
            state['attempts'][key] = state['attempts'].get(key, 0)+1
            state['status'] = 'executing'; state['active_action'] = str(action_dir/'action.json'); save()
            executor.execute(action, action_dir)
            state['actions_executed'] += 1; state['status'] = 'observing'; save()
            # This is the only observation after all motion stages have returned.
            # Knock needs only its target. After Pick, confirm removal and refresh
            # the next fixed-plan target in the same capture. Other poses remain
            # their last accepted estimates, never disguised as fresh detections.
            targets=[action['target']]
            next_cursor=state['plan_cursor']+1
            if action['kind']=='pick' and next_cursor<len(fixed_plan) and not single_action:
                targets.append(fixed_plan[next_cursor]['target'])
            latest = perception.observe(step+1, run_dir/f'observations/{step+1:04d}',targets=targets,
                                        active_objects=current,last_action=action)
            outcome = assess_action(current, latest, action, cfg, workflow)
            pick_exhausted=(action['kind']=='pick' and not outcome['completed']
                            and state['attempts'][key]>=workflow['max_pick_attempts'])
            if pick_exhausted:outcome['decision']='stop_pick_attempt_limit'
            write_json(action_dir/'outcome.json', outcome)
            if pick_exhausted:
                raise UnsafePlan(f'{key}: attempt limit reached ({state["attempts"][key]}/{limit}); '
                                 'Pick target remains in source area; no retry')
            if outcome['completed']:
                active.remove(action['target']); state['completed'].append(action['target']); state['pending'] = None
                state['plan_cursor']+=1
            else:
                state['pending'] = {k:action[k] for k in ('target','kind','base_direction') if k in action}
                if action['kind'] == 'knock':
                    state['no_progress'][key] = 0 if outcome['progress'] else state['no_progress'].get(key, 0)+1
                    # Progress is diagnostic; noisy estimates do not introduce a
                    # second retry cutoff before the configured Knock attempt limit.
            fresh={o['name']:o for o in latest}
            missing=set(targets)-set(fresh)
            if missing:raise UnsafePlan(f'requested action target missing from perception: {sorted(missing)}')
            by_name.update(fresh)
            pose_observations.update({name:step+1 for name in fresh})
            observed = list(by_name.values()); state['status'] = 'running'; save()
    except BaseException as exc:
        state.update(status='stopped', reason=str(exc)); save(); raise
