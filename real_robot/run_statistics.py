"""Run summaries from saved artifacts; no perception or hardware access."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def timed_call(run_dir, phase, function, *args, timing_details=None, **kwargs):
    start=time.monotonic(); start_unix=time.time(); status='failed'
    try:
        result=function(*args,**kwargs)
        status='complete'
        return result
    finally:
        path=Path(run_dir)/'run_timings.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as handle:
            handle.write(json.dumps(dict(phase=phase,start_unix=start_unix,
                elapsed_seconds=time.monotonic()-start,status=status,**(timing_details or {})))+'\n')


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def read_events(path):
    if not path.exists():return []
    events=[]
    for line in path.read_text().splitlines():
        try:events.append(json.loads(line))
        except json.JSONDecodeError:continue  # An interrupted last write is not a completed event.
    return events


def group_durations(rows, key):
    grouped={}
    for row in rows:
        name=row[key]
        bucket=grouped.setdefault(name,dict(calls=0,failed_calls=0,elapsed_seconds=0.))
        bucket['calls']+=1
        bucket['failed_calls']+=row.get('status')=='failed'
        bucket['elapsed_seconds']+=row['elapsed_seconds']
    return grouped


def sdk_stages(events):
    """Pair stage/home events; retain time spent in the stage that aborted."""
    rows=[];pending=None
    def stamp(event):return event.get('elapsed_seconds',event['time'])
    for event in events:
        kind=event['event']
        if kind in ('stage_start','startup_home_start','final_home_start'):
            pending=dict(phase=event['stage']['phase'] if kind=='stage_start' else ('final-home' if kind=='final_home_start' else 'startup-home'),
                         stage_index=event.get('stage_index'),start=stamp(event))
        elif pending and kind in ('stage_complete','startup_home_complete','final_home_complete','aborted'):
            rows.append(dict(phase=pending['phase'],stage_index=pending['stage_index'],
                status='failed' if kind=='aborted' else 'complete',
                elapsed_seconds=max(0.,stamp(event)-pending['start'])))
            pending=None
    if pending and events:
        rows.append(dict(phase=pending['phase'],stage_index=pending['stage_index'],status='incomplete',
                         elapsed_seconds=max(0.,stamp(events[-1])-pending['start'])))
    return rows


def write_run_summary(run_dir, *, started_unix, elapsed_seconds, status):
    """Counts describe dispatched batches, SDK stages, and target retirement separately."""
    from pipeline_core import write_json
    folder=Path(run_dir)
    state=read_json(folder/'state.json',{})
    metadata=read_json(folder/'run_metadata.json',{})
    config=read_json(folder/'pipeline_config.json',{})
    failure=read_json(folder/'failure.json',{})
    fixed=read_json(folder/state.get('fixed_plan_file','observations/0000/planning/fixed_execution_plan.json'))
    steps=None if fixed is None else fixed['steps']
    planned_knocks=None if steps is None else sum(s['kind']=='knock' for s in steps)
    planned_picks=None if steps is None else len(steps)
    timings=read_events(folder/'run_timings.jsonl')
    execution_times={row['action']:row for row in timings if row['phase']=='execution'}
    actions=[];stage_rows=[]
    for path in sorted((folder/'actions').glob('*/action.json')):
        action=read_json(path); index=int(path.parent.name)
        execution=execution_times.get(index)
        events=read_events(path.parent/'execution.jsonl')
        synthetic=(path.parent/'simulated_execution.json').exists()
        stages=sdk_stages(events)
        stage_rows.extend(dict(row,phase=action['kind']+'/'+row['phase']) for row in stages)
        attempted=execution is not None or bool(events) or synthetic
        complete=(execution is not None and execution['status']=='complete') or any(e['event']=='batch_complete' for e in events) or synthetic
        place=next((s['tcp_mm_deg'] for s in action['stages'] if s['phase']=='place'),None)
        actions.append(dict(index=index,target=action['target'],kind=action['kind'],
            object_class=action['source_pose']['kind'],bowl=action.get('bowl'),release_tcp_mm_deg=place,
            attempted=attempted,batch_completed=complete,
            execution_seconds=execution['elapsed_seconds'] if execution else None,
            robot_setup_and_teardown_seconds=(max(0.,execution['elapsed_seconds']-sum(r['elapsed_seconds'] for r in stages))
                                              if execution and events else None),
            sdk_action_stages_started=sum(e['event']=='stage_start' for e in events),
            knock_strokes_started=sum(e['event']=='stage_start' and e['stage']['phase']=='horizontal-knock' for e in events),
            knock_strokes_completed=sum(e['event']=='stage_complete' and e['phase']=='horizontal-knock' for e in events),
            synthetic=synthetic,stages=stages,outcome=read_json(path.parent/'outcome.json')))
    # State increments attempts before executor invocation, including startup failures.
    attempts=state.get('attempts',{})
    counts={kind:sum(value for key,value in attempts.items() if key.endswith(':'+kind)) for kind in ('pick','knock')}
    high_level=group_durations([r for r in timings if not r.get('detail',False)],'phase')
    for phase in ('observation','planning','selection','waypoints','execution','assessment','final_home'):
        high_level.setdefault(phase,dict(calls=0,failed_calls=0,elapsed_seconds=0.))
    accounted=sum(r['elapsed_seconds'] for r in high_level.values())
    subprocesses=read_events(folder/'perception_timings.jsonl')
    subprocess_rows=[dict(row,phase=Path(row['log']).stem,
                          status='complete' if row['returncode']==0 else 'failed') for row in subprocesses]
    captures=[r for r in timings if r['phase']=='capture']
    observation_total=high_level['observation']['elapsed_seconds']
    perception_detail=group_durations(subprocess_rows+captures,'phase')
    # Keep nested timing separate: capture/process time is already inside observation.
    perception_detail['other_observation_work']=dict(elapsed_seconds=max(0.,observation_total-
        sum(r['elapsed_seconds'] for r in subprocess_rows+captures)))
    completed=state.get('completed',[])
    report=dict(schema_version=1,run_directory=str(folder.resolve()),
        scene_id=config.get('experiment_reference',{}).get('scene_id'),mode=metadata.get('mode','unknown'),
        status=state.get('status',status),reason=state.get('reason') or failure.get('error'),
        started_at=datetime.fromtimestamp(started_unix,timezone.utc).isoformat(),
        finished_at=datetime.fromtimestamp(started_unix+elapsed_seconds,timezone.utc).isoformat(),
        planned=dict(objects=planned_picks,picks=planned_picks,knocks=planned_knocks,
                     total_actions=None if steps is None else planned_picks+planned_knocks,
                     order=None if steps is None else [s['target'] for s in steps]),
        actual=dict(total_attempts=sum(counts.values()),pick_attempts=counts['pick'],knock_attempts=counts['knock'],
            knock_retries=sum(max(0,v-1) for k,v in attempts.items() if k.endswith(':knock')),
            attempts_by_target=attempts,completed_batches=sum(a['batch_completed'] for a in actions),
            completed_pick_batches=sum(a['batch_completed'] and a['kind']=='pick' for a in actions),
            completed_knock_batches=sum(a['batch_completed'] and a['kind']=='knock' for a in actions),
            physical_batches_with_action_stage_started=sum(a['sdk_action_stages_started']>0 for a in actions) if metadata.get('mode')=='physical' else 0,
            knock_strokes_started=sum(a['knock_strokes_started'] for a in actions),
            knock_strokes_completed=sum(a['knock_strokes_completed'] for a in actions),
            objects_removed_from_source=len(completed),completed_targets=completed,
            pick_completion_policy=state.get('pick_completion_policy','observed_source_removal'),
            retired_targets=len(completed),
            remaining_targets=None if steps is None else [s['target'] for s in steps if s['target'] not in completed],
            observations_attempted=high_level['observation']['calls'],
            observations_completed=high_level['observation']['calls']-high_level['observation']['failed_calls']),
        timing=dict(total_seconds=elapsed_seconds,phases=high_level,
                    setup_and_other_seconds=max(0.,elapsed_seconds-accounted),
                    perception_detail=perception_detail,robot_stage_detail=group_durations(stage_rows,'phase'),
                    robot_setup_and_teardown_seconds=sum(a['robot_setup_and_teardown_seconds'] or 0 for a in actions)),
        final_home=dict(state.get('final_home',{'status':'not_run'}),
                        stages=sdk_stages(read_events(folder/'final_home/execution.jsonl'))),
        actions=actions,
        definitions=dict(planned_total_actions='One Pick batch per object, plus one Knock batch for each planned Knock. Retries are additional.',
            attempts='Executor invocations, including startup failures. Preview batches are not attempts. SDK stroke counters distinguish commanded slides.',
            completion=('Successful Pick drop/open/retreat retires the target; grasp and placement are not visually verified.'
                        if state.get('pick_completion_policy')=='successful_drop_and_retreat_batch' else
                        'Confirmed source removal; correct placement inside a bowl is not automatically verified.'),
            timing='Monotonic elapsed seconds, including failed calls. High-level phases are exclusive. Perception and robot stage details are nested; do not add them to the total.',
            simulation='Synthetic execution tests control flow, not physical motion or contact timing.'))
    write_json(folder/'summary.json',report)
    def value(v):return 'not available' if v is None else str(v)
    lines=[f"Run summary — {report['scene_id'] or folder.name}",f"Status: {report['status']} | Mode: {report['mode']}",
           f"Total elapsed: {elapsed_seconds:.3f} s",
           f"Final return home: {report['final_home']['status']}",'']
    if report['reason']:lines += [f"Stop reason: {report['reason']}",'']
    lines += ['| Statistic | Count |','|---|---:|',
        f"| Planned total actions (Knock + Pick) | {value(report['planned']['total_actions'])} |",
        f'| Planned Knocks | {value(planned_knocks)} |',f'| Planned Picks | {value(planned_picks)} |',
        f"| Actual Knock attempts / retries | {counts['knock']} / {report['actual']['knock_retries']} |",
        f"| Actual Pick attempts | {counts['pick']} |",
        f"| Completed batches | {report['actual']['completed_batches']} |",
        f"| SDK Knock slides started / completed | {report['actual']['knock_strokes_started']} / {report['actual']['knock_strokes_completed']} |",
        f'| Retired targets | {len(completed)} |','',
        'Attempts count executor invocations; startup failures can occur before a slide starts. '+report['definitions']['completion'],
        '', '| Phase | Calls | Failed | Seconds |','|---|---:|---:|---:|']
    for phase,row in high_level.items():
        lines.append(f"| {phase} | {row['calls']} | {row['failed_calls']} | {row['elapsed_seconds']:.3f} |")
    lines += [f"| setup and other work | — | — | {report['timing']['setup_and_other_seconds']:.3f} |",'',
              'Nested detail (already included above):','', '| Detail | Seconds |','|---|---:|']
    for category,rows in [('perception',perception_detail),('robot',report['timing']['robot_stage_detail'])]:
        lines.extend(f"| {category}: {phase} | {row['elapsed_seconds']:.3f} |" for phase,row in rows.items())
    lines += [f"| robot: connection, setup and teardown | {report['timing']['robot_setup_and_teardown_seconds']:.3f} |"]
    lines += ['', 'Per-action destinations, stage durations and outcomes are in summary.json.',
              'Synthetic timings do not estimate physical execution time.' if report['mode']=='simulation' else '']
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    return report
