#!/usr/bin/env python3
"""One initial plan: capture/track/FoundationPose/solve, then batch/observe/retry-or-advance.

Default is a live perception + trajectory preview with no robot connection.
--simulate exercises the full decision loop using synthetic motion outcomes.
Only --execute imports the xArm SDK, after validating the physical configuration.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import uuid
import time

import numpy as np
from scipy.spatial.transform import Rotation
from batch_executor import XArmBatchExecutor, checked_batch_config, physical_config_report, height_available
from pipeline_core import run_loop, write_json
from pose_to_plan import convert, apply_base_pose_offset, checked_base_pose_offset_mm
from xarm_executor import UnsafePlan

ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT/p


class LivePerception:
    def __init__(self, config, run_dir, recorded_capture=None, *, workflow=None):
        self.recorded_capture = recorded_capture
        self.config = config; self.run_dir = Path(run_dir)
        self.base_pose_offset_mm = checked_base_pose_offset_mm(config.get('base_pose_offset_mm',[0,0,0]))
        self.frames = self.run_dir/'frames'; self.frames.mkdir(parents=True)
        self.points = self.run_dir/'initial_points.json'
        self.automatic_scene=config.get('automatic_scene',{})
        self.workflow=workflow
        if self.automatic_scene.get('enabled',False):
            if workflow is None:raise UnsafePlan('automatic initialization needs the source workspace configuration')
            if self.automatic_scene.get('kind')!='protex':raise UnsafePlan('automatic scene initialization currently supports Protex-only scenes')
            maximum=self.automatic_scene.get('max_instances',4)
            if not isinstance(maximum,int) or isinstance(maximum,bool) or not 1<=maximum<=4:
                raise UnsafePlan('automatic Protex inventory limit must be 1..4')
            expected=self.automatic_scene.get('expected_instances')
            if expected is not None and (not isinstance(expected,int) or isinstance(expected,bool) or not 1<=expected<=maximum):
                raise UnsafePlan('expected automatic instance count must be 1..max_instances')
            if not resolve(self.automatic_scene['mesh']).is_file():raise FileNotFoundError(self.automatic_scene['mesh'])
            self.initial_names=set();self.specs={}
        else:
            shutil.copy2(resolve(config['points']), self.points)
            self.initial_names = set(json.loads(self.points.read_text()))
            self.specs = config['objects']
            if self.initial_names != set(self.specs):
                raise UnsafePlan('initial prompt names and configured object names differ')
            for name, meta in self.specs.items():
                if name != meta['class']+'_'+str(meta['instance_id']):
                    raise UnsafePlan('object name must be class_instance_id')
                if not resolve(meta['mesh']).is_file(): raise FileNotFoundError(meta['mesh'])
        for key in ('python', 'sam2_python', 'checkpoint', 'library', 'refiner', 'scorer'):
            if not resolve(config[key]).is_file(): raise FileNotFoundError(config[key])
        write_json(self.run_dir/'object_specs.json', self.specs)

    def run(self, argv, logfile):
        started=time.time();clock=time.monotonic()
        with Path(logfile).open('w') as log:
            process = subprocess.run([str(x) for x in argv], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        with (self.run_dir/'perception_timings.jsonl').open('a') as timing:
            timing.write(json.dumps(dict(log=str(logfile),start_unix=started,
                elapsed_seconds=time.monotonic()-clock,returncode=process.returncode))+'\n')
        if process.returncode:
            tail = Path(logfile).read_text()[-2000:]
            raise RuntimeError(f'perception command failed ({process.returncode}); see {logfile}\n{tail}')

    def observe(self, index, folder, *, targets=None, active_objects=None, last_action=None):
        import cv2
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        requested=sorted(self.specs if targets is None else set(targets))
        if targets is not None and (not requested or not set(requested)<=set(self.specs)):
            raise UnsafePlan('observation targets must be configured object identities')
        path = folder/'capture.npz'
        print(f'[OBSERVE {index}] '+('Recorded RGB-D input' if self.recorded_capture is not None else 'One RGB-D capture')+', then SAM 2 and FoundationPose'+(' for '+', '.join(requested) if targets is not None else ' for all boxes'), flush=True)
        if self.recorded_capture is not None:
            if index != 0: raise UnsafePlan('recorded input is only supported for a non-executing preview')
            shutil.copy2(self.recorded_capture, path)
        else:
            from capture_rgbd import capture
            capture(path)
        with np.load(path) as data:
            rgb_bgr = data['color']
            cv2.imwrite(str(folder/'capture_rgb.png'), rgb_bgr)
            cv2.imwrite(str(self.frames/f'{index:06d}.jpg'), rgb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        c = self.config
        if index==0 and getattr(self,'automatic_scene',{}).get('enabled',False):
            from automatic_scene import prepare_grid,select_instances
            automatic=folder/'automatic_scene'
            for attempt,spacing in enumerate((5,3)):
                trial=automatic if attempt==0 else automatic/'recovery'
                grid,count=prepare_grid(path,self.workflow,self.base_pose_offset_mm,trial,spacing_px=spacing)
                records=[];masks={}
                if count:
                    command=[resolve(c['sam2_python']),ROOT/'real_robot/run_sam2_masks.py',
                             '--capture',path,'--checkpoint',resolve(c['checkpoint']),
                             '--point-grid',grid,'--out',trial/'proposals']
                    if attempt:command.append('--refine-proposals')
                    self.run(command,trial/'sam2_automatic.log')
                    records=json.loads((trial/'proposals/sam2_candidates.json').read_text())
                    with np.load(trial/'proposals/sam2_masks.npz') as data:
                        masks={name:data[name] for name in data.files}
                try:
                    points=select_instances(path,self.workflow,self.base_pose_offset_mm,masks,records,trial,
                        max_instances=self.automatic_scene.get('max_instances',4),
                        expected_instances=self.automatic_scene.get('expected_instances'))
                    write_json(automatic/'initialization.json',dict(status='accepted',attempts=attempt+1,
                               selected_from=str(trial),expected_instances=self.automatic_scene.get('expected_instances'),
                               generated_instances=len(points),new_capture=False))
                    break
                except UnsafePlan as error:
                    write_json(trial/'selection_failure.json',dict(error=str(error)))
                    if attempt:
                        write_json(automatic/'initialization.json',dict(status='failed',attempts=2,error=str(error)))
                        raise
                    print('[SCENE RECOVERY] Retrying automatic seeds once with denser SAM2 prompts on the same capture.',flush=True)
            write_json(self.points,points)
            self.initial_names=set(points)
            self.specs={name:dict(**{'class':'protex'},instance_id=name.split('_')[-1],mesh=self.automatic_scene['mesh']) for name in points}
            write_json(self.run_dir/'object_specs.json',self.specs)
            print(f'[SCENE] Automatically initialized {len(points)} Protex boxes; seeds and object specs saved in this run.',flush=True)
            if not points:
                write_json(folder/'observed_objects_uncorrected.json',[])
                write_json(folder/'observed_objects.json',[])
                return []
        requested=sorted(self.specs if targets is None else set(targets))
        write_json(folder/'perception_scope.json',dict(estimated_instances=requested,
                   policy='all_initial_then_action_targets',sam2_tracks_all_identities=True))
        tracking=[resolve(c['sam2_python']), ROOT/'real_robot/run_sam2_tracking.py',
                  '--frames', self.frames, '--points', self.points,
                  '--checkpoint', resolve(c['checkpoint']), '--out', folder/'masks']
        if targets is not None:tracking.extend(['--required-instances',*requested])
        if active_objects is not None:
            tracking.extend(['--defer-identity-check','--active-instances',*[o['name'] for o in active_objects]])
        self.run(tracking,folder/'sam2.log')
        removed=[];association=None
        if active_objects is not None:
            from instance_identity import associate_masks
            masks_path=folder/'masks/instance_masks.npz'
            shutil.copy2(masks_path,folder/'masks/tracker_masks.npz')
            previous=getattr(self,'_previous_observation_folder',None)
            masks,association=associate_masks(path,masks_path,active_objects,self.workflow,
                self.base_pose_offset_mm,last_action,previous/'capture.npz' if previous else None,
                mask_references={name:r['last_mask_center_xy_mm'] for name,r in getattr(self,'_identity_registry',{}).items()
                                 if r.get('status')!='removed' and 'last_mask_center_xy_mm' in r})
            write_json(folder/'identity_association.json',association)
            np.savez_compressed(masks_path,**masks)
            raw_preview=folder/'masks/tracked_masks_preview.png'
            if raw_preview.exists():shutil.copy2(raw_preview,folder/'masks/tracker_masks_preview.png')
            preview=rgb_bgr.copy();mask_records=[]
            for name,mask in masks.items():
                ys,xs=np.where(mask)
                if len(xs):
                    box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                    cv2.rectangle(preview,tuple(box[:2]),tuple(box[2:]),(0,0,255),2)
                    cv2.putText(preview,name,tuple(box[:2]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
                    mask_records.append(dict(instance_id=name,pixels=len(xs),bbox_xyxy=box,spatially_associated=True))
            raw_records=folder/'masks/instance_masks.json'
            if raw_records.exists():shutil.copy2(raw_records,folder/'masks/tracker_masks.json')
            write_json(raw_records,mask_records)
            cv2.imwrite(str(raw_preview),preview)
            if association['removal'].get('confirmed'):
                obj=copy.deepcopy(next(o for o in active_objects if o['name']==last_action['target']))
                obj['perception_status']='source_removed'
                obj['source_removal_evidence']=association['removal']
                removed=[obj];requested=[n for n in requested if n!=obj['name']]
                print(f"[IDENTITY] Retired {obj['name']}: depth confirms removal from source.",flush=True)
            missing=set(requested)-set(masks)
            if missing:
                raise UnsafePlan(f'no source mask matched the saved identity for {sorted(missing)}; see {folder}/identity_association.json')
            write_json(folder/'perception_scope.json',dict(estimated_instances=requested,
                active_instances=[o['name'] for o in active_objects if o['name'] not in {r['name'] for r in removed}],
                retired_instances=[o['name'] for o in removed],policy='spatially_bound_active_source_identities'))
        def container_path(path): return '/project/'+str(Path(path).resolve().relative_to(ROOT))
        cmd = ['docker','run','--rm','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}',
               '-e','PYTHONPATH=/project/real_robot/external/foundation-pose-inference-library/python/src',
               '-e','LD_LIBRARY_PATH=/project/real_robot/external/foundation-pose-inference-library/build:/usr/lib/x86_64-linux-gnu',
               '-v',f'{ROOT}:/project','-w','/project',c['docker_image'],
               'python','real_robot/run_foundationpose.py','--capture',container_path(path),
               '--masks',container_path(folder/'masks/instance_masks.npz'),
               '--mesh',container_path(resolve(next(iter(self.specs.values()))['mesh'])),
               '--object-specs',container_path(self.run_dir/'object_specs.json'),
               '--library',container_path(resolve(c['library'])),
               '--refiner',container_path(resolve(c['refiner'])), '--scorer',container_path(resolve(c['scorer'])),
               '--engine-cache',container_path(self.run_dir/'engine_cache'),
               '--n-hypotheses',str(c.get('n_hypotheses',128)), '--n-refine',str(c.get('n_refine',5)),
               '--out',container_path(folder/'foundationpose')]
        if targets is not None:cmd.extend(['--instances',*requested])
        if requested:self.run(cmd, folder/'foundationpose.log')
        else:write_json(folder/'foundationpose/foundationpose_instances.json',dict(instances=[]))
        records = json.loads((folder/'foundationpose/foundationpose_instances.json').read_text())['instances']
        returned=[f"{r['class']}_{r['instance_id']}" for r in records]
        if len(returned)!=len(requested) or set(returned)!=set(requested):
            raise UnsafePlan(f'FoundationPose did not return the requested identities: {requested}; got {returned}')
        from perception_validation import validate_model_poses,PoseValidationError
        sources={f"{r['class']}_{r['instance_id']}":'registration' for r in records}
        try:
            accepted_checks=validate_model_poses(records,path,folder/'masks/instance_masks.npz',
                                                folder/'foundationpose/validation.json')
        except PoseValidationError as exc:
            failed=exc.failed_names
            previous=getattr(self,'_previous_observation_folder',None)
            recovery=folder/'foundationpose/recovery';recovery.mkdir(parents=True,exist_ok=True)
            mode='previous_frame_tracking' if previous is not None else 'higher_hypothesis_registration'
            summary=dict(failed_instances=failed,mode=mode,attempt_limit=1,status='running',
                         capture=str(path),new_capture=False)
            write_json(recovery/'summary.json',summary)
            print(f'[PERCEPTION RECOVERY] {failed}: one {mode} attempt on saved RGB-D; no new capture or plan.',flush=True)
            retry=cmd.copy()
            if '--instances' in retry:retry=retry[:retry.index('--instances')]
            retry[retry.index('--out')+1]=container_path(recovery)
            retry[retry.index('--n-refine')+1]=str(max(10,int(c.get('n_refine',5))))
            if previous is not None:
                retry.extend(['--previous-capture',container_path(previous/'capture.npz'),
                              '--previous-masks',container_path(previous/'masks/instance_masks.npz')])
            else:
                retry[retry.index('--n-hypotheses')+1]=str(max(256,int(c.get('n_hypotheses',128))))
            retry.extend(['--instances',*failed])
            write_json(recovery/'command.json',retry)
            try:
                self.run(retry,recovery/'process.log')
                recovered=json.loads((recovery/'foundationpose_instances.json').read_text())['instances']
                names=[f"{r['class']}_{r['instance_id']}" for r in recovered]
                if len(names)!=len(failed) or set(names)!=set(failed):
                    raise UnsafePlan('pose recovery returned unexpected instance identities')
                recovered_checks=validate_model_poses(recovered,path,folder/'masks/instance_masks.npz',recovery/'validation.json')
                by_name=dict(zip(names,recovered))
                records=[by_name.get(f"{r['class']}_{r['instance_id']}",r) for r in records]
                by_check={r['name']:r for r in recovered_checks}
                accepted_checks=[by_check.get(r['name'],r) for r in exc.checks]
                sources.update({name:mode for name in names})
                summary['status']='accepted';write_json(recovery/'summary.json',summary)
                print(f'[PERCEPTION RECOVERY] Accepted recovered poses for {failed}; continuing the fixed plan.',flush=True)
            except Exception as recovery_error:
                summary.update(status='failed',reason=str(recovery_error));write_json(recovery/'summary.json',summary)
                raise UnsafePlan(f'pose recovery failed after one attempt for {failed}: {recovery_error}; see {recovery}') from recovery_error
        write_json(folder/'foundationpose/accepted_instances.json',dict(instances=records,pose_sources=sources))
        write_json(folder/'foundationpose/accepted_validation.json',accepted_checks)
        uncorrected = convert(records, require_flat=False) if records else []
        write_json(folder/'observed_objects_uncorrected.json', uncorrected)
        objects = apply_base_pose_offset(uncorrected, self.base_pose_offset_mm)
        registry=getattr(self,'_identity_registry',{})
        for obj in objects:
            entry=registry.setdefault(obj['name'],dict(initial_pose=copy.deepcopy(obj),status='active'))
            entry.update(last_valid_pose=copy.deepcopy(obj),last_valid_observation=index)
        if association is not None:
            mask_centers={m['name']:m['center_base_xy_mm'] for m in association['matches']}
        elif (folder/'masks/instance_masks.npz').exists():
            from instance_identity import mask_reference_centers
            mask_centers=mask_reference_centers(path,folder/'masks/instance_masks.npz',self.base_pose_offset_mm)
        else:mask_centers={}
        for name,center in mask_centers.items():
            if name in registry and registry[name]['status']!='removed':
                registry[name].setdefault('initial_mask_center_xy_mm',center)
                registry[name].update(last_mask_center_xy_mm=center,last_mask_observation=index)
        for obj in removed:
            registry[obj['name']]['status']='removed'
            registry[obj['name']]['removed_at_observation']=index
        self._identity_registry=registry
        write_json(self.run_dir/'identity_registry.json',registry)
        write_json(folder/'identity_registry.json',registry)
        objects.extend(removed)
        write_json(folder/'observed_objects.json', objects)
        self._previous_observation_folder=folder
        return objects


class SimulatedWorld:
    """Deterministic software test double, not a physical contact simulator."""
    def __init__(self, cfg, objects=None, *, knock_scale=1, pick_success=True):
        self.cfg=cfg; self.knock_scale=knock_scale; self.pick_success=pick_success
        self.objects = copy.deepcopy(objects) if objects is not None else self.initial_scene()
        self.initial_names = {o['name'] for o in self.objects}
        self.events = []

    @staticmethod
    def initial_scene():
        objects = []
        for i,(x,y) in enumerate(((.435,-.046),(.435,.046),(.495,-.046),(.495,.046)),1):
            t=np.eye(4); t[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix(); t[:3,3]=[x,y,.015]
            objects.append(dict(name=f'protex_{i}', instance_id=str(i), kind='protex',
                                dimensions_m=[.09,.06,.03], score=1,
                                base_center_m=t[:3,3].tolist(),yaw_rad=np.pi/2,T_base_object=t.tolist()))
        return objects

    def observe(self, index, folder, *, targets=None, active_objects=None, last_action=None):
        self.events.append(('observe',index))
        objects=[o for o in self.objects if targets is None or o['name'] in targets]
        write_json(Path(folder)/'observed_objects.json',objects)
        write_json(Path(folder)/'perception_scope.json',dict(estimated_instances=sorted(o['name'] for o in objects)))
        write_json(Path(folder)/'simulation.json', {'synthetic':True,'physics_validated':False})
        return copy.deepcopy(objects)

    def execute(self, action, folder):
        self.events.append(('execute', action['kind'],action['target']))
        obj = next(o for o in self.objects if o['name']==action['target'])
        if action['kind']=='knock':
            obj['base_center_m']=(np.asarray(obj['base_center_m'])+np.asarray(action['base_direction'])*action['intended_slide_mm']*self.knock_scale/1000).tolist()
        elif self.pick_success:
            from batch_executor import pick_placement_reference
            goal=pick_placement_reference(action)['expected_box_center_xy_mm']
            obj['base_center_m']=[*(np.asarray(goal)/1000).tolist(),.015]
        t=np.asarray(obj['T_base_object']); t[:3,3]=obj['base_center_m']; obj['T_base_object']=t.tolist()
        write_json(Path(folder)/'simulated_execution.json', {'synthetic':True,'stages_completed':[s['phase'] for s in action['stages']]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    default_config = ROOT/'real_robot/pipeline_config.local.json'
    if not default_config.exists(): default_config = ROOT/'real_robot/pipeline_config.example.json'
    parser.add_argument('--config',type=Path,default=default_config)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true',help='enable real xArm action batches; no per-stage prompts')
    mode.add_argument('--check-config',action='store_true',help='check motion settings and calculated-height inputs; no hardware access')
    mode.add_argument('--simulate',action='store_true',help='full loop with synthetic poses and motion outcomes; no hardware')
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--single-action',action='store_true',
                        help='execute one complete action, retreat, observe and update the fixed-plan cursor, then stop')
    parser.add_argument('--recorded-capture',type=Path,help='test full learned-perception preview on an existing RGB-D file; forbids execution')
    args=parser.parse_args()
    if args.recorded_capture and (args.execute or args.simulate):
        parser.error('--recorded-capture is a preview-only option')
    if args.single_action and not (args.execute or args.simulate):
        parser.error('--single-action requires --execute or --simulate')
    config=json.loads(args.config.read_text())
    output=resolve(args.output_root or config.get('output_root','real_robot/experiments/pipeline'))
    # Every invocation and observation receives its own immutable data folder.
    run_dir=output/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]); run_dir.mkdir(parents=True)
    print(f'RUN DIRECTORY: {run_dir}',flush=True)
    write_json(run_dir/'pipeline_config.json',config)
    write_json(run_dir/'run_metadata.json', {'mode': 'simulation' if args.simulate else ('physical' if args.execute else 'preview'),
                                           'recorded_input': str(args.recorded_capture) if args.recorded_capture else None,
                                           'single_action': args.single_action,
                                           'contact_physics_validated_by_simulation': False})
    execution_path = resolve(config['execution_config'])
    raw=json.loads(execution_path.read_text())
    write_json(run_dir/'execution_config.json', raw)
    report = dict(config_file=str(execution_path), **physical_config_report(raw))
    write_json(run_dir/'configuration_report.json', report)
    if args.check_config:
        checked_batch_config(raw, physical=False)
        print(json.dumps(report, indent=2))
        return
    if args.simulate:
        raw=copy.deepcopy(raw)
        if not height_available(raw,'pick_tcp_z_mm'): raw['pick_tcp_z_mm']=15
        if not height_available(raw,'release_tcp_z_mm'): raw['release_tcp_z_mm']=90
        if not height_available(raw,'knock.contact_tcp_z_mm'): raw['knock']['contact_tcp_z_mm']=30
    try:
        cfg=checked_batch_config(raw,physical=args.execute)
        write_json(run_dir/'execution_config.json',cfg)
        if args.simulate:
            world=SimulatedWorld(cfg); perception=executor=world
        else:
            perception=LivePerception(config['perception'],run_dir,resolve(args.recorded_capture) if args.recorded_capture else None,workflow=config['workflow'])
            executor=XArmBatchExecutor(cfg) if args.execute else None
        state=run_loop(perception,executor,cfg,config['workflow'],run_dir,
                       preview=not(args.execute or args.simulate),single_action=args.single_action)
        print(json.dumps(state,indent=2))
    except (ValueError,RuntimeError,OSError) as exc:
        write_json(run_dir/'failure.json',{'error':str(exc),'physical_execution_requested':args.execute})
        raise SystemExit(f'STOPPED: {exc}\nRun data: {run_dir}') from None


if __name__=='__main__':
    main()
