"""Fixed-plan loop regressions: one solver call, one overlay, no hardware."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pipeline_core
from pipeline_core import run_loop
from run_pipeline import SimulatedWorld
from test_pipeline import configs
from fixed_execution import latest_layout,select_fixed_candidate
from xarm_executor import UnsafePlan


class FixedExecutionTests(unittest.TestCase):
    def test_one_solver_and_overlay_for_entire_loop_and_fixed_witnesses(self):
        cfg,w=configs();world=SimulatedWorld(cfg);actions=[];plan_bytes=[]
        execute=world.execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def recording_execute(action,folder):
                current=next(o for o in world.objects if o['name']==action['target'])
                # Every action uses the latest pose, even with a fixed grasp witness.
                self.assertEqual(action['source_pose']['base_center_m'],current['base_center_m'])
                actions.append(copy.deepcopy(action))
                plan_bytes.append((root/'observations/0000/planning/fixed_execution_plan.json').read_bytes())
                execute(action,folder)
            world.execute=recording_execute
            with patch('pipeline_core.solve_scene',wraps=pipeline_core.solve_scene) as solve, \
                 patch('pipeline_core.correct_pose_overlaps',wraps=pipeline_core.correct_pose_overlaps) as correct, \
                 patch('pipeline_core.plot_instance') as plot, \
                 patch('pipeline_core.plot_solution_overlay') as overlay, \
                 patch('pipeline_core.plot_solution_sequence') as sequence:
                state=run_loop(world,world,cfg,w,root,render=True)
            self.assertEqual(state['status'],'complete')
            for call in (solve,correct,plot,overlay,sequence):self.assertEqual(call.call_count,1)
            self.assertTrue(all(b==plan_bytes[0] for b in plan_bytes))
            fixed=json.loads(plan_bytes[0])['steps']
            self.assertEqual(state['completed'],[s['target'] for s in fixed])
            self.assertEqual(len(list(root.glob('observations/*/planning'))),1)
            for action in actions:
                entry=next(s for s in fixed if s['target']==action['target'])
                self.assertEqual(action['witness'],entry[action['kind']]['witness'])
            indices=[json.loads(p.read_text())['fixed_plan_step'] for p in sorted(root.glob('observations/*/execution_selection.json'))]
            self.assertEqual(indices,sorted(indices))

    def test_partial_observations_refresh_target_at_handoff_and_keep_neighbors(self):
        cfg,w=configs();world=SimulatedWorld(cfg);requests=[];actions=[]
        observe=world.observe;execute=world.execute
        def scoped_observe(index,folder,*,targets=None,**kwargs):
            requests.append(None if targets is None else list(targets))
            return observe(index,folder,targets=targets)
        def scoped_execute(action,folder):
            actions.append(copy.deepcopy(action))
            execute(action,folder)
        world.observe=scoped_observe;world.execute=scoped_execute
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            state=run_loop(world,world,cfg,w,root,render=False)
            self.assertEqual(state['status'],'complete')
            self.assertIsNone(requests[0])
            for i,action in enumerate(actions):
                if i+1<len(actions):
                    expected=[action['target']] if action['kind']=='knock' else [actions[i+1]['target']]
                    self.assertEqual(requests[i+1],expected)
                else:self.assertEqual(len(requests),len(actions))
                scene=json.loads((root/f'observations/{i:04d}/execution_scene.json').read_text())
                self.assertEqual(scene['pose_observation_indices'][action['target']],i)
                self.assertIn(action['target'],[o['name'] for o in scene['objects']])
                if i==1:
                    self.assertTrue(any(n!=action['target'] and idx==0
                                        for n,idx in scene['pose_observation_indices'].items()))
            # Every recorded detection after initialization contains only requested boxes.
            for i,request in enumerate(requests[1:],1):
                detected=json.loads((root/f'observations/{i:04d}/observed_objects.json').read_text())
                self.assertEqual({o['name'] for o in detected},set(request))

    def test_after_pick_far_from_bowl_advances_in_fixed_order(self):
        cfg,w=configs()
        class Outside(SimulatedWorld):
            def execute(self,action,folder):
                super().execute(action,folder)
                if action['kind']=='pick':
                    obj=next(o for o in self.objects if o['name']==action['target'])
                    obj['base_center_m']=[.720,0,.015]
                    obj['T_base_object'][0][3]=.720;obj['T_base_object'][1][3]=0
        world=Outside(cfg)
        with tempfile.TemporaryDirectory() as d:
            state=run_loop(world,world,cfg,w,Path(d),render=False)
            self.assertEqual(state['status'],'complete')
            fixed=json.loads((Path(d)/state['fixed_plan_file']).read_text())['steps']
            self.assertEqual(state['completed'],[s['target'] for s in fixed])

    def test_recorded_starting_overlap_does_not_veto_fixed_knock(self):
        from batch_executor import build_batch,direction_base
        from scipy.spatial.transform import Rotation
        cfg,w=configs();objects=SimulatedWorld.initial_scene()[:2]
        # Observation 4 in 20260915_055503_8b8921: neighbor is beside target.
        poses=[('protex_2',[.45116009075,.03421290005,.01478598889],90.58812127),
               ('protex_4',[.51090742266,.03054037189,.01641794213],83.55008606)]
        for obj,(name,xyz,yaw) in zip(objects,poses):
            t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix();t[:3,3]=xyz
            obj.update(name=name,instance_id=name[-1],base_center_m=xyz,yaw_rad=np.deg2rad(yaw),T_base_object=t.tolist())
        entry=dict(target='protex_4',kind='knock',
            knock=dict(target='protex_4',kind='knock',witness='knock-+x',blockers=[]),
            pick=dict(target='protex_4',kind='pick',witness='pick-y@-0.044',blockers=[]))
        original=copy.deepcopy(entry);layout=latest_layout(objects,w)
        for has_knocked in (False,True):
            with patch('cuboid_knock_pick_solvers.knock_blockers',side_effect=AssertionError('no repeated corridor check')):
                chosen,checks=select_fixed_candidate(entry,layout,cfg,has_knocked=has_knocked)
            self.assertEqual(chosen,entry['knock'])
            if has_knocked:self.assertFalse(checks['pick_clearance']['passed'])
        self.assertEqual(entry,original)
        action=build_batch(layout,chosen,cfg)
        start=next(s['tcp_mm_deg'] for s in action['stages'] if s['phase']=='contact')
        end=next(s['tcp_mm_deg'] for s in action['stages'] if s['phase']=='horizontal-knock')
        np.testing.assert_allclose(np.array(end[:2])-start[:2],50*direction_base(objects[1],'+x')[:2])

    def test_initial_knock_does_not_need_a_pick_clearance_check(self):
        entry=dict(kind='knock',knock=dict(target='protex_4',kind='knock',witness='knock-+x',blockers=[]))
        with patch('fixed_execution.pick_clearance',side_effect=AssertionError('initial Knock needs no Pick check')):
            chosen,checks=select_fixed_candidate(entry,{}, {},has_knocked=False)
        self.assertEqual(chosen,entry['knock'])


if __name__=='__main__':unittest.main()
