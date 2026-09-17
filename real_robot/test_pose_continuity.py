"""Symmetry flips must not reverse fixed Knock directions or swap grasp ends."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from batch_executor import checked_batch_config, direction_base, build_batch
from fixed_execution import freeze_plan, latest_layout, select_fixed_candidate
from pipeline_core import run_loop
from pose_continuity import align_execution_yaws
from run_pipeline import SimulatedWorld
from test_pipeline import configs

HERE=Path(__file__).parent


class PoseContinuityTests(unittest.TestCase):
    def fixture(self):return json.loads((HERE/'testdata/s07_yaw_flip_001038.json').read_text())

    def test_nearest_half_turn_preserves_real_rotation_and_input_poses(self):
        for before,detected,expected in [(1,177,-3),(179,1,181),(-3,178,-2),(20,40,40),(0,180,0),(181,3,183)]:
            previous=[dict(name='box',yaw_rad=math.radians(before))]
            raw=[dict(name='box',yaw_rad=math.radians(detected),base_center_m=[.4,0,.015],T_base_object=np.eye(4).tolist())]
            original=copy.deepcopy(raw)
            obj=align_execution_yaws(raw,previous)[0]
            self.assertAlmostEqual(math.degrees(obj['yaw_rad']),expected)
            self.assertEqual(raw,original)
            self.assertEqual(obj['T_base_object'],raw[0]['T_base_object'])
            self.assertEqual(obj['base_center_m'],raw[0]['base_center_m'])
            self.assertEqual(obj['detected_yaw_rad'],raw[0]['yaw_rad'])
            self.assertEqual(align_execution_yaws([obj],previous),[obj])

    def test_all_signed_directions_stay_consistent_across_the_pi_boundary(self):
        for direction in ('+x','-x','+y','-y'):
            previous=[dict(name='box',yaw_rad=math.radians(.8))]
            initial=direction_base(previous[0],direction)
            for yaw in (176.99,178.21,.09):
                aligned=align_execution_yaws([dict(name='box',yaw_rad=math.radians(yaw))],previous)
                self.assertGreater(float(initial@direction_base(aligned[0],direction)),.99)
                previous=aligned

    def test_saved_first_knock_is_followed_by_clear_pick_with_same_witness(self):
        fixture=self.fixture();cfg,w=configs();initial=fixture['initial']
        plan=freeze_plan(initial,cfg);entry=plan[0];original=copy.deepcopy(plan)
        objects={o['name']:o for o in initial['objects']}
        raw=fixture['observations'][0]
        before=latest_layout(list(objects.values()),w)
        before['objects']=[next((o for o in raw if o['name']==old['name']),old) for old in before['objects']]
        self.assertEqual(select_fixed_candidate(entry,before,cfg,has_knocked=True)[0]['kind'],'knock')
        objects.update({o['name']:o for o in align_execution_yaws(raw,list(objects.values()))})
        chosen,checks=select_fixed_candidate(entry,latest_layout(list(objects.values()),w),cfg,has_knocked=True)
        self.assertEqual(chosen['kind'],'pick')
        self.assertEqual(chosen['witness'],entry['pick']['witness'])
        self.assertEqual(checks['pick_clearance']['overlap_details'],[])
        self.assertEqual(plan,original)
        action=build_batch(latest_layout(list(objects.values()),w),chosen,cfg,bowl='bowl_2')
        grasp=next(s['tcp_mm_deg'] for s in action['stages'] if s['phase']=='pick')
        np.testing.assert_allclose(grasp[:2],[279.1421451826,39.9141771695])

    def test_loop_applies_continuity_before_selecting_the_second_action(self):
        fixture=self.fixture();cfg,w=configs();cfg['knock']['stroke_mm']=80
        world=SimulatedWorld(cfg,objects=fixture['initial']['objects']);actions=[]
        observe=world.observe
        def replay_observe(index,folder,**kwargs):
            return copy.deepcopy(fixture['observations'][0]) if index==1 else observe(index,folder,**kwargs)
        def record_action(action,folder):
            actions.append(copy.deepcopy(action))
            if len(actions)==2:raise StopIteration('recorded next action')
        world.observe=replay_observe;world.execute=record_action
        with tempfile.TemporaryDirectory() as d,patch('pipeline_core.snapshot_plan',return_value=fixture['initial']):
            with self.assertRaisesRegex(StopIteration,'recorded next action'):
                run_loop(world,world,cfg,w,Path(d),render=False)
            diagnostic=json.loads((Path(d)/'observations/0001/yaw_alignment.json').read_text())
            self.assertEqual(diagnostic[0]['half_turns_applied'],-1)
        self.assertEqual([a['kind'] for a in actions],['knock','pick'])
        self.assertAlmostEqual(math.degrees(actions[1]['source_pose']['yaw_rad']),-3.0125146746)


if __name__=='__main__':unittest.main()
