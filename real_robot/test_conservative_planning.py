"""Planner admissibility must imply execution admissibility for the same scene."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from cuboid_knock_pick_solvers import GeometryInstance,Rect
from conservative_pick_planning import conservative_rule
from pipeline_core import snapshot_plan,run_loop
from fixed_execution import freeze_plan
from pick_clearance import pick_clearance
from run_pipeline import SimulatedWorld
from test_pipeline import configs
import test_pipeline


class ConservativePlanningTests(unittest.TestCase):
    def geometry(self,objects,cfg,w):
        rects=[]
        for obj in objects:
            l,h=np.array(obj['dimensions_m'][:2])-.002;x,y=obj['base_center_m'][:2]
            rects.append(Rect(obj['name'],x-l/2,x+l/2,y-h/2,y+h/2,obj['yaw_rad']))
        inst=GeometryInstance('test',rects,F=.030,D=.080,finger_clearance=.020)
        model=dict(open_aperture_mm=cfg['gripper_open_mm'],finger_thickness_mm=20,margin_mm=3,
                   usable_jaw_opening_mm=cfg['usable_jaw_opening_mm'],footprint_shrink_mm=2)
        return conservative_rule(inst,objects,model)

    def assert_dominates(self,objects,cfg,w):
        rule,picks=self.geometry(objects,cfg,w)
        checked=0
        for i,options in picks.items():
            for option in options:
                candidate=dict(target=rule.names[i],kind='pick',witness=option.witness,blockers=[])
                check=pick_clearance(dict(objects=objects,finger_thickness_mm=20),candidate,cfg)
                planned={rule.names[j] for j in option.blockers}
                self.assertLessEqual(set(check['blockers']),planned,(objects,candidate,check,planned))
                remaining=[o for o in objects if o['name'] not in planned]
                self.assertTrue(pick_clearance(dict(objects=remaining,finger_thickness_mm=20),candidate,cfg)['passed'])
                checked+=1
        return checked

    def test_planning_stricter_than_execution_at_parallel_gap_boundary(self):
        cfg,w=configs()
        for yaw in (0,45,90,-30):
            for gap in (5,29.5,30,33,35.5,35.51,40):
                objects=test_pipeline.PhysicalPickClearanceTests.scene(gap,yaw)
                rule,picks=self.geometry(objects,cfg,w)
                self.assertTrue(picks[0])
                if gap<=35.5:self.assertTrue(all(a.blockers for a in picks[0]))
                else:self.assertTrue(any(not a.blockers for a in picks[0]))
                self.assertGreater(self.assert_dominates(objects,cfg,w),0)

    def test_execution_blockers_subset_of_planning_blockers_under_pose_variation(self):
        cfg,w=configs();cfg['pick_overlap_tolerance_mm']=0
        rng=np.random.default_rng(61015);checked=0
        for trial in range(240):
            objects=SimulatedWorld.initial_scene()
            for obj in objects:
                xyz=np.array([.46,0,.015])+rng.uniform([-.09,-.14,0],[.09,.14,0])
                yaw=rng.uniform(-np.pi,np.pi);t=np.eye(4)
                t[:3,:3]=Rotation.from_euler('z',yaw).as_matrix();t[:3,3]=xyz
                obj.update(base_center_m=xyz.tolist(),yaw_rad=float(yaw),T_base_object=t.tolist())
            checked+=self.assert_dominates(objects,cfg,w)
        self.assertGreater(checked,900)

    def test_omitted_finger_setting_has_same_planning_and_execution_default(self):
        from fixed_execution import latest_layout
        cfg,w=configs();w.pop('finger_thickness_mm')
        objects=SimulatedWorld.initial_scene()
        with tempfile.TemporaryDirectory() as d:
            planned=snapshot_plan(objects,Path(d),w,render=False,execution_cfg=cfg)
        self.assertEqual(planned['planning_gripper_envelope']['finger_thickness_mm'],20)
        self.assertEqual(planned['finger_thickness_mm'],latest_layout(objects,w)['finger_thickness_mm'])

    def test_saved_failure_has_no_execution_veto_of_initial_plan(self):
        cfg,w=configs()
        fixture=json.loads((Path(__file__).parent/'testdata/initial_pick_mismatch_061015.json').read_text())
        objects=fixture['objects']
        with tempfile.TemporaryDirectory() as d:
            # Initial planning/freezing cannot call the executor's acceptance rule.
            with patch('pick_clearance.pick_clearance',side_effect=AssertionError('planning must be independent')), \
                 patch('fixed_execution.pick_clearance',side_effect=AssertionError('freeze must respect planning')):
                layout=snapshot_plan(objects,Path(d)/'initial',w,render=False,execution_cfg=cfg)
                fixed=freeze_plan(layout,cfg)
            self.assertEqual(fixed[0]['kind'],'knock')
            remaining=copy.deepcopy(objects)
            for step in layout['action_details']:
                if step['kind']=='pick':
                    self.assertTrue(pick_clearance(dict(objects=remaining,finger_thickness_mm=20),step,cfg)['passed'])
                remaining=[o for o in remaining if o['name']!=step['target']]
            world=SimulatedWorld(cfg,objects=objects)
            state=run_loop(world,world,cfg,w,Path(d)/'simulation',render=False)
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['completed'],[s['target'] for s in fixed])


if __name__=='__main__':unittest.main()
