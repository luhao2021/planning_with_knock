"""Bounded pose-noise tolerance; no hardware access."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from batch_executor import checked_batch_config,build_batch
from fixed_execution import latest_layout,select_fixed_candidate
from pick_clearance import pick_clearance,rectangle,overlap_depth_mm
from pipeline_core import run_loop
from run_pipeline import SimulatedWorld
from test_pipeline import configs
import test_pipeline
from xarm_executor import UnsafePlan


class PoseNoiseToleranceTests(unittest.TestCase):
    def test_three_mm_overlap_passes_larger_obstructions_fail_at_rotated_yaws(self):
        cfg,w=configs();self.assertEqual(cfg['pick_overlap_tolerance_mm'],3)
        for yaw in (0,45,90,-30):
            for gap,passed in ((33,True),(32.5,True),(30,True),(29.5,True),(29.49,False),(5,False)):
                with self.subTest(yaw=yaw,gap=gap):
                    objects=test_pipeline.PhysicalPickClearanceTests.scene(gap,yaw);original=copy.deepcopy(objects)
                    candidate=dict(target=objects[0]['name'],kind='pick',witness='pick-y@-0.044',blockers=[])
                    layout=latest_layout(objects,w);check=pick_clearance(layout,candidate,cfg)
                    self.assertEqual(check['passed'],passed)
                    self.assertEqual(objects,original);self.assertEqual(check['finger_thickness_mm'],20)
                    if 29.5<=gap<=32.5:
                        self.assertEqual(check['tolerated_neighbors'],[objects[1]['name']])
                        self.assertAlmostEqual(check['overlap_details'][0]['estimated_overlap_mm'],32.5-gap,places=5)
                        entry=dict(target=objects[0]['name'],kind='pick',pick=candidate,knock=None)
                        chosen,_=select_fixed_candidate(entry,layout,cfg,has_knocked=False)
                        action=build_batch(layout,chosen,cfg,bowl='bowl_1')
                        self.assertEqual(action['pick_clearance'],check)
                    elif not passed:
                        self.assertEqual(check['blockers'],[objects[1]['name']])
                        with self.assertRaisesRegex(UnsafePlan,'Pick clearance failed'):
                            build_batch(layout,candidate,cfg,bowl='bowl_1')

    def test_containment_is_not_mistaken_for_a_shallow_intersection(self):
        x=np.array([1.,0]);y=np.array([0.,1]);center=np.zeros(2)
        outer=rectangle(center,x,y,10,10);inner=rectangle(center,x,y,1,1)
        self.assertAlmostEqual(overlap_depth_mm(outer,inner),11)
        self.assertIsNone(overlap_depth_mm(outer,inner+30))

    def test_invalid_overlap_tolerances_fail_configuration_validation(self):
        cfg,_=configs()
        for value in (-1,5.01,float('nan'),float('inf')):
            with self.subTest(value=value):
                bad=copy.deepcopy(cfg);bad['pick_overlap_tolerance_mm']=value
                with self.assertRaisesRegex(UnsafePlan,'pick_overlap_tolerance_mm'):
                    checked_batch_config(bad,physical=True)

    def test_three_no_progress_observations_do_not_prevent_later_success(self):
        cfg,w=configs()
        class DelayedContact(SimulatedWorld):
            def __init__(self,cfg):super().__init__(cfg);self.knocks={}
            def execute(self,action,folder):
                if action['kind']=='knock':
                    target=action['target'];self.knocks[target]=self.knocks.get(target,0)+1
                    self.knock_scale=0 if self.knocks[target]<=3 else 1
                super().execute(action,folder)
        world=DelayedContact(cfg)
        with tempfile.TemporaryDirectory() as d:
            result=run_loop(world,world,cfg,w,Path(d),render=False)
            self.assertEqual(result['status'],'complete')
            self.assertTrue(all(n>3 for n in world.knocks.values()))
            self.assertEqual(len(result['completed']),4)


if __name__=='__main__':unittest.main()
