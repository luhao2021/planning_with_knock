"""Boundary tolerance follows active source identities, not deposited bowl objects."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from instance_identity import associate_masks
from pipeline_core import inside_source, assess_action, run_loop
from run_pipeline import SimulatedWorld
from source_workspace import source_xy_bounds
from test_pipeline import configs
import test_instance_identity as identity_tests
from xarm_executor import UnsafePlan


class SourceBoundaryTests(unittest.TestCase):
    workflow=dict(source_xy_min_mm=[300,-250],source_xy_max_mm=[570,250],source_boundary_tolerance_mm=100)

    def test_centers_within_100mm_of_each_edge_pass_but_initial_inventory_stays_strict(self):
        for xy in ([200,0],[670,0],[400,-350],[400,350]):
            obj=dict(base_center_m=[xy[0]/1000,xy[1]/1000,.015])
            self.assertTrue(inside_source(obj,self.workflow,with_tolerance=True))
            self.assertFalse(inside_source(obj,self.workflow))
        for xy in ([199.9,0],[670.1,0],[400,-350.1],[400,350.1]):
            self.assertFalse(inside_source(dict(base_center_m=[xy[0]/1000,xy[1]/1000,.015]),self.workflow,with_tolerance=True))
        for tolerance in (-1,101,float('nan')):
            with self.assertRaises(UnsafePlan):source_xy_bounds(dict(self.workflow,source_boundary_tolerance_mm=tolerance))

    def test_whole_mask_is_kept_when_centroid_is_inside_margin(self):
        obj=SimulatedWorld.initial_scene()[0];obj['base_center_m']=[.66,0,.015];obj['yaw_rad']=0
        render=identity_tests.IdentityTests().render
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);masks=render(d/'capture.npz',[obj]);np.savez(d/'masks.npz',**masks)
            assigned,diag=associate_masks(d/'capture.npz',d/'masks.npz',[obj],self.workflow,[0,0,0])
            self.assertIn(obj['name'],assigned)
            np.testing.assert_array_equal(assigned[obj['name']],masks[obj['name']])
            self.assertTrue(diag['boundary_tolerance_matches'])

    def test_pick_removal_is_not_blocked_by_a_mask_at_the_bowl_inside_margin(self):
        objects=SimulatedWorld.initial_scene()[:2]
        bowl=copy.deepcopy(objects[0]);bowl['base_center_m']=[.600,-.200,.015]
        render=identity_tests.IdentityTests().render
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);render(d/'before.npz',objects)
            masks=render(d/'after.npz',[objects[1],bowl]);np.savez(d/'masks.npz',**masks)
            assigned,diag=associate_masks(d/'after.npz',d/'masks.npz',objects,self.workflow,[0,0,0],
                                         dict(kind='pick',target=objects[0]['name']),d/'before.npz')
            self.assertTrue(diag['removal']['confirmed'])
            self.assertEqual(set(assigned),{objects[1]['name']})
            self.assertTrue(any(x['reason']=='boundary_mask_not_near_active_identity' for x in diag['discarded']))

    def test_failed_pick_just_outside_nominal_boundary_is_not_success(self):
        obj=SimulatedWorld.initial_scene()[0];obj['base_center_m'][0]=.610
        outcome=assess_action([obj],[copy.deepcopy(obj)],dict(kind='pick',target=obj['name']),{},self.workflow)
        self.assertFalse(outcome['completed'])

    def test_loop_can_pick_after_knock_moves_target_beyond_nominal_boundary(self):
        cfg,w=configs();w.update(self.workflow);cfg['knock']['stroke_mm']=80
        obj=SimulatedWorld.initial_scene()[0];obj['base_center_m']=[.550,0,.015]
        obj['yaw_rad']=0;obj['T_base_object']=np.eye(4).tolist()
        for i,v in enumerate(obj['base_center_m']):obj['T_base_object'][i][3]=v
        initial=dict(objects=[obj],action_details=[dict(target=obj['name'],kind='knock',witness='knock-+x',blockers=[])],
                     footprint_shrink_total_m=.002)
        world=SimulatedWorld(cfg,objects=[obj])
        with tempfile.TemporaryDirectory() as d,patch('pipeline_core.snapshot_plan',return_value=initial):
            state=run_loop(world,world,cfg,w,Path(d),render=False)
        self.assertEqual(state['status'],'complete')
        self.assertEqual(state['actions_executed'],2)
        self.assertEqual(state['final_home']['status'],'complete')


if __name__=='__main__':unittest.main()
