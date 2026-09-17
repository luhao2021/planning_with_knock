"""Post-Knock grasp follows the exposed end without changing the solver order."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from batch_executor import build_batch, checked_batch_config, direction_base
from fixed_execution import freeze_plan, select_fixed_candidate
from pick_clearance import pick_clearance
from test_pipeline import configs

HERE=Path(__file__).parent


class ExposedEndGraspTests(unittest.TestCase):
    def fixture(self):
        return json.loads((HERE/'testdata/s07_trailing_end_grasp_230300.json').read_text())

    def test_saved_s07_picks_after_first_knock_without_second_stroke(self):
        fixture=self.fixture()
        cfg=checked_batch_config(json.loads((HERE/'execution_config.batch.example.json').read_text()))
        initial=fixture['initial'];layout=fixture['post_knock_layout']
        old=initial['initial_pick_options']['beach_2'][0]
        check=pick_clearance(layout,old,cfg)
        self.assertFalse(check['passed'])
        self.assertEqual(check['blockers'],['beach_1'])
        self.assertAlmostEqual(check['overlap_details'][0]['estimated_overlap_mm'],28.7226856)
        with patch('fixed_execution.pick_clearance',side_effect=AssertionError('do not guide initial plan using execution checks')):
            frozen=freeze_plan(initial,cfg)
        self.assertEqual([e['target'] for e in frozen],[a['target'] for a in initial['action_details']])
        for selected,entry in zip(initial['action_details'],frozen):
            if selected['kind']=='knock':self.assertEqual(entry['knock']['witness'],selected['witness'])
            else:self.assertEqual(entry['pick']['witness'],selected['witness'])
        self.assertEqual(frozen[0]['pick']['witness'],'pick-y@0.024000000')
        candidate,checks=select_fixed_candidate(frozen[0],layout,cfg,has_knocked=True)
        self.assertEqual(candidate['kind'],'pick')
        self.assertTrue(checks['pick_clearance']['passed'])
        self.assertEqual(checks['pick_clearance']['overlap_details'],[])
        batch=build_batch(layout,candidate,cfg,bowl='bowl_2')
        grasp=next(s['tcp_mm_deg'] for s in batch['stages'] if s['phase']=='pick')
        np.testing.assert_allclose(grasp[:2],[291.5189499428,36.1449259461])
        self.assertEqual(batch['stages'][-1]['phase'],'retreat')

    def test_all_local_directions_and_yaws_place_grasp_toward_slide(self):
        cfg,_=configs()
        for direction in ('+x','-x','+y','-y'):
            for yaw in (0,np.pi/2,-np.pi/3,np.pi-.1):
                with self.subTest(direction=direction,yaw=yaw):
                    obj=dict(name='test',kind='protex',dimensions_m=[.06,.065,.03],
                             base_center_m=[.4,0,.015],yaw_rad=yaw)
                    chosen=dict(target='test',kind='knock',witness='knock-'+direction,blockers=[])
                    initial=dict(objects=[obj],action_details=[chosen],footprint_shrink_total_m=.002)
                    entry=freeze_plan(initial,cfg)[0]
                    candidate,_=select_fixed_candidate(entry,{'objects':[obj]},cfg,has_knocked=True)
                    self.assertEqual(candidate['kind'],'pick')
                    action=build_batch({'objects':[obj]},candidate,cfg,bowl='bowl_1')
                    grasp=np.asarray(next(s['tcp_mm_deg'] for s in action['stages'] if s['phase']=='pick'))[:2]
                    offset=grasp-np.asarray(obj['base_center_m'][:2])*1000
                    expected=direction_base(obj,direction)[:2]
                    np.testing.assert_allclose(offset/np.linalg.norm(offset),expected,atol=1e-8)

    def test_real_soap_short_axis_keeps_jaw_compatible_pick(self):
        cfg,_=configs();initial=self.fixture()['initial']
        initial=copy.deepcopy(initial);initial['action_details']=initial['action_details'][:1]
        initial['action_details'][0]['witness']='knock-+y'
        entry=freeze_plan(initial,cfg)[0]
        # BEACH's 110 mm long side cannot fit in the 80 mm usable jaw opening.
        self.assertEqual(entry['pick']['witness'],initial['initial_pick_options']['beach_2'][0]['witness'])

    def test_more_than_30mm_motion_is_not_a_blanket_overlap_override(self):
        cfg,_=configs();fixture=self.fixture();entry=freeze_plan(fixture['initial'],cfg)[0]
        layout=copy.deepcopy(fixture['post_knock_layout'])
        target=next(o for o in layout['objects'] if o['name']=='beach_2')
        neighbor=next(o for o in layout['objects'] if o['name']=='beach_1')
        # Place a real obstacle alongside the leading window after the saved 72 mm move.
        yaw=target['yaw_rad'];normal=np.array([-np.sin(yaw),np.cos(yaw)])
        neighbor['yaw_rad']=yaw
        neighbor['base_center_m'][:2]=(np.array(target['base_center_m'][:2])+.065*normal).tolist()
        candidate,checks=select_fixed_candidate(entry,layout,cfg,has_knocked=True)
        self.assertEqual(candidate['kind'],'knock')
        self.assertIn('beach_1',checks['pick_clearance']['blockers'])


if __name__=='__main__':unittest.main()
