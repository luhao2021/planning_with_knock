"""Pick completion verifies removal from source; no drop-distance gate or hardware."""
import copy
import unittest
import numpy as np
from batch_executor import pick_placement_reference
from pipeline_core import assess_action
from xarm_executor import UnsafePlan


class PlacementVerificationTests(unittest.TestCase):
    def fixture(self):
        # Numeric regression from the 20260915_052707 Pick; independent of run files.
        before=[dict(name='protex_3',base_center_m=[.5010469652,-.0996562319,.0152140769])]
        action=dict(kind='pick',target='protex_3',bowl='bowl_1',source_pose=copy.deepcopy(before[0]),stages=[
            dict(phase='pick',tcp_mm_deg=[496.82389245,-128.34709553,13.09843639,180,0,81.62663112]),
            dict(phase='place',tcp_mm_deg=[600,-200,60,180,0,81.62663112])])
        after=[dict(name='protex_3',base_center_m=[.6316359002,-.1717133286,.0316896846])]
        cfg=dict(drop_points_xy_mm={'bowl_1':[600,-200]},placement_tolerance_mm=40,release_tcp_z_mm=60)
        workflow=dict(source_xy_min_mm=[300,-250],source_xy_max_mm=[570,250])
        return before,after,action,cfg,workflow

    def test_recorded_pick_advances_when_target_is_detected_outside_source(self):
        result=assess_action(*self.fixture())
        self.assertTrue(result['completed'])
        self.assertEqual(result['decision'],'target_removed_from_source')
        self.assertNotIn('placement_distance_mm',result)

    def test_drop_distance_does_not_gate_source_removal(self):
        before,after,action,cfg,w=self.fixture()
        # Outside source, irrespective of the configured bowl or grasp offset.
        for xy in ([.600,-.229],[.700,.100],[.620,-.171]):
            after[0]['base_center_m']=[*xy,.030]
            self.assertTrue(assess_action(before,after,action,cfg,w)['completed'])

    def test_orientation_change_rotates_the_carried_offset(self):
        _,_,action,_,_=self.fixture()
        action['source_pose']['base_center_m']=[.500,0,.015]
        action['stages'][0]['tcp_mm_deg']=[470,0,13,180,0,0]
        action['stages'][1]['tcp_mm_deg']=[600,-200,60,180,0,90]
        ref=pick_placement_reference(action)
        np.testing.assert_allclose(ref['expected_box_center_xy_mm'],[600,-170],atol=1e-8)

    def test_missing_still_in_source_or_invalid_detection_is_not_completion(self):
        before,after,action,cfg,w=self.fixture()
        with self.assertRaisesRegex(UnsafePlan,'absence alone'):
            assess_action(before,[],action,cfg,w)
        result=assess_action(before,copy.deepcopy(before),action,cfg,w)
        self.assertFalse(result['completed'])
        after[0]['base_center_m'][0]=float('nan')
        with self.assertRaisesRegex(UnsafePlan,'invalid detected centre'):
            assess_action(before,after,action,cfg,w)

    def test_missing_or_nonfinite_action_geometry_is_rejected(self):
        _,_,action,_,_=self.fixture()
        action['stages'][0]['tcp_mm_deg'][0]=float('nan')
        with self.assertRaisesRegex(UnsafePlan,'invalid Pick placement reference'):
            pick_placement_reference(action)
        action['stages']=[]
        with self.assertRaisesRegex(UnsafePlan,'invalid Pick placement reference'):
            pick_placement_reference(action)


if __name__=='__main__':unittest.main()
