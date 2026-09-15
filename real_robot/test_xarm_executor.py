"""Disconnected geometry and safety-gate tests; never imports or connects to SDK."""
import json
import unittest
from pathlib import Path

from xarm_executor import UnsafePlan, aperture_pulse, build_action, checked_config

ROOT = Path(__file__).parent


def synthetic_calibration():
    config = json.loads((ROOT / 'execution_config.example.json').read_text())
    config.update(tcp_calibrated=True, gripper_calibrated=True,
                  table_plane_verified=True,
                  workspace_min_mm=[250, -300, 0], workspace_max_mm=[850, 300, 600],
                  transit_z_mm=260, approach_z_mm=180, lift_z_mm=220,
                  tool_roll_pitch_deg=[180, 0], tcp_offset_mm_deg=[0,0,0,0,0,0],
                  tool_closing_axis_yaw_offset_deg=0,
                  tcp_contact_z_offset_mm=80, usable_jaw_opening_mm=80,
                  gripper_speed_setting=10, knock_stroke_mm=10,
                  knock_precontact_clearance_mm=0, knock_max_drop_mm=.5,
                  gripper_aperture_calibration={
                      'closed_pulse': 0, 'closed_aperture_mm': 0,
                      'open_pulse': 850, 'open_aperture_mm': 85})
    for bowl in config['bowls'].values():
        bowl.update(release_z_mm=90, inner_radius_mm=90)
    return config


class ExecutorPreviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layout = json.loads((ROOT / 'testdata/recorded_protex_layout.json').read_text())

    def test_template_blocks_motion_preparation(self):
        template=json.loads((ROOT / 'execution_config.example.json').read_text())
        with self.assertRaises(UnsafePlan):
            checked_config(template)

    def test_object_aperture_and_bowl_assignment(self):
        cfg=checked_config(synthetic_calibration())
        self.assertEqual(aperture_pulse(57, cfg),570)
        action=build_action(self.layout,cfg,2)
        self.assertEqual(action['kind'],'pick')
        self.assertEqual(action['bowl'],'bowl_1')
        self.assertEqual([s['phase'] for s in action['stages']],
                         ['pre-pick','open-fingers','approach','pick','close-fingers',
                          'lift','clearance-lift','transfer','place','open-fingers','retreat'])
        self.assertEqual(action['stages'][4]['gripper_pulse'],570)
        self.assertEqual(action['stages'][7]['tcp_mm_deg'][:2],(600,-100))

    def test_knock_stops_for_reobservation(self):
        cfg=checked_config(synthetic_calibration())
        action=build_action(self.layout,cfg,0)
        self.assertEqual(action['kind'],'knock')
        self.assertEqual(action['stages'][-1]['phase'],'pause-for-reobservation')
        self.assertLessEqual(action['stages'][4]['speed_mm_s'],3)
        self.assertIsNone(action['bowl'])

    def test_grasp_rejects_unreachable_aperture(self):
        cfg=checked_config(synthetic_calibration())
        cfg['usable_jaw_opening_mm']=55
        with self.assertRaisesRegex(UnsafePlan,'exceeds jaw opening'):
            build_action(self.layout,cfg,2)


if __name__=='__main__':
    unittest.main()
