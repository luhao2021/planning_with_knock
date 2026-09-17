import unittest
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pose_to_plan import base_to_optical, convert, plan, solve_scene


class PoseAdapterTest(unittest.TestCase):
    def test_ros_transform_and_distinct_instances(self):
        T = base_to_optical()
        self.assertTrue(np.allclose(T[:3, 3], [1.0494764484901562,
                                               -0.05633311429031207,
                                               0.6130679657159153]))
        self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0, places=6)
        records = []
        for idx, x in enumerate([0.35, 0.50, 0.65, 0.80]):
            T_base = np.eye(4)
            T_base[:3, :3] = Rotation.from_euler('z', 0.1 * idx).as_matrix()
            T_base[:3, 3] = [x, 0.0, 0.015]
            records.append({'class': 'protex', 'instance_id': str(idx+1),
                            'score': 0.9,
                            'T_camera_object': (np.linalg.inv(T) @ T_base).tolist()})
        objects = convert(records)
        self.assertEqual([o['name'] for o in objects],
                         ['protex_1', 'protex_2', 'protex_3', 'protex_4'])
        self.assertAlmostEqual(objects[0]['base_center_m'][0], 0.35)
        actions, _, status = plan(objects)
        self.assertEqual(status, "optimal")
        self.assertEqual(len(actions), 4)

    def test_correction_finds_feasible_solution_within_existing_5mm_bound(self):
        from pose_to_plan import correct_pose_overlaps
        import copy
        objects=json.loads((Path(__file__).parent/'testdata/automatic_seed_bounded_correction.json').read_text())
        original=copy.deepcopy(objects)
        corrected,_=correct_pose_overlaps(objects,.005)
        self.assertEqual(objects,original)
        for before,after in zip(objects,corrected):
            self.assertLessEqual(np.linalg.norm(np.array(after['base_center_m'])-before['base_center_m']),.005+1e-8)
            self.assertEqual(before['dimensions_m'],after['dimensions_m'])
            self.assertEqual(before['yaw_rad'],after['yaw_rad'])
        # The same scene has no valid solution within a 1 mm per-box budget.
        with self.assertRaises(ValueError):correct_pose_overlaps(objects,.001)

    def test_rearranged_scene_opening_threshold(self):
        poses = Path(__file__).parent / 'testdata/recorded_protex_poses.json'
        objects = convert(json.loads(poses.read_text())['instances'])
        for opening, expected in ((0.080, 2), (0.085, 2), (0.088, 1)):
            rule, result = solve_scene(objects, footprint_shrink_m=0.002,
                                       jaw_opening_m=opening)
            self.assertEqual(result.status, 'optimal')
            self.assertEqual(result.optimum, expected)
            self.assertEqual([round(r.w, 3) for r in rule.geometry.rects], [0.088]*4)
            self.assertEqual([round(r.h, 3) for r in rule.geometry.rects], [0.058]*4)
        self.assertTrue(all(obj['dimensions_m'] == (0.09, 0.06, 0.028) for obj in objects))

    def test_duplicate_instance_rejected(self):
        record = {'class': 'protex', 'instance_id': '1', 'score': 0.9,
                  'T_camera_object': np.linalg.inv(base_to_optical()).tolist()}
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            convert([record, record])


if __name__ == '__main__':
    unittest.main()
