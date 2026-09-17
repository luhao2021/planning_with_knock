"""Camera startup tests with all RealSense operations mocked; no hardware access."""
from pathlib import Path
import tempfile
import json
import numpy as np
import unittest
from unittest.mock import MagicMock, patch

import capture_rgbd


class CameraResetTests(unittest.TestCase):
    def test_reset_waits_for_same_serial(self):
        original=MagicMock();original.get_info.return_value='camera-A'
        other=MagicMock();other.get_info.return_value='camera-B'
        reconnected=MagicMock();reconnected.get_info.return_value='camera-A'
        context=MagicMock()
        context.query_devices.side_effect=[[original],[],[other],[reconnected]]
        with patch.object(capture_rgbd.rs,'context',return_value=context), \
             patch.object(capture_rgbd.time,'sleep') as sleep:
            self.assertEqual(capture_rgbd.reset_camera(),'camera-A')
        original.hardware_reset.assert_called_once_with()
        other.hardware_reset.assert_not_called()
        sleep.assert_any_call(5)
        self.assertEqual(context.query_devices.call_count,4)

    def test_reconnection_timeout_is_bounded(self):
        original=MagicMock();original.get_info.return_value='camera-A'
        context=MagicMock();context.query_devices.return_value=[original]
        with patch.object(capture_rgbd.rs,'context',return_value=context), \
             patch.object(capture_rgbd.time,'sleep'), \
             patch.object(capture_rgbd.time,'monotonic',side_effect=[0,21]):
            with self.assertRaisesRegex(RuntimeError,'did not reconnect'):
                capture_rgbd.reset_camera()
        original.hardware_reset.assert_called_once_with()

    def test_no_camera_or_multiple_cameras_are_not_reset(self):
        for count in (0,2):
            devices=[MagicMock() for _ in range(count)]
            context=MagicMock();context.query_devices.return_value=devices
            with patch.object(capture_rgbd.rs,'context',return_value=context):
                with self.assertRaisesRegex(RuntimeError,'Expected one RealSense'):
                    capture_rgbd.reset_camera()
            for device in devices:device.hardware_reset.assert_not_called()

    def test_capture_defaults_to_reset_and_binds_that_serial(self):
        pipeline=MagicMock();pipeline.wait_for_frames.side_effect=RuntimeError('test frame timeout')
        config=MagicMock()
        with tempfile.TemporaryDirectory() as d, \
             patch.object(capture_rgbd,'reset_camera',return_value='camera-A') as reset, \
             patch.object(capture_rgbd.rs,'pipeline',return_value=pipeline), \
             patch.object(capture_rgbd.rs,'config',return_value=config), \
             patch.object(capture_rgbd.rs,'align'):
            with self.assertRaisesRegex(RuntimeError,'test frame timeout'):
                capture_rgbd.capture(Path(d)/'capture.npz')
        reset.assert_called_once_with()
        config.enable_device.assert_called_once_with('camera-A')
        pipeline.stop.assert_called_once_with()

    def test_scene_setup_requests_one_reset_but_recorded_setup_uses_no_camera(self):
        import prepare_experiment_scene as setup
        for recorded in (False,True):
            with self.subTest(recorded=recorded), tempfile.TemporaryDirectory() as d:
                root=Path(d)
                mesh=root/'beach.obj';mesh.write_text('test mesh placeholder')
                names=['beach_1','beach_2','beach_3']
                config=root/'S06.json'
                config.write_text(json.dumps({'perception':{'automatic_scene':{'enabled':False},
                    'objects':{name:{'class':'beach','instance_id':str(i),'mesh':str(mesh)}
                               for i,name in enumerate(names,1)}}}))
                points={name:[10+i*10,20] for i,name in enumerate(names)}
                def fake_capture(path, **kwargs):
                    np.savez(path,color=np.zeros((48,64,3),dtype=np.uint8))
                argv=['prepare_experiment_scene.py','--config',str(config)]
                if recorded:
                    source=root/'source.npz';fake_capture(source)
                    argv.extend(['--capture',str(source)])
                with patch.object(setup,'ROOT',root), patch('sys.argv',argv), \
                     patch('capture_rgbd.capture',side_effect=fake_capture) as camera, \
                     patch.object(setup,'select_points',return_value=points):
                    setup.main()
                if recorded:
                    camera.assert_not_called()
                else:
                    self.assertEqual(camera.call_count,1)
                    self.assertEqual(camera.call_args.kwargs,{'reset_before':True})
                self.assertEqual(len(list((root/'real_robot/experiments/scene_setup').glob('*/pipeline_config.json'))),1)

    def test_post_action_capture_does_not_reset(self):
        pipeline=MagicMock();pipeline.wait_for_frames.side_effect=RuntimeError('test frame timeout')
        with tempfile.TemporaryDirectory() as d, \
             patch.object(capture_rgbd,'reset_camera') as reset, \
             patch.object(capture_rgbd.rs,'pipeline',return_value=pipeline), \
             patch.object(capture_rgbd.rs,'config'), \
             patch.object(capture_rgbd.rs,'align'):
            with self.assertRaisesRegex(RuntimeError,'test frame timeout'):
                capture_rgbd.capture(Path(d)/'capture.npz',reset_before=False)
        reset.assert_not_called()
        pipeline.stop.assert_called_once_with()


if __name__=='__main__':
    unittest.main()
