"""Bounded recovery orchestration with real depth validation; no hardware/GPU."""
import copy
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from run_pipeline import LivePerception
from xarm_executor import UnsafePlan

HERE=Path(__file__).parent.resolve()


class PerceptionRecoveryTests(unittest.TestCase):
    def setup_world(self,p,*,fail_first=False,recovery_success=True,wrong_identity=False):
        capture=p/'fixture.npz'
        np.savez(capture,color=np.zeros((200,320,3),np.uint8),depth=np.full((200,320),.985),depth_scale=1.,K=[1000,1000,160,100])
        masks={}
        for name,x in [('protex_1',100),('protex_2',220)]:
            mask=np.zeros((200,320),bool);mask[71:129,x-44:x+44]=True;masks[name]=mask
        world=LivePerception.__new__(LivePerception)
        world.run_dir=p/'run';world.frames=world.run_dir/'frames';world.frames.mkdir(parents=True)
        world.recorded_capture=None;world.base_pose_offset_mm=[0,0,0]
        world.points=world.run_dir/'initial_points.json';world.points.write_text('{}')
        world.config={key:str(HERE/'dummy') for key in ('python','sam2_python','checkpoint','library','refiner','scorer')}
        world.config['docker_image']='unused'
        world.specs={name:dict(mesh=str(HERE/'models/protex_90x60x30mm.obj')) for name in masks}
        calls={'sam':0,'registration':0,'recovery':[]}
        def record(i,bad=False):
            t=np.eye(4);t[:3,3]=[-.06 if i==1 else (.26 if bad else .06),0,1]
            return {'class':'protex','instance_id':str(i),'score':123+i,'T_camera_object':t.tolist()}
        def fake_run(argv,logfile):
            if any(str(x).endswith('run_sam2_tracking.py') for x in argv):
                calls['sam']+=1;folder=Path(logfile).parent/'masks';folder.mkdir()
                np.savez(folder/'instance_masks.npz',**masks);return
            retry=Path(logfile).name=='process.log'
            if retry:
                calls['recovery'].append(copy.deepcopy(argv))
                folder=Path(logfile).parent
                records=[record(2,bad=not recovery_success)]
                if wrong_identity:records[0]['instance_id']='9'
            else:
                calls['registration']+=1;folder=Path(logfile).parent/'foundationpose';folder.mkdir()
                bad=fail_first or calls['registration']==2
                records=[record(1),record(2,bad=bad)]
                if '--instances' in argv:
                    requested=argv[argv.index('--instances')+1:]
                    records=[r for r in records if f"{r['class']}_{r['instance_id']}" in requested]
            (folder/'foundationpose_instances.json').write_text(json.dumps(dict(instances=records)))
        world.run=fake_run
        return world,capture,calls

    def test_previous_frame_recovery_keeps_good_pose_and_uses_no_extra_capture(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);world,capture,calls=self.setup_world(p)
            with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)) as camera:
                first=world.run_dir/'observations/0000';second=world.run_dir/'observations/0001'
                world.observe(0,first)
                world._last_accepted_pose_records={}  # Exercise model retry when no cached pose is available.
                result=world.observe(1,second)
            self.assertEqual(camera.call_count,2);self.assertEqual(calls['sam'],2)
            self.assertEqual([call.kwargs['reset_before'] for call in camera.call_args_list],[True,False])
            self.assertEqual(calls['registration'],2);self.assertEqual(len(calls['recovery']),1)
            retry=calls['recovery'][0];self.assertIn('--previous-capture',retry)
            self.assertEqual(retry[retry.index('--instances')+1:],['protex_2'])
            self.assertEqual(world._previous_observation_folder,second)
            accepted=json.loads((second/'foundationpose/accepted_instances.json').read_text())
            raw=json.loads((second/'foundationpose/foundationpose_instances.json').read_text())
            self.assertEqual(accepted['instances'][0],raw['instances'][0])
            self.assertNotEqual(accepted['instances'][1],raw['instances'][1])
            self.assertEqual(accepted['pose_sources']['protex_2'],'previous_frame_tracking')
            self.assertEqual(len(result),2)
            self.assertTrue(all(c['passed'] for c in json.loads((second/'foundationpose/accepted_validation.json').read_text())))

    def test_previous_accepted_pose_is_revalidated_without_gpu_retry(self):
        for targets in (None,['protex_2']):
            with self.subTest(targets=targets),tempfile.TemporaryDirectory(dir=HERE) as d:
                p=Path(d);world,capture,calls=self.setup_world(p,recovery_success=False)
                first=world.run_dir/'observations/0000';second=world.run_dir/'observations/0001'
                with patch('capture_rgbd.capture',side_effect=lambda path,**kwargs:shutil.copy2(capture,path)) as camera:
                    world.observe(0,first)
                    result=world.observe(1,second,targets=targets)
                self.assertEqual(camera.call_count,2)
                self.assertEqual(calls['recovery'],[])
                self.assertEqual(calls['sam'],2)
                accepted=json.loads((second/'foundationpose/accepted_instances.json').read_text())
                original=json.loads((first/'foundationpose/accepted_instances.json').read_text())
                target=lambda records:next(r for r in records if r['instance_id']=='2')
                self.assertEqual(target(accepted['instances']),target(original['instances']))
                self.assertEqual(accepted['pose_sources']['protex_2'],'previous_accepted_pose_revalidated')
                summary=json.loads((second/'foundationpose/recovery/previous_accepted_pose/summary.json').read_text())
                self.assertEqual(summary['accepted_instances'],['protex_2'])
                self.assertEqual(summary['attempts'][0]['previous_observation'],0)
                self.assertTrue(all(c['passed'] for c in json.loads((second/'foundationpose/accepted_validation.json').read_text())))
                self.assertIn('protex_2',[o['name'] for o in result])

    def test_target_only_skips_unrelated_bad_registration(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);world,capture,calls=self.setup_world(p)
            first=world.run_dir/'observations/0000';second=world.run_dir/'observations/0001'
            with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)) as camera:
                world.observe(0,first)
                result=world.observe(1,second,targets=['protex_1'])
            self.assertEqual(camera.call_count,2)
            self.assertEqual([o['name'] for o in result],['protex_1'])
            self.assertEqual(calls['recovery'],[])
            checks=json.loads((second/'foundationpose/accepted_validation.json').read_text())
            self.assertEqual([c['name'] for c in checks],['protex_1'])
            self.assertEqual(json.loads((second/'perception_scope.json').read_text())['estimated_instances'],['protex_1'])

    def test_target_only_still_recovers_target_with_previous_frame(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);world,capture,calls=self.setup_world(p)
            first=world.run_dir/'observations/0000';second=world.run_dir/'observations/0001'
            with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)):
                world.observe(0,first)
                world._last_accepted_pose_records={}
                result=world.observe(1,second,targets=['protex_2'])
            self.assertEqual([o['name'] for o in result],['protex_2'])
            self.assertEqual(len(calls['recovery']),1)
            retry=calls['recovery'][0]
            self.assertEqual(retry.count('--instances'),1)
            self.assertEqual(retry[retry.index('--instances')+1:],['protex_2'])
            self.assertIn('--previous-capture',retry)
            self.assertIn('--previous-masks',retry)

    def test_only_unrelated_mask_ambiguity_is_ignored(self):
        from run_sam2_tracking import check_mask_identity
        left=np.zeros((20,20),bool);left[:,:10]=True
        right=~left
        masks={'protex_1':left,'protex_2':left.copy(),'protex_4':right}
        check_mask_identity(masks,{'protex_4'})
        for required in (None,{'protex_1'}):
            with self.assertRaisesRegex(RuntimeError,'ambiguous tracked instances'):
                check_mask_identity(masks,required)

    def test_first_observation_uses_one_higher_hypothesis_retry(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);world,capture,calls=self.setup_world(p,fail_first=True)
            with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)) as camera:
                world.observe(0,world.run_dir/'observations/0000')
            self.assertEqual(camera.call_count,1);self.assertEqual(len(calls['recovery']),1)
            retry=calls['recovery'][0];self.assertNotIn('--previous-capture',retry)
            self.assertEqual(retry[retry.index('--n-hypotheses')+1],'256')

    def test_failed_recovery_stops_once_without_stale_pose_fallback(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);world,capture,calls=self.setup_world(p,recovery_success=False)
            first=world.run_dir/'observations/0000';second=world.run_dir/'observations/0001'
            with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)) as camera:
                world.observe(0,first)
                # A genuinely moved/bad cached pose must fail the same current-depth checks.
                world._last_accepted_pose_records['protex_2']['record']['T_camera_object'][0][3]=.26
                with self.assertRaisesRegex(UnsafePlan,'pose recovery failed after one attempt'):
                    world.observe(1,second)
            self.assertEqual(camera.call_count,2);self.assertEqual(len(calls['recovery']),1)
            self.assertEqual(world._previous_observation_folder,first)
            self.assertFalse((second/'observed_objects.json').exists())
            self.assertEqual(json.loads((second/'foundationpose/recovery/summary.json').read_text())['status'],'failed')

    def test_recovery_cannot_replace_an_unrequested_instance(self):
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d);world,capture,calls=self.setup_world(p,fail_first=True,wrong_identity=True)
            with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)):
                with self.assertRaisesRegex(UnsafePlan,'unexpected instance identities'):
                    world.observe(0,world.run_dir/'observations/0000')
            self.assertEqual(len(calls['recovery']),1)


if __name__=='__main__':unittest.main()
