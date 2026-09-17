"""Offline initialization tests; no camera, SDK, or learned-model dependency."""
import json
import tempfile
from pathlib import Path
import unittest
import shutil
from unittest.mock import patch
import numpy as np
from automatic_scene import prepare_grid,select_instances
from pose_to_plan import base_to_optical
from xarm_executor import UnsafePlan

HERE=Path(__file__).parent


class AutomaticSceneTests(unittest.TestCase):
    def synthetic(self,folder,count,yaw=0):
        h,w=480,640;fx=fy=600.;cx=320.;cy=240.
        v,u=np.indices((h,w));rays=np.stack(((u-cx)/fx,(v-cy)/fy,np.ones((h,w))),-1)
        t=base_to_optical();direction=rays@t[:3,:3].T
        floor=(0-t[2,3])/direction[:,:,2]
        top=(.033-t[2,3])/direction[:,:,2]
        xy=(top[...,None]*direction+t[:3,3])[:,:,:2]
        depth=floor.copy();masks={};records=[]
        angle=np.deg2rad(yaw);r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        locations=[(.40,-.15),(.40,0),(.40,.15),(.52,-.15),(.52,0)]
        for i,center in enumerate(locations[:count]):
            local=(xy-np.array(center))@r
            mask=(np.abs(local[:,:,0])<.045)&(np.abs(local[:,:,1])<.030)
            depth[mask]=top[mask];name=f'candidate_{i}'
            masks[name]=mask;records.append(dict(name=name,predicted_iou=.95,stability_score=.98))
        path=folder/'capture.npz'
        np.savez(path,color=np.zeros((h,w,3),np.uint8),depth=depth,depth_scale=1.,K=[fx,fy,cx,cy])
        workflow=json.loads((HERE/'pipeline_config.example.json').read_text())['workflow']
        return path,workflow,masks,records

    def test_variable_count_and_rotated_layouts_generate_interior_seeds(self):
        for count in range(1,5):
            for angle in (0,35,90):
                with self.subTest(count=count,angle=angle), tempfile.TemporaryDirectory() as d:
                    p=Path(d);capture,w,masks,records=self.synthetic(p,count,angle)
                    points=select_instances(capture,w,[15,-10,0],masks,records,p/'selection')
                    self.assertEqual(set(points),{f'protex_{i}' for i in range(1,count+1)})
                    for u,v in points.values():self.assertTrue(any(mask[v,u] for mask in masks.values()))
                    grid,n=prepare_grid(capture,w,[15,-10,0],p/'grid')
                    self.assertGreater(n,0);self.assertTrue(grid.is_file())

    def test_live_startup_generates_specs_without_manual_fields_and_only_seeds_once(self):
        from run_pipeline import LivePerception
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d).resolve();capture,workflow,masks,records=self.synthetic(p,2)
            dummy=p/'dummy';dummy.write_text('test fixture')
            config={key:str(dummy) for key in ('python','sam2_python','checkpoint','library','refiner','scorer')}
            config.update(docker_image='unused',base_pose_offset_mm=[15,-10,0],
                          automatic_scene=dict(enabled=True,kind='protex',max_instances=4,mesh=str(dummy)))
            world=LivePerception(config,p/'run',recorded_capture=capture,workflow=workflow)
            calls=[]
            def fake_run(argv,logfile):
                if 'run_sam2_masks.py' in str(argv[1]):
                    calls.append('automatic');out=Path(argv[argv.index('--out')+1]);out.mkdir(parents=True)
                    np.savez_compressed(out/'sam2_masks.npz',**masks)
                    (out/'sam2_candidates.json').write_text(json.dumps(records))
                elif '--object-specs' in argv:
                    calls.append('poses');folder=Path(logfile).parent/'foundationpose';folder.mkdir()
                    instances=[]
                    for i in range(len(world.initial_names)):
                        t=np.eye(4);t[:3,3]=[.40,-.15+i*.15,.015]
                        instances.append(dict(**{'class':'protex'},instance_id=str(i+1),score=1,
                                              T_camera_object=(np.linalg.inv(base_to_optical())@t).tolist()))
                    (folder/'foundationpose_instances.json').write_text(json.dumps(dict(instances=instances)))
                else:calls.append('tracking')
            world.run=fake_run
            with patch('perception_validation.validate_model_poses',return_value=[]):
                first=world.observe(0,p/'run/observations/0000')
                world.recorded_capture=None
                with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)):
                    second=world.observe(1,p/'run/observations/0001')
            self.assertEqual(len(first),2);self.assertEqual(len(second),2)
            self.assertEqual(calls.count('automatic'),1);self.assertEqual(calls.count('tracking'),2)
            self.assertEqual(set(json.loads(world.points.read_text())),world.initial_names)
            self.assertEqual(set(json.loads((world.run_dir/'object_specs.json').read_text())),world.initial_names)

    def test_expected_four_never_silently_uses_two(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);capture,w,masks,records=self.synthetic(p,2)
            with self.assertRaisesRegex(UnsafePlan,'found 2 boxes; expected 4'):
                select_instances(capture,w,[15,-10,0],masks,records,p/'selection',expected_instances=4)

    def test_startup_seed_recovery_once_on_same_capture(self):
        from run_pipeline import LivePerception
        for succeeds in (True,False):
            with self.subTest(succeeds=succeeds),tempfile.TemporaryDirectory(dir=HERE) as d:
                p=Path(d).resolve();capture,workflow,masks,records=self.synthetic(p,2)
                dummy=p/'dummy';dummy.write_text('fixture')
                config={key:str(dummy) for key in ('python','sam2_python','checkpoint','library','refiner','scorer')}
                config.update(docker_image='unused',base_pose_offset_mm=[15,-10,0],
                    automatic_scene=dict(enabled=True,kind='protex',max_instances=4,expected_instances=2,mesh=str(dummy)))
                world=LivePerception(config,p/'run',workflow=workflow);commands=[]
                def fake_run(argv,log):
                    if 'run_sam2_masks.py' in str(argv[1]):
                        commands.append(argv);out=Path(argv[argv.index('--out')+1]);out.mkdir(parents=True)
                        np.savez(out/'sam2_masks.npz',**masks)
                        selected=records if len(commands)==2 and succeeds else records[:1]
                        (out/'sam2_candidates.json').write_text(json.dumps(selected))
                    elif '--object-specs' in argv:
                        out=Path(log).parent/'foundationpose';out.mkdir()
                        items=[]
                        for i in range(2):
                            t=np.eye(4);t[:3,3]=[.40,-.15+i*.15,.015]
                            items.append(dict(**{'class':'protex'},instance_id=str(i+1),score=1,
                                T_camera_object=(np.linalg.inv(base_to_optical())@t).tolist()))
                        (out/'foundationpose_instances.json').write_text(json.dumps(dict(instances=items)))
                world.run=fake_run
                with patch('capture_rgbd.capture',side_effect=lambda path, **kwargs:shutil.copy2(capture,path)) as camera, \
                     patch('perception_validation.validate_model_poses',return_value=[]):
                    if succeeds:
                        result=world.observe(0,p/'run/observations/0000');self.assertEqual(len(result),2)
                    else:
                        with self.assertRaisesRegex(UnsafePlan,'expected 2'):
                            world.observe(0,p/'run/observations/0000')
                self.assertEqual(camera.call_count,1);self.assertEqual(len(commands),2)
                self.assertNotIn('--refine-proposals',commands[0]);self.assertIn('--refine-proposals',commands[1])
                self.assertEqual(commands[0][commands[0].index('--capture')+1],commands[1][commands[1].index('--capture')+1])

    def test_duplicate_masks_do_not_inflate_count(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);capture,w,masks,records=self.synthetic(p,2)
            masks['duplicate']=masks['candidate_0'].copy()
            records.append(dict(name='duplicate',predicted_iou=.94,stability_score=.98))
            self.assertEqual(len(select_instances(capture,w,[15,-10,0],masks,records,p/'selection')),2)

    def test_missing_mask_stops_instead_of_silently_reducing_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);capture,w,masks,records=self.synthetic(p,3)
            with self.assertRaisesRegex(UnsafePlan,'unexplained raised'):
                select_instances(capture,w,[15,-10,0],masks,records[:2],p/'selection')

    def test_more_than_four_protex_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);capture,w,masks,records=self.synthetic(p,5)
            with self.assertRaisesRegex(UnsafePlan,'above configured inventory'):
                select_instances(capture,w,[15,-10,0],masks,records,p/'selection')

    def test_empty_table_generates_empty_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);capture,w,masks,records=self.synthetic(p,0)
            self.assertEqual(select_instances(capture,w,[15,-10,0],masks,records,p/'selection'),{})

    def test_invalid_depth_does_not_look_like_an_empty_table(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);capture,w,_,_=self.synthetic(p,0)
            np.savez(capture,color=np.zeros((480,640,3),np.uint8),depth=np.zeros((480,640)),depth_scale=1.,K=[600,600,320,240])
            with self.assertRaisesRegex(UnsafePlan,'insufficient valid'):
                prepare_grid(capture,w,[15,-10,0],p/'grid')


if __name__=='__main__':unittest.main()
