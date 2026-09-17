"""Physical-name association and removal checks on synthetic RGB-D; no hardware."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from unittest.mock import patch
from instance_identity import associate_masks,base_depth_points,removal_check,reference_distance
from pose_to_plan import base_to_optical
from pipeline_core import assess_action
from run_pipeline import SimulatedWorld


class IdentityTests(unittest.TestCase):
    workflow=dict(source_xy_min_mm=[300,-250],source_xy_max_mm=[570,250])

    def render(self,path,objects):
        h,w=480,640;fx=fy=600.;cx=320.;cy=240.
        yy,xx=np.indices((h,w));rays=np.stack(((xx-cx)/fx,(yy-cy)/fy,np.ones((h,w))),axis=-1)
        t=base_to_optical();rays=rays@t[:3,:3].T;origin=t[:3,3]
        depth=-origin[2]/rays[...,2];masks={}
        for obj in objects:
            center=np.asarray(obj['base_center_m']);top=center[2]+obj['dimensions_m'][2]/2
            boxdepth=(top-origin[2])/rays[...,2];p=origin+rays*boxdepth[...,None]
            c,s=np.cos(obj['yaw_rad']),np.sin(obj['yaw_rad'])
            local=(p[...,:2]-center[:2])@np.array([[c,-s],[s,c]])
            mask=np.all(np.abs(local)<=np.asarray(obj['dimensions_m'][:2])/2,axis=-1)&(boxdepth<depth)
            depth[mask]=boxdepth[mask];masks[obj['name']]=mask
        np.savez(path,color=np.zeros((h,w,3),np.uint8),depth=depth,depth_scale=1.,K=[fx,fy,cx,cy])
        return masks

    def test_swapped_labels_are_bound_to_positions_and_bowls_are_excluded(self):
        objects=SimulatedWorld.initial_scene()
        bowl=copy.deepcopy(objects[2]);bowl['base_center_m']=[.650,-.2,.015]
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);masks=self.render(d/'c.npz',objects[:2]+[bowl])
            np.savez(d/'m.npz',protex_1=masks['protex_2'],protex_2=masks['protex_1'],protex_3=masks['protex_3'])
            assigned,diag=associate_masks(d/'c.npz',d/'m.npz',objects[:2],self.workflow,[0,0,0])
            self.assertEqual(set(assigned),{'protex_1','protex_2'})
            self.assertTrue(np.array_equal(assigned['protex_1'],masks['protex_1']))
            self.assertTrue(np.array_equal(assigned['protex_2'],masks['protex_2']))
            self.assertEqual({(m['name'],m['tracker_label']) for m in diag['matches']},
                             {('protex_1','protex_2'),('protex_2','protex_1')})

    def test_knock_reference_follows_last_pose_and_commanded_direction(self):
        obj=SimulatedWorld.initial_scene()[0];obj['base_center_m'][1]=.12
        action=dict(kind='knock',target=obj['name'],base_direction=[0,1,0],intended_slide_mm=50)
        distance,limit=reference_distance(np.array([435,165]),obj,action)
        self.assertEqual(distance,0)
        self.assertGreater(reference_distance(np.array([435,40]),obj,action)[0],limit)
        self.assertGreater(reference_distance(np.array([435,180]),obj,None)[0],limit)

    def test_50mm_identity_allowance_and_separate_surface_reference(self):
        obj=SimulatedWorld.initial_scene()[0]
        base=np.asarray(obj['base_center_m'][:2])*1000
        for delta in (30.3975,49.9,50.):
            distance,limit=reference_distance(base+[delta,0],obj,None)
            self.assertLessEqual(distance,limit)
        distance,limit=reference_distance(base+[22.7+8,0],obj,None,base+[22.7,0])
        self.assertAlmostEqual(distance,8.)
        self.assertEqual(limit,50.)

    def test_depth_confirms_removal_without_any_bowl_identity(self):
        objects=SimulatedWorld.initial_scene()[:2]
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);self.render(d/'before.npz',objects)
            masks=self.render(d/'after.npz',objects[1:]);np.savez(d/'m.npz',**masks)
            action=dict(kind='pick',target='protex_1')
            assigned,diag=associate_masks(d/'after.npz',d/'m.npz',objects,self.workflow,[0,0,0],action,d/'before.npz')
            self.assertTrue(diag['removal']['confirmed'])
            self.assertEqual(set(assigned),{'protex_2'})
            removed=dict(objects[0],perception_status='source_removed',source_removal_evidence=diag['removal'])
            self.assertTrue(assess_action(objects,[removed],action,{},self.workflow)['completed'])

    def test_missing_label_with_box_still_present_or_invalid_depth_is_not_removal(self):
        obj=SimulatedWorld.initial_scene()[0]
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);self.render(d/'c.npz',[obj]);np.savez(d/'m.npz')
            _,diag=associate_masks(d/'c.npz',d/'m.npz',[obj],self.workflow,[0,0,0],dict(kind='pick',target=obj['name']),d/'c.npz')
            self.assertFalse(diag['removal']['confirmed'])
            points=base_depth_points(d/'c.npz',[0,0,0]);points[:]=np.nan
            self.assertFalse(removal_check(d/'c.npz',points,obj,[0,0,0],d/'c.npz')['confirmed'])

    def test_sdk_retirement_removes_label_and_cached_pose_before_detection(self):
        from run_pipeline import LivePerception
        with tempfile.TemporaryDirectory() as d:
            world=LivePerception.__new__(LivePerception);world.run_dir=Path(d)
            world._identity_registry={'protex_1':dict(status='active',initial_pose={'name':'protex_1'})}
            world._last_accepted_pose_records={'protex_1':{'record':{}},'protex_2':{'record':{}}}
            world.retire_target('protex_1',Path(d)/'actions/0000')
            registry=json.loads((Path(d)/'identity_registry.json').read_text())
            self.assertEqual(registry['protex_1']['status'],'removed')
            self.assertFalse(registry['protex_1']['visually_verified'])
            self.assertEqual(set(world._last_accepted_pose_records),{'protex_2'})
            self.assertTrue((Path(d)/'actions/0000/identity_retirement.json').exists())

    def test_live_perception_retires_names_and_handles_last_pick_without_pose_call(self):
        from run_pipeline import LivePerception
        import shutil
        root=Path(__file__).parent.resolve()
        objects=SimulatedWorld.initial_scene()[:2]
        with tempfile.TemporaryDirectory(dir=root) as d:
            d=Path(d);captures=[];masks=[]
            for i,scene in enumerate((objects,objects[1:],[])):
                captures.append(d/f'input{i}.npz');masks.append(self.render(captures[-1],scene))
            # Simulate a remaining box carrying the just-picked box's SAM label.
            masks[1]={'protex_1':masks[1]['protex_2']}
            world=LivePerception.__new__(LivePerception)
            world.run_dir=d/'run';world.frames=world.run_dir/'frames';world.frames.mkdir(parents=True)
            world.recorded_capture=None;world.base_pose_offset_mm=[0,0,0];world.workflow=self.workflow
            world.points=d/'points.json';world.points.write_text('{}')
            world.specs={o['name']:dict(mesh=str(root/'unused.obj')) for o in objects}
            world.config={key:str(root/'unused') for key in ('sam2_python','checkpoint','library','refiner','scorer')}
            world.config['docker_image']='unused'
            sam_calls=[];pose_calls=[];frame=[-1]
            def capture(path, **kwargs):
                frame[0]+=1;shutil.copy2(captures[frame[0]],path)
            def run(argv,log):
                if any(str(x).endswith('run_sam2_tracking.py') for x in argv):
                    sam_calls.append(argv);out=Path(log).parent/'masks';out.mkdir()
                    np.savez(out/'instance_masks.npz',**masks[frame[0]])
                else:
                    pose_calls.append(argv);out=Path(log).parent/'foundationpose';out.mkdir()
                    requested=argv[argv.index('--instances')+1:] if '--instances' in argv else list(world.specs)
                    records=[dict(**{'class':o['kind']},instance_id=o['instance_id'],score=1,
                        T_camera_object=(np.linalg.inv(base_to_optical())@np.asarray(o['T_base_object'])).tolist())
                        for o in objects if o['name'] in requested]
                    (out/'foundationpose_instances.json').write_text(json.dumps(dict(instances=records)))
            world.run=run
            with patch('capture_rgbd.capture',side_effect=capture):
                first=world.observe(0,world.run_dir/'observations/0000')
                second=world.observe(1,world.run_dir/'observations/0001',targets=['protex_1','protex_2'],
                    active_objects=first,last_action=dict(kind='pick',target='protex_1'))
                remaining=[o for o in second if o['name']=='protex_2']
                last=world.observe(2,world.run_dir/'observations/0002',targets=['protex_2'],
                    active_objects=remaining,last_action=dict(kind='pick',target='protex_2'))
            self.assertEqual(len(pose_calls),2)
            self.assertEqual(pose_calls[-1][pose_calls[-1].index('--instances')+1:],['protex_2'])
            self.assertEqual(sam_calls[-1][sam_calls[-1].index('--active-instances')+1:],['protex_2'])
            self.assertEqual(last[0]['perception_status'],'source_removed')
            registry=json.loads((world.run_dir/'identity_registry.json').read_text())
            self.assertTrue(all(v['status']=='removed' for v in registry.values()))
            self.assertEqual(registry['protex_1']['initial_pose'],json.loads(json.dumps(first[0])))

    def test_pick_that_only_displaces_box_within_source_does_not_retire_it(self):
        obj=SimulatedWorld.initial_scene()[0]
        moved=copy.deepcopy(obj);moved['base_center_m'][1]-=.10
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);self.render(d/'before.npz',[obj]);masks=self.render(d/'after.npz',[moved])
            np.savez(d/'m.npz',**masks)
            _,diag=associate_masks(d/'after.npz',d/'m.npz',[obj],self.workflow,[0,0,0],
                dict(kind='pick',target=obj['name']),d/'before.npz')
            self.assertFalse(diag['removal']['confirmed'])

    def test_recorded_last_box_side_fragment_does_not_veto_pick_removal(self):
        fixture=Path(__file__).parent/'testdata/pick_side_fragment_014917'
        meta=json.loads(fixture.with_suffix('.json').read_text())
        with np.load(fixture.with_suffix('.npz')) as data:
            points=data['points'];masks={n:data[n] for n in data.files if n!='points'}
        for reference_shift in (0.,-5.):
            # Also cover the fragment lying just inside the picked identity's
            # distance gate. It still must not keep a removed identity alive.
            refs=copy.deepcopy(meta['mask_references']);refs['protex_2'][0]+=reference_shift
            with self.subTest(reference_shift=reference_shift),tempfile.TemporaryDirectory() as d:
                path=Path(d)/'masks.npz';np.savez(path,**masks)
                with patch('instance_identity.base_depth_points',return_value=points), \
                     patch('instance_identity.removal_check',return_value=copy.deepcopy(meta['removal'])):
                    assigned,diag=associate_masks('unused',path,meta['objects'],meta['workflow'],[15,-10,0],
                                                  meta['action'],mask_references=refs)
                self.assertTrue(diag['removal']['confirmed'])
                self.assertEqual(diag['removal']['source_mask_count'],2)
                self.assertEqual(diag['removal']['distinct_source_box_count'],1)
                self.assertEqual(diag['removal']['surface_fragments'][0]['belongs_to'],'protex_1')
                self.assertEqual(set(assigned),{'protex_1'})
                np.testing.assert_array_equal(assigned['protex_1'],masks['protex_1'])

    def test_separate_small_source_mask_still_prevents_false_removal(self):
        fixture=Path(__file__).parent/'testdata/pick_side_fragment_014917'
        meta=json.loads(fixture.with_suffix('.json').read_text())
        with np.load(fixture.with_suffix('.npz')) as data:
            points=data['points'].copy();masks={n:data[n] for n in data.files if n!='points'}
        # A small mask of a displaced target away from the remaining box is
        # still a separate source candidate, even if the old source is clear.
        points[masks['protex_2'],0]+=70
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'masks.npz';np.savez(path,**masks)
            with patch('instance_identity.base_depth_points',return_value=points), \
                 patch('instance_identity.removal_check',return_value=copy.deepcopy(meta['removal'])):
                _,diag=associate_masks('unused',path,meta['objects'],meta['workflow'],[15,-10,0],
                                      meta['action'],mask_references=meta['mask_references'])
            self.assertFalse(diag['removal']['confirmed'])
            self.assertEqual(diag['removal']['distinct_source_box_count'],2)
            self.assertEqual(diag['removal']['surface_fragments'],[])

    def test_duplicate_mask_never_assigns_two_physical_names(self):
        objects=SimulatedWorld.initial_scene()[:2]
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);mask=self.render(d/'c.npz',objects)['protex_1']
            np.savez(d/'m.npz',protex_1=mask,protex_2=mask)
            assigned,_=associate_masks(d/'c.npz',d/'m.npz',objects,self.workflow,[0,0,0])
            self.assertEqual(set(assigned),{'protex_1'})


if __name__=='__main__':unittest.main()
