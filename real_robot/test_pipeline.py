"""Offline workflow and SDK-boundary tests. Never connect to hardware."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from batch_executor import XArmBatchExecutor, build_batch, checked_batch_config, direction_base
from pipeline_core import run_loop, snapshot_plan
from run_pipeline import SimulatedWorld
from xarm_executor import UnsafePlan

HERE=Path(__file__).parent


def configs(physical=False):
    p=json.loads((HERE/'pipeline_config.example.json').read_text())
    c=json.loads((HERE/'execution_config.batch.example.json').read_text())
    c.update(pick_tcp_z_mm=15,release_tcp_z_mm=90)
    c['knock'].update(contact_tcp_z_mm=30,top_indent_mm=None,slide_descent_mm=0)
    c['pick_fingertip_clearance_mm']=None
    return checked_batch_config(c,physical=physical),p['workflow']


class LoopTests(unittest.TestCase):
    def test_complete_loop_observes_only_between_batches(self):
        cfg,w=configs(); world=SimulatedWorld(cfg)
        with tempfile.TemporaryDirectory() as d:
            state=run_loop(world,world,cfg,w,Path(d),render=False)
            self.assertEqual(state['status'],'complete')
            self.assertEqual(set(state['completed']),world.initial_names)
            self.assertEqual([e[0] for e in world.events], ['observe']+['execute','observe']*state['actions_executed'])
            # Repeated Knocks keep the same target until a clear Pick is available.
            events=[e for e in world.events if e[0]=='execute']
            for i,e in enumerate(events):
                if e[1]=='knock': self.assertEqual(events[i+1][2],e[2])
            self.assertGreater(len(events),4)

    def test_single_action_observes_after_retreat_without_replanning(self):
        cfg,w=configs(); world=SimulatedWorld(cfg)
        with tempfile.TemporaryDirectory() as d:
            state=run_loop(world,world,cfg,w,Path(d),render=False,single_action=True)
            self.assertEqual(state['status'],'single_action_complete')
            self.assertEqual(state['actions_executed'],1)
            self.assertEqual([e[0] for e in world.events],['observe','execute','observe'])
            self.assertTrue((Path(d)/'actions/0000/outcome.json').is_file())
            self.assertFalse((Path(d)/'observations/0001/planning').exists())
            self.assertTrue((Path(d)/'observations/0000/planning/fixed_execution_plan.json').is_file())
            execution=json.loads((Path(d)/'actions/0000/simulated_execution.json').read_text())
            self.assertEqual(execution['stages_completed'][-1],'retreat')
            self.assertFalse((Path(d)/'actions/0001').exists())

    def test_no_progress_uses_full_knock_budget_without_escalating_depth(self):
        cfg,w=configs(); world=SimulatedWorld(cfg,knock_scale=0)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(UnsafePlan,r'attempt limit reached \(25/25\)'):
                run_loop(world,world,cfg,w,Path(d),render=False)
            self.assertEqual(sum(e[0]=='execute' for e in world.events),25)
            a=json.loads((Path(d)/'actions/0000/action.json').read_text())
            b=json.loads((Path(d)/'actions/0024/action.json').read_text())
            self.assertEqual(a['stages'],b['stages'])

    def test_failed_pick_is_bounded(self):
        cfg,w=configs(); world=SimulatedWorld(cfg,pick_success=False)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(UnsafePlan,'attempt limit'):
                run_loop(world,world,cfg,w,Path(d),render=False)
            state=json.loads((Path(d)/'state.json').read_text())
            self.assertEqual(state['completed'],[])
            self.assertEqual(max(v for k,v in state['attempts'].items() if k.endswith(':pick')),1)

    def test_twenty_five_progressing_knocks_allowed_but_twenty_sixth_stops(self):
        cfg,w=configs();world=SimulatedWorld(cfg,knock_scale=.1)
        candidate=dict(target='protex_3',kind='knock',witness='knock--x',blockers=[])
        with tempfile.TemporaryDirectory() as d, patch('fixed_execution.select_fixed_candidate',return_value=(candidate,{})):
            with self.assertRaisesRegex(UnsafePlan,r'attempt limit reached \(25/25\)'):
                run_loop(world,world,cfg,w,Path(d),render=False)
            self.assertEqual(sum(e[0]=='execute' for e in world.events),25)
            self.assertEqual(sum(e[0]=='observe' for e in world.events),26)
            first=json.loads((Path(d)/'actions/0000/action.json').read_text())
            last=json.loads((Path(d)/'actions/0024/action.json').read_text())
            self.assertEqual(first['height_geometry'],last['height_geometry'])
            self.assertEqual(first['intended_slide_mm'],last['intended_slide_mm'])

    def test_missing_after_pick_does_not_count_as_placement(self):
        cfg,w=configs()
        class Missing(SimulatedWorld):
            def execute(self,action,folder):
                super().execute(action,folder)
                if action['kind']=='pick': self.objects=[o for o in self.objects if o['name']!=action['target']]
        world=Missing(cfg)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(UnsafePlan,'absence alone'):
                run_loop(world,world,cfg,w,Path(d),render=False)
            self.assertFalse(json.loads((Path(d)/'state.json').read_text())['completed'])

    def test_horizontal_ros_direction_and_closed_fingers(self):
        cfg,w=configs(); objects=SimulatedWorld.initial_scene()
        with tempfile.TemporaryDirectory() as d:
            layout=snapshot_plan(objects,d,w,render=False)
            action=build_batch(layout,dict(target='protex_3',kind='knock',witness='knock--x',blockers=[]),cfg)
            self.assertTrue(np.allclose(action['base_direction'],[0,-1,0]))
            contact=next(s for s in action['stages'] if s['phase']=='contact')['tcp_mm_deg']
            stroke=next(s for s in action['stages'] if s['phase']=='horizontal-knock')['tcp_mm_deg']
            self.assertEqual(contact[2],stroke[2])
            self.assertAlmostEqual(stroke[1]-contact[1],-50)
            self.assertEqual(action['stages'][0]['gripper_pulse'],0)
            self.assertEqual(action['stages'][2]['speed_mm_s'],112.5)
            # Height is the explicitly supplied TCP contact height.
            self.assertAlmostEqual(contact[2],30)

    def test_old_approval_flags_are_not_used(self):
        cfg,_=configs(physical=True)
        cfg.update(provisional=True,tcp_calibrated=False,gripper_calibrated=False,
                   contact_execution_reviewed=False,table_plane_verified=False,
                   knock_contact_validated=False,bowls_placed_and_checked=False)
        checked_batch_config(cfg,physical=True)

    def test_preview_saves_plan_without_inventing_contact_height(self):
        cfg,w=configs();cfg['knock']['contact_tcp_z_mm']=None
        world=SimulatedWorld(cfg)
        with tempfile.TemporaryDirectory() as d:
            state=run_loop(world,None,cfg,w,Path(d),preview=True,render=False)
            self.assertEqual(state['status'],'preview')
            self.assertEqual(state['missing_motion_values'],['knock.contact_tcp_z_mm'])
            self.assertEqual(world.events,[('observe',0)])
            self.assertTrue((Path(d)/'observations/0000/planning/layout_and_plan.json').exists())
            self.assertTrue((Path(d)/'actions/0000/action_request.json').exists())

    def test_missing_numeric_contact_height_is_actionable(self):
        cfg,_=configs();cfg['knock']['contact_tcp_z_mm']=None
        with self.assertRaisesRegex(UnsafePlan,'knock.contact_tcp_z_mm'):
            XArmBatchExecutor(cfg)

    def test_standard_gripper_y_axis_matches_solver_closing_axis(self):
        from scipy.spatial.transform import Rotation
        cfg,w=configs();world=SimulatedWorld(cfg)
        world.objects[2]['base_center_m'][1]-=.05
        world.objects[2]['T_base_object'][1][3]-=.05
        with tempfile.TemporaryDirectory() as d:
            layout=snapshot_plan(world.objects,d,w,render=False)
            candidate=next(a for a in layout['immediately_feasible_actions'] if a['kind']=='pick' and a['target']=='protex_3')
            action=build_batch(layout,candidate,cfg,bowl='bowl_1')
            pose=next(s['tcp_mm_deg'] for s in action['stages'] if s['phase']=='pick')
            actual=Rotation.from_euler('xyz',pose[3:],degrees=True).as_matrix()[:,1]
            desired=direction_base(world.objects[2],'+'+candidate['witness'][5])
            self.assertAlmostEqual(abs(float(actual@desired)),1)


class RequestedHeightTests(unittest.TestCase):
    def test_geometry_preserves_clearance_and_requested_knock_slope(self):
        from standard_gripper_geometry import tip_extension_mm, tip_distance_from_flange_mm
        cfg=json.loads((HERE/'execution_config.batch.example.json').read_text())
        cfg['knock']['nominal_height_tcp_classes']=[]
        cfg=checked_batch_config(cfg,physical=True)
        _,workflow=configs()
        objects=SimulatedWorld.initial_scene()
        # A nonzero support height verifies that base Z and table clearance differ.
        for obj in objects:
            obj['base_center_m'][2]+=.005
            obj['T_base_object'][2][3]+=.005
        objects[2]['base_center_m'][1]-=.05
        objects[2]['T_base_object'][1][3]-=.05
        with tempfile.TemporaryDirectory() as d:
            layout=snapshot_plan(objects,d,workflow,render=False)
            pick=next(a for a in layout['immediately_feasible_actions'] if a['kind']=='pick' and a['target']=='protex_3')
            batch=build_batch(layout,pick,cfg,bowl='bowl_1')
            z=next(s['tcp_mm_deg'][2] for s in batch['stages'] if s['phase']=='pick')
            for q in np.linspace(0,.85,1001):
                tip_z=z-(tip_distance_from_flange_mm(q)-172)
                self.assertGreaterEqual(tip_z-5,10-1e-8)
            self.assertAlmostEqual(z,5+10+tip_extension_mm(172))
            self.assertEqual(next(s['tcp_mm_deg'][2] for s in batch['stages'] if s['phase']=='place'),60)
            batch=build_batch(layout,dict(target='protex_3',kind='knock',witness='knock--x',blockers=[]),cfg)
            start=next(s['tcp_mm_deg'] for s in batch['stages'] if s['phase']=='contact')
            end=next(s['tcp_mm_deg'] for s in batch['stages'] if s['phase']=='horizontal-knock')
            self.assertAlmostEqual(start[2]-tip_extension_mm(172,closed=True),33)
            self.assertAlmostEqual(end[2]-tip_extension_mm(172,closed=True),32)
            self.assertAlmostEqual(end[1]-start[1],-50)
            self.assertAlmostEqual(end[0],start[0])
            self.assertEqual(batch['stages'][0]['gripper_pulse'],0)
            self.assertAlmostEqual(tip_extension_mm(172,closed=True),2.49039547326933)

    def test_nominal_jaw_width_is_independent_of_planning_shrink(self):
        cfg=json.loads((HERE/'execution_config.batch.example.json').read_text())
        cfg=checked_batch_config(cfg,physical=True)
        cfg['pick_aperture_mm_by_class']={}
        _,workflow=configs()
        objects=[SimulatedWorld.initial_scene()[0]]
        with tempfile.TemporaryDirectory() as d:
            for shrink in (2,3):
                workflow['footprint_shrink_mm']=shrink
                layout=snapshot_plan(objects,d,workflow,render=False)
                batch=build_batch(layout,layout['action_details'][0],cfg,bowl='bowl_1')
                self.assertEqual(batch['nominal_closing_width_mm'],60)
                self.assertEqual(batch['target_aperture_mm'],59)
                close=next(s for s in batch['stages'] if s['phase']=='close-fingers')
                self.assertEqual(close['gripper_pulse'],round(842*59/85))
                place=next(s for s in batch['stages'] if s['phase']=='place')
                self.assertEqual(place['tcp_mm_deg'][:3],[600,-200,60])

    def test_mixed_dimensions_convert_plan_and_use_per_object_heights(self):
        from object_catalog import DIMENSIONS
        from pose_to_plan import base_to_optical, convert
        from standard_gripper_geometry import tip_extension_mm
        cfg=checked_batch_config(json.loads((HERE/'execution_config.batch.example.json').read_text()),physical=True)
        _,workflow=configs()
        records=[]
        for (kind,dims),y in zip(DIMENSIONS.items(),(-.15,0,.15)):
            t=np.eye(4);t[:3,3]=[.435,y,dims[2]/2+.005]
            records.append({'class':kind,'instance_id':'1','score':1,
                            'T_camera_object':(np.linalg.inv(base_to_optical())@t).tolist()})
        objects=convert(records)
        with tempfile.TemporaryDirectory() as d:
            layout=snapshot_plan(objects,d,workflow,render=False)
            self.assertTrue(layout['plan_verified'])
            for obj in objects:
                self.assertEqual(obj['dimensions_m'],DIMENSIONS[obj['kind']])
                candidate=next(a for a in layout['immediately_feasible_actions']
                               if a['kind']=='pick' and a['target']==obj['name'])
                pick=build_batch(layout,candidate,cfg,bowl='bowl_2')
                self.assertAlmostEqual(pick['height_geometry']['table_base_z_estimate_mm'],5)
                self.assertAlmostEqual(pick['target_aperture_mm'],50 if obj['kind']=='protex' else obj['dimensions_m'][1]*1000-1)
                knock=build_batch(layout,dict(target=obj['name'],kind='knock',witness='knock-+x',blockers=[]),cfg)
                expected_top=5+obj['dimensions_m'][2]*1000
                geometry=knock['height_geometry']
                if obj['kind'] in ('protex','beach'):
                    self.assertEqual(geometry['contact_tcp_z_mm'],28)
                    self.assertEqual(geometry['end_tcp_z_mm'],27)
                else:
                    self.assertAlmostEqual(geometry['contact_tcp_z_mm']-tip_extension_mm(172,closed=True),expected_top-2)
                    self.assertAlmostEqual(geometry['end_tcp_z_mm']-tip_extension_mm(172,closed=True),expected_top-3)

    def test_soap_knock_ignores_detected_z_and_preserves_raw_pose(self):
        from object_catalog import DIMENSIONS
        cfg=checked_batch_config(json.loads((HERE/'execution_config.batch.example.json').read_text()),physical=True)
        for kind in ('protex','beach'):
            for center_z in (.012,.018145815,.025):
                obj=copy.deepcopy(SimulatedWorld.initial_scene()[0])
                obj.update(kind=kind,name=kind+'_1',dimensions_m=DIMENSIONS[kind])
                obj['base_center_m'][2]=center_z
                obj['T_base_object'][2][3]=center_z
                layout={'objects':[obj]};original=copy.deepcopy(layout)
                action=build_batch(layout,dict(target=obj['name'],kind='knock',witness='knock--x',blockers=[]),cfg)
                start=next(s for s in action['stages'] if s['phase']=='contact')
                end=next(s for s in action['stages'] if s['phase']=='horizontal-knock')
                self.assertEqual(start['tcp_mm_deg'][2],28)
                self.assertEqual(end['tcp_mm_deg'][2],27)
                self.assertAlmostEqual(end['tcp_mm_deg'][1]-start['tcp_mm_deg'][1],-50)
                self.assertEqual(end['speed_mm_s'],4)
                self.assertEqual(action['height_geometry']['fitted_center_base_z_mm'],15)
                self.assertEqual(action['height_geometry']['fingertip_compensation_applied_mm'],0)
                self.assertEqual(action['source_pose'],obj)
                self.assertEqual(layout,original)

    def test_explicit_knock_height_overrides_nominal_soap_rule(self):
        cfg=checked_batch_config(json.loads((HERE/'execution_config.batch.example.json').read_text()),physical=True)
        cfg['knock']['contact_tcp_z_mm']=31
        obj=SimulatedWorld.initial_scene()[0]
        action=build_batch({'objects':[obj]},dict(target=obj['name'],kind='knock',witness='knock--x',blockers=[]),cfg)
        self.assertEqual(action['height_geometry']['mode'],'explicit_tcp_z')
        self.assertEqual(action['height_geometry']['contact_tcp_z_mm'],31)
        self.assertEqual(action['height_geometry']['end_tcp_z_mm'],30)

    def test_nominal_soap_rule_rejects_unknown_or_toothpaste_classes(self):
        cfg=json.loads((HERE/'execution_config.batch.example.json').read_text())
        for classes in ('protex',['toothpaste'],['bench'],None):
            cfg['knock']['nominal_height_tcp_classes']=classes
            with self.assertRaisesRegex(UnsafePlan,'nominal_height_tcp_classes'):
                checked_batch_config(cfg,physical=True)

    def test_geometry_rejects_incompatible_tool_rotation(self):
        cfg=json.loads((HERE/'execution_config.batch.example.json').read_text())
        cfg['tool_roll_pitch_deg']=[90,0]
        with self.assertRaisesRegex(UnsafePlan,'top-down'):
            checked_batch_config(cfg,physical=True)


class BasePoseOffsetTests(unittest.TestCase):
    def test_base_axes_translation_preserves_geometry_for_all_classes(self):
        from pose_to_plan import apply_base_pose_offset
        from object_catalog import DIMENSIONS
        objects=SimulatedWorld.initial_scene()[:3]
        for obj,(kind,dims) in zip(objects,DIMENSIONS.items()):
            obj.update(name=kind+'_1',kind=kind,dimensions_m=dims)
        original=copy.deepcopy(objects)
        shifted=apply_base_pose_offset(objects,[15,-10,0])
        for old,new in zip(objects,shifted):
            self.assertTrue(np.allclose(np.asarray(new['base_center_m'])-old['base_center_m'],[.015,-.010,0]))
            self.assertTrue(np.allclose(np.asarray(new['T_base_object'])[:3,3],new['base_center_m']))
            self.assertEqual(np.asarray(new['T_base_object'])[:3,:3].tolist(),np.asarray(old['T_base_object'])[:3,:3].tolist())
            self.assertEqual(new['yaw_rad'],old['yaw_rad'])
            self.assertEqual(new['dimensions_m'],old['dimensions_m'])
            self.assertEqual(new['T_base_object_uncorrected'],old['T_base_object'])
        self.assertEqual(objects,original)
        self.assertTrue(np.allclose(np.asarray(shifted[1]['base_center_m'])-shifted[0]['base_center_m'],
                                    np.asarray(objects[1]['base_center_m'])-objects[0]['base_center_m']))

    def test_offset_validation_and_double_application_guard(self):
        from pose_to_plan import apply_base_pose_offset
        objects=SimulatedWorld.initial_scene()
        for offset in ([15,-10], [15,-10,0,0], [15,float('nan'),0], [float('inf'),0,0], None):
            with self.assertRaisesRegex(ValueError,'three finite numbers'):
                apply_base_pose_offset(objects,offset)
        shifted=apply_base_pose_offset(objects,[15,-10,0])
        with self.assertRaisesRegex(ValueError,'already applied'):
            apply_base_pose_offset(shifted,[15,-10,0])

    def test_pick_and_knock_shift_source_waypoints_but_not_drop_points(self):
        from pose_to_plan import apply_base_pose_offset
        cfg=checked_batch_config(json.loads((HERE/'execution_config.batch.example.json').read_text()),physical=True)
        _,workflow=configs();objects=[SimulatedWorld.initial_scene()[0]]
        with tempfile.TemporaryDirectory() as d:
            before=snapshot_plan(objects,Path(d)/'before',workflow,render=False)
            after=snapshot_plan(apply_base_pose_offset(objects,[15,-10,0]),Path(d)/'after',workflow,render=False)
            for kind,witness,bowl in [('pick','pick-y@-0.044','bowl_1'),('knock','knock--x',None)]:
                candidate=dict(target=objects[0]['name'],kind=kind,witness=witness,blockers=[])
                old=build_batch(before,candidate,cfg,bowl=bowl)
                new=build_batch(after,candidate,cfg,bowl=bowl)
                self.assertEqual(new['pose_source'],'detected_pose_with_manual_base_offset')
                self.assertEqual(old['height_geometry'],new['height_geometry'])
                for a,b in zip(old['stages'],new['stages'],strict=True):
                    self.assertEqual(a['speed_mm_s'],b['speed_mm_s'])
                    if a['tcp_mm_deg'] is None:
                        self.assertEqual(a,b)
                    else:
                        delta=np.asarray(b['tcp_mm_deg'])-a['tcp_mm_deg']
                        expected=[0]*6 if kind=='pick' and a['phase'] in ('transfer','place','retreat') else [15,-10,0,0,0,0]
                        self.assertTrue(np.allclose(delta,expected))

    def test_live_observation_applies_offset_after_camera_conversion(self):
        from run_pipeline import LivePerception
        from pose_to_plan import base_to_optical
        obj=SimulatedWorld.initial_scene()[0]
        record={'class':'protex','instance_id':'1','score':1,
                'T_camera_object':(np.linalg.inv(base_to_optical())@np.asarray(obj['T_base_object'])).tolist()}
        # Exercise the real observation entry point with recorded RGB-D and mocked
        # learned-model subprocesses. No camera capture or SDK connection occurs.
        with tempfile.TemporaryDirectory(dir=HERE) as d:
            p=Path(d).resolve();world=LivePerception.__new__(LivePerception)
            world.run_dir=p;world.frames=p/'frames';world.frames.mkdir()
            world.recorded_capture=p/'input.npz'
            np.savez(world.recorded_capture,color=np.zeros((20,20,3),np.uint8))
            world.points=p/'points.json';world.base_pose_offset_mm=[15,-10,0]
            world.specs={'protex_1':{'mesh':'real_robot/unused.obj'}}
            world.config={key:'real_robot/unused' for key in ('sam2_python','checkpoint','library','refiner','scorer')}
            world.config['docker_image']='unused'
            folder=p/'observation'
            def fake_model_run(argv,logfile):
                if '--object-specs' in argv:
                    out=folder/'foundationpose';out.mkdir()
                    (out/'foundationpose_instances.json').write_text(json.dumps({'instances':[record]}))
            world.run=fake_model_run
            with patch('perception_validation.validate_model_poses',return_value=[]):
                observed=world.observe(0,folder)
            raw=json.loads((folder/'observed_objects_uncorrected.json').read_text())
            saved=json.loads((folder/'observed_objects.json').read_text())
            self.assertTrue(np.allclose(raw[0]['base_center_m'],obj['base_center_m']))
            self.assertTrue(np.allclose(np.asarray(observed[0]['base_center_m'])-obj['base_center_m'],[.015,-.010,0]))
            self.assertEqual(saved,json.loads(json.dumps(observed)))


class FingerClearanceTests(unittest.TestCase):
    def test_20mm_thickness_blocks_gap_that_15mm_allowed(self):
        _,workflow=configs()
        objects=SimulatedWorld.initial_scene()[:2]
        for obj,x in zip(objects,(.400,.475)):
            obj['base_center_m']=[x,0.,.015]
            obj['T_base_object'][0][3]=x
            obj['T_base_object'][1][3]=0.
        original=copy.deepcopy(objects)
        with tempfile.TemporaryDirectory() as d:
            workflow['finger_thickness_mm']=15
            old=snapshot_plan(objects,Path(d)/'old',workflow,render=False)
            self.assertEqual(old['action_details'][0]['kind'],'pick')
            workflow['finger_thickness_mm']=20
            new=snapshot_plan(objects,Path(d)/'new',workflow,render=False)
            self.assertEqual(new['action_details'][0]['kind'],'knock')
            self.assertEqual(new['finger_thickness_mm'],20)
            self.assertFalse(any(a['kind']=='pick' for a in new['immediately_feasible_actions']))
            self.assertEqual(old['planning_objects'],new['planning_objects'])
            self.assertEqual(objects,original)

    def test_invalid_finger_thickness_rejected(self):
        _,workflow=configs()
        with tempfile.TemporaryDirectory() as d:
            for thickness in (0,-1,float('nan'),float('inf')):
                workflow['finger_thickness_mm']=thickness
                with self.assertRaisesRegex(UnsafePlan,'finger_thickness_mm'):
                    snapshot_plan([],d,workflow,render=False)


class PhysicalPickClearanceTests(unittest.TestCase):
    @staticmethod
    def scene(gap_mm=30,yaw_deg=0):
        from scipy.spatial.transform import Rotation
        objects=SimulatedWorld.initial_scene()[:2]
        yaw=np.deg2rad(yaw_deg);r=Rotation.from_euler('z',yaw).as_matrix()
        for obj,dy in zip(objects,(0,(60+gap_mm)/1000)):
            center=np.array([.435,0,.015])+r@np.array([0,dy,0])
            t=np.eye(4);t[:3,:3]=r;t[:3,3]=center
            obj.update(base_center_m=center.tolist(),T_base_object=t.tolist(),yaw_rad=yaw)
        return objects

    def test_conservative_planner_requires_knock_for_blocked_open_finger_sweep(self):
        from pick_clearance import pick_clearance
        cfg,w=configs();cfg['pick_overlap_tolerance_mm']=0;objects=self.scene()
        with tempfile.TemporaryDirectory() as d:
            layout=snapshot_plan(objects,d,w,render=False,execution_cfg=cfg)
            self.assertEqual(layout['action_details'][0]['kind'],'knock')
            self.assertFalse(any(c['kind']=='pick' for c in layout['immediately_feasible_actions']))
            candidate=dict(target=objects[0]['name'],kind='pick',witness='pick-y@-0.044',blockers=[])
            check=pick_clearance(layout,candidate,cfg)
            self.assertEqual(check['finger_thickness_mm'],20)
            self.assertEqual(check['open_aperture_mm'],85)
            self.assertTrue(check['blockers'])
            candidate=dict(target=objects[0]['name'],kind='pick',witness='pick-y@-0.044',blockers=[])
            with self.assertRaisesRegex(UnsafePlan,'Pick clearance failed'):
                build_batch(layout,candidate,cfg,bowl='bowl_1')
            # Manipulating planning-only centers must not hide observed obstruction.
            layout['planning_objects'][1]['base_center_m'][1]+=1
            self.assertFalse(pick_clearance(layout,candidate,cfg)['passed'])

    def test_fixed_knock_continues_until_fixed_grasp_has_clearance_at_rotated_yaws(self):
        from fixed_execution import select_fixed_candidate,latest_layout
        cfg,w=configs();cfg['pick_overlap_tolerance_mm']=0
        for yaw in (0,45,90,-30):
            for gap,expected in ((30,'knock'),(32.5,'knock'),(33,'pick')):
                with self.subTest(yaw=yaw,gap=gap):
                    objects=self.scene(gap,yaw);target=objects[0]['name']
                    entry=dict(target=target,kind='knock',
                        knock=dict(target=target,kind='knock',witness='knock--y',blockers=[]),
                        pick=dict(target=target,kind='pick',witness='pick-y@-0.044',blockers=[]))
                    layout=latest_layout(objects,w)
                    chosen,_=select_fixed_candidate(entry,layout,cfg,has_knocked=True)
                    self.assertEqual(chosen['kind'],expected)
                    self.assertEqual(chosen['target'],target)
                    if expected=='pick':
                        self.assertTrue(build_batch(layout,chosen,cfg,bowl='bowl_1')['pick_clearance']['passed'])

    def test_blocked_fixed_pick_stops_without_inventing_a_knock(self):
        from fixed_execution import select_fixed_candidate,latest_layout
        cfg,w=configs();cfg['pick_overlap_tolerance_mm']=0;objects=self.scene();target=objects[0]['name']
        entry=dict(target=target,kind='pick',knock=None,
                   pick=dict(target=target,kind='pick',witness='pick-y@-0.044',blockers=[]))
        with self.assertRaisesRegex(UnsafePlan,'fixed planned Pick is blocked'):
            select_fixed_candidate(entry,latest_layout(objects,w),cfg,has_knocked=False)

    def test_sweep_catches_neighbor_between_closed_and_fully_open_fingers(self):
        from pick_clearance import pick_clearance
        cfg,w=configs();cfg['pick_overlap_tolerance_mm']=0;objects=self.scene()
        # A narrow obstacle lies between the closed and open finger positions.
        objects[1]['dimensions_m']=[.09,.002,.03]
        objects[1]['base_center_m'][1]=.036
        objects[1]['T_base_object'][1][3]=.036
        cfg['gripper_open_mm']=160
        layout=dict(objects=objects,finger_thickness_mm=2)
        candidate=dict(target=objects[0]['name'],kind='pick',witness='pick-y@-0.044',blockers=[])
        self.assertFalse(pick_clearance(layout,candidate,cfg)['passed'])


class GraspAndDropCorrectionTests(unittest.TestCase):
    def test_protex_override_closes_to_495_pulses_and_drops_at_600(self):
        cfg,w=configs();objects=[SimulatedWorld.initial_scene()[0]]
        with tempfile.TemporaryDirectory() as d:
            layout=snapshot_plan(objects,d,w,render=False,execution_cfg=cfg)
            candidate=layout['action_details'][0]
            for bowl,y in (('bowl_1',-200),('bowl_2',200)):
                action=build_batch(layout,candidate,cfg,bowl=bowl)
                self.assertEqual(action['target_aperture_mm'],50)
                self.assertEqual(action['grasp_aperture_mode'],'class_override')
                self.assertEqual(action['aperture_mapping'],'uncalibrated_linear_pulse_estimate')
                self.assertEqual(next(s['gripper_pulse'] for s in action['stages'] if s['phase']=='close-fingers'),495)
                self.assertEqual([s['gripper_pulse'] for s in action['stages'] if s['phase']=='open-fingers'],[842,842])
                for stage in action['stages']:
                    if stage['phase'] in ('transfer','place','retreat'):
                        self.assertEqual(stage['tcp_mm_deg'][:2],[600,y])
                self.assertEqual(action['pick_clearance']['commanded_close_aperture_estimate_mm'],50)

    def test_invalid_aperture_overrides_rejected(self):
        cfg,w=configs()
        for overrides in ({'protex':0},{'protex':float('nan')},{'protex':90},{'unknown':50}):
            with self.subTest(overrides=overrides):
                bad=copy.deepcopy(cfg);bad['pick_aperture_mm_by_class']=overrides
                with self.assertRaises(UnsafePlan):checked_batch_config(bad,physical=True)


class MotionSpeedTests(unittest.TestCase):
    def test_requested_speed_changes_preserve_paths(self):
        cfg=checked_batch_config(json.loads((HERE/'execution_config.batch.example.json').read_text()),physical=True)
        previous=copy.deepcopy(cfg)
        previous.update(motion_speed_scale=1.5,approach_speed_scale=1.5)
        self.assertEqual(cfg['gripper_speed_setting'],1500)
        obj=SimulatedWorld.initial_scene()[0]
        layout={'objects':[obj]}
        for kind,witness,bowl in [('knock','knock--x',None),('pick','pick-y@-0.029','bowl_1')]:
            candidate=dict(target=obj['name'],kind=kind,witness=witness,blockers=[])
            old=build_batch(layout,candidate,previous,bowl=bowl)
            new=build_batch(layout,candidate,cfg,bowl=bowl)
            self.assertEqual(old['height_geometry'],new['height_geometry'])
            for before,after in zip(old['stages'],new['stages'],strict=True):
                self.assertEqual(before['tcp_mm_deg'],after['tcp_mm_deg'])
                self.assertEqual(before['gripper_pulse'],after['gripper_pulse'])
                if after['tcp_mm_deg'] is not None:
                    factor=2.5/1.5 if after['phase'] in ('pre-pick','pre-knock','approach','near-contact','contact','pick') else 1.
                    self.assertAlmostEqual(after['speed_mm_s'],before['speed_mm_s']*factor)

    def test_invalid_speed_scale_rejected(self):
        cfg=json.loads((HERE/'execution_config.batch.example.json').read_text())
        for scale in (0,-1,float('nan'),float('inf')):
            cfg['motion_speed_scale']=scale
            with self.assertRaisesRegex(UnsafePlan,'motion_speed_scale'):
                checked_batch_config(cfg,physical=True)


class PoseValidationTests(unittest.TestCase):
    def test_depth_consistent_pose_passes_and_bad_registration_stops(self):
        from perception_validation import validate_model_poses
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); mask=np.zeros((100,100),bool); mask[21:79,6:94]=True
            depth=np.full((100,100),.985)
            np.savez(p/'capture.npz',depth=depth,depth_scale=1.,K=[1000,1000,50,50])
            np.savez(p/'masks.npz',protex_1=mask)
            t=np.eye(4);t[2,3]=1
            record={'class':'protex','instance_id':'1','T_camera_object':t.tolist()}
            result=validate_model_poses([record],p/'capture.npz',p/'masks.npz',p/'validation.json')
            self.assertTrue(result[0]['passed'])
            record['T_camera_object'][0][3]=.2
            with self.assertRaisesRegex(ValueError,'validation failed'):
                validate_model_poses([record],p/'capture.npz',p/'masks.npz',p/'validation.json')


class FakeArm:
    def __init__(self,cfg,fail_motion=None,fail_enable=False,initial_state=2,stuck_state=None,errors=None,pose_z=300):
        self.cfg=cfg;self.connected=True;self.collision_sensitivity=3;self.tcp_load=[.82,[0,0,48]]
        self.tcp_offset=cfg['tcp_offset_mm_deg'];self.calls=[];self.moves=0;self.motion_commands=[]
        self.pose=[400,0,pose_z,180,0,0];self.ignore_motion=False
        self.joints=[0]*7;self.home_commands=[];self.fail_home=False;self.ignore_home=False
        self.home_pose=[515.95,-.029,553.176,178.54,-60,1.103];self.home_limited=False
        self.fail_motion=fail_motion;self.fail_enable=fail_enable
        self.state=initial_state;self.stuck_state=stuck_state;self.errors=errors or [0,0];self.pose_z=pose_z
    def connect(self):self.calls.append('connect')
    def disconnect(self):self.calls.append('disconnect')
    def get_state(self):self.calls.append('get_state');return 0,self.state
    def get_err_warn_code(self):self.calls.append('get_err_warn_code');return 0,self.errors
    def get_position(self,**kwargs):self.calls.append('get_position');return 0,list(self.pose)
    def set_tcp_offset(self,*args,**kwargs):self.calls.append('tcp_offset');self.state=5;return 0
    def get_inverse_kinematics(self,*args,**kwargs):raise AssertionError('redundant per-waypoint IK call')
    def motion_enable(self,*args):self.calls.append('enable');return 1 if self.fail_enable else 0
    def set_mode(self,*args):self.calls.append('mode');return 0
    def set_state(self,state):
        self.calls.append('stop' if state==4 else 'state')
        self.state=self.stuck_state if state==0 and self.stuck_state is not None else state
        return 0
    def is_joint_limit(self,*args,**kwargs):self.calls.append('is_joint_limit');return 0,self.home_limited
    def get_forward_kinematics(self,*args,**kwargs):self.calls.append('get_forward_kinematics');return 0,list(self.home_pose)
    def get_servo_angle(self,**kwargs):self.calls.append('get_servo_angle');return 0,list(self.joints)
    def set_servo_angle(self,**kwargs):
        self.calls.append('home_motion');self.home_commands.append(kwargs)
        if self.fail_home:return 9
        if not self.ignore_home:self.pose=list(self.home_pose);self.joints=list(kwargs['angle'])
        return 0
    def set_gripper_mode(self,*args):return 0
    def set_gripper_enable(self,*args):return 0
    def set_gripper_speed(self,*args):return 0
    def set_gripper_position(self,*args,**kwargs):self.calls.append('gripper_motion');return 0
    def set_position(self,**kwargs):
        self.moves+=1;self.calls.append('arm_motion');self.motion_commands.append(kwargs)
        if self.moves==self.fail_motion:return 9
        if not self.ignore_motion:self.pose=[kwargs[k] for k in ('x','y','z','roll','pitch','yaw')]
        return 0


class SDKTests(unittest.TestCase):
    def batch(self,d):
        cfg,w=configs(physical=True)
        layout=snapshot_plan(SimulatedWorld.initial_scene(),d,w,render=False)
        action=build_batch(layout,layout['action_details'][0],cfg)
        return cfg,action
    def test_no_explicit_feedback_queries_between_stages(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg)
            XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            start=arm.calls.index('gripper_motion')
            self.assertFalse(any(c.startswith('get_') or c=='IK' for c in arm.calls[start:]))
            self.assertEqual(arm.moves,len([s for s in action['stages'] if s['tcp_mm_deg'] is not None]))
    def test_sdk_error_stops_remaining_stages(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg,fail_motion=2)
            with self.assertRaisesRegex(UnsafePlan,'SDK code 9'):
                XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            self.assertEqual(arm.moves,2);self.assertIn('stop',arm.calls)
    def test_initialization_failure_is_not_eager(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg,fail_enable=True)
            with self.assertRaisesRegex(UnsafePlan,'SDK code 1'):
                XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            self.assertNotIn('mode',arm.calls);self.assertEqual(arm.moves,0)
    def test_stopped_states_reinitialize_before_batch_without_resuming_commands(self):
        for initial_state in (4,5):
            with self.subTest(initial_state=initial_state), tempfile.TemporaryDirectory() as d:
                cfg,action=self.batch(d);arm=FakeArm(cfg,initial_state=initial_state)
                XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
                self.assertLess(arm.calls.index('stop'),arm.calls.index('tcp_offset'))
                self.assertLess(arm.calls.index('tcp_offset'),arm.calls.index('enable'))
                self.assertLess(arm.calls.index('enable'),arm.calls.index('mode'))
                self.assertLess(arm.calls.index('mode'),arm.calls.index('state'))
                self.assertLess(arm.calls.index('state'),arm.calls.index('gripper_motion'))
                self.assertEqual(arm.calls.count('enable'),1)
                self.assertEqual(arm.calls.count('get_position'),2)
                events=[json.loads(line) for line in (Path(d)/'exec/execution.jsonl').read_text().splitlines()]
                self.assertTrue(any(e['event']=='startup_recovery' and e['initial_state']==initial_state for e in events))

    def test_moving_or_paused_robot_is_not_taken_over(self):
        for initial_state in (1,3,6):
            with self.subTest(initial_state=initial_state), tempfile.TemporaryDirectory() as d:
                cfg,action=self.batch(d);arm=FakeArm(cfg,initial_state=initial_state)
                with self.assertRaisesRegex(UnsafePlan,'refusing to take over'):
                    XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
                self.assertNotIn('enable',arm.calls)
                self.assertNotIn('stop',arm.calls)
                self.assertEqual(arm.moves,0)

    def test_faults_are_not_cleared_by_startup_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg,initial_state=5,errors=[22,0])
            with self.assertRaisesRegex(UnsafePlan,'resolve the fault'):
                XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            self.assertNotIn('enable',arm.calls)
            self.assertNotIn('stop',arm.calls)
            self.assertEqual(arm.moves,0)

    def test_recovery_does_not_retry_enable_when_controller_stays_stopped(self):
        with tempfile.TemporaryDirectory() as d, patch('batch_executor.time.sleep'):
            cfg,action=self.batch(d);arm=FakeArm(cfg,initial_state=5,stuck_state=5)
            with self.assertRaisesRegex(UnsafePlan,'did not become ready'):
                XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            self.assertEqual(arm.calls.count('enable'),1)
            self.assertEqual(arm.calls.count('state'),1)
            self.assertNotIn('gripper_motion',arm.calls)
            self.assertEqual(arm.moves,0)
            self.assertEqual(arm.state,4)

    def test_stopped_near_table_does_not_automatically_reposition(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg,initial_state=5,pose_z=15)
            with self.assertRaisesRegex(UnsafePlan,'initial TCP Z is 15 mm'):
                XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            self.assertNotIn('enable',arm.calls)
            self.assertNotIn('gripper_motion',arm.calls)
            self.assertEqual(arm.moves,0)

    def test_joint_home_runs_even_when_initial_tcp_above_transit(self):
        for z in (100.001,270.881592,300,550):
            with self.subTest(z=z), tempfile.TemporaryDirectory() as d:
                cfg,action=self.batch(d);arm=FakeArm(cfg,initial_state=5,pose_z=z)
                with patch('builtins.print') as announce:
                    XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
                self.assertEqual(len(arm.home_commands),1)
                home=arm.home_commands[0]
                self.assertEqual(home['angle'],[0,-30,0,60,0,30,0])
                self.assertEqual(home['speed'],20);self.assertEqual(home['mvacc'],50)
                self.assertTrue(home['wait']);self.assertFalse(home['is_radian'])
                self.assertEqual(arm.calls.count('get_position'),2)
                self.assertLess(arm.calls.index('home_motion'),arm.calls.index('gripper_motion'))
                self.assertEqual(arm.moves,sum(s['tcp_mm_deg'] is not None for s in action['stages']))
                self.assertTrue(any('startup joint home' in call.args[0] for call in announce.call_args_list))
                start=arm.calls.index('gripper_motion')
                self.assertFalse(any(c.startswith('get_') for c in arm.calls[start:]))
                events=[json.loads(line)['event'] for line in (Path(d)/'exec/execution.jsonl').read_text().splitlines()]
                self.assertLess(events.index('startup_home_complete'),events.index('stage_start'))

    def test_joint_home_rejects_threshold_and_invalid_pose(self):
        for z in (100,99.9,0,float('nan'),float('inf')):
            with self.subTest(z=z), tempfile.TemporaryDirectory() as d:
                cfg,action=self.batch(d);arm=FakeArm(cfg,pose_z=z)
                with self.assertRaises(UnsafePlan):
                    XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
                self.assertNotIn('enable',arm.calls);self.assertEqual(arm.moves,0)
                self.assertFalse(arm.home_commands);self.assertNotIn('gripper_motion',arm.calls)

    def test_joint_home_checks_endpoint_before_enable(self):
        for limited,z in ((True,553),(False,300),(False,float('nan'))):
            with self.subTest(limited=limited,z=z), tempfile.TemporaryDirectory() as d:
                cfg,action=self.batch(d);arm=FakeArm(cfg)
                arm.home_limited=limited;arm.home_pose[2]=z
                with self.assertRaises(UnsafePlan):
                    XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
                self.assertNotIn('enable',arm.calls);self.assertFalse(arm.home_commands)

    def test_joint_home_failure_never_starts_action(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg,pose_z=270);arm.fail_home=True
            with self.assertRaisesRegex(UnsafePlan,'startup joint home: SDK code 9'):
                XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            self.assertEqual(len(arm.home_commands),1);self.assertEqual(arm.state,4)
            self.assertEqual(arm.moves,0);self.assertNotIn('gripper_motion',arm.calls)
            self.assertEqual(arm.calls.count('enable'),1)

    def test_joint_home_verifies_arrival_before_action(self):
        for z in (270,550):
            with self.subTest(z=z), tempfile.TemporaryDirectory() as d:
                cfg,action=self.batch(d);arm=FakeArm(cfg,pose_z=z);arm.ignore_home=True
                with self.assertRaisesRegex(UnsafePlan,'did not reach'):
                    XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
                self.assertEqual(arm.moves,0);self.assertEqual(arm.state,4)
                self.assertNotIn('gripper_motion',arm.calls)

    def test_joint_home_fault_never_starts_action(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg,pose_z=270)
            with patch.object(arm,'get_err_warn_code',side_effect=[(0,[0,0]),(0,[0,0]),(0,[22,0])]):
                with self.assertRaisesRegex(UnsafePlan,'not ready after startup home'):
                    XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm).execute(action,Path(d)/'exec')
            self.assertEqual(arm.moves,0);self.assertEqual(arm.state,4)
            self.assertNotIn('gripper_motion',arm.calls)

    def test_joint_home_only_once_across_batches(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,action=self.batch(d);arm=FakeArm(cfg,pose_z=270)
            executor=XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm)
            executor.execute(action,Path(d)/'first')
            executor.execute(action,Path(d)/'second')
            self.assertEqual(len(arm.home_commands),1)
            self.assertEqual(arm.calls.count('get_forward_kinematics'),1)
            self.assertEqual(arm.calls.count('get_position'),3)
            arm.pose[2]=270
            with self.assertRaisesRegex(UnsafePlan,'mid-run homing'):
                executor.execute(action,Path(d)/'third')
            self.assertEqual(len(arm.home_commands),1)

    def test_joint_home_config_requires_seven_finite_angles_and_positive_speeds(self):
        cfg,_=configs()
        for field,value in (('joint_angles_deg',[0]*6),('joint_angles_deg',[float('nan')]*7),
                            ('speed_deg_s',0),('acceleration_deg_s2',-1)):
            with self.subTest(field=field,value=value):
                bad=copy.deepcopy(cfg);bad['startup_home'][field]=value
                with self.assertRaises(UnsafePlan):checked_batch_config(bad,physical=True)

    def test_manual_script_import_has_no_hardware_side_effects(self):
        spec=importlib.util.spec_from_file_location('manual_example',HERE/'test_xarm.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(callable(module.main));self.assertFalse(hasattr(module,'arm'))


if __name__=='__main__':unittest.main()
