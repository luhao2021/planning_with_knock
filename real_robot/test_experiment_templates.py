"""Focused mixed-inventory regressions; cameras and robot calls are never used."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from batch_executor import checked_batch_config, apply_execution_overrides
from run_statistics import write_run_summary
from object_catalog import DIMENSIONS
from pipeline_core import run_loop
from run_pipeline import LivePerception, SimulatedWorld
import prepare_experiment_scene as setup

HERE=Path(__file__).parent


class FocusTemplateTests(unittest.TestCase):
    def template(self, name):
        return json.loads((HERE/'experiment_design/configs'/f'{name}.json').read_text())

    def test_focus_inventories_and_per_object_budget(self):
        for name,protex,beach in [('S07',2,2),('S12',3,2),('S13',4,2)]:
            cfg=self.template(name)
            expected={f'protex_{i}' for i in range(1,protex+1)}|{f'beach_{i}' for i in range(1,beach+1)}
            self.assertEqual(set(cfg['perception']['objects']),expected)
            self.assertFalse(cfg['perception']['automatic_scene']['enabled'])
            for key,spec in cfg['perception']['objects'].items():
                self.assertEqual(key,f"{spec['class']}_{spec['instance_id']}")
                size='x'.join(f'{x*1000:g}' for x in DIMENSIONS[spec['class']])
                self.assertEqual(spec['mesh'],f"real_robot/models/{spec['class']}_{size}mm.obj")
            if name in ('S12','S13'):
                self.assertGreaterEqual(cfg['workflow']['max_actions'],(protex+beach)*(cfg['workflow']['max_knock_attempts']+cfg['workflow']['max_pick_attempts']))

    def test_five_box_setup_resets_once_and_loads_all_named_seeds(self):
        self.check_setup('S12')

    def test_six_box_setup_resets_once_and_loads_all_named_seeds(self):
        self.check_setup('S13')

    def check_setup(self, template_name):
        cfg=self.template(template_name)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            asset=root/'mock_asset';asset.write_text('test asset')
            for spec in cfg['perception']['objects'].values():spec['mesh']=str(asset)
            for key in ('python','sam2_python','checkpoint','library','refiner','scorer'):
                cfg['perception'][key]=str(asset)
            template=root/f'{template_name}.json';template.write_text(json.dumps(cfg))
            names=list(cfg['perception']['objects'])
            points={name:[50+i*80,200] for i,name in enumerate(names)}
            def fake_capture(path, **kwargs):
                np.savez(path,color=np.zeros((480,640,3),dtype=np.uint8))
            with patch.object(setup,'ROOT',root), patch('sys.argv',['setup','--config',str(template)]), \
                 patch('capture_rgbd.capture',side_effect=fake_capture) as camera, \
                 patch.object(setup,'select_points',return_value=points) as select:
                setup.main()
            camera.assert_called_once()
            self.assertEqual(camera.call_args.kwargs,{'reset_before':True})
            self.assertEqual(select.call_args.args[1],names)
            prepared=list((root/'real_robot/experiments/scene_setup').glob('*/pipeline_config.json'))
            self.assertEqual(len(prepared),1)
            prepared_config=json.loads(prepared[0].read_text())
            self.assertEqual(prepared_config['execution_overrides'],cfg['execution_overrides'])
            pc=prepared_config['perception']
            pc['points']=str(root/pc['points'])
            self.assertEqual(json.loads(Path(pc['points']).read_text()),points)
            perception=LivePerception(pc,root/'perception',workflow=cfg['workflow'])
            self.assertEqual(perception.initial_names,set(names))

    def test_four_box_fixed_plan_completes_in_synthetic_loop(self):
        self.check_complete_loop('S07',4)

    def test_five_box_fixed_plan_completes_in_synthetic_loop(self):
        self.check_complete_loop('S12',5)

    def test_six_box_fixed_plan_completes_in_synthetic_loop(self):
        self.check_complete_loop('S13',6)

    def check_complete_loop(self, template_name, count):
        reference=json.loads((HERE/'experiment_design/reference_scenes.json').read_text())
        scene=next(s for s in reference['scenes'] if s['id']==template_name)
        config=self.template(template_name)
        cfg=checked_batch_config(apply_execution_overrides(
            json.loads((HERE/'execution_config.batch.example.json').read_text()),config),physical=False)
        objects=[]
        for box in scene['boxes']:
            kind=box['kind'];dims=DIMENSIONS[kind]
            transform=np.eye(4)
            transform[:3,:3]=Rotation.from_euler('z',box['yaw_deg'],degrees=True).as_matrix()
            transform[:3,3]=[box['x_mm']/1000,box['y_mm']/1000,dims[2]/2+.005]
            objects.append(dict(name=box['name'],kind=kind,instance_id=box['name'].split('_')[-1],
                dimensions_m=dims,score=1,base_center_m=transform[:3,3].tolist(),
                yaw_rad=float(np.deg2rad(box['yaw_deg'])),T_base_object=transform.tolist()))
        world=SimulatedWorld(cfg,objects=objects)
        with tempfile.TemporaryDirectory() as d:
            start=time.monotonic();start_unix=time.time()
            (Path(d)/'run_metadata.json').write_text(json.dumps({'mode':'simulation'}))
            (Path(d)/'pipeline_config.json').write_text(json.dumps(config))
            state=run_loop(world,world,cfg,config['workflow'],Path(d),render=False)
            self.assertEqual(state['status'],'complete')
            self.assertEqual(set(state['completed']),{o['name'] for o in objects})
            plan=json.loads((Path(d)/state['fixed_plan_file']).read_text())['steps']
            self.assertEqual(len(plan),count)
            picks=[]
            for path in (Path(d)/'actions').glob('*/action.json'):
                action=json.loads(path.read_text())
                if action['kind']!='pick':continue
                picks.append(action)
                protex=action['source_pose']['kind']=='protex'
                self.assertEqual(action['bowl'],'bowl_1' if protex else 'bowl_2')
                place=next(stage['tcp_mm_deg'] for stage in action['stages'] if stage['phase']=='place')
                expected_z={'S07':60,'S12':80,'S13':110}[template_name] if protex else 60
                self.assertEqual(place[:3],[600,-200 if protex else 200,expected_z])
            self.assertEqual(len(picks),count)
            summary=write_run_summary(Path(d),started_unix=start_unix,
                                      elapsed_seconds=time.monotonic()-start,status=state['status'])
            self.assertEqual(summary['planned']['picks'],count)
            self.assertEqual(summary['planned']['total_actions'],count+sum(p['kind']=='knock' for p in plan))
            self.assertEqual(summary['actual']['completed_pick_batches'],count)
            self.assertEqual(summary['actual']['completed_batches'],state['actions_executed'])
            self.assertEqual(summary['actual']['total_attempts'],state['actions_executed'])
            self.assertEqual(summary['actual']['knock_attempts'],sum(v for k,v in state['attempts'].items() if k.endswith(':knock')))
            self.assertEqual(summary['actual']['physical_batches_with_action_stage_started'],0)
            self.assertEqual(summary['timing']['phases']['planning']['calls'],1)
            self.assertEqual(summary['timing']['phases']['observation']['calls'],state['actions_executed'])
            self.assertTrue((Path(d)/'summary.md').exists())
            self.assertEqual(sum(e[0]=='observe' for e in world.events),state['actions_executed'])


if __name__=='__main__':
    unittest.main()
