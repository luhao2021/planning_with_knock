"""Statistics include preview, interrupted motion and failures before planning."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from batch_executor import apply_execution_overrides, checked_batch_config, placement_bowl
from pipeline_core import write_json
from run_statistics import write_run_summary, timed_call, sdk_stages
import run_pipeline

HERE=Path(__file__).parent


class RunStatisticsTests(unittest.TestCase):
    def test_failed_call_records_elapsed_time_and_preserves_exception(self):
        with tempfile.TemporaryDirectory() as d:
            def fail():raise RuntimeError('detection failed')
            with patch('run_statistics.time.monotonic',side_effect=[10.,12.5]),self.assertRaisesRegex(RuntimeError,'detection failed'):
                timed_call(d,'observation',fail)
            result=write_run_summary(Path(d),started_unix=1000,elapsed_seconds=3,status='stopped')
            self.assertIsNone(result['planned']['total_actions'])
            self.assertEqual(result['actual']['total_attempts'],0)
            self.assertEqual(result['timing']['phases']['observation'],dict(calls=1,failed_calls=1,elapsed_seconds=2.5))
            self.assertAlmostEqual(result['timing']['setup_and_other_seconds'],.5)

    def test_sdk_stage_duration_includes_aborted_slide_and_repeated_open(self):
        events=[]
        for at,event,kwargs in [(0,'startup_home_start',{}),(2,'startup_home_complete',{}),
            (3,'stage_start',dict(stage={'phase':'open-fingers'},stage_index=1)),
            (4,'stage_complete',dict(phase='open-fingers',stage_index=1)),
            (5,'stage_start',dict(stage={'phase':'open-fingers'},stage_index=8)),
            (7,'stage_complete',dict(phase='open-fingers',stage_index=8)),
            (8,'stage_start',dict(stage={'phase':'horizontal-knock'},stage_index=4)),
            (11,'aborted',dict(reason='SDK code -9'))]:
            events.append(dict(time=100-at,elapsed_seconds=at,event=event,**kwargs))
        stages=sdk_stages(events)
        self.assertEqual([s['elapsed_seconds'] for s in stages],[2,1,2,3])
        self.assertEqual([s['stage_index'] for s in stages],[None,1,8,4])
        self.assertEqual(stages[-1]['status'],'failed')

    def test_preview_waypoints_are_not_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            write_json(folder/'state.json',dict(status='preview',attempts={},completed=[]))
            write_json(folder/'actions/0000/action.json',dict(kind='pick',target='protex_1',
                source_pose={'kind':'protex'},stages=[]))
            result=write_run_summary(folder,started_unix=1000,elapsed_seconds=1,status='preview')
            self.assertEqual(result['actual']['total_attempts'],0)
            self.assertFalse(result['actions'][0]['attempted'])
            self.assertEqual(result['actual']['completed_batches'],0)

    def test_failed_sdk_startup_is_attempt_not_slide_or_completed_batch(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            write_json(folder/'run_metadata.json',{'mode':'physical'})
            write_json(folder/'state.json',dict(status='stopped',attempts={'protex_1:knock':1},completed=[]))
            action_dir=folder/'actions/0000'
            write_json(action_dir/'action.json',dict(kind='knock',target='protex_1',source_pose={'kind':'protex'},stages=[]))
            (action_dir/'execution.jsonl').write_text(json.dumps(dict(time=100,event='aborted',reason='not ready'))+'\n')
            result=write_run_summary(folder,started_unix=1000,elapsed_seconds=1,status='stopped')
            self.assertEqual(result['actual']['knock_attempts'],1)
            self.assertEqual(result['actual']['knock_strokes_started'],0)
            self.assertEqual(result['actual']['physical_batches_with_action_stage_started'],0)
            self.assertEqual(result['actual']['completed_batches'],0)

    def test_cli_config_error_still_writes_summary_without_hardware(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            config=folder/'config.json'
            write_json(config,dict(execution_config=str(folder/'missing.json'),output_root=str(folder/'runs')))
            with patch('sys.argv',['run_pipeline','--config',str(config),'--check-config']),self.assertRaises(SystemExit):
                run_pipeline.main()
            summaries=list((folder/'runs').glob('*/summary.json'))
            self.assertEqual(len(summaries),1)
            result=json.loads(summaries[0].read_text())
            self.assertEqual(result['status'],'stopped')
            self.assertIn('missing.json',result['reason'])
            self.assertIsNone(result['planned']['total_actions'])

    def test_template_policy_does_not_change_shared_motion_config(self):
        base=json.loads((HERE/'execution_config.batch.example.json').read_text())
        for scene,z in [('S07',60),('S12',80),('S13',110)]:
            template=json.loads((HERE/'experiment_design/configs'/f'{scene}.json').read_text())
            merged=checked_batch_config(apply_execution_overrides(base,template))
            self.assertEqual(merged['release_tcp_z_mm_by_class'],{'protex':z,'beach':60})
            self.assertEqual(merged['knock'],checked_batch_config(base)['knock'])
            for n in range(6):
                self.assertEqual(placement_bowl(merged,{'name':'protex_1','kind':'protex'},n),'bowl_1')
                self.assertEqual(placement_bowl(merged,{'name':'beach_1','kind':'beach'},n),'bowl_2')
        self.assertEqual(apply_execution_overrides(base,{}),base)
        self.assertNotIn('release_tcp_z_mm_by_class',base)
        self.assertEqual(placement_bowl(base,{'name':'protex_1','kind':'protex'},1),'bowl_2')


if __name__=='__main__':unittest.main()
