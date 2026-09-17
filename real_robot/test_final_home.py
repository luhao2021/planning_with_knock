"""Final homing is a single announced joint move after complete plan execution."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from batch_executor import XArmBatchExecutor
from pipeline_core import run_loop
from run_pipeline import SimulatedWorld
from run_statistics import write_run_summary
from test_pipeline import configs, FakeArm
from xarm_executor import UnsafePlan


class FinalHomeTests(unittest.TestCase):
    def executor(self,**kwargs):
        cfg,_=configs(physical=True)
        arm=FakeArm(cfg,**kwargs)
        executor=XArmBatchExecutor(cfg,api_factory=lambda *a,**k:arm)
        executor._startup_home_done=True
        return executor,arm

    def test_final_home_announced_before_joint_motion_without_rearming_or_gripper_commands(self):
        executor,arm=self.executor()
        original=arm.set_servo_angle
        with tempfile.TemporaryDirectory() as d,patch('builtins.print') as announce:
            def joint_move(**kwargs):
                self.assertTrue(any('entire plan completed; return to joint home' in call.args[0]
                                    for call in announce.call_args_list))
                return original(**kwargs)
            arm.set_servo_angle=joint_move
            executor.return_home(Path(d))
            self.assertEqual(len(arm.home_commands),1)
            command=arm.home_commands[0]
            self.assertEqual(command['angle'],[0,-30,0,60,0,30,0])
            self.assertEqual(command['speed'],20)
            self.assertTrue(command['wait'])
            self.assertEqual(command['timeout'],180)
            self.assertEqual(arm.moves,0)
            for operation in ('enable','mode','state','stop','tcp_offset','gripper_motion'):
                self.assertNotIn(operation,arm.calls)
            events=[json.loads(line) for line in (Path(d)/'execution.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['event'],'final_home_complete')
            self.assertEqual(arm.calls[-1],'disconnect')

    def test_stopped_faulted_or_low_robot_does_not_return_home(self):
        for kwargs in ({'initial_state':4},{'initial_state':5},{'initial_state':3},
                       {'errors':[35,0]},{'pose_z':100},{'pose_z':270},{'pose_z':float('nan')}):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as d:
                executor,arm=self.executor(**kwargs)
                with self.assertRaises(UnsafePlan):executor.return_home(Path(d))
                self.assertFalse(arm.home_commands)
                self.assertNotIn('enable',arm.calls)
                self.assertNotIn('state',arm.calls)

    def test_joint_failure_or_unreached_home_stops_without_retry(self):
        for fault in ('fail_home','ignore_home','home_limited'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                executor,arm=self.executor();setattr(arm,fault,True)
                with self.assertRaises(UnsafePlan):executor.return_home(Path(d))
                self.assertEqual(len(arm.home_commands),0 if fault=='home_limited' else 1)
                if fault!='home_limited':self.assertIn('stop',arm.calls)
                self.assertEqual(arm.calls[-1],'disconnect')

    def test_uninitialized_executor_cannot_issue_final_home(self):
        executor,arm=self.executor();executor._startup_home_done=False
        with tempfile.TemporaryDirectory() as d,self.assertRaises(UnsafePlan):executor.return_home(Path(d))
        self.assertEqual(arm.calls,[])

    def test_final_home_failure_keeps_removal_counts_and_records_failed_phase(self):
        cfg,w=configs();world=SimulatedWorld(cfg)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);start=time.monotonic();unix=time.time()
            with patch.object(world,'return_home',side_effect=UnsafePlan('final home failed')) as home:
                with self.assertRaisesRegex(UnsafePlan,'final home failed'):
                    run_loop(world,world,cfg,w,folder,render=False)
            home.assert_called_once_with(folder/'final_home')
            state=json.loads((folder/'state.json').read_text())
            self.assertEqual(state['status'],'stopped')
            self.assertTrue(state['plan_completed'])
            self.assertEqual(set(state['completed']),world.initial_names)
            self.assertEqual(state['final_home']['status'],'failed')
            summary=write_run_summary(folder,started_unix=unix,elapsed_seconds=time.monotonic()-start,status='stopped')
            self.assertEqual(summary['final_home']['status'],'failed')
            self.assertEqual(summary['actual']['objects_removed_from_source'],4)
            self.assertEqual(summary['timing']['phases']['final_home']['failed_calls'],1)

    def test_preview_single_action_and_failed_plan_do_not_home(self):
        cfg,w=configs()
        for mode in ('preview','single_action','failed_plan'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                world=SimulatedWorld(cfg,knock_scale=0)
                with patch.object(world,'return_home') as home:
                    if mode=='failed_plan':
                        limited=dict(w,max_knock_attempts=1)
                        with self.assertRaises(UnsafePlan):run_loop(world,world,cfg,limited,Path(d),render=False)
                    else:run_loop(world,world,cfg,w,Path(d),render=False,**{mode:True})
                home.assert_not_called()


if __name__=='__main__':unittest.main()
