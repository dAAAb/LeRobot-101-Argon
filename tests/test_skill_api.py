"""Natural-language command API, fake planner/runner and one queued worker.

Never enter application lifespan or connect camera/motors/network.
"""
import copy
import importlib
import threading
import time
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class SkillAPITests(unittest.TestCase):
    def setUp(self):
        self.app = importlib.import_module('server.app')
        self.frame = {'id':'api-test-observation', 'joints':dict.fromkeys(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'),2047), 'source': 'wrist', 'path': 'private-wrist-image.jpg',
                      'camera_generation': 'current-wrist', 'capture_monotonic': time.monotonic()}
        self.plan = {'status': 'ready', 'summary': '先看一下。', 'unsupported_reason': '',
                     'steps': [{'skill': 'observe', 'direction': 'none', 'distance_mm': 0.,
                                'target': '', 'label': '看看眼前'}]}
        self.robot = SimpleNamespace(stop=threading.Event(), pwm_owner=None, connect=Mock())
        self.camera = SimpleNamespace(fresh=Mock(return_value=True), error='', generation='current-wrist')
        self.grasp = SimpleNamespace(capture=Mock(side_effect=lambda: copy.deepcopy(self.frame)))
        self.brain = object()
        self.tactician = SimpleNamespace(status=Mock(return_value={'provider': 'jev', 'enabled': False}))
        from server.world_state import WorldStateMemory
        self.memory = WorldStateMemory(self.app.spatial.fk)
        self.queue = []
        for name, value in (('robot', self.robot), ('camera', self.camera), ('grasp', self.grasp), ('brain', self.brain),
                            ('tactician', self.tactician), ('world_memory',self.memory), ('active', None), ('jobs', {}),
                            ('snapshot_owner', threading.local()),
                            ('executor', SimpleNamespace(submit=self.queue.append))):
            guard = patch.object(self.app, name, value)
            guard.start(); self.addCleanup(guard.stop)
        self.planner = Mock(side_effect=lambda *args, **kwargs: copy.deepcopy(self.plan))
        guard = patch('server.skill_plan.plan_skills', self.planner)
        guard.start(); self.addCleanup(guard.stop)
        self.runner = SimpleNamespace(execute=Mock(side_effect=self.execute))
        self.runner_factory = Mock(return_value=self.runner)
        guard = patch('server.skill_runtime.SkillRunner', self.runner_factory)
        guard.start(); self.addCleanup(guard.stop)
        self.client = TestClient(self.app.app)
        self.addCleanup(self.client.close)
        self.headers = {'x-lerobot-token': self.app.TOKEN}

    def execute(self, plan, report, cancel, publish):
        result = {'plan': plan, 'completed': True, 'steps': [{'skill': 'observe', 'completed': True}],
                  'status': 'completed'}
        publish(result)
        return result

    def post(self, data=None, **kwargs):
        return self.client.post('/api/command', headers=self.headers,
                                json={'instruction': '看一下眼前的物件'} if data is None else data, **kwargs)

    def finish(self, accepted):
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(len(self.queue), 1)
        self.queue.pop(0)()
        result = self.client.get('/api/jobs/' + accepted.json()['id'])
        self.assertEqual(result.status_code, 200)
        self.assertFalse(hasattr(self.app.snapshot_owner, 'job'))
        return result.json()

    def test_instruction_only_request_plans_and_runs_in_single_job(self):
        with patch.dict(self.app.CONFIG, {'experimental_pick_enabled': False}):
            accepted = self.post()
            self.assertEqual(accepted.status_code, 200)
            self.assertEqual(accepted.json()['kind'], 'command')
            self.assertEqual(accepted.json()['status'], 'running')
            self.grasp.capture.assert_not_called()
            self.planner.assert_not_called()
            done = self.finish(accepted)
        self.grasp.capture.assert_called_once_with()
        self.planner.assert_called_once()
        args = self.planner.call_args.args
        self.assertIs(args[0], self.brain)
        self.assertEqual(args[1], '看一下眼前的物件')
        self.assertEqual(args[2], self.frame)
        self.assertIs(args[3], self.app.jobs[done['id']]['cancel'])
        self.runner_factory.assert_called_once_with(self.grasp, self.brain, pick_enabled=False, tactician=self.tactician)
        self.runner.execute.assert_called_once()
        self.assertEqual(done['status'], 'completed')
        self.assertTrue(done['command_result']['completed'])
        self.assertNotIn('path', str(done))
        self.assertNotIn('cancel', done)

    def test_missing_or_wrong_auth_and_foreign_origin_cannot_queue(self):
        for headers in ({}, {'x-lerobot-token': 'invalid'},
                        {**self.headers, 'origin': 'https://other.example'}):
            response = self.client.post('/api/command', headers=headers, json={'instruction': '揮手'})
            self.assertEqual(response.status_code, 403)
        self.assertEqual(self.queue, [])
        self.grasp.capture.assert_not_called()

    def test_arbitrary_plan_motor_arguments_and_invalid_text_are_rejected(self):
        for data in ({}, {'instruction': ''}, {'instruction': '  '}, {'instruction': 'x' * 501},
                     {'instruction': 25}, {'instruction': ['揮手']},
                     {'instruction': '揮手', 'steps': self.plan['steps']},
                     {'instruction': '抬高', 'joint_targets': {'elbow_flex': 1234}},
                     {'instruction': '抬高', 'clear_path_verified': True},
                     {'instruction': '抬高', 'source': 'observer'}):
            with self.subTest(data=data):
                response = self.post(data)
                self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.queue, [])
        self.planner.assert_not_called()

    def test_second_command_and_other_motion_share_single_owner(self):
        first = self.post()
        self.assertEqual(first.status_code, 200)
        second = self.post({'instruction': '揮手'})
        self.assertEqual(second.status_code, 409)
        wave = self.client.post('/api/object-action', headers=self.headers, json={'kind': 'wave'})
        self.assertEqual(wave.status_code, 409)
        self.assertEqual(len(self.queue), 1)
        self.grasp.capture.assert_not_called()

    def test_stopped_arm_cannot_begin_a_new_command(self):
        self.robot.stop.set()
        self.assertEqual(self.post().status_code, 409)
        self.assertEqual(self.queue, [])
        self.planner.assert_not_called()

    def test_owned_jaw_can_reach_runner_for_release_without_api_rearming(self):
        self.robot.pwm_owner = object()
        self.plan['steps'] = [{'skill': 'release', 'direction': 'none', 'distance_mm': 0.,
                               'target': '', 'label': '放開'}]
        owner = self.robot.pwm_owner
        done = self.finish(self.post({'instruction': '放開'}))
        self.assertEqual(done['status'], 'completed')
        self.assertIs(self.robot.pwm_owner, owner)
        self.assertEqual(self.runner.execute.call_args.args[0]['steps'][0]['skill'], 'release')

    def test_camera_or_planner_failure_never_reaches_execution(self):
        self.grasp.capture.side_effect = RuntimeError('腕部畫面中斷')
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'failed')
        self.assertIn('腕部畫面中斷', done['message'])
        self.planner.assert_not_called()
        self.runner.execute.assert_not_called()
        self.grasp.capture.side_effect = lambda: copy.deepcopy(self.frame)
        self.planner.side_effect = RuntimeError('規劃沒有完成')
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'failed')
        self.assertIn('規劃沒有完成', done['message'])
        self.runner.execute.assert_not_called()

    def test_dark_camera_rejects_command_before_robot_capture_or_stop(self):
        self.camera.fresh.return_value = False
        self.camera.error = '腕部相機畫面太暗，請確認照明與鏡頭是否被遮住'
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'failed')
        self.assertEqual(done['message'], self.camera.error)
        self.grasp.capture.assert_not_called()
        self.planner.assert_not_called()
        self.runner.execute.assert_not_called()
        self.assertFalse(self.robot.stop.is_set())
        self.robot.connect.assert_not_called()

    def test_precancelled_command_does_not_connect_or_capture(self):
        accepted=self.post();self.app.jobs[accepted.json()['id']]['cancel'].set()
        done=self.finish(accepted)
        self.assertEqual(done['status'],'cancelled')
        self.robot.connect.assert_not_called();self.grasp.capture.assert_not_called()
        self.planner.assert_not_called();self.runner.execute.assert_not_called()

    def test_cancellation_during_connect_prevents_planning_capture(self):
        accepted=self.post()
        self.robot.connect.side_effect=self.app.jobs[accepted.json()['id']]['cancel'].set
        done=self.finish(accepted)
        self.assertEqual(done['status'],'cancelled')
        self.robot.connect.assert_called_once();self.grasp.capture.assert_not_called()
        self.planner.assert_not_called();self.runner.execute.assert_not_called()

    def test_cold_robot_connects_readonly_before_first_planning_capture(self):
        from server.robot import Robot,JOINTS
        with tempfile.TemporaryDirectory() as runtime:
            calibration={name:{'id':index+1,'range_min':0,'range_max':4095,'homing_offset':0}
                         for index,name in enumerate(JOINTS)}
            cold=Robot({'follower_port':'fake://no-hardware','max_ticks_per_second':90,
                        'max_tracking_error_ticks':90},calibration,runtime,self.camera)
            self.assertFalse(cold.connected);self.assertIsNone(cold.ph)
            port=SimpleNamespace(openPort=Mock(return_value=True),setBaudRate=Mock(),
                                 closePort=Mock(),ser=SimpleNamespace(exclusive=False))
            def read(_port,motor,address):
                return {3:777,31:0,33:0,9:0,11:4095,56:2000}[address],0,0
            protocol=SimpleNamespace(read1ByteTxRx=read,read2ByteTxRx=read,
                                     write1ByteTxRx=Mock(),write2ByteTxRx=Mock())
            def capture():
                self.assertTrue(cold.connected)
                self.assertEqual(cold.read(),dict.fromkeys(JOINTS,2000))
                return copy.deepcopy(self.frame)
            self.grasp.capture.side_effect=capture
            try:
                with patch.object(self.app,'robot',cold),patch('server.robot.PortHandler',return_value=port),\
                        patch('server.robot.PacketHandler',return_value=protocol):
                    done=self.finish(self.post())
                self.assertEqual(done['status'],'completed',done['message'])
                self.assertFalse(cold.armed)
                protocol.write1ByteTxRx.assert_not_called();protocol.write2ByteTxRx.assert_not_called()
            finally:
                if getattr(cold,'lockfile',None):cold.lockfile.close()

    def test_blocked_runner_preserves_partial_receipt_without_completion(self):
        def blocked(plan, report, cancel, publish):
            publish({'plan': plan, 'completed': False, 'status': 'blocked',
                     'steps': [], 'reason': '這一步還不能走'})
            raise RuntimeError('這一步還不能走')
        self.runner.execute.side_effect = blocked
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'failed')
        self.assertFalse(done['command_result']['completed'])
        self.assertEqual(done['command_result']['status'], 'blocked')
        self.assertIn('還不能走', done['message'])

    def test_world_state_endpoint_is_read_only_and_marks_dark_snapshot_stale(self):
        empty=self.client.get('/api/world-state').json()
        self.assertFalse(empty['available'])
        self.memory.remember(self.frame)
        self.camera.fresh.return_value=False
        state=self.client.get('/api/world-state').json()
        self.assertTrue(state['available']);self.assertFalse(state['fresh'])
        self.assertFalse(state['motion_authorized'])
        self.grasp.capture.assert_not_called();self.assertEqual(self.queue,[])

    def test_catalogue_declares_experimental_conditional_stack_without_credentials(self):
        with patch.dict('os.environ', {'JEV_API_KEY': 'DO_NOT_EXPOSE_PROVIDER_CREDENTIAL'}):
            response = self.client.get('/api/skills')
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['unavailable'], [])
        self.assertTrue({'place', 'stack'} <= {s['skill'] for s in result['experimental']})
        self.assertFalse(result['learned_motor_policy'])
        self.assertFalse(result['limits']['plan_is_motion_authorization'])
        self.assertEqual(result['confidence_policy']['local_move'], .7)
        self.assertEqual(result['confidence_policy']['wave'], .8)
        self.assertEqual(result['confidence_policy']['close'], .9)
        self.assertNotIn('DO_NOT_EXPOSE', response.text)
        self.assertNotIn('api_key', response.text.lower())
        self.assertEqual(self.queue, [])

    def test_wide_estimate_enriches_same_observation_without_refreshing_its_age(self):
        from server.world_state import policy_state
        stamp=time.monotonic()-20
        views=[{**self.frame,'id':f'wide-{i}','capture_monotonic':stamp+i*.01,
                'captured_at':time.time()-20+i*.01} for i in range(10)]
        self.robot.revision=0
        self.robot.wide_observation=Mock(return_value=views)
        scene={'objects':[{'name':'cup','frame_index':3,'confidence':.8,'safe_to_touch':False,
                          'bbox':{'x':.2,'y':.2,'w':.3,'h':.3},'actions':[]}]}
        brain=SimpleNamespace(last_latency=.1,perceive=Mock(return_value=scene))
        seen_ids=[]
        def estimate(*args,**kwargs):
            before=self.memory.latest(camera_generation=self.camera.generation,camera_fresh=True)
            self.assertIsNone(before['objects'][0]['surface_3d'])
            seen_ids.append(scene['id'])
            return {'source':'wrist-only','stability':'uncertain','needs_more_views':True,
                    'evidence':{'observations':[{k:f[k] for k in ('id','camera_generation','capture_monotonic')} for f in views]},
                    'objects':[{'name':'cup','object_index':0,'scene_frame_id':'wide-9','status':'estimated',
                                'position_reference':'median-visible-surface-not-object-center',
                                'position_m':[.1,.2,.3],'same_point_p95_disagreement_m':.02}]}
        with patch.object(self.app,'brain',brain),patch.object(self.app,'capture_segment',return_value={}), \
             patch.object(self.app.spatial,'record_inputs'),patch.object(self.app.spatial,'estimate',side_effect=estimate):
            accepted=self.client.post('/api/observe-wide',headers=self.headers,json={'query':'cup'})
            done=self.finish(accepted)
        self.assertEqual(done['status'],'completed')
        self.assertEqual(done['result']['id'],seen_ids[0])
        after=self.client.get('/api/world-state').json()
        self.assertEqual(after['observation_id'],'wide-9')
        self.assertEqual(after['captured_monotonic'],views[-1]['capture_monotonic'])
        self.assertGreaterEqual(after['age_s'],19.9)
        self.assertFalse(after['fresh'])
        self.assertEqual(policy_state(after)['objects'][0]['surface_position_m'],[.1,.2,.3])
        self.assertFalse(after['objects'][0]['surface_3d']['metric_accuracy_verified'])


if __name__ == '__main__':
    unittest.main()
