"""Product semantic actions with fake camera/brain/motor owners only.

Never enter TestClient's lifespan. The fake robot deliberately has no motor,
connection, read, rearm, or hold methods: a camera-only inspection must finish
without any of them. Worker execution is explicit and synchronous in tests.
"""
import copy
import importlib
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class SemanticObjectActionTests(unittest.TestCase):
    def setUp(self):
        self.m = importlib.import_module('server.app')
        self.robot = types.SimpleNamespace(
            stop=threading.Event(), pwm_owner=None, revision=41)
        self.wrist = {
            'id': 'new-wrist-frame', 'source': 'wrist',
            'url': '/api/frames/new-wrist-frame', 'path': 'not-read-or-written.jpg',
            'captured_at': time.time(), 'capture_monotonic': time.monotonic(),
            'camera_generation': 'current-generation', 'sequence': 42,
            'image_size': [1280, 720],
        }
        self.details = {
            'visible': True, 'object_name': '新影像中的圓筒', 'confidence': .73,
            'summary': '白色外觀，標籤部分可見。', 'attributes': ['白色外觀'],
            'visible_text': ['ABC'], 'unknowns': ['內容物無法確認'],
            'bbox': {'x': .2, 'y': .1, 'w': .2, 'h': .5},
            'safe_to_touch': False,
            'actions': [{'kind': 'inspect', 'label': '查看細節',
                         'reason': '辨識可見文字', 'target': ''}],
        }
        self.camera = types.SimpleNamespace(
            snapshot=Mock(side_effect=lambda: copy.deepcopy(self.wrist)),
            fresh=Mock(return_value=True), error='')
        self.search_scene = {
            'query': '圓筒', 'summary': '看見圓筒', 'empty': False,
            'selected_index': 0, 'uncertainty': '', 'objects': [{
                'name': '圓筒', 'description': '白色外觀', 'confidence': .7,
                'bbox': self.details['bbox'], 'frame_index': 0,
                'matches_query': True, 'safe_to_touch': False, 'actions': [],
            }],
        }
        self.brain = types.SimpleNamespace(
            _infer_text=Mock(return_value={'intent':'search','title':'尋找物件','summary':'搜尋眼前物件。',
                                          'label':'','availability':'none','reason':''}),
            inspect_details=Mock(side_effect=lambda *args: copy.deepcopy(self.details)),
            perceive=Mock(side_effect=lambda *args: copy.deepcopy(self.search_scene)),
            last_latency=.3)
        self.views = [{**self.wrist, 'id': f'view-{i}', 'sequence': 42 + i,
                       'url': f'/api/frames/view-{i}', 'path': f'not-read-view-{i}.jpg'}
                      for i in range(7)]
        self.grasp = types.SimpleNamespace(
            latest=None,
            run_program=Mock(return_value={'grasp_confirmed': False, 'state': 'holding_unverified'}),
            planned_wave=Mock(return_value={'wave_completed': True, 'cycles': 2,
                                   'measured_roll_range_deg': [-5.2, 5.1]}),
            observe_views=Mock(return_value={
                'frames': self.views, 'measured_roll_range_deg': [-19.9, 19.8],
                'return_error_deg': .1, 'camera_baseline_model_mm': 50.2,
                'not_external_metrology': True,
                'initial_frame': {'id': 'preflight', 'path': 'private-preflight.jpg'}}))
        self.spatial = types.SimpleNamespace(estimate=Mock(
            side_effect=AssertionError('single-axis search must not fit metric geometry')))
        self.robot.refinement_scan = Mock(side_effect=AssertionError('product roll search must not start refinement'))
        self.robot.scan = Mock(side_effect=AssertionError('product roll search must not start legacy body scan'))
        self.queue = []
        self.executor = types.SimpleNamespace(submit=self.queue.append)
        for name, value in (
            ('robot', self.robot), ('camera', self.camera), ('brain', self.brain),
            ('grasp', self.grasp), ('executor', self.executor), ('active', None),
            ('spatial', self.spatial),
            ('tactician', types.SimpleNamespace(choose=Mock(return_value={'status':'abstain'}))),
            ('jobs', {}), ('scenes', {}), ('frames', {}),
            ('snapshot_owner', threading.local()),
        ):
            guard = patch.object(self.m, name, value)
            guard.start()
            self.addCleanup(guard.stop)
        self.client = TestClient(self.m.app)  # No lifespan, sockets or hardware.
        self.addCleanup(self.client.close)
        self.headers = {'x-lerobot-token': self.m.TOKEN}

    def post(self, kind='inspect', query='原先找到的圓筒', **extra):
        return self.client.post('/api/object-action', headers=self.headers,
                                json={'kind': kind, 'query': query, **extra})

    def finish(self, accepted):
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json()['status'], 'running')
        self.assertEqual(len(self.queue), 1)
        self.queue.pop(0)()
        result = self.client.get('/api/jobs/' + accepted.json()['id'])
        self.assertEqual(result.status_code, 200)
        self.assertFalse(hasattr(self.m.snapshot_owner, 'job'))
        return result.json()

    def test_inspect_stopped_and_holding_uses_only_new_wrist_frame_without_rearming(self):
        self.robot.stop.set()
        owner = object()
        self.robot.pwm_owner = owner
        self.m.scenes['expired'] = {
            'id': 'expired', 'created_at': 0, 'frames': [],
            'objects': [{'name': 'old unrelated item', 'bbox': {'x': .9}}],
        }
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'completed', done['message'])
        self.assertEqual(done['kind'], 'inspect')
        self.assertTrue(self.robot.stop.is_set())
        self.assertIs(self.robot.pwm_owner, owner)
        self.assertEqual(self.robot.revision, 41)
        self.camera.snapshot.assert_called_once_with()
        args = self.brain.inspect_details.call_args.args
        self.assertEqual(args[0], '原先找到的圓筒')
        self.assertEqual([f['id'] for f in args[1]], ['new-wrist-frame'])
        self.assertIs(args[2], self.m.jobs[done['id']]['cancel'])
        result = done['result']
        self.assertNotEqual(result['id'], 'expired')
        self.assertEqual(result['details']['frame']['id'], 'new-wrist-frame')
        self.assertNotIn('path', result['details']['frame'])
        self.assertNotIn('path', result['frames'][0])
        self.assertEqual(result['objects'][0]['confidence'], .73)
        self.assertEqual(result['details']['visible_text'], ['ABC'])
        self.assertEqual(result['details']['unknowns'], ['內容物無法確認'])
        self.assertEqual(self.m.jobs[done['id']]['frame_pins'], {'new-wrist-frame'})
        self.grasp.run_program.assert_not_called()
        self.grasp.planned_wave.assert_not_called()
        self.grasp.observe_views.assert_not_called()

    def test_pending_job_rejects_inspect_even_when_stopped_and_holding(self):
        first = self.post()
        self.assertEqual(first.status_code, 200)
        self.robot.stop.set()
        self.robot.pwm_owner = object()
        second = self.post()
        self.assertEqual(second.status_code, 409)
        self.assertEqual(len(self.queue), 1)
        self.camera.snapshot.assert_not_called()
        self.brain.inspect_details.assert_not_called()

    def test_camera_failure_does_not_substitute_an_expired_frame_or_result(self):
        self.m.frames['old'] = {**self.wrist, 'id': 'old', 'captured_at': 0}
        self.camera.snapshot.side_effect = RuntimeError('腕部相機畫面已過期')
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'failed')
        self.assertIn('畫面已過期', done['message'])
        self.assertIsNone(done['result'])
        self.assertEqual(self.m.scenes, {})
        self.brain.inspect_details.assert_not_called()

    def test_absent_target_keeps_details_and_unknowns_without_a_phantom_pick(self):
        self.details.update(visible=False, object_name='', confidence=0.,
                            bbox=None, attributes=[], visible_text=[],
                            summary='目前無法確認目標。')
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'completed')
        result = done['result']
        self.assertEqual(result['objects'], [])
        self.assertEqual(result['selected_index'], -1)
        self.assertTrue(result['empty'])
        self.assertFalse(result['details']['visible'])
        self.assertEqual(result['details']['unknowns'], ['內容物無法確認'])
        self.grasp.run_program.assert_not_called()

    def test_inspection_failure_is_a_failed_job_with_no_false_result(self):
        self.brain.inspect_details.side_effect = RuntimeError('無法辨識這次影像')
        done = self.finish(self.post())
        self.assertEqual(done['status'], 'failed')
        self.assertIsNone(done['result'])
        self.assertIn('無法辨識', done['message'])

    def test_pick_semantics_use_live_flag_and_named_fresh_grasp_without_a_scene(self):
        with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': False}):
            rejected = self.post('pick')
        self.assertEqual(rejected.status_code, 409)
        self.assertEqual(self.queue, [])
        with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
            done = self.finish(self.post('pick', '新指定的物件'))
        self.assertEqual(done['kind'], 'grasp-attempt')
        self.assertEqual(done['status'], 'failed')
        self.assertEqual(self.grasp.run_program.call_args.args[0], '新指定的物件')
        self.assertFalse(done['engineering_result']['grasp_confirmed'])
        self.assertIn('尚未確認', done['message'])
        self.assertEqual(self.m.scenes, {})
        self.brain.inspect_details.assert_not_called()

    def test_place_stack_semantic_routes_keep_full_instruction_and_matching_verified_receipt(self):
        instruction = '把白黃圓筒抓起來，放到藍色盒子的左上方'
        for kind in ('place', 'stack'):
            with self.subTest(kind=kind):
                self.grasp.run_program.reset_mock()
                self.grasp.run_program.return_value = {'task': kind, 'completed': True,
                    'grasp_confirmed': True, kind + '_confirmed': True}
                with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
                    done = self.finish(self.post(kind, instruction))
                self.assertEqual(done['kind'], 'object-program')
                self.assertEqual(done['status'], 'completed', done['message'])
                self.assertEqual(done['stage'], 'done')
                self.assertIsNone(done['result'])
                self.assertTrue(done['engineering_result'][kind + '_confirmed'])
                self.grasp.run_program.assert_called_once()
                call = self.grasp.run_program.call_args
                self.assertEqual(call.args[0], instruction)
                self.assertEqual(call.kwargs['task'], kind)
                self.assertTrue(callable(call.kwargs['report_stage']))
                self.assertIs(call.args[2], self.m.jobs[done['id']]['cancel'])
        self.camera.snapshot.assert_not_called()
        self.grasp.observe_views.assert_not_called()

    def test_place_stack_need_enabled_idle_controller_before_queueing(self):
        for kind in ('place', 'stack'):
            with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': False}):
                self.assertEqual(self.post(kind, '把杯子放到盒子上').status_code, 409)
            with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
                self.robot.stop.set()
                self.assertEqual(self.post(kind, '把杯子放到盒子上').status_code, 409)
                self.robot.stop.clear()
                self.robot.pwm_owner = object()
                self.assertEqual(self.post(kind, '把杯子放到盒子上').status_code, 409)
                self.robot.pwm_owner = None
        self.assertEqual(self.queue, [])
        self.grasp.run_program.assert_not_called()

    def test_place_stack_flag_without_complete_program_cannot_complete_api_job(self):
        for kind in ('place', 'stack'):
            key = kind + '_confirmed'
            for evidence in ({'completed': False, key: True}, {key: True},
                             {'completed': True, key: False}, {'completed': True, key: 'true'},
                             {'completed': True, 'grasp_confirmed': True, 'released': True}):
                with self.subTest(kind=kind, evidence=evidence):
                    self.grasp.run_program.return_value = evidence
                    with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
                        done = self.finish(self.post(kind, '把杯子放到盒子上'))
                    self.assertEqual(done['status'], 'failed')
                    self.assertIn('尚未確認', done['message'])
                    self.assertIsNone(done['result'])
                    self.assertNotIn('已確認堆疊完成', done['message'])

    def test_object_program_reports_true_phases_and_retains_partial_failure_evidence(self):
        previous = {'task': 'stack', 'completed': True, 'stack_confirmed': True}
        self.grasp.latest = previous
        seen = []
        def partial(query, report, cancel, *, task, report_stage):
            for phase in ('planning', 'moving', 'verifying'):
                report_stage(phase)
                report('這是目前階段')
                seen.append(self.m.jobs[self.m.active]['stage'])
            self.grasp.latest = {'task': task, 'completed': False, 'grasp_confirmed': True,
                                 'stack_confirmed': False, 'reason': '物件尚未放穩'}
            raise RuntimeError('物件尚未放穩')
        self.grasp.run_program.side_effect = partial
        with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
            done = self.finish(self.post('stack', '把白黃圓筒堆在藍色盒子上'))
        self.assertEqual(seen, ['planning', 'moving', 'verifying'])
        self.assertEqual(done['status'], 'failed')
        self.assertFalse(done['engineering_result']['completed'])
        self.assertTrue(done['engineering_result']['grasp_confirmed'])
        self.assertFalse(done['engineering_result']['stack_confirmed'])
        self.assertNotEqual(done['engineering_result'], previous)

    def test_program_failure_before_new_evidence_does_not_reuse_old_success(self):
        self.grasp.latest = {'completed': True, 'place_confirmed': True}
        self.grasp.run_program.side_effect = RuntimeError('這次規劃逾時')
        with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
            done = self.finish(self.post('place', '把白黃圓筒放到藍色盒子上'))
        self.assertEqual(done['status'], 'failed')
        self.assertNotIn('engineering_result', done)

    def test_cached_placement_recommendation_requires_safe_source_and_destination(self):
        for kind in ('place', 'stack'):
            with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
                for safe, destination, expected in ((False, '藍色盒子', 409), (True, ' ', 422)):
                    self.m.scenes['selected'] = {'id': 'selected', 'created_at': time.time(),
                        'objects': [{'name': '白黃圓筒', 'safe_to_touch': safe, 'actions': [
                            {'kind': kind, 'target': destination, 'available': True, 'unavailable_reason': ''}]}]}
                    response = self.client.post('/api/actions', headers=self.headers,
                        json={'scene_id': 'selected', 'object_index': 0, 'action_index': 0})
                    self.assertEqual(response.status_code, expected, response.text)
        self.assertEqual(self.queue, [])
        self.grasp.run_program.assert_not_called()

    def test_cached_stack_recommendation_preserves_source_and_destination(self):
        self.m.scenes['selected'] = {'id': 'selected', 'created_at': time.time(),
            'objects': [{'name': '白黃圓筒', 'safe_to_touch': True, 'actions': [
                {'kind': 'stack', 'target': '藍色盒子', 'available': True, 'unavailable_reason': ''}]}]}
        self.grasp.run_program.return_value = {'task': 'stack', 'completed': True, 'stack_confirmed': True}
        with patch.dict(self.m.CONFIG, {'experimental_pick_enabled': True}):
            done = self.finish(self.client.post('/api/actions', headers=self.headers,
                json={'scene_id': 'selected', 'object_index': 0, 'action_index': 0}))
        self.assertEqual(done['status'], 'completed', done['message'])
        call = self.grasp.run_program.call_args
        self.assertIn('白黃圓筒', call.args[0])
        self.assertIn('藍色盒子', call.args[0])
        self.assertEqual(call.kwargs['task'], 'stack')
        self.assertTrue(callable(call.kwargs['report_stage']))

    def test_wave_has_real_worker_receipt_and_rejects_stopped_or_owned_state(self):
        self.robot.stop.set()
        self.assertEqual(self.post('wave', '').status_code, 409)
        self.robot.stop.clear()
        self.robot.pwm_owner = object()
        self.assertEqual(self.post('wave', '').status_code, 409)
        self.assertEqual(self.queue, [])
        self.robot.pwm_owner = None
        done = self.finish(self.post('wave', ''))
        self.assertEqual(done['kind'], 'wave')
        self.assertEqual(done['status'], 'completed')
        self.assertEqual(done['engineering_result'], {
            'wave_completed': True, 'cycles': 2,
            'measured_roll_range_deg': [-5.2, 5.1]})
        self.grasp.planned_wave.assert_called_once()
        self.camera.snapshot.assert_not_called()

    def force_wave(self, jid, **kwargs):
        return self.client.post('/api/jobs/' + jid + '/force-wave',
                                headers=self.headers, json=kwargs)

    def test_manual_wave_signals_same_worker_once_without_starting_motion(self):
        accepted = self.post('wave', '')
        jid = accepted.json()['id']
        job = self.m.jobs[jid]
        self.assertFalse(job['manual_wave_signal'].is_set())
        self.assertNotIn('manual_wave_signal', accepted.json())
        first = self.force_wave(jid)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertTrue(first.json()['manual_wave_requested'])
        self.assertTrue(job['manual_wave_signal'].is_set())
        self.assertFalse(job['cancel'].is_set())
        self.assertNotIn('manual_wave_signal', first.json())
        self.grasp.planned_wave.assert_not_called()
        self.assertEqual(len(self.queue), 1)
        second = self.force_wave(jid)
        self.assertEqual(second.json(), first.json())
        done = self.finish(accepted)
        self.assertEqual(done['status'], 'completed')
        self.assertIs(self.grasp.planned_wave.call_args.kwargs['manual_override'], job['manual_wave_signal'])
        self.assertEqual(self.force_wave(jid).status_code, 409)

    def test_manual_wave_does_not_carry_to_next_wave(self):
        first = self.post('wave', '')
        self.force_wave(first.json()['id'])
        self.finish(first)
        second = self.post('wave', '')
        self.assertFalse(second.json()['manual_wave_requested'])
        self.assertFalse(self.m.jobs[second.json()['id']]['manual_wave_signal'].is_set())
        self.assertEqual(self.force_wave(first.json()['id']).status_code, 409)

    def test_manual_wave_rejects_stop_cancel_holding_and_executing_wave(self):
        accepted = self.post('wave', '')
        jid = accepted.json()['id']
        job = self.m.jobs[jid]
        self.robot.stop.set()
        self.assertEqual(self.force_wave(jid).status_code, 409)
        self.robot.stop.clear()
        job['cancel'].set()
        self.assertEqual(self.force_wave(jid).status_code, 409)
        job['cancel'].clear()
        self.robot.pwm_owner = object()
        self.assertEqual(self.force_wave(jid).status_code, 409)
        self.robot.pwm_owner = None
        job['manual_wave_phase'] = 'waving'
        self.assertEqual(self.force_wave(jid).status_code, 409)
        self.assertFalse(job['manual_wave_signal'].is_set())
        self.grasp.planned_wave.assert_not_called()
        self.assertEqual(len(self.queue), 1)

    def test_manual_wave_cannot_override_inspection_or_missing_jobs(self):
        accepted = self.post('inspect')
        self.assertEqual(self.force_wave(accepted.json()['id']).status_code, 409)
        self.assertEqual(self.force_wave('missing').status_code, 404)
        self.assertNotIn('manual_wave_signal', self.m.jobs[accepted.json()['id']])
        self.assertEqual(len(self.queue), 1)

    def test_manual_wave_requires_session_and_has_no_motor_parameters(self):
        accepted = self.post('wave', '')
        jid = accepted.json()['id']
        for headers in ({}, {'x-lerobot-token': 'wrong'},
                        {**self.headers, 'Origin': 'https://untrusted.invalid'}):
            response = self.client.post('/api/jobs/' + jid + '/force-wave', headers=headers, json={})
            self.assertEqual(response.status_code, 403)
        self.assertEqual(self.force_wave(jid, amplitude=90).status_code, 422)
        self.assertFalse(self.m.jobs[jid]['manual_wave_signal'].is_set())

    def test_wave_progress_marks_waving_before_first_motor_leg(self):
        def wave(report, cancel, *, manual_override):
            report('向你揮揮手 · 1 / 2')
            jid = self.m.active
            self.assertEqual(self.m.jobs[jid]['manual_wave_phase'], 'waving')
            self.assertEqual(self.force_wave(jid).status_code, 409)
            return {'wave_completed': True}
        self.grasp.planned_wave.side_effect = wave
        self.finish(self.post('wave', ''))

    def test_schema_rejects_cached_coordinates_and_blank_object_names(self):
        for extra in ({'scene_id': 'old'}, {'object_index': 0},
                      {'action_index': 0}, {'x': .5}, {'joints': {'gripper': 2000}}):
            with self.subTest(extra=extra):
                self.assertEqual(self.post(**extra).status_code, 422)
        for kind in ('inspect', 'pick', 'place', 'stack'):
            with self.subTest(kind=kind):
                self.assertEqual(self.post(kind, '  ').status_code, 422)
        self.assertEqual(self.post('teleport').status_code, 422)
        self.assertEqual(self.queue, [])

    def test_semantic_endpoint_keeps_token_and_origin_boundary(self):
        for headers in ({}, {'x-lerobot-token': 'wrong'},
                        {**self.headers, 'Origin': 'https://untrusted.invalid'}):
            response = self.client.post('/api/object-action', headers=headers,
                                        json={'kind': 'inspect', 'query': '圓筒'})
            self.assertEqual(response.status_code, 403)
        self.assertEqual(self.queue, [])
        self.camera.snapshot.assert_not_called()

    def search(self, **changes):
        return self.client.post('/api/search', headers=self.headers,
                                json={'query': '圓筒', 'mode': 'search', **changes})

    def assert_no_metric_or_body_scan(self):
        self.spatial.estimate.assert_not_called()
        self.robot.refinement_scan.assert_not_called()
        self.robot.scan.assert_not_called()

    def test_point_recognition_does_not_report_an_unperformed_grasp_precision_check(self):
        self.m.frames[self.wrist['id']] = copy.deepcopy(self.wrist)
        done = self.finish(self.search(mode='point', query='', frame_id=self.wrist['id'], x=.3, y=.4))
        self.assertEqual(done['status'], 'completed', done['message'])
        scene = done['result']
        self.assertEqual(scene['notice_kind'], 'recognition_only')
        self.assertIn('尚未嘗試抓取', scene['notice'])
        self.assertNotIn('尚未通過精度驗證', scene['notice'])
        self.assertEqual(scene['query'], '圓筒')
        self.grasp.run_program.assert_not_called()
        self.grasp.observe_views.assert_not_called()
        self.camera.snapshot.assert_not_called()
        self.assert_no_metric_or_body_scan()

    def test_seven_view_search_selects_distinct_inference_views_without_metric_fit(self):
        done = self.finish(self.search())
        self.assertEqual(done['status'], 'completed', done['message'])
        args = self.brain.perceive.call_args.args
        self.assertEqual([f['id'] for f in args[1]], ['view-0', 'view-2', 'view-5', 'view-6'])
        self.assertIsNone(args[2])
        self.assertIsNone(args[3])
        self.assertIs(args[4], self.m.jobs[done['id']]['cancel'])
        observation = done['result']['observation']
        self.assertEqual(observation['view_count'], 7)
        self.assertEqual(observation['mode'], 'wrist_roll_arc')
        self.assertIs(observation['metric_calibrated'], False)
        self.assertIs(observation['not_external_metrology'], True)
        self.assertNotIn('frames', observation)
        self.assertNotIn('initial_frame', observation)
        self.assertEqual(observation['return_error_deg'], .1)
        self.assertNotIn('spatial', done['result'])
        self.assertEqual(len(done['result']['frames']), 4)
        self.camera.snapshot.assert_not_called()
        self.grasp.observe_views.assert_called_once()
        self.assert_no_metric_or_body_scan()

    def test_refused_sweep_before_movement_falls_back_to_new_single_wrist_frame(self):
        self.grasp.observe_views.side_effect = RuntimeError('sweep clearance insufficient')
        done = self.finish(self.search())
        self.assertEqual(done['status'], 'completed', done['message'])
        self.assertEqual([f['id'] for f in self.brain.perceive.call_args.args[1]], ['new-wrist-frame'])
        self.camera.snapshot.assert_called_once_with()
        self.assertNotIn('observation', done['result'])
        self.assertIn('目前先原地觀察', done['result']['notice'])
        self.assertEqual(self.robot.revision, 41)
        self.assert_no_metric_or_body_scan()

    def test_inference_failure_during_scan_does_not_start_another_cloud_request(self):
        from server.brain import BrainTransportError
        for kind in ('network','timeout','authentication','rate_limit','invalid_response'):
            with self.subTest(kind=kind):
                self.grasp.observe_views.side_effect = BrainTransportError(kind)
                done = self.finish(self.search())
                self.assertEqual(done['status'], 'failed')
                self.assertIsNone(done['result'])
                self.camera.snapshot.assert_not_called()
                self.brain.perceive.assert_not_called()
                self.assertEqual(self.robot.revision, 41)
                self.assert_no_metric_or_body_scan()

    def test_partial_motion_stop_or_cancellation_never_becomes_single_frame_success(self):
        for reason in ('revision', 'stop', 'cancel'):
            with self.subTest(reason=reason):
                self.robot.stop.clear()
                def interrupted(report, cancel):
                    if reason == 'revision': self.robot.revision += 1
                    elif reason == 'stop': self.robot.stop.set()
                    else: cancel.set()
                    raise RuntimeError('sweep interrupted: ' + reason)
                self.grasp.observe_views.side_effect = interrupted
                done = self.finish(self.search())
                self.assertEqual(done['status'], 'cancelled' if reason == 'cancel' else 'failed')
                self.assertIn('sweep interrupted: ' + reason, done['message'])
                self.assertIsNone(done['result'])
                self.camera.snapshot.assert_not_called()
                self.brain.perceive.assert_not_called()
                self.assertEqual(self.m.scenes, {})
                self.assert_no_metric_or_body_scan()

    def test_unavailable_wrist_camera_rejects_before_sweep_or_fallback(self):
        self.camera.fresh.return_value = False
        done = self.finish(self.search())
        self.assertEqual(done['status'], 'failed')
        self.grasp.observe_views.assert_not_called()
        self.camera.snapshot.assert_not_called()
        self.brain.perceive.assert_not_called()
        self.assert_no_metric_or_body_scan()


if __name__ == '__main__':
    unittest.main(verbosity=2)
