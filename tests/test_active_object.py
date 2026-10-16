"""Active perception resumes the same object task; no camera, motors or network."""
import copy
import json
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from pydantic import ValidationError

from server import active_observation as ao
from server import motion_program as mp
from server import object_program as op
from server.gripper_power import SessionCancelled
from test_object_program import Fixture as ObjectFixture, program


def fixture_camera_pose(raw):
    """Synthetic camera translation for loop tests, not a hardware FK model."""
    transform = np.eye(4)
    transform[0, 3] = (raw['wrist_roll'] - 2000) * .001
    return transform


class Fixture(ObjectFixture):
    def __init__(self):
        super().__init__()
        transforms = self.model.fk.transforms
        self.model.fk.transforms = lambda raw: {
            **transforms(raw), 'wrist_camera_optical': fixture_camera_pose(raw)}


def observe(task='grasp', candidate_id='view-1', **verdict):
    value = program(task)
    value.update(status='observe', steps=[], target_visible=False,
                 path_description='先換個角度看清楚夾爪後面的物件',
                 observation={'candidate_id': candidate_id, 'gripper_empty': True,
                              'human_clear': True, 'clear_path': True, 'confidence': .85,
                              'reason': '從旁邊看看被夾爪擋住的地方', **verdict})
    return value


def unsupported(task='grasp'):
    value = program(task)
    value.update(status='unsupported', steps=[], target_visible=False,
                 unsupported_reason='夾爪擋住物件，旁邊也沒有能換角度的空間')
    return value


def candidate(frame, candidate_id='view-1'):
    """A sealed fixture contract; real geometry production has separate tests."""
    target = {**frame['joints'], 'wrist_roll': frame['joints']['wrist_roll'] + 10}
    return ao._seal({'id': candidate_id, 'kind': 'body', 'label': '往旁邊看看',
                     'origin': copy.deepcopy(frame['joints']), 'reference': copy.deepcopy(frame),
                     'descriptor': {'id': candidate_id, 'kind': 'body',
                                    'label': '往旁邊看看', 'jaw_changes': False,
                                    'camera_change_model': {
                                        'before_camera_transform_base_m': fixture_camera_pose(frame['joints']).tolist(),
                                        'after_camera_transform_base_m': fixture_camera_pose(target).tolist()}}})


class ActiveObjectLoopTests(unittest.TestCase):
    def setUp(self):
        self.g = Fixture()
        self.cancel = threading.Event()
        self.report = Mock()
        self.plans = []
        self.planning_calls = []
        self.stage = None
        self.messages = []
        self.observation_charge = 1.0
        self.observation_deadlines = []
        self.motion_deadlines = []
        self.g.brain._infer.side_effect = self.infer
        self.patches = [patch.object(op, 'unchanged_wrist', return_value={}),
                        patch.object(op, 'build_observation_candidates', side_effect=self.candidates),
                        patch.object(op, 'execute_observation', side_effect=self.execute_observation),
                        patch.object(mp, 'execute_prepared_segment', side_effect=self.execute_motion)]
        self.unchanged, self.build, self.observe_motor, self.move_motor = [p.start() for p in self.patches]
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])

    def candidates(self, grasp, frame, *, origin, used, history, cancel,
                   target_bbox=None, image_size=None, diagnostics=None):
        return [candidate(frame, f'view-{len(used) + 1}')]

    def infer(self, prompt, frames, schema, **kwargs):
        if schema is not op.ObjectProgram:
            return self.g.infer(prompt, frames, schema, **kwargs)
        self.planning_calls.append({'context': json.loads(prompt.split('INPUT_DATA_JSON\n')[1]),
                                    'frames': copy.deepcopy(frames), 'prompt': prompt})
        if not self.plans:
            raise AssertionError('Unexpected additional planning call')
        value = self.plans.pop(0)
        return op.ObjectProgram.model_validate(value() if callable(value) else value)

    def execute_observation(self, grasp, selected, reference, cancel, report, *, verdict, deadline):
        self.observation_deadlines.append(deadline - op.time.monotonic())
        self.g.events.append('observe')
        self.g.current['wrist_roll'] += 10
        self.g.robot.goals = dict(self.g.current)
        return {'candidate_id': selected['id'], 'kind': 'body', 'completed': True,
                'before': copy.deepcopy(reference), 'after': self.g.capture(),
                'physical_seconds_charged': self.observation_charge,
                'task_completed': False, 'information_gain_verified': False}

    def execute_motion(self, *args, **kwargs):
        self.motion_deadlines.append(kwargs['deadline'] - op.time.monotonic())
        return self.g.execute(*args, **kwargs)

    def run_program(self, task='grasp', query='把白色圓筒放在方塊上面'):
        def stage(value):
            self.stage = value
        def report(message):
            self.messages.append((self.stage, message))
            self.report(message)
        return op.run_object_program(self.g, query, report, self.cancel, task=task,
                                     report_stage=stage)

    def assert_no_task_success(self):
        self.assertFalse(self.g.latest['completed'])
        for key in ('grasp_confirmed', 'place_confirmed', 'stack_confirmed'):
            self.assertFalse(self.g.latest[key])

    def test_observe_then_ready_retains_task_query_source_and_destination(self):
        for task in ('grasp', 'place', 'stack'):
            with self.subTest(task=task):
                # Each task receives a fresh controller and its own motor ledger.
                self.g = Fixture()
                self.g.brain._infer.side_effect = self.infer
                self.planning_calls.clear()
                self.messages.clear()
                self.plans = [observe(task), program(task)]
                result = self.run_program(task)
                self.assertTrue(result['completed'])
                self.assertTrue(result[f'{task}_confirmed'])
                self.assertEqual(result['planning_attempts'], 2)
                self.assertEqual(result['plan']['source_object'], 'toy')
                self.assertEqual(result['plan']['destination'], '' if task == 'grasp' else 'block')
                self.assertEqual([p['context']['task'] for p in self.planning_calls], [task, task])
                self.assertEqual([p['context']['user_request'] for p in self.planning_calls],
                                 ['把白色圓筒放在方塊上面'] * 2)
                self.assertEqual(self.g.events[0], 'observe')
                self.assertEqual(len(result['observations']), 1)
                self.assertFalse(result['observations'][0]['task_completed'])
                reobserve = next(i for i, (stage, _) in enumerate(self.messages) if stage == 'reobserving')
                self.assertEqual(self.messages[reobserve + 1][0], 'planning')

    def test_history_has_distinct_same_attempt_wrist_views_and_current_last(self):
        self.plans = [observe(candidate_id='view-1'), observe(candidate_id='view-2'), program()]
        self.run_program()
        self.assertEqual([len(p['frames']) for p in self.planning_calls], [1, 2, 3])
        final = self.planning_calls[-1]
        frames = final['frames']
        self.assertEqual(len({f['sequence'] for f in frames}), 3)
        self.assertEqual([f['joints']['wrist_roll'] for f in frames], [2000, 2010, 2020])
        self.assertTrue(all(f['source'] == 'wrist' and f['camera_generation'] == 'wrist-1' for f in frames))
        self.assertEqual([r['role'] for r in final['context']['observation_history']],
                         ['earlier_same_attempt', 'earlier_same_attempt', 'current'])
        self.assertEqual(final['context']['frame']['sequence'], frames[-1]['sequence'])
        self.assertEqual([c.kwargs['used'] for c in self.build.call_args_list],
                         [(), ('view-1',), ('view-1', 'view-2')])
        self.assertEqual([[f['sequence'] for f in c.kwargs['history']] for c in self.build.call_args_list],
                         [[], [frames[0]['sequence']], [frames[0]['sequence'], frames[1]['sequence']]])
        self.assertTrue(all(c.kwargs['origin'] == self.g.attempt_start['joints']
                            for c in self.build.call_args_list))
        # Exercise the real novelty filter through the orchestration loop.
        novelty = final['context']['observation_candidates'][0]['view_novelty_model']
        self.assertEqual(novelty['eligible_prior_views'], 2)
        self.assertEqual(novelty['nearest_prior']['frame_id'], frames[1]['id'])
        self.assertEqual(novelty['nearest_prior']['translation_mm'], 20.)
        self.assertFalse(novelty['predicted_revisit'])

    def test_five_observations_are_maximum_even_when_model_requests_six(self):
        self.plans = [observe(candidate_id=f'view-{i}') for i in range(1, 7)]
        reason = '已換過幾個角度，仍看不清楚怎麼操作，先保持目前姿勢'
        with self.assertRaisesRegex(op.ObjectProgramBlocked, reason):
            self.run_program()
        self.assertEqual(self.observe_motor.call_count, 5)
        self.assertEqual(self.build.call_count, 5)
        self.assertEqual(len(self.planning_calls), 6)
        self.assertEqual(self.planning_calls[-1]['context']['observation_candidates'], [])
        self.assertEqual(len(self.planning_calls[-1]['frames']), 6)
        self.g.close_candidate.assert_not_called()
        self.move_motor.assert_not_called()
        self.assertEqual(self.g.latest['reason'], reason)
        self.assert_no_task_success()

    def test_repeated_candidate_is_not_executed_twice(self):
        self.build.side_effect = lambda grasp, frame, **kwargs: [candidate(frame)]
        self.plans = [observe(), observe()]
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '無法執行'):
            self.run_program()
        self.assertEqual(self.observe_motor.call_count, 1)
        self.g.close_candidate.assert_not_called()
        self.assert_no_task_success()

    def test_unavailable_candidate_never_moves(self):
        self.plans = [observe(candidate_id='invented')]
        with self.assertRaisesRegex(ValueError, '未提供'):
            self.run_program()
        self.observe_motor.assert_not_called()
        self.move_motor.assert_not_called()
        self.g.close_candidate.assert_not_called()
        self.assert_no_task_success()

    def test_cancel_during_observation_does_not_resume_or_close(self):
        def execute(*args, **kwargs):
            result = self.execute_observation(*args, **kwargs)
            self.cancel.set()
            return result
        self.observe_motor.side_effect = execute
        self.plans = [observe(), program()]
        with self.assertRaises(SessionCancelled):
            self.run_program()
        self.assertEqual(len(self.planning_calls), 1)
        self.assertEqual(self.g.latest['status'], 'cancelled')
        self.move_motor.assert_not_called()
        self.g.close_candidate.assert_not_called()
        self.assert_no_task_success()

    def test_cancel_while_ready_response_arrives_prevents_manipulation(self):
        def ready():
            self.cancel.set()
            return program()
        self.plans = [observe(), ready]
        with self.assertRaises(SessionCancelled):
            self.run_program()
        self.move_motor.assert_not_called()
        self.g.close_candidate.assert_not_called()
        self.assert_no_task_success()

    def test_bad_current_frame_after_observation_never_reaches_brain(self):
        for change in ({'capture_monotonic': -100}, {'camera_generation': 'reconnected'},
                       {'source': 'engineering-observer'}):
            with self.subTest(change=change):
                self.g = Fixture()
                self.g.brain._infer.side_effect = self.infer
                self.planning_calls.clear()
                self.plans = [observe(), program()]
                def execute(*args, **kwargs):
                    result = self.execute_observation(*args, **kwargs)
                    result['after'].update(change)
                    return result
                self.observe_motor.side_effect = execute
                with self.assertRaises(ValueError):
                    self.run_program()
                self.assertEqual(len(self.planning_calls), 1)
                self.g.close_candidate.assert_not_called()
                self.assert_no_task_success()

    def test_initial_stale_refresh_never_reaches_brain(self):
        capture = self.g.capture
        def stale_capture():
            value = capture()
            if value['sequence'] >= 2:
                value['capture_monotonic'] -= op.MAX_FRAME_AGE_SECONDS + 1
            return value
        self.g.capture = stale_capture
        self.plans = [observe()]
        with self.assertRaisesRegex(ValueError, '過期'):
            self.run_program()
        self.g.brain._infer.assert_not_called()
        self.observe_motor.assert_not_called()
        self.move_motor.assert_not_called()

    def test_no_locally_clear_observation_has_plain_reason_and_no_motion(self):
        self.build.side_effect = None
        self.build.return_value = []
        self.plans = [unsupported()]
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '沒有能換角度'):
            self.run_program()
        self.assertEqual(self.g.latest['status'], 'unsupported')
        self.observe_motor.assert_not_called()
        self.move_motor.assert_not_called()
        self.g.close_candidate.assert_not_called()
        self.assert_no_task_success()

    def test_false_or_low_confidence_observation_reaches_real_guard_but_not_motors(self):
        for denial in ({'confidence': .69}, {'clear_path': False}, {'human_clear': False},
                       {'gripper_empty': False}):
            with self.subTest(denial=denial):
                self.plans = [observe(**denial)]
                self.observe_motor.side_effect = ao.execute_observation
                with patch.object(ao, 'execute_prepared_segment') as motor:
                    with self.assertRaisesRegex(ValueError, '尚未確認'):
                        self.run_program()
                    motor.assert_not_called()
                self.move_motor.assert_not_called()
                self.g.close_candidate.assert_not_called()
                self.assert_no_task_success()

    def test_observation_then_unsupported_is_not_grasp_success(self):
        self.plans = [observe(), unsupported()]
        with self.assertRaises(op.ObjectProgramBlocked):
            self.run_program()
        self.assertEqual(len(self.g.latest['observations']), 1)
        self.assertFalse(self.g.latest['observations'][0]['task_completed'])
        self.assert_no_task_success()
        self.g.brain.verify_grasp.assert_not_called()
        self.g.close_candidate.assert_not_called()

    def test_incomplete_observation_stops_without_replanning(self):
        def incomplete(*args, **kwargs):
            result = self.execute_observation(*args, **kwargs)
            result['completed'] = False
            return result
        self.observe_motor.side_effect = incomplete
        self.plans = [observe(), program()]
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '尚未完成'):
            self.run_program()
        self.assertEqual(len(self.planning_calls), 1)
        self.g.close_candidate.assert_not_called()
        self.assert_no_task_success()

    def test_observations_and_manipulation_share_physical_budget(self):
        self.observation_charge = 10
        self.plans = [observe(candidate_id='view-1'), observe(candidate_id='view-2'), program()]
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '時間不足以完成合爪'):
            self.run_program()
        self.assertEqual(self.observe_motor.call_count, 2)
        self.assertAlmostEqual(self.observation_deadlines[0], 30, delta=.1)
        self.assertAlmostEqual(self.observation_deadlines[1], 20, delta=.1)
        self.assertAlmostEqual(self.motion_deadlines[0], 10, delta=.1)
        limits = [call['context']['limits'] for call in self.planning_calls]
        self.assertEqual([item['physical_execution_seconds'] for item in limits], [30, 20, 10])
        self.assertEqual([item['remaining_observation_actions'] for item in limits], [5, 4, 3])
        self.assertGreaterEqual(self.g.latest['physical_seconds_charged'], 20)
        self.assertEqual(self.move_motor.call_count, 1)
        self.g.close_candidate.assert_not_called()
        self.assert_no_task_success()

    def test_preflight_rejects_whole_body_path_exceeding_remaining_time(self):
        self.observation_charge = 27
        self.plans = [observe(), program(opened=True)]
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '完整路徑超過本次動作時間'):
            self.run_program()
        self.assertEqual(self.observe_motor.call_count, 1)
        self.assertEqual(self.planning_calls[-1]['context']['limits']['physical_execution_seconds'], 3)
        self.assertEqual(self.planning_calls[-1]['context']['limits']['remaining_observation_actions'], 4)
        # Reject the entire manipulation before its optional jaw-open and
        # first approach, even though each individual segment would fit.
        self.g.open_empty.assert_not_called()
        self.move_motor.assert_not_called()
        self.g.close_candidate.assert_not_called()
        self.assertEqual(self.g.latest['physical_seconds_charged'], 27)
        self.assert_no_task_success()


class ActiveObjectPlanningTests(unittest.TestCase):
    def setUp(self):
        self.g = Fixture()
        self.cancel = threading.Event()
        self.g.brain._infer.side_effect = None
        self.g.brain._infer.return_value = op.ObjectProgram.model_validate(program())

    def plan(self, current=None, *, history=(), candidates=()):
        return op.plan_object_program(self.g, '拿起原本那個白色物件', current or self.g.capture(),
                                      self.cancel, history=history, observation_candidates=candidates)

    def test_history_deduplicates_and_bounds_frames_with_current_last(self):
        history = [self.g.capture() for _ in range(7)]
        current = self.g.capture()
        self.plan(current, history=history + [history[-1]])
        prompt, frames, schema = self.g.brain._infer.call_args.args
        self.assertIs(schema, op.ObjectProgram)
        self.assertEqual([v['sequence'] for v in frames], [3, 4, 5, 6, 7, 8])
        context = json.loads(prompt.split('INPUT_DATA_JSON\n')[1])
        self.assertEqual(context['observation_history'][-1]['role'], 'current')
        self.assertEqual(context['observation_history'][-1]['sequence'], current['sequence'])
        self.assertEqual(context['user_request'], '拿起原本那個白色物件')

    def test_d435_or_other_generation_in_history_never_reaches_model(self):
        for changes in ({'source': 'engineering-observer'}, {'source': 'd435'},
                        {'camera_generation': 'another-wrist'}, {'sequence': True},
                        {'capture_monotonic': float('nan')}, {'capture_monotonic': True}):
            with self.subTest(changes=changes):
                prior = self.g.capture()
                prior.update(changes)
                with self.assertRaises(ValueError):
                    self.plan(self.g.capture(), history=[prior])
                self.g.brain._infer.assert_not_called()

    def test_history_must_precede_current_sequence_and_capture_time(self):
        for field in ('sequence', 'capture_monotonic'):
            prior, current = self.g.capture(), self.g.capture()
            prior[field] = current[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, '觀察歷史'):
                self.plan(current, history=[prior])
            self.g.brain._infer.assert_not_called()

    def test_nonpositive_historical_sequence_rejected_before_brain(self):
        for value in (-1, 0):
            with self.subTest(sequence=value):
                prior = self.g.capture()
                prior['sequence'] = value
                with self.assertRaises(ValueError):
                    self.plan(self.g.capture(), history=[prior])
                self.g.brain._infer.assert_not_called()

    def test_blank_or_missing_historical_image_path_rejected_before_brain(self):
        for value in ('', '   ', None):
            with self.subTest(path=value):
                prior = self.g.capture()
                if value is None:
                    del prior['path']
                else:
                    prior['path'] = value
                with self.assertRaises(ValueError):
                    self.plan(self.g.capture(), history=[prior])
                self.g.brain._infer.assert_not_called()

    def test_missing_historical_joint_pose_rejected_before_brain(self):
        for missing in (None, 'shoulder_pan', 'gripper'):
            with self.subTest(missing=missing):
                prior = self.g.capture()
                if missing is None:
                    del prior['joints']
                else:
                    del prior['joints'][missing]
                with self.assertRaises(ValueError):
                    self.plan(self.g.capture(), history=[prior])
                self.g.brain._infer.assert_not_called()

    def test_invalid_historical_joint_ticks_rejected_before_brain(self):
        for axis in ('shoulder_pan', 'gripper'):
            for value in (2000.0, '2000', True, -1, 4096):
                with self.subTest(axis=axis, ticks=value):
                    prior = self.g.capture()
                    prior['joints'][axis] = value
                    with self.assertRaises(ValueError):
                        self.plan(self.g.capture(), history=[prior])
                    self.g.brain._infer.assert_not_called()

    def test_old_views_may_be_compared_but_current_must_be_fresh(self):
        prior = self.g.capture()
        prior['capture_monotonic'] -= 60
        current = self.g.capture()
        self.plan(current, history=[prior])
        self.assertEqual(len(self.g.brain._infer.call_args.args[1]), 2)
        self.g.brain._infer.reset_mock()
        current['capture_monotonic'] -= 60
        with self.assertRaisesRegex(ValueError, '過期'):
            self.plan(current)
        self.g.brain._infer.assert_not_called()

    def test_observe_binds_response_to_current_candidate_id(self):
        current = self.g.capture()
        self.g.brain._infer.return_value = op.ObjectProgram.model_validate(observe(candidate_id='other'))
        with self.assertRaisesRegex(ValueError, '未提供'):
            self.plan(current, candidates=[candidate(current)])
        self.g.brain._infer.return_value = op.ObjectProgram.model_validate(observe())
        result = self.plan(current, candidates=[candidate(current)])
        self.assertEqual(result.status, 'observe')
        self.assertFalse(result.target_visible)
        self.assertEqual(result.observation.candidate_id, 'view-1')

    def test_observation_schema_cannot_include_manipulation_steps(self):
        value = observe()
        value['steps'] = program()['steps']
        with self.assertRaises(ValidationError):
            op.ObjectProgram.model_validate(value)
        value = observe()
        value['initial_jaw_empty'] = False
        with self.assertRaises(ValidationError):
            op.ObjectProgram.model_validate(value)

    def test_prompt_requests_local_observation_without_proving_entire_grasp(self):
        self.plan()
        prompt = self.g.brain._infer.call_args.args[0].split('INPUT_DATA_JSON\n')[0]
        self.assertIn('prefer status="observe"', prompt)
        self.assertIn("Judge only that observation action's disclosed swept volume", prompt)
        self.assertIn('need not prove the whole future task is clear', prompt)
        self.assertIn('ORIGINAL task', prompt)
        self.assertIn('LAST supplied image is the current WRIST camera', prompt)
        self.assertIn('ONE short everyday sentence', prompt)
        self.assertIn('Never invent observation targets', prompt)


if __name__ == '__main__':
    unittest.main()
