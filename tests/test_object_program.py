"""Object-program causal execution, with no hardware/network or visual claims."""
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pydantic import ValidationError

from server import object_program as op
from server import motion_program as mp
from server.gripper_power import SessionCancelled
from server.local_cartesian import NAMES


def pose(**changes):
    return {**{n: 2000 for n in NAMES}, 'gripper': 1860, **changes}


def step(kind, target=None, **changes):
    return {'kind': kind, 'body_target': None if target is None else {n: target[n] for n in NAMES},
            'description': kind, 'external_path_clear': True, 'human_clear': True,
            'confidence': .95, **changes}


def program(task='grasp', *, opened=False):
    approach = pose(shoulder_pan=2060)
    lifted = {**approach, 'elbow_flex': 2040}
    steps = ([step('open')] if opened else []) + [step('approach', approach), step('close'), step('verify_lift', lifted)]
    if task != 'grasp':
        moved = {**lifted, 'shoulder_pan': 2100}
        placed = {**moved, 'elbow_flex': 2000}
        steps += [step('transport', moved), step('lower', placed), step('release')]
    return {'task': task, 'status': 'ready', 'source_object': 'toy',
            'destination': '' if task == 'grasp' else 'block',
            'initial_jaw_empty': True, 'target_visible': True, 'path_description': 'clear test path',
            'unsupported_reason': '', 'steps': steps, 'observation': None, 'target_bbox': None}


class Model:
    def __init__(self):
        self.calibration = {n: {'range_min': 1000, 'range_max': 3000} for n in (*NAMES, 'gripper')}
        self.fk = SimpleNamespace(transforms=lambda raw: {'gripper_frame_link': self.tip(raw)}, joints=[])
        self.rejected_pan = None
    def tip(self, raw):
        value = np.eye(4)
        value[:3, 3] = [raw['shoulder_pan'] * .0001, 0, raw['elbow_flex'] * .0001]
        return value
    def validate_raw(self, raw, **kwargs):
        if any(not 1000 <= v <= 3000 for v in raw.values()):
            raise ValueError('range')
    def screen_segment(self, origin, target, **kwargs):
        return {'model_table_pass': target['shoulder_pan'] != self.rejected_pan}
    def floor_minimum(self, raw):
        return {'base_link': 0, 'gripper_link': .1}


class Power:
    def __init__(self):
        self.state = 'idle'
        self.contact = False
        self.events = []
    def _record(self):
        return {'contact_possible': self.contact, 'phase': self.state}
    def defer_holding(self, owner):
        if owner != 'owned' or not self.contact:
            raise RuntimeError('missing contact owner')
        self.events.append('defer')
        self.state = 'holding'
    def leave_pending(self):
        self.events.append('pending')
        self.state = 'blocked'
    def stop_contact(self, owner):
        self.events.append('stop_contact')
        self.state = 'blocked'


class Fixture:
    def __init__(self):
        self.model = Model()
        self.robot = SimpleNamespace(stop=threading.Event(),
            camera=SimpleNamespace(fresh=lambda: True, generation='wrist-1'),
            goals=pose(), calibration=self.model.calibration,
            gripper_power=Power(), pwm_owner=None, begin=Mock(), hold=Mock(),
            io=threading.RLock(), bias={n: 0 for n in (*NAMES, 'gripper')}, _write=Mock())
        self.current = pose()
        self.sequence = 0
        self.state = 'idle'
        self.latest = None
        self.record = Mock()
        self.events = []
        self.closed = False
        self.contact_check = {k: True for k in ('same_target', 'both_inner_pads_visible', 'target_between_inner_pads', 'closure_path_clear', 'human_clear')}
        self.support_check = {k: True for k in ('same_target', 'correct_destination', 'supported_by_destination', 'release_path_clear', 'human_clear')}
        self.placement_check = {k: True for k in ('same_target', 'correct_destination', 'resting_on_destination', 'visibly_stable', 'gripper_empty')}
        self.grasp_check = {k: True for k in ('same_object', 'between_fingers', 'clear_of_table', 'follows_gripper')}
        self.grasp_check.update(gripper_empty=False, confidence=.95, reason='test')
        self.brain = SimpleNamespace(_infer=Mock(side_effect=self.infer),
                                     verify_grasp=Mock(side_effect=lambda *a: copy.deepcopy(self.grasp_check)))
        self.open_empty = Mock(side_effect=self.open)
        self.close_candidate = Mock(side_effect=self.close)
        self.release = Mock(side_effect=self.free)
    def capture(self):
        self.sequence += 1
        return {'source': 'wrist', 'id': str(self.sequence), 'camera_generation': 'wrist-1',
                'capture_monotonic': op.time.monotonic(), 'sequence': self.sequence,
                'joints': dict(self.current), 'path': 'mocked'}
    def infer(self, prompt, frames, schema, **kwargs):
        self.events.append(schema.__name__)
        values = {op.ContactCheckpoint: self.contact_check, op.SupportCheckpoint: self.support_check,
                  op.PlacementVerification: self.placement_check}
        return schema.model_validate({**values[schema], 'confidence': .95, 'reason': 'test'})
    def open(self, frame, report, *, cancel=None):
        if cancel is not None and cancel.is_set():
            raise SessionCancelled('opening stopped')
        self.events.append('open')
        self.robot.pwm_owner = 'owned'
        self.robot.gripper_power.state = 'active'
        self.current['gripper'] = self.robot.goals['gripper'] = 1860
        return self.capture()
    def close(self, frame, report, cancel):
        self.events.append('close')
        self.closed = True
        self.state = 'closing'
        self.robot.pwm_owner = 'owned'
        self.robot.gripper_power.state = 'active'
        self.robot.gripper_power.contact = True
        self.current['gripper'] = 1810
        self.robot.goals['gripper'] = 1800
        return {'obstruction_candidate': True}
    def free(self, frame, report, cancel, *, support_verified=False):
        assert support_verified
        self.events.append('release')
        self.robot.gripper_power.contact = False
        self.robot.gripper_power.state = 'completed'
        self.robot.pwm_owner = None
        self.current['gripper'] = self.robot.goals['gripper'] = 1860
        self.state = 'idle'
        return {'completed': True}
    def execute(self, grasp, segment, frame, cancel, report, **kwargs):
        assert grasp is self
        assert segment['held_goals'] == self.robot.goals
        assert segment['measured_origin']['gripper'] == self.current['gripper']
        self.events.append('move')
        before = self.capture()
        self.current = dict(segment['joint_targets'])
        self.robot.goals = dict(segment['physical_goal_targets'])
        return {'before': before, 'after': self.capture(), 'elapsed': .2}


class ObjectProgramSchemaTests(unittest.TestCase):
    def test_target_box_is_visible_pixels_not_metric_dimensions(self):
        p = program(); p['target_bbox'] = {'x': .4, 'y': .1, 'w': .15, 'h': .5}
        self.assertEqual(op.ObjectProgram.model_validate(p).target_bbox.w, .15)
        for box in ({'x': .9, 'y': 0., 'w': .2, 'h': .2},
                    {'x': 0., 'y': 0., 'w': float('nan'), 'h': .2},
                    {'x': 0., 'y': 0., 'w': 0., 'h': .2}):
            p['target_bbox'] = box
            with self.assertRaises(ValidationError): op.ObjectProgram.model_validate(p)

    def test_model_response_schema_requires_every_nullable_field(self):
        schema = op.ObjectProgram.model_json_schema()
        for model in (schema, *schema.get('$defs', {}).values()):
            if model.get('type') == 'object':
                self.assertEqual(set(model['properties']), set(model['required']))
                self.assertFalse(model['additionalProperties'])

    def test_valid_grasp_place_and_stack(self):
        for task in ('grasp', 'place', 'stack'):
            self.assertEqual(op.ObjectProgram.model_validate(program(task)).task, task)

    def test_invalid_contact_and_release_order_rejected(self):
        variants = []
        p = program('stack'); p['steps'][2], p['steps'][3] = p['steps'][3], p['steps'][2]; variants.append(p)
        p = program('stack'); p['steps'].pop(1); variants.append(p)
        p = program('stack'); p['steps'].insert(4, step('open')); variants.append(p)
        p = program('stack'); p['steps'][-1] = step('close'); variants.append(p)
        p = program('grasp'); p['steps'].append(step('transport', pose(shoulder_pan=2200))); variants.append(p)
        for p in variants:
            with self.subTest(steps=[s['kind'] for s in p['steps']]), self.assertRaises(ValidationError):
                op.ObjectProgram.model_validate(p)

    def test_no_raw_gripper_target_or_false_clearance(self):
        for key, value in (('gripper', 1500), ('shoulder_pan', True)):
            p = program(); p['steps'][0]['body_target'][key] = value
            with self.assertRaises(ValidationError): op.ObjectProgram.model_validate(p)
        for key in ('external_path_clear', 'human_clear'):
            p = program(); p['steps'][0][key] = False
            with self.assertRaises(ValidationError): op.ObjectProgram.model_validate(p)

    def test_semantic_jaw_cannot_smuggle_body_target(self):
        p = program(); p['steps'][1]['body_target'] = {n: 2000 for n in NAMES}
        with self.assertRaises(ValidationError): op.ObjectProgram.model_validate(p)

    def test_unsupported_has_no_executable_steps(self):
        p = program(); p.update(status='unsupported', unsupported_reason='not visible')
        with self.assertRaises(ValidationError): op.ObjectProgram.model_validate(p)
        p['steps'] = []
        self.assertEqual(op.ObjectProgram.model_validate(p).status, 'unsupported')


class ObjectProgramExecutionTests(unittest.TestCase):
    def setUp(self):
        self.g = Fixture()
        self.cancel = threading.Event()
        self.report = Mock()
        self.real_execute = mp.execute_prepared_segment
        self.patches = [patch.object(op, 'unchanged_wrist', return_value={}),
                        patch.object(mp, 'execute_prepared_segment', side_effect=self.g.execute),
                        patch.object(op, 'build_observation_candidates', return_value=[]),
                        patch.object(op, 'filter_observation_novelty', side_effect=lambda values, *a, **k: values)]
        self.mocks = [p.start() for p in self.patches]
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patches)])

    def run_program(self, task='grasp', value=None, report_stage=None):
        with patch.object(op, 'plan_object_program', return_value=op.ObjectProgram.model_validate(value or program(task))) as planner:
            try:
                return op.run_object_program(self.g, 'toy on block', self.report, self.cancel, task=task, report_stage=report_stage)
            finally:
                self.assertEqual(planner.call_count, 1)

    def test_successful_grasp_has_one_plan_then_contact_and_grasp_check(self):
        result = self.run_program()
        self.assertTrue(result['grasp_confirmed'])
        self.assertTrue(result['completed'])
        self.assertEqual(self.g.events, ['move', 'ContactCheckpoint', 'close', 'move'])
        self.assertEqual(self.g.brain.verify_grasp.call_count, 1)
        self.assertEqual(self.g.state, 'holding')
        self.assertEqual(self.g.robot.pwm_owner, 'owned')
        self.assertFalse(result['held_object_mesh_certified'])

    def test_missing_v2_jaw_skips_candidates_cloud_and_motion_before_target_views(self):
        self.g.robot.config = {'spatial_backend': 'v2', 'motion_backend': 'task_space_v2'}
        packet = SimpleNamespace(metric_calibrated=True, jaw_geometry=None,
            calibration=SimpleNamespace(metric_calibrated=True),
            to_dict=lambda: {'validation_scope_reason': ''})
        with patch('server.spatial_runtime.observation_packet', return_value=packet), \
                patch('server.spatial_object_program.observation_packet', return_value=packet), \
                patch('server.spatial_object_program.build_object_geometry', return_value={'status': 'unmeasured'}):
            with self.assertRaisesRegex(op.ObjectProgramBlocked, '抓取設定尚未完成.*夾墊'):
                op.run_object_program(self.g, 'toy', self.report, self.cancel)
        self.mocks[2].assert_not_called()
        self.mocks[1].assert_not_called()
        self.g.brain._infer.assert_not_called()
        self.g.robot._write.assert_not_called()
        self.g.close_candidate.assert_not_called()
        self.assertEqual(self.g.latest['observations'], [])
        self.assertEqual(self.g.latest['spatial_v2']['required_setups'], ['jaw_calibration'])
        self.assertFalse(self.g.robot.stop.is_set())

    def test_successful_stack_requires_support_then_release_then_final_check(self):
        result = self.run_program('stack')
        self.assertTrue(result['stack_confirmed'])
        self.assertFalse(result['place_confirmed'])
        self.assertEqual(self.g.events[-3:], ['SupportCheckpoint', 'release', 'PlacementVerification'])
        self.assertIsNone(self.g.robot.pwm_owner)
        self.assertEqual(self.mocks[1].call_count, 4)
        self.assertFalse(self.mocks[1].call_args_list[0].kwargs['continuation'])
        self.assertTrue(all(c.kwargs['continuation'] for c in self.mocks[1].call_args_list[1:]))

    def test_successful_place_receipt(self):
        result = self.run_program('place')
        self.assertTrue(result['place_confirmed'])
        self.assertFalse(result['stack_confirmed'])

    def test_entire_program_screened_before_first_open_or_move(self):
        self.g.model.rejected_pan = 2100
        with self.assertRaises(ValueError): self.run_program('stack', program('stack', opened=True))
        self.assertEqual(self.g.events, [])
        self.assertFalse(self.g.robot.stop.is_set())

    def test_large_later_segment_blocks_all_earlier_operations(self):
        p = program('stack'); p['steps'][3]['body_target']['shoulder_pan'] = 2300
        with self.assertRaisesRegex(ValueError, '180'): self.run_program('stack', p)
        self.assertEqual(self.g.events, [])

    def test_bad_verification_lift_blocks_before_approach(self):
        p = program(); p['steps'][2]['body_target']['elbow_flex'] = 1990
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '向上'): self.run_program(value=p)
        self.assertEqual(self.g.events, [])

    def test_failed_contact_checkpoint_never_closes(self):
        self.g.contact_check['target_between_inner_pads'] = False
        with self.assertRaises(op.ObjectProgramBlocked): self.run_program()
        self.g.close_candidate.assert_not_called()
        self.assertEqual(self.mocks[1].call_count, 1)
        self.assertEqual(self.g.latest['status'], 'blocked')

    def test_failed_grasp_stops_queued_transport_release(self):
        self.g.grasp_check['clear_of_table'] = False
        with self.assertRaises(op.ObjectProgramBlocked): self.run_program('stack')
        self.assertEqual(self.mocks[1].call_count, 2)
        self.g.release.assert_not_called()
        self.assertTrue(self.g.robot.gripper_power.contact)
        self.assertEqual(self.g.robot.pwm_owner, 'owned')
        self.assertEqual(self.g.state, 'holding_unverified')
        self.assertFalse(self.g.latest['completed'])

    def test_motor_obstruction_alone_is_not_grasp_confirmation(self):
        self.g.grasp_check['follows_gripper'] = False
        with self.assertRaises(op.ObjectProgramBlocked): self.run_program()
        self.assertFalse(self.g.latest['grasp_confirmed'])

    def test_missing_destination_support_keeps_low_power_contact(self):
        self.g.support_check['supported_by_destination'] = False
        with self.assertRaises(op.ObjectProgramBlocked): self.run_program('stack')
        self.g.release.assert_not_called()
        self.assertTrue(self.g.robot.gripper_power.contact)
        self.assertFalse(self.g.latest['stack_confirmed'])

    def test_released_object_without_final_visual_success_is_not_completed_stack(self):
        self.g.placement_check['resting_on_destination'] = False
        with self.assertRaises(op.ObjectProgramBlocked): self.run_program('stack')
        self.g.release.assert_called_once()
        self.assertFalse(self.g.latest['stack_confirmed'])
        self.assertFalse(self.g.latest['completed'])
        self.assertEqual(self.g.state, 'idle')

    def test_cancel_after_contact_preserves_owned_contact(self):
        def verify(*args):
            self.cancel.set()
            raise SessionCancelled('test cancelled')
        self.g.brain.verify_grasp.side_effect = verify
        with self.assertRaises(SessionCancelled): self.run_program('stack')
        self.g.release.assert_not_called()
        self.assertIn('stop_contact', self.g.robot.gripper_power.events)
        self.assertTrue(self.g.robot.gripper_power.contact)
        self.assertEqual(self.g.latest['status'], 'cancelled')

    def test_verification_transport_failure_keeps_contact_no_release(self):
        self.g.brain.verify_grasp.side_effect = RuntimeError('model unavailable')
        with self.assertRaisesRegex(RuntimeError, 'unavailable'): self.run_program('place')
        self.g.release.assert_not_called()
        self.assertTrue(self.g.robot.gripper_power.contact)
        self.assertEqual(self.g.latest['status'], 'failed')

    def test_unsupported_plan_does_not_stop_idle_arm(self):
        p = program(); p.update(status='unsupported', steps=[], unsupported_reason='target is hidden')
        with self.assertRaisesRegex(op.ObjectProgramBlocked, 'hidden'): self.run_program(value=p)
        self.assertEqual(self.g.events, [])
        self.assertFalse(self.g.robot.stop.is_set())
        self.assertEqual(self.g.latest['status'], 'unsupported')

    def test_postclose_segments_bind_measured_jaw_not_open_jaw(self):
        self.run_program('place')
        segments = [c.args[1] for c in self.mocks[1].call_args_list]
        self.assertEqual(segments[0]['measured_origin']['gripper'], 1860)
        for segment in segments[1:]:
            self.assertEqual(segment['measured_origin']['gripper'], 1810)
            self.assertEqual(segment['physical_goal_targets']['gripper'], 1800)

    def test_cloud_milestone_waits_do_not_consume_motor_budget(self):
        clock = [0.0]
        class ClockEvent(threading.Event):
            def wait(self, timeout=None):
                clock[0] += timeout or 0
                return self.is_set()
        self.cancel = ClockEvent()
        original_execute = self.g.execute
        original_infer = self.g.infer
        original_close = self.g.close
        def execute(*args, **kwargs):
            clock[0] += 2
            return original_execute(*args, **kwargs)
        def infer(*args, **kwargs):
            clock[0] += 40
            return original_infer(*args, **kwargs)
        def close(*args, **kwargs):
            clock[0] += 3
            return original_close(*args, **kwargs)
        self.mocks[1].side_effect = execute
        self.g.brain._infer.side_effect = infer
        self.g.close_candidate.side_effect = close
        with patch.object(op, 'time', SimpleNamespace(monotonic=lambda: clock[0])):
            result = self.run_program('stack')
        self.assertTrue(result['completed'])
        self.assertEqual(result['physical_seconds_charged'], 23)
        self.assertGreater(clock[0], 120)
        self.assertAlmostEqual(result['steps'][-1]['observation_interval_seconds'], .35)

    def test_insufficient_remaining_jaw_budget_never_starts_closing(self):
        clock = [0.0]
        original_execute = self.g.execute
        def execute(*args, **kwargs):
            clock[0] += 19
            return original_execute(*args, **kwargs)
        self.mocks[1].side_effect = execute
        with patch.object(op, 'time', SimpleNamespace(monotonic=lambda: clock[0])):
            with self.assertRaisesRegex(op.ObjectProgramBlocked, '時間不足'):
                self.run_program('stack')
        self.g.close_candidate.assert_not_called()
        self.g.release.assert_not_called()
        self.assertEqual(self.g.latest['physical_seconds_charged'], 19)

    def test_motor_deadline_does_not_latch_the_user_cancel_event(self):
        clock = [1.0]
        with patch.object(op, 'time', SimpleNamespace(monotonic=lambda: clock[0])):
            motor_cancel = op._PhysicalCancel(self.cancel, 2)
            self.assertFalse(motor_cancel.is_set())
            clock[0] = 2
            self.assertTrue(motor_cancel.is_set())
            self.assertFalse(self.cancel.is_set())

    def test_open_step_receives_live_user_cancellation_and_blocks_following_steps(self):
        def opening(frame, report, *, cancel):
            self.assertIs(cancel.cancel, self.cancel)
            self.assertFalse(cancel.is_set())
            self.cancel.set()
            return self.g.open(frame, report, cancel=cancel)
        self.g.open_empty.side_effect = opening
        with self.assertRaisesRegex(SessionCancelled, 'opening stopped'):
            self.run_program(value=program(opened=True))
        self.g.open_empty.assert_called_once()
        self.g.close_candidate.assert_not_called()
        self.mocks[1].assert_not_called()
        self.assertEqual(self.g.latest['status'], 'cancelled')
        self.assertFalse(self.g.latest['completed'])

    def test_open_deadline_uses_budget_remaining_after_observation(self):
        clock = [100.0]
        candidate = {'id': 'jaw_open', 'label': '張開空夾爪再看'}
        observing = program()
        observing.update(status='observe', steps=[], observation={
            'candidate_id': 'jaw_open', 'gripper_empty': True, 'human_clear': True,
            'clear_path': True, 'confidence': .95, 'reason': '看清物件'})
        def observe(*args, **kwargs):
            self.assertEqual(kwargs['deadline'], 130)
            before = self.g.capture()
            clock[0] += 8
            return {'completed': True, 'before': before, 'after': self.g.capture(),
                    'physical_seconds_charged': 8}
        def opening(frame, report, *, cancel):
            self.assertEqual(clock[0], 108)
            self.assertEqual(cancel.deadline, 130)
            self.assertFalse(cancel.is_set())
            clock[0] = cancel.deadline
            self.assertTrue(cancel.is_set())
            self.assertFalse(self.cancel.is_set())
            return self.g.open(frame, report, cancel=cancel)
        self.g.open_empty.side_effect = opening
        with patch.object(op, 'time', SimpleNamespace(monotonic=lambda: clock[0])), \
                patch.object(op, 'build_observation_candidates', return_value=[candidate]), \
                patch.object(op, 'execute_observation', side_effect=observe), \
                patch.object(op, 'plan_object_program', side_effect=[
                    op.ObjectProgram.model_validate(observing),
                    op.ObjectProgram.model_validate(program(opened=True))]):
            with self.assertRaisesRegex(SessionCancelled, 'opening stopped'):
                op.run_object_program(self.g, 'toy', self.report, self.cancel)
        self.assertEqual(self.g.latest['physical_seconds_charged'], 30)
        self.assertEqual(len(self.g.latest['observations']), 1)
        self.g.close_candidate.assert_not_called()
        self.mocks[1].assert_not_called()
        self.assertFalse(self.g.latest['completed'])

    def test_failed_observation_still_charges_elapsed_physical_time(self):
        clock = [100.0]
        candidate = {'id': 'jaw_open', 'label': '張開空夾爪再看'}
        observing = program()
        observing.update(status='observe', steps=[], observation={
            'candidate_id': 'jaw_open', 'gripper_empty': True, 'human_clear': True,
            'clear_path': True, 'confidence': .95, 'reason': '看清物件'})
        def observe(*args, **kwargs):
            clock[0] += 6.25
            raise RuntimeError('opening guard failed after movement')
        with patch.object(op, 'time', SimpleNamespace(monotonic=lambda: clock[0])), \
                patch.object(op, 'build_observation_candidates', return_value=[candidate]), \
                patch.object(op, 'execute_observation', side_effect=observe), \
                patch.object(op, 'plan_object_program', return_value=op.ObjectProgram.model_validate(observing)):
            with self.assertRaisesRegex(RuntimeError, 'guard failed'):
                op.run_object_program(self.g, 'toy', self.report, self.cancel)
        self.assertEqual(self.g.latest['physical_seconds_charged'], 6.25)
        self.assertEqual(self.g.latest['status'], 'failed')
        self.assertEqual(self.g.latest['observations'], [])
        self.assertFalse(self.g.latest['completed'])
        self.g.close_candidate.assert_not_called()
        self.mocks[1].assert_not_called()

    def test_d435_cannot_enter_planner(self):
        frame = self.g.capture(); frame['source'] = 'engineering-observer'
        with self.assertRaisesRegex(ValueError, '腕部'):
            op.plan_object_program(self.g, 'toy', frame, self.cancel)
        self.g.brain._infer.assert_not_called()

    def test_planning_payload_has_measured_limits_and_geometry(self):
        self.g.brain._infer.side_effect = None
        self.g.brain._infer.return_value = op.ObjectProgram.model_validate(program())
        op.plan_object_program(self.g, 'toy', self.g.capture(), self.cancel)
        prompt = self.g.brain._infer.call_args.args[0]
        self.assertIn('onboard_limits_raw', prompt)
        self.assertIn('cad_link_transforms_base_m', prompt)
        self.assertIn('measured_joints_raw', prompt)
        self.assertIn('wrist-only', prompt)
        self.assertIn('active_joint_actual_tracking_ticks', prompt)
        self.assertIn('stationary_joint_actual_tracking_ticks', prompt)
        self.assertIn('continued_stationary_joint_start_tolerance_ticks', prompt)
        self.assertIn('body_held_goal_bias_ticks', prompt)
        self.assertIn('physical_held_goals_raw', prompt)

    def test_planner_receives_controller_rejection_reasons_without_claiming_an_obstacle(self):
        self.g.brain._infer.side_effect = None
        self.g.brain._infer.return_value = op.ObjectProgram.model_validate(program())
        op.plan_object_program(self.g, 'toy', self.g.capture(), self.cancel,
            observation_diagnostics={'rejection_counts': {'Onboard range/envelope exceeded: elbow_flex': 4},
                'rejections': [{'candidate_id': 'pitch_negative', 'reason': 'large detailed log'}]})
        prompt = self.g.brain._infer.call_args.args[0]
        context = __import__('json').loads(prompt.split('INPUT_DATA_JSON\n')[1])
        diagnostics = context['observation_candidate_diagnostics']
        self.assertEqual(diagnostics['presented_candidate_ids'], [])
        self.assertEqual(diagnostics['rejection_counts']['Onboard range/envelope exceeded: elbow_flex'], 4)
        self.assertNotIn('rejections', diagnostics)
        self.assertIn('does not prove that the physical workspace has no room', prompt)

    def test_depth_is_anchored_to_previous_target_image_then_passed_to_next_plan(self):
        candidate = {'id': 'camera_left', 'label': '側移觀察'}
        observing = program()
        bbox = {'x': .4, 'y': .1, 'w': .15, 'h': .5}
        observing.update(status='observe', steps=[], target_bbox=bbox, observation={
            'candidate_id': candidate['id'], 'gripper_empty': True, 'human_clear': True,
            'clear_path': True, 'confidence': .85, 'reason': '用側面視差估距'})
        calls = []
        def planning(*args, **kwargs):
            calls.append((args[2], kwargs))
            return op.ObjectProgram.model_validate(observing if len(calls) == 1 else program())
        def observe(*args, **kwargs):
            return {'completed': True, 'before': args[2], 'after': self.g.capture(),
                    'physical_seconds_charged': 2.}
        evidence = {'source': 'wrist-only', 'status': 'insufficient', 'reason': 'test'}
        with patch.object(op, 'build_observation_candidates', return_value=[candidate]), \
                patch.object(op, 'execute_observation', side_effect=observe), \
                patch.object(op, 'filter_observation_framing', side_effect=lambda values, *a, **k: values), \
                patch.object(op, 'estimate_active_depth', return_value=evidence) as estimate, \
                patch.object(op, 'plan_object_program', side_effect=planning):
            result = op.run_object_program(self.g, 'toy', self.report, self.cancel)
        original_frame, _ = calls[0]
        next_frame, next_options = calls[1]
        depth_frames = estimate.call_args.args[0]
        self.assertEqual(depth_frames[0], original_frame)
        self.assertLess(depth_frames[-1]['sequence'], next_frame['sequence'])
        self.assertEqual(estimate.call_args.kwargs['roi_frame_id'], original_frame['id'])
        self.assertEqual(estimate.call_args.kwargs['roi'], bbox)
        self.assertEqual(next_options['depth_evidence']['stationary_refresh']['current_frame_id'], next_frame['id'])
        self.assertEqual(result['active_depth']['source'], 'wrist-only')

    def test_tracked_pixels_frame_candidates_before_cloud_without_authorizing_motion(self):
        candidate = {'id': 'camera_left', 'label': '側移觀察'}
        observing = program()
        bbox = {'x': .4, 'y': .1, 'w': .15, 'h': .5}
        tracked = {'x': .3, 'y': .2, 'w': .16, 'h': .5}
        observing.update(status='observe', steps=[], target_bbox=bbox, observation={
            'candidate_id': candidate['id'], 'gripper_empty': True, 'human_clear': True,
            'clear_path': True, 'confidence': .85, 'reason': '比較側面'})
        def estimate(frames, model, **kwargs):
            return {'status': 'insufficient', 'source': 'wrist-only',
                    'current_frame_id': frames[-1]['id'],
                    'provenance': {'camera_generation': frames[-1]['camera_generation']},
                    'tracked_roi_current': {'status': 'tracked', 'source': 'wrist-only',
                        'framing_hint_only': True, 'current_frame_id': frames[-1]['id'], 'bbox': tracked}}
        def observe(*args, **kwargs):
            return {'completed': True, 'after': self.g.capture(), 'physical_seconds_charged': 2.}
        with patch.object(op, 'build_observation_candidates', return_value=[candidate]) as builder, \
                patch.object(op, 'filter_observation_framing', side_effect=lambda values, *a, **k: values), \
                patch.object(op, 'execute_observation', side_effect=observe) as execute, \
                patch.object(op, 'unchanged_wrist', return_value={'changed_fraction': 0., 'p99_difference': 0.}), \
                patch.object(op, 'estimate_active_depth', side_effect=estimate), \
                patch.object(op, 'plan_object_program', side_effect=[op.ObjectProgram.model_validate(p)
                    for p in (observing, program())]) as planner:
            result = op.run_object_program(self.g, 'toy', self.report, self.cancel)
        self.assertEqual(result['planning_attempts'], 2)
        self.assertEqual(builder.call_args_list[-1].kwargs['target_bbox'], tracked)
        self.assertEqual(planner.call_count, 2)
        execute.assert_called_once()

    def test_framing_hint_requires_exact_depth_and_stationary_refresh_identity(self):
        frame = self.g.capture()
        depth = {'source': 'wrist-only', 'current_frame_id': 'depth-view',
            'provenance': {'camera_generation': frame['camera_generation']},
            'stationary_refresh': {'current_frame_id': frame['id'], 'image_comparison': {'changed_fraction': 0}},
            'tracked_roi_current': {'status': 'tracked', 'source': 'wrist-only', 'framing_hint_only': True,
                'current_frame_id': 'depth-view', 'bbox': {'x': .3, 'y': .2, 'w': .2, 'h': .5}}}
        self.assertIsNotNone(op._tracked_framing_hint(depth, frame))
        for field, bad in (('current_frame_id', 'other'), ('source', 'engineering-observer'),
                           ('stationary_refresh', {}), ('provenance', {'camera_generation': 'other'})):
            self.assertIsNone(op._tracked_framing_hint({**depth, field: bad}, frame))

    def test_cropped_view_is_replanned_before_motion_without_substituting_approval(self):
        bad = {'id': 'cropped', 'label': '後退'}
        good = {'id': 'lateral', 'label': '側移'}
        first = program()
        first.update(status='observe', steps=[], target_bbox={'x': .4, 'y': .2, 'w': .2, 'h': .6},
                     observation={'candidate_id': 'cropped', 'gripper_empty': True, 'human_clear': True,
                                  'clear_path': True, 'confidence': .8, 'reason': '觀察'})
        second = copy.deepcopy(first); second['observation']['candidate_id'] = 'lateral'
        def observe(*args, **kwargs):
            self.assertEqual(args[1]['id'], 'lateral')
            self.assertEqual(kwargs['verdict']['candidate_id'], 'lateral')
            return {'completed': True, 'after': self.g.capture(), 'physical_seconds_charged': 2.}
        with patch.object(op, 'build_observation_candidates', side_effect=[[bad], [good], []]) as builder, \
                patch.object(op, 'filter_observation_framing', side_effect=[[], [good]]), \
                patch.object(op, 'execute_observation', side_effect=observe) as execute, \
                patch.object(op, 'estimate_active_depth', return_value={'status': 'insufficient'}), \
                patch.object(op, 'plan_object_program', side_effect=[op.ObjectProgram.model_validate(p)
                    for p in (first, second, program())]) as planner:
            result = op.run_object_program(self.g, 'toy', self.report, self.cancel)
        execute.assert_called_once()
        self.assertEqual(result['planning_attempts'], 3)
        self.assertEqual(len(result['observations']), 1)
        self.assertEqual(planner.call_args_list[1].kwargs['observation_candidates'], [good])
        self.assertEqual(planner.call_args_list[1].kwargs['remaining_seconds'], op.MAX_PHYSICAL_SECONDS)
        self.assertEqual(builder.call_args_list[1].kwargs['target_bbox'], first['target_bbox'])

    def test_unsupported_localization_can_offer_recenter_but_needs_new_approval(self):
        bbox = {'x': .4, 'y': .5, 'w': .2, 'h': .5}
        blocked = program()
        blocked.update(status='unsupported', steps=[], target_bbox=bbox,
                       unsupported_reason='目前的角度無法換位')
        candidate = {'id': 'pitch_positive', 'label': '重新對準'}
        observing = copy.deepcopy(blocked)
        observing.update(status='observe', unsupported_reason='', observation={
            'candidate_id': 'pitch_positive', 'gripper_empty': True, 'human_clear': True,
            'clear_path': True, 'confidence': .8, 'reason': '先把物件移回畫面中央'})
        def execute(*args, **kwargs):
            self.assertEqual(kwargs['verdict']['candidate_id'], 'pitch_positive')
            return {'completed': True, 'after': self.g.capture(), 'physical_seconds_charged': 2.}
        with patch.object(op, 'build_observation_candidates', side_effect=[[], [candidate], []]), \
                patch.object(op, 'filter_observation_framing', side_effect=lambda values, *a, **k: values), \
                patch.object(op, 'execute_observation', side_effect=execute) as moving, \
                patch.object(op, 'estimate_active_depth', return_value={'status': 'insufficient'}), \
                patch.object(op, 'plan_object_program', side_effect=[op.ObjectProgram.model_validate(p)
                    for p in (blocked, observing, program())]) as planner:
            result = op.run_object_program(self.g, 'toy', self.report, self.cancel)
        self.assertEqual(planner.call_count, 3)
        moving.assert_called_once()
        self.assertEqual(result['observations'][0]['completed'], True)

    def test_planner_rejects_stale_future_and_changed_camera_frames(self):
        for change in ({'capture_monotonic': op.time.monotonic() - 6},
                       {'capture_monotonic': op.time.monotonic() + 30},
                       {'capture_monotonic': float('nan')},
                       {'camera_generation': 'old-camera'}, {'sequence': 0}):
            with self.subTest(change=change):
                frame = self.g.capture(); frame.update(change)
                with self.assertRaises(ValueError):
                    op.plan_object_program(self.g, 'toy', frame, self.cancel)
        self.g.brain._infer.assert_not_called()

    def test_planner_requires_current_live_camera(self):
        frame = self.g.capture()
        self.g.robot.camera.fresh = lambda: False
        with self.assertRaises(RuntimeError):
            op.plan_object_program(self.g, 'toy', frame, self.cancel)
        self.g.brain._infer.assert_not_called()

    def test_bad_query_rejected_before_begin(self):
        for query in ('', '   ', 'x' * 501, None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                op.run_object_program(self.g, query, self.report, self.cancel)
        self.g.robot.begin.assert_not_called()

    def test_final_placement_uses_two_separated_frames(self):
        result = self.run_program('stack')
        calls = [c for c in self.g.brain._infer.call_args_list if c.args[2] is op.PlacementVerification]
        self.assertEqual(len(calls), 1)
        frames = calls[0].args[1]
        self.assertEqual(len(frames), 2)
        self.assertGreater(frames[1]['sequence'], frames[0]['sequence'])
        self.assertGreaterEqual(frames[1]['capture_monotonic'] - frames[0]['capture_monotonic'], .3)
        self.assertLess(result['steps'][-1]['observation_interval_seconds'], 2)

    def test_duplicate_postrelease_frame_is_not_stability_evidence(self):
        first = self.g.capture()
        self.g.capture = Mock(return_value=dict(first))
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '不同時間'):
            op._placement_observations(self.g, first, self.cancel)
        self.g.brain._infer.assert_not_called()

    def test_postrelease_arm_drift_blocks_stability_verification(self):
        first = self.g.capture()
        self.g.current['shoulder_pan'] += 5
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '手臂還在變動'):
            op._placement_observations(self.g, first, self.cancel)

    def test_prior_moving_axis_cannot_inherit_tolerance_when_next_fixed(self):
        original_execute = self.g.execute
        def execute(*args, **kwargs):
            result = original_execute(*args, **kwargs)
            self.g.current['shoulder_pan'] += 6
            result['after'] = self.g.capture()
            return result
        self.mocks[1].side_effect = execute
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '固定的關節'):
            self.run_program('stack')
        self.assertEqual(self.mocks[1].call_count, 1)
        self.g.release.assert_not_called()

    def test_optional_phase_reporting_preserves_one_argument_messages(self):
        stages = []
        self.run_program(report_stage=stages.append)
        self.assertEqual(stages, ['planning', 'moving', 'moving', 'verifying', 'verifying',
                                  'moving', 'moving', 'moving', 'verifying'])
        self.assertTrue(all(len(c.args) == 1 for c in self.report.call_args_list))

    def test_phase_precedes_message_publishing_and_grasp_inference(self):
        current = ['unset']
        messages = []
        def phase(stage):
            current[0] = stage
        self.report = lambda message: messages.append((current[0], message))
        def verify(*args):
            self.assertEqual(current[0], 'verifying')
            self.assertEqual(messages[-1], ('verifying', '確認物件真的離桌、跟著夾爪抬起'))
            return copy.deepcopy(self.g.grasp_check)
        self.g.brain.verify_grasp.side_effect = verify
        self.run_program(report_stage=phase)
        descriptions = [(stage, message.split(' · ')[-1]) for stage, message in messages if ' · ' in message]
        self.assertEqual(descriptions, [('moving', 'approach'), ('verifying', 'close'), ('moving', 'verify_lift')])

    def use_real_segment_ledger_with_stale_bias(self):
        """Run the real segment adoption; emulate only motor transport."""
        self.g.robot.goals = {n: self.g.current[n] + (5 if n in NAMES else 0)
                              for n in (*NAMES, 'gripper')}
        self.g.robot.bias = {n: 0 for n in (*NAMES, 'gripper')}
        def glide(targets, **kwargs):
            for name, target in targets.items():
                self.g.robot.goals[name] = target + self.g.robot.bias[name]
                self.g.current[name] = target
            kwargs['guard'](self.g.current)
            kwargs['measured_guard'](self.g.current)
            return dict(self.g.current)
        self.g.robot.glide = Mock(side_effect=glide)
        self.mocks[1].side_effect = self.real_execute
        guard = patch('server.grasp.unchanged_wrist', return_value={})
        guard.start(); self.addCleanup(guard.stop)

    def test_open_before_first_move_binds_plan_bias_before_pwm_ownership(self):
        self.use_real_segment_ledger_with_stale_bias()
        original_open = self.g.open
        def open_empty(*args, **kwargs):
            self.assertEqual(self.g.robot.bias, {n: 5 if n in NAMES else 0 for n in (*NAMES, 'gripper')})
            self.assertIsNone(self.g.robot.pwm_owner)
            return original_open(*args, **kwargs)
        self.g.open_empty.side_effect = open_empty
        result = self.run_program(value=program(opened=True))
        self.assertTrue(result['grasp_confirmed'])
        self.assertEqual(self.g.robot.glide.call_count, 2)
        self.assertEqual(self.g.robot.bias['gripper'], 0)

    def test_close_first_then_lift_preserves_new_plan_bias(self):
        self.use_real_segment_ledger_with_stale_bias()
        p = program()
        p['steps'] = [step('close'), step('verify_lift', pose(elbow_flex=2040))]
        original_close = self.g.close
        def close_candidate(*args, **kwargs):
            self.assertEqual(self.g.robot.bias['elbow_flex'], 5)
            self.assertIsNone(self.g.robot.pwm_owner)
            return original_close(*args, **kwargs)
        self.g.close_candidate.side_effect = close_candidate
        result = self.run_program(value=p)
        self.assertTrue(result['grasp_confirmed'])
        self.assertEqual(self.g.robot.glide.call_count, 1)
        self.assertEqual(self.g.robot.bias['gripper'], 0)

    def test_initial_anchor_is_read_only_for_goals_and_pwm(self):
        self.use_real_segment_ledger_with_stale_bias()
        self.g.robot.bias['gripper'] = 5  # An obsolete ledger must not become jaw compensation.
        frame = self.g.capture()
        prepared = op.prepare_object_program(self.g, program(opened=True), frame)
        goals = dict(self.g.robot.goals)
        power = copy.deepcopy(self.g.robot.gripper_power.__dict__)
        op._bind_initial_body_anchor(self.g, prepared, frame, self.cancel)
        self.assertEqual(self.g.robot.goals, goals)
        self.assertEqual(self.g.robot.gripper_power.__dict__, power)
        self.assertIsNone(self.g.robot.pwm_owner)
        self.assertEqual(self.g.robot.bias, {n: 5 if n in NAMES else 0 for n in (*NAMES, 'gripper')})
        self.g.robot._write.assert_not_called()
        self.g.robot.glide.assert_not_called()

    def test_initial_anchor_rejects_changed_held_goal_without_rebasing(self):
        frame = self.g.capture()
        prepared = op.prepare_object_program(self.g, program(), frame)
        original_bias = dict(self.g.robot.bias)
        self.g.robot.goals['elbow_flex'] += 1
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '保持目標已改變'):
            op._bind_initial_body_anchor(self.g, prepared, frame, self.cancel)
        self.assertEqual(self.g.robot.bias, original_bias)
        self.g.robot._write.assert_not_called()

    def test_later_segment_still_rejects_modified_compensation_ledger(self):
        self.use_real_segment_ledger_with_stale_bias()
        calls = [0]
        def execute(*args, **kwargs):
            value = self.real_execute(*args, **kwargs)
            calls[0] += 1
            if calls[0] == 1:
                self.g.robot.bias['shoulder_pan'] += 1
            return value
        self.mocks[1].side_effect = execute
        with self.assertRaisesRegex(RuntimeError, '保持補償已改變'):
            self.run_program()
        self.assertEqual(self.g.robot.glide.call_count, 1)
        self.assertEqual(self.g.robot.bias['shoulder_pan'], 6)
        self.g.release.assert_not_called()

    def test_initial_anchor_never_rebases_an_existing_contact_session(self):
        frame = self.g.capture()
        prepared = op.prepare_object_program(self.g, program(), frame)
        self.g.robot.gripper_power.contact = True
        self.g.robot.pwm_owner = 'owned'
        bias = dict(self.g.robot.bias)
        with self.assertRaisesRegex(op.ObjectProgramBlocked, '待確認的物件'):
            op._bind_initial_body_anchor(self.g, prepared, frame, self.cancel)
        self.assertEqual(self.g.robot.bias, bias)
        self.assertTrue(self.g.robot.gripper_power.contact)
        self.g.robot._write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
