"""No-hardware tests of exact-plan local motion and monitored limits."""
from copy import deepcopy
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from server.gripper_power import SessionCancelled
from server.local_cartesian import ALL, NAMES, default_model
from server.motion_program import (prepare_joint_segment, execute_prepared_segment,
                                   rebind_segment_jaw, _tube)


class Model:
    table_pass = True
    floor_height = .1
    calibration = {n: {'range_min': 20, 'range_max': 4070} for n in ALL}
    def validate_raw(self, raw, **kwargs):
        if any(not 20 <= v <= 4070 for v in raw.values()):
            raise ValueError('range')
    def screen_segment(self, start, target, **kwargs):
        return {'model_table_pass': self.table_pass,
                'actual_tracking_envelope_ticks': kwargs['uncertainty_ticks']}
    def floor_minimum(self, pose):
        return {'base_link': -.002, 'arm': self.floor_height}
    def tip(self, pose):
        result = np.eye(4)
        result[:3, 3] = np.array([pose['shoulder_pan'], pose['elbow_flex'], pose['wrist_flex']]) * .0001
        return result


class Camera:
    generation = 'one'
    okay = True
    def fresh(self):
        return self.okay


class Robot:
    def __init__(self, raw, held):
        self.position, self.goals = dict(raw), dict(held)
        self.bias = {n: held[n] - raw[n] if n in NAMES else 0 for n in ALL}
        self.io = threading.RLock()
        self.camera = Camera()
        self.stop = threading.Event()
        self.pwm_owner = None
        self.calls = []
        self.hold_calls = 0
        self.callback = None
        self.endpoint_error = {}
    def begin(self, **kwargs):
        assert kwargs == {'allow_rearm': False}
    def hold(self):
        self.hold_calls += 1
    def glide(self, targets, **kwargs):
        self.calls.append((dict(targets), kwargs))
        assert 'gripper' not in targets
        assert kwargs['allow_rearm'] is False
        assert kwargs['feedback'] is False
        kwargs['guard'](self.position)
        if self.callback:
            self.callback(self, kwargs)
        self.goals.update({n: v + self.bias[n] for n, v in targets.items()})
        self.position.update({n: v + self.endpoint_error.get(n, 0) for n, v in targets.items()})
        kwargs['measured_guard'](self.position)
        kwargs['guard'](self.position)
        kwargs['observe']({})


class Grasp:
    def __init__(self, raw, held):
        self.model = Model()
        self.robot = Robot(raw, held)
        self.frames = 0
        self.events = []
    def capture(self):
        self.frames += 1
        return {'source': 'wrist', 'camera_generation': self.robot.camera.generation,
                'joints': dict(self.robot.position), 'capture_monotonic': self.frames + 1}
    def record(self, event):
        self.events.append(event)


class MotionProgramTests(unittest.TestCase):
    def setUp(self):
        self.raw = {n: 2000 for n in ALL}
        self.held = {**self.raw, 'elbow_flex': 1980}
        self.target = {**self.raw, 'elbow_flex': 2100}
        self.grasp = Grasp(self.raw, self.held)
        self.cancel = threading.Event()
        self.ref = {'source': 'wrist', 'camera_generation': 'one', 'joints': self.raw,
                    'capture_monotonic': 0}
        self.scene = patch('server.grasp.unchanged_wrist', return_value={'unchanged': True})
        self.scene.start()
        self.addCleanup(self.scene.stop)
    def prepare(self, **kwargs):
        return prepare_joint_segment(self.grasp.model, self.raw, self.held, self.target, **kwargs)
    def execute(self, segment=None, **kwargs):
        return execute_prepared_segment(self.grasp, segment or self.prepare(), self.ref,
                                        self.cancel, lambda _: None, **kwargs)
    def test_executes_whole_segment_once_preserves_offset_and_target(self):
        segment = self.prepare()
        result = self.execute(segment)
        self.assertEqual(len(self.grasp.robot.calls), 1)
        self.assertEqual(self.grasp.robot.calls[0][0], {'elbow_flex': 2100})
        self.assertEqual(self.grasp.robot.goals['elbow_flex'], 2080)
        self.assertEqual(result['joint_targets'], self.target)
        self.assertAlmostEqual(result['actual_delta_model_mm'][1], 10.)
        self.assertEqual(result['cloud_calls_during_segment'], 0)
        self.assertEqual(segment['uncertainty_ticks']['elbow_flex'], 26)
        self.assertEqual(len(segment['sampled_path']), 101)
        self.assertIn('no visual obstacle detection', result['camera_monitor_scope'])
        start_event = next(e for e in self.grasp.events if e['type'] == 'motion_program_segment_start')
        self.assertEqual(start_event['motion_contract']['held_goals'], self.held)
        self.assertEqual(start_event['motion_contract']['measured_origin'], self.raw)
        self.assertEqual(start_event['motion_contract']['actual_tracking_envelope_ticks'],
                         segment['actual_tracking_envelope_ticks'])
        self.assertEqual(start_event['motion_contract']['uncertainty_ticks'], segment['uncertainty_ticks'])
        self.assertEqual(start_event['motion_contract']['stationary_tolerance_ticks'], 4)
    def test_partial_endpoint_keeps_original_next_target_and_compensation(self):
        self.grasp.robot.endpoint_error = {'elbow_flex': -8}
        first = self.execute()
        nominal_next = {**self.target, 'elbow_flex': 2200}
        second = prepare_joint_segment(self.grasp.model, self.target,
                                       {**self.held, 'elbow_flex': 2080}, nominal_next)
        self.ref = first['after']
        result = self.execute(second, continuation=True)
        self.assertEqual(result['joint_targets']['elbow_flex'], 2200)
        self.assertEqual(self.grasp.robot.goals['elbow_flex'], 2180)
        self.assertEqual(result['after']['joints']['elbow_flex'], 2192)
    def test_first_dispatch_only_three_ticks_settling(self):
        self.grasp.robot.position['elbow_flex'] += 4
        with self.assertRaisesRegex(RuntimeError, '起點'):
            self.execute()
        self.assertEqual(self.grasp.robot.calls, [])
    def test_three_tick_settling_does_not_reanchor_target(self):
        self.grasp.robot.position['elbow_flex'] += 3
        self.execute()
        self.assertEqual(self.grasp.robot.goals['elbow_flex'], 2080)
    def test_changed_hold_rejects_before_dispatch(self):
        self.grasp.robot.goals['elbow_flex'] += 1
        with self.assertRaisesRegex(RuntimeError, '保持目標'):
            self.execute()
        self.assertFalse(self.grasp.robot.calls)
    def test_contract_target_mutation_rejects(self):
        segment = self.prepare()
        segment['joint_targets']['elbow_flex'] += 1
        with self.assertRaisesRegex(ValueError, '計畫不同'):
            self.execute(segment)
    def test_cancel_before_and_during_dispatch(self):
        self.cancel.set()
        with self.assertRaises(SessionCancelled):
            self.execute()
        self.assertFalse(self.grasp.robot.calls)
        self.cancel.clear()
        self.grasp.robot.callback = lambda *_: self.cancel.set()
        with self.assertRaises(SessionCancelled):
            self.execute()
        self.assertTrue(self.grasp.robot.stop.is_set())
        self.assertEqual(self.grasp.robot.hold_calls, 1)
    def test_deadline_enforced_inside_segment(self):
        with patch('server.motion_program.time.monotonic', side_effect=[1., 1., 1., 1., 31.]):
            with self.assertRaisesRegex(RuntimeError, '執行時間上限'):
                self.execute(deadline=30.)
        self.assertFalse(self.grasp.robot.calls)
    def test_camera_source_generation_and_live_freshness(self):
        self.ref['source'] = 'd435'
        with self.assertRaisesRegex(ValueError, '腕部相機'):
            self.execute()
        self.ref['source'] = 'wrist'
        self.ref['camera_generation'] = 'old'
        with self.assertRaisesRegex(RuntimeError, '重新連接'):
            self.execute()
        self.ref['camera_generation'] = 'one'
        self.grasp.robot.callback = lambda r, _: setattr(r.camera, 'okay', False)
        with self.assertRaisesRegex(RuntimeError, '畫面中斷'):
            self.execute()
        self.assertTrue(self.grasp.robot.stop.is_set())
    def test_camera_reconnect_during_segment(self):
        self.grasp.robot.callback = lambda r, _: setattr(r.camera, 'generation', 'two')
        with self.assertRaisesRegex(RuntimeError, '重新連接'):
            self.execute()
        self.assertEqual(self.grasp.robot.hold_calls, 1)
    def test_stationary_axis_drift_and_jaw_drift_abort(self):
        for name in ('wrist_roll', 'gripper'):
            with self.subTest(name=name):
                self.grasp = Grasp(self.raw, self.held)
                def change(r, _, name=name):
                    r.position[name] += 5
                self.grasp.robot.callback = change
                with self.assertRaises(RuntimeError):
                    self.execute()
                self.assertTrue(self.grasp.robot.stop.is_set())
    def test_asynchronous_axes_box_is_not_enough(self):
        origin = {n: 2000 for n in ALL}
        target = {**origin, 'shoulder_pan': 2120, 'elbow_flex': 2120}
        actual = {**origin, 'shoulder_pan': 2060}
        with self.assertRaisesRegex(RuntimeError, '不同步'):
            _tube(actual, origin, target, {n: 12 for n in NAMES})
        _tube({**actual, 'elbow_flex': 2050}, origin, target, {n: 12 for n in NAMES})
    def test_model_table_preflight_and_runtime(self):
        self.grasp.model.table_pass = False
        with self.assertRaisesRegex(ValueError, '桌面'):
            self.prepare()
        self.grasp.model.table_pass = True
        segment = self.prepare()
        self.grasp.robot.callback = lambda *_: setattr(self.grasp.model, 'floor_height', .004)
        with self.assertRaisesRegex(RuntimeError, '桌面'):
            self.execute(segment)
    def test_endpoint_failure_is_not_success(self):
        self.grasp.robot.endpoint_error = {'elbow_flex': -11}
        with self.assertRaisesRegex(RuntimeError, '終點'):
            self.execute()
        self.assertTrue(self.grasp.robot.stop.is_set())
    def test_jaw_rebind_changes_only_jaw_and_preserves_contact_owner(self):
        segment = self.prepare()
        bound = rebind_segment_jaw(segment, 1900, 1870)
        self.grasp.robot.position['gripper'] = 1900
        self.grasp.robot.goals['gripper'] = 1870
        self.grasp.robot.pwm_owner = object()
        self.ref['joints'] = {**self.raw, 'gripper': 1900}
        result = self.execute(bound)
        self.assertEqual(self.grasp.robot.goals['gripper'], 1870)
        self.assertEqual(result['after']['joints']['gripper'], 1900)
        for field in ('delta_ticks', 'sampled_path', 'uncertainty_ticks', 'clearance'):
            self.assertEqual(bound[field], segment[field])
    def test_limits_cannot_be_relaxed_by_caller(self):
        for value in (True, 0, 21, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare(endpoint_tolerance_ticks=value)
        with self.assertRaises(ValueError):
            self.prepare(approved_envelope=61)
        with self.assertRaises(ValueError):
            self.prepare(approved_envelope=12)
        with self.assertRaises(ValueError):
            prepare_joint_segment(self.grasp.model, self.raw, self.held,
                                  {**self.target, 'gripper': 1999})
        with self.assertRaises(ValueError):
            prepare_joint_segment(self.grasp.model, self.raw, self.held,
                                  {**self.target, 'elbow_flex': 2181})
    def test_real_model_screen_and_curve_are_available_without_hardware(self):
        model = default_model()
        origin = dict(zip(ALL, [2161, 1135, 2490, 2328, 1100, 1858]))
        target = {**origin, 'wrist_roll': 1213}
        segment = prepare_joint_segment(model, origin, origin, target)
        self.assertTrue(segment['clearance']['model_table_pass'])
        self.assertEqual(len(segment['sampled_path']), 114)
        self.assertLess(segment['nominal_tip_path_mm'], 80)

    def test_stationary_bias_has_no_hypothetical_goal_allowance(self):
        self.target = {**self.raw, 'wrist_roll': 2113}
        segment = self.prepare()
        self.assertEqual(segment['uncertainty_ticks']['elbow_flex'], 20)
        self.assertEqual(segment['actual_tracking_envelope_ticks']['elbow_flex'], 4)
        self.assertEqual(segment['uncertainty_ticks']['wrist_roll'], 12)
        self.assertEqual(segment['onboard_margin_ticks']['elbow_flex'], 0)
        self.assertEqual(segment['onboard_margin_ticks']['wrist_roll'], 12)
        self.assertFalse(segment['stationary_goal_resends'])
        self.execute(segment)
        self.assertEqual(self.grasp.robot.calls[0][0], {'wrist_roll': 2113})
        self.assertEqual(self.grasp.robot.goals['elbow_flex'], 1980)

    def test_stationary_tolerance_is_explicit_bounded_and_contract_bound(self):
        for value in (True, False, 0, -1, 13, 4.0, float('nan'), '12', None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '固定關節容許偏移'):
                self.prepare(stationary_tolerance_ticks=value)
        self.target = {**self.raw, 'wrist_roll': 2113}
        normal = self.prepare()
        relaxed = self.prepare(stationary_tolerance_ticks=12)
        self.assertEqual(normal['stationary_tolerance_ticks'], 4)
        self.assertEqual(relaxed['stationary_tolerance_ticks'], 12)
        self.assertEqual(relaxed['actual_tracking_envelope_ticks']['elbow_flex'], 12)
        self.assertEqual(relaxed['uncertainty_ticks']['elbow_flex'], 20)
        self.assertEqual(relaxed['measured_bounds']['elbow_flex'], [1988, 2012])
        self.assertEqual(relaxed['joint_targets'], normal['joint_targets'])
        self.assertEqual(relaxed['physical_goal_targets'], normal['physical_goal_targets'])
        self.assertNotEqual(relaxed['contract_sha256'], normal['contract_sha256'])
        changed = deepcopy(normal); changed['stationary_tolerance_ticks'] = 12
        with self.assertRaisesRegex(ValueError, '計畫不同'):
            self.execute(changed)
        self.assertFalse(self.grasp.robot.calls)

    def test_relaxed_stationary_drift_uses_original_anchor_and_keeps_hold_goals(self):
        self.target = {**self.raw, 'wrist_roll': 2113}
        segment = self.prepare(stationary_tolerance_ticks=12)
        samples = []
        def drift(r, kwargs):
            for ticks in (4, 8, 12):
                r.position['elbow_flex'] = self.raw['elbow_flex'] - ticks
                kwargs['measured_guard'](r.position)
                samples.append(ticks)
        self.grasp.robot.callback = drift
        result = self.execute(segment)
        self.assertEqual(samples, [4, 8, 12])
        self.assertEqual(result['after']['joints']['elbow_flex'], 1988)
        self.assertEqual(self.grasp.robot.goals['elbow_flex'], 1980)
        self.assertEqual(self.grasp.robot.calls[0][0], {'wrist_roll': 2113})
        self.assertEqual(segment['measured_origin']['elbow_flex'], 2000)

        # Several individually small changes do not reset the original box.
        self.grasp = Grasp(self.raw, self.held)
        def excessive_drift(r, kwargs):
            drift(r, kwargs)
            r.position['elbow_flex'] -= 1
            kwargs['measured_guard'](r.position)
        self.grasp.robot.callback = excessive_drift
        with self.assertRaisesRegex(RuntimeError, r'手肘.*1\.14°（13 刻度）.*1\.05°（12 刻度）'):
            self.execute(segment)
        self.assertTrue(self.grasp.robot.stop.is_set())
        self.assertEqual(self.grasp.robot.hold_calls, 1)
        self.assertEqual(self.grasp.robot.goals['elbow_flex'], 1980)

    def test_relaxed_stationary_policy_does_not_relax_start_jaw_or_approved_geometry(self):
        self.target = {**self.raw, 'wrist_roll': 2113}
        segment = self.prepare(stationary_tolerance_ticks=12)
        self.grasp.robot.position['elbow_flex'] -= 4
        with self.assertRaisesRegex(RuntimeError, '起點'):
            self.execute(segment)
        self.assertFalse(self.grasp.robot.calls)
        self.grasp.robot.position['elbow_flex'] = self.raw['elbow_flex']
        self.grasp.robot.callback = lambda r, _: r.position.update(gripper=self.raw['gripper'] + 5)
        with self.assertRaisesRegex(RuntimeError, '夾爪開口'):
            self.execute(segment)
        self.assertTrue(self.grasp.robot.stop.is_set())
        with self.assertRaisesRegex(ValueError, '已確認的路徑範圍'):
            self.prepare(stationary_tolerance_ticks=12,
                         approved_envelope={**segment['uncertainty_ticks'], 'shoulder_pan': 4})

    def test_observation_hold_box_expands_geometry_but_clips_onboard_boundary(self):
        model = default_model()
        raw = dict(zip(ALL, [2155, 1096, 3042, 2302, 805, 1858]))
        held = {**raw, 'elbow_flex': 3037}
        target = {**raw, 'wrist_roll': raw['wrist_roll'] + 113}
        normal = prepare_joint_segment(model, raw, held, target)
        segment = prepare_joint_segment(model, raw, held, target, stationary_tolerance_ticks=12)
        self.assertEqual(segment['geometry_bounds']['elbow_flex'], [3030, 3054])
        self.assertEqual(segment['raw_bounds']['elbow_flex'], [3030, 3045])
        self.assertEqual(segment['measured_bounds']['elbow_flex'], [3030, 3045])
        self.assertEqual(segment['clearance']['actual_tracking_envelope_ticks']['elbow_flex'], 12)
        self.assertLess(segment['clearance']['continuous_model_lower_bound_mm'][model.jaw_label],
                        normal['clearance']['continuous_model_lower_bound_mm'][model.jaw_label])
        for elbow in (3037, 3030, 3045, 3029, 3046):
            with self.subTest(elbow=elbow):
                g = Grasp(raw, held); g.model = model
                g.robot.callback = lambda r, _, value=elbow: r.position.update(elbow_flex=value)
                if elbow in (3029, 3046):
                    with self.assertRaises((RuntimeError, ValueError)):
                        execute_prepared_segment(g, segment, g.capture(), self.cancel, lambda _: None)
                    self.assertTrue(g.robot.stop.is_set())
                else:
                    receipt = execute_prepared_segment(g, segment, g.capture(), self.cancel, lambda _: None)
                    self.assertEqual(receipt['after']['joints']['elbow_flex'], elbow)
                self.assertEqual(g.robot.goals['elbow_flex'], 3037)
                self.assertEqual(g.robot.calls[0][0], {'wrist_roll': target['wrist_roll']})

    def test_real_boundary_pose_allows_screened_wrist_only_paths(self):
        model = default_model()
        raw = dict(zip(ALL, [2152, 931, 3040, 2334, 1037, 1857]))
        for joint, delta in (('wrist_roll', 113), ('wrist_roll', -113),
                             ('wrist_flex', 68), ('wrist_flex', -68)):
            with self.subTest(joint=joint, delta=delta):
                target = {**raw, joint: raw[joint] + delta}
                segment = prepare_joint_segment(model, raw, raw, target)
                self.assertTrue(segment['clearance']['model_table_pass'])
                self.assertIn(model.jaw_label, segment['clearance']['continuous_model_lower_bound_mm'])
                self.assertEqual(segment['active_joints'], [joint])
                self.assertEqual(segment['measured_bounds']['elbow_flex'], [3036, 3044])
                self.assertEqual(segment['uncertainty_ticks']['elbow_flex'], 4)
                self.assertEqual(segment['onboard_margin_ticks'][joint], 12)
                # Other users of the mesh helper retain the original default.
                with self.assertRaises(ValueError):
                    model.screen_segment(raw, target, uncertainty_ticks=segment['uncertainty_ticks'])
        for margin in (True, -1, 13):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                model.screen_segment(raw, raw, margin_ticks=margin)

    def test_stationary_geometry_can_exceed_limits_but_measured_and_held_positions_cannot(self):
        model = default_model()
        raw = dict(zip(ALL, [2152, 931, 3040, 2334, 1037, 1857]))
        target = {**raw, 'wrist_roll': 1150}
        held = {**raw, 'elbow_flex': 3035}
        segment = prepare_joint_segment(model, raw, held, target)
        self.assertEqual(segment['uncertainty_ticks']['elbow_flex'], 5)
        self.assertEqual(segment['physical_goal_targets']['elbow_flex'], 3035)
        # A larger inward held offset enlarges geometry symmetrically, but
        # does not command or permit the hypothetical outward mirror point.
        inward = prepare_joint_segment(model, raw, {**held, 'elbow_flex': 3034}, target)
        self.assertEqual(inward['geometry_bounds']['elbow_flex'], [3034, 3046])
        self.assertEqual(inward['raw_bounds']['elbow_flex'], [3034, 3045])
        for value in (840, 3046):
            with self.subTest(held_elbow=value), self.assertRaises(ValueError):
                prepare_joint_segment(model, raw, {**held, 'elbow_flex': value}, target)
        outside = {**raw, 'elbow_flex': 3046}
        with self.assertRaises(ValueError):
            prepare_joint_segment(model, outside, outside, {**outside, 'wrist_roll': 1150})
        limits = {**model.joint_limits,
                  'elbow_flex': (model.joint_limits['elbow_flex'][0], (3043 - 2047) * 2 * np.pi / 4096)}
        with patch.object(model, 'joint_limits', limits):
            clipped = prepare_joint_segment(model, raw, raw, target)
            self.assertEqual(clipped['measured_bounds']['elbow_flex'], [3036, 3043])
            self.assertEqual(clipped['geometry_bounds']['elbow_flex'], [3036, 3044])
            with self.assertRaisesRegex(ValueError, '模型關節'):
                prepare_joint_segment(model, raw, {**raw, 'elbow_flex': 3044}, target)

    def test_live_failure_pose_keeps_full_geometry_and_clips_only_stationary_runtime_bounds(self):
        model = default_model()
        raw = dict(zip(ALL, [2155, 1096, 3042, 2302, 805, 1858]))
        held = dict(zip(ALL, [2152, 1098, 3040, 2304, 810, 1857]))
        target = {**raw, 'wrist_flex': raw['wrist_flex'] - 68}
        segment = prepare_joint_segment(model, raw, held, target, endpoint_tolerance_ticks=20)
        self.assertEqual(segment['actual_tracking_envelope_ticks']['elbow_flex'], 4)
        self.assertEqual(segment['uncertainty_ticks']['elbow_flex'], 4)
        self.assertEqual(segment['geometry_bounds']['elbow_flex'], [3038, 3046])
        self.assertEqual(segment['raw_bounds']['elbow_flex'], [3038, 3045])
        self.assertEqual(segment['measured_bounds']['elbow_flex'], [3038, 3045])
        self.assertEqual(segment['physical_goal_targets']['elbow_flex'], 3040)
        self.assertTrue(segment['clearance']['model_table_pass'])
        self.assertEqual(segment['clearance']['actual_tracking_envelope_ticks']['elbow_flex'], 4)
        self.assertEqual(segment['clearance']['limit_validation_uncertainty_ticks']['elbow_flex'], 0)
        # Compare against an incorrectly reduced geometry envelope: the real
        # contract must retain the stricter full-four-tick mesh lower bound.
        reduced = model.screen_segment(raw, target, margin_ticks=0,
            uncertainty_ticks={**segment['uncertainty_ticks'], 'elbow_flex': 3})
        self.assertLess(segment['clearance']['continuous_model_lower_bound_mm'][model.jaw_label],
                        reduced['continuous_model_lower_bound_mm'][model.jaw_label])
        with self.assertRaisesRegex(ValueError, 'Onboard'):
            model.screen_segment(raw, target, margin_ticks=0, uncertainty_ticks=segment['uncertainty_ticks'])
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            model.screen_segment(raw, target, margin_ticks=0,
                uncertainty_ticks=segment['uncertainty_ticks'], stationary_limit_joints=['wrist_flex'])
        with self.assertRaises(ValueError):
            prepare_joint_segment(model, raw, held, {**raw, 'elbow_flex': 3000})

    def test_clipped_stationary_interval_is_enforced_during_motion_on_both_edges(self):
        raw = dict(zip(ALL, [2155, 1096, 3042, 2302, 805, 1858]))
        held = dict(zip(ALL, [2152, 1098, 3040, 2304, 810, 1857]))
        model = default_model()
        target = {**raw, 'wrist_flex': raw['wrist_flex'] - 68}
        segment = prepare_joint_segment(model, raw, held, target, endpoint_tolerance_ticks=20)
        for elbow in (3038, 3045, 3046):
            with self.subTest(elbow=elbow):
                g = Grasp(raw, held); g.model = model
                g.robot.callback = lambda r, _, value=elbow: r.position.update(elbow_flex=value)
                if elbow == 3046:
                    # +4 is inside the old symmetric box but outside the real
                    # onboard range, so it must stop under the new contract.
                    with self.assertRaisesRegex(ValueError, 'screened raw interval'):
                        execute_prepared_segment(g, segment, g.capture(), self.cancel, lambda _: None)
                    self.assertTrue(g.robot.stop.is_set())
                    self.assertEqual(g.robot.hold_calls, 1)
                else:
                    result = execute_prepared_segment(g, segment, g.capture(), self.cancel, lambda _: None)
                    self.assertEqual(result['after']['joints']['elbow_flex'], elbow)
                self.assertEqual(g.robot.goals['elbow_flex'], 3040)
                self.assertEqual(set(g.robot.calls[0][0]), {'wrist_flex'})

    def test_moving_elbow_retains_both_tracking_box_and_extra_margin(self):
        model = default_model()
        boundary = dict(zip(ALL, [2152, 931, 3040, 2334, 1037, 1857]))
        with self.assertRaises(ValueError):
            prepare_joint_segment(model, boundary, boundary, {**boundary, 'elbow_flex': 3000})
        raw = {**boundary, 'elbow_flex': 3000}
        safe = prepare_joint_segment(model, raw, raw, {**raw, 'elbow_flex': 3021})
        self.assertEqual(safe['raw_bounds']['elbow_flex'][1], 3033)
        self.assertEqual(model.calibration['elbow_flex']['range_max'] - 3033, 12)
        with self.assertRaisesRegex(ValueError, '原有行程餘量'):
            prepare_joint_segment(model, raw, raw, {**raw, 'elbow_flex': 3022})

    def test_twenty_tick_opt_in_expands_screened_geometry_and_not_default(self):
        model = default_model()
        raw = dict(zip(ALL, [2152, 931, 3000, 2334, 1037, 1857]))
        target = {**raw, 'elbow_flex': 3013}
        normal = prepare_joint_segment(model, raw, raw, target)
        relaxed = prepare_joint_segment(model, raw, raw, target, endpoint_tolerance_ticks=20)
        self.assertEqual(normal['endpoint_tolerance_ticks'], 10)
        self.assertEqual(normal['actual_tracking_envelope_ticks']['elbow_flex'], 12)
        self.assertEqual(relaxed['endpoint_tolerance_ticks'], 20)
        self.assertEqual(relaxed['actual_tracking_envelope_ticks']['elbow_flex'], 20)
        self.assertEqual(relaxed['uncertainty_ticks']['elbow_flex'], 20)
        self.assertEqual(relaxed['clearance']['actual_tracking_envelope_ticks']['elbow_flex'], 20)
        self.assertEqual(relaxed['measured_bounds']['elbow_flex'], [2980, 3033])
        self.assertEqual(relaxed['onboard_margin_ticks']['elbow_flex'], 12)
        self.assertEqual(relaxed['actual_tracking_envelope_ticks']['wrist_flex'], 4)
        self.assertEqual(relaxed['joint_targets'], normal['joint_targets'])
        self.assertEqual(relaxed['physical_goal_targets'], normal['physical_goal_targets'])
        jaw = model.jaw_label
        self.assertLess(relaxed['clearance']['continuous_model_lower_bound_mm'][jaw],
                        normal['clearance']['continuous_model_lower_bound_mm'][jaw])
        # Default remains valid here, but the newly disclosed larger box cannot
        # cross the old moving-axis reserve even by one tick.
        further = {**raw, 'elbow_flex': 3014}
        prepare_joint_segment(model, raw, raw, further)
        with self.assertRaisesRegex(ValueError, '原有行程餘量'):
            prepare_joint_segment(model, raw, raw, further, endpoint_tolerance_ticks=20)
        with self.assertRaisesRegex(ValueError, '已確認的路徑範圍'):
            prepare_joint_segment(model, raw, raw, target,
                                  endpoint_tolerance_ticks=20, approved_envelope=12)

    def test_runtime_rechecks_stationary_calibration_before_any_dispatch(self):
        raw = dict(zip(ALL, [2152, 931, 3040, 2334, 1037, 1857]))
        g = Grasp(raw, raw); g.model = default_model()
        segment = prepare_joint_segment(g.model, raw, raw, {**raw, 'wrist_roll': 1150})
        changed = deepcopy(g.model.calibration)
        changed['elbow_flex']['range_max'] = 3043
        with patch.object(g.model, 'calibration', changed), self.assertRaises(ValueError):
            execute_prepared_segment(g, segment, g.capture(), self.cancel, lambda _: None)
        self.assertEqual(g.robot.calls, [])


class ActualGlideProgramTests(unittest.TestCase):
    model = default_model()
    """Real Robot.glide, synthetic register bus, no hardware access."""
    def setUp(self):
        import test_wave_tolerance as helpers
        import test_wave as wave
        helpers.WaveEndpointToleranceTests.setUp(self)
        self.model = self.grasp.model
        for name, entry in self.robot.calibration.items():
            servo = entry['id']
            self.bus.offset[servo] = self.initial[name] - self.initial_holds[name]
            self.bus.reg[servo].update({60: 0, 69: 0})

    def capture(self):
        import test_wave as wave
        return wave.WaveTests.capture(self)

    def run_segment(self, **kwargs):
        reference = self.capture()
        target = {**reference['joints'], 'elbow_flex': reference['joints']['elbow_flex'] - 40}
        segment = prepare_joint_segment(self.model, reference['joints'], self.robot.goals, target)
        return execute_prepared_segment(self.grasp, segment, reference,
                                        self.cancel, self.reports.append, **kwargs)

    def test_real_controller_streams_whole_segment_without_cloud(self):
        result = self.run_segment()
        self.assertEqual(len(self.calls), 1)
        self.assertGreater(len(self.bus.events), 15)
        self.assertEqual({(servo, address) for servo, address, _ in self.bus.events}, {(3, 42)})
        self.assertGreater(result['actual_delta_model_mm'][2], 10)
        self.assertFalse(self.robot.stop.is_set())
        self.grasp._wave_options.assert_not_called()

    def test_extra_guard_only_receives_measured_pose_and_stops_immediately(self):
        seen = []
        def measured(actual):
            seen.append(dict(actual))
            self.assertEqual(actual, self.robot.read())
            if self.initial['elbow_flex'] - actual['elbow_flex'] > 12:
                raise RuntimeError('whole program envelope')
        with self.assertRaisesRegex(RuntimeError, 'whole program envelope'):
            self.run_segment(measured_guard=measured)
        self.assertGreater(len(seen), 2)
        self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(any(e.get('type') == 'motion_program_segment_result' for e in self.events))

    def test_real_glide_cancellation_stops_remaining_interpolation(self):
        original = self.robot._write
        moving_writes = []
        def write(name, address, value):
            result = original(name, address, value)
            if not self.robot.stop.is_set():
                moving_writes.append((name, address, value))
                if len(moving_writes) == 4:
                    self.cancel.set()
            return result
        self.robot._write = write
        with self.assertRaises(SessionCancelled):
            self.run_segment()
        self.assertEqual(len(moving_writes), 4)
        self.assertTrue(self.robot.stop.is_set())

    def test_real_glide_thermal_fault_retains_stop(self):
        self.bus.reg[3][63] = 61
        with self.assertRaisesRegex(RuntimeError, '溫度'):
            self.run_segment()
        self.assertTrue(self.robot.stop.is_set())

    def test_real_glide_near_limit_wrist_motion_never_rewrites_stationary_axes(self):
        raw = dict(zip(ALL, [2155, 1096, 3042, 2302, 805, 1858]))
        held = dict(zip(ALL, [2152, 1098, 3040, 2304, 810, 1857]))
        for name, entry in self.robot.calibration.items():
            servo = entry['id']
            entry.update(self.model.calibration[name])
            self.bus.reg[servo].update({56: raw[name], 42: held[name]})
            self.bus.offset[servo] = raw[name] - held[name]
        self.robot.goals = dict(held)
        self.robot.bias = {n: held[n] - raw[n] for n in ALL}
        reference = self.capture()
        target = {**raw, 'wrist_roll': raw['wrist_roll'] + 113}
        segment = prepare_joint_segment(self.model, raw, held, target)
        self.assertEqual(segment['measured_bounds']['elbow_flex'], [3038, 3045])
        self.assertEqual(segment['geometry_bounds']['elbow_flex'], [3038, 3046])
        result = execute_prepared_segment(self.grasp, segment, reference,
                                          self.cancel, self.reports.append)
        self.assertEqual(len(self.calls), 1)
        self.assertGreater(len(self.bus.events), 20)
        self.assertEqual({(servo, address) for servo, address, _ in self.bus.events}, {(5, 42)})
        self.assertLessEqual(abs(result['after']['joints']['wrist_roll'] - target['wrist_roll']), 10)
        for name in ALL:
            if name != 'wrist_roll':
                self.assertEqual(result['after']['joints'][name], raw[name])
                self.assertEqual(self.robot.goals[name], held[name])
        self.assertFalse(self.robot.stop.is_set())

    def test_real_glide_allows_five_tick_elbow_settle_without_an_elbow_write(self):
        raw = dict(zip(ALL, [2155, 1096, 3042, 2302, 805, 1858]))
        held = {**raw, 'elbow_flex': 3037}
        for name, entry in self.robot.calibration.items():
            servo = entry['id']
            entry.update(self.model.calibration[name])
            self.bus.reg[servo].update({56: raw[name], 42: held[name]})
            self.bus.offset[servo] = raw[name] - held[name]
        self.robot.goals = dict(held)
        self.robot.bias = {n: held[n] - raw[n] for n in ALL}
        reference = self.capture()
        target = {**raw, 'wrist_roll': raw['wrist_roll'] + 113}
        segment = prepare_joint_segment(self.model, raw, held, target, stationary_tolerance_ticks=12)
        original_write = self.robot._write
        def write(name, address, value):
            result = original_write(name, address, value)
            # The held elbow settles as the wrist changes load. Its motor goal
            # already is 3037; no synthetic write to the elbow causes this.
            if name == 'wrist_roll' and address == 42:
                self.bus.reg[3][56] = 3037
            return result
        self.robot._write = write
        receipt = execute_prepared_segment(self.grasp, segment, reference,
                                            self.cancel, self.reports.append)
        self.assertEqual(receipt['after']['joints']['elbow_flex'], 3037)
        self.assertEqual(self.robot.goals['elbow_flex'], 3037)
        self.assertEqual(self.robot.bias['elbow_flex'], -5)
        self.assertEqual({(servo, address) for servo, address, _ in self.bus.events}, {(5, 42)})
        self.assertFalse(self.robot.stop.is_set())

    def test_declared_twenty_tick_endpoint_accepts_real_eighteen_tick_shortfall(self):
        self.bus.residual = 18
        reference = self.capture()
        target = {**reference['joints'], 'wrist_roll': reference['joints']['wrist_roll'] + 68}
        segment = prepare_joint_segment(self.model, reference['joints'], self.robot.goals,
                                        target, endpoint_tolerance_ticks=20)
        result = execute_prepared_segment(self.grasp, segment, reference,
                                          self.cancel, self.reports.append)
        self.assertEqual(result['endpoint_error_ticks']['wrist_roll'], -18)
        self.assertEqual(result['joint_targets'], target)
        self.assertEqual(self.robot.goals, segment['physical_goal_targets'])
        self.assertEqual(segment['actual_tracking_envelope_ticks']['wrist_roll'], 20)
        self.assertGreaterEqual(result['after']['joints']['wrist_roll'] - reference['joints']['wrist_roll'], .4 * 68)
        self.assertEqual({(servo, address) for servo, address, _ in self.bus.events}, {(5, 42)})
        self.assertFalse(self.robot.stop.is_set())


if __name__ == '__main__':
    unittest.main()
