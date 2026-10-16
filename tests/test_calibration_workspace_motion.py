"""Finite shoulder/elbow survey contracts; real CAD, no devices or cloud."""
from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from server import calibration_workspace_motion as workspace
from server.local_cartesian import ALL, NAMES, default_model
from server.motion_program import (prepare_joint_segment, execute_prepared_segment,
    _fixed_origin, WORKSPACE_ORIGIN_PROFILE, WORKSPACE_ORIGIN_DELTA_BOUNDS)


class WorkspaceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = default_model()
        cls.raw = dict(zip(ALL, (2181, 1032, 3042, 2174, 884, 1861)))
        cls.profile = json.loads((Path(__file__).resolve().parents[1] / 'config/calibration/'
            '0a6bf0697d0f4c2d99b1bfe0f6b68bd3-193ed03c8661474a9f6226eb983dfccb-profile.json').read_text())
        grasp = SimpleNamespace(model=cls.model, state='idle',
            robot=SimpleNamespace(goals=dict(cls.raw), pwm_owner=None))
        cls.protocol = workspace.build_workspace_protocol(grasp, cls.frame(cls.raw),
            session_id='fixed-session', camera_profile=cls.profile,
            camera_id=cls.profile['camera_id'], mount_revision=cls.profile['mount_revision'])

    @staticmethod
    def frame(raw):
        return {'source': 'wrist', 'id': 'fresh-frame', 'path': '/unused/wrist.jpg',
            'camera_generation': 'fixed-generation', 'capture_monotonic': time.monotonic(),
            'joints': dict(raw), 'image_size': [1920, 1080]}

    def setUp(self):
        self.events = []
        self.grasp = SimpleNamespace(model=self.model, state='idle', record=self.events.append,
            robot=SimpleNamespace(goals=dict(self.raw), pwm_owner=None))

    def prepare(self, index=0, **overrides):
        previous = self.raw if index == 0 else self.protocol['stations'][index - 1]['target_joints']
        self.grasp.robot.goals = dict(previous)
        kwargs = dict(station_index=index,
            completed_station_ids=[s['station_id'] for s in self.protocol['stations'][:index]])
        if index:
            kwargs['previous_receipt'] = dict(completed=True,
                protocol_sha256=self.protocol['contract_sha256'],
                station_id=self.protocol['stations'][index - 1]['station_id'],
                after=self.frame(previous))
        kwargs.update(overrides)
        return workspace.prepare_workspace_station(self.grasp, self.protocol, self.frame(previous), **kwargs)

    def approval(self, candidate):
        return dict(approved=True, gripper_empty=True, human_clear=True, clear_path=True,
            protocol_sha256=self.protocol['contract_sha256'],
            candidate_sha256=candidate['contract_sha256'], frame_id=candidate['reference']['id'],
            reason='Offline fixture: one exact segment has been reviewed.')

    def execute(self, candidate, **overrides):
        kwargs = dict(session_id='fixed-session', completed_station_ids=[],
                      approval=self.approval(candidate))
        kwargs.update(overrides)
        return workspace.execute_workspace_station(self.grasp, self.protocol, candidate,
            self.frame(self.raw), threading.Event(), lambda _: None, **kwargs)

    def test_full_finite_manifest_has_preregistered_twelve_distinct_views(self):
        p = self.protocol
        self.assertEqual(len(p['stations']), 18)
        self.assertEqual(p['origin'], self.raw)
        self.assertFalse(p['motors_commanded'])
        self.assertFalse(p['automatic_station_chaining'])
        captures = [s for s in p['stations'] if s['capture_required']]
        self.assertEqual(len(captures), 12)
        self.assertEqual(len(p['stations']) - len(captures), 6)
        self.assertEqual(p['stations'][0]['nominal_segment']['active_joints'], ['elbow_flex'])
        self.assertEqual(p['stations'][0]['target_joints'], {**self.raw, 'elbow_flex': 3000})
        self.assertEqual(p['stations'][1]['nominal_segment']['active_joints'],
                         ['shoulder_pan', 'shoulder_lift', 'wrist_flex'])
        for capture, offsets in zip(captures, workspace.CAPTURE_OFFSETS):
            self.assertEqual(capture['target_joints'],
                             {**self.raw, **{n: self.raw[n] + d for n, d in zip(NAMES, offsets)}})
        self.assertEqual([s['split'] for s in captures], ['training', 'heldout'] * 6)
        from server.geometry import observation_families
        self.assertEqual(len(observation_families([{'joints': s['target_joints']} for s in captures])), 12)
        self.assertGreater(captures[-1]['target_joints']['shoulder_lift'] - p['origin']['shoulder_lift'], 1000)
        self.assertLess(captures[-1]['target_joints']['elbow_flex'] - p['origin']['elbow_flex'], -400)
        for s in p['stations']:
            segment = s['nominal_segment']
            self.assertEqual(segment['fixed_origin'], self.raw)
            self.assertEqual(segment['origin_limit_profile'], WORKSPACE_ORIGIN_PROFILE)
            self.assertTrue(segment['clearance']['model_table_pass'])
            self.assertLessEqual(max(abs(v) for v in segment['delta_ticks'].values()), 160)
            self.assertLessEqual(segment['nominal_tip_path_mm'], 80)
            self.assertEqual(s['target_joints']['gripper'], self.raw['gripper'])
            self.assertEqual(s['nominal_board_projection']['visible_corner_count'], 54)
            self.assertFalse(s['external_clearance_verified'])
            self.assertFalse(s['nominal_board_projection']['occlusion_checked'])

    def test_default_360_rule_does_not_change_and_unknown_profiles_reject(self):
        far = {**self.raw, 'shoulder_lift': self.raw['shoulder_lift'] + 361}
        with self.assertRaisesRegex(ValueError, '關節範圍'):
            _fixed_origin(far, self.raw)
        _fixed_origin(far, self.raw, WORKSPACE_ORIGIN_PROFILE)
        for profile in ('anything', {}, True):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                _fixed_origin(far, self.raw, profile)
        with self.assertRaises(ValueError):
            _fixed_origin(far, None, WORKSPACE_ORIGIN_PROFILE)
        with self.assertRaises(ValueError):
            _fixed_origin({**far, 'shoulder_lift': self.raw['shoulder_lift'] + 1251},
                          self.raw, WORKSPACE_ORIGIN_PROFILE)

    def test_production_executor_cannot_run_engineering_segment(self):
        segment = self.protocol['stations'][0]['nominal_segment']
        with self.assertRaisesRegex(ValueError, '工程校準'):
            execute_prepared_segment(self.grasp, segment, self.frame(self.raw), None, lambda _: None)
        normal = prepare_joint_segment(self.model, self.raw, self.raw,
            {**self.raw, 'elbow_flex': 3000}, inward_limit_joints=['elbow_flex'])
        with self.assertRaisesRegex(ValueError, '契約'):
            execute_prepared_segment(self.grasp, normal, self.frame(self.raw), None, lambda _: None,
                                     engineering_workspace=True)

    def test_fresh_first_station_retains_global_origin_and_inward_contract(self):
        c = self.prepare()
        self.assertEqual(c['segment']['fixed_origin'], self.raw)
        self.assertEqual(c['segment']['inward_limit_joints'], ['elbow_flex'])
        self.assertEqual(c['segment']['actual_tracking_envelope_ticks']['elbow_flex'], 16)
        self.assertEqual(c['segment']['geometry_bounds']['elbow_flex'][1], 3058)
        self.assertEqual(c['segment']['measured_bounds']['elbow_flex'][1], 3045)
        self.assertEqual(c['segment']['joint_targets'], self.protocol['stations'][0]['target_joints'])
        self.assertTrue(c['framing_is_not_obstacle_clearance'])

    def test_late_station_keeps_original_origin_instead_of_local_rebase(self):
        c = self.prepare(17)
        self.assertEqual(c['origin'], self.raw)
        self.assertEqual(c['segment']['fixed_origin'], self.raw)
        self.assertNotEqual(c['segment']['measured_origin'], self.raw)
        self.assertEqual(c['split'], 'heldout')
        self.assertGreater(c['segment']['joint_targets']['shoulder_lift'] - self.raw['shoulder_lift'], 1000)

    def test_skipped_and_repeated_station_do_not_rebase_origin(self):
        with self.assertRaisesRegex(ValueError, '跳站'):
            self.prepare(1, completed_station_ids=[])
        with self.assertRaisesRegex(ValueError, '跳站'):
            self.prepare(0, completed_station_ids=['station-01'])
        with self.assertRaisesRegex(ValueError, '上一站'):
            workspace.prepare_workspace_station(self.grasp, self.protocol,
                self.frame({**self.raw, 'shoulder_lift': 1100}),
                station_index=0, completed_station_ids=[])

    def test_changed_generation_or_tampered_manifest_rejects(self):
        f = {**self.frame(self.raw), 'camera_generation': 'reconnected'}
        with self.assertRaisesRegex(ValueError, '相機'):
            workspace.prepare_workspace_station(self.grasp, self.protocol, f,
                station_index=0, completed_station_ids=[])
        bad = deepcopy(self.protocol); bad['origin']['shoulder_pan'] += 1
        with self.assertRaisesRegex(ValueError, '改變'):
            workspace.prepare_workspace_station(self.grasp, bad, self.frame(self.raw),
                station_index=0, completed_station_ids=[])

    def test_dispatch_forwards_one_exact_segment_with_global_guard(self):
        c = self.prepare()
        with patch('server.grasp.unchanged_wrist') as scene, patch.object(workspace,
                'execute_prepared_segment', return_value={'after': {'joints': c['segment']['joint_targets']}}) as execute:
            result = self.execute(c)
        self.assertEqual(execute.call_count, 1)
        self.assertIs(execute.call_args.args[1], c['segment'])
        self.assertTrue(execute.call_args.kwargs['engineering_workspace'])
        self.assertEqual(execute.call_args.kwargs['speed'], 60)
        self.assertFalse(result['next_station_executed'])
        self.assertFalse(result['calibration_sample_recorded'])
        self.assertTrue(result['completed'])
        guard = execute.call_args.kwargs['measured_guard']
        guard(self.raw)
        with self.assertRaises(ValueError):
            guard({**self.raw, 'shoulder_pan': self.raw['shoulder_pan'] + 301})
        scene.assert_called_once()

    def test_preflight_is_read_only_and_returns_the_exact_next_station(self):
        c = self.prepare()
        with patch('server.grasp.unchanged_wrist') as scene, patch.object(workspace,
                'execute_prepared_segment') as execute:
            station = workspace.validate_workspace_station(self.grasp, self.protocol, c,
                self.frame(self.raw), session_id='fixed-session', completed_station_ids=[],
                approval=self.approval(c))
        self.assertIs(station, self.protocol['stations'][0])
        self.assertEqual(self.events, [])
        self.assertEqual(self.grasp.robot.goals, self.raw)
        self.assertEqual(self.grasp.state, 'idle')
        execute.assert_not_called()
        scene.assert_called_once()

    def test_expired_or_changed_preflight_fails_without_motor_or_record(self):
        c = self.prepare()
        with patch.object(workspace, 'execute_prepared_segment') as execute:
            with patch.object(workspace.time, 'monotonic', return_value=c['expires_monotonic'] + 1), \
                 patch.object(workspace.observation, '_frame', return_value={}):
                with self.assertRaisesRegex(ValueError, '過期'):
                    workspace.validate_workspace_station(self.grasp, self.protocol, c,
                        self.frame(self.raw), session_id='fixed-session', completed_station_ids=[],
                        approval=self.approval(c))
            c['station_id'] = 'changed-after-preflight'
            with self.assertRaisesRegex(ValueError, '改變'):
                self.execute(c)
        self.assertEqual(self.events, [])
        execute.assert_not_called()

    def test_dispatch_rejects_missing_approval_changed_hold_and_expiry(self):
        c = self.prepare()
        with patch.object(workspace, 'execute_prepared_segment') as execute:
            approval = self.approval(c); approval['clear_path'] = False
            with self.assertRaisesRegex(ValueError, '核對'):
                self.execute(c, approval=approval)
            self.grasp.robot.goals['shoulder_pan'] += 1
            with self.assertRaisesRegex(ValueError, '保持'):
                self.execute(c)
            self.grasp.robot.goals = dict(self.raw)
            with patch.object(workspace.time, 'monotonic', return_value=c['expires_monotonic'] + 1), \
                 patch.object(workspace.observation, '_frame', return_value={}):
                with self.assertRaisesRegex(ValueError, '過期'):
                    self.execute(c)
            execute.assert_not_called()

    def test_board_reprojection_identity_mismatch_rejected(self):
        ref = workspace._seal({'version': 1, 'kind': 'engineering_workspace_board_reference',
            'camera_generation': 'different', 'camera_profile_sha256': self.protocol['camera_profile_sha256'],
            'geometry_revision': self.protocol['geometry_revision'], 'frame_id': 'board',
            'board_transform_base': self.profile['validation_scope']['board_plane_base']})
        with self.assertRaisesRegex(ValueError, '重投影'):
            self.prepare(board_reference=ref)

    def test_large_nominal_step_still_rejected(self):
        with self.assertRaisesRegex(ValueError, '180'):
            prepare_joint_segment(self.model, self.raw, self.raw,
                {**self.raw, 'shoulder_lift': self.raw['shoulder_lift'] + 181},
                fixed_origin=self.raw, origin_limit_profile=WORKSPACE_ORIGIN_PROFILE)


class WorkspaceDriverTests(unittest.TestCase):
    # Exercise actual Robot.glide/register commands rather than a mock executor.
    from test_motion_program import ActualGlideProgramTests as Fixtures
    model = default_model()
    capture = Fixtures.capture

    def setUp(self):
        self.Fixtures.setUp(self)
        self.raw = dict(zip(ALL, (2181, 1032, 3042, 2174, 884, 1861)))
        self.held = {**self.raw, 'elbow_flex': 3037}
        for n, entry in self.robot.calibration.items():
            servo = entry['id']; entry.update(self.model.calibration[n])
            self.bus.reg[servo].update({56: self.raw[n], 42: self.held[n], 60: 0, 69: 0})
            self.bus.offset[servo] = self.raw[n] - self.held[n]
        self.robot.goals = dict(self.held)
        self.robot.bias = {n: self.held[n] - self.raw[n] for n in ALL}

    def test_elbow_prelude_clears_starting_deadband_then_body_station_holds_elbow(self):
        # Register-model stiction: the loaded elbow remains at its start until
        # the inward goal is 25 ticks away, then follows with the same 5-tick
        # load offset. This is not a permanent 25-tick endpoint error.
        original_write = self.robot._write
        released = False
        def write(name, address, value):
            nonlocal released
            result = original_write(name, address, value)
            if name == 'elbow_flex' and address == 42 and not released:
                released = self.held[name] - value >= 25
                if not released:
                    self.bus.reg[3][56] = self.raw['elbow_flex']
            return result
        self.robot._write = write
        prelude = {**self.raw, 'elbow_flex': 3000}
        segment = workspace._segment(self.model, self.raw, self.held, prelude, self.raw)
        first = execute_prepared_segment(self.grasp, segment, self.capture(),
            self.cancel, self.reports.append, engineering_workspace=True,
            measured_guard=lambda actual: workspace._origin_guard(self.model, self.raw, actual))
        self.assertTrue(released)
        self.assertEqual(first['after']['joints'], prelude)
        self.assertEqual({(servo, address) for servo, address, _ in self.bus.events}, {(3, 42)})
        self.assertEqual(self.robot.goals['elbow_flex'], 2995)
        self.assertEqual(segment['endpoint_tolerance_ticks'], 16)
        self.assertEqual(segment['stationary_tolerance_ticks'], 12)
        self.bus.events.clear()
        target = {**self.raw, 'shoulder_pan': 2341, 'shoulder_lift': 1122,
                  'elbow_flex': 3000, 'wrist_flex': 2109}
        segment = workspace._segment(self.model, first['after']['joints'],
                                     self.robot.goals, target, self.raw)
        result = execute_prepared_segment(self.grasp, segment, self.capture(),
            self.cancel, self.reports.append, engineering_workspace=True,
            measured_guard=lambda actual: workspace._origin_guard(self.model, self.raw, actual))
        self.assertEqual({(servo, address) for servo, address, _ in self.bus.events},
                         {(1, 42), (2, 42), (4, 42)})
        self.assertEqual(self.robot.goals['elbow_flex'], 2995)
        self.assertEqual(self.robot.goals['wrist_roll'], 884)
        self.assertEqual(self.robot.goals['gripper'], 1861)
        self.assertEqual(result['after']['joints'], target)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(self.robot.stop.is_set())


if __name__ == '__main__':
    unittest.main()
