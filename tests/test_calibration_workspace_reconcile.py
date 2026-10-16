"""Evidence-only handling of an interrupted engineering prelude; no devices."""
from copy import deepcopy
from types import SimpleNamespace
import time
import unittest

import test_calibration_workspace_motion as motion_fixtures
import test_calibration_workspace_recovery as recovery_fixtures
from server import calibration_workspace_motion as workspace
from server.calibration_workspace_reconcile import build_workspace_interrupted_observation


class InterruptedObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recovery_fixtures.WorkspaceSettlingRecoveryTests.setUpClass()
        prior = recovery_fixtures.WorkspaceSettlingRecoveryTests(); prior.setUp()
        cls.protocol = prior.build()
        cls.model, cls.profile, cls.start = prior.model, prior.profile, prior.current

    def setUp(self):
        original = SimpleNamespace(model=self.model, state='idle',
            robot=SimpleNamespace(goals=dict(self.start), pwm_owner=None, moving=False))
        self.candidate = workspace.prepare_workspace_station(original, self.protocol,
            dict(motion_fixtures.WorkspaceProtocolTests.frame(self.start), camera_generation='new-retry-camera'),
            station_index=0, completed_station_ids=[])
        self.candidate['candidate_id'] = 'second-prelude-candidate'
        self.parent = dict(session_id=self.protocol['session_id'], protocol=deepcopy(self.protocol),
            completed_station_ids=[], receipts=[], captures=[],
            inflight=dict(station_id='recovery-elbow-01', candidate_id=self.candidate['candidate_id']),
            recovered_from=self.protocol['recovery']['parent_protocol_sha256'])
        failed = {**self.start, 'elbow_flex':3031}
        goals = deepcopy(self.candidate['segment']['physical_goal_targets'])
        self.failure = dict(error="未到達目標位置，已保持姿勢：{'elbow_flex': 31}",
            time=1790766060., actual=failed, goals=goals)
        self.telemetry = [dict(time=1790766057. + i*.1,
            type='motion_program_endpoint_settling', kind='endpoint_settling',
            contract_sha256=self.candidate['segment']['contract_sha256'],
            elapsed_s=i*.1, actual=deepcopy(failed), goals=deepcopy(goals),
            logical_targets={'elbow_flex':3000}, endpoint_error_ticks={'elbow_flex':31},
            timeout_s=2.5, sample_index=i) for i in range(1,27)]
        # Holding after a failed command changes the observed position. It is
        # this new pose, not the old 3031 plateau, that must drive the next path.
        self.current = {**self.start, 'elbow_flex':3040}
        self.grasp = SimpleNamespace(model=self.model, state='idle',
            robot=SimpleNamespace(goals=dict(self.current), pwm_owner=None, moving=False))

    def build(self, **overrides):
        now = time.monotonic()
        frame = overrides.pop('frame', dict(motion_fixtures.WorkspaceProtocolTests.frame(self.current),
            id='fresh-after-hold', camera_generation='new-retry-camera'))
        samples = overrides.pop('settled_samples', [
            {'monotonic':now-.4, 'joints':dict(self.current)},
            {'monotonic':now, 'joints':dict(self.current)}])
        kw = dict(camera_profile=self.profile, camera_id=self.profile['camera_id'],
            mount_revision=self.profile['mount_revision'], settled_samples=samples)
        kw.update(overrides)
        return build_workspace_interrupted_observation(self.grasp, self.parent, self.candidate,
            self.failure, self.telemetry, frame, **kw)

    def test_resolves_incident_not_endpoint_preserves_fresh_pose_and_all_evidence(self):
        before = deepcopy((self.parent,self.candidate,self.failure,self.telemetry,self.grasp.robot.goals))
        result = self.build()
        workspace._validate(result, 'engineering_workspace_interrupted_observation')
        receipt = result['receipt']
        self.assertTrue(receipt['completed']); self.assertTrue(receipt['interrupted_observation_resolved'])
        self.assertEqual(receipt['completion'], 'stable_partial_observation')
        self.assertFalse(receipt['endpoint_reached']); self.assertFalse(receipt['motion_completed'])
        self.assertTrue(receipt['nominal_unreached'])
        self.assertEqual(receipt['after']['joints'], self.current)
        self.assertEqual(receipt['after']['joints']['elbow_flex'], 3040)
        self.assertEqual(result['failure_evidence']['actual']['elbow_flex'], 3031)
        self.assertEqual(result['telemetry'], self.telemetry)
        self.assertEqual(result['original_reference'], self.protocol['reference'])
        self.assertEqual(result['origin'], self.protocol['origin'])
        self.assertEqual(result['protocol_sha256'], self.protocol['contract_sha256'])
        self.assertEqual(result['parent_record_sha256'], workspace._hash(self.parent))
        self.assertFalse(result['motors_commanded']); self.assertFalse(result['original_protocol_changed'])
        self.assertFalse(result['old_path_approval_reused'])
        self.assertTrue(result['next_station_requires_fresh_review'])
        self.assertIsNone(result['camera_rebind'])
        self.assertEqual(before, (self.parent,self.candidate,self.failure,self.telemetry,self.grasp.robot.goals))

    def test_restart_requires_explicit_same_camera_profile_binding(self):
        frame = dict(motion_fixtures.WorkspaceProtocolTests.frame(self.current), id='restarted', camera_generation='next-generation')
        with self.assertRaisesRegex(ValueError,'明確綁定'):self.build(frame=frame)
        result = self.build(frame=frame, rebind_camera_generation=True)
        rebind = result['camera_rebind']
        workspace._validate(rebind, 'engineering_workspace_camera_rebind')
        self.assertEqual(rebind['from_generation'], self.protocol['reference']['camera_generation'])
        self.assertEqual(rebind['to_generation'], 'next-generation')
        self.assertEqual(rebind['original_reference_sha256'], workspace._hash(self.protocol['reference']))
        self.assertEqual(rebind['camera_id'], self.profile['camera_id'])
        self.assertEqual(rebind['mount_revision'], self.profile['mount_revision'])
        profile = deepcopy(self.profile); profile['handeye'][0][3] += .001
        with self.assertRaisesRegex(ValueError, '不能更換'):self.build(camera_profile=profile)

    def test_wrong_attempt_prior_progress_and_candidate_reject(self):
        self.parent['completed_station_ids'] = ['recovery-elbow-01']
        with self.assertRaisesRegex(ValueError,'最後一次'):self.build()
        self.parent['completed_station_ids'] = []
        self.parent['inflight']['candidate_id'] = 'another-candidate'
        with self.assertRaisesRegex(ValueError,'未完成紀錄'):self.build()

    def test_telemetry_wrong_contract_goal_and_gap_reject(self):
        original = deepcopy(self.telemetry)
        for field,value in (('contract_sha256','unrelated'),('goals',{}),('sample_index',10)):
            self.telemetry = deepcopy(original); self.telemetry[1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError,'不連續'):self.build()

    def test_unstable_telemetry_rejects_partial_observation(self):
        for index,row in enumerate(self.telemetry[-5:-1]):
            row['actual']['elbow_flex'] += index + 3
            row['endpoint_error_ticks']['elbow_flex'] += index + 3
        with self.assertRaisesRegex(ValueError,'停穩'):self.build()

    def test_no_progress_and_already_reached_failure_reject(self):
        self.failure['actual']['elbow_flex'] = self.start['elbow_flex']
        with self.assertRaisesRegex(ValueError,'實際前進'):self.build()
        self.failure['actual']['elbow_flex'] = 3002
        with self.assertRaisesRegex(ValueError,'實際前進'):self.build()

    def test_unrecorded_post_hold_motion_rejects(self):
        self.current['elbow_flex'] = 3044
        with self.assertRaisesRegex(ValueError,'保持範圍'):self.build()

    def test_fresh_stability_required_in_addition_to_historical_plateau(self):
        now = time.monotonic()
        for samples in ([], [{'monotonic':now,'joints':self.current}],
            [{'monotonic':now-.4,'joints':{**self.current,'elbow_flex':3036}},
             {'monotonic':now,'joints':self.current}],
            [{'monotonic':now-8,'joints':self.current},{'monotonic':now-7,'joints':self.current}]):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError,'停穩'):self.build(settled_samples=samples)

    def test_stationary_axis_drift_and_active_motion_reject(self):
        self.telemetry[-1]['actual']['wrist_roll'] += 30
        with self.assertRaisesRegex(RuntimeError,'偏離保持位置'):self.build()
        self.telemetry[-1]['actual']['wrist_roll'] -= 30
        self.grasp.robot.moving = True
        with self.assertRaisesRegex(ValueError,'保持手臂'):self.build()


if __name__=='__main__':unittest.main()
