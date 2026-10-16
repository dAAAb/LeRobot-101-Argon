"""Interrupted engineering survey rebuilds retain the first origin; no devices."""
from copy import deepcopy
from types import SimpleNamespace
import time
import unittest

import test_calibration_workspace_motion as fixtures
from server import calibration_workspace_motion as workspace
from server.calibration_workspace_recovery import (build_workspace_recovery_protocol,
    build_workspace_settling_recovery_protocol)
from server.local_cartesian import ALL


class WorkspaceRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WorkspaceProtocolTests.setUpClass()
        cls.model=fixtures.WorkspaceProtocolTests.model
        cls.profile=fixtures.WorkspaceProtocolTests.profile
        cls.protocol=deepcopy(fixtures.WorkspaceProtocolTests.protocol)
        cls.origin=fixtures.WorkspaceProtocolTests.raw
        # Recovery is intentionally for the interrupted legacy 17-station
        # manifest. A future normal builder already has its own prelude;
        # reconstruct this frozen historical template rather than broadening
        # recovery to accept another retry of that newer protocol.
        if len(cls.protocol['stations'])==18:
            cls.protocol['stations'].pop(0)
            for i,station in enumerate(cls.protocol['stations']):
                station.update(station_id=f'station-{i+1:02d}',station_index=i)
            cls.protocol['stations'][0]['nominal_segment']=workspace._segment(cls.model,
                cls.origin,cls.origin,cls.protocol['stations'][0]['target_joints'],cls.origin)
            workspace._seal(cls.protocol)

    def setUp(self):
        original_grasp=SimpleNamespace(model=self.model,state='idle',
            robot=SimpleNamespace(goals=dict(self.origin),pwm_owner=None))
        self.candidate=workspace.prepare_workspace_station(original_grasp,self.protocol,
            fixtures.WorkspaceProtocolTests.frame(self.origin),station_index=0,completed_station_ids=[])
        self.candidate['candidate_id']='failed-candidate'
        self.candidate['frame']={'url':'/display-only.jpg'}
        self.parent=dict(session_id=self.protocol['session_id'],protocol=deepcopy(self.protocol),
            completed_station_ids=[],receipts=[],captures=[],
            inflight=dict(station_id='station-01',candidate_id='failed-candidate'))
        self.failure=dict(error='Synthetic recorded no-progress failure',time=1790764505.,
            actual=dict(zip(ALL,(2254,1074,3042,2154,884,1861))),
            goals=dict(zip(ALL,(2257,1075,3022,2143,884,1861))))
        self.current=dict(zip(ALL,(2254,1074,3044,2157,884,1861)))
        self.grasp=SimpleNamespace(model=self.model,state='idle',
            robot=SimpleNamespace(goals=dict(self.current),pwm_owner=None,moving=False))

    def build(self,**overrides):
        kw=dict(session_id=self.protocol['session_id'],camera_profile=self.profile,
                camera_id=self.profile['camera_id'],mount_revision=self.profile['mount_revision'])
        frame=overrides.pop('frame',dict(fixtures.WorkspaceProtocolTests.frame(self.current),
            id='new-stopped-image',camera_generation='restarted-camera'))
        kw.update(overrides)
        return build_workspace_recovery_protocol(self.grasp,self.parent,self.candidate,
            self.failure,frame,**kw)

    def test_single_axis_prelude_keeps_all_original_targets_and_global_bounds(self):
        before=deepcopy((self.parent,self.candidate,self.failure,self.grasp.robot.goals))
        result=self.build()
        self.assertEqual(result['origin'],self.origin)
        self.assertEqual(result['session_id'],self.parent['session_id'])
        self.assertEqual(len(result['stations']),18)
        first=result['stations'][0]
        self.assertEqual(first['station_id'],'recovery-elbow-01')
        self.assertEqual(first['target_joints'],{**self.current,'elbow_flex':3000})
        self.assertEqual(first['nominal_segment']['active_joints'],['elbow_flex'])
        self.assertFalse(first['capture_required']);self.assertIsNone(first['split'])
        for new,old in zip(result['stations'][1:],self.protocol['stations']):
            for field in ('station_id','target_joints','capture_id','capture_required','split'):
                self.assertEqual(new[field],old[field])
        for station in result['stations']:
            self.assertEqual(station['nominal_segment']['fixed_origin'],self.origin)
            self.assertTrue(station['nominal_segment']['clearance']['model_table_pass'])
            self.assertFalse(station['external_clearance_verified'])
            self.assertFalse(station['nominal_segment']['gripper_command_included'])
        self.assertEqual(result['joint_delta_bounds_from_original_origin'],self.protocol['joint_delta_bounds_from_original_origin'])
        self.assertEqual(result['maximum_tip_distance_from_original_origin_mm'],180.)
        self.assertFalse(result['automatic_station_chaining']);self.assertFalse(result['motors_commanded'])
        self.assertEqual(before,(self.parent,self.candidate,self.failure,self.grasp.robot.goals))

    def test_generation_restart_is_bound_to_same_frozen_camera_and_onboard_geometry(self):
        result=self.build();r=result['recovery']
        self.assertEqual(result['reference']['camera_generation'],'restarted-camera')
        self.assertEqual(r['original_camera_generation'],self.protocol['reference']['camera_generation'])
        self.assertEqual(r['parent_protocol_sha256'],self.protocol['contract_sha256'])
        self.assertEqual(r['parent_record_sha256'],workspace._hash(self.parent))
        self.assertEqual(r['original_failed_candidate_sha256'],self.candidate['contract_sha256'])
        self.assertEqual(r['failure_evidence'],self.failure)
        self.assertEqual(r['failure_evidence_sha256'],workspace._hash(self.failure))
        self.assertEqual(r['recovery_start'],self.current)
        self.assertEqual(r['failed_station_id'],'station-01')
        self.assertEqual(result['camera_profile'],self.profile)
        self.assertEqual(result['geometry_revision'],self.protocol['geometry_revision'])
        self.assertFalse(r['old_path_approval_reused']);self.assertFalse(r['original_origin_reset'])
        self.assertTrue(r['parent_journal_must_be_preserved'])

    def test_missing_or_wrong_inflight_and_candidate_do_not_resume(self):
        original=deepcopy(self.parent)
        for inflight in (None,{'station_id':'station-02','candidate_id':'failed-candidate'},
                         {'station_id':'station-01','candidate_id':'another-candidate'}):
            self.parent=deepcopy(original);self.parent['inflight']=inflight
            with self.subTest(inflight=inflight),self.assertRaises(ValueError):self.build()
        self.parent=original
        self.candidate['segment']['joint_targets']['elbow_flex']-=1
        with self.assertRaisesRegex(ValueError,'改變'):self.build()

    def test_completed_or_already_recovered_protocol_is_not_rebased_again(self):
        self.parent['completed_station_ids']=['station-01']
        with self.assertRaisesRegex(ValueError,'不能再次'):self.build()
        self.parent['completed_station_ids']=[]
        self.parent['protocol']['recovery']={'prior':'attempt'}
        workspace._seal(self.parent['protocol'])
        with self.assertRaisesRegex(ValueError,'不能再次'):self.build()

    def test_changed_calibration_and_expanded_global_bounds_reject(self):
        profile=deepcopy(self.profile);profile['camera_matrix'][0][0]+=1.
        with self.assertRaisesRegex(ValueError,'不能更換'):self.build(camera_profile=profile)
        self.parent['protocol']['joint_delta_bounds_from_original_origin']['elbow_flex'][0]-=100
        workspace._seal(self.parent['protocol'])
        self.candidate['protocol_sha256']=self.parent['protocol']['contract_sha256']
        display=self.candidate.pop('frame');identity=self.candidate.pop('candidate_id')
        workspace._seal(self.candidate);self.candidate.update(frame=display,candidate_id=identity)
        with self.assertRaisesRegex(ValueError,'不能更換'):self.build()

    def test_fresh_pose_cannot_rebase_a_later_unrecorded_move(self):
        self.current['shoulder_pan']+=30
        with self.assertRaisesRegex(ValueError,'保持範圍'):self.build()

    def test_failed_pose_must_belong_to_the_original_segment(self):
        self.failure['actual']['wrist_roll']+=30
        with self.assertRaisesRegex(RuntimeError,'偏離保持位置'):self.build()

    def test_old_frame_and_active_hardware_owner_are_rejected(self):
        frame=dict(fixtures.WorkspaceProtocolTests.frame(self.current),capture_monotonic=time.monotonic()-6)
        with self.assertRaisesRegex(ValueError,'過期'):self.build(frame=frame)
        self.grasp.robot.pwm_owner='active-contact'
        with self.assertRaisesRegex(ValueError,'空夾爪'):self.build()


class WorkspaceSettlingRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        WorkspaceRecoveryTests.setUpClass()
        first = WorkspaceRecoveryTests()
        first.setUp()
        cls.protocol = first.build()
        cls.model, cls.profile = first.model, first.profile
        cls.start = first.current

    def setUp(self):
        first_grasp = SimpleNamespace(model=self.model, state='idle',
            robot=SimpleNamespace(goals=dict(self.start), pwm_owner=None, moving=False))
        self.candidate = workspace.prepare_workspace_station(first_grasp, self.protocol,
            dict(fixtures.WorkspaceProtocolTests.frame(self.start), camera_generation='restarted-camera'),
            station_index=0, completed_station_ids=[])
        self.candidate.update(candidate_id='failed-elbow', frame={'url':'/first-recovery.jpg'})
        self.parent = dict(session_id=self.protocol['session_id'], protocol=deepcopy(self.protocol),
            completed_station_ids=[], receipts=[], captures=[],
            recovered_from=self.protocol['recovery']['parent_protocol_sha256'],
            inflight=dict(station_id='recovery-elbow-01', candidate_id='failed-elbow'))
        self.failure = dict(error="未到達目標位置，已保持姿勢：{'elbow_flex': 31}",
            time=1790765246.7037828, actual={**self.start, 'elbow_flex':3031},
            goals=deepcopy(self.candidate['segment']['physical_goal_targets']))
        # Include real post-hold drift: the stationary pan must not be moved.
        self.current = {**self.start, 'shoulder_pan':2255, 'elbow_flex':3042}
        self.grasp = SimpleNamespace(model=self.model, state='idle',
            robot=SimpleNamespace(goals=dict(self.current), pwm_owner=None, moving=False))

    def build(self, **overrides):
        frame = overrides.pop('frame', dict(fixtures.WorkspaceProtocolTests.frame(self.current),
            id='fresh-final-retry', camera_generation='new-retry-camera'))
        kw = dict(session_id=self.protocol['session_id'], camera_profile=self.profile,
            camera_id=self.profile['camera_id'], mount_revision=self.profile['mount_revision'])
        kw.update(overrides)
        return build_workspace_settling_recovery_protocol(self.grasp, self.parent, self.candidate,
            self.failure, frame, **kw)

    def test_final_retry_retains_root_origin_and_targets_rebuilds_cad_and_lineage(self):
        before = deepcopy((self.parent, self.candidate, self.failure, self.grasp.robot.goals))
        result = self.build()
        r = result['recovery']
        self.assertEqual((r['kind'], r['attempt']), ('first_station_settling_retry', 2))
        self.assertEqual(r['parent_protocol_sha256'], self.protocol['contract_sha256'])
        self.assertEqual(r['root_original_protocol_sha256'], self.protocol['recovery']['parent_protocol_sha256'])
        self.assertEqual(r['parent_record_sha256'], workspace._hash(self.parent))
        self.assertEqual(r['failure_evidence'], self.failure)
        self.assertEqual(r['failure_evidence_sha256'], workspace._hash(self.failure))
        self.assertEqual(r['failed_station_id'], 'recovery-elbow-01')
        self.assertEqual(r['recovery_start'], self.current)
        self.assertEqual(result['reference']['camera_generation'], 'new-retry-camera')
        self.assertEqual(result['origin'], self.protocol['origin'])
        for field in ('camera_profile_sha256','geometry_revision','origin_limit_profile',
                      'joint_delta_bounds_from_original_origin','maximum_tip_distance_from_original_origin_mm',
                      'maximum_segment_ticks','maximum_segment_tip_mm','maximum_segment_seconds'):
            self.assertEqual(result[field], self.protocol[field])
        for new, old in zip(result['stations'], self.protocol['stations']):
            for field in ('station_id','station_index','capture_id','capture_required','split'):
                self.assertEqual(new[field], old[field])
            self.assertEqual(new['nominal_segment']['fixed_origin'], self.protocol['origin'])
            self.assertTrue(new['nominal_segment']['clearance']['model_table_pass'])
            self.assertFalse(new['external_clearance_verified'])
        first = result['stations'][0]
        self.assertEqual(first['target_joints'], {**self.current, 'elbow_flex':3000})
        self.assertEqual(first['nominal_segment']['active_joints'], ['elbow_flex'])
        self.assertEqual([s['target_joints'] for s in result['stations'][1:]],
                         [s['target_joints'] for s in self.protocol['stations'][1:]])
        self.assertFalse(r['old_path_approval_reused'])
        self.assertFalse(result['motors_commanded'])
        self.assertFalse(result['automatic_station_chaining'])
        self.assertEqual(before, (self.parent, self.candidate, self.failure, self.grasp.robot.goals))

    def test_completed_wrong_attempt_and_third_retry_reject(self):
        for mutate in (lambda p:p.update(completed_station_ids=['recovery-elbow-01']),
                       lambda p:p['protocol']['recovery'].update(attempt=2),
                       lambda p:p['protocol']['recovery'].update(kind='first_station_settling_retry')):
            with self.subTest(mutate=mutate):
                self.parent['protocol'] = deepcopy(self.protocol)
                self.parent['completed_station_ids'] = []
                mutate(self.parent)
                workspace._seal(self.parent['protocol'])
                with self.assertRaisesRegex(ValueError, '再驗證一次'):self.build()

    def test_wrong_failure_kind_and_unfinished_command_reject(self):
        for error in ('追蹤偏差', "卡住：{'elbow_flex': 31}", '未到達目標位置：wrist_roll'):
            self.failure['error'] = error
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, '等待後未到位'):self.build()
        self.failure['error'] = "未到達目標位置，已保持姿勢：{'elbow_flex': 31}"
        self.failure['goals']['elbow_flex'] = 3022
        with self.assertRaisesRegex(ValueError, '已送到原目標'):self.build()

    def test_unrecorded_motion_and_already_arrived_failure_reject(self):
        self.current['elbow_flex'] = 3044
        with self.assertRaisesRegex(ValueError, '保持範圍'):self.build()
        self.current['elbow_flex'] = 3042
        self.failure['actual']['elbow_flex'] = 3010
        with self.assertRaisesRegex(ValueError, '已送到原目標'):self.build()

    def test_changed_calibration_stale_image_and_active_owner_reject(self):
        profile = deepcopy(self.profile); profile['camera_matrix'][0][0] += 1.
        with self.assertRaisesRegex(ValueError, '不能更換'):self.build(camera_profile=profile)
        old = dict(fixtures.WorkspaceProtocolTests.frame(self.current), capture_monotonic=time.monotonic()-6)
        with self.assertRaisesRegex(ValueError, '過期'):self.build(frame=old)
        self.grasp.robot.moving = True
        with self.assertRaisesRegex(ValueError, '空夾爪'):self.build()


if __name__=='__main__':unittest.main()
