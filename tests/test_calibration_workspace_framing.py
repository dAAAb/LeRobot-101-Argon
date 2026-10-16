"""Offline wrist framing with full held-goal CAD and immutable survey bounds."""
from copy import deepcopy
import tempfile
from types import SimpleNamespace
import unittest

import test_calibration_workspace_motion as fixtures
from server import calibration_workspace_motion as workspace
from server.calibration_workspace_framing import build_wrist_framing_addendum
from server.calibration_workspace_journal import WorkspaceJournal


class WorkspaceFramingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WorkspaceProtocolTests.setUpClass()
        cls.model = fixtures.WorkspaceProtocolTests.model
        cls.original = deepcopy(fixtures.WorkspaceProtocolTests.protocol)
        cls.original['session_id'] = 'a' * 32
        cls.original = workspace._seal(cls.original)
        cls.frame = staticmethod(fixtures.WorkspaceProtocolTests.frame)

    def setUp(self):
        self.parent = dict(session_id=self.original['session_id'], protocol=deepcopy(self.original),
            completed_station_ids=[s['station_id'] for s in self.original['stations']],
            receipts=[], captures=[], inflight=None)
        self.raw = dict(zip(workspace.ALL, (2367, 2190, 2621, 1388, 1043, 1861)))
        self.held = dict(zip(workspace.ALL, (2369, 2185, 2595, 1390, 1047, 1861)))
        self.grasp = SimpleNamespace(model=self.model, state='idle',
            robot=SimpleNamespace(goals=dict(self.held), pwm_owner=None))
        self.reference = {**self.frame(self.raw), 'camera_id': self.original['camera_id']}
        self.report = dict(status='validated', source='wrist-only',
            method='frozen-camera-workspace-extension', board_session_id=self.original['session_id'],
            calibration_profile=deepcopy(self.original['camera_profile']), validation=dict(board_plane_base=[
                [-.03576743108178293, -.9987978528781468, .03351924760213376, .4423000575914095],
                [-.9993589721272468, .03579855791643649, .0003287550388841841, -.17971723189018596],
                [-.0015283005535607665, -.03348600210695102, -.9994380180683099, -.008283916595846168],
                [0., 0., 0., 1.]]))

    def build(self):
        return build_wrist_framing_addendum(self.grasp, self.parent, self.reference,
            extension_report=self.report, camera_id=self.original['camera_id'],
            mount_revision=self.original['mount_revision'])

    def test_complete_four_view_plan_preserves_held_bias_camera_and_origin(self):
        parent = deepcopy(self.parent)
        plan = self.build()
        workspace._validate(plan, 'engineering_workspace_calibration')
        self.assertEqual(self.parent, parent)
        self.assertEqual(self.grasp.robot.goals, self.held)
        self.assertEqual(plan['origin'], self.original['origin'])
        self.assertEqual(plan['camera_profile'], self.original['camera_profile'])
        self.assertEqual(plan['initial_held_goals'], self.held)
        self.assertEqual(plan['replan']['checkpoint_start_joints'], self.raw)
        self.assertEqual([s['split'] for s in plan['stations']], ['training', 'heldout'] * 2)
        self.assertFalse(plan['automatic_station_chaining'])
        self.assertFalse(plan['motors_commanded'])
        previous = self.raw
        for station in plan['stations']:
            segment = station['nominal_segment']
            self.assertTrue(segment['clearance']['model_table_pass'])
            self.assertEqual(segment['uncertainty_ticks']['elbow_flex'], 26)
            self.assertEqual(segment['physical_goal_targets']['elbow_flex'], 2595)
            self.assertEqual(segment['fixed_origin'], self.original['origin'])
            self.assertTrue(set(station['planned_active_joints']) <= {'wrist_flex', 'wrist_roll'})
            self.assertTrue(any(abs(station['target_joints'][n] - previous[n]) >= 50
                                for n in ('wrist_flex', 'wrist_roll')))
            self.assertEqual(station['nominal_board_projection']['visible_corner_count'], 54)
            self.assertLess(station['nominal_board_projection']['bbox_px'][1][1], 650)
            self.assertFalse(station['external_clearance_verified'])
            previous = station['target_joints']
        with tempfile.TemporaryDirectory() as directory:
            journal = WorkspaceJournal(directory)
            journal._save(parent)
            published = journal.publish_addendum(parent['session_id'], plan)
            self.assertEqual(published['previous_complete_record'], parent)
            self.assertEqual(published['completed_station_ids'], [])

    def test_different_generation_and_incomplete_survey_reject(self):
        self.reference['camera_generation'] = 'new-generation'
        with self.assertRaises(ValueError):
            self.build()
        self.setUp()
        self.parent['completed_station_ids'].pop()
        with self.assertRaises(ValueError):
            self.build()

    def test_recalibrated_camera_is_not_silently_substituted(self):
        self.report['calibration_profile']['handeye'][0][3] += .001
        with self.assertRaisesRegex(ValueError, 'handeye'):
            self.build()
        self.setUp()
        self.report['status'] = 'rejected'
        with self.assertRaises(ValueError):
            self.build()


if __name__ == '__main__':
    unittest.main()
