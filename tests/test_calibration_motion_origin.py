"""Calibration motion excursion origins survive processes and camera reconnects."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest

from server.calibration_store import CalibrationSessionStore


class CalibrationMotionOriginTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = CalibrationSessionStore(self.root)
        self.session = self.store.start(camera_id='wrist-a', mount_revision='mount-1',
            geometry_revision='geometry-1', board_static=True)['session_id']
        self.directory = self.store.root / self.session
        self.manifest = self.directory / 'session.json'
        self.origin_path = self.directory / 'motion-origin.json'

    def frame(self, number=1, generation='generation-1'):
        joints = dict(shoulder_pan=2294 + number - 1, shoulder_lift=938,
                      elbow_flex=3038, wrist_flex=2192, wrist_roll=747, gripper=1859)
        stamp = number * 3.
        return {'id': f'frame-{number}', 'source': 'wrist', 'camera_id': 'wrist-a',
            'camera_generation': generation, 'sequence': number, 'capture_monotonic': stamp,
            'image_size': [1920, 1080], 'joints': joints,
            'joint_capture_span_ticks': 0, 'image_wait_joint_span_ticks': 0,
            'joint_observation_bracket_monotonic': [stamp - .02, stamp + .02],
            'settled_joint_window': [{'read_start_monotonic': stamp - .5,
                'read_end_monotonic': stamp - .49, 'joints': dict(joints)},
                {'read_start_monotonic': stamp - .1, 'read_end_monotonic': stamp - .09,
                 'joints': dict(joints)}]}

    def mutate_session(self, **fields):
        session = json.loads(self.manifest.read_text())
        session.update(fields)
        self.manifest.write_text(json.dumps(session))

    def test_restart_reuses_first_actual_pose_without_rewriting_session_or_origin(self):
        manifest_before = self.manifest.read_bytes()
        first = self.frame()
        origin = self.store.motion_origin(self.session, first)
        origin_before = self.origin_path.read_bytes()
        first['joints']['elbow_flex'] = 1
        origin['joints']['wrist_roll'] = 1
        restarted = CalibrationSessionStore(self.root)
        reused = restarted.motion_origin(self.session, self.frame(10))
        self.assertEqual(reused['joints'], self.frame()['joints'])
        self.assertEqual(reused['frame_id'], 'frame-1')
        self.assertEqual(reused['geometry_revision'], 'geometry-1')
        self.assertEqual(self.origin_path.read_bytes(), origin_before)
        self.assertEqual(self.manifest.read_bytes(), manifest_before)
        self.assertEqual(restarted.status(self.session)['collected_view_count'], 0)
        self.assertEqual(list((self.directory / 'frames').iterdir()), [])

    def test_empty_session_reconnect_keeps_origin_but_binds_current_generation(self):
        self.store.motion_origin(self.session, self.frame())
        original = self.origin_path.read_bytes()
        restarted = CalibrationSessionStore(self.root)
        returned = restarted.motion_origin(self.session, self.frame(3, 'generation-2'))
        self.assertEqual(returned['joints'], self.frame()['joints'])
        self.assertEqual(returned['generation'], 'generation-2')
        self.assertEqual(returned['origin_camera_generation'], 'generation-1')
        self.assertEqual(self.origin_path.read_bytes(), original)

    def test_accepted_views_block_reconnected_camera_without_rebasing(self):
        self.store.motion_origin(self.session, self.frame())
        original = self.origin_path.read_bytes()
        self.mutate_session(camera_generation='generation-1', frames=[self.frame()])
        with self.assertRaisesRegex(ValueError, '重新連接'):
            self.store.motion_origin(self.session, self.frame(3, 'generation-2'))
        self.assertEqual(self.origin_path.read_bytes(), original)
        self.assertEqual(self.store.motion_origin(self.session, self.frame(3))['joints'],
                         self.frame()['joints'])

    def test_first_saved_plan_can_migrate_without_a_live_camera_or_jpeg(self):
        saved_plan = {'calibration_motion': {'frame': self.frame()}}
        json_path = self.root / 'saved-first-plan.json'
        json_path.write_text(json.dumps(saved_plan))
        migrated = self.store.motion_origin(self.session,
            json.loads(json_path.read_text())['calibration_motion']['frame'])
        returned = CalibrationSessionStore(self.root).motion_origin(self.session, self.frame(7, 'new-generation'))
        self.assertEqual(returned['joints'], migrated['joints'])
        self.assertEqual(returned['joints']['elbow_flex'], 3038)

    def test_foreign_or_unsettled_snapshots_never_create_an_origin(self):
        for update in ({'source': 'D435'}, {'camera_id': 'external'}, {'board_session_id': 'b' * 32},
                       {'joint_capture_span_ticks': 4}, {'settled_joint_window': []},
                       {'joints': {'elbow_flex': 2000}}):
            with self.subTest(update=update):
                frame = self.frame(); frame.update(update)
                with self.assertRaises(ValueError): self.store.motion_origin(self.session, frame)
                self.assertFalse(self.origin_path.exists())

    def test_invalid_session_cannot_set_or_reuse_origin(self):
        self.store.motion_origin(self.session, self.frame())
        original = self.origin_path.read_bytes()
        for updates in ({'status': 'invalidated'}, {'status': 'solving'},
                        {'status': 'collecting', 'board_static_declared': False}):
            self.mutate_session(**updates)
            with self.assertRaises(ValueError): self.store.motion_origin(self.session, self.frame(2))
            self.assertEqual(self.origin_path.read_bytes(), original)

    def test_geometry_or_mount_change_cannot_reuse_or_replace_saved_origin(self):
        self.store.motion_origin(self.session, self.frame())
        original = self.origin_path.read_bytes()
        for field in ('geometry_revision', 'mount_revision'):
            old = json.loads(self.manifest.read_text())[field]
            self.mutate_session(**{field: 'changed'})
            with self.assertRaises(ValueError): self.store.motion_origin(self.session, self.frame(2))
            self.assertEqual(self.origin_path.read_bytes(), original)
            self.mutate_session(**{field: old})

    def test_corrupt_origin_is_not_silently_replaced_by_current_pose(self):
        for contents in ('[1,2]', '{', '{"schema_version":1}', '{"sha256":"tampered"}'):
            self.origin_path.write_text(contents)
            with self.assertRaises(ValueError): self.store.motion_origin(self.session, self.frame(2))
            self.assertEqual(self.origin_path.read_text(), contents)

    def test_two_store_instances_cannot_replace_each_others_first_origin(self):
        barrier = threading.Barrier(2)
        def initialize(number):
            store = CalibrationSessionStore(self.root)
            barrier.wait()
            return store.motion_origin(self.session, self.frame(number))
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(initialize, (1, 2)))
        self.assertEqual(results[0]['joints'], results[1]['joints'])
        self.assertEqual(results[0]['frame_id'], results[1]['frame_id'])
        self.assertIn(results[0]['frame_id'], ('frame-1', 'frame-2'))
        self.assertEqual(list(self.directory.glob('motion-origin-*.tmp')), [])

    def test_new_session_has_separate_origin(self):
        self.store.motion_origin(self.session, self.frame())
        new = self.store.start(camera_id='wrist-a', mount_revision='mount-1',
            geometry_revision='geometry-1', board_static=True)['session_id']
        later = self.frame(8)
        self.assertEqual(self.store.motion_origin(new, later)['joints'], later['joints'])
        self.assertEqual(self.store.motion_origin(self.session, later)['joints'], self.frame()['joints'])


if __name__ == '__main__':
    unittest.main()
