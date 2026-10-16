"""Elbow-first route transformations with full CAD and no hardware."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import test_calibration_workspace_motion as fixtures
from server import calibration_workspace_motion as workspace
from server.calibration_workspace_route import (
    decouple_workspace_stations, build_decoupled_workspace_protocol,
)


class DecoupledWorkspaceRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.WorkspaceProtocolTests.setUpClass()
        cls.model = fixtures.WorkspaceProtocolTests.model
        cls.original = fixtures.WorkspaceProtocolTests.protocol
        cls.frame = staticmethod(fixtures.WorkspaceProtocolTests.frame)

    def setUp(self):
        self.parent = dict(session_id=self.original['session_id'], protocol=deepcopy(self.original),
            completed_station_ids=[s['station_id'] for s in self.original['stations'][:3]],
            receipts=[{'after': {'joints': self.original['origin']}}], captures=[{'frame': 'old-1'}, {'frame': 'old-2'}],
            inflight=dict(station_id=self.original['stations'][3]['station_id'], candidate_id='interrupted'))
        self.raw = dict(zip(workspace.ALL, (2345, 1171, 3036, 2053, 939, 1861)))
        self.held = {**self.raw, 'elbow_flex': 3029}
        self.grasp = SimpleNamespace(model=self.model, state='idle',
            robot=SimpleNamespace(goals=self.held, pwm_owner=None))
        self.reference = {**self.frame(self.raw), 'camera_generation': 'fresh-child-generation',
                          'camera_id': self.original['camera_id']}

    def build(self, **kwargs):
        arguments = dict(session_id='new-child-session', camera_id=self.original['camera_id'],
            mount_revision=self.original['mount_revision'], reason='Separate loaded elbow from other survey joints.')
        arguments.update(kwargs)
        return build_decoupled_workspace_protocol(self.grasp, self.parent, self.reference, **arguments)

    def test_full_eighteen_station_route_decouples_and_retains_all_twelve_views(self):
        stations, mapping = decouple_workspace_stations(self.model, self.original,
            self.original['stations'], start=self.original['origin'],
            held_goals=self.original['initial_held_goals'])
        self.assertEqual(len(stations), 29)
        self.assertEqual(sum(m['phase'] == 'elbow_prelude' for m in mapping), 11)
        self.assertEqual([s['capture_id'] for s in stations if s['capture_required']],
                         [s['capture_id'] for s in self.original['stations'] if s['capture_required']])
        for station, source in zip(stations, mapping):
            self.assertTrue(station['nominal_segment']['clearance']['model_table_pass'])
            self.assertEqual(station['nominal_segment']['fixed_origin'], self.original['origin'])
            if source['phase'] == 'elbow_prelude':
                self.assertEqual(station['planned_active_joints'], ['elbow_flex'])
                self.assertFalse(station['capture_required'])
            else:
                original = self.original['stations'][source['source_station_index']]
                self.assertEqual(station['target_joints'], original['target_joints'])
                self.assertEqual(station['split'], original['split'])
            active = station['nominal_segment']['active_joints']
            self.assertFalse('elbow_flex' in active and len(active) > 1)

    def test_child_remaining_route_preserves_parent_and_global_origin(self):
        old = deepcopy(self.parent)
        child = self.build()
        workspace._validate(child, 'engineering_workspace_calibration')
        self.assertEqual(self.parent, old)
        self.assertEqual(child['origin'], self.original['origin'])
        self.assertNotEqual(child['origin'], self.raw)
        self.assertEqual(child['reference']['joints'], self.raw)
        self.assertEqual(child['reference']['camera_generation'], 'fresh-child-generation')
        self.assertEqual(child['replan']['checkpoint_held_goals'], self.held)
        self.assertEqual(child['replan']['parent_record_sha256'], workspace._hash(old))
        self.assertEqual(child['replan']['parent_protocol_sha256'], self.original['contract_sha256'])
        self.assertEqual(child['replan']['completed_station_ids'], old['completed_station_ids'])
        self.assertEqual(len(child['stations']), 26)
        self.assertEqual(child['capture_count'], 10)
        captures = [s for s in child['stations'] if s['capture_required']]
        self.assertEqual([s['split'] for s in captures], ['training', 'heldout'] * 5)
        self.assertEqual([s['capture_id'] for s in captures], [f'capture-{n:02d}' for n in range(3, 13)])
        self.assertFalse(child['motors_commanded'])
        self.assertFalse(child['automatic_station_chaining'])
        first = child['stations'][0]['nominal_segment']
        self.assertEqual(first['active_joints'], ['elbow_flex'])
        for n in workspace.ALL:
            if n != 'elbow_flex':
                self.assertEqual(first['joint_targets'][n], self.raw[n])
                self.assertEqual(first['physical_goal_targets'][n], self.held[n])
        for key in ('camera_profile_sha256', 'geometry_revision', 'joint_delta_bounds_from_original_origin',
                    'maximum_tip_distance_from_original_origin_mm'):
            self.assertEqual(child[key], self.original[key])

    def test_checkpoint_generation_change_does_not_allow_new_camera_or_mount(self):
        for kwargs in (dict(camera_id='other'), dict(mount_revision='other'),
                       dict(session_id=self.original['session_id'])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)
        self.reference['camera_id'] = 'other'
        with self.assertRaises(ValueError):
            self.build()

    def test_prefix_and_original_range_still_apply_to_new_checkpoint(self):
        self.parent['completed_station_ids'] = [self.original['stations'][2]['station_id']]
        with self.assertRaises(ValueError):
            self.build()
        self.setUp()
        self.reference['joints']['shoulder_pan'] = self.original['origin']['shoulder_pan'] + 301
        with self.assertRaisesRegex(ValueError, '關節範圍'):
            self.build()


if __name__ == '__main__':
    unittest.main()
