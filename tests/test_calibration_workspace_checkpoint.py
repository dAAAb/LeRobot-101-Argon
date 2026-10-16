"""Explicit child route publication retains parent photos and interruption."""
from copy import deepcopy
import json
from tempfile import TemporaryDirectory
import unittest

from server import calibration_workspace_motion as workspace
from server.calibration_workspace_journal import WorkspaceJournal


class WorkspaceCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.journal = WorkspaceJournal(self.temp.name)
        self.parent_id, self.child_id = 'a'*32, 'b'*32
        self.raw = dict(shoulder_pan=2181, shoulder_lift=1032, elbow_flex=3042,
                       wrist_flex=2174, wrist_roll=884, gripper=1861)
        stations = [dict(station_id=f's{i}', station_index=i,
            target_joints={**self.raw, 'shoulder_pan': 2200+i*20, 'elbow_flex': 3000-i*20},
            nominal_segment={'active_joints': ['shoulder_pan', 'elbow_flex']},
            capture_id=f'c{i}', capture_required=True, split='training' if i%2 == 0 else 'heldout') for i in range(4)]
        self.protocol = workspace._seal(dict(version=1, kind='engineering_workspace_calibration',
            session_id=self.parent_id, origin=self.raw, reference={'joints':self.raw, 'camera_generation':'old'},
            camera_id='wrist', mount_revision='mount', camera_profile={'unchanged':True}, camera_profile_sha256='profile',
            geometry_revision='model', origin_limit_profile='fixed', joint_delta_bounds_from_original_origin={'same':True},
            maximum_tip_distance_from_original_origin_mm=180, maximum_segment_ticks=180,
            maximum_segment_tip_mm=80, maximum_segment_seconds=30, stations=stations))
        self.parent = self.journal.create(self.parent_id, self.protocol)
        self.parent.update(completed_station_ids=['s0','s1'], captures=[{'id':'photo1'},{'id':'photo2'}],
            receipts=[{'old':'preserved'}], inflight={'station_id':'s2', 'candidate_id':'failed'})
        self.journal._save(self.parent)
        self.checkpoint = {**self.raw, 'shoulder_pan':2345, 'elbow_flex':3036}
        self.child = deepcopy(self.protocol)
        self.child.update(session_id=self.child_id,
            reference={'joints':self.checkpoint, 'camera_generation':'new'}, initial_held_goals=self.checkpoint, stations=[])
        mapping = []
        for index, source in enumerate(stations[2:], 2):
            for phase in ('elbow_prelude','source_target'):
                active = ['elbow_flex'] if phase=='elbow_prelude' else ['shoulder_pan']
                row = {**deepcopy(source), 'station_id':f'{source["station_id"]}-{phase}',
                       'station_index':len(self.child['stations']), 'planned_active_joints':active}
                if phase=='elbow_prelude':row.update(capture_id=None,capture_required=False,split=None)
                self.child['stations'].append(row)
                mapping.append(dict(station_id=row['station_id'],source_station_id=source['station_id'],
                    source_station_index=index, phase=phase, original_target_joints=source['target_joints'],
                    planned_active_joints=active))
        self.child['replan'] = dict(parent_record_sha256=workspace._hash(self.parent),
            parent_protocol_sha256=self.protocol['contract_sha256'],parent_session_id=self.parent_id,
            root_original_origin=self.raw,completed_station_ids=['s0','s1'],
            interrupted_station_id='s2',interrupted_candidate_id='failed',
            checkpoint_start_joints=self.checkpoint, checkpoint_frame=self.child['reference'],
            checkpoint_held_goals=self.checkpoint,source_station_mapping=mapping,
            source_camera_generation='old',checkpoint_camera_generation='new',
            camera_id='wrist',mount_revision='mount',camera_profile_sha256='profile')
        workspace._seal(self.child)
        self.candidate = workspace._seal(dict(version=1,kind='engineering_workspace_station',
            session_id=self.parent_id,station_id='s2',station_index=2,origin=self.raw,
            protocol_sha256=self.protocol['contract_sha256']))
        self.candidate['candidate_id']='failed'
        self.failure = {'error':'phase stopped','time':1.,'actual':self.checkpoint,'goals':self.checkpoint}

    def publish(self, child=None):
        return self.journal.fork_remaining(self.parent_id, child or self.child,
            failed_candidate=self.candidate,failure_evidence=self.failure)

    def test_child_keeps_global_origin_parent_photos_and_failed_stage_unchanged(self):
        before = self.journal._path(self.parent_id).read_bytes()
        child = self.publish()
        self.assertEqual(self.journal._path(self.parent_id).read_bytes(),before)
        self.assertEqual(child['protocol']['origin'],self.raw)
        self.assertEqual(child['protocol']['reference']['joints'],self.checkpoint)
        for field in ('completed_station_ids','captures','receipts'):self.assertEqual(child[field],[])
        self.assertIsNone(child['inflight'])
        self.assertEqual(child['checkpoint_evidence']['failure_evidence'],self.failure)
        self.assertEqual(child['checkpoint_evidence']['parent_record']['captures'],self.parent['captures'])
        archive=self.journal.root/'revisions'/(workspace._hash(self.parent)+'.json')
        self.assertEqual(json.loads(archive.read_text()),self.parent)
        self.assertEqual(archive.stat().st_mode & 0o222,0)
        with self.assertRaises(ValueError):self.publish()

    def test_stale_parent_new_origin_or_changed_split_reject_before_publication(self):
        variants=[]
        changed=deepcopy(self.child);changed['replan']['parent_record_sha256']='stale';variants.append(changed)
        changed=deepcopy(self.child);changed['origin']={**self.raw,'elbow_flex':3036};variants.append(changed)
        changed=deepcopy(self.child);changed['stations'][1]['split']='heldout';variants.append(changed)
        changed=deepcopy(self.child);changed['replan']['checkpoint_start_joints']={**self.checkpoint,'elbow_flex':1};variants.append(changed)
        for child in variants:
            with self.assertRaises(ValueError):self.publish(workspace._seal(child))
        self.assertIsNone(self.journal.load(self.child_id))
        self.assertEqual(self.journal.load(self.parent_id),self.parent)

    def test_missing_targets_extra_captures_and_changed_active_axes_reject(self):
        variants=[]
        changed=deepcopy(self.child);changed['stations'].pop();changed['replan']['source_station_mapping'].pop();variants.append(changed)
        changed=deepcopy(self.child);changed['stations'][0]['capture_required']=True;variants.append(changed)
        changed=deepcopy(self.child);changed['stations'][1]['target_joints']['elbow_flex']-=10;variants.append(changed)
        changed=deepcopy(self.child);changed['stations'][0]['planned_active_joints'].append('wrist_roll');variants.append(changed)
        for child in variants:
            with self.assertRaises(ValueError):self.publish(workspace._seal(child))
        self.assertIsNone(self.journal.load(self.child_id))

    def test_unrelated_failed_segment_or_no_inflight_cannot_publish(self):
        self.candidate['candidate_id']='different'
        with self.assertRaises(ValueError):self.publish()
        self.candidate['candidate_id']='failed'
        self.parent['inflight']=None;self.journal._save(self.parent)
        with self.assertRaises(ValueError):self.publish()
        self.assertIsNone(self.journal.load(self.child_id))

    def test_malformed_failure_poses_and_timestamps_cannot_be_archived(self):
        original=deepcopy(self.failure)
        variants=[]
        for field in ('actual','goals'):
            for pose in ({'not_an_axis':1},{**self.checkpoint,'elbow_flex':4096},
                         {**self.checkpoint,'elbow_flex':True}):
                variants.append({**original,field:pose})
        for stamp in (None,0,-1,True,float('inf')):
            variants.append({**original,'time':stamp})
        for failure in variants:
            self.failure=failure
            with self.subTest(failure=failure),self.assertRaises(ValueError):self.publish()
            self.assertIsNone(self.journal.load(self.child_id))
        self.assertFalse((self.journal.root/'revisions').exists())

    def test_checkpoint_camera_provenance_must_match_both_generations_and_setup(self):
        for field in ('source_camera_generation','checkpoint_camera_generation','camera_id',
                      'mount_revision','camera_profile_sha256'):
            child=deepcopy(self.child);child['replan'][field]='different'
            with self.subTest(field=field),self.assertRaises(ValueError):
                self.publish(workspace._seal(child))
        self.assertIsNone(self.journal.load(self.child_id))
        self.assertFalse((self.journal.root/'revisions').exists())


class WorkspaceCheckpointCadIntegrationTests(unittest.TestCase):
    def test_real_route_builder_publishes_and_prepares_elbow_only_from_checkpoint(self):
        import test_calibration_workspace_route as route_fixtures
        route_fixtures.DecoupledWorkspaceRouteTests.setUpClass()
        fixture=route_fixtures.DecoupledWorkspaceRouteTests()
        fixture.setUp()
        parent_id,child_id='c'*32,'d'*32
        fixture.parent['session_id']=parent_id
        fixture.parent['protocol']['session_id']=parent_id
        workspace._seal(fixture.parent['protocol'])
        child=fixture.build(session_id=child_id)
        source=fixture.parent['protocol']['stations'][3]
        candidate=workspace._seal(dict(version=1,kind='engineering_workspace_station',
            session_id=parent_id,station_id=source['station_id'],station_index=3,
            origin=fixture.original['origin'],protocol_sha256=fixture.parent['protocol']['contract_sha256']))
        candidate['candidate_id']='interrupted'
        with TemporaryDirectory() as root:
            journal=WorkspaceJournal(root)
            journal.create(parent_id,fixture.parent['protocol']);journal._save(fixture.parent)
            record=journal.fork_remaining(parent_id,child,failed_candidate=candidate,
                failure_evidence={'error':'phase stopped','time':1.,'actual':fixture.raw,'goals':fixture.held})
            prepared=workspace.prepare_workspace_station(fixture.grasp,record['protocol'],
                fixture.reference,station_index=0,completed_station_ids=[])
        self.assertEqual(prepared['segment']['active_joints'],['elbow_flex'])
        self.assertEqual(prepared['segment']['measured_origin'],fixture.raw)
        self.assertEqual(prepared['segment']['fixed_origin'],fixture.original['origin'])
        self.assertEqual(prepared['resolved_target_joints']['shoulder_pan'],fixture.raw['shoulder_pan'])
        self.assertEqual(prepared['segment']['physical_goal_targets']['shoulder_pan'],fixture.held['shoulder_pan'])
