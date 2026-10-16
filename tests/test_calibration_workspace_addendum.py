"""One explicit completed-survey wrist addendum, journal only; no devices."""
from copy import deepcopy
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from server import calibration_workspace_motion as workspace
from server.calibration_workspace_journal import WorkspaceJournal


class WorkspaceAddendumTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.journal=WorkspaceJournal(self.temp.name)
        self.sid='a'*32
        self.raw=dict(shoulder_pan=2368,shoulder_lift=2196,elbow_flex=2756,
            wrist_flex=1457,wrist_roll=1012,gripper=1861)
        self.origin={**self.raw,'shoulder_pan':2181,'shoulder_lift':1032,'elbow_flex':3042,'wrist_flex':2174,'wrist_roll':884}
        self.previous=workspace._seal(dict(version=1,kind='engineering_workspace_calibration',session_id=self.sid,
            origin=self.origin,camera_id='wrist',mount_revision='mount',camera_profile={'same':True},
            camera_profile_sha256='profile',geometry_revision='model',origin_limit_profile='fixed',
            joint_delta_bounds_from_original_origin={'same':True},maximum_tip_distance_from_original_origin_mm=180,
            maximum_segment_ticks=180,maximum_segment_tip_mm=80,maximum_segment_seconds=30,
            reference={'id':'old','camera_generation':'same','joints':self.origin},
            stations=[dict(station_id=f'old-{i}',capture_id=f'oldcap-{i}') for i in range(26)]))
        self.old=self.journal.create(self.sid,self.previous)
        self.old.update(completed_station_ids=[s['station_id'] for s in self.previous['stations']],
            receipts=[{'station_id':f'old-{i}','actual':i} for i in range(26)],
            captures=[{'result':{'accepted':i not in (3,7)},'station_id':f'old-{i}'} for i in range(10)])
        self.journal._save(self.old)
        frame={'id':'new-checkpoint','camera_generation':'same','joints':self.raw}
        self.protocol=deepcopy(self.previous)
        self.protocol.update(reference=frame,initial_held_goals=self.raw,capture_count=4,
            addendum=dict(kind='completed_survey_wrist_addendum',parent_session_id=self.sid,
                parent_record_sha256=workspace._hash(self.old),parent_protocol_sha256=self.previous['contract_sha256']),
            replan=dict(checkpoint_start_joints=self.raw,checkpoint_frame=frame,checkpoint_held_goals=self.raw),
            stations=[dict(station_id=f'extra-{i}',station_index=i,capture_id=f'extracap-{i}',capture_required=True,
                planned_active_joints=['wrist_flex','wrist_roll'],split='training' if i%2==0 else 'heldout',
                target_joints={**self.raw,'wrist_flex':self.raw['wrist_flex']+160+i*5,
                               'wrist_roll':self.raw['wrist_roll']+(-50 if i%2==0 else 50)}) for i in range(4)])
        workspace._seal(self.protocol)

    def publish(self,protocol=None):
        return self.journal.publish_addendum(self.sid,protocol or self.protocol)

    def test_completed_record_including_failed_captures_is_archived_unchanged(self):
        record=self.publish()
        archive=self.journal.root/'revisions'/(workspace._hash(self.old)+'.json')
        self.assertEqual(json.loads(archive.read_text()),self.old)
        self.assertEqual(archive.stat().st_mode & 0o222,0)
        self.assertEqual(record['previous_complete_record'],self.old)
        self.assertEqual(record['previous_complete_record_sha256'],workspace._hash(self.old))
        self.assertEqual(sum(not c['result']['accepted'] for c in record['previous_complete_record']['captures']),2)
        self.assertEqual(record['protocol']['origin'],self.origin)
        self.assertEqual(record['protocol']['session_id'],self.sid)
        for key in ('receipts','captures','completed_station_ids'):self.assertEqual(record[key],[])
        self.assertIsNone(record['inflight'])

    def test_inflight_incomplete_or_second_addendum_is_forbidden(self):
        for change in ({'inflight':{'station_id':'old-25'}}, {'completed_station_ids':self.old['completed_station_ids'][:-1]}):
            self.journal._save({**deepcopy(self.old),**change})
            with self.assertRaises(ValueError):self.publish()
        self.journal._save(self.old)
        first=self.publish()
        first['completed_station_ids']=[s['station_id'] for s in first['protocol']['stations']]
        self.journal._save(first)
        with self.assertRaisesRegex(ValueError,'一次'):self.publish()

    def test_stale_parent_new_session_generation_or_scope_is_rejected(self):
        variants=[]
        for key,value in (('session_id','b'*32),('origin',self.raw),('maximum_segment_ticks',999),
                          ('camera_profile_sha256','other')):
            variants.append({**deepcopy(self.protocol),key:value})
        changed=deepcopy(self.protocol);changed['addendum']['parent_record_sha256']='stale';variants.append(changed)
        changed=deepcopy(self.protocol);changed['reference']['camera_generation']='restarted';variants.append(changed)
        for protocol in variants:
            with self.subTest(protocol=protocol),self.assertRaises(ValueError):self.publish(workspace._seal(protocol))
        self.assertEqual(self.journal.load(self.sid),self.old)
        self.assertFalse((self.journal.root/'revisions').exists())

    def test_nonwrist_motion_duplicate_capture_or_more_than_four_views_rejected(self):
        variants=[]
        changed=deepcopy(self.protocol);changed['stations'][0]['planned_active_joints']=['elbow_flex'];variants.append(changed)
        changed=deepcopy(self.protocol);changed['stations'][0]['target_joints']['shoulder_lift']+=1;variants.append(changed)
        changed=deepcopy(self.protocol);changed['stations'][1]['capture_id']=changed['stations'][0]['capture_id'];variants.append(changed)
        changed=deepcopy(self.protocol);changed['stations'][0]['capture_id']='oldcap-0';variants.append(changed)
        changed=deepcopy(self.protocol);changed['stations'].append(deepcopy(changed['stations'][-1]));variants.append(changed)
        changed=deepcopy(self.protocol);changed['stations'][0]['split']='heldout';variants.append(changed)
        for protocol in variants:
            with self.assertRaises(ValueError):self.publish(workspace._seal(protocol))
        self.assertEqual(self.journal.load(self.sid),self.old)

    def test_archive_precedes_current_save_and_identical_archive_is_retryable(self):
        with patch.object(self.journal,'_save',side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):self.publish()
        self.assertEqual(self.journal.load(self.sid),self.old)
        archive=self.journal.root/'revisions'/(workspace._hash(self.old)+'.json')
        self.assertEqual(json.loads(archive.read_text()),self.old)
        self.publish()
        self.assertEqual(self.journal.load(self.sid)['previous_complete_record'],self.old)
