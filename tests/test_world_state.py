"""Synthetic paired observations; no camera, serial, network or cached images."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from server.geometry import Kinematics
from server.local_cartesian import ALL
from server.world_state import build_world_state, policy_state, WorldStateMemory
from server.jev import _semantic_json

ROOT=Path(__file__).resolve().parents[1]


def frame(**changes):
    return {'id':'f1','source':'wrist','camera_generation':'g1','capture_monotonic':100.,
            'joints':dict.fromkeys(ALL,2047),**changes}


def obj(name='cup',**changes):
    return {'name':name,'bbox':{'x':.1,'y':.2,'w':.15,'h':.2},'confidence':.8,'frame_index':0,**changes}


class WorldStateTests(unittest.TestCase):
    def setUp(self):
        self.fk=Kinematics(ROOT/'assets/so101_follower_cam.urdf')

    def state(self,f=None,**kw):return build_world_state(f or frame(),self.fk,now=100.,**kw)

    def test_pose_matches_encoder_fk_and_never_claims_jaw_center(self):
        value=self.state();tip=value['robot']['poses_model']['fixed_finger_pad']
        np.testing.assert_allclose(tip['position_m'],self.fk.transforms(frame()['joints'])['gripper_frame_link'][:3,3],atol=1e-6)
        self.assertFalse(tip['metric_accuracy_verified'])
        self.assertIsNone(value['robot']['jaw_center_m'])
        self.assertIsNone(value['robot']['held_object_verified'])
        self.assertFalse(value['motion_authorized'])

    def test_rgb_box_never_becomes_3d_or_known_size(self):
        value=self.state(objects=[obj('a 35 mm bottle')])['objects'][0]
        for key in ('object_center_m','dimensions_3d_m','collision_volume','surface_3d'):self.assertIsNone(value[key])

    def test_new_image_without_perception_is_unknown_not_zero_objects(self):
        value=policy_state(self.state(),now=100.)
        self.assertEqual(value['object_observation_status'],'not_yet_analyzed')
        self.assertIsNone(value['object_count'])
        analyzed=policy_state(self.state(objects=[]),now=100.)
        self.assertEqual(analyzed['object_observation_status'],'analyzed')
        self.assertEqual(analyzed['object_count'],0)

    def test_explicit_updated_empty_analysis_replaces_prior_labels(self):
        memory=WorldStateMemory(self.fk)
        memory.remember(frame(),objects=[obj()],now=100.)
        value=memory.remember(frame(),objects=[],now=101.)
        self.assertEqual(value['objects'],[])
        cached=memory.remember(frame(),now=102.)
        self.assertTrue(cached['objects_analyzed']);self.assertEqual(cached['objects'],[])

    def test_observer_and_incomplete_raw_data_rejected(self):
        for f in (frame(source='engineering_observer'),frame(joints={}),frame(capture_monotonic=float('nan')),frame(capture_monotonic=101.)):
            with self.assertRaises(ValueError):self.state(f)
        for invalid in (True,1.2,float('inf'),-1,4096):
            with self.assertRaises(ValueError):self.state(frame(joints={**frame()['joints'],'elbow_flex':invalid}))

    def test_prior_view_objects_do_not_become_current_objects(self):
        old=frame(id='old',capture_monotonic=99.)
        self.assertEqual(self.state(objects=[obj()],object_frames=[old])['objects'],[])

    def test_image_relations_preserve_referential_ids_without_metric_claim(self):
        state=self.state(objects=[obj(),obj('box',bbox={'x':.6,'y':.6,'w':.2,'h':.2})])
        result=policy_state(state,now=100.)
        ids={o['id'] for o in result['objects']}
        self.assertTrue(result['relations'])
        for relation in result['relations']:
            self.assertIn(relation['subject'],ids);self.assertIn(relation['object'],ids)
            self.assertTrue(relation['predicate'].startswith('image_'))

    def test_normalized_roundoff_and_long_labels_do_not_break_perception(self):
        state=self.state(objects=[obj('字'*150,bbox={'x':.9,'y':.2,'w':.105,'h':.2})])
        self.assertEqual(len(state['objects'][0]['name']),120)
        self.assertAlmostEqual(state['objects'][0]['image_bbox']['w'],.1)

    def spatial(self):
        return {'source':'wrist-only','evidence':{'observations':[{k:frame()[k] for k in ('id','camera_generation','capture_monotonic')}]},
            'stability':'uncertain','needs_more_views':True,'objects':[{'name':'cup','object_index':0,'scene_frame_id':'f1',
            'status':'estimated','position_reference':'median-visible-surface-not-object-center',
            'position_m':[.1,.2,.3],'same_point_p95_disagreement_m':.04}]}

    def test_surface_position_requires_current_binding_and_retains_uncertainty(self):
        value=policy_state(self.state(objects=[obj()],spatial=self.spatial()),now=100.)['objects'][0]
        self.assertEqual(value['surface_position_m'],[.1,.2,.3])
        self.assertEqual(value['surface_consistency_p95_m'],.04)
        self.assertFalse(value['surface_repeatable']);self.assertFalse(value['metric_accuracy_verified'])
        for change in ('evidence','object_index','scene_frame_id'):
            spatial=self.spatial()
            if change=='evidence':spatial.pop('evidence')
            else:spatial['objects'][0][change]='different'
            self.assertIsNone(self.state(objects=[obj()],spatial=spatial)['objects'][0]['surface_3d'])

    def test_foreign_and_mixed_generation_depth_rejected(self):
        spatial=self.spatial();spatial['source']='engineering_observer'
        with self.assertRaises(ValueError):self.state(objects=[obj()],spatial=spatial)
        spatial=self.spatial();spatial['evidence']['observations'].append({'camera_generation':'old'})
        with self.assertRaises(ValueError):self.state(objects=[obj()],spatial=spatial)

    def test_policy_refreshes_age_and_removes_raw_motor_and_path_fields(self):
        state=self.state()
        value=policy_state(state,now=110.)
        self.assertFalse(value['fresh']);self.assertEqual(value['age_s'],10.)
        self.assertNotIn('measured_joint_ticks',str(value));self.assertNotIn('path',value)
        _semantic_json(value)

    def test_memory_enrichment_does_not_freeze_freshness(self):
        memory=WorldStateMemory(self.fk)
        memory.remember(frame(),objects=[obj()],now=100.)
        value=memory.remember(frame(),now=110.)
        self.assertEqual(len(value['objects']),1);self.assertFalse(value['fresh']);self.assertEqual(value['age_s'],10.)

    def test_new_capture_does_not_inherit_objects(self):
        memory=WorldStateMemory(self.fk)
        memory.remember(frame(),objects=[obj()],now=100.)
        memory.remember(frame(id='new',capture_monotonic=101.),now=101.)
        self.assertEqual(memory.latest(camera_generation='g1',camera_fresh=True,now=101.)['objects'],[])

    def test_late_old_generation_cannot_replace_head_or_claim_current(self):
        memory=WorldStateMemory(self.fk)
        memory.remember(frame(id='new',camera_generation='g2',capture_monotonic=101.),now=101.)
        old=memory.remember(frame(),objects=[obj()],now=101.)
        self.assertFalse(old['fresh'])
        self.assertEqual(memory.latest(camera_generation='g2',camera_fresh=True,now=101.)['observation_id'],'new')

    def test_cached_state_becomes_stale_when_camera_dark_or_reconnected(self):
        memory=WorldStateMemory(self.fk);memory.remember(frame(),now=100.)
        for generation,fresh,now in [('g1',False,100.),('g2',True,100.),('g1',True,106.)]:
            value=memory.latest(camera_generation=generation,camera_fresh=fresh,now=now)
            self.assertFalse(value['fresh']);self.assertFalse(value['current_observation_available'])

    def test_observation_id_cannot_reuse_different_encoder_measurement(self):
        memory=WorldStateMemory(self.fk);memory.remember(frame(),objects=[obj()],now=100.)
        changed=frame(joints={**frame()['joints'],'elbow_flex':2050})
        with self.assertRaisesRegex(ValueError,'identity'):memory.remember(changed,now=100.)

    def test_compact_jev_projection_bounds_large_scene(self):
        import json
        objects=[obj('長標籤'*30,bbox={'x':i*.1,'y':i*.1,'w':.05,'h':.05}) for i in range(8)]
        value=policy_state(self.state(objects=objects),now=100.)
        self.assertEqual(value['object_count'],8);self.assertEqual(len(value['objects']),4)
        self.assertTrue(value['objects_truncated'])
        _semantic_json({'world_state':value})
        self.assertLess(len(json.dumps(value,ensure_ascii=False).encode()),8000)


if __name__=='__main__':unittest.main()
