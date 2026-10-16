"""Content-bound activation and ownership; never starts a device lifespan."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from server.jaw_profile_selection import reviewed_jaw_profile
from server.spatial_runtime import _read_profile
import test_jaw_model_assisted as fixtures


class JawSelectionTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ConditionalProfileTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.directory=self.root/'config/calibration';self.directory.mkdir(parents=True)
        self.profile=self.fixture.profile()
        self.config={'wrist_camera_id':'wrist','camera_mount_revision':'mount-v1',
                     'spatial_calibration_profile':'selected-camera.json'}
        for name,value in [('server.jaw_profile_selection.geometry_revision','geometry-v1'),
                           ('server.jaw_profile_selection._read_profile',self.fixture.camera)]:
            guard=patch(name,return_value=value);guard.start();self.addCleanup(guard.stop)
        self.sha=self.publish(self.profile)

    def publish(self,profile):
        data=json.dumps(profile).encode();sha=hashlib.sha256(data).hexdigest()
        (self.directory/('jaw-'+sha+'-profile.json')).write_bytes(data)
        return sha

    def select(self,sha=None):
        return reviewed_jaw_profile(self.root,sha or self.sha,self.config,object())

    def test_reviewed_estimate_preserves_meaning_and_local_content_identity(self):
        selected,jaw=self.select()
        self.assertEqual(selected,'config/calibration/jaw-'+self.sha+'-profile.json')
        self.assertTrue(jaw.at(1702).planning_eligible)
        self.assertFalse(jaw.at(1702).metric_calibrated)

    def test_content_mutation_prior_observer_and_geometry_mismatch_are_rejected(self):
        path=self.directory/('jaw-'+self.sha+'-profile.json')
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'版本不同'):self.select()
        for change in ({'source':'observer'}, {'geometry_revision':'other'},
                       {'status':'prior','model_assisted':None}):
            profile={**deepcopy(self.profile),**change}
            with self.assertRaises(ValueError):self.select(self.publish(profile))

    def test_camera_change_symlinks_and_traversal_do_not_select(self):
        changed=deepcopy(self.fixture.camera);changed['camera_matrix'][0][0]+=1.
        with patch('server.jaw_profile_selection._read_profile',return_value=changed):
            with self.assertRaisesRegex(ValueError,'相機校準不同'):self.select()
        path=self.directory/('jaw-'+self.sha+'-profile.json')
        target=self.root/'elsewhere.json';path.replace(target);path.symlink_to(target)
        with self.assertRaises(ValueError):self.select()
        with self.assertRaises(ValueError):self.select('../elsewhere')

    def test_runtime_rejects_in_place_edit_after_activation(self):
        path=self.directory/('jaw-'+self.sha+'-profile.json')
        config={'jaw_calibration_profile':str(path)}
        self.assertEqual(_read_profile(config,'jaw_calibration_profile'),self.profile)
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'reviewed version'):
            _read_profile(config,'jaw_calibration_profile')


class JawActivationApiTests(JawSelectionTests):
    def setUp(self):
        super().setUp()
        self.app=importlib.import_module('server.app')
        self.robot=SimpleNamespace(stop=threading.Event(),pwm_owner=None,moving=False,connected=True,
            read=Mock(return_value={'gripper':1702}),
            gripper_power=SimpleNamespace(assert_motion_allowed=Mock()))
        self.robot.stop.set()
        self.save=Mock(return_value={'changed':True,'motors_commanded':False})
        for name,value in dict(ROOT=self.root,CONFIG=self.config,robot=self.robot,
                grasp=SimpleNamespace(state='idle',model=object()),active=None,jobs={},
                _save_spatial_config=self.save).items():
            guard=patch.object(self.app,name,value);guard.start();self.addCleanup(guard.stop)
        self.client=TestClient(self.app.app);self.addCleanup(self.client.close)
        self.headers={'x-lerobot-token':self.app.TOKEN}

    def post(self,**changes):
        return self.client.post('/api/engineering/jaw-calibration/activate',
            json={'profile_sha256':self.sha,**changes},headers=self.headers)

    def test_activation_checks_actual_opening_and_uses_existing_config_transaction(self):
        response=self.post();self.assertEqual(response.status_code,200,response.text)
        self.assertFalse(response.json()['motors_commanded'])
        self.assertTrue(response.json()['current_opening_checked'])
        self.assertFalse(response.json()['jaw_geometry']['metric_calibrated'])
        self.assertTrue(response.json()['jaw_geometry']['planning_eligible'])
        self.robot.read.assert_called_once_with()
        self.save.assert_called_once_with({'jaw_calibration_profile':
            'config/calibration/jaw-'+self.sha+'-profile.json'})

    def test_motion_contact_stop_and_out_of_range_do_not_mutate_config(self):
        self.robot.stop.clear();self.assertEqual(self.post().status_code,409);self.robot.stop.set()
        self.robot.moving=True;self.assertEqual(self.post().status_code,409);self.robot.moving=False
        self.robot.pwm_owner='contact';self.assertEqual(self.post().status_code,409);self.robot.pwm_owner=None
        self.robot.read.assert_not_called()
        self.robot.read.return_value={'gripper':1699}
        self.assertEqual(self.post().status_code,409)
        self.save.assert_not_called()

    def test_request_is_content_id_only_and_requires_token(self):
        self.assertEqual(self.post(path='/tmp/candidate').status_code,422)
        self.assertEqual(self.post(profile_sha256='../file').status_code,422)
        self.assertEqual(self.client.post('/api/engineering/jaw-calibration/activate',
            json={'profile_sha256':self.sha}).status_code,403)
        self.save.assert_not_called();self.robot.read.assert_not_called()


if __name__=='__main__':unittest.main()
