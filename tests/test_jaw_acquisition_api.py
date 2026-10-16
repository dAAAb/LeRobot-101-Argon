"""Engineering route ownership and explicit empty-image authorization, no lifespan."""
import importlib
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class JawAcquisitionApiTests(unittest.TestCase):
    def setUp(self):
        self.app=importlib.import_module('server.app');self.queue=[]
        self.robot=SimpleNamespace(stop=threading.Event(),pwm_owner=None,moving=False,
            gripper_power=SimpleNamespace(assert_motion_allowed=Mock()))
        self.grasp=SimpleNamespace(state='idle',model=object())
        self.frame=dict(id='inspected',source='wrist',sequence=42,camera_generation='camera-A',
            captured_at=time.time(),capture_monotonic=time.monotonic())
        values=dict(robot=self.robot,grasp=self.grasp,camera=SimpleNamespace(generation='camera-A'),
            frames={'inspected':self.frame},jobs={},active=None,snapshot_owner=threading.local(),
            executor=SimpleNamespace(submit=self.queue.append),
            CONFIG={'wrist_camera_id':'wrist-test','camera_mount_revision':'mount-A'})
        for name,value in values.items():
            guard=patch.object(self.app,name,value);guard.start();self.addCleanup(guard.stop)
        for name,value in [('server.spatial_runtime.geometry_revision','geometry-A'),
                           ('server.spatial_runtime._read_profile',{'calibration_revision':'camera-profile'})]:
            guard=patch(name,return_value=value);guard.start();self.addCleanup(guard.stop)
        guard=patch('server.jaw_acquisition.collect_jaw_images',return_value={
            'session_id':'acquisition','completed':True,'frame_count':5,'metric_calibrated':False})
        self.collect=guard.start();self.addCleanup(guard.stop)
        self.client=TestClient(self.app.app);self.addCleanup(self.client.close)
        self.headers={'x-lerobot-token':self.app.TOKEN}
    def post(self,**changes):
        return self.client.post('/api/engineering/jaw-calibration/capture',
            json={'frame_id':'inspected','empty_verified':True,**changes},headers=self.headers)
    def test_explicit_request_queues_owned_capture_and_returns_evidence_only(self):
        response=self.post();self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(self.post().status_code,409)
        self.queue.pop()()
        job=self.client.get('/api/jobs/'+response.json()['id']).json()
        self.assertEqual(job['status'],'completed')
        self.assertEqual(job['jaw_calibration_result']['frame_count'],5)
        self.assertFalse(job['jaw_calibration_result']['metric_calibrated'])
        kwargs=self.collect.call_args.kwargs
        self.assertTrue(kwargs['empty_verified']);self.assertEqual(kwargs['camera_id'],'wrist-test')
        self.assertEqual(kwargs['camera_profile'],{'calibration_revision':'camera-profile'})
    def test_missing_false_or_extra_motion_inputs_are_rejected(self):
        self.assertEqual(self.post(empty_verified=False).status_code,422)
        self.assertEqual(self.post(targets=[1000]).status_code,422)
        self.assertEqual(self.post(protocol='arbitrary').status_code,422)
        self.assertEqual(self.client.post('/api/engineering/jaw-calibration/capture',
            json={'frame_id':'inspected'},headers=self.headers).status_code,422)
        self.collect.assert_not_called()
    def test_wide_protocol_is_explicit_and_forwarded(self):
        self.assertEqual(self.post(protocol='wide').status_code,200)
        self.queue.pop()()
        self.assertEqual(self.collect.call_args.kwargs['protocol'],'wide')
    def test_stale_external_and_restarted_frames_cannot_authorize_capture(self):
        for key,value in [('source','D435'),('captured_at',time.time()-31),('camera_generation','old-camera')]:
            before=self.frame[key];self.frame[key]=value
            self.assertEqual(self.post().status_code,409)
            self.frame[key]=before
        self.assertEqual(self.queue,[]);self.collect.assert_not_called()
    def test_stop_motion_holding_and_pending_power_reject_before_queue(self):
        self.robot.stop.set();self.assertEqual(self.post().status_code,409);self.robot.stop.clear()
        self.robot.moving=True;self.assertEqual(self.post().status_code,409);self.robot.moving=False
        self.robot.pwm_owner='held';self.assertEqual(self.post().status_code,409);self.robot.pwm_owner=None
        self.grasp.state='holding';self.assertEqual(self.post().status_code,409);self.grasp.state='idle'
        self.robot.gripper_power.assert_motion_allowed.side_effect=RuntimeError('pending recovery')
        self.assertEqual(self.post().status_code,409)
        self.assertEqual(self.queue,[]);self.collect.assert_not_called()


if __name__=='__main__':unittest.main()
