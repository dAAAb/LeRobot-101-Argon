import unittest
from fastapi.testclient import TestClient
from server.app import app, TOKEN

class LocalRequestTests(unittest.TestCase):
    # No lifespan context: never open a camera or serial port during these tests.
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_external_origin_cannot_read_session(self):
        self.assertEqual(self.client.get('/api/session', headers={'Origin': 'https://example.com'}).status_code, 403)

    def test_mutation_requires_current_session(self):
        for headers in ({}, {'x-lerobot-token': 'expired'}):
            self.assertEqual(self.client.post('/api/camera/snapshot', json={}, headers=headers).status_code, 403)

    def test_correct_token_does_not_override_origin(self):
        self.assertEqual(self.client.post('/api/camera/snapshot', json={}, headers={'x-lerobot-token': TOKEN, 'Origin': 'https://example.com'}).status_code, 403)

    def test_local_session_and_host(self):
        self.assertEqual(self.client.get('/api/session', headers={'Origin': 'http://localhost:3100'}).json()['token'], TOKEN)
        self.assertEqual(self.client.get('/api/session', headers={'Host': 'evil.example'}).status_code, 400)

class JobPayloadTests(unittest.TestCase):
 def test_search_poll_keeps_conclusion_without_mutating_engineering_evidence(self):
  from server.app import public_job
  job={'id':'a','result':{'spatial':{'status':'estimated','grasp_ready':False,'tracking':[{'status':'ok','pairs':[{'i':1,'j':2}]}],'objects':[{'name':'tube','calibration_stability':{'status':'uncertain','folds':[{'pose':1,'p95_difference_m':.01,'pairs':[{'shared_indices':list(range(100))}]}]}}]}}}
  public=public_job(job)
  self.assertFalse(public['result']['spatial']['grasp_ready'])
  fold=public['result']['spatial']['objects'][0]['calibration_stability']['folds'][0]
  self.assertEqual(fold['p95_difference_m'],.01);self.assertNotIn('pairs',fold)
  self.assertIn('pairs',job['result']['spatial']['objects'][0]['calibration_stability']['folds'][0])

class LocalGazeFailureTests(unittest.TestCase):
 def test_partial_capture_is_archived_before_failure_propagates(self):
  from unittest.mock import patch
  import importlib
  m=importlib.import_module('server.app');frame={'id':'first','source':'wrist','sequence':1}
  def fail(snap,report,on_capture):
   on_capture(frame)
   raise RuntimeError('actual wrist did not return')
  with patch.object(m,'submit',lambda kind,fn:fn({})),patch.object(m.robot,'local_gaze_observation',side_effect=fail),patch.object(m,'capture_segment',return_value={'frames':[],'keyframes':['first']}),patch.object(m.spatial,'record_inputs') as save,patch.object(m.brain,'perceive') as brain:
   with self.assertRaisesRegex(RuntimeError,'did not return'):m.observe_local(m.SearchRequest())
   self.assertEqual(save.call_args.args[0],[frame]);self.assertEqual(save.call_args.args[1],[])
   self.assertEqual(save.call_args.kwargs['archive_name'],'wrist-local-gaze-input.zip');brain.assert_not_called()
