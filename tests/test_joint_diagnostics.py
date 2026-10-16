"""Diagnosing a stopped actuator must not move or rearm it."""
import json, tempfile, threading, unittest
from pathlib import Path
from unittest.mock import Mock, patch
from server.robot import Robot, JOINTS

ROOT=Path(__file__).resolve().parents[1]

class JointDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.r=Robot(json.loads((ROOT/'config/robot.json').read_text()),
                     json.loads((ROOT/'config/follower-calibration.json').read_text()),
                     self.temp.name,Mock())
        self.r.connect=Mock(side_effect=lambda:setattr(self.r,'error',''))
        self.r._write=Mock(side_effect=AssertionError('Diagnostics must not write'))
        self.r._read=Mock(side_effect=lambda n,a,w:{56:2968,42:2937,65:8,40:1,
            60:0x480,69:0x8008,62:120,63:33,48:1000}.get(a,0))
        self.r.error='original motion failure'
        self.r.stop.set();self.r.goals={'elbow_flex':2937};self.r.bias={'elbow_flex':-21}

    def tearDown(self):self.temp.cleanup()

    def test_stopped_faulted_motors_are_read_without_changing_control_state(self):
        result=self.r.joint_diagnostics();elbow=result['joints']['elbow_flex']
        self.assertEqual(set(result['joints']),set(JOINTS))
        self.assertEqual(elbow['status'],8)
        self.assertEqual(elbow['goal_error_ticks'],-31)
        self.assertEqual(elbow['signed_pwm'],-128)
        self.assertEqual(elbow['current_milliamps'],-52)
        self.assertEqual(elbow['voltage'],12)
        self.assertLessEqual(elbow['read_start_monotonic'],elbow['read_end_monotonic'])
        self.assertTrue(self.r.stop.is_set());self.assertFalse(self.r.armed)
        self.assertEqual(self.r.error,'original motion failure')
        self.assertEqual(self.r.goals,{'elbow_flex':2937})
        self.assertEqual(self.r.bias,{'elbow_flex':-21})
        self.r._write.assert_not_called()

    def test_moving_refuses_before_connect_or_read(self):
        self.r.moving=True
        with self.assertRaisesRegex(RuntimeError,'忙碌'):self.r.joint_diagnostics()
        self.r.connect.assert_not_called();self.r._read.assert_not_called()

    def test_busy_io_does_not_wait_behind_motion(self):
        acquired=threading.Event();release=threading.Event()
        def occupy():
            with self.r.io:acquired.set();release.wait(2)
        thread=threading.Thread(target=occupy);thread.start();acquired.wait(1)
        try:
            with self.assertRaisesRegex(RuntimeError,'忙碌'):self.r.joint_diagnostics()
            self.r.connect.assert_not_called();self.r._read.assert_not_called()
        finally:release.set();thread.join()

    def test_read_failure_returns_no_success_snapshot_and_releases_lock(self):
        self.r._read.side_effect=RuntimeError('transport failed')
        with self.assertRaisesRegex(RuntimeError,'transport failed'):self.r.joint_diagnostics()
        self.r._write.assert_not_called();self.assertTrue(self.r.stop.is_set())
        # Another thread, not this reentrant-lock owner, must be able to acquire.
        free=[]
        def check():
            ok=self.r.io.acquire(blocking=False);free.append(ok)
            if ok:self.r.io.release()
        thread=threading.Thread(target=check);thread.start();thread.join()
        self.assertEqual(free,[True])

    def test_failed_motion_reason_survives_restart_until_explicit_resume(self):
        self.r.begin=Mock(side_effect=RuntimeError('elbow did not move'))
        self.r.hold=Mock()
        with self.assertRaisesRegex(RuntimeError,'elbow did not move'):
            self.r.glide({'elbow_flex':2968})
        self.r.hold.assert_called_once()
        restarted=Robot(self.r.config,self.r.calibration,self.temp.name,Mock())
        self.assertTrue(restarted.stop.is_set())
        self.assertEqual(restarted.error,'elbow did not move')
        restarted.resume()
        self.assertFalse(restarted.stop.is_set())
        self.assertFalse(restarted.failure_path.exists())
        self.assertEqual(restarted.error,'')

    def test_disk_failure_does_not_delay_hold_or_mask_motion_error(self):
        self.r.begin=Mock(side_effect=RuntimeError('original motion failure'))
        self.r.hold=Mock()
        def failed_write(*args,**kwargs):
            self.r.hold.assert_called_once()
            raise OSError('disk unavailable')
        with patch.object(Path,'write_text',side_effect=failed_write):
            with self.assertRaisesRegex(RuntimeError,'original motion failure'):
                self.r.glide({'elbow_flex':2968})
        self.assertTrue(self.r.stop.is_set())

    def test_failed_resume_keeps_failure_for_next_restart(self):
        self.r.failure_path.write_text(json.dumps({'error':'original motion failure'}))
        with patch.object(self.r.stop,'clear',side_effect=RuntimeError('new stop won')):
            with self.assertRaisesRegex(RuntimeError,'new stop won'):self.r.resume()
        restarted=Robot(self.r.config,self.r.calibration,self.temp.name,Mock())
        self.assertTrue(restarted.stop.is_set())
        self.assertEqual(restarted.error,'original motion failure')

    def test_unreadable_failure_cleanup_relatches_stop(self):
        self.r.failure_path.write_text(json.dumps({'error':'original motion failure'}))
        unlink=Path.unlink
        def fail(path,*args,**kwargs):
            if path==self.r.failure_path:raise OSError('denied')
            return unlink(path,*args,**kwargs)
        with patch.object(Path,'unlink',fail):
            with self.assertRaisesRegex(RuntimeError,'保持停止'):self.r.resume()
        self.assertTrue(self.r.stop.is_set())
        self.assertEqual(self.r.error,'original motion failure')

    def test_http_endpoint_delegates_only_to_read_only_diagnostics(self):
        from fastapi.testclient import TestClient
        from importlib import import_module
        m=import_module('server.app')
        with patch.object(m.robot,'joint_diagnostics',return_value={'read_only':True}) as read:
            client=TestClient(m.app)  # No lifespan: no hardware in tests.
            try:
                self.assertEqual(client.get('/api/robot/diagnostics').json(),{'read_only':True})
                self.assertEqual(client.get('/api/robot/diagnostics',headers={'Origin':'https://example.com'}).status_code,403)
                read.assert_called_once()
                with patch.object(m,'active','releasing'),patch.dict(m.jobs,{'releasing':{'status':'running'}}):
                    self.assertEqual(client.get('/api/robot/diagnostics').status_code,409)
                    read.assert_called_once()
                read.side_effect=RuntimeError('busy')
                self.assertEqual(client.get('/api/robot/diagnostics').status_code,503)
            finally:client.close()
