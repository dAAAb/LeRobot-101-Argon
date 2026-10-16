"""Maintenance admission and stationary hold use reads only, never real devices."""
import importlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from server.robot import Robot, JOINTS, MaintenanceRejected
from server.gripper_power import RecoveryRequired


class MaintenanceQuiesceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        root = Path(__file__).resolve().parents[1]
        self.config = json.loads((root / 'config/robot.json').read_text())
        self.calibration = json.loads((root / 'config/follower-calibration.json').read_text())
        self.robot = Robot(self.config, self.calibration, self.directory.name, Mock())
        self.actual = {n: (c['range_min'] + c['range_max']) // 2 for n, c in self.calibration.items()}
        self.goals = {**self.actual, 'elbow_flex': self.actual['elbow_flex'] - 26}
        self.robot.port = Mock(); self.robot.connected = True
        self.robot.goals = dict(self.goals); self.robot.position = dict(self.actual)
        self.robot._read = Mock(side_effect=self.read_register)
        self.robot._write = Mock()
        self.robot.connect = Mock()
        self.app = importlib.import_module('server.app')
        for name, value in dict(robot=self.robot, grasp=SimpleNamespace(state='idle'),
                jobs={}, active=None, job_lock=threading.RLock(), executor=Mock()).items():
            guard = patch.object(self.app, name, value); guard.start(); self.addCleanup(guard.stop)
        self.client = TestClient(self.app.app); self.addCleanup(self.client.close)
        self.headers = {'x-lerobot-token': self.app.TOKEN}
        self.sleep = patch('server.robot.time.sleep'); self.sleep.start(); self.addCleanup(self.sleep.stop)

    def read_register(self, name, address, width=2):
        return {56: self.actual[name], 42: self.goals[name], 40: 1, 65: 0, 66: 1}[address]

    def post(self, path='/api/engineering/maintenance-quiesce', **kwargs):
        return self.client.post(path, json={}, headers=kwargs.pop('headers', self.headers), **kwargs)

    def test_repeated_maintenance_preserves_loaded_goals_and_all_motor_registers(self):
        for _ in range(2):
            response = self.post()
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['preserved_goals'], self.goals)
            self.assertTrue(response.json()['zero_motor_writes'])
            self.assertEqual(len(response.json()['samples']), 4)
        self.assertTrue(self.robot.stop.is_set())
        self.assertTrue(self.robot.maintenance_latched)
        self.assertNotEqual(self.robot.goals['elbow_flex'], self.actual['elbow_flex'])
        self.robot._write.assert_not_called()  # goal, torque, PWM and EEPROM all untouched
        self.robot.connect.assert_not_called()
        port = self.robot.port
        self.robot.close()
        port.closePort.assert_called_once()
        self.robot._write.assert_not_called()
        replacement = Robot(self.config, self.calibration, self.directory.name, Mock())
        self.assertTrue(replacement.stop.is_set())
        self.assertFalse(replacement.maintenance_latched)

    def test_admission_resume_recovery_and_direct_begin_are_blocked_after_latch(self):
        self.assertEqual(self.post().status_code, 200)
        for path in ('/api/resume', '/api/gripper-power/recover', '/api/look'):
            with self.subTest(path=path):
                self.assertEqual(self.post(path).status_code, 409)
        for operation in (self.robot.begin, self.robot.resume, self.robot.recover_gripper_power):
            with self.subTest(operation=operation), self.assertRaisesRegex(RuntimeError, '重新啟動'):
                operation()
        self.app.executor.submit.assert_not_called()
        self.robot._write.assert_not_called()

    def test_direct_motion_wrappers_reject_maintenance_without_error_hold(self):
        self.assertEqual(self.post().status_code, 200)
        operations = (
            lambda: self.robot.glide({'wrist_roll': self.actual['wrist_roll'] + 10}),
            lambda: self.robot.demonstrate_gripper(Mock()),
            lambda: self.robot.gesture('wave'),
            lambda: self.robot.scan(Mock(), Mock()),
            lambda: self.robot.wide_observation(Mock(), Mock()),
            lambda: self.robot.settled_snapshot(Mock()),
            lambda: self.robot.refinement_scan(Mock(), Mock(), dict(self.actual)),
            lambda: self.robot.local_gaze_observation(Mock(), Mock()),
        )
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaises(MaintenanceRejected):
                operation()
            self.robot._write.assert_not_called()
            self.assertEqual(self.robot.goals, self.goals)
            self.assertFalse(self.robot.moving)

    def test_glide_race_before_begin_preserves_maintenance_goals(self):
        admitted, release = threading.Event(), threading.Event()
        validate = self.robot.validate
        def delayed_validate(*args, **kwargs):
            validate(*args, **kwargs)
            if threading.current_thread().name == 'direct-glide' and not admitted.is_set():
                admitted.set()
                if not release.wait(2):raise RuntimeError('test synchronization timeout')
        outcome = []
        def run():
            try:self.robot.glide({'wrist_roll': self.actual['wrist_roll'] + 10})
            except Exception as exc:outcome.append(exc)
        with patch.object(self.robot, 'validate', side_effect=delayed_validate):
            thread = threading.Thread(target=run, name='direct-glide'); thread.start()
            self.assertTrue(admitted.wait(1))
            self.assertEqual(self.post().status_code, 200)
            release.set(); thread.join(2); self.assertFalse(thread.is_alive())
        self.assertEqual(len(outcome), 1)
        self.assertIsInstance(outcome[0], MaintenanceRejected)
        self.robot._write.assert_not_called()
        self.assertEqual(self.robot.goals, self.goals)

    def test_begin_race_after_first_admission_rejects_without_generic_stop_failure(self):
        admitted, release = threading.Event(), threading.Event()
        def pause_connect():
            admitted.set()
            if not release.wait(2):raise RuntimeError('test synchronization timeout')
        self.robot.connect.side_effect = pause_connect
        outcome = []
        def run():
            try:self.robot.begin()
            except Exception as exc:outcome.append(exc)
        thread = threading.Thread(target=run); thread.start(); self.assertTrue(admitted.wait(1))
        self.assertEqual(self.post().status_code, 200)
        release.set(); thread.join(2); self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcome[0], MaintenanceRejected)
        self.robot._write.assert_not_called()

    def test_regular_glide_fault_still_writes_emergency_hold(self):
        with patch.object(self.robot, 'begin', side_effect=RuntimeError('ordinary motion fault')):
            with self.assertRaisesRegex(RuntimeError, 'ordinary motion fault'):
                self.robot.glide({'wrist_roll': self.actual['wrist_roll'] + 10})
        self.assertEqual([(c.args[0], c.args[1], c.args[2]) for c in self.robot._write.call_args_list],
                         [(n, 42, self.actual[n]) for n in JOINTS])
        self.assertTrue(self.robot.stop.is_set())

    def test_connect_checks_every_onboard_homing_offset_without_writing_calibration(self):
        for mismatched in [*JOINTS, None]:
            with self.subTest(joint=mismatched), tempfile.TemporaryDirectory() as runtime:
                robot = Robot(self.config, self.calibration, runtime, Mock())
                port = Mock()
                def read(name, address, width=2):
                    offset = self.calibration[name]['homing_offset'] + int(name == mismatched)
                    raw = (0x800 | -offset) if offset < 0 else offset
                    return {56: self.actual[name], 3: 777, 31: raw, 33: 0,
                        9: self.calibration[name]['range_min'],
                        11: self.calibration[name]['range_max']}[address]
                robot._read = Mock(side_effect=read); robot._write = Mock()
                try:
                    with patch('server.robot.PortHandler', return_value=port), patch('server.robot.PacketHandler'):
                        if mismatched is None:
                            robot.connect()
                            self.assertTrue(robot.connected)
                            self.assertEqual([c.args[0] for c in robot._read.call_args_list if c.args[1] == 31], JOINTS)
                        else:
                            with self.assertRaisesRegex(RuntimeError, mismatched + ' 機內校準與目前快照不同'):
                                robot.connect()
                            self.assertFalse(robot.connected)
                            port.closePort.assert_called_once()
                    robot._write.assert_not_called()
                finally:
                    if robot.port:robot.port.closePort()
                    if robot.lockfile:robot.lockfile.close()

    def test_running_motion_pwm_contact_and_pending_journal_refuse_quiesce(self):
        for attribute, value in (('moving', True), ('pwm_owner', 'owner')):
            setattr(self.robot, attribute, value)
            self.assertEqual(self.post().status_code, 409)
            setattr(self.robot, attribute, False if attribute == 'moving' else None)
        self.app.grasp.state = 'holding'
        self.assertEqual(self.post().status_code, 409)
        self.app.grasp.state = 'idle'
        self.app.jobs['pending'] = {'status': 'running'}
        self.assertEqual(self.post().status_code, 409)
        self.app.jobs.clear()
        with patch.object(self.robot.gripper_power, 'assert_motion_allowed', side_effect=RecoveryRequired('pending contact')):
            self.assertEqual(self.post().status_code, 409)
        self.assertFalse(self.robot.maintenance_latched)
        self.robot._write.assert_not_called()

    def test_changed_goals_or_drifting_actual_never_claim_a_preserved_hold(self):
        calls = 0
        def changing(name, address, width=2):
            nonlocal calls
            if name == 'shoulder_pan' and address == 56:
                calls += 1
                if calls == 2:
                    self.goals['elbow_flex'] += 1
            return self.read_register(name, address, width)
        self.robot._read.side_effect = changing
        self.assertEqual(self.post().status_code, 409)
        self.assertFalse(self.robot.maintenance_latched)
        calls = 0
        def drifting(name, address, width=2):
            nonlocal calls
            if name == 'shoulder_pan' and address == 56:
                calls += 1
                if calls == 2:
                    self.actual['elbow_flex'] += 3
            return self.read_register(name, address, width)
        self.robot._read.side_effect = drifting
        self.assertEqual(self.post().status_code, 409)
        self.assertFalse(self.robot.maintenance_latched)
        self.robot._write.assert_not_called()

    def test_quiesce_lock_closes_submit_resume_recovery_race_window(self):
        reading, finish = threading.Event(), threading.Event()
        first = True
        def paused(name, address, width=2):
            nonlocal first
            if first:
                first = False; reading.set()
                if not finish.wait(2):
                    raise RuntimeError('test synchronization timeout')
            return self.read_register(name, address, width)
        self.robot._read.side_effect = paused
        outcomes = {}
        def run(name, operation):
            try:outcomes[name] = operation()
            except Exception as exc:outcomes[name] = exc
        latch = threading.Thread(target=run, args=('latch', self.app.maintenance_quiesce))
        latch.start(); self.assertTrue(reading.wait(1))
        competitors = [threading.Thread(target=run, args=(name, fn)) for name, fn in (
            ('submit', lambda: self.app.submit('test', lambda j: None, allow_stopped=True)),
            ('resume', self.app.resume), ('recover', self.app.recover_gripper_power),
            ('direct_begin', self.robot.begin))]
        for thread in competitors:thread.start()
        self.assertEqual(outcomes, {})
        finish.set(); latch.join(2)
        for thread in competitors:thread.join(2); self.assertFalse(thread.is_alive())
        self.assertIsInstance(outcomes['latch'], dict)
        for name in ('submit', 'resume', 'recover'):
            self.assertIsInstance(outcomes[name], HTTPException)
            self.assertEqual(outcomes[name].status_code, 409)
        self.assertIsInstance(outcomes['direct_begin'], RuntimeError)
        self.robot._write.assert_not_called()
        self.app.executor.submit.assert_not_called()

    def test_emergency_stop_still_holds_actual_and_csrf_is_required(self):
        self.assertEqual(self.post(headers={}).status_code, 403)
        self.assertEqual(self.post().status_code, 200)
        response = self.post('/api/stop')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['held'])
        self.assertEqual([(c.args[0], c.args[1], c.args[2]) for c in self.robot._write.call_args_list],
                         [(n, 42, self.actual[n]) for n in JOINTS])


if __name__ == '__main__':
    unittest.main()
