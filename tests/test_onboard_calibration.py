"""Motor-resident calibration inspection is read-only and preserves ownership."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock

from server.robot import JOINTS, Robot

ROOT=Path(__file__).resolve().parents[1]


class OnboardCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.robot=Robot(json.loads((ROOT/'config/robot.json').read_text()),
            json.loads((ROOT/'config/follower-calibration.json').read_text()),self.directory.name,Mock())
        self.original=deepcopy(self.robot.calibration)
        self.robot.models={n:777 for n in JOINTS}
        self.robot.connect=Mock(side_effect=lambda:setattr(self.robot,'error',''))
        self.robot._write=Mock(side_effect=AssertionError('Must not write motor registers'))
        self.robot._read=Mock(side_effect=self.read)
        self.robot.error='preserved failure';self.robot.stop.set()
        self.robot.goals={'elbow_flex':3000};self.robot.bias={'elbow_flex':4}

    def read(self,n,address,width):
        self.assertTrue(self.robot.io._is_owned())
        c=self.original[n];offset=c['homing_offset']
        return {3:777,33:c['drive_mode'],9:c['range_min'],11:c['range_max'],
                31:(abs(offset)|0x800) if offset<0 else offset}[address]

    def test_reads_all_six_actual_registers_and_keeps_snapshot_and_control_state(self):
        before_files={p.name:p.read_bytes() for p in Path(self.directory.name).iterdir() if p.is_file()}
        result=self.robot.read_onboard_calibration()
        self.assertTrue(result['read_only']);self.assertEqual(result['source'],'motor_registers')
        self.assertTrue(result['matches_startup_snapshot']);self.assertEqual(set(result['joints']),set(JOINTS))
        for n,s in result['joints'].items():
            for key in ('id','drive_mode','homing_offset','range_min','range_max'):
                self.assertEqual(s[key],self.original[n][key])
            self.assertEqual(s['model_number'],777);self.assertEqual(s['mismatched_fields'],[])
            self.assertLessEqual(s['read_start_monotonic'],s['read_end_monotonic'])
        self.assertEqual(self.robot._read.call_count,30)
        self.robot._write.assert_not_called();self.assertTrue(self.robot.stop.is_set())
        self.assertFalse(self.robot.armed);self.assertEqual(self.robot.error,'preserved failure')
        self.assertEqual(self.robot.goals,{'elbow_flex':3000});self.assertEqual(self.robot.bias,{'elbow_flex':4})
        self.assertEqual(self.robot.calibration,self.original)
        self.assertEqual(before_files,{p.name:p.read_bytes() for p in Path(self.directory.name).iterdir() if p.is_file()})

    def test_changed_register_is_reported_and_never_adopted(self):
        for field,address,value in [('homing_offset',31,10),('range_min',9,900),
                                    ('range_max',11,2900),('model_number',3,123),('drive_mode',33,1)]:
            with self.subTest(field=field):
                self.robot._read.side_effect=lambda n,a,w: value if n=='elbow_flex' and a==address else self.read(n,a,w)
                result=self.robot.read_onboard_calibration()
                self.assertFalse(result['matches_startup_snapshot'])
                self.assertEqual(result['joints']['elbow_flex']['mismatched_fields'],[field])
                self.assertEqual(result['joints']['elbow_flex'][field],value)
                self.assertEqual(self.robot.calibration,self.original)
                self.assertEqual(self.robot.models,{n:777 for n in JOINTS})
        self.robot._write.assert_not_called()

    def test_active_motion_pwm_and_sdk_owners_are_rejected_before_any_read(self):
        for owner in ('moving','pwm_owner','sdk_owner','port_owner','pwm_state'):
            with self.subTest(owner=owner):
                self.robot.moving=owner=='moving'
                self.robot.pwm_owner='owner' if owner=='pwm_owner' else None
                self.robot._sdk_transaction_owner=object() if owner=='sdk_owner' else None
                self.robot.port=Mock(is_using=owner=='port_owner')
                self.robot.gripper_power.state='holding' if owner=='pwm_state' else 'idle'
                with self.assertRaisesRegex(RuntimeError,'忙碌'):self.robot.read_onboard_calibration()
        self.robot.connect.assert_not_called();self.robot._read.assert_not_called()

    def test_busy_lock_returns_without_waiting_or_connecting(self):
        occupied=threading.Event();release=threading.Event()
        def own():
            with self.robot.io:occupied.set();release.wait(2)
        thread=threading.Thread(target=own);thread.start();occupied.wait(1)
        try:
            with self.assertRaisesRegex(RuntimeError,'忙碌'):self.robot.read_onboard_calibration()
            self.robot.connect.assert_not_called();self.robot._read.assert_not_called()
        finally:release.set();thread.join()

    def test_failure_returns_no_partial_snapshot_and_releases_lock(self):
        self.robot._read.side_effect=RuntimeError('transport failed')
        with self.assertRaisesRegex(RuntimeError,'transport failed'):self.robot.read_onboard_calibration()
        free=[]
        def check():
            ok=self.robot.io.acquire(blocking=False);free.append(ok)
            if ok:self.robot.io.release()
        thread=threading.Thread(target=check);thread.start();thread.join()
        self.assertEqual(free,[True]);self.assertEqual(self.robot.error,'preserved failure')
        self.robot._write.assert_not_called();self.assertEqual(self.robot.calibration,self.original)


if __name__=='__main__':unittest.main()
