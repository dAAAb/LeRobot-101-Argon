"""Empty-jaw capture with real PWM ownership over a stateful fake servo bus."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from server.gripper_power import AtomicJournal, GripperPWMSession, SessionCancelled
from server.jaw_acquisition import TARGETS, PROTOCOLS, collect_jaw_images
from server.spatial_contract import JOINTS
import test_gripper_power as fixtures
import test_local_feedback as control_fixtures


class AcquisitionRobot:
    def __init__(self, root):
        self.runtime=root; self.stop=threading.Event(); self.cancel=threading.Event()
        self.connected=True; self.moving=False; self.pwm_owner=None; self.error=''
        self.body={n:2000 for n in JOINTS[:-1]}; self.goals={**self.body,'gripper':1859}
        self.bias=dict.fromkeys(JOINTS,0)
        self.journal=AtomicJournal(root/'pwm.json'); self.bus=fixtures.FakeBus(self.journal)
        self.bus.reg.update({56:1859,42:1859})
        self.gripper_power=GripperPWMSession(read=self.bus.read,write=self.bus.write,
            lock=threading.RLock(),stop_latch=self.stop.set,journal=self.journal,device_key='fake:6',
            position_bounds=(1400,2100),clock=lambda:self.bus.now,sleep=self.bus.sleep)
        self.camera=SimpleNamespace(generation='camera-A',fresh=lambda:True)
        self.moves=[]; self.begins=[]; self.holds=0; self.sequence=1
        self.failure=None; self.drift=False; self.cancel_during_move=False; self.capture_stop=False
        self.image=root/'image.jpg'; cv2.imwrite(str(self.image),np.full((48,64),127,np.uint8))

    def read(self):return {**self.body,'gripper':self.bus.reg[56]}
    def validate(self,targets):
        if set(targets)!={'gripper'} or not 1412<=targets['gripper']<=2088:raise ValueError('range')
    def begin(self,*,allow_rearm):
        self.begins.append(allow_rearm); self.gripper_power.assert_motion_allowed(self.pwm_owner)
    def hold(self):self.stop.set();self.holds+=1
    def gripper_telemetry(self):
        stamp=time.monotonic()
        return dict(position=self.bus.reg[56],goal=self.bus.reg[42],pwm_limit=self.bus.reg[48],
            status=self.bus.reg[65],temperature_c=30,body_position=dict(self.body),
            capture_monotonic=stamp,read_start_monotonic=stamp-.001,read_end_monotonic=stamp)
    def glide(self,targets,**kwargs):
        self.moves.append((targets.copy(),kwargs.copy()))
        assert set(targets)=={'gripper'} and kwargs['allow_rearm'] is False and kwargs['speed']==50
        assert self.pwm_owner is not None and self.bus.reg[48]==100
        kwargs['measured_guard'](self.read())
        if self.failure:
            self.bus.reg[65]=1
            raise self.failure
        target=targets['gripper']
        self.bus.write(6,42,target,2);self.goals['gripper']=target
        self.bus.reg[56]=target-1  # Actual encoder must not be replaced by label.
        if self.drift:self.body['shoulder_pan']+=6
        if self.cancel_during_move:self.cancel.set()
        kwargs['measured_guard'](self.read())
        kwargs['observe']({'elapsed':1.,'duration':2.})
    def snapshot(self):
        self.sequence+=1;stamp=time.monotonic()
        return dict(id=f'frame-{self.sequence}',path=str(self.image),source='wrist',image_size=[64,48],
            camera_generation=self.camera.generation,sequence=self.sequence,
            captured_at=time.time(),capture_monotonic=stamp)
    def settled_snapshot(self,snap,guard):
        if self.capture_stop:
            self.stop.set();raise RuntimeError('環顧已停止')
        guard(self.read());frame=snap();stamp=frame['capture_monotonic'];joints=self.read()
        frame.update(joints=joints,joint_capture_span_ticks=0,image_wait_joint_span_ticks=0,
            timestamp_basis='jpeg-arrival after settled joint window',settled_duration_s=.6,
            joint_observation_bracket_monotonic=[stamp-.05,stamp+.05],
            settled_joint_window=[dict(read_start_monotonic=stamp-.7,read_end_monotonic=stamp-.6,joints=deepcopy(joints)),
                                  dict(read_start_monotonic=stamp-.2,read_end_monotonic=stamp-.1,joints=deepcopy(joints))])
        return frame


class JawAcquisitionTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.robot=AcquisitionRobot(self.root)
        self.empty=self.robot.snapshot()
    def acquire(self,**kwargs):
        return collect_jaw_images(self.robot,self.robot.snapshot,lambda text:None,self.robot.cancel,self.empty,
            empty_verified=kwargs.pop('empty_verified',True),camera_id='wrist-test',mount_revision='mount',
            geometry_revision='geometry',camera_profile={'calibration_revision':'profile-A'},**kwargs)
    def manifest(self):
        return json.loads(next((self.root/'jaw-calibration-captures').glob('*/session.json')).read_text())
    def test_five_measured_images_then_return_and_verified_cap_restore(self):
        result=self.acquire();m=self.manifest()
        self.assertTrue(result['completed']);self.assertTrue(result['restore_verified'])
        self.assertEqual([t['gripper'] for t,_ in self.robot.moves],[*TARGETS,1859])
        self.assertEqual(self.robot.begins,[False])
        self.assertEqual(m['targets_raw'],list(TARGETS))
        self.assertEqual([f['actual_gripper_raw'] for f in m['frames']],[t-1 for t in TARGETS])
        for target,f in zip(TARGETS,m['frames']):
            self.assertEqual(f['target_raw'],target);self.assertEqual(f['joints']['gripper'],target-1)
            self.assertEqual(f['image_sha256'],hashlib.sha256(Path(f['path']).read_bytes()).hexdigest())
            self.assertEqual({n:f['joints'][n] for n in JOINTS[:-1]},m['body_reference'])
        self.assertEqual(self.robot.bus.reg[48],500)
        self.assertEqual(self.robot.bus.reg[42],self.robot.bus.reg[56])
        self.assertIsNone(self.robot.pwm_owner)
        self.assertFalse(m['metric_calibrated']);self.assertFalse(m['profile_produced'])
        self.assertEqual(m['camera_profile']['calibration_revision'],'profile-A')
        self.assertTrue(all(servo==6 and address in (42,48) for servo,address,_ in self.robot.bus.writes()))
        self.assertEqual(self.robot.bus.writes()[-2:],[(6,42,1858),(6,48,500)])
    def test_cancel_neutralizes_low_cap_without_returning_to_original_opening(self):
        self.robot.cancel_during_move=True
        with self.assertRaises(SessionCancelled):self.acquire()
        self.assertEqual(len(self.robot.moves),1)
        self.assertTrue(self.robot.stop.is_set());self.assertGreater(self.robot.holds,0)
        self.assertEqual(self.robot.bus.reg[48],500)
        self.assertEqual(self.robot.bus.reg[42],self.robot.bus.reg[56])
        self.assertFalse(self.manifest()['completed'])
        self.assertFalse(self.manifest()['return_to_original_completed'])
    def test_wide_protocol_preregisters_split_and_keeps_actual_encoders(self):
        self.acquire(protocol='wide');m=self.manifest()
        self.assertEqual(m['opening_protocol'],dict(name='wide',targets_raw=list(PROTOCOLS['wide']),
            training_targets_raw=[1550,1800,2050],heldout_targets_raw=[1675,1925],declared_before_capture=True))
        self.assertEqual([f['actual_gripper_raw'] for f in m['frames']],[t-1 for t in PROTOCOLS['wide']])
        self.assertEqual(self.robot.bus.reg[48],500)
        self.assertEqual([t['gripper'] for t,_ in self.robot.moves],[*PROTOCOLS['wide'],1859])
    def test_unknown_protocol_rejected_without_motion(self):
        with self.assertRaises(ValueError):self.acquire(protocol='arbitrary')
        self.assertEqual(self.robot.moves,[]);self.assertEqual(self.robot.bus.writes(),[])
    def test_snapshot_stop_is_cancellation_but_io_fault_never_restores_or_returns(self):
        self.robot.capture_stop=True
        with self.assertRaises(SessionCancelled):self.acquire()
        self.assertEqual(self.robot.bus.reg[48],500)
        self.assertEqual(len(self.robot.moves),1)
    def test_servo_fault_preserves_pending_power_barrier(self):
        self.robot.failure=IOError('servo read fault')
        with self.assertRaisesRegex(IOError,'servo read fault'):self.acquire()
        self.assertEqual(self.robot.bus.writes(),[(6,48,100)])
        self.assertEqual(self.robot.gripper_power.state,'blocked')
        self.assertEqual(len(self.robot.moves),1)
        self.assertFalse(self.manifest()['completed'])
        self.assertIsNone(self.manifest()['cleanup_receipt'])
        self.assertIsNone(self.robot.pwm_owner)
    def test_body_drift_aborts_and_does_not_continue_or_restore(self):
        self.robot.drift=True
        with self.assertRaisesRegex(ValueError,'Body pose'):self.acquire()
        self.assertEqual(len(self.robot.moves),1)
        self.assertEqual(self.robot.bus.reg[48],100)
        self.assertEqual(self.robot.gripper_power.state,'blocked')
    def test_missing_empty_verification_stale_frame_stop_and_owner_reject_before_motion(self):
        with self.assertRaises(ValueError):self.acquire(empty_verified=False)
        self.empty['captured_at']-=31
        with self.assertRaises(ValueError):self.acquire()
        self.empty=self.robot.snapshot();self.robot.stop.set()
        with self.assertRaises(RuntimeError):self.acquire()
        self.robot.stop.clear();self.robot.pwm_owner='held'
        with self.assertRaises(RuntimeError):self.acquire()
        self.assertEqual(self.robot.moves,[]);self.assertEqual(self.robot.bus.writes(),[])
    def test_bound_inspection_pose_mismatch_rejects_before_power_entry(self):
        self.empty['joints']=self.robot.read();self.empty['joints']['shoulder_pan']+=6
        with self.assertRaisesRegex(ValueError,'Body pose'):self.acquire()
        self.assertEqual(self.robot.bus.writes(),[])
    def test_restoration_fault_keeps_pending_and_reports_incomplete(self):
        self.robot.bus.write_failure=lambda bus,address,value:'before' if address==48 and value==500 else None
        with self.assertRaises(RuntimeError):self.acquire()
        self.assertFalse(self.manifest()['completed'])
        self.assertEqual(self.robot.gripper_power.state,'blocked')
        self.assertEqual(self.robot.bus.reg[48],100)


class RealGlideAcquisitionTests(unittest.TestCase):
    def setUp(self):
        case=control_fixtures.RobotIntegration();case.setUp();self.addCleanup(case.doCleanups)
        self.case=case;case.held();case.clock.now=10.
        self.robot=case.robot;self.robot.models=dict.fromkeys(JOINTS,777)
        self.robot.goals['gripper']=1859
        case.bus.reg[6].update({3:777,33:0,48:500,56:1859,42:1859,60:0,62:120,69:0})
        self.robot.gripper_power.clock=case.clock.time
        self.robot.gripper_power.sleep=lambda seconds:setattr(case.clock,'now',case.clock.now+seconds)
        self.robot.camera.generation='camera-A'
        self.robot.camera.latest_stamp=lambda:dict(camera_source='wrist',frame_capture_monotonic=case.clock.now)
        self.cancel=threading.Event();self.sequence=0
        self.image=Path(case.tmp.name)/'image.jpg';cv2.imwrite(str(self.image),np.full((48,64),127,np.uint8))
        self.empty=self.snap()
    def snap(self):
        self.sequence+=1
        return dict(id=f'real-glide-frame-{self.sequence}',path=str(self.image),source='wrist',image_size=[64,48],
            camera_generation='camera-A',sequence=self.sequence,captured_at=time.time(),
            capture_monotonic=self.case.clock.now)
    def acquire(self):
        return collect_jaw_images(self.robot,self.snap,lambda message:None,self.cancel,self.empty,
            empty_verified=True,camera_id='wrist',mount_revision='mount',geometry_revision='geometry')
    def test_actual_glide_and_settled_snapshot_only_send_gripper_ram_goals(self):
        result=self.acquire()
        self.assertTrue(result['completed']);self.assertEqual(result['frame_count'],5)
        self.assertTrue(all(servo==6 and address in (42,48) for servo,address,value in self.case.bus.events))
        self.assertEqual(self.case.bus.reg[6][48],500)
        self.assertEqual(self.case.bus.reg[6][56],1859)
        manifest=json.loads(Path(result['manifest_path']).read_text())
        self.assertTrue(all(len(f['settled_joint_window'])==6 for f in manifest['frames']))
    def test_actual_glide_stop_hold_and_power_cleanup_do_not_return_trajectory(self):
        telemetry=self.robot.gripper_telemetry
        def cancel_after_first_goal():
            sample=telemetry()
            if sample['pwm_limit']==100 and sample['position']<1859:self.cancel.set()
            return sample
        self.robot.gripper_telemetry=cancel_after_first_goal
        with self.assertRaises(SessionCancelled):self.acquire()
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.case.bus.reg[6][48],500)
        self.assertLess(self.case.bus.reg[6][56],1859)
        self.assertEqual(self.case.bus.reg[6][42],self.case.bus.reg[6][56])
        self.assertFalse(any(address==40 for _,address,_ in self.case.bus.events))


if __name__=='__main__':unittest.main()
