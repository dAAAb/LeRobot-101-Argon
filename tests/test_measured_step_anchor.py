"""Real WristGrasp.move; synthetic images, fake geometry/encoders/goal ledger.
No Robot/app import, network, files containing photos, or hardware I/O.
"""
import copy,tempfile,threading,types,unittest
from unittest.mock import patch
import numpy as np
from server import grasp as gm
from server.local_feedback import check_bounds

NAMES=gm.NAMES
ORIGIN={'shoulder_pan':2198,'shoulder_lift':1182,'elbow_flex':2946,'wrist_flex':2270,'wrist_roll':985,'gripper':1857}
HELD={'shoulder_pan':2199,'shoulder_lift':1176,'elbow_flex':2937,'wrist_flex':2274,'wrist_roll':989,'gripper':1857}
STALE={'shoulder_pan':2,'shoulder_lift':0,'elbow_flex':-3,'wrist_flex':-6,'wrist_roll':0,'gripper':0}
DELTA={'shoulder_pan':6,'shoulder_lift':-1,'elbow_flex':-1,'wrist_flex':2,'wrist_roll':2}

class Stop:
    def __init__(self):self.flag=False
    def is_set(self):return self.flag
    def set(self):self.flag=True

class FakeRobot:
    def __init__(self):
        self.actual=dict(ORIGIN);self.goals=dict(HELD);self.bias=dict(STALE)
        self.stop=Stop();self.io=threading.RLock();self.sequence=0;self.drift_at=None
        self.camera=types.SimpleNamespace(generation='synthetic',fresh=lambda:True)
        self.glide_calls=[];self.accepted_residuals=[];self.hold_calls=0
    def begin(self,*,allow_rearm):
        assert allow_rearm is False
        if self.stop.is_set():raise RuntimeError('prior operation remains stopped')
    def read(self):return dict(self.actual)
    def _read(self,n,address,width=2):
        if address==63:return 30
        if address in (60,65,69):return 0
        if address==42:return self.goals[n]
        raise AssertionError('unexpected fake telemetry register')
    def frame(self):
        self.sequence+=1
        if self.sequence==self.drift_at:self.actual['elbow_flex']+=4
        return {'source':'wrist','camera_generation':'synthetic','sequence':self.sequence,
                'capture_monotonic':self.sequence*.1,'path':'synthetic-memory-only',
                'joints':dict(self.actual),'settled_joint_window':[dict(self.actual)]*6}
    def settled_snapshot(self,snapshot):return self.frame()
    def glide(self,logical,**kwargs):
        held=dict(self.goals);bias=dict(self.bias)
        physical={n:logical[n]+bias[n] for n in NAMES}
        check_bounds(self.actual,kwargs['measured_bounds'])
        check_bounds({**self.actual,**physical},kwargs['raw_bounds'])
        kwargs['guard'](self.actual);kwargs['guard']({**self.actual,**physical})
        self.glide_calls.append({'logical':dict(logical),'physical':physical,'held_before':held,
                                 'bias':bias,'actual_before':dict(self.actual)})
        self.goals.update(physical)
        residual=self.accepted_residuals.pop(0) if self.accepted_residuals else {}
        self.actual.update({n:logical[n]+residual.get(n,0) for n in NAMES})
        check_bounds(self.actual,kwargs['measured_bounds'])
        assert kwargs['endpoint_accept'](self.actual), 'fake endpoint must satisfy accepted Cartesian criterion'
        kwargs['feedback_observe']({'phase':'cartesian_endpoint','status':'accepted'})
        return dict(self.actual)
    def hold(self):self.stop.set();self.hold_calls+=1

class FakeModel:
    def __init__(self,robot):self.robot=robot;self.proposals=[];self.corrupt=False;self.change_ledger=False
    def tip(self,raw):
        # Pan is the measured Cartesian axis; other axes deliberately cancel
        # from this fake tip to model accepted nonzero scalar residuals.
        t=np.eye(4);t[0,3]=(raw['shoulder_pan']-ORIGIN['shoulder_pan'])/3000
        return t
    def floor_minimum(self,raw):return {'base_link':0,'fake_tool':.1}
    def propose_adaptive(self,raw,delta,*,held_goals,uncertainty_ticks):
        target={**raw,**{n:raw[n]+DELTA[n] for n in NAMES}}
        physical={n:held_goals[n]+DELTA[n] for n in NAMES}
        if self.corrupt:physical['wrist_flex']+=1
        out={'requested_delta_m':delta.tolist(),'joint_targets':target,'physical_goal_targets':physical}
        self.proposals.append(copy.deepcopy(out))
        if self.change_ledger:self.robot.goals['elbow_flex']+=1
        return out

class MeasuredStepAnchor(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r=FakeRobot();self.m=FakeModel(self.r)
        self.g=gm.WristGrasp(self.r,object(),self.r.frame,self.tmp.name);self.g._model=self.m
        self.g.vector=lambda *a:np.array([.002,0,0])
        self.records=[];self.g.record=lambda r:self.records.append(copy.deepcopy(r))
        self.image=patch.object(gm.cv2,'imread',return_value=np.zeros((24,24),np.uint8))
        self.image.start();self.addCleanup(self.image.stop)
    def move(self):return self.g.move('right',2,self.r.frame(),lambda message:None)
    def test_recorded_stale_elbow_and_wrist_targets_are_replaced_without_changing_first_physical_command(self):
        held=dict(self.r.goals);old_bias=dict(self.r.bias);origin=dict(self.r.actual)
        self.move();call=self.r.glide_calls[0];proposal=self.m.proposals[0]
        self.assertEqual(call['held_before'],held,'bookkeeping rebase must not rewrite goals')
        self.assertEqual(call['logical'],{n:proposal['joint_targets'][n] for n in NAMES})
        self.assertEqual(call['physical'],proposal['physical_goal_targets'])
        self.assertEqual(call['logical']['elbow_flex'],2945)
        self.assertEqual(call['logical']['wrist_flex'],2272)
        self.assertEqual(call['bias']['elbow_flex'],-9)
        self.assertEqual(call['bias']['wrist_flex'],4)
        self.assertEqual(proposal['physical_goal_targets']['elbow_flex']-old_bias['elbow_flex'],2939)
        self.assertEqual(proposal['physical_goal_targets']['wrist_flex']-old_bias['wrist_flex'],2282)
        self.assertEqual(call['physical'],{n:held[n]+proposal['joint_targets'][n]-origin[n] for n in NAMES})
        self.assertEqual(call['bias']['gripper'],0);self.assertEqual(self.r.hold_calls,0)
    def test_second_accepted_step_reanchors_large_cancelling_scalar_residuals(self):
        self.r.accepted_residuals=[{'elbow_flex':6,'wrist_flex':-10,'shoulder_lift':6,'wrist_roll':-4},{}]
        self.move();held_after_first=dict(self.r.goals);measured_after_first=dict(self.r.actual)
        self.move();second=self.r.glide_calls[1];proposal=self.m.proposals[1]
        self.assertEqual(second['held_before'],held_after_first)
        self.assertEqual(second['logical'],{n:proposal['joint_targets'][n] for n in NAMES})
        self.assertEqual(second['physical'],{n:held_after_first[n]+DELTA[n] for n in NAMES})
        self.assertEqual(second['bias'],{n:held_after_first[n]-measured_after_first[n] if n in NAMES else 0 for n in held_after_first})
        self.assertEqual(second['bias']['wrist_flex'],14)
        self.assertEqual(len([r for r in self.records if r['type']=='measured_step_anchor']),2)
        self.assertEqual(self.r.hold_calls,0)
    def test_proposal_inconsistent_with_held_goals_is_rejected_before_rebase_or_motion(self):
        self.m.corrupt=True;old_bias=dict(self.r.bias);old_goals=dict(self.r.goals)
        with self.assertRaisesRegex(RuntimeError,'補償不一致'):self.move()
        self.assertEqual(self.r.glide_calls,[]);self.assertEqual(self.r.bias,old_bias);self.assertEqual(self.r.goals,old_goals)
    def test_goal_ledger_changed_after_planning_is_rejected_before_rebase(self):
        self.m.change_ledger=True;old_bias=dict(self.r.bias)
        with self.assertRaisesRegex(RuntimeError,'補償不一致'):self.move()
        self.assertEqual(self.r.glide_calls,[]);self.assertEqual(self.r.bias,old_bias)
    def test_fresh_capture_drift_does_not_rebase_or_dispatch(self):
        self.r.drift_at=3;old_bias=dict(self.r.bias);old_goals=dict(self.r.goals)
        with self.assertRaisesRegex(ValueError,'姿勢已改變'):self.move()
        self.assertEqual(self.r.glide_calls,[]);self.assertEqual(self.r.bias,old_bias);self.assertEqual(self.r.goals,old_goals)
    def test_stopped_prior_operation_cannot_be_rebased_into_a_new_move(self):
        self.r.stop.set();old_bias=dict(self.r.bias)
        with self.assertRaisesRegex(RuntimeError,'prior operation'):self.move()
        self.assertEqual(self.r.glide_calls,[]);self.assertEqual(self.r.bias,old_bias)
    def test_excessive_body_hold_offset_is_rejected_without_motion(self):
        self.r.goals['elbow_flex']=self.r.actual['elbow_flex']-55;old_bias=dict(self.r.bias)
        with self.assertRaisesRegex(RuntimeError,'保持補償'):self.move()
        self.assertEqual(self.r.glide_calls,[]);self.assertEqual(self.r.bias,old_bias)

if __name__=='__main__':unittest.main(verbosity=2)
