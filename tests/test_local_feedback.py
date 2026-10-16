"""No serial access: deterministic load-offset plant and pure-control tests."""
import tempfile, threading, unittest
from types import SimpleNamespace
from unittest.mock import patch
from server.local_feedback import ALL_JOINTS, LocalEndpoint, adopt_goal_anchor, check_bounds
from server.robot import Robot, JOINTS
from server.gripper_power import SessionCancelled
from scservo_sdk import COMM_SUCCESS

class PureControl(unittest.TestCase):
    def test_cold_and_repeated_warm_adoption_does_not_ratchet(self):
        actual={n:2000 for n in ALL_JOINTS}; goals=dict(actual)
        goals.update(shoulder_pan=1994,wrist_flex=1990,gripper=1998)
        anchor=adopt_goal_anchor(actual,goals)
        self.assertEqual(anchor['bias']['shoulder_pan'],-6)
        self.assertEqual(anchor['bias']['wrist_flex'],-10)
        self.assertEqual(anchor['bias']['gripper'],0)
        for drift in [1,2,3,4]:
            a={n:v+drift for n,v in actual.items()}
            anchor=adopt_goal_anchor(a,goals,previous_goals=anchor['physical_goals'],previous_bias=anchor['bias'])
            self.assertEqual(anchor['logical_goals']['wrist_flex'],2000)
        with self.assertRaisesRegex(ValueError,'ledger'):
            adopt_goal_anchor(actual,{**goals,'wrist_flex':1989},previous_goals=goals,previous_bias=anchor['bias'])

    def test_restart_reanchors_current_actual_without_guessing_old_intent(self):
        goals={n:2000 for n in ALL_JOINTS}; actual={**goals,'wrist_flex':2009}
        anchor=adopt_goal_anchor(actual,goals)
        self.assertEqual(anchor['logical_goals']['wrist_flex'],2009)
        self.assertEqual(anchor['physical_goals'],goals)
        self.assertEqual(anchor['bias']['wrist_flex'],-9)

    def test_cold_loaded_body_anchor_accepts_33_ticks_within_existing_60_budget(self):
        goals={n:2000 for n in ALL_JOINTS};goals['elbow_flex']=2912
        actual={**goals,'elbow_flex':2945,'gripper':2016}
        anchor=adopt_goal_anchor(actual,goals)
        self.assertEqual(anchor['bias']['elbow_flex'],-33)
        self.assertEqual(anchor['logical_goals']['elbow_flex'],2945)
        self.assertEqual(anchor['physical_goals']['elbow_flex'],2912)
        self.assertEqual(anchor['bias']['gripper'],0)
        self.assertEqual(anchor['logical_goals']['gripper'],2000)

    def test_cold_adoption_does_not_expand_body_bias_or_gripper_offset_limits(self):
        goals={n:2000 for n in ALL_JOINTS}
        for sign in (-1,1):
            with self.subTest(sign=sign):
                actual={**goals,'elbow_flex':2000+sign*60}
                self.assertEqual(adopt_goal_anchor(actual,goals)['bias']['elbow_flex'],-sign*60)
                with self.assertRaises(ValueError):adopt_goal_anchor({**actual,'elbow_flex':2000+sign*61},goals)
                with self.assertRaises(ValueError):adopt_goal_anchor({**goals,'gripper':2000+sign*17},goals)

    def make(self,targets=None,bounds=None):
        targets=targets or {'wrist_flex':2000}
        return LocalEndpoint(targets,{n:0 for n in targets},targets,bounds or {n:(1950,2050) for n in targets})

    def test_actual_and_candidate_bounds_are_both_enforced(self):
        c=self.make(bounds={'wrist_flex':(1996,2024)})
        d=c.observe({'wrist_flex':2010},0);self.assertEqual(d.status,'propose');c.commit(d,0)
        d=c.observe({'wrist_flex':2010},.12)
        self.assertEqual(d.status,'abort');self.assertIn('limit',d.reason)
        with self.assertRaises(ValueError):self.make().observe({'wrist_flex':2100},0)
        for v in [float('nan'),True,2000.5]:
            with self.assertRaises(ValueError):check_bounds({'wrist_flex':v},{'wrist_flex':(1950,2050)})

    def test_four_successful_writes_without_motion_aborts(self):
        c=self.make()
        for i in range(4):
            d=c.observe({'wrist_flex':2010},i*.12)
            self.assertEqual(d.status,'propose');c.commit(d,i*.12)
        d=c.observe({'wrist_flex':2010},.48)
        self.assertEqual(d.status,'abort');self.assertIn('no progress',d.reason)

    def test_one_tick_precision_preserves_four_write_stall_limit(self):
        c=LocalEndpoint({'wrist_flex':2000},{'wrist_flex':0},{'wrist_flex':2000},
                        {'wrist_flex':(1950,2050)},precision=1)
        for i in range(4):
            d=c.observe({'wrist_flex':2003},i*.12)
            self.assertEqual(d.status,'propose')
            self.assertEqual(d.correction.goal,1999-i)
            c.commit(d,i*.12)
        d=c.observe({'wrist_flex':2003},.48)
        self.assertEqual(d.status,'abort');self.assertIn('no progress',d.reason)

    def test_precision_validation_remains_bounded(self):
        for precision in (1,2,3):
            with self.subTest(precision=precision):
                c=LocalEndpoint({'wrist_flex':2000},{'wrist_flex':0},{'wrist_flex':2000},
                                {'wrist_flex':(1950,2050)},precision=precision)
                self.assertEqual(c.precision,precision)
        for precision in (True,False,0,-1,4,1.5,float('nan'),float('inf'),None,'1'):
            with self.subTest(precision=precision):
                with self.assertRaises(ValueError):
                    LocalEndpoint({'wrist_flex':2000},{'wrist_flex':0},{'wrist_flex':2000},
                                  {'wrist_flex':(1950,2050)},precision=precision)

    def test_simultaneous_stability_restarts_after_other_axis_couples(self):
        c=self.make({'wrist_flex':2000,'shoulder_pan':2000})
        d=c.observe({'wrist_flex':2010,'shoulder_pan':2000},0);c.commit(d,0)
        self.assertEqual(c.observe({'wrist_flex':2002,'shoulder_pan':2000},.12).status,'settling')
        self.assertEqual(c.observe({'wrist_flex':2002,'shoulder_pan':2000},.24).status,'settling')
        d=c.observe({'wrist_flex':2002,'shoulder_pan':2006},.36)
        self.assertEqual(d.correction.joint,'shoulder_pan');c.commit(d,.36)
        for t in [.48,.60]:self.assertEqual(c.observe({'wrist_flex':2002,'shoulder_pan':2002},t).status,'settling')
        self.assertEqual(c.observe({'wrist_flex':2002,'shoulder_pan':2002},.72).status,'done')

    def test_unselected_axis_large_reversal_is_not_hidden(self):
        c=self.make({'wrist_flex':2000,'shoulder_pan':2000})
        d=c.observe({'wrist_flex':2010,'shoulder_pan':2009},0);c.commit(d,0)
        d=c.observe({'wrist_flex':2020,'shoulder_pan':1990},.12)
        self.assertEqual(d.status,'abort');self.assertIn('shoulder_pan',d.reason)

    def test_global_budget_is_not_restarted_for_a_new_axis(self):
        c=self.make({'wrist_flex':2000,'shoulder_pan':2000})
        self.assertEqual(c.observe({'wrist_flex':2000,'shoulder_pan':2000},0).status,'settling')
        d=c.observe({'wrist_flex':2000,'shoulder_pan':2006},2.99)
        self.assertEqual(d.status,'propose')
        with self.assertRaisesRegex(RuntimeError,'expired'):c.validate_before_write(d,3.0)
        self.assertEqual(c.controllers['shoulder_pan'].bias,0)

    def test_failed_write_never_commits_or_changes_bias(self):
        c=self.make();d=c.observe({'wrist_flex':2010},0)
        self.assertEqual(c.controllers['wrist_flex'].bias,0)
        self.assertEqual(c.observe({'wrist_flex':2010},.12).status,'abort')
        with self.assertRaises(ValueError):c.commit(d,.12)

class Clock:
    def __init__(self):self.now=0.
    def time(self):return self.now
class FakeStop(threading.Event):
    def __init__(self,clock):super().__init__();self.clock=clock
    def wait(self,seconds=None):self.clock.now+=seconds or 0;return self.is_set()
class FakeCamera:
    def fresh(self):return True
class FakeProtocol:
    def __init__(self,cal):
        self.reg={c['id']:{40:1,42:2000,56:2000,63:30,65:0,66:0} for c in cal.values()}
        self.events=[];self.offset={c['id']:0 for c in cal.values()};self.stuck=set();self.fail_goal=False
    def read(self,port,servo,address):return self.reg[servo][address],COMM_SUCCESS,0
    read2ByteTxRx=read;read1ByteTxRx=read
    def write(self,port,servo,address,value):
        self.events.append((servo,address,value))
        if address==42 and self.fail_goal:return -3001,0
        self.reg[servo][address]=value
        if address==42 and servo not in self.stuck:self.reg[servo][56]=value+self.offset[servo]
        return COMM_SUCCESS,0
    write2ByteTxRx=write;write1ByteTxRx=write

class RobotIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        cal={n:{'id':i+1,'range_min':0,'range_max':4095} for i,n in enumerate(JOINTS)}
        r=self.robot=Robot({'follower_port':'fake://no-hardware','max_ticks_per_second':90,'max_tracking_error_ticks':90},cal,self.tmp.name,FakeCamera())
        self.bus=FakeProtocol(cal);r.ph=self.bus;r.port=SimpleNamespace(is_using=False);r.connected=True
        self.clock=Clock();r.stop=FakeStop(self.clock);r.gripper_power.stop_latch=r.stop.set
        self.addCleanup(patch.stopall);patch('server.robot.time.monotonic',self.clock.time).start()
        self.bounds={n:(1976,2024) for n in JOINTS}
    def held(self):
        self.robot.armed=True;self.robot.goals={n:2000 for n in JOINTS}
    def test_cold_begin_adopts_existing_goals_without_writes_even_allow_rearm_false(self):
        self.bus.reg[1][56]=2006;self.bus.reg[4][56]=2010
        self.robot.begin(allow_rearm=False)
        self.assertEqual(self.bus.events,[])
        self.assertEqual(self.robot.bias['shoulder_pan'],-6)
        self.assertEqual(self.robot.bias['wrist_flex'],-10)
        self.robot.adopt_stationary_hold();self.robot.begin()
        self.assertEqual(self.bus.events,[]);self.assertEqual(self.robot.bias['wrist_flex'],-10)
    def test_unstable_cold_hold_fails_read_only(self):
        # Each adjacent sample moves only one tick, but the entire hold drifts.
        base={n:2000 for n in JOINTS}
        poses=[{**base,'wrist_flex':2000+step} for step in range(6)]
        with patch.object(self.robot,'read',side_effect=poses):
            with self.assertRaisesRegex(RuntimeError,'不穩定'):self.robot.adopt_stationary_hold()
        self.assertEqual(self.bus.events,[])

    def test_loaded_stationary_moving_flag_one_adopts_six_samples_without_writes(self):
        self.bus.reg[3].update({42:2912,56:2945,66:1})
        anchor=self.robot.adopt_stationary_hold()
        self.assertEqual(len(anchor['stationary_samples']),6)
        self.assertGreaterEqual(anchor['stationary_duration_s'],.5)
        self.assertEqual(self.robot.bias['elbow_flex'],-33)
        self.assertEqual(self.robot.goals['elbow_flex'],2912)
        self.assertEqual(anchor['logical_goals']['elbow_flex'],2945)
        self.assertTrue(all(s['servo_health']['elbow_flex']['moving']==1 for s in anchor['stationary_samples']))
        self.robot.begin(allow_rearm=False)
        self.assertEqual(self.bus.events,[])

    def test_stationary_adoption_rejects_changed_goal_and_invalid_moving_value_without_writes(self):
        self.bus.reg[4][66]=2
        with self.assertRaises(RuntimeError):self.robot.adopt_stationary_hold()
        self.assertEqual(self.bus.events,[])
        self.bus.reg[4][66]=0
        original=self.robot._read;reads=0
        def changing_goal(name,address,size=2):
            nonlocal reads
            if name=='wrist_flex' and address==42:
                reads+=1
                return 2000 if reads<2 else 1999
            return original(name,address,size)
        with patch.object(self.robot,'_read',side_effect=changing_goal):
            with self.assertRaisesRegex(RuntimeError,'目標仍在改變'):self.robot.adopt_stationary_hold()
        self.assertEqual(self.bus.events,[])

    def test_pending_contact_still_blocks_read_only_adoption(self):
        from server.gripper_power import RecoveryRequired
        self.bus.reg[6].update({3:777,33:0,48:500})
        self.robot.pwm_owner=self.robot.gripper_power.enter()
        self.robot.gripper_power.mark_contact_possible(self.robot.pwm_owner)
        before=list(self.bus.events)
        with patch.object(self.robot,'connect',side_effect=AssertionError('pending contact must block before connect')):
            with self.assertRaises(RecoveryRequired):self.robot.adopt_stationary_hold()
        self.assertEqual(self.bus.events,before)
    def test_actual_torque_off_still_requires_explicit_rearm_permission(self):
        self.bus.reg[2][40]=0
        with self.assertRaises(RuntimeError):self.robot.begin(allow_rearm=False)
        self.assertEqual(self.bus.events,[])
        self.robot.begin();self.assertEqual(sum(a==40 for _,a,_ in self.bus.events),6)
    def test_return_pan_plus6_wrist_plus10_converges_without_reseeding(self):
        self.held();self.bus.offset[1]=6;self.bus.offset[4]=10
        self.bus.reg[1][56]=2006;self.bus.reg[4][56]=2010
        events=[];guards=[]
        def guard(p):guards.append(dict(p));check_bounds(p,self.bounds)
        actual=self.robot.glide({'shoulder_pan':2000,'wrist_flex':2000},speed=35,feedback=True,
            raw_bounds=self.bounds,guard=guard,feedback_observe=events.append,allow_rearm=False)
        self.assertLessEqual(abs(actual['shoulder_pan']-2000),3)
        self.assertLessEqual(abs(actual['wrist_flex']-2000),3)
        self.assertTrue(-10<=self.robot.bias['wrist_flex']<-3)
        self.assertLess(self.clock.now,3)
        committed=[e for e in events if e['status']=='committed']
        self.assertGreater(len(committed),1)
        for e in committed:
            self.assertEqual(e['goal'],e['logical_targets'][e['joint']]+e['bias'][e['joint']])
            self.assertLessEqual(e['write_read_end_monotonic'],e['write_start_monotonic'])
            self.assertLessEqual(e['write_start_monotonic'],e['write_completed_monotonic'])
        self.assertEqual(events[-1]['status'],'done')
        self.assertEqual([e for e in self.bus.events if e[1]==40],[])
        # A new partial movement keeps acknowledged goals and compensation.
        before=len(self.bus.events);wrist_goal=self.robot.goals['wrist_flex']
        self.robot.glide({'shoulder_pan':2002},speed=35,feedback=True,raw_bounds=self.bounds,guard=guard)
        self.assertFalse(any(servo==4 for servo,address,value in self.bus.events[before:]))
        self.assertEqual(self.robot.goals['wrist_flex'],wrist_goal)
    def test_feedback_refuses_missing_bounds_before_connect(self):
        self.robot.connected=False
        with patch.object(self.robot,'connect',side_effect=AssertionError('must not connect')):
            with self.assertRaises(ValueError):self.robot.glide({'wrist_flex':2000},feedback=True)
    def test_feedback_candidate_guard_veto_precedes_write_and_commit(self):
        self.held();self.bus.reg[4][56]=2010
        def guard(p):
            if p['wrist_flex']<2000:raise RuntimeError('screened candidate veto')
        with self.assertRaisesRegex(RuntimeError,'candidate veto'):
            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,guard=guard)
        self.assertEqual(self.bus.events,[]);self.assertEqual(self.robot.bias['wrist_flex'],0)
    def test_feedback_actual_outside_bounds_precedes_write(self):
        self.held();self.bus.reg[4][56]=2025
        with self.assertRaises(ValueError):self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds)
        self.assertEqual(self.bus.events,[])
    def test_guard_latency_cannot_cross_deadline_then_write(self):
        self.held();self.bus.reg[4][56]=2010
        def guard(p):
            if p['wrist_flex']<2000:self.clock.now+=3
        with self.assertRaisesRegex(RuntimeError,'expired'):
            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,guard=guard)
        self.assertEqual(self.bus.events,[]);self.assertEqual(self.robot.bias['wrist_flex'],0)
    def test_write_failure_leaves_bias_uncommitted(self):
        self.held();self.bus.reg[4][56]=2010;self.bus.fail_goal=True
        with self.assertRaisesRegex(RuntimeError,'寫入失敗'):
            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds)
        self.assertEqual(self.robot.bias['wrist_flex'],0);self.assertEqual(self.robot.goals['wrist_flex'],2000)
    def test_glide_screens_each_sequential_candidate_not_only_final_pose(self):
        self.held();self.robot.hold=lambda:None
        def guard(p):
            if p['shoulder_pan']>2000 and p['wrist_flex']==2000:raise RuntimeError('intermediate forbidden')
        with self.assertRaisesRegex(RuntimeError,'intermediate forbidden'):
            self.robot.glide({'shoulder_pan':2004,'wrist_flex':2004},speed=35,raw_bounds=self.bounds,guard=guard)
        self.assertFalse(any(a==42 and v>2000 for _,a,v in self.bus.events))
    def test_static_servo_stops_after_four_corrections(self):
        self.held();self.bus.reg[4][56]=2010;self.bus.stuck.add(4)
        with self.assertRaisesRegex(RuntimeError,'no progress'):
            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds)
        self.assertEqual(len(self.bus.events),4);self.assertLess(self.clock.now,1)
    def test_screened_physical_lead_can_exceed_measured_bounds(self):
        self.held();self.robot.bias['wrist_flex']=-10;self.robot.goals['wrist_flex']=1990
        self.bus.reg[4][42]=1990;self.bus.reg[4][56]=2010;self.bus.offset[4]=20
        raw={n:(1970,2030) for n in JOINTS};measured={n:(1988,2012) for n in JOINTS}
        events=[]
        actual=self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=raw,measured_bounds=measured,observe=events.append)
        self.assertLessEqual(abs(actual['wrist_flex']-2000),3)
        self.assertTrue(any(a==42 and v<1988 for _,a,v in self.bus.events))
        for e in events:check_bounds(e['actual'],measured)
    def test_second_prewrite_actual_read_cannot_cross_measured_bounds(self):
        self.held();raw={n:(1970,2030) for n in JOINTS};measured={n:(1988,2012) for n in JOINTS}
        a={n:2000 for n in JOINTS};a['wrist_flex']=2010
        b={**a,'wrist_flex':2013}
        with patch.object(self.robot,'read',side_effect=[a,b]):
            with self.assertRaises(ValueError):
                self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=raw,measured_bounds=measured)
        self.assertEqual(self.bus.events,[]);self.assertEqual(self.robot.bias['wrist_flex'],0)
    def test_glide_checks_measured_bounds_before_each_serial_goal(self):
        self.held();self.robot.hold=lambda:None
        raw={n:(1970,2030) for n in JOINTS};measured={n:(1988,2012) for n in JOINTS}
        original=self.bus.write2ByteTxRx
        def drift_after_first_write(port,servo,address,value):
            result=original(port,servo,address,value)
            if servo==1 and address==42:self.bus.reg[4][56]=2013
            return result
        self.bus.write2ByteTxRx=drift_after_first_write
        with self.assertRaises(ValueError):
            self.robot.glide({'shoulder_pan':2002,'wrist_flex':2002},raw_bounds=raw,measured_bounds=measured)
        self.assertEqual(self.bus.events,[(1,42,2000)])

    def test_deferred_wrist_holds_during_trajectory_then_feedback_uses_valid_rebased_ledger(self):
        self.held();r=self.robot
        # The loaded wrist already holds with a real five-tick physical offset.
        r.goals['wrist_flex']=1995;r.bias['wrist_flex']=-5
        self.bus.reg[4][42]=1995;self.bus.offset[4]=5
        targets={'elbow_flex':1990,'wrist_flex':2010};trajectory=[];events=[];transition=[]
        original=r.finish_feedback
        def observe(event):
            trajectory.append(event)
            self.assertNotIn('wrist_flex',event['goals'])
            self.assertEqual(self.bus.reg[4][42],1995)
            self.assertFalse(any(servo==4 for servo,_,_ in self.bus.events))
        def feedback(*args,**kwargs):
            transition.append(len(self.bus.events))
            self.assertEqual(r.goals['wrist_flex'],1995)
            self.assertEqual(r.bias['wrist_flex'],-15)
            for name,target in targets.items():
                self.assertEqual(target+r.bias[name],r.goals[name])
            # This construction rejects a fabricated physical goal or stale bias.
            LocalEndpoint(targets,r.bias,r.goals,{n:self.bounds[n] for n in targets})
            self.assertEqual(len(self.bus.events),transition[-1])
            return original(*args,**kwargs)
        with patch.object(r,'finish_feedback',side_effect=feedback):
            actual=r.glide(targets,feedback=True,raw_bounds=self.bounds,allow_rearm=False,
                deferred_joints=('wrist_flex',),observe=observe,feedback_observe=events.append)
        self.assertGreater(len(trajectory),1)
        self.assertEqual(len(transition),1)
        self.assertTrue(any(servo==3 and value==1990 for servo,address,value in self.bus.events[:transition[0]]))
        self.assertTrue(any(servo==4 for servo,_,_ in self.bus.events[transition[0]:]))
        self.assertTrue(all(address==42 for _,address,_ in self.bus.events))
        for name,target in targets.items():self.assertLessEqual(abs(actual[name]-target),3)
        self.assertEqual(events[-1]['status'],'done')

    def test_deferred_feedback_corrects_worst_measured_elbow_before_wrist(self):
        self.held();self.bus.stuck.add(3);events=[]
        raw={n:(1950,2050) for n in JOINTS}
        def observe(event):
            events.append(event)
            if event['status']=='committed' and len([e for e in events if e['status']=='committed'])==1:
                self.assertEqual(event['joint'],'elbow_flex')
                self.assertEqual(self.bus.reg[4][42],2000)
                # A bounded correction releases this synthetic load. No driver
                # limit, torque state, or feedback budget changes in this fixture.
                self.bus.stuck.remove(3);self.bus.offset[3]=20
        actual=self.robot.glide({'elbow_flex':1980,'wrist_flex':2010},feedback=True,
            raw_bounds=raw,deferred_joints=('wrist_flex',),feedback_observe=observe)
        self.assertLessEqual(abs(actual['elbow_flex']-1980),3)
        self.assertLessEqual(abs(actual['wrist_flex']-2010),3)
        self.assertEqual(events[-1]['status'],'done')

    def test_invalid_deferred_joints_fail_before_begin_or_writes(self):
        invalid=[(False,('wrist_flex',)),(True,'wrist_flex'),(True,True),
                 (True,('gripper',)),(True,('unknown',)),(True,('elbow_flex',)),
                 (True,('wrist_flex',1)),(True,{'wrist_flex':True})]
        with patch.object(self.robot,'begin',side_effect=AssertionError('must not begin')):
            for feedback,deferred in invalid:
                with self.subTest(feedback=feedback,deferred=deferred):
                    with self.assertRaises(ValueError):
                        self.robot.glide({'wrist_flex':2010},feedback=feedback,
                            raw_bounds=self.bounds,deferred_joints=deferred)
        self.assertEqual(self.bus.events,[])

    def test_measured_guard_sees_actual_only_while_generic_guard_still_screens_motor_lead(self):
        self.held();r=self.robot;r.bias['wrist_flex']=-10;r.goals['wrist_flex']=1990
        self.bus.reg[4][42]=1990;self.bus.reg[4][56]=2010;self.bus.offset[4]=20
        raw={n:(1970,2030) for n in JOINTS};measured={n:(1988,2012) for n in JOINTS}
        actual_samples=[];candidate_samples=[]
        def measured_guard(pose):
            actual_samples.append(dict(pose));check_bounds(pose,measured)
            self.assertEqual(pose,{n:self.bus.reg[i+1][56] for i,n in enumerate(JOINTS)})
        actual=r.glide({'wrist_flex':2000},feedback=True,raw_bounds=raw,
            measured_guard=measured_guard,guard=lambda pose:candidate_samples.append(dict(pose)))
        self.assertLessEqual(abs(actual['wrist_flex']-2000),3)
        self.assertGreater(len(actual_samples),3)
        self.assertTrue(any(p['wrist_flex']<1988 for p in candidate_samples))
        self.assertTrue(all(p['wrist_flex']>=1988 for p in actual_samples))

    def test_measured_guard_fresh_drift_between_serial_goals_stops_and_holds(self):
        self.held();original=self.bus.write2ByteTxRx
        def drift_after_first_write(port,servo,address,value):
            result=original(port,servo,address,value)
            if servo==1 and address==42:self.bus.reg[4][56]=2013
            return result
        self.bus.write2ByteTxRx=drift_after_first_write
        def measured_guard(pose):
            if pose['wrist_flex']>2012:raise RuntimeError('actual lift corridor drift')
        with patch.object(self.robot,'hold') as hold:
            with self.assertRaisesRegex(RuntimeError,'actual lift corridor drift'):
                self.robot.glide({'shoulder_pan':2002,'wrist_flex':2002},
                    raw_bounds=self.bounds,measured_guard=measured_guard)
            hold.assert_called_once_with()
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.bus.events,[(1,42,2000)])

    def test_measured_guard_cancellation_stops_before_first_goal(self):
        self.held();r=self.robot
        def measured_guard(actual):r.stop.set()
        with patch.object(r,'hold') as hold:
            with self.assertRaises(SessionCancelled):
                r.glide({'wrist_flex':2010},feedback=True,raw_bounds=self.bounds,
                    measured_guard=measured_guard,deferred_joints=('wrist_flex',))
            hold.assert_called_once_with()
        self.assertEqual(self.bus.events,[])
        self.assertEqual(r.bias['wrist_flex'],0)

    def test_invalid_measured_guard_fails_before_any_read_or_write(self):
        with patch.object(self.robot,'read',side_effect=AssertionError('must not read')):
            with patch.object(self.robot,'begin',side_effect=AssertionError('must not begin')):
                for invalid in (True,{},1,'guard'):
                    with self.subTest(guard=invalid):
                        with self.assertRaises(ValueError):
                            self.robot.glide({'wrist_flex':2000},measured_guard=invalid)
                        with self.assertRaises(ValueError):
                            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,measured_guard=invalid)
                        with self.assertRaises(ValueError):
                            self.robot._screen_local_goal({},self.bounds,None,12,measured_guard=invalid)
        self.assertEqual(self.bus.events,[])

    def test_cartesian_accept_needs_three_fresh_samples_preserves_held_bias_and_emits_no_goals(self):
        self.held();r=self.robot
        r.bias['wrist_flex']=-10;r.goals['wrist_flex']=1990
        self.bus.reg[4][42]=1990;self.bus.reg[4][56]=2010
        events=[];sample_times=[]
        def accepted(actual):
            sample_times.append(self.clock.now);return True
        actual=r.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
            endpoint_accept=accepted,observe=events.append)
        self.assertEqual(len(sample_times),3)
        self.assertTrue(all(b-a>=.12-1e-9 for a,b in zip(sample_times,sample_times[1:])))
        self.assertEqual(self.bus.events,[])
        self.assertEqual(actual['wrist_flex'],2010)
        self.assertEqual(r.goals['wrist_flex'],1990);self.assertEqual(r.bias['wrist_flex'],-10)
        self.assertEqual([e['accepted_samples'] for e in events],[1,2,3])
        self.assertEqual(events[-1]['status'],'accepted')
        for e in events:
            self.assertEqual(e['goals']['wrist_flex'],e['logical_targets']['wrist_flex']+e['bias']['wrist_flex'])
            self.assertLessEqual(e['read_start_monotonic'],e['read_end_monotonic'])
        # Measured acceptance does not rewrite the acknowledged goal/bias ledger.
        warm=adopt_goal_anchor(actual,r.goals,previous_goals=r.goals,previous_bias=r.bias)
        self.assertEqual(warm['physical_goals']['wrist_flex'],1990)
        self.assertEqual(warm['logical_goals']['wrist_flex'],2000)
        self.assertEqual(warm['bias']['wrist_flex'],-10)

    def test_cartesian_streak_resets_when_one_sample_fails(self):
        self.held();self.bus.reg[4][56]=2010;self.bus.offset[4]=10
        sequence=iter([True,True,False,True,True,True]);calls=[];events=[]
        def accepted(actual):calls.append(dict(actual));return next(sequence)
        self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
            endpoint_accept=accepted,observe=events.append)
        self.assertEqual(len(calls),6)
        self.assertEqual([e['accepted_samples'] for e in events if e['phase']=='cartesian_endpoint'],[1,2,1,2,3])
        self.assertEqual(len(self.bus.events),1)  # only the original controller's rejected-sample correction
        self.assertEqual(events[-1]['status'],'accepted')

    def test_partial_endpoint_must_stop_moving_before_acceptance(self):
        self.held();positions=iter([2000,2002,2004,2006,2006,2006]);events=[]
        original=self.robot._screen_local_goal
        def sample(*args,**kwargs):
            self.bus.reg[4][56]=next(positions)
            return original(*args,**kwargs)
        self.robot._screen_local_goal=sample
        actual=self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
            endpoint_accept=lambda actual:True,endpoint_stability_ticks=2,observe=events.append)
        self.assertEqual(actual['wrist_flex'],2006)
        self.assertEqual([e['accepted_samples'] for e in events],[1,2,1,2,3])
        self.assertEqual(self.bus.events,[])

    def test_cartesian_progress_below_half_step_gets_one_more_bounded_correction(self):
        # Regression for a 2-mm request that had made only 0.94 mm progress
        # while all raw errors were already within the old 3-tick tolerance.
        # This deterministic local linear plant is a test fixture, not an arm
        # calibration. A one-tick correction crosses both Cartesian gates;
        # acceptance must then use three fresh samples, without chasing zero.
        self.held();self.bus.reg[4][56]=2003;self.bus.offset[4]=3
        events=[];measurements=[]
        def accepted(actual):
            progress_mm=.94+(2003-actual['wrist_flex'])*.10
            error_mm=abs(2.-progress_mm)
            measurements.append((progress_mm,error_mm))
            return error_mm<1. and progress_mm>=.5*2.
        actual=self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
            endpoint_accept=accepted,observe=events.append)
        self.assertEqual(measurements[0],(.94,1.06))
        self.assertEqual(self.bus.events,[(4,42,1999)])
        self.assertEqual(actual['wrist_flex'],2002)
        self.assertEqual(self.robot.goals['wrist_flex'],1999)
        self.assertEqual(self.robot.bias['wrist_flex'],-1)
        self.assertEqual([e['accepted_samples'] for e in events if e['phase']=='cartesian_endpoint'],[1,2,3])
        self.assertEqual(events[-1]['status'],'accepted')
        self.assertTrue(all(progress>=1. and error<1. for progress,error in measurements[1:]))
        self.assertLess(self.clock.now,3.)
        self.assertFalse(any(e['status'] in ('done','rejected') for e in events))

    def test_normal_endpoint_still_accepts_three_tick_error_without_correction(self):
        self.held();self.bus.reg[4][56]=2003;self.bus.offset[4]=3;events=[]
        actual=self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,observe=events.append)
        self.assertEqual(actual['wrist_flex'],2003)
        self.assertEqual(self.bus.events,[])
        self.assertEqual([e['status'] for e in events],['settling','settling','done'])
        self.assertEqual(self.robot.goals['wrist_flex'],2000)
        self.assertEqual(self.robot.bias['wrist_flex'],0)

    def test_cartesian_accept_cannot_bypass_raw_or_measured_bounds(self):
        for outside_measured in (False,True):
            with self.subTest(outside_measured=outside_measured):
                self.held();self.bus.events=[]
                self.bus.reg[4][56]=2013 if outside_measured else 2025
                measured={n:(1988,2012) for n in JOINTS} if outside_measured else None
                criterion_calls=[]
                with self.assertRaises(ValueError):
                    self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
                        measured_bounds=measured,endpoint_accept=lambda actual: criterion_calls.append(actual) or True)
                self.assertEqual(criterion_calls,[]);self.assertEqual(self.bus.events,[])

    def test_cartesian_callback_latency_counts_toward_original_three_second_budget(self):
        self.held();events=[]
        def observe(event):
            events.append(dict(event))
            if event['accepted_samples']==3:self.clock.now+=3
        with self.assertRaisesRegex(RuntimeError,'逾時'):
            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
                endpoint_accept=lambda actual:True,observe=observe)
        self.assertEqual(self.bus.events,[])
        self.assertEqual([e['accepted_samples'] for e in events],[1,2,3])
        self.assertFalse(any(e['status']=='done' for e in events))

    def test_cartesian_callback_cannot_reset_global_budget_between_acceptance_samples(self):
        self.held();events=[]
        def accepted(actual):self.clock.now+=1;return True
        with self.assertRaisesRegex(RuntimeError,'逾時'):
            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
                endpoint_accept=accepted,observe=events.append)
        self.assertEqual(self.bus.events,[]);self.assertEqual(len(events),2)

    def test_scalar_settling_cannot_bridge_an_intervening_cartesian_only_sample(self):
        self.held();events=[];calls=[]
        base={n:2000 for n in JOINTS}
        poses=[base,base,{**base,'wrist_flex':2006},base,base,base]
        accepted=iter([False,False,True,False,False,False])
        def criterion(actual):calls.append(dict(actual));return next(accepted)
        with patch.object(self.robot,'read',side_effect=poses):
            with self.assertRaisesRegex(RuntimeError,'末端位置'):
                self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
                    endpoint_accept=criterion,observe=events.append)
        self.assertEqual(len(calls),6);self.assertEqual(self.bus.events,[])
        self.assertEqual(events[-1]['status'],'rejected')

    def test_false_cartesian_criterion_uses_original_corrections_then_rejects_scalar_completion(self):
        self.held();self.bus.offset[4]=10;self.bus.reg[4][56]=2010;events=[]
        with self.assertRaisesRegex(RuntimeError,'末端位置'):
            self.robot.finish_feedback({'wrist_flex':2000},raw_bounds=self.bounds,
                endpoint_accept=lambda actual:False,observe=events.append)
        committed=[e for e in events if e['status']=='committed']
        self.assertGreater(len(committed),1)
        self.assertEqual(events[-1]['status'],'rejected')
        self.assertLess(self.clock.now,3)
        for e in committed:
            n=e['joint'];self.assertEqual(e['goal'],e['logical_targets'][n]+e['bias'][n])
            self.assertLessEqual(e['write_read_end_monotonic'],e['write_start_monotonic'])
        self.assertFalse(any(e['status']=='done' for e in events))

if __name__=='__main__':unittest.main(verbosity=2)
