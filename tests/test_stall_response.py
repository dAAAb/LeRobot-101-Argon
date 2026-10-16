"""Delayed empty-hand elbow response, with real endpoint control and fake SDK."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from server.feedback import EndpointFeedback
from server.local_feedback import LocalEndpoint
from server.robot import JOINTS, Robot
from server.gripper_power import SessionCancelled
import test_local_feedback as helpers


class StalledResponseControlTests(unittest.TestCase):
    def controller(self,**kwargs):
        return EndpointFeedback('elbow_flex',2000,1900,2100,stall_response_wait=True,**kwargs)

    def four(self,c,actual=2010):
        for i in range(4):
            d=c.observe(actual,i*.12);self.assertEqual(d.status,'propose');c.commit(d,i*.12)

    def test_delayed_three_tick_response_allows_existing_correction_flow(self):
        c=self.controller();self.four(c)
        for now in (.48,.60):
            d=c.observe(2009,now);self.assertEqual(d.status,'wait');self.assertIsNone(d.goal)
        self.assertEqual(c.corrections,4)
        d=c.observe(2007,.72)
        self.assertEqual(d.status,'propose')
        self.assertEqual(c.corrections,4)  # still uncommitted, no implied fifth write

    def test_one_tick_response_expires_half_second_after_fourth_commit(self):
        c=self.controller();self.four(c)
        self.assertAlmostEqual(c.stall_response_deadline,.86)
        self.assertEqual(c.observe(2009,.85).status,'wait')
        d=c.observe(2009,.86)
        self.assertEqual(d.status,'abort');self.assertIn('response observation expired',d.reason)
        self.assertEqual(c.corrections,4)

    def test_wait_never_extends_three_second_feedback_deadline(self):
        c=self.controller()
        for now in (0.,.12,.24,2.7):
            d=c.observe(2010,now);c.commit(d,now)
        self.assertEqual(c.observe(2010,2.9).status,'wait')
        d=c.observe(2010,3.)
        self.assertEqual(d.status,'abort');self.assertIn('3 seconds',d.reason)
        self.assertEqual(c.corrections,4)

    def test_reversal_and_position_bounds_still_abort_during_wait(self):
        for actual,reason in ((1989,'reversed'),(2101,'outside')):
            c=self.controller();self.four(c)
            self.assertEqual(c.observe(2010,.48).status,'wait')
            d=c.observe(actual,.49)
            self.assertEqual(d.status,'abort');self.assertIn(reason,d.reason)
            self.assertEqual(c.corrections,4)

    def test_default_stall_behavior_and_elbow_only_opt_in(self):
        c=EndpointFeedback('elbow_flex',2000,1900,2100);self.four(c)
        self.assertEqual(c.observe(2010,.48).status,'abort')
        for flag in (1,None,'true'):
            with self.assertRaises(ValueError):
                EndpointFeedback('elbow_flex',2000,1900,2100,stall_response_wait=flag)
        with self.assertRaises(ValueError):
            EndpointFeedback('wrist_flex',2000,1900,2100,stall_response_wait=True)
        with self.assertRaises(ValueError):
            LocalEndpoint({'wrist_flex':2000},{'wrist_flex':0},{'wrist_flex':2000},
                          {'wrist_flex':(1900,2100)},stall_response_wait=True)

    def test_local_stable_scalar_cannot_hide_insufficient_three_tick_progress(self):
        c=LocalEndpoint({'elbow_flex':2000},{'elbow_flex':0},{'elbow_flex':2000},
                        {'elbow_flex':(1900,2100)},stall_response_wait=True)
        for i in range(4):
            d=c.observe({'elbow_flex':2004},i*.12);c.commit(d,i*.12)
        for now in (.48,.60,.72):
            self.assertEqual(c.observe({'elbow_flex':2003},now).status,'wait')
        self.assertAlmostEqual(c.poll_delay(.84),.02)
        self.assertEqual(c.observe({'elbow_flex':2003},.86).status,'abort')


class DelayedPlant(helpers.FakeProtocol):
    def __init__(self,cal,clock,*,delay=.30):
        super().__init__(cal);self.clock=clock;self.delay=delay;self.writes=[];self.stuck.add(3)
        self.reg[3][56]=2010;self.fourth_at=None;self.read_hook=None
    def read(self,port,servo,address):
        if self.fourth_at is not None and self.delay is not None and self.clock.now-self.fourth_at>=self.delay:
            self.reg[3][56]=2002
        if self.read_hook:self.read_hook(servo,address)
        return super().read(port,servo,address)
    read1ByteTxRx=read
    read2ByteTxRx=read
    def write(self,port,servo,address,value):
        self.writes.append((self.clock.now,servo,address,value))
        result=super().write(port,servo,address,value)
        if servo==3 and address==42 and len(self.writes)==4:self.fourth_at=self.clock.now
        return result
    write1ByteTxRx=write
    write2ByteTxRx=write


class StalledResponseRobotTests(unittest.TestCase):
    def rig(self,delay=.30):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        cal={n:{'id':i+1,'range_min':0,'range_max':4095} for i,n in enumerate(JOINTS)}
        r=Robot({'follower_port':'fake://no-hardware','max_ticks_per_second':90,
                 'max_tracking_error_ticks':90},cal,tmp.name,helpers.FakeCamera())
        clock=helpers.Clock();bus=DelayedPlant(cal,clock,delay=delay)
        r.ph=bus;r.port=SimpleNamespace(is_using=False);r.connected=True;r.armed=True;r.goals=dict.fromkeys(JOINTS,2000)
        r.stop=helpers.FakeStop(clock);r.gripper_power.stop_latch=r.stop.set
        return r,bus,clock

    def finish(self,r,**options):
        return r.finish_feedback({'elbow_flex':2000},raw_bounds={n:(1970,2030) for n in JOINTS},
                                 stall_response_wait=True,**options)

    def test_delayed_plant_reaches_endpoint_without_a_fifth_write(self):
        r,bus,clock=self.rig();events=[]
        with patch('server.robot.time.monotonic',clock.time):
            actual=self.finish(r,observe=events.append)
        self.assertEqual(actual['elbow_flex'],2002)
        self.assertEqual(len(bus.writes),4)
        self.assertTrue(any('stalled motor response' in e.get('reason','') for e in events))
        self.assertEqual(events[-1]['status'],'done')
        self.assertLess(clock.now,3)

    def test_same_delayed_plant_with_default_precision_path_still_stops_at_four(self):
        r,bus,clock=self.rig()
        with patch('server.robot.time.monotonic',clock.time):
            with self.assertRaisesRegex(RuntimeError,'no progress'):
                r.finish_feedback({'elbow_flex':2000},raw_bounds={n:(1970,2030) for n in JOINTS})
        self.assertEqual(len(bus.writes),4);self.assertLess(clock.now,.7)

    def test_truly_stuck_waits_at_most_half_second_without_extra_write(self):
        r,bus,clock=self.rig(delay=None)
        with patch('server.robot.time.monotonic',clock.time):
            with self.assertRaisesRegex(RuntimeError,'response observation expired'):self.finish(r)
        self.assertEqual(len(bus.writes),4)
        self.assertAlmostEqual(clock.now-bus.fourth_at,.5)

    def test_three_second_global_deadline_preempts_remaining_response_window(self):
        r,bus,clock=self.rig(delay=None)
        def consume_time(event):
            if event['status']=='committed' and len(bus.writes)==1:clock.now+=2.2
        with patch('server.robot.time.monotonic',clock.time):
            with self.assertRaisesRegex(RuntimeError,'3 seconds'):self.finish(r,observe=consume_time)
        self.assertEqual(len(bus.writes),4);self.assertAlmostEqual(clock.now,3.)
        self.assertLess(clock.now-bus.fourth_at,.5)

    def test_custom_endpoint_acceptance_keeps_three_spaced_samples_after_wait(self):
        r,bus,clock=self.rig(delay=.45);events=[]
        with patch('server.robot.time.monotonic',clock.time):
            self.finish(r,endpoint_accept=lambda p:p['elbow_flex']<=2002,
                        endpoint_stability_ticks=2,observe=lambda e:events.append((clock.now,e)))
        accepted=[t for t,e in events if e['phase']=='cartesian_endpoint']
        self.assertEqual(len(accepted),3)
        self.assertTrue(all(b-a>=.12-1e-9 for a,b in zip(accepted,accepted[1:])))
        self.assertEqual(len(bus.writes),4)

    def test_camera_temperature_bounds_reversal_and_cancel_remain_live_while_waiting(self):
        for kind in ('camera','temperature','bounds','reversal','cancel','stop'):
            with self.subTest(kind=kind):
                r,bus,clock=self.rig(delay=None)
                def after_wait():return bus.fourth_at is not None and clock.now-bus.fourth_at>=.24-1e-9
                def guard(p):
                    if kind=='cancel' and after_wait():raise SessionCancelled('cancelled in guard')
                def read_hook(servo,address):
                    if not after_wait():return
                    if kind=='temperature':bus.reg[3][63]=61
                    if kind=='bounds':bus.reg[3][56]=2031
                    if kind=='reversal':bus.reg[3][56]=1989
                bus.read_hook=read_hook
                r.camera.fresh=lambda:not (kind=='camera' and after_wait())
                original_wait=r.stop.wait
                def wait(seconds):
                    result=original_wait(seconds)
                    if kind=='stop' and after_wait():r.stop.set();return True
                    return result
                r.stop.wait=wait
                with patch('server.robot.time.monotonic',clock.time):
                    with self.assertRaises((RuntimeError,ValueError,SessionCancelled)):
                        self.finish(r,guard=guard)
                self.assertEqual(len(bus.writes),4)
                self.assertLess(clock.now-bus.fourth_at,.5)

    def test_other_joint_or_held_jaw_cannot_enable_robot_opt_in(self):
        r,bus,clock=self.rig()
        bounds={n:(1970,2030) for n in JOINTS}
        with self.assertRaises(ValueError):
            r.glide({'wrist_flex':2000},feedback=True,allow_rearm=False,
                    raw_bounds=bounds,stall_response_wait=True)
        r.pwm_owner=object()
        with self.assertRaises(ValueError):self.finish(r)
        self.assertEqual(bus.writes,[])


if __name__=='__main__':unittest.main()
