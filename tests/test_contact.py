import dataclasses, unittest
from server import contact as m
S=m.Sample
BODY=dict.fromkeys(('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'),2000)

class ContactCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=m.Config(position_min=0,position_max=1000,closing_sign=1,
                         angle_bin_width=100,endpoint_margin=30)
        cls.cycles=[]
        for cycle in range(3):
            rows=[]
            for n,pos in enumerate(range(100,801,5)):
                # A normal, repeatable friction peak in bin 3 must be learned.
                pwm=80 if 300<=pos<400 else 20
                rows.append(S(cycle*100+n*.1,'closing',pos+8,pos,pwm,100,50,
                              current_raw=0,current_valid=True,body_position=BODY))
            cls.cycles.append(rows)
        cls.baseline=m.fit_empty_baseline(cls.cycles,cls.cfg)
    def setUp(self):
        self.det=m.ContactCandidateDetector(self.cfg,self.baseline);self.t=1000.
    def sample(self,pos=450,goal=None,pwm=80,**kw):
        self.t+=.1
        return S(timestamp=self.t,direction=kw.pop('direction','closing'),
                 goal=goal if goal is not None else pos+100,position=pos,
                 load_signed_pwm=pwm,pwm_limit=kw.pop('pwm_limit',100),
                 speed=kw.pop('speed',50),current_raw=0,current_valid=True,
                 body_position=kw.pop('body_position',BODY),**kw)
    def feed(self,s):return self.det.update(s,now=s.timestamp)
    def stall(self,pos=450,pwm=80,count=7,**kw):
        return [self.feed(self.sample(pos=pos,pwm=pwm,**kw)) for _ in range(count)]
    def test_sustained_closing_anomaly_is_candidate_never_grasp_or_force(self):
        result=self.stall()[-1]
        self.assertEqual(result['state'],'contact_candidate');self.assertFalse(result['true_grasp'])
        self.assertIsNone(result['force_estimate'])
        self.assertEqual(result['current_evidence'],'unavailable_or_flat_not_force_evidence')
        self.assertGreaterEqual(result['fresh_samples'],3)
    def test_stall_without_effort_increase_is_only_stalled(self):
        self.assertEqual(self.stall(pwm=20)[-1]['state'],'stalled')
    def test_matched_friction_peak_is_not_contact(self):
        self.assertEqual(self.stall(pos=350,pwm=80)[-1]['state'],'stalled')
        self.det.reset()
        results=[self.feed(self.sample(pos=320+n*5,goal=328+n*5,pwm=80)) for n in range(7)]
        self.assertEqual(results[-1]['state'],'free')
        self.assertNotIn('contact_candidate',[r['state'] for r in results])
    def test_endpoint_exclusion(self):
        for pos,goal in [(975,1000),(25,125)]:
            result=self.stall(pos=pos,goal=goal)[-1]
            self.assertEqual(result['state'],'unknown');self.assertEqual(result['reason'],'endpoint exclusion')
    def test_reversal_clears_existing_candidate_and_requires_new_guard(self):
        self.assertEqual(self.stall()[-1]['state'],'contact_candidate')
        opening=self.feed(self.sample(direction='opening',goal=350,pwm=-80))
        self.assertEqual(opening['state'],'unknown')
        closing=[self.feed(self.sample()) for _ in range(3)]
        self.assertNotIn('contact_candidate',[r['state'] for r in closing])
    def test_acceleration_phase_never_counts(self):
        self.assertEqual(self.stall(phase='accelerating')[-1]['state'],'unknown')
    def test_invalid_or_fault_clears_candidate(self):
        for kw in [{'valid':False},{'servo_status':1}]:
            self.assertEqual(self.stall()[-1]['state'],'contact_candidate')
            self.assertEqual(self.feed(self.sample(**kw))['state'],'unknown')
            self.assertEqual(self.feed(self.sample())['state'],'unknown')
    def test_unmatched_pwm_cap_or_speed_remains_unknown(self):
        for kw in [{'pwm_limit':80},{'speed':40}]:
            self.det.reset();result=self.stall(**kw)[-1]
            self.assertEqual(result['state'],'unknown')
            self.assertEqual(result['reason'],'no matched independent empty-jaw baseline')
    def test_cancel_resets_candidate(self):
        self.assertEqual(self.stall()[-1]['state'],'contact_candidate')
        s=self.sample();result=self.det.update(s,now=s.timestamp,cancelled=True)
        self.assertEqual(result['state'],'unknown');self.assertEqual(result['reason'],'cancelled')
        self.assertEqual(self.feed(self.sample())['state'],'unknown')

    def test_missing_or_changed_body_pose_clears_candidate(self):
        for body in [None,dict(BODY,wrist_roll=2005),dict(BODY,elbow_flex=True)]:
            self.assertEqual(self.stall()[-1]['state'],'contact_candidate')
            self.assertEqual(self.feed(self.sample(body_position=body))['state'],'unknown')
            self.assertEqual(self.feed(self.sample())['state'],'unknown')

    def test_baseline_cannot_pool_other_arm_orientations(self):
        cycles=list(self.cycles)
        cycles[2]=[dataclasses.replace(s,body_position=dict(BODY,wrist_roll=2010)) for s in cycles[2]]
        with self.assertRaisesRegex(ValueError,'Body pose'):m.fit_empty_baseline(cycles,self.cfg)
    def test_duplicate_stale_and_gap_samples_never_bridge_evidence(self):
        s=self.sample();self.feed(s)
        self.assertEqual(self.feed(s)['reason'],'duplicate or non-increasing capture timestamp')
        s=self.sample();self.assertEqual(self.det.update(s,now=s.timestamp+1)['state'],'unknown')
        self.stall();self.t+=1
        self.assertEqual(self.feed(self.sample())['state'],'unknown')
    def test_opposing_pwm_and_single_effort_spike_do_not_claim_contact(self):
        self.assertEqual(self.stall(pwm=-80)[-1]['state'],'stalled')
        self.det.reset();self.stall(pwm=20)
        self.assertEqual(self.feed(self.sample(pwm=80))['state'],'stalled')
    def test_effort_without_tracking_stall_is_not_contact(self):
        result=None
        for n in range(7):result=self.feed(self.sample(pos=420+n*5,goal=428+n*5,pwm=80))
        self.assertEqual(result['state'],'unknown')
    def test_three_cycles_required_and_duplicate_times_rejected(self):
        with self.assertRaises(ValueError):m.fit_empty_baseline(self.cycles[:2],self.cfg)
        with self.assertRaises(ValueError):m.fit_empty_baseline([self.cycles[0]]*3,self.cfg)
    def test_rare_recorded_friction_peak_cannot_hide_below_quantile(self):
        cycles=[]
        for cycle in self.cycles:
            rows=[dataclasses.replace(row,load_signed_pwm=90) if row.position==450 else row for row in cycle]
            cycles.append(rows)
        baseline=m.fit_empty_baseline(cycles,self.cfg)
        self.det=m.ContactCandidateDetector(self.cfg,baseline)
        self.assertEqual(self.stall(pos=450,pwm=90)[-1]['state'],'stalled')

    def test_minimum_samples_per_cycle_and_config_provenance(self):
        bad=m.fit_empty_baseline([[row for row in cycle if 300<=row.position<320] for cycle in self.cycles],self.cfg)
        self.assertEqual(len(bad.bins),0)
        other=dataclasses.replace(self.cfg,angle_bin_width=50)
        with self.assertRaises(ValueError):m.ContactCandidateDetector(other,self.baseline)

if __name__=='__main__':unittest.main(verbosity=2)
