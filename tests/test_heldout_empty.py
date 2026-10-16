import copy,dataclasses,unittest
from unittest.mock import patch
from server import heldout_empty as m
from server.gripper_context import body_context

def saved_report():
    report={'fit_allowed':True,'errors':[],'config':dataclasses.asdict(m.EXPECTED_CONFIG),
            'pwm_polarity':{'valid':True,'agreement_fraction':1.,'multiply_signed_raw_by':-1},'bins':[],
            'body_context':body_context({n:2000 for n in m.BODY})}
    for b in range(2,10):
        for direction in ['opening','closing']:
            report['bins'].append({'angle_bin':b,'position_interval':[1456+b*40,1456+(b+1)*40],
                'direction':direction,'speed_ticks_per_second':50.,'pwm_limit':100.,'cycles':3,'samples':40,
                'tracking_error_upper':14.,'pwm_effort_upper':60.,'progress_median':50.,'progress_low_limit':10.})
    return report

def fresh_cycle():
    rows=[dict(type='metadata',empty_verified=True,original_pwm_limit=500,pwm_limit=100,
               speed_ticks_per_second=50,sweep_min=1520,sweep_max=1860,camera_source='wrist',
               feedback=False,body_reference={n:2000 for n in m.BODY})]
    t=2000.
    for direction,start,sign,endpoint in [('opening',1520,1,1860),('closing',1860,-1,1520)]:
        for j in range(137):
            pos=round(start+sign*j*2.5);goal=max(1520,min(1860,pos+sign*4));load=-30*sign
            rows.append(dict(type='sample',cycle=0,direction=direction,phase='cruise',
                capture_start_monotonic=t-.002,capture_end_monotonic=t+.002,capture_monotonic=t,
                position=pos,goal=goal,load_raw=abs(load)|(1024 if load<0 else 0),load_signed_pwm=load,
                current_raw=0,status=0,model_number=777,pwm_limit=100,speed_ticks_per_second=50,
                communication_ok=True,frame_sequence=int(t*8),frame_capture_monotonic=t-.05,
                camera_source='wrist',leg_elapsed_seconds=j*.05,goal_distance_to_endpoint=abs(endpoint-goal),
                body_position={n:2000 for n in m.BODY}))
            t+=.05
        t+=.4
    rows.append(dict(type='end',complete=True,restored_pwm_limit=500,restore_verified=True,outstanding_closing_goal=False))
    return rows

class HeldoutTests(unittest.TestCase):
    def run_check(self,rows=None,report=None,**kw):
        return m.validate_heldout_empty(rows or fresh_cycle(),report or saved_report(),
                                       training_last_capture_monotonic=kw.pop('cutoff',1000.),**kw)
    def test_frozen_baseline_no_refit_and_bidirectional_coverage(self):
        baseline=saved_report();unchanged=copy.deepcopy(baseline)
        with patch.object(m.mod,'fit_empty_baseline',side_effect=AssertionError('MUST NOT FIT')):
            result=self.run_check(report=baseline)
        self.assertTrue(result['passed'],result['errors']);self.assertEqual(baseline,unchanged)
        self.assertFalse(result['refit_performed']);self.assertFalse(result['true_grasp']);self.assertIsNone(result['force_estimate'])
        self.assertGreaterEqual(min(result['evaluated_by_direction'].values()),30)
        self.assertEqual(result['counts']['contact_candidate'],0)
    def test_empty_jaw_candidate_is_an_observed_false_positive_and_fails(self):
        rows=fresh_cycle();closing=[r for r in rows if r.get('direction')=='closing']
        position=closing[45]['position']
        for row in closing[45:70]:row.update(position=position,load_raw=90,load_signed_pwm=90)
        result=self.run_check(rows)
        self.assertFalse(result['passed']);self.assertGreater(result['counts']['contact_candidate'],0)
        self.assertIn('Empty-jaw contact_candidate false positive observed',result['errors'])
    def test_reused_training_timestamps_rejected(self):
        result=self.run_check(cutoff=3000.)
        self.assertFalse(result['passed']);self.assertTrue(any('training time' in e for e in result['errors']))
    def test_wrong_polarity_is_rejected_without_relearning(self):
        rows=fresh_cycle()
        for row in rows:
            if row.get('type')=='sample':
                row['load_signed_pwm']*=-1;load=row['load_signed_pwm'];row['load_raw']=abs(load)|(1024 if load<0 else 0)
        result=self.run_check(rows)
        self.assertFalse(result['passed']);self.assertEqual(result['frozen_pwm_polarity'],-1)
        self.assertTrue(any('frozen PWM polarity' in e for e in result['errors']))

    def test_pose_context_checked_in_each_sample_without_refitting(self):
        for change in [None,{n:2000+(8 if n=='wrist_roll' else 0) for n in m.BODY}]:
            rows=fresh_cycle();rows[50]['body_position']=change
            result=self.run_check(rows)
            self.assertFalse(result['passed']);self.assertFalse(result['refit_performed'])
        report=saved_report();report.pop('body_context')
        self.assertFalse(self.run_check(report=report)['passed'])
    def test_cap_speed_fault_decode_and_snapshot_span_fail(self):
        for changed in [{'pwm_limit':500},{'speed_ticks_per_second':70},{'status':1},
                        {'load_signed_pwm':999},{'capture_end_monotonic':5000.}]:
            with self.subTest(changed=changed):
                rows=fresh_cycle();rows[50].update(changed)
                self.assertFalse(self.run_check(rows)['passed'])
    def test_restore_receipt_failure_is_not_ignored(self):
        rows=fresh_cycle();rows[-1]['restore_verified']=False
        self.assertFalse(self.run_check(rows)['passed'])
    def test_changed_config_and_insufficient_training_are_rejected(self):
        for config_change in [True,False]:
            r=saved_report()
            if config_change:r['config']['angle_bin_width']=80
            else:r['bins'][0]['cycles']=2
            self.assertFalse(self.run_check(report=r)['passed'])
    def test_insufficient_directional_coverage_is_rejected(self):
        rows=fresh_cycle()
        for row in rows:
            if row.get('direction')=='closing':row['phase']='excluded'
        result=self.run_check(rows)
        self.assertFalse(result['passed']);self.assertEqual(result['evaluated_by_direction']['closing'],0)
    def test_goal42_distance_and_direction_have_to_match_protocol(self):
        for change in [{'goal_distance_to_endpoint':999},{'goal':1800,'goal_distance_to_endpoint':60}]:
            rows=fresh_cycle();rows[50].update(change)
            self.assertFalse(self.run_check(rows)['passed'])

if __name__=='__main__':unittest.main(verbosity=2)
