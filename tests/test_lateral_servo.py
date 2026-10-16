"""Pure state tests using explicit synthetic correspondence, no hardware."""
import importlib.util,unittest,copy
from server import lateral_servo as m
def frame(seq=1,t=100):return {'source':'wrist','camera_generation':'synthetic','sequence':seq,'capture_monotonic':t,'image_size':[640,480]}
def aperture(x=300):
    edge={'status':'ok','inlier_fraction':.9,'residual_p95_px':.5,'alternating_rows_fit_disagreement_p95_px':.2}
    return {'status':'ok','image_size':[640,480],'aperture_center_px':[x,350],
        'visible_gap_px':80,'left':dict(edge),'right':dict(edge)}
def evidence(before,after,dx,*,retained=.95):
    return {'source':'wrist-only','camera_generation':'synthetic','image_size':[640,480],
        'frame_sequences':[before['sequence'],after['sequence']],
        'frame_capture_monotonic':[before['capture_monotonic'],after['capture_monotonic']],
        'target_bbox_input_px':[0,0,640,480],
        'target':{'status':'ok','track_count':30,'retained_fraction':retained,
            'fb_p95_px':.1,'ncc_median':.95,'similarity_residual_p95_px':.5,'median_scale':1.,
            'similarity_matrix':[[1.,0.,dx],[0.,1.,0.]]}}
def state(x=500,budget=40):return m.LateralServo(frame(),[x,200],aperture(),target_token='selected-one',now_monotonic=100.1,total_travel_mm=budget)
def observe(c,dx=-30,actual=3,**kwargs):
    b=c.frame;a=frame(b['sequence']+1,b['capture_monotonic']+1)
    ev=evidence(b,a,dx)
    return c.observe_step(a,ev,aperture(),target_token='selected-one',actual_camera_right_mm=actual,now_monotonic=a['capture_monotonic']+.1,**kwargs)
class Tests(unittest.TestCase):
    def test_requires_positive_measured_pilot_before_proposals(self):
        c=state();self.assertIsNone(c.propose(now_monotonic=100.1)['proposal'])
        r=observe(c);self.assertEqual(r['state'],'ready');self.assertAlmostEqual(r['observed_response_px_per_mm'],10)
        p=c.propose(now_monotonic=101.2)['proposal'];self.assertEqual(p['direction'],'right');self.assertLessEqual(p['distance_mm'],3)
        self.assertIsNone(c.propose(now_monotonic=101.3)['proposal'])
    def test_repeated_steps_use_actual_displacement_and_converge(self):
        c=state();observe(c)
        for _ in range(20):
            p=c.propose(now_monotonic=c.frame['capture_monotonic']+.2)['proposal']
            if p is None:break
            actual=p['distance_mm']*.9
            r=observe(c,dx=-10*actual,actual=actual,proposal_id=p['proposal_id'])
            self.assertFalse(r['confirmed_grasp'])
        self.assertEqual(c.state,'aligned');self.assertLess(abs(c.error),4)
        self.assertLessEqual(c.measured_travel,40);self.assertLessEqual(c.commanded_travel,40)
    def test_left_error_yields_left_proposal(self):
        c=state(100);observe(c,dx=30,actual=-3)
        self.assertEqual(c.propose(now_monotonic=101.2)['proposal']['direction'],'left')
    def test_lost_identity_stops_without_zero_motion_fallback(self):
        c=state();b=c.frame;a=frame(2,101);ev=evidence(b,a,-30,retained=.1)
        r=c.observe_step(a,ev,aperture(),target_token='selected-one',actual_camera_right_mm=3,now_monotonic=101.1)
        self.assertEqual(r['state'],'stopped');self.assertIsNone(c.propose(now_monotonic=101.2)['proposal'])
    def test_wrong_direction_no_progress_and_zero_actual_stop(self):
        for dx,actual in [(10,3),(0,3),(-.1,3),(-30,0),(-30,-3)]:
            with self.subTest(dx=dx,actual=actual):self.assertEqual(observe(state(),dx=dx,actual=actual)['state'],'stopped')
    def test_stale_frame_and_generation_change_stop(self):
        for change in ('stale','generation'):
            c=state();b=c.frame;a=frame(2,101);ev=evidence(b,a,-30)
            if change=='generation':a['camera_generation']='reconnected'
            r=c.observe_step(a,ev,aperture(),target_token='selected-one',actual_camera_right_mm=3,now_monotonic=103 if change=='stale' else 101.1)
            self.assertEqual(r['state'],'stopped')
    def test_mismatched_evidence_endpoint_is_rejected(self):
        c=state();a=frame(2,101);ev=evidence(c.frame,a,-30);ev['frame_sequences'][0]=0
        self.assertEqual(c.observe_step(a,ev,aperture(),target_token='selected-one',actual_camera_right_mm=3,now_monotonic=101.1)['state'],'stopped')
    def test_changed_jaw_or_target_identity_requires_reacquisition(self):
        for mode in ('jaw','token'):
            c=state();a=frame(2,101);ev=evidence(c.frame,a,-30)
            r=c.observe_step(a,ev,aperture(310 if mode=='jaw' else 300),target_token='other' if mode=='token' else 'selected-one',actual_camera_right_mm=3,now_monotonic=101.1)
            self.assertEqual(r['state'],'stopped')
    def test_low_ncc_or_bad_fb_stops(self):
        for key,value in [('ncc_median',.7),('fb_p95_px',.8),('similarity_residual_p95_px',3)]:
            c=state();a=frame(2,101);ev=evidence(c.frame,a,-30);ev['target'][key]=value
            self.assertEqual(c.observe_step(a,ev,aperture(),target_token='selected-one',actual_camera_right_mm=3,now_monotonic=101.1)['state'],'stopped')
    def test_budget_reserves_actual_endpoint_slack(self):
        c=state(budget=9);observe(c,dx=-3)
        p=c.propose(now_monotonic=101.2)['proposal'];self.assertEqual(p['distance_mm'],3)
        observe(c,dx=-3,proposal_id=p['proposal_id'])
        p=c.propose(now_monotonic=102.2)['proposal'];self.assertEqual(p['distance_mm'],2)
        observe(c,dx=-3,actual=3,proposal_id=p['proposal_id'])
        self.assertIsNone(c.propose(now_monotonic=103.2)['proposal']);self.assertEqual(c.measured_travel,9)
    def test_overshoot_beyond_image_tolerance_does_not_reverse_automatically(self):
        c=state(310);self.assertEqual(observe(c,dx=-30)['state'],'stopped')
    def test_wrong_proposal_id_stops(self):
        c=state();observe(c);c.propose(now_monotonic=101.2)
        self.assertEqual(observe(c,proposal_id=999)['state'],'stopped')
    def test_stale_before_proposal_stops(self):
        c=state();observe(c);self.assertIsNone(c.propose(now_monotonic=103)['proposal']);self.assertEqual(c.state,'stopped')
if __name__=='__main__':unittest.main()
