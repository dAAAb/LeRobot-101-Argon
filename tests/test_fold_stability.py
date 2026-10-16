import itertools, threading, unittest
import numpy as np
from server.selfcal import inv, intrinsics, reconstruct_pair, sampson_errors
from server.fold_stability import target_fold_stability as validate

class MetricFoldStabilityTests(unittest.TestCase):
    def setUp(self):
        self.G=[]
        for x in [-.06,-.02,.02,.06]:
            T=np.eye(4);T[0,3]=x;self.G.append(T)
        self.X=np.eye(4);self.K=intrinsics(800.,(960,540))
        rng=np.random.default_rng(37)
        self.points=np.c_[rng.uniform(-.04,.04,24),rng.uniform(-.025,.025,24),rng.uniform(.45,.6,24)]
        pixels=[]
        for g in self.G:
            cam=(inv(g@self.X)@np.c_[self.points,np.ones(len(self.points))].T).T[:,:3]
            uv=(self.K@cam.T).T;pixels.append(uv[:,:2]/uv[:,2,None])
        self.matches=[(i,j,pixels[i],pixels[j]) for i,j in itertools.combinations(range(4),2)]
    def folds(self,X=None,K=None):
        return [{'pose':p,'passed':True,'X':self.X if X is None else X,
                 'K':self.K if K is None else K} for p in range(3)]
    def run_check(self,folds,**kwargs):
        return validate(self.G,self.X,self.K,self.matches,folds,
                        reconstruct_pair=reconstruct_pair,**kwargs)
    def test_identical_complete_calibrations_are_repeatable(self):
        result=self.run_check(self.folds())
        self.assertEqual(result['status'],'repeatable');self.assertFalse(result['grasp_ready'])
        self.assertEqual(result['validated_folds'],3)
        self.assertLess(result['worst_fold_p95_difference_m'],1e-10)
        self.assertEqual(result['folds'][0]['shared_valid_rows'],144)
    def test_low_epipolar_error_same_translation_but_wrong_focal_is_uncertain(self):
        kf=intrinsics(940.,(960,540))
        self.assertLess(np.max(np.abs(sampson_errors(self.G,self.X,kf,self.matches))),1e-9)
        result=self.run_check(self.folds(K=kf))
        self.assertEqual(result['status'],'uncertain');self.assertEqual(result['validated_folds'],3)
        self.assertGreater(result['worst_fold_p95_difference_m'],.07)
        self.assertEqual(result['translation_change_mm']['max'],0.)
        self.assertEqual(result['focal_x_px']['median'],940.)
        self.assertGreater(result['folds'][0]['focal_change_percent'][0],17.)
    def test_low_epipolar_error_same_translation_but_wrong_rotation_is_uncertain(self):
        theta=np.deg2rad(4.)
        xf=np.eye(4);xf[1:3,1:3]=[[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]
        self.assertLess(np.max(np.abs(sampson_errors(self.G,xf,self.K,self.matches))),1e-9)
        result=self.run_check(self.folds(X=xf))
        self.assertEqual(result['status'],'uncertain');self.assertEqual(result['validated_folds'],3)
        self.assertGreater(result['worst_fold_p95_difference_m'],.03)
        self.assertEqual(result['translation_change_mm']['max'],0.)
        self.assertAlmostEqual(result['rotation_change_deg']['max'],4.,places=8)
    def test_fewer_than_three_distinct_folds_is_uncertain(self):
        self.assertEqual(self.run_check(self.folds()[:2])['status'],'uncertain')
        duplicate=self.folds();duplicate[2]['pose']=1
        result=self.run_check(duplicate)
        self.assertEqual(result['status'],'uncertain');self.assertEqual(result['validated_folds'],2)
    def test_failed_fold_cannot_be_hidden_by_three_other_folds(self):
        result=self.run_check(self.folds()+[{'pose':3,'passed':False}])
        self.assertEqual(result['status'],'uncertain')
    def test_reconstruct_pair_existing_positive_depth_mask_is_preserved(self):
        # Appended negative-depth world points still reproject exactly, but cannot
        # inflate the shared positive-depth evidence count.
        points=np.r_[self.points,[[.01,.02,-.5],[.02,.01,-.6]]]
        uv=[]
        for g in self.G:
            c=(inv(g)@np.c_[points,np.ones(len(points))].T).T[:,:3]
            h=(self.K@c.T).T;uv.append(h[:,:2]/h[:,2,None])
        self.matches=[(i,j,uv[i],uv[j]) for i,j in itertools.combinations(range(4),2)]
        result=self.run_check(self.folds())
        self.assertEqual(result['status'],'repeatable')
        for fold in result['folds']:
            for pair in fold['pairs']:
                self.assertEqual(pair['input_rows'],26)
                self.assertEqual(pair['shared_indices'],list(range(24)))
    def test_cancellation_never_reports_repeatable(self):
        event=threading.Event();event.set()
        result=self.run_check(self.folds(),cancel=event)
        self.assertEqual(result['status'],'uncertain');self.assertTrue(result['cancelled'])
    def test_unsafe_evidence_threshold_relaxation_is_rejected(self):
        result=self.run_check(self.folds(),max_p95_difference_m=.009)
        self.assertEqual(result['status'],'uncertain')

