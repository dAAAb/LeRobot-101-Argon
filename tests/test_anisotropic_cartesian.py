"""Pure geometry regressions; no hardware, cameras or known object dimensions."""
import itertools, json, math, sys, unittest
from pathlib import Path
import numpy as np
from server.local_cartesian import ALL,NAMES,RAD_PER_TICK,default_model,tracking_envelope

class AnisotropicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=default_model()
        cls.raw=dict(zip(ALL,[2161,1123,2945,2330,985,1525]))
        cls.envelope={n:12 for n in NAMES};cls.envelope['elbow_flex']=39
        cls.held={**cls.raw,'elbow_flex':2912}
    def test_measured_elbow_load_offset_passes_without_inflating_other_axes(self):
        p=self.model.propose(self.raw,[0,0,.003],held_goals=self.held,uncertainty_ticks=self.envelope)
        self.assertTrue(p['clearance']['model_table_pass'])
        self.assertEqual(p['physical_goal_targets']['elbow_flex'],2912+p['delta_ticks']['elbow_flex'])
        self.assertEqual(p['clearance']['actual_tracking_envelope_ticks'],{**self.envelope,'gripper':0})
        isotropic=self.model.screen_segment(self.raw,p['joint_targets'],uncertainty_ticks=39)
        self.assertLess(p['clearance']['continuous_residual_bound_mm'][self.model.jaw_label],isotropic['continuous_residual_bound_mm'][self.model.jaw_label])
    def test_changed_hold_outside_its_axis_fails(self):
        for n in NAMES:
            bad={**self.held,n:self.raw[n]+self.envelope[n]+1}
            with self.subTest(joint=n),self.assertRaisesRegex(ValueError,'hold offset'):
                self.model.propose(self.raw,[0,0,.003],held_goals=bad,uncertainty_ticks=self.envelope)
    def test_small_scalar_rejects_large_elbow_hold(self):
        with self.assertRaisesRegex(ValueError,'hold offset'):
            self.model.propose(self.raw,[0,0,.003],held_goals=self.held,uncertainty_ticks=30)
    def test_each_axis_onboard_envelope_is_checked(self):
        for n in NAMES:
            raw={**self.raw,n:self.model.calibration[n]['range_max']-12-self.envelope[n]+1}
            with self.subTest(joint=n),self.assertRaisesRegex(ValueError,'Onboard'):
                self.model.screen_segment(raw,raw,uncertainty_ticks=self.envelope)
    def test_full_jaw_envelope_does_not_consume_motor_range(self):
        raw={**self.raw,'gripper':self.model.calibration['gripper']['range_min']+12}
        r=self.model.screen_segment(raw,raw,uncertainty_ticks=60)
        self.assertEqual(r['onboard_margin_including_envelope_ticks']['gripper'],12)
        self.assertEqual(r['actual_tracking_envelope_ticks']['gripper'],0)
    def test_invalid_envelopes_are_rejected(self):
        invalid=[True,1.0,-1,61,{'elbow_flex':39},{**self.envelope,'elbow_flex':float('nan')},
                 {**self.envelope,'wrist_roll':True},{**self.envelope,'gripper':1},
                 {**self.envelope,'unknown':0}]
        for value in invalid:
            with self.subTest(value=value),self.assertRaises(ValueError):tracking_envelope(value)
    def test_scalar_and_equal_mapping_match(self):
        a=self.model.screen_segment(self.raw,self.raw,uncertainty_ticks=12)
        b=self.model.screen_segment(self.raw,self.raw,uncertainty_ticks={n:12 for n in NAMES})
        self.assertEqual(a,b)
    def test_nan_hold_cannot_bypass_offset_check(self):
        with self.assertRaisesRegex(ValueError,'Bad joint'):
            self.model.propose(self.raw,[0,0,.003],held_goals={**self.held,'elbow_flex':float('nan')},uncertainty_ticks=self.envelope)
    def test_analytic_lower_bound_covers_corners_and_intermediate_path(self):
        end=self.model.local_ik(self.raw,[0,0,.003])['joint_targets']
        result=self.model.screen_segment(self.raw,end,uncertainty_ticks=self.envelope,path_step_ticks=2)
        for t in [.031,.273,.519,.887]:
            center={n:self.raw[n]+t*(end[n]-self.raw[n]) for n in ALL}
            for signs in itertools.product([-1,1],repeat=5):
                actual={**center,**{n:center[n]+s*self.envelope[n] for n,s in zip(NAMES,signs)}}
                for link,z in self.model.floor_minimum(actual).items():
                    self.assertLessEqual(result['continuous_model_lower_bound_mm'][link],z*1000+1e-8)
        radii=self.model.radii[self.model.jaw_label]
        expected=sum(2*r*math.sin((self.envelope[n]+1)*RAD_PER_TICK/2) for n,r in radii.items())*1000
        self.assertAlmostEqual(result['continuous_residual_bound_mm'][self.model.jaw_label],expected,places=10)

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AnisotropicTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    p=AnisotropicTests.model.propose(AnisotropicTests.raw,[0,0,.003],held_goals=AnisotropicTests.held,uncertainty_ticks=AnisotropicTests.envelope)
    Path('/private/tmp/lerobot-anisotropic-cartesian-report.json').write_text(json.dumps(p,indent=2))
