"""Exercise the actual arc dispatch with the existing in-memory motor plant."""
import unittest

import numpy as np

import test_wave as helpers


class ArcMoveTests(unittest.TestCase):
    model=helpers.default_model()
    setUp=helpers.WaveTests.setUp
    capture=helpers.WaveTests.capture

    def move_arc(self,cap=5):
        before=self.capture()
        proposal,envelope=self.grasp._wave_lift_proposal(cap,before['joints'],before['joints'])
        return self.grasp.move('up',float(np.linalg.norm(proposal['requested_delta_m'])*1000),
            before,self.reports.append,free_space_lift=True,lift_origin=before['joints'],cancel=self.cancel,
            assessed_lift={'proposal':proposal,'envelope':envelope,
                           'origin':before['joints'],'hold_goals':dict(self.robot.goals)})

    def test_arc_writes_elbow_only_and_reports_measured_raise_without_wave_success(self):
        before=self.robot.read();goals=dict(self.robot.goals)
        result=self.move_arc()
        self.assertEqual(set(n for n,_ in self.robot.writes),{'elbow_flex'})
        for n in before:
            if n!='elbow_flex':
                self.assertEqual(self.robot.actual[n],before[n])
                self.assertEqual(self.robot.goals[n],goals[n])
        self.assertGreater(result['actual_delta_model_mm'][2],.5)
        self.assertEqual(result['motion_kind'],'elbow_raise')
        self.assertNotIn('wave_completed',result)
        self.assertNotIn('grasp_confirmed',result)
        options=self.robot.calls[0]['kwargs']
        self.assertEqual(options['deferred_joints'],())
        self.assertEqual(options['endpoint_stability_ticks'],2)
        self.assertTrue(options['stall_response_wait'])
        self.assertTrue(options['endpoint_accept'](self.robot.read()))
        drift={**self.robot.read(),'wrist_flex':before['wrist_flex']+5}
        with self.assertRaises(RuntimeError):options['measured_guard'](drift)

    def test_larger_assessed_arc_executes_its_exact_path_with_existing_live_corridor(self):
        before=self.robot.read()
        result=self.move_arc(8)
        self.assertGreater(result['requested_mm'],5)
        self.assertLessEqual(result['requested_mm'],8)
        self.assertEqual(set(n for n,_ in self.robot.writes),{'elbow_flex'})
        options=self.robot.calls[0]['kwargs']
        options['measured_guard'](self.robot.read())
        self.assertTrue(options['endpoint_accept'](self.robot.read()))
        with self.assertRaises(RuntimeError):
            options['measured_guard']({**before,'elbow_flex':before['elbow_flex']-40})
        self.assertNotIn('wave_completed',result)

    def test_larger_step_requires_the_approved_elbow_path_and_matching_distance(self):
        before=self.capture()
        proposal,envelope=self.grasp._wave_lift_proposal(8,before['joints'],before['joints'])
        distance=float(np.linalg.norm(proposal['requested_delta_m'])*1000)
        assessed={'proposal':proposal,'envelope':envelope,
                  'origin':before['joints'],'hold_goals':dict(self.robot.goals)}
        for kwargs in ({},{'free_space_lift':True}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                self.grasp.move('up',distance,before,self.reports.append,**kwargs)
        for wrong in (distance-.1,8.01,float('nan'),True,0):
            with self.subTest(distance=wrong),self.assertRaises(ValueError):
                self.grasp.move('up',wrong,before,self.reports.append,free_space_lift=True,
                                assessed_lift=assessed,lift_origin=before['joints'])
        self.assertEqual(self.robot.writes,[])


if __name__=='__main__':unittest.main()
