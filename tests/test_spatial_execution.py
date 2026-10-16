"""Keep assessed paths exact; post-contact plans use the new measured jaw."""
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from server import motion_program as mp, object_program as op
from test_object_program import Fixture, program


def attach_proposal(g, plan, frame):
    g.spatial_attempt = {}
    prepared = op.prepare_object_program(g, plan, frame)
    proposal = {'binding': {'evidence': {'calibration_fingerprint': 'camera-v1', 'geometry_revision': 'model-v1'},
                            'tcp': {'fingerprint': 'jaw-v1'}},
                'segments': [{'kind': step.kind, **({'contract': prepared[i]} if i in prepared else {})}
                            for i, step in enumerate(plan.steps)]}
    g.spatial_attempt = {'proposal': proposal}
    return prepared


class SpatialExecutionTests(unittest.TestCase):
    def setUp(self):
        self.g = Fixture()
        self.cancel = threading.Event()
        self.plan = op.ObjectProgram.model_validate(program())
        self.g.robot.config = {}
        self.packet = SimpleNamespace(metric_calibrated=True,
            calibration=SimpleNamespace(fingerprint='camera-v1', geometry_revision='model-v1'),
            jaw_geometry=SimpleNamespace(metric_calibrated=True, fingerprint='jaw-v1'))
        guard = patch('server.spatial_runtime.observation_packet', return_value=self.packet)
        guard.start();self.addCleanup(guard.stop)

    def test_calibration_replacement_invalidates_previously_assessed_path(self):
        attach_proposal(self.g, self.plan, self.g.capture())
        self.packet.calibration.fingerprint = 'camera-v2'
        with self.assertRaises(op.ObjectProgramBlocked):
            op.prepare_object_program(self.g, self.plan, self.g.capture())

    def test_fresh_frame_settling_does_not_rebase_approved_trajectory(self):
        original = self.g.capture()
        approved = attach_proposal(self.g, self.plan, original)
        self.g.current['shoulder_pan'] += 2
        fresh = self.g.capture()
        reused = op.prepare_object_program(self.g, self.plan, fresh)
        self.assertEqual(reused, approved)
        op._bind_initial_body_anchor(self.g, reused, fresh, self.cancel)
        self.assertEqual(self.g.robot.bias['shoulder_pan'], 0)
        self.g.robot._write.assert_not_called()

    def test_changed_target_contract_or_goal_cannot_use_prior_assessment(self):
        original = self.g.capture()
        attach_proposal(self.g, self.plan, original)
        changed = self.plan.model_copy(deep=True)
        changed.steps[0].body_target.shoulder_pan += 1
        with self.assertRaises(op.ObjectProgramBlocked):
            op.prepare_object_program(self.g, changed, self.g.capture())
        self.g.robot.goals['shoulder_pan'] += 1
        with self.assertRaises(op.ObjectProgramBlocked):
            op.prepare_object_program(self.g, self.plan, self.g.capture())
        self.g.robot.goals['shoulder_pan'] -= 1
        self.g.spatial_attempt['proposal']['segments'][0]['contract']['joint_targets']['shoulder_pan'] += 1
        with self.assertRaises(ValueError):
            op.prepare_object_program(self.g, self.plan, self.g.capture())

    def test_post_close_lift_dispatches_new_contract_and_preserves_pwm_owner(self):
        def planning(g, query, frame, *args, **kwargs):
            attach_proposal(g, self.plan, frame)
            return self.plan
        def close(*args, **kwargs):
            result = self.g.close(*args, **kwargs)
            self.g.state = 'contact_candidate'
            return result
        self.g.close_candidate.side_effect = close
        replacement = []
        def lift(g, name, frame, context, cancel, *, capture_fresh, remaining_seconds):
            self.assertEqual(frame['joints']['gripper'], 1810)
            self.assertTrue(context['target_between_inner_pads'])
            self.assertTrue(context['pwm_owner_active'])
            origin = capture_fresh()
            target = {**origin['joints'], 'elbow_flex': 2050}
            segment = mp.prepare_joint_segment(g.model, origin['joints'], g.robot.goals, target)
            replacement.append(copy.deepcopy(segment))
            return {'segment': segment, 'frame': origin, 'proposal': {'new_lift': True},
                    'expected_pwm_owner': g.robot.pwm_owner, 'expected_state': g.state,
                    'expected_goals': dict(g.robot.goals)}
        with patch.object(op, 'unchanged_wrist', return_value={}), \
                patch.object(op, 'plan_object_program', side_effect=planning), \
                patch.object(op, 'build_observation_candidates', return_value=[]), \
                patch('server.spatial_object_program.prepare_contact_lift', side_effect=lift) as replan, \
                patch.object(mp, 'execute_prepared_segment', side_effect=self.g.execute) as execute:
            result = op.run_object_program(self.g, 'toy', Mock(), self.cancel)
        self.assertTrue(result['grasp_confirmed'])
        replan.assert_called_once()
        last = execute.call_args
        self.assertEqual(last.args[1], replacement[0])
        self.assertFalse(last.kwargs['continuation'])
        self.assertEqual(self.g.current['elbow_flex'], 2050)
        self.assertEqual(self.g.robot.pwm_owner, 'owned')
        self.g.robot._write.assert_not_called()


if __name__ == '__main__':unittest.main()
