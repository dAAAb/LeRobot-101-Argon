"""Offline contracts for independent wave/lift visual decisions; no hardware."""
from copy import deepcopy
import json
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from server.gripper_power import SessionCancelled
from server.wave_planner import BODY_JOINTS, WaveOptions, RetreatOptions, assess_wave_options, assess_retreat_options


class WavePlannerTests(unittest.TestCase):
    def setUp(self):
        self.frames = [{'source': 'wrist', 'camera_generation': 'camera-1', 'path': 'current.jpg'}]
        self.geometry = {'full_mesh_table_guard': True, 'max_roll_degrees': 20.,
                         'tracking_margin_degrees': 1.05}
        self.proposal = {
            'requested_delta_m': [0., 0., .005],
            'predicted_delta_m': [.00001, -.00001, .004925],
            'full_orientation_error_deg': .08,
            'delta_ticks': dict(zip(BODY_JOINTS, [0., 1., -24., 23., 0.])),
            'clearance': {'model_table_pass': True,
                          'actual_tracking_envelope_ticks': {**dict.fromkeys(BODY_JOINTS, 30), 'gripper': 0}},
        }
        self.response = {
            'wave': {'gripper_empty': True, 'clear_sweep': False, 'human_clear': True,
                     'confidence': .96, 'reason': '圓筒靠近手指，轉動時可能碰到。'},
            'lift': {'gripper_empty': True, 'clear_path': True, 'human_clear': True,
                     'confidence': .94, 'reason': '手指上方可見空間足夠，可先小步抬高再看。'},
        }
        self.brain = SimpleNamespace(_infer=Mock(side_effect=lambda *a, **kw: WaveOptions.model_validate(self.response)))

    def assess(self, **changes):
        args = {'roll_geometry': self.geometry, 'lift_proposal': self.proposal,
                'lifted_mm': 0., 'step_count': 0}
        args.update(changes)
        return assess_wave_options(self.brain, self.frames, **args)

    def payload(self):
        return json.loads(self.brain._infer.call_args.args[0].split('\nINPUT_DATA_JSON\n')[1])

    def test_blocked_wave_and_independently_clear_lift_survive_one_inference(self):
        result = self.assess(lifted_mm=7.8, step_count=3)
        self.assertFalse(result['wave']['clear_sweep'])
        self.assertTrue(result['lift']['clear_path'])
        self.brain._infer.assert_called_once()
        self.assertIs(self.brain._infer.call_args.args[1], self.frames)
        self.assertIs(self.brain._infer.call_args.args[2], WaveOptions)
        data = self.payload()
        self.assertEqual(data['roll_geometry'], self.geometry)
        self.assertEqual(data['repositioning']['lifted_model_mm'], 7.8)
        self.assertTrue(data['repositioning']['model_displacement_not_external_metrology'])

    def test_only_allowlisted_controller_data_reaches_prompt(self):
        self.proposal.update(observer_image='DO-NOT-FORWARD', known_height='DO-NOT-FORWARD',
                             arbitrary_instruction='DO-NOT-FORWARD', physical_goal_targets={'gripper': 1500})
        self.proposal['clearance']['notes'] = 'DO-NOT-FORWARD'
        self.assess()
        prompt = self.brain._infer.call_args.args[0]
        self.assertNotIn('DO-NOT-FORWARD', prompt)
        self.assertEqual(set(self.payload()['lift_proposal']),
                         {'requested_delta_m', 'predicted_delta_m', 'full_orientation_error_deg',
                          'delta_ticks', 'clearance'})

    def test_wave_and_lift_use_local_evidence_without_assuming_projected_object_is_held(self):
        result=self.assess()
        prompt=self.brain._infer.call_args.args[0]
        self.assertIn('Apply these evidence rules independently to wave, lift',prompt)
        self.assertIn('no object being held or engaged between the inner finger pads',prompt)
        self.assertIn('An object merely projected between open fingers',prompt)
        self.assertIn('if contact or depth relevant to pad engagement is unresolved, set gripper_empty=false',prompt)
        self.assertIn('Missing a full-body or shoulder view, or an unrelated unseen region, is not by itself evidence',prompt)
        self.assertIn('concrete occlusion that leaves such a nearby obstruction unresolved',prompt)
        self.assertIn('requires the corresponding clearance flag=false',prompt)
        self.assertIn('COMPLETE stated swept path',prompt)
        self.assertNotIn('gripper_empty requires a visibly empty jaw gap',prompt)
        self.assertEqual(result,self.response)
        forwarded=self.payload()['lift_proposal']['clearance']
        for key,value in self.proposal['clearance'].items():self.assertEqual(forwarded[key],value)
        self.brain._infer.assert_called_once()

    def coordination(self):
        return {'intermediate_delta_m': [.00256, -.00045, .01019],
                'intermediate_orientation_deg': 2.1, 'model_table_pass': True,
                'actual_tip_corridor_mm': {'minimum_up': -1., 'maximum_up': 12.,
                                          'maximum_lateral': 3., 'maximum_rotation_degrees': 3.}}

    def elbow_arc(self):
        proposal=deepcopy(self.proposal)
        endpoint=[.0008,-.0001,.004]
        orientation=10*360/4096
        proposal.update(motion_kind='elbow_raise',requested_delta_m=endpoint,
                        predicted_delta_m=list(endpoint),full_orientation_error_deg=orientation,
                        delta_ticks={**dict.fromkeys(BODY_JOINTS,0),'elbow_flex':-10})
        proposal['clearance']['actual_tracking_envelope_ticks'].update(
            {n:12 for n in BODY_JOINTS if n!='elbow_flex'})
        proposal['arc_path']={
            'sampled_elbow_offsets_ticks':list(range(0,-11,-1)),
            'sampled_delta_m':[[v*i/10 for v in endpoint] for i in range(11)],
            'sampled_orientation_deg':[orientation*i/10 for i in range(11)],
            'actual_tip_corridor_mm':deepcopy(self.coordination()['actual_tip_corridor_mm']),
            'attempt_limits':{'minimum_up_mm':-1.,'maximum_up_mm':60.,'maximum_lateral_mm':15.,
                              'maximum_rotation_degrees':15.,'maximum_other_joint_drift_ticks':4},
        }
        return proposal

    def test_elbow_only_arc_is_explicit_and_independently_assessed_once(self):
        proposal=self.elbow_arc()
        proposal['active_joint_targets']={'elbow_flex':2900}
        proposal['untrusted_notes']='DO-NOT-FORWARD'
        result=self.assess(lift_proposal=proposal,lifted_mm=15.,step_count=3)
        self.assertFalse(result['wave']['clear_sweep']);self.assertTrue(result['lift']['clear_path'])
        self.brain._infer.assert_called_once()
        data=self.payload()['lift_proposal'];prompt=self.brain._infer.call_args.args[0]
        self.assertEqual(data['motion_kind'],'elbow_raise')
        self.assertEqual(data['arc_path'],proposal['arc_path'])
        self.assertEqual(data['requested_delta_m'],data['predicted_delta_m'])
        self.assertNotEqual(data['requested_delta_m'][:2],[0,0])
        self.assertNotIn('coordination',data);self.assertNotIn('active_joint_targets',data)
        self.assertNotIn('DO-NOT-FORWARD',prompt)
        self.assertIn('rotates ONLY the elbow',prompt)
        self.assertIn('does NOT preserve hand orientation',prompt)
        self.assertIn('no later catch-up',prompt)
        self.assertIn('15 degrees rotation',prompt)
        self.assertIn('15 mm lateral displacement',prompt)
        self.assertIn('reserves the entire 3 mm local lateral corridor',prompt)
        self.assertIn('must be at most 12 mm',prompt)
        self.assertIn('Every further arc requires a new image',prompt)

    def test_elbow_arc_declares_upward_vector_not_cartesian_request(self):
        invalid=[]
        for vector in ([0,0,0],[0,0,-.001],[0,0,.0081],[.00201,0,.003],
                       [.002,0,.008],[0,0,float('nan')]):
            p=self.elbow_arc();p['requested_delta_m']=vector;p['predicted_delta_m']=vector
            invalid.append(p)
        p=self.elbow_arc();p['requested_delta_m']=[0,0,.004];invalid.append(p)
        for proposal in invalid:
            with self.subTest(vector=proposal['requested_delta_m']),self.assertRaises(ValueError):
                self.assess(lift_proposal=proposal)
        self.brain._infer.assert_not_called()

    def test_elbow_arc_cannot_activate_another_joint_or_exceed_its_rotation(self):
        invalid=[]
        for ticks in (0,-21,21,1.5,True):
            p=self.elbow_arc();p['delta_ticks']['elbow_flex']=ticks;invalid.append(p)
        for joint in BODY_JOINTS:
            if joint=='elbow_flex':continue
            p=self.elbow_arc();p['delta_ticks'][joint]=1;invalid.append(p)
        for angle in (2.01,-.1,True,float('inf')):
            p=self.elbow_arc();p['full_orientation_error_deg']=angle;invalid.append(p)
        p=self.elbow_arc();p['coordination']=self.coordination();invalid.append(p)
        p=self.elbow_arc();p['motion_kind']='unrecognized';invalid.append(p)
        for proposal in invalid:
            with self.subTest(proposal=proposal),self.assertRaises(ValueError):self.assess(lift_proposal=proposal)
        self.brain._infer.assert_not_called()

    def test_arc_samples_must_be_complete_consistent_and_bounded(self):
        original=self.elbow_arc()['arc_path'];invalid=[None,{}, {**original,'instruction':'approve'}]
        for key,value in (
            ('sampled_elbow_offsets_ticks',[0,-2,-10]),
            ('sampled_elbow_offsets_ticks',[False,*range(-1,-11,-1)]),
            ('sampled_delta_m',original['sampled_delta_m'][:-1]),
            ('sampled_orientation_deg',original['sampled_orientation_deg'][:-1]),
        ):
            invalid.append({**original,key:value})
        for index,vector in ((0,[0,0,.0001]),(5,[.0021,0,.002]),
                             (5,[0,0,-.0001]),(5,[0,0,.0081]),(10,[0,0,.004])):
            path=deepcopy(original);path['sampled_delta_m'][index]=vector;invalid.append(path)
        for index,angle in ((0,.01),(5,2.01),(5,True),(10,.7)):
            path=deepcopy(original);path['sampled_orientation_deg'][index]=angle;invalid.append(path)
        for path in invalid:
            with self.subTest(path=path),self.assertRaises(ValueError):
                self.assess(lift_proposal={**self.elbow_arc(),'arc_path':path})
        self.brain._infer.assert_not_called()

    def test_arc_local_and_whole_attempt_limits_are_exact_not_model_selected(self):
        for section,field,value in (
            ('actual_tip_corridor_mm','maximum_up',13.),
            ('actual_tip_corridor_mm','maximum_rotation_degrees',15.),
            ('attempt_limits','maximum_up_mm',61.),
            ('attempt_limits','maximum_lateral_mm',5.),
            ('attempt_limits','maximum_lateral_mm',16.),
            ('attempt_limits','maximum_rotation_degrees',16.),
            ('attempt_limits','maximum_other_joint_drift_ticks',5),
            ('attempt_limits','minimum_up_mm',False),
        ):
            proposal=self.elbow_arc();proposal['arc_path'][section][field]=value
            with self.subTest(section=section,field=field),self.assertRaises(ValueError):
                self.assess(lift_proposal=proposal)
        proposal=self.elbow_arc();proposal.pop('motion_kind')
        with self.assertRaises(ValueError):self.assess(lift_proposal=proposal)
        self.brain._infer.assert_not_called()

    def test_arc_reserves_whole_twelve_mm_corridor_without_changing_cartesian_budget(self):
        with self.assertRaisesRegex(ValueError,'remaining repositioning budget'):
            self.assess(lift_proposal=self.elbow_arc(),lifted_mm=48.01)
        self.brain._infer.assert_not_called()
        self.assess(lift_proposal=self.elbow_arc(),lifted_mm=48.)
        self.assess(lifted_mm=55.)

    def test_real_larger_elbow_arc_reaches_assessor_with_unchanged_actual_corridor(self):
        from server.local_cartesian import ALL, default_model
        from server.wave_arc import propose_raise
        raw=dict(zip(ALL,[2161,1138,2654,2332,984,1858]))
        proposal,_=propose_raise(default_model(),raw,{**raw,'elbow_flex':2647},raw)
        self.assertEqual(proposal['delta_ticks']['elbow_flex'],-17)
        result=self.assess(lift_proposal=proposal)
        self.assertFalse(result['wave']['clear_sweep'])
        self.assertTrue(result['lift']['clear_path'])
        data=self.payload()['lift_proposal'];prompt=self.brain._infer.call_args.args[0]
        self.assertEqual(data['arc_path'],proposal['arc_path'])
        self.assertGreater(data['predicted_delta_m'][2],.005)
        self.assertIn('at-most-20 ticks',prompt)
        self.assertIn('sideways up to 2 mm',prompt)
        self.assertIn('total length is at most 8 mm',prompt)
        self.assertIn('at most 2 degrees nominal rotation',prompt)
        self.assertIn('3 mm lateral displacement, and 3 degrees actual local rotation',prompt)
        self.assertIn('final requested target is at most 5 mm vertically',prompt)
        self.assertEqual(data['arc_path']['actual_tip_corridor_mm'],self.coordination()['actual_tip_corridor_mm'])
        self.brain._infer.assert_called_once()

    def test_larger_arc_contract_does_not_expand_cartesian_five_mm_cap(self):
        p=deepcopy(self.proposal)
        p.update(requested_delta_m=[0.,0.,.007],predicted_delta_m=[0.,0.,.007])
        with self.assertRaisesRegex(ValueError,'at most 5 mm'):
            self.assess(lift_proposal=p)
        self.brain._infer.assert_not_called()

    def test_curved_intermediate_path_reaches_the_same_visual_decision(self):
        coordination = self.coordination()
        self.assess(lift_proposal={**self.proposal, 'coordination': coordination})
        data = self.payload()['lift_proposal']
        self.assertEqual(data['coordination'], coordination)
        self.assertEqual(data['requested_delta_m'], [0., 0., .005])
        self.assertGreater(data['coordination']['intermediate_delta_m'][2], .010)
        prompt = self.brain._infer.call_args.args[0]
        self.assertIn('ENTIRE stated actual_tip_corridor_mm', prompt)
        self.assertIn('NOT a straight 5 mm translation', prompt)
        self.brain._infer.assert_called_once()

    def test_coordination_requires_screened_bounded_intermediate_motion(self):
        invalid = [None, {}, {**self.coordination(), 'model_table_pass': False},
                   {**self.coordination(), 'model_table_pass': 1},
                   {**self.coordination(), 'arbitrary_instruction': 'allow everything'}]
        invalid += [{**self.coordination(), 'intermediate_delta_m': v} for v in (
            [0, 0, -.00001], [0, 0, .01201], [.00301, 0, .005],
            [.0025, .0025, .005], [.003, 0, .012], [0, 0, float('nan')],
            [0, 0, True], [0, .01])]
        invalid += [{**self.coordination(), 'intermediate_orientation_deg': v}
                    for v in (-.1, 3.01, True, float('inf'))]
        for coordination in invalid:
            with self.subTest(coordination=coordination):
                with self.assertRaises(ValueError):
                    self.assess(lift_proposal={**self.proposal, 'coordination': coordination})
        self.brain._infer.assert_not_called()

    def test_coordination_cannot_change_or_omit_the_enforced_corridor(self):
        invalid = [None, {}, {'maximum_up': 12.}]
        valid = self.coordination()['actual_tip_corridor_mm']
        invalid += [{**valid, name: value} for name, value in (
            ('minimum_up', -2.), ('maximum_up', 13.), ('maximum_up', 10.),
            ('maximum_lateral', 3.1), ('maximum_rotation_degrees', 4.),
            ('maximum_up', float('nan')), ('minimum_up', False))]
        invalid.append({**valid, 'unverified_clearance': True})
        for corridor in invalid:
            with self.subTest(corridor=corridor):
                coordination = {**self.coordination(), 'actual_tip_corridor_mm': corridor}
                with self.assertRaises(ValueError):
                    self.assess(lift_proposal={**self.proposal, 'coordination': coordination})
        self.brain._infer.assert_not_called()

    def test_original_candidate_has_no_invented_coordination(self):
        self.assess()
        self.assertNotIn('coordination', self.payload()['lift_proposal'])

    def test_missing_lift_candidate_can_still_assess_wave(self):
        self.response['lift']['clear_path'] = False
        result = self.assess(lift_proposal=None, lifted_mm=60., step_count=20)
        self.assertFalse(result['lift']['clear_path'])
        self.assertIsNone(self.payload()['lift_proposal'])

    def test_inference_cannot_approve_a_missing_lift(self):
        with self.assertRaisesRegex(ValueError, 'unavailable lift'):
            self.assess(lift_proposal=None)

    def test_wrist_only_and_same_generation_required_before_inference(self):
        for frames in ([], [{'source': 'observer', 'camera_generation': 'a'}],
                       [{'source': 'wrist'}], [{'source': 'wrist', 'camera_generation': ''}],
                       [{'source': 'wrist', 'camera_generation': 'a'},
                        {'source': 'wrist', 'camera_generation': 'b'}]):
            with self.subTest(frames=frames):
                self.frames = frames
                with self.assertRaises(ValueError): self.assess()
        self.brain._infer.assert_not_called()

    def test_previous_wrist_views_are_passed_in_order(self):
        self.frames.insert(0, {**self.frames[0], 'path': 'before.jpg'})
        self.assess()
        self.assertEqual(self.payload()['current_frame_index'], 1)
        self.assertEqual(self.payload()['wrist_frame_count'], 2)
        self.assertEqual([f['path'] for f in self.brain._infer.call_args.args[1]], ['before.jpg', 'current.jpg'])

    def test_invalid_candidate_vectors_fail_before_inference(self):
        cases = [('requested_delta_m', v) for v in (
            [0, 0, 0], [0, 0, -.001], [.0001, 0, .003], [0, 0, .0051],
            [0, 0, float('nan')], [0, 0, True], [0, .003])]
        cases += [('predicted_delta_m', v) for v in (
            [0, 0, -.001], [0, 0, .0052], [0, 0, .003], [.002, 0, .004],
            [0, 0, float('inf')])]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                proposal = {**self.proposal, key: value}
                with self.assertRaises(ValueError): self.assess(lift_proposal=proposal)
        self.brain._infer.assert_not_called()

    def test_model_guard_and_complete_conservative_error_box_are_required(self):
        cases = [None, {'model_table_pass': False}, {'model_table_pass': 1},
                 {'model_table_pass': True}]
        for field, value in [('elbow_flex', 29), ('wrist_flex', 61), ('gripper', 1),
                             ('shoulder_pan', True), ('shoulder_lift', 30.)]:
            clearance = deepcopy(self.proposal['clearance'])
            clearance['actual_tracking_envelope_ticks'][field] = value
            cases.append(clearance)
        incomplete = deepcopy(self.proposal['clearance'])
        del incomplete['actual_tracking_envelope_ticks']['elbow_flex']
        cases.append(incomplete)
        for clearance in cases:
            with self.subTest(clearance=clearance):
                with self.assertRaises(ValueError):
                    self.assess(lift_proposal={**self.proposal, 'clearance': clearance})
        self.brain._infer.assert_not_called()

    def test_stationary_axes_accept_their_tighter_enforced_error_box(self):
        proposal = deepcopy(self.proposal)
        envelope = proposal['clearance']['actual_tracking_envelope_ticks']
        envelope.update(shoulder_pan=12, wrist_roll=12)
        self.assess(lift_proposal=proposal)
        forwarded = self.payload()['lift_proposal']['clearance']['actual_tracking_envelope_ticks']
        self.assertEqual(forwarded, envelope)
        self.assertEqual(forwarded['elbow_flex'], 30)
        self.assertEqual(forwarded['wrist_flex'], 30)

    def test_active_axes_cannot_use_the_stationary_axis_envelope(self):
        for axis in BODY_JOINTS:
            with self.subTest(axis=axis):
                proposal = deepcopy(self.proposal)
                proposal['delta_ticks'][axis] = 1
                proposal['clearance']['actual_tracking_envelope_ticks'][axis] = 29
                with self.assertRaises(ValueError): self.assess(lift_proposal=proposal)
        self.brain._infer.assert_not_called()

    def test_stationary_axis_envelope_cannot_be_smaller_than_twelve_ticks(self):
        for axis in ('shoulder_pan', 'wrist_roll'):
            with self.subTest(axis=axis):
                proposal = deepcopy(self.proposal)
                proposal['clearance']['actual_tracking_envelope_ticks'][axis] = 11
                with self.assertRaises(ValueError): self.assess(lift_proposal=proposal)
        self.brain._infer.assert_not_called()

    def test_body_trust_region_and_orientation_cannot_expand(self):
        bad_deltas = [{**self.proposal['delta_ticks'], 'elbow_flex': -31},
                      {**self.proposal['delta_ticks'], 'gripper': 0},
                      {'wrist_flex': 20}, {**self.proposal['delta_ticks'], 'wrist_flex': 2.2}]
        for delta in bad_deltas:
            with self.subTest(delta=delta):
                with self.assertRaises(ValueError): self.assess(lift_proposal={**self.proposal, 'delta_ticks': delta})
        for orientation in (2.01, -1, True, float('nan')):
            with self.subTest(orientation=orientation):
                with self.assertRaises(ValueError):
                    self.assess(lift_proposal={**self.proposal, 'full_orientation_error_deg': orientation})
        self.brain._infer.assert_not_called()

    def test_bounded_attempt_context_and_remaining_budget(self):
        for changes in ({'lifted_mm': -1}, {'lifted_mm': 61}, {'lifted_mm': float('nan')},
                        {'step_count': True}, {'step_count': 21}, {'step_count': -1},
                        {'step_count': 20}, {'lifted_mm': 55.01}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError): self.assess(**changes)
        self.brain._infer.assert_not_called()
        self.assess(lifted_mm=55., step_count=19)

    def test_strict_response_cannot_coerce_flags_or_add_action(self):
        for changes in ({'clear_path': 'true'}, {'human_clear': 1}, {'confidence': 1.01},
                        {'direction': 'forward'}, {'reason': '字' * 161}):
            with self.subTest(changes=changes):
                response = {**self.response, 'lift': {**self.response['lift'], **changes}}
                with self.assertRaises(ValidationError): WaveOptions.model_validate(response)
        with self.assertRaises(ValidationError):
            WaveOptions.model_validate({**self.response, 'motor_goal': 2000})

    def test_inference_result_is_revalidated(self):
        self.brain._infer.side_effect = None
        self.brain._infer.return_value = {**self.response, 'lift': {**self.response['lift'], 'clear_path': 'yes'}}
        with self.assertRaises(ValidationError): self.assess()

    def test_cancelled_attempt_does_not_start_inference(self):
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(SessionCancelled): self.assess(cancel=cancel)
        self.brain._infer.assert_not_called()

    def test_real_pure_cartesian_candidate_is_accepted(self):
        from server.local_cartesian import default_model
        raw = dict(zip((*BODY_JOINTS, 'gripper'), [2161, 1139, 2983, 2342, 985, 1857]))
        model = default_model()
        proposal = model.propose_adaptive(raw, [0., 0., .005], held_goals=raw, uncertainty_ticks=30)
        self.assess(lift_proposal=proposal)
        self.assertTrue(self.payload()['lift_proposal']['clearance']['model_table_pass'])
        self.assertGreater(self.payload()['lift_proposal']['predicted_delta_m'][2], .004)

    def test_real_elbow_arc_geometry_matches_visual_contract(self):
        from server.local_cartesian import default_model
        from server.wave_arc import propose_raise
        raw=dict(zip((*BODY_JOINTS,'gripper'),[2162,1141,2839,2429,984,1857]))
        held={**raw,'elbow_flex':raw['elbow_flex']-32}
        proposal,_=propose_raise(default_model(),raw,held,raw)
        self.assess(lift_proposal=proposal)
        data=self.payload()['lift_proposal']
        self.assertEqual(data['motion_kind'],'elbow_raise')
        self.assertEqual(data['arc_path']['sampled_delta_m'][-1],data['predicted_delta_m'])
        self.assertGreater(data['predicted_delta_m'][2],.004)
        self.assertEqual(data['clearance']['actual_tracking_envelope_ticks']['elbow_flex'],38)


class RetreatPlannerTests(unittest.TestCase):
    def setUp(self):
        self.frames=[{'source':'wrist','camera_generation':'camera-1','path':'current.jpg'}]
        self.geometry={'full_mesh_table_guard':True,'max_roll_degrees':10.,'tracking_margin_degrees':1.05}
        self.candidates={side:self.candidate(side) for side in ('left','right')}
        self.candidates['backward']=None
        self.response={
            'wave':{'gripper_empty':True,'clear_sweep':False,'human_clear':True,
                    'confidence':.82,'reason':'圓筒還靠近指尖，先不要轉腕。'},
            'left':{'gripper_empty':True,'clear_path':True,'human_clear':True,
                    'confidence':.78,'reason':'左側路徑可見為空，可稍微移開指尖。'},
            'right':{'gripper_empty':True,'clear_path':False,'human_clear':True,
                     'confidence':.91,'reason':'右側可能往圓筒靠近，先不要移動。'},
            'backward':{'gripper_empty':True,'clear_path':False,'human_clear':True,
                        'confidence':.8,'reason':'這次尚未提供後退路徑。'},
        }
        self.brain=SimpleNamespace(_infer=Mock(side_effect=lambda *a,**kw:RetreatOptions.model_validate(self.response)))

    def candidate(self,side):
        sign=-1 if side=='left' else 1
        vector=[sign*.004,.0003,.0002];angle=10*360/4096
        return {'motion_kind':'pan_retreat','image_side':side,'camera_relative_delta_m':list(vector),
                'requested_delta_m':list(vector),'predicted_delta_m':list(vector),
                'full_orientation_error_deg':angle,
                'delta_ticks':{**dict.fromkeys(BODY_JOINTS,0),'shoulder_pan':sign*10},
                'clearance':{'model_table_pass':True,'actual_tracking_envelope_ticks':{
                    **dict.fromkeys(BODY_JOINTS,12),'shoulder_pan':30,'gripper':0}},
                'retreat_path':{
                    'sampled_pan_offsets_ticks':list(range(0,sign*11,sign)),
                    'sampled_delta_m':[[v*i/10 for v in vector] for i in range(11)],
                    'sampled_orientation_deg':[angle*i/10 for i in range(11)],
                    'actual_tip_corridor_mm':{'minimum_up':-3.,'maximum_up':3.,'maximum_lateral':12.,'maximum_rotation_degrees':3.},
                    'attempt_limits':{'minimum_up_mm':-5.,'maximum_up_mm':5.,'maximum_lateral_mm':40.,
                                      'maximum_rotation_degrees':12.,'maximum_other_joint_drift_ticks':4}}}

    def backward_candidate(self):
        joints=['shoulder_lift','elbow_flex','wrist_flex']
        offsets=[[a,b,c] for a in (0,-18) for b in (0,-1) for c in (0,19)]
        vectors=[];angles=[]
        for a,b,c in offsets:
            sa=a/-18;eb=b/-1;wr=c/19
            vectors.append([.002*sa-.002*wr,0.,-.001*sa-.0001*eb-.0019*wr])
            angles.append(abs(-.02*a+.01*b+.01*c))
        return {'motion_kind':'cartesian_backward','image_side':'backward',
                'requested_delta_m':[0.,0.,-.003],'predicted_delta_m':vectors[-1],
                'camera_relative_delta_m':vectors[-1],
                'full_orientation_error_deg':angles[-1],'orientation_reference_error_deg':angles[-1],
                'delta_ticks':dict(zip(BODY_JOINTS,[0,-18,-1,19,0])),
                'clearance':{'model_table_pass':True,'actual_tracking_envelope_ticks':{
                    **dict.fromkeys(BODY_JOINTS,30),'shoulder_pan':12,'wrist_roll':12,'gripper':0}},
                'asynchronous_clearance':[{'model_table_pass':True} for _ in range(16)],
                'backward_path':{
                    'backward_direction_base':[0.,0.,-1.],'coordinated_joints':joints,
                    'sampled_corner_offsets_ticks':offsets,'sampled_corner_delta_m':vectors,
                    'sampled_corner_orientation_deg':angles,
                    'actual_tip_corridor_mm':{'minimum_back':-1.,'maximum_back':12.,
                                               'maximum_transverse':12.,'maximum_rotation_degrees':3.},
                    'attempt_limits':{'minimum_back_mm':-1.,'maximum_back_mm':40.,
                                      'maximum_transverse_mm':12.,'maximum_rotation_degrees':3.,
                                      'maximum_fixed_joint_drift_ticks':4}}}

    def assess(self,**changes):
        args={'roll_geometry':self.geometry,'retreat_proposals':self.candidates,'retreated_mm':0.,'step_count':0}
        args.update(changes)
        return assess_retreat_options(self.brain,self.frames,**args)

    payload=WavePlannerTests.payload

    def test_wave_and_two_distinct_sides_are_assessed_once_without_selection_or_score_changes(self):
        result=self.assess(retreated_mm=12.,step_count=3)
        self.assertEqual(result,self.response)
        self.brain._infer.assert_called_once()
        self.assertIs(self.brain._infer.call_args.args[1],self.frames)
        self.assertIs(self.brain._infer.call_args.args[2],RetreatOptions)
        data=self.payload()
        self.assertEqual(set(data['retreat_proposals']),{'left','right','backward'})
        self.assertLess(data['retreat_proposals']['left']['camera_relative_delta_m'][0],0)
        self.assertGreater(data['retreat_proposals']['right']['camera_relative_delta_m'][0],0)
        self.assertEqual(data['repositioning']['retreated_model_mm'],12.)
        self.assertTrue(data['repositioning']['new_observation_required_after_each_step'])
        prompt=self.brain._infer.call_args.args[0]
        self.assertIn('ONLY shoulder_pan is commanded',prompt)
        self.assertIn('DO NOT move a finger through the cylinder',prompt)
        self.assertIn('or retry a rejected judgment',prompt)
        self.assertIn('NOT a calibrated pixel projection',prompt)

    def test_retreat_assesses_future_corners_without_demanding_future_photos_or_rewriting_blocks(self):
        self.candidates['backward']=self.backward_candidate()
        for kind,answer in self.response.items():
            answer['human_clear']=False
            answer['clear_sweep' if kind=='wave' else 'clear_path']=False
        result=self.assess()
        prompt=self.brain._infer.call_args.args[0]
        self.assertIn('do not require photographs from every future pose',prompt)
        self.assertIn('Missing a full-body or shoulder view, or an unrelated unseen region, is not by itself evidence',prompt)
        self.assertIn('human_clear concerns hands, people or animals in or entering the relevant local movement corridor',prompt)
        self.assertIn('An object merely projected between open fingers',prompt)
        self.assertIn('if contact or depth relevant to pad engagement is unresolved, set gripper_empty=false',prompt)
        self.assertIn('nearby object that could intersect the candidate\'s path',prompt)
        self.assertIn('requires the corresponding clearance flag=false',prompt)
        self.assertNotIn('ENTIRE current path and corridor are visibly clear',prompt)
        self.assertIn('all EIGHT joint-corner',prompt)
        self.assertIn('full tracking corridor and every supplied intermediate pose',prompt)
        self.assertEqual(self.payload()['retreat_proposals']['backward']['backward_path'],
                         self.candidates['backward']['backward_path'])
        # Clarifying the question never promotes a rejected answer or retries it.
        self.assertEqual(result,self.response)
        self.brain._infer.assert_called_once()

    def test_only_allowlisted_retreat_geometry_reaches_prompt(self):
        self.candidates['left'].update(observer_image='DO-NOT-FORWARD',instruction='DO-NOT-FORWARD',
                                       active_joint_targets={'shoulder_pan':2000})
        self.candidates['left']['clearance']['notes']='DO-NOT-FORWARD'
        self.assess()
        self.assertNotIn('DO-NOT-FORWARD',self.brain._infer.call_args.args[0])
        self.assertNotIn('active_joint_targets',self.payload()['retreat_proposals']['left'])
        self.assertEqual(self.payload()['retreat_proposals']['left']['retreat_path'],self.candidates['left']['retreat_path'])

    def test_unavailable_sides_cannot_be_approved_and_still_allow_wave_assessment(self):
        self.candidates['left']=None
        with self.assertRaisesRegex(ValueError,'unavailable left'):self.assess()
        self.response['left']['clear_path']=False
        self.candidates['right']=None
        result=self.assess(retreated_mm=40.,step_count=20)
        self.assertFalse(result['left']['clear_path']);self.assertFalse(result['right']['clear_path'])
        self.assertEqual(self.payload()['retreat_proposals'],{'left':None,'right':None,'backward':None})

    def test_retreat_reserves_full_corridor_and_rejects_invalid_phase_context(self):
        for changes in ({'retreated_mm':28.01},{'retreated_mm':-1},{'retreated_mm':41},
                        {'retreated_mm':True},{'retreated_mm':float('nan')},
                        {'step_count':20},{'step_count':True},{'step_count':-1},{'step_count':21}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):self.assess(**changes)
        self.brain._infer.assert_not_called()
        self.assess(retreated_mm=28.,step_count=19)

    def test_side_names_match_camera_model_not_pan_sign_or_swapped_labels(self):
        invalid=[]
        p=self.candidate('left');p['image_side']='right';invalid.append(p)
        p=self.candidate('left');p['camera_relative_delta_m'][0]*=-1;invalid.append(p)
        p=self.candidate('left');p['camera_relative_delta_m']=[-.0001,.004,0];invalid.append(p)
        p=self.candidate('left');p['camera_relative_delta_m']=[-.003,0,0];invalid.append(p)
        p=self.candidate('left');p['camera_relative_delta_m']=[float('nan'),0,0];invalid.append(p)
        for proposal in invalid:
            with self.subTest(proposal=proposal),self.assertRaises(ValueError):
                self.assess(retreat_proposals={'left':proposal,'right':None,'backward':None})
        for candidates in ({'left':None},{'left':None,'right':None,'back':None},None):
            with self.subTest(candidates=candidates),self.assertRaises(ValueError):self.assess(retreat_proposals=candidates)
        self.brain._infer.assert_not_called()

    def test_retreat_nominal_vectors_and_active_axis_remain_bounded(self):
        invalid=[]
        for vector in ([-.006,0,0],[-.004,0,.0011],[0,.0004,0],[0,0,float('inf')]):
            p=self.candidate('left');p.update(requested_delta_m=vector,predicted_delta_m=vector,camera_relative_delta_m=vector)
            invalid.append(p)
        p=self.candidate('left');p['requested_delta_m']=[-.003,0,0];invalid.append(p)
        for joint,value in (('shoulder_pan',0),('shoulder_pan',-11),('shoulder_pan',-.5),
                            ('elbow_flex',1),('wrist_flex',1),('shoulder_pan',True)):
            p=self.candidate('left');p['delta_ticks'][joint]=value;invalid.append(p)
        for angle in (1.01,-1,True):
            p=self.candidate('left');p['full_orientation_error_deg']=angle;invalid.append(p)
        for proposal in invalid:
            with self.subTest(proposal=proposal),self.assertRaises(ValueError):
                self.assess(retreat_proposals={'left':proposal,'right':None,'backward':None})
        self.brain._infer.assert_not_called()

    def test_retreat_sample_path_and_declared_limits_cannot_diverge(self):
        invalid=[]
        for key,value in (('sampled_pan_offsets_ticks',[0,-10]),('sampled_delta_m',[]),
                          ('sampled_orientation_deg',[0]),('unknown_instruction','approve')):
            p=self.candidate('left');p['retreat_path'][key]=value;invalid.append(p)
        for index,vector in ((0,[.0001,0,0]),(5,[-.006,0,0]),(5,[-.002,0,.0011]),(10,[-.003,0,0])):
            p=self.candidate('left');p['retreat_path']['sampled_delta_m'][index]=vector;invalid.append(p)
        for index,angle in ((0,.1),(5,1.1),(10,.7)):
            p=self.candidate('left');p['retreat_path']['sampled_orientation_deg'][index]=angle;invalid.append(p)
        for section,key,value in (
            ('actual_tip_corridor_mm','maximum_lateral',13.),('actual_tip_corridor_mm','minimum_up',-4.),
            ('attempt_limits','maximum_lateral_mm',41.),('attempt_limits','maximum_rotation_degrees',15.),
            ('attempt_limits','maximum_other_joint_drift_ticks',5)):
            p=self.candidate('left');p['retreat_path'][section][key]=value;invalid.append(p)
        for proposal in invalid:
            with self.subTest(proposal=proposal),self.assertRaises(ValueError):
                self.assess(retreat_proposals={'left':proposal,'right':None,'backward':None})
        self.brain._infer.assert_not_called()

    def test_retreat_requires_table_screen_and_complete_conservative_error_box(self):
        invalid=[]
        for value in (False,1):
            p=self.candidate('left');p['clearance']['model_table_pass']=value;invalid.append(p)
        for joint,value in (('shoulder_pan',29),('elbow_flex',11),('wrist_roll',61),('gripper',1),('wrist_flex',True)):
            p=self.candidate('left');p['clearance']['actual_tracking_envelope_ticks'][joint]=value;invalid.append(p)
        for proposal in invalid:
            with self.subTest(proposal=proposal),self.assertRaises(ValueError):
                self.assess(retreat_proposals={'left':proposal,'right':None,'backward':None})
        self.brain._infer.assert_not_called()

    def test_retreat_keeps_wrist_only_generation_and_cancellation_guards(self):
        original=self.frames
        for frames in ([],[{'source':'observer','camera_generation':'a'}],
                       [{'source':'wrist','camera_generation':'a'},{'source':'wrist','camera_generation':'b'}]):
            self.frames=frames
            with self.subTest(frames=frames),self.assertRaises(ValueError):self.assess()
        self.frames=original;cancel=threading.Event();cancel.set()
        with self.assertRaises(SessionCancelled):self.assess(cancel=cancel)
        self.brain._infer.assert_not_called()

    def test_retreat_response_revalidation_rejects_coercion_or_extra_motion(self):
        self.brain._infer.side_effect=None
        for addition in ({'clear_path':'true'},{'human_clear':1},{'confidence':1.1},{'motor_goal':2100}):
            self.brain._infer.return_value={**self.response,'left':{**self.response['left'],**addition}}
            with self.subTest(addition=addition),self.assertRaises(ValidationError):self.assess()

    def test_real_pan_retreat_candidates_match_camera_side_and_visual_contract(self):
        from server.local_cartesian import default_model
        from server.wave_retreat import propose_retreat
        model=default_model()
        raw=dict(zip((*BODY_JOINTS,'gripper'),[2162,1141,2727,2429,984,1857]))
        held={**raw,'elbow_flex':raw['elbow_flex']-32}
        candidates={'backward':None}
        for sign in (-1,1):
            proposal,_=propose_retreat(model,raw,held,raw,sign)
            candidates[proposal['image_side']]=proposal
        self.assess(retreat_proposals=candidates)
        data=self.payload()['retreat_proposals']
        self.assertEqual(set(data),{'left','right','backward'})
        for side in ('left','right'):
            self.assertEqual(data[side]['retreat_path']['sampled_delta_m'][-1],data[side]['predicted_delta_m'])
            self.assertEqual(data[side]['clearance']['actual_tracking_envelope_ticks']['elbow_flex'],38)
        self.assertLess(data['left']['camera_relative_delta_m'][0],-.0005)
        self.assertGreater(data['right']['camera_relative_delta_m'][0],.0005)

    def test_backward_withdrawal_is_assessed_with_both_sides_and_wave_in_one_call(self):
        proposal=self.backward_candidate();proposal['operator_note']='DO-NOT-FORWARD'
        proposal['asynchronous_clearance'][0]['notes']='DO-NOT-FORWARD'
        self.candidates['backward']=proposal
        self.response['backward'].update(clear_path=True,confidence=.92,reason='後方通道可見為空，可以小步退出圓筒前方。')
        result=self.assess()
        self.assertEqual(result,self.response)
        self.brain._infer.assert_called_once()
        forwarded=self.payload()['retreat_proposals']['backward']
        self.assertEqual(forwarded['backward_path'],proposal['backward_path'])
        self.assertTrue(forwarded['asynchronous_model_table_pass'])
        self.assertNotIn('asynchronous_clearance',forwarded)
        prompt=self.brain._infer.call_args.args[0]
        self.assertNotIn('DO-NOT-FORWARD',prompt)
        self.assertIn('FOUR DISTINCT candidate actions',prompt)
        self.assertIn('all EIGHT joint-corner',prompt)
        self.assertIn('An exact endpoint remains within 1 mm',prompt)
        self.assertIn('useful partial progress',prompt)
        self.assertIn('not task completion or grasp verification',prompt)
        self.assertIn('DO NOT move a finger through the cylinder',prompt)

    def test_backward_is_required_in_schema_and_unavailable_candidate_cannot_be_approved(self):
        response=deepcopy(self.response);response.pop('backward')
        with self.assertRaises(ValidationError):RetreatOptions.model_validate(response)
        self.response['backward']['clear_path']=True
        with self.assertRaisesRegex(ValueError,'unavailable backward'):self.assess()

    def test_backward_cannot_omit_or_fake_asynchronous_table_checks(self):
        valid=self.backward_candidate()
        invalid=[None,valid['asynchronous_clearance'][:-1]]
        invalid.extend([[{'model_table_pass':value}]+deepcopy(valid['asynchronous_clearance'][1:])
                        for value in (False,1,'true')])
        for checks in invalid:
            proposal={**valid,'asynchronous_clearance':checks}
            with self.subTest(checks=checks),self.assertRaises(ValueError):
                self.assess(retreat_proposals={**self.candidates,'backward':proposal})
        self.brain._infer.assert_not_called()

    def test_backward_corner_box_is_complete_bounded_and_matches_endpoint(self):
        invalid=[]
        for field,value in (('sampled_corner_offsets_ticks',[[0,0,0],[-18,-1,19]]),
                            ('sampled_corner_delta_m',[]),('sampled_corner_orientation_deg',[0]),
                            ('coordinated_joints',['wrist_flex','elbow_flex','shoulder_lift']),
                            ('backward_direction_base',[0,0,-2])):
            p=self.backward_candidate();p['backward_path'][field]=value;invalid.append(p)
        for index,vector in ((0,[.001,0,0]),(3,[0,0,.0011]),(3,[0,0,-.0121]),
                             (3,[.0121,0,0]),(7,[0,0,-.002])):
            p=self.backward_candidate();p['backward_path']['sampled_corner_delta_m'][index]=vector;invalid.append(p)
        p=self.backward_candidate();p['backward_path']['sampled_corner_orientation_deg'][3]=3.01;invalid.append(p)
        for section,field,value in (('actual_tip_corridor_mm','maximum_transverse',13),
                                   ('attempt_limits','maximum_back_mm',41),
                                   ('attempt_limits','maximum_fixed_joint_drift_ticks',5)):
            p=self.backward_candidate();p['backward_path'][section][field]=value;invalid.append(p)
        for proposal in invalid:
            with self.subTest(proposal=proposal),self.assertRaises(ValueError):
                self.assess(retreat_proposals={**self.candidates,'backward':proposal})
        self.brain._infer.assert_not_called()

    def test_backward_nominal_direction_joint_axes_and_orientation_stay_constrained(self):
        invalid=[]
        for field,value in (('requested_delta_m',[0,0,-.0031]),('predicted_delta_m',[0,0,-.0041]),
                            ('camera_relative_delta_m',[0,0,.003]),('image_side','left'),
                            ('orientation_reference_error_deg',1.01),('full_orientation_error_deg',2.01)):
            p=self.backward_candidate();p[field]=value;invalid.append(p)
        for name,value in (('shoulder_pan',1),('wrist_roll',1),('wrist_flex',31),('elbow_flex',True)):
            p=self.backward_candidate();p['delta_ticks'][name]=value;invalid.append(p)
        p=self.backward_candidate();p['clearance']['actual_tracking_envelope_ticks']['shoulder_lift']=29;invalid.append(p)
        for proposal in invalid:
            with self.subTest(proposal=proposal),self.assertRaises(ValueError):
                self.assess(retreat_proposals={**self.candidates,'backward':proposal})
        self.brain._infer.assert_not_called()

    def test_real_backward_geometry_and_all_asynchronous_checks_match_visual_contract(self):
        from server.local_cartesian import default_model
        from server.wave_backward import propose_backward
        raw=dict(zip((*BODY_JOINTS,'gripper'),[2162,1141,2694,2429,984,1857]))
        held={n:v+1 if n!='gripper' else v for n,v in raw.items()}
        held['elbow_flex']=raw['elbow_flex']-41
        proposal,_=propose_backward(default_model(),raw,held,raw)
        self.assess(retreat_proposals={**self.candidates,'backward':proposal})
        data=self.payload()['retreat_proposals']['backward']
        self.assertTrue(data['asynchronous_model_table_pass'])
        self.assertEqual(len(data['backward_path']['sampled_corner_delta_m']),8)
        self.assertEqual(data['delta_ticks'],dict(zip(BODY_JOINTS,[0.,-18.,-1.,19.,0.])))
        self.assertLess(data['camera_relative_delta_m'][2],-.002)


if __name__ == '__main__':
    unittest.main()
