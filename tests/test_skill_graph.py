import unittest
from copy import deepcopy
from server.skill_graph import compile_graph, checkpoint, SkillGraphRun, replay


def step(skill='observe',**kw):
    return {'skill':skill,'direction':'up' if skill=='move' else 'none',
            'distance_mm':3. if skill=='move' else 0.,'target':'cup' if skill=='grasp' else 'put cup on box' if skill in ('place','stack') else '',
            'label':skill,**kw}

def plan(*steps):return {'status':'ready','summary':'test','steps':list(steps),'unsupported_reason':''}

def observation(visible=True):return {'skill':'observe','completed':True,'observation':{'visible':visible}}


class SkillGraphTests(unittest.TestCase):
    def test_compiled_graph_is_finite_with_fixed_failure_edges(self):
        value=compile_graph(plan(step('move'),step('wave')))
        self.assertEqual(value['maximum_node_visits'],2)
        self.assertEqual(value['automatic_motion_retries'],0)
        self.assertEqual(value['nodes'][0]['on_verified'],'step_1')
        self.assertEqual(value['nodes'][1]['on_verified'],'completed')
        self.assertFalse(value['generated_code'])

    def test_same_target_observation_binds_grasp_but_not_wave(self):
        value=compile_graph(plan(step('observe',target=' cup '),step('grasp')))
        self.assertEqual(value['nodes'][1]['requires_visible_from'],'step_0')
        for steps in ([step('observe',target='cup'),step('wave')],
                      [step('observe',target='box'),step('grasp')],
                      [step('observe',target='cup'),step('move'),step('grasp')],
                      [step('grasp')]):
            self.assertIsNone(compile_graph(plan(*steps))['nodes'][-1]['requires_visible_from'])

    def test_observation_absence_is_successful_inspection(self):
        value=replay(plan(step()),[{'node':'step_0','receipt':observation(False)}])
        self.assertTrue(value['completed']);self.assertTrue(value['replay'])
        self.assertFalse(value['physical_success_verified'])

    def test_observation_absent_then_wave_can_complete(self):
        value=replay(plan(step('observe',target='cup'),step('wave')),[
            {'node':'step_0','receipt':observation(False)},
            {'node':'step_1','receipt':{'skill':'wave','completed':True,'receipt':{'wave_completed':True}}}])
        self.assertTrue(value['completed'])

    def test_dependent_grasp_requires_current_visibility_binding(self):
        for dependency in (None,{'source_node':'other','current':True,'visible':True},
                           {'source_node':'step_0','current':False,'visible':True},
                           {'source_node':'step_0','current':True,'visible':False}):
            value=replay(plan(step('observe',target='cup'),step('grasp')),[
                {'node':'step_0','receipt':observation()},
                {'node':'step_1','receipt':{'skill':'grasp','completed':True,'receipt':{'completed':True,'grasp_confirmed':True}},'dependency':dependency}])
            self.assertEqual(value['active_node'],'blocked')

    def test_partial_distance_never_completes_move_node(self):
        base={'skill':'move','completed':True,'requested_mm':3.,'measured_model_progress_mm':1.5,'step_count':1}
        self.assertFalse(checkpoint(step('move'),base)['passed'])
        self.assertTrue(checkpoint(step('move'),{**base,'measured_model_progress_mm':3.})['passed'])
        for bad in (float('nan'),True,7.):
            self.assertFalse(checkpoint(step('move'),{**base,'measured_model_progress_mm':bad})['passed'])

    def test_flag_alone_cannot_prove_wave_grasp_or_jaw_completion(self):
        for skill in ('wave','grasp','open','release','observe'):
            self.assertFalse(checkpoint(step(skill),{'skill':skill,'completed':True})['passed'])
        self.assertFalse(checkpoint(step('wave'),{'skill':'wave','completed':True,'receipt':{'lifted_model_mm':40}})['passed'])

    def test_place_and_stack_need_program_completion_and_matching_terminal_evidence(self):
        for kind in ('place', 'stack'):
            key = kind + '_confirmed'
            primitive = step(kind)
            good = {'skill': kind, 'completed': True,
                    'receipt': {'completed': True, key: True, 'grasp_confirmed': True}}
            self.assertTrue(checkpoint(primitive, good)['passed'])
            for evidence in ({}, {'grasp_confirmed': True}, {'released': True},
                             {key: True}, {'completed': False, key: True},
                             {'completed': True, key: False}, {'completed': True, key: 1},
                             {'completed': True, ('place_confirmed' if kind == 'stack' else 'stack_confirmed'): True}):
                with self.subTest(kind=kind, evidence=evidence):
                    self.assertFalse(checkpoint(primitive, {**good, 'receipt': evidence})['passed'])
            definition = compile_graph(plan(primitive, step('wave')))
            self.assertEqual(definition['nodes'][0]['postcondition'], 'stack_verified' if kind == 'stack' else 'placement_verified')
            failed = replay(plan(primitive, step('wave')), [{'node': 'step_0',
                'receipt': {**good, 'receipt': {'completed': True, key: False, 'grasp_confirmed': True}}}])
            self.assertEqual(failed['active_node'], 'blocked')
            self.assertFalse(failed['completed'])
            with self.assertRaises(ValueError):
                replay(plan(primitive, step('wave')), [{'node': 'step_0',
                    'receipt': {**good, 'receipt': {'completed': True, key: False}}},
                    {'node': 'step_1', 'receipt': {'skill': 'wave', 'completed': True,
                                                'receipt': {'wave_completed': True}}}])

    def test_release_variants_match_actual_controller_receipts(self):
        for receipt in ({'already_empty':True},
                        {'receipt':{'completed':True,'restored_cap':500,'neutral_goal':1860}},
                        {'receipt':{'completed':True,'released':True,'contact_possible':False,'restored_cap':500,'neutral_goal':1860}}):
            self.assertTrue(checkpoint(step('release'),{'skill':'release','completed':True,**receipt})['passed'])

    def test_replay_rejects_gaps_duplicates_extra_and_out_of_order(self):
        p=plan(step(),step())
        bad=[[],[{'node':'step_1','receipt':observation()}],
             [{'node':'step_0','receipt':observation()}],
             [{'node':'step_0','receipt':observation()},{'node':'step_0','receipt':observation()}],
             [{'node':'step_0','receipt':observation(),'script':'forbidden'}]]
        for records in bad:
            with self.assertRaises(ValueError):replay(p,records)

    def test_error_and_cancellation_are_terminal_without_implicit_retries(self):
        p=plan(step(),step('wave'))
        for cancelled in (False,True):
            record={'node':'step_0','error':'camera missing','cancelled':cancelled}
            value=replay(p,[record]);self.assertFalse(value['completed'])
            self.assertEqual(value['active_node'],'cancelled' if cancelled else 'blocked')
            with self.assertRaises(ValueError):replay(p,[record,{'node':'step_1','receipt':observation()}])

    def test_replay_binding_cannot_overwrite_source_visibility(self):
        result=replay(plan(step('observe',target='cup'),step('grasp')),[
            {'node':'step_0','receipt':observation(False)},
            {'node':'step_1','receipt':{'skill':'grasp','completed':True,'receipt':{'completed':True,'grasp_confirmed':True}},
             'dependency':{'source_node':'step_0','current':True,'visible':True}}])
        self.assertFalse(result['completed']);self.assertEqual(result['active_node'],'blocked')

    def test_malformed_replay_records_and_jaw_receipts_are_rejected(self):
        for record in ({'node':'step_0'}, {'node':'step_0','receipt':observation(),'cancelled':True},
                       {'node':'step_0','error':{}}, {'node':'step_0','error':'x','dependency':{}}):
            with self.assertRaises(ValueError):replay(plan(step()),[record])
        for value in ('wrong',[1],12,{'completed':True,'restored_cap':-500,'neutral_goal':1e99},
                      {'completed':True,'restored_cap':500,'neutral_goal':True},
                      {'completed':True,'restored_cap':500,'neutral_goal':1860,'contact_possible':True}):
            self.assertFalse(checkpoint(step('release'),{'skill':'release','completed':True,'receipt':value})['passed'])

    def test_unsupported_plan_is_explicit_terminal(self):
        value=replay({'status':'unsupported','summary':'unsupported','steps':[],'unsupported_reason':'stack unavailable'},[])
        self.assertEqual(value['active_node'],'unsupported');self.assertFalse(value['completed'])


if __name__=='__main__':unittest.main()
