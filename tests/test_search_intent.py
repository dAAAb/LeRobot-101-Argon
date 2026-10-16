"""Semantic text routing and camera-free suggestion jobs; all providers faked."""
from copy import deepcopy
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from pydantic import ValidationError

from server.gripper_power import SessionCancelled
from server.search_intent import SearchIntent, action_suggestion, classify_search_intent
import test_object_actions as fixtures


def intent(kind='action', **changes):
    return {'intent':kind,'title':'建議的動作','summary':'點選後再看現場並嘗試。',
            'label':'揮揮手' if kind!='search' else '',
            'availability':'experimental' if kind=='action' else 'none', 'reason':'', **changes}


class SearchIntentTests(unittest.TestCase):
    def setUp(self):
        self.brain=SimpleNamespace(_infer_text=Mock(return_value=SearchIntent.model_validate(intent())))
        self.policy=SimpleNamespace(choose=Mock(return_value={'status':'abstain'}))

    def test_semantic_provider_is_used_instead_of_keyword_matching(self):
        for query,selected in (('揮手','action_experimental'),('手套','search'),
                               ('不要揮手，找杯子','search'),('請跟我打招呼','action_experimental'),
                               ('把杯子疊上去','unsupported_action')):
            self.policy.choose.return_value={'status':'selected','selected':selected}
            result=classify_search_intent(self.brain,query,tactician=self.policy)
            self.assertEqual(result['intent'],'search' if selected=='search' else 'action')
            state,candidates=self.policy.choose.call_args.args
            self.assertEqual(state['query'],query)
            self.assertEqual(set(candidates),{'search','action_available','action_experimental','unsupported_action','ambiguous'})
        self.brain._infer_text.assert_not_called()

    def test_jev_abstain_or_unknown_selection_falls_back_to_astra_text(self):
        for decision in ({'status':'abstain'}, {'status':'selected','selected':'unknown'}):
            self.policy.choose.return_value=decision
            result=classify_search_intent(self.brain,'揮手',tactician=self.policy)
            self.assertEqual(result['intent'],'action')
        self.assertEqual(self.brain._infer_text.call_count,2)
        prompt,schema=self.brain._infer_text.call_args.args
        self.assertIs(schema,SearchIntent)
        self.assertEqual(json.loads(prompt.split('INPUT_DATA_JSON\n')[1])['query'],'揮手')
        self.assertIn('NO camera image',prompt)
        self.assertIn('Interpret the whole phrase',prompt)

    def test_suggestion_preserves_original_instruction_and_never_contains_motor_plan(self):
        query='  先抬高一公分，再揮揮手  '
        result=action_suggestion(query,intent(label='A model-written label'))
        self.assertEqual(result['instruction'],query.strip())
        self.assertEqual(result['query'],query.strip())
        self.assertEqual(set(result),{'kind','query','instruction','title','summary','label','availability','reason'})
        self.assertIsNone(action_suggestion('杯子',intent('search')))

    def test_place_stack_suggestion_keeps_full_instruction_with_experimental_catalogue(self):
        for query in ('把白色圓筒放到藍色盒子左邊', '把紅色積木堆在黃色積木上面'):
            self.policy.choose.return_value = {'status': 'selected', 'selected': 'action_experimental'}
            value = classify_search_intent(self.brain, query, tactician=self.policy)
            suggestion = action_suggestion(query, value)
            self.assertEqual(suggestion['instruction'], query)
            self.assertEqual(suggestion['availability'], 'experimental')
            state, choices = self.policy.choose.call_args.args
            experimental = {item['skill'] for item in state['capabilities']['experimental']}
            self.assertTrue({'place', 'stack'} <= experimental)
            self.assertIn('place', choices['action_experimental'])
            self.assertIn('stack', choices['action_experimental'])
            self.assertNotIn('steps', suggestion)
        self.brain._infer_text.assert_not_called()

    def test_ambiguity_has_no_executable_instruction(self):
        self.policy.choose.return_value={'status':'selected','selected':'ambiguous'}
        result=action_suggestion('弄一下',classify_search_intent(self.brain,'弄一下',tactician=self.policy))
        self.assertEqual(result['availability'],'clarify')
        self.assertEqual(result['instruction'],'')
        self.assertTrue(result['reason'])

    def test_capability_class_does_not_claim_physical_feasibility(self):
        for choice,status in (('action_available','available'),('action_experimental','experimental'),('unsupported_action','unsupported')):
            self.policy.choose.return_value={'status':'selected','selected':choice}
            result=classify_search_intent(self.brain,'請做這個動作',tactician=self.policy)
            self.assertEqual(result['availability'],status)
        self.assertFalse(self.policy.choose.call_args.args[0]['capabilities']['learned_motor_policy'])

    def test_strict_route_contract_and_no_hidden_execution_fields(self):
        for value in (intent('search',availability='available'), intent(availability='none'),
                      intent(availability='unsupported'), intent(label=' '),
                      intent('ambiguous',availability='clarify'), intent(title=' '),
                      intent(summary='x'*301), intent(steps=[]), intent(joint_targets={})):
            with self.subTest(value=value),self.assertRaises(ValidationError):SearchIntent.model_validate(value)

    def test_blank_overlong_and_nontext_input_never_calls_provider(self):
        for query in ('','  ','x'*501,None,1):
            with self.subTest(query=query),self.assertRaises(ValueError):
                classify_search_intent(self.brain,query,tactician=self.policy)
        self.policy.choose.assert_not_called();self.brain._infer_text.assert_not_called()

    def test_cancellation_before_or_during_each_provider_never_publishes_suggestion(self):
        cancel=threading.Event();cancel.set()
        with self.assertRaises(SessionCancelled):classify_search_intent(self.brain,'揮手',cancel,tactician=self.policy)
        self.policy.choose.assert_not_called()
        cancel.clear()
        def choose(*args,**kwargs):
            cancel.set();return {'status':'selected','selected':'action_experimental'}
        self.policy.choose.side_effect=choose
        with self.assertRaises(SessionCancelled):classify_search_intent(self.brain,'揮手',cancel,tactician=self.policy)
        self.brain._infer_text.assert_not_called()
        cancel.clear()
        def infer(*args,**kwargs):cancel.set();return intent()
        self.brain._infer_text.side_effect=infer
        with self.assertRaises(SessionCancelled):classify_search_intent(self.brain,'揮手',cancel)


class SearchSuggestionAPITests(unittest.TestCase):
    def setUp(self):
        fixtures.SemanticObjectActionTests.setUp(self)
        self.policy=self.m.tactician
        self.policy.choose.return_value={'status':'selected','selected':'action_experimental'}

    finish=fixtures.SemanticObjectActionTests.finish

    def search(self,**changes):
        return self.client.post('/api/search',headers=self.headers,json={'query':'揮手','mode':'search',**changes})

    def assert_no_camera_or_motion(self):
        self.camera.snapshot.assert_not_called()
        for name in ('run_program','planned_wave','observe_views'):getattr(self.grasp,name).assert_not_called()
        self.robot.scan.assert_not_called();self.robot.refinement_scan.assert_not_called()
        self.brain.perceive.assert_not_called()

    def test_stopped_holding_and_dark_camera_still_produce_suggestion_without_motor_access(self):
        self.robot.stop.set();self.robot.pwm_owner=object();self.camera.fresh.return_value=False
        done=self.finish(self.search())
        self.assertEqual(done['status'],'completed',done['message'])
        self.assertIsNone(done['result'])
        self.assertEqual(done['action_suggestion']['instruction'],'揮手')
        self.assertEqual(done['action_suggestion']['availability'],'experimental')
        self.assertEqual(done['stage'],'done')
        self.assertTrue(self.robot.stop.is_set())
        self.assertEqual(self.robot.revision,41)
        self.camera.fresh.assert_not_called()
        self.assert_no_camera_or_motion()

    def test_complete_placement_search_produces_suggestion_without_dispatching_program(self):
        instruction = '把白黃圓筒放到藍色盒子上面'
        done = self.finish(self.search(query=instruction))
        self.assertEqual(done['status'], 'completed', done['message'])
        self.assertEqual(done['action_suggestion']['instruction'], instruction)
        self.assertEqual(done['action_suggestion']['availability'], 'experimental')
        self.assertIsNone(done['result'])
        self.assert_no_camera_or_motion()

    def test_ambiguous_or_unsupported_result_is_not_an_executable_command(self):
        for choice,status in (('ambiguous','clarify'),('unsupported_action','unsupported')):
            self.policy.choose.return_value={'status':'selected','selected':choice}
            done=self.finish(self.search(query='把那個弄成一疊'))
            self.assertEqual(done['status'],'completed')
            self.assertEqual(done['action_suggestion']['availability'],status)
            self.assertTrue(done['action_suggestion']['reason'])
            self.assertIsNone(done['result'])
        self.assert_no_camera_or_motion()

    def test_noun_search_keeps_existing_observation_and_scene_flow(self):
        self.policy.choose.return_value={'status':'selected','selected':'search'}
        done=self.finish(self.search(query='白色圓筒'))
        self.assertEqual(done['status'],'completed',done['message'])
        self.assertNotIn('action_suggestion',done)
        self.assertIsNotNone(done['result'])
        self.grasp.observe_views.assert_called_once();self.brain.perceive.assert_called_once()
        self.grasp.planned_wave.assert_not_called();self.grasp.run_program.assert_not_called()

    def test_noun_search_still_checks_stop_and_holding_after_readonly_classification(self):
        self.policy.choose.return_value={'status':'selected','selected':'search'}
        for stopped,holding in ((True,None),(False,object())):
            self.robot.stop.clear()
            if stopped:self.robot.stop.set()
            self.robot.pwm_owner=holding
            done=self.finish(self.search(query='杯子'))
            self.assertEqual(done['status'],'failed')
            self.assertNotIn('action_suggestion',done)
        self.assert_no_camera_or_motion()

    def test_upload_inspect_point_and_empty_search_bypass_intent(self):
        for changes in ({'reference_id':'missing-reference'}, {'mode':'point','frame_id':'missing-frame','x':.2,'y':.3},
                        {'mode':'inspect'}, {'query':''}):
            done=self.finish(self.search(**changes))
            self.assertNotIn('action_suggestion',done)
        self.policy.choose.assert_not_called()
        self.brain._infer_text.assert_not_called()

    def test_lucky_keeps_its_explicit_action_behavior_without_intent_classification(self):
        self.search_scene.update(empty=True,selected_index=-1,objects=[])
        done=self.finish(self.search(mode='lucky'))
        self.assertEqual(done['status'],'completed',done['message'])
        self.grasp.planned_wave.assert_called_once()
        self.policy.choose.assert_not_called()
        self.assertNotIn('action_suggestion',done)

    def test_provider_failure_never_falls_through_to_motor_search(self):
        self.policy.choose.return_value={'status':'abstain'}
        self.brain._infer_text.side_effect=RuntimeError('文字理解暫時無法完成')
        done=self.finish(self.search())
        self.assertEqual(done['status'],'failed')
        self.assertIn('文字理解',done['message'])
        self.assert_no_camera_or_motion()

    def test_suggestion_query_still_requires_auth_and_exclusive_job(self):
        denied=self.client.post('/api/search',json={'query':'揮手'})
        self.assertEqual(denied.status_code,403)
        first=self.search();self.assertEqual(first.status_code,200)
        self.assertEqual(first.json()['stage'],'understanding')
        self.assertEqual(first.json()['message'],'正在理解你的搜尋')
        self.assertEqual(self.search().status_code,409)
        self.assertEqual(len(self.queue),1)
        self.policy.choose.assert_not_called()


if __name__=='__main__':unittest.main()
