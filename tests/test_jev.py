"""Offline provider contract: no real credentials, network, vision or motors."""
from copy import deepcopy
import json
import os
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from server.jev import ABSTAIN, ENDPOINT, MAX_RESPONSE_BYTES, JevPolicy, _NoRedirect, _post


class JevTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-only-fake-key'}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.state = {'goal': 'Make room to wave', 'observed': {'gripper_empty': True, 'obstruction': 'cylinder'}}
        self.candidates = {'left': 'Take the already cleared left step', 'backward': 'Take the already cleared backward step'}
        self.response = {'model': 'jev-1.13.0', 'answers': {'next_skill': {
            'type': 'choice', 'choice': 'backward', 'confidence': .8,
            'probabilities': {'left': .05, 'backward': .9, ABSTAIN: .05}}},
            'usage': {'input_tokens': 100, 'output_tokens': 20}}
        self.transport = Mock(side_effect=lambda *args: (200, json.dumps(self.response).encode()))
        self.policy = JevPolicy(transport=self.transport)

    def choose(self, **kwargs):
        return self.policy.choose(self.state, self.candidates, **kwargs)

    def test_official_wire_contract_returns_only_supplied_skill(self):
        result = self.choose()
        self.assertEqual((result['status'], result['selected'], result['reason']), ('selected', 'backward', 'selected'))
        request, timeout = self.transport.call_args.args
        self.assertEqual(request.full_url, ENDPOINT); self.assertEqual(request.method, 'POST')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-only-fake-key')
        self.assertEqual(timeout, 3.)
        data = json.loads(request.data)
        self.assertEqual(set(data), {'model', 'state', 'questions'})
        self.assertEqual(data['state'], self.state)
        self.assertEqual(set(data['questions']), {'next_skill'})
        self.assertEqual(set(data['questions']['next_skill']['criteria']), {*self.candidates, ABSTAIN})
        self.assertEqual(result['usage'], self.response['usage'])
        self.assertEqual(result['model'], 'jev-1.13.0')
        self.assertNotIn('test-only-fake-key', json.dumps(result) + json.dumps(self.policy.status()))
        self.assertFalse(self.policy.status()['vision'])

    def test_missing_key_and_empty_candidates_make_no_request(self):
        os.environ.clear()
        self.assertFalse(self.policy.status()['available'])
        self.assertEqual(self.choose()['reason'], 'missing_key')
        os.environ['JEV_API_KEY'] = 'alias-only-fake-key'
        self.assertEqual(self.policy.choose(self.state, {})['reason'], 'invalid_input')
        self.transport.assert_not_called()

    def test_provider_access_denied_is_explicit_without_body_disclosure_or_retry(self):
        error=HTTPError(ENDPOINT,403,'test-only-fake-key sensitive upstream text',{},None)
        self.transport.side_effect=error
        result=self.choose()
        self.assertEqual(result['status'],'abstain')
        self.assertEqual(result['reason'],'provider_access_denied')
        self.assertEqual(self.policy.status()['error'],'provider_access_denied')
        self.transport.assert_called_once()
        self.assertNotIn('sensitive',json.dumps(result)+json.dumps(self.policy.status()))
        self.assertNotIn('test-only-fake-key',json.dumps(result)+json.dumps(self.policy.status()))
        self.transport.reset_mock();self.transport.side_effect=None
        self.transport.return_value=(403,b'private upstream body')
        self.assertEqual(self.choose()['reason'],'provider_access_denied')
        self.transport.assert_called_once()

    def test_documented_key_precedence_and_alias(self):
        os.environ['JEV_API_KEY'] = 'alias-only-fake-key'
        self.choose()
        self.assertEqual(self.transport.call_args.args[0].get_header('Authorization'), 'Bearer test-only-fake-key')
        del os.environ['TYPESAFE_API_KEY']
        self.choose()
        self.assertEqual(self.transport.call_args.args[0].get_header('Authorization'), 'Bearer alias-only-fake-key')

    def test_semantic_input_excludes_media_code_motor_fields_and_secret(self):
        invalid = [dict(self.state, frames=['image.jpg']), dict(self.state, joint_targets={'elbow': 2000}),
                   dict(self.state, code='run()'), {'nested': {'image_url': 'https://example.invalid/image.jpg'}},
                   {'value': float('nan')}, {'image_description': b'pixels'}, {'x': 'data:image/png;base64,x'},
                   {'x': 'test-only-fake-key'}, {'many': list(range(65))}, {'x': 'x' * 2049}]
        for state in invalid:
            with self.subTest(state_type=list(state)):
                self.assertEqual(self.policy.choose(state, self.candidates)['reason'], 'invalid_input')
        self.transport.assert_not_called()

    def test_invalid_candidate_set_rejected_before_network(self):
        for candidates in ({ABSTAIN: 'reserved'}, {'../../execute': 'bad'}, {'left': {'code': 'run()'}},
                           {f'a{i}': 'description' for i in range(33)}):
            with self.subTest(candidates=list(candidates)):
                self.assertEqual(self.policy.choose(self.state, candidates)['reason'], 'invalid_input')
        self.transport.assert_not_called()

    def test_unknown_choice_extra_answer_missing_options_and_non_argmax_abstain(self):
        original = deepcopy(self.response)
        changes = [lambda a: a.update(choice='close_gripper'),
                   lambda a: a['probabilities'].pop('left'),
                   lambda a: a['probabilities'].update(unlisted=0.),
                   lambda a: a.update(choice='left'),
                   lambda a: a.update(motor_command=1800),
                   lambda a: a.update(type='score')]
        for change in changes:
            self.response = deepcopy(original); change(self.response['answers']['next_skill'])
            with self.subTest(change=change):
                result = self.choose()
                self.assertEqual(result['reason'], 'invalid_response'); self.assertIsNone(result['selected'])
        self.response = deepcopy(original); self.response['answers']['extra'] = {}
        self.assertEqual(self.choose()['reason'], 'invalid_response')

    def test_invalid_probabilities_confidence_and_token_usage_are_not_coerced(self):
        original = deepcopy(self.response)
        for value in (float('nan'), float('inf'), -0.1, 1.1, True, '0.9'):
            for field in ('confidence', 'probability'):
                self.response = deepcopy(original)
                answer = self.response['answers']['next_skill']
                if field == 'confidence': answer['confidence'] = value
                else: answer['probabilities']['backward'] = value
                with self.subTest(value=value, field=field):
                    self.assertEqual(self.choose()['reason'], 'invalid_response')
        self.response = deepcopy(original); self.response['answers']['next_skill']['probabilities']['backward'] = .8
        self.assertEqual(self.choose()['reason'], 'invalid_response')
        self.response = deepcopy(original); self.response['usage']['input_tokens'] = True
        self.assertEqual(self.choose()['reason'], 'invalid_response')

    def test_low_confidence_and_explicit_abstention_preserve_evidence_without_choice(self):
        self.response['answers']['next_skill']['confidence'] = .49
        result = self.choose()
        self.assertEqual(result['reason'], 'low_confidence'); self.assertIsNone(result['selected'])
        self.assertEqual(result['confidence'], .49); self.assertTrue(result['probabilities'])
        self.response['answers']['next_skill'].update(choice=ABSTAIN, confidence=.9,
                                                    probabilities={'left': .02, 'backward': .03, ABSTAIN: .95})
        result = self.choose()
        self.assertEqual(result['reason'], 'model_abstained'); self.assertIsNone(result['selected'])

    def test_invalid_json_duplicate_keys_and_oversized_response_abstain(self):
        for body in (b'not JSON', b'\xff', b'{}', b'{"model":"jev-latest","model":"jev-1.13.0"}',
                     b'x' * (MAX_RESPONSE_BYTES + 1)):
            self.transport.side_effect = None; self.transport.return_value = (200, body)
            self.assertEqual(self.choose()['reason'], 'invalid_response')

    def test_http_transport_and_timeout_failures_are_sanitized_without_retries(self):
        for error, reason in ((HTTPError(ENDPOINT, 401, 'test-only-fake-key', {}, None), 'http_error'),
                              (URLError('test-only-fake-key'), 'transport_error'),
                              (TimeoutError('test-only-fake-key'), 'timeout')):
            self.transport.reset_mock(); self.transport.side_effect = error
            result = self.choose()
            self.assertEqual(result['reason'], reason); self.transport.assert_called_once()
            self.assertNotIn('test-only-fake-key', json.dumps(result) + json.dumps(self.policy.status()))
        self.transport.side_effect = None; self.transport.return_value = (429, b'test-only-fake-key')
        self.assertEqual(self.choose()['reason'], 'http_error')

    def test_wall_deadline_discards_late_success_and_bounds_inflight_requests(self):
        release = threading.Event(); entered = threading.Event()
        def slow(*args):
            entered.set(); release.wait(1.)
            return 200, json.dumps(self.response).encode()
        policy = JevPolicy(timeout_s=.03, transport=Mock(side_effect=slow))
        try:
            start = time.monotonic(); result = policy.choose(self.state, self.candidates)
            self.assertLess(time.monotonic() - start, .3)
            self.assertTrue(entered.is_set()); self.assertEqual(result['reason'], 'timeout')
            self.assertEqual(policy.choose(self.state, self.candidates)['reason'], 'busy')
            policy._transport.assert_called_once()
        finally:
            release.set()
        self.assertIsNone(result['selected'])

    def test_cancellation_before_and_during_inference_discards_response(self):
        cancel = threading.Event(); cancel.set()
        self.assertEqual(self.choose(cancel=cancel)['reason'], 'cancelled')
        self.transport.assert_not_called(); cancel.clear()
        def cancelling(*args):
            cancel.set()
            return 200, json.dumps(self.response).encode()
        self.transport.side_effect = cancelling
        self.assertEqual(self.choose(cancel=cancel)['reason'], 'cancelled')

    def test_default_transport_caps_read_and_disallows_redirects(self):
        response = Mock(); response.status = 200; response.read.return_value = b'{}'
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        opener = Mock(); opener.open.return_value = response
        with patch('server.jev.build_opener', return_value=opener) as build:
            self.assertEqual(_post('request', .25), (200, b'{}'))
        self.assertIsInstance(build.call_args.args[0], _NoRedirect)
        self.assertIsNone(build.call_args.args[0].redirect_request(None, None, 302, '', {}, 'https://other.invalid'))
        opener.open.assert_called_once_with('request', timeout=.25)
        response.read.assert_called_once_with(MAX_RESPONSE_BYTES + 1)


if __name__ == '__main__':
    unittest.main()
