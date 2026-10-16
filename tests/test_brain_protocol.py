"""Offline regression tests for server.brain.Brain's subprocess protocol.

Run from the repository with its Python environment:
    PYTHONPATH="$PWD" /path/to/python /private/tmp/brain-regression-tests.py -v

These tests replace every Brain Popen call with a tiny local Python child. They
never invoke Codex, load a camera, import the robot driver, or access hardware.
The fake hanging child exits on its own after three seconds so a regression
cannot leave the test suite hung indefinitely.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from server.brain import Brain, BrainTransportError
from server.search_intent import SearchIntent

_REAL_POPEN = subprocess.Popen


class BrainSubprocessRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='astra-brain-regression-')
        self.root = Path(self.temporary.name)
        self.brain = Brain({'model': 'gpt-6-astra', 'reasoning_effort': 'high'}, self.root)
        # The replacement child never reads this path; no image is captured.
        self.frames = [{'source': 'wrist', 'path': str(self.root / 'unused-frame.png')}]
        self.children = []
        self.seen_invocations = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=2)
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        self.temporary.cleanup()

    def fake_popen(self, source):
        def launch(codex_args, **kwargs):
            self.seen_invocations.append(list(codex_args))
            # Preserve the real pipe/new-session settings under test, but replace
            # the executable and arguments completely before launching anything.
            child = _REAL_POPEN([sys.executable, '-u', '-c', source], **kwargs)
            self.children.append(child)
            return child
        return launch

    def test_final_jsonl_without_trailing_newline_is_accepted(self):
        expected = {
            'query': '桌上物件', 'summary': '眼前沒有合適物件。', 'objects': [],
            'selected_index': -1, 'empty': True, 'uncertainty': '',
        }
        events = [
            {'type': 'item.completed', 'item': {
                'type': 'agent_message', 'text': json.dumps(expected, ensure_ascii=False),
            }},
            {'type': 'turn.completed'},
        ]
        payload = '\n'.join(json.dumps(event, ensure_ascii=False) for event in events)
        self.assertFalse(payload.endswith('\n'))
        source = 'import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write(' + repr(payload.encode()) + '); sys.stdout.buffer.flush()'
        with patch('server.brain.subprocess.Popen', self.fake_popen(source)):
            actual = self.brain.perceive('桌上物件', self.frames)
        self.assertEqual(actual, expected)
        self.assertTrue(self.brain.verified)
        self.assertEqual(self.children[-1].returncode, 0)
        self.assertEqual(len(self.seen_invocations), 1)

    def test_partial_stdout_and_stderr_remain_cancellable(self):
        for stream_name in ('stdout', 'stderr'):
            with self.subTest(stream=stream_name):
                marker = self.root / (stream_name + '.ready')
                cancel = threading.Event()
                cancel_times = []
                source = (
                    'import pathlib,sys,time; sys.stdin.buffer.read(); '
                    f'sys.{stream_name}.buffer.write(b"partial line without newline"); '
                    f'sys.{stream_name}.buffer.flush(); '
                    f'pathlib.Path({str(marker)!r}).write_text("ready"); '
                    'time.sleep(3)'
                )

                def cancel_after_partial_line():
                    deadline = time.monotonic() + 2
                    while not marker.exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    # Give the reader time to consume readability; a blocking
                    # readline would now be waiting for a newline that never comes.
                    time.sleep(.10)
                    cancel_times.append(time.monotonic())
                    cancel.set()

                canceller = threading.Thread(target=cancel_after_partial_line, daemon=True)
                canceller.start()
                try:
                    with patch('server.brain.subprocess.Popen', self.fake_popen(source)):
                        with self.assertRaisesRegex(RuntimeError, '搜尋已停止'):
                            self.brain.perceive('cancel test', self.frames, cancel=cancel)
                    completed_at = time.monotonic()
                finally:
                    canceller.join(timeout=3)
                self.assertTrue(marker.exists(), 'Fake child did not emit its partial line')
                self.assertEqual(len(cancel_times), 1)
                self.assertLess(completed_at - cancel_times[0], 1.0,
                                'Cancellation blocked waiting for a newline')
                self.assertIsNotNone(self.children[-1].poll(), 'Cancelled child was not reaped')

    def test_text_intent_uses_same_tool_disabled_transport_without_any_image(self):
        expected={'intent':'action','title':'試著揮手','summary':'點選後再判斷現場並執行。',
                  'label':'揮揮手','availability':'experimental','reason':''}
        events=[{'type':'item.completed','item':{'type':'agent_message','text':json.dumps(expected)}},
                {'type':'turn.completed'}]
        payload='\n'.join(json.dumps(event) for event in events)
        source='import sys; sys.stdin.buffer.read(); sys.stdout.buffer.write('+repr(payload.encode())+'); sys.stdout.buffer.flush()'
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            actual=self.brain._infer_text('Classify this text only.',SearchIntent)
        self.assertEqual(actual.model_dump(),expected)
        args=self.seen_invocations[0]
        self.assertNotIn('--image',args)
        self.assertIn('gpt-6-astra',args)
        for tool in ('shell_tool','apps','plugins','computer_use','browser_use','multi_agent','hooks'):
            index=args.index(tool)
            self.assertEqual(args[index-1],'--disable')

    def test_text_path_does_not_relax_existing_wrist_only_vision_contract(self):
        with patch('server.brain.subprocess.Popen') as launch:
            for frames in ([],[{'source':'observer','path':'unused.png'}],
                           [{'source':'wrist','path':'one.png'},{'source':'d435','path':'two.png'}]):
                with self.subTest(frames=frames),self.assertRaises(ValueError):
                    self.brain._infer('vision',frames,SearchIntent)
            launch.assert_not_called()

    def test_cancelled_text_intent_never_starts_subprocess(self):
        cancel=threading.Event();cancel.set()
        with patch('server.brain.subprocess.Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'搜尋已停止'):
                self.brain._infer_text('intent',SearchIntent,cancel=cancel)
            launch.assert_not_called()

    def test_dns_retry_has_short_connection_budget_and_sanitized_receipt(self):
        self.brain.config['connection_failure_timeout_seconds']=.15
        secret='apikey_DO_NOT_RECORD_THIS_or_user_prompt'
        event={'type':'error','message':'Reconnecting... failed to lookup address information '+secret}
        source=('import sys,time; sys.stdin.buffer.read(); '+
                '\nfor _ in range(4):\n print('+repr(json.dumps(event))+',flush=True); time.sleep(.08)\n'+
                'time.sleep(3)')
        started=time.monotonic()
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            with self.assertRaises(BrainTransportError) as caught:
                self.brain._infer_text('private target '+secret,SearchIntent)
        self.assertEqual(caught.exception.kind,'network')
        self.assertIn('連不上',str(caught.exception))
        self.assertLess(time.monotonic()-started,1.2)
        self.assertIsNotNone(self.children[-1].poll())
        status=self.brain.status()
        self.assertEqual(status['phase'],'idle')
        self.assertIsNone(status['active'])
        self.assertEqual(status['error_kind'],'network')
        self.assertGreater(status['last_call']['network_events'],0)
        self.assertFalse(status['last_call']['ok'])
        for path in ('last-failure.json','inferences.jsonl'):
            text=(self.brain.directory/path).read_text()
            self.assertNotIn(secret,text)
            self.assertNotIn('private target',text)
            self.assertNotIn('failed to lookup',text)

    def test_transient_network_retry_recovers_and_clears_network_deadline(self):
        self.brain.config['connection_failure_timeout_seconds']=.15
        expected={'intent':'action','title':'揮手','summary':'試著揮手。','label':'揮手',
                  'availability':'experimental','reason':''}
        events=[{'type':'error','message':'Reconnecting... waiting for network'},
                {'type':'item.started','item':{'type':'reasoning'}},
                {'type':'item.completed','item':{'type':'agent_message','text':json.dumps(expected)}},
                {'type':'turn.completed'}]
        source=('import sys,time; sys.stdin.buffer.read(); '
                'print('+repr(json.dumps(events[0]))+',flush=True); '
                'print('+repr(json.dumps(events[1]))+',flush=True); time.sleep(.3); '
                'print('+repr(json.dumps(events[2]))+',flush=True); '
                'print('+repr(json.dumps(events[3]))+',flush=True)')
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            self.assertEqual(self.brain._infer_text('intent',SearchIntent).model_dump(),expected)
        status=self.brain.status()
        self.assertEqual(status['error_kind'],'')
        self.assertTrue(status['last_call']['ok'])
        self.assertIsNotNone(status['last_call']['first_model_output_seconds'])
        self.assertGreaterEqual(status['last_call']['elapsed'],.3)

    def test_transient_retry_then_silent_reasoning_outlives_connection_budget(self):
        self.brain.config['connection_failure_timeout_seconds']=.1
        self.brain.config['inference_timeout_seconds']=2
        expected={'intent':'action','title':'揮手','summary':'試著揮手。','label':'揮手',
                  'availability':'experimental','reason':''}
        retry={'type':'error','message':'Reconnecting... waiting for network'}
        answer={'type':'item.completed','item':{'type':'agent_message','text':json.dumps(expected)}}
        # There is deliberately no model/recovery event during the quiet wait.
        source=('import sys,time; sys.stdin.buffer.read(); '
                'print('+repr(json.dumps(retry))+',flush=True); time.sleep(.3); '
                'print('+repr(json.dumps(answer))+',flush=True); '
                'print(\'{"type":"turn.completed"}\',flush=True)')
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            self.assertEqual(self.brain._infer_text('intent',SearchIntent).model_dump(),expected)
        self.assertTrue(self.brain.last_call['ok'])
        self.assertGreaterEqual(self.brain.last_call['elapsed'],.3)

    def test_single_stale_network_event_does_not_relabel_inference_timeout(self):
        self.brain.config['connection_failure_timeout_seconds']=.1
        self.brain.config['inference_timeout_seconds']=.3
        event={'type':'error','message':'Reconnecting... waiting for network'}
        source='import sys,time; sys.stdin.buffer.read(); print('+repr(json.dumps(event))+',flush=True); time.sleep(3)'
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            with self.assertRaises(BrainTransportError) as caught:
                self.brain._infer_text('intent',SearchIntent)
        self.assertEqual(caught.exception.kind,'timeout')
        self.assertGreaterEqual(self.brain.last_call['elapsed'],.3)

    def test_terminal_failure_authentication_and_rate_limits_do_not_wait(self):
        cases=[('turn.failed',{'message':'unrecognized service failure'},'inference'),
               ('turn.failed',{'message':'Connection failed'},'network'),
               ('error',{'message':'401 Unauthorized private-token'},'authentication'),
               ('error',{'message':'usage limit reached private-token'},'rate_limit')]
        for event_type,error,expected_kind in cases:
            with self.subTest(kind=expected_kind):
                event={'type':event_type,'error':error}
                source='import sys,time; sys.stdin.buffer.read(); print('+repr(json.dumps(event))+',flush=True); time.sleep(3)'
                started=time.monotonic()
                with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
                    with self.assertRaises(BrainTransportError) as caught:
                        self.brain._infer_text('intent',SearchIntent)
                self.assertEqual(caught.exception.kind,expected_kind)
                self.assertLess(time.monotonic()-started,1.0)
                self.assertNotIn('private-token',str(caught.exception))
                self.assertIsNotNone(self.children[-1].poll())

    def test_true_inference_timeout_is_distinct_from_network_failure(self):
        self.brain.config['inference_timeout_seconds']=.15
        source='import sys,time; sys.stdin.buffer.read(); print(\'{"type":"turn.started"}\',flush=True); time.sleep(3)'
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            with self.assertRaises(BrainTransportError) as caught:
                self.brain._infer_text('intent',SearchIntent)
        self.assertEqual(caught.exception.kind,'timeout')
        self.assertEqual(self.brain.status()['last_call']['network_events'],0)
        self.assertIsNotNone(self.children[-1].poll())

    def test_invalid_output_schema_is_immediate_and_sanitized(self):
        # Match the Codex event's nested, JSON-encoded API error, not a model
        # reply. This local child exercises the actual subprocess parser.
        detail={'type':'invalid_request_error','code':'invalid_json_schema',
                'message':"Invalid schema for response_format 'private-target': Missing 'support_boundary_normalized'.",
                'param':'text.format.schema'}
        event={'type':'error','message':json.dumps({'type':'error','error':detail,'status':400})}
        source='import sys,time; sys.stdin.buffer.read(); print('+repr(json.dumps(event))+',flush=True); time.sleep(3)'
        started=time.monotonic()
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            with self.assertRaises(BrainTransportError) as caught:
                self.brain._infer_text('private prompt',SearchIntent)
        self.assertLess(time.monotonic()-started,1.)
        self.assertEqual(caught.exception.kind,'invalid_schema')
        self.assertNotIn('重試',str(caught.exception))
        self.assertEqual(self.brain.last_call['network_events'],0)
        self.assertIsNone(self.brain.last_call['first_model_output_seconds'])
        self.assertIsNotNone(self.children[-1].poll())
        for name in ('last-failure.json','inferences.jsonl'):
            saved=(self.brain.directory/name).read_text()
            self.assertIn('invalid_schema',saved)
            for private in ('private-target','private prompt','support_boundary_normalized'):
                self.assertNotIn(private,saved)

    def test_large_unread_prompt_remains_cancellable(self):
        cancel=threading.Event();marker=self.root/'stdin.ready'
        source='import pathlib,time; pathlib.Path('+repr(str(marker))+').write_text("ready"); time.sleep(3)'
        cancel_times=[]
        def interrupt():
            deadline=time.monotonic()+2
            while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
            cancel_times.append(time.monotonic());cancel.set()
        thread=threading.Thread(target=interrupt,daemon=True);thread.start()
        try:
            with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
                with self.assertRaises(BrainTransportError) as caught:
                    self.brain._infer_text('x'*2_000_000,SearchIntent,cancel=cancel)
            self.assertEqual(caught.exception.kind,'cancelled')
            self.assertLess(time.monotonic()-cancel_times[0],1.0)
            self.assertIsNotNone(self.children[-1].poll())
        finally:thread.join(timeout=3)

    def test_malformed_model_reply_has_safe_typed_error(self):
        event={'type':'item.completed','item':{'type':'agent_message','text':'private-invalid-answer'}}
        source='import sys; sys.stdin.buffer.read(); print('+repr(json.dumps(event))+'); print(\'{"type":"turn.completed"}\')'
        with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
            with self.assertRaises(BrainTransportError) as caught:
                self.brain._infer_text('intent',SearchIntent)
        self.assertEqual(caught.exception.kind,'invalid_response')
        self.assertNotIn('private-invalid-answer',str(caught.exception))
        self.assertNotIn('private-invalid-answer',(self.brain.directory/'last-failure.json').read_text())

    def test_cli_start_failure_is_logged_without_exposing_exception_text(self):
        with patch('server.brain.subprocess.Popen',side_effect=OSError('private executable details')):
            with self.assertRaises(BrainTransportError) as caught:
                self.brain._infer_text('intent',SearchIntent)
        self.assertEqual(caught.exception.kind,'transport')
        self.assertNotIn('private executable details',str(caught.exception))
        self.assertIsNone(self.brain.status()['active'])

    def test_active_status_identifies_reconnection_without_private_input(self):
        cancel=threading.Event();seen=[]
        event={'type':'error','message':'Reconnecting... waiting for network'}
        source='import sys,time; sys.stdin.buffer.read(); print('+repr(json.dumps(event))+',flush=True); time.sleep(3)'
        def observe():
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                status=self.brain.status()
                if status['phase']=='reconnecting':
                    seen.append(status);cancel.set();return
                time.sleep(.01)
            cancel.set()
        watcher=threading.Thread(target=observe,daemon=True);watcher.start()
        try:
            with patch('server.brain.subprocess.Popen',self.fake_popen(source)):
                with self.assertRaises(BrainTransportError):
                    self.brain._infer_text('private input',SearchIntent,cancel=cancel)
        finally:watcher.join(timeout=3)
        self.assertEqual(len(seen),1)
        self.assertEqual(seen[0]['active']['schema'],'SearchIntent')
        self.assertGreaterEqual(seen[0]['active']['elapsed'],0)
        self.assertNotIn('started_monotonic',seen[0]['active'])
        self.assertNotIn('private input',json.dumps(seen[0]))


if __name__ == '__main__':
    unittest.main()
