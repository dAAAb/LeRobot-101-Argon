"""Brain grasp-advice contracts with in-memory fake process and local pipes.
No CLI, child process, network, robot import or camera is used.
"""
import json,os,tempfile,threading,unittest
from unittest.mock import patch
from server.brain import BrainTransportError, Brain

VALID={'step':'observe','direction':'none','distance_mm':0,'object_name':'合成測試物件',
       'target_visible':False,'gripper_empty':False,'target_between_fingers':False,
       'clear_path':False,'confidence':0.2,'bbox':None,'reason':'資訊不足','warning':''}

class CapturedInput:
    """Selectable pipe that also preserves the complete prompt for assertions."""
    def __init__(self):
        r,w=os.pipe();self.writer=os.fdopen(w,'wb',buffering=0)
        self.reader=os.fdopen(r,'rb',buffering=0);self.captured=b''
        self.worker=threading.Thread(target=self._read,daemon=True);self.worker.start()
    def _read(self):
        chunks=[]
        with self.reader:
            while chunk:=self.reader.read(65536):chunks.append(chunk)
        self.captured=b''.join(chunks)
    def fileno(self):return self.writer.fileno()
    @property
    def closed(self):return self.writer.closed
    def close(self):
        if not self.closed:self.writer.close()
        self.worker.join(timeout=2)
        if self.worker.is_alive():raise RuntimeError('Fake stdin reader did not finish')

class PipeProcess:
    def __init__(self,answer):
        self.stdin=CapturedInput();self.returncode=0;self.pid=-1
        r,w=os.pipe();self.stdout=os.fdopen(r,'rb',buffering=0)
        payload='\n'.join(json.dumps(e,ensure_ascii=False) for e in [
            {'type':'thread.started','thread_id':'mock'}, {'type':'turn.started'},
            {'type':'item.completed','item':{'type':'agent_message','text':json.dumps(answer,ensure_ascii=False)}},
            {'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':20}}])
        os.write(w,payload.encode());os.close(w)
        r,w=os.pipe();self.stderr=os.fdopen(r,'rb',buffering=0);os.close(w)
    def wait(self,timeout=None):return 0
    def poll(self):return 0
    def close(self):
        for stream in (self.stdin,self.stdout,self.stderr):stream.close()

class GraspAdviceContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.brain=Brain({'model':'gpt-6-astra','reasoning_effort':'high'},self.temp.name)
        self.frames=[{'source':'wrist','path':'unused-synthetic.png'}]
    def infer(self,answer,frames=None):
        def process(*args,**kwargs):
            p=PipeProcess(answer);self.addCleanup(p.close);return p
        with patch('server.brain.subprocess.Popen',side_effect=process) as popen:
            result=self.brain.grasp_advice('合成測試',self.frames if frames is None else frames,{'stage':'offline'})
            self.assertEqual(popen.call_count,1)
            args=popen.call_args.args[0]
            self.assertEqual(args[args.index('--model')+1],'gpt-6-astra')
            self.assertIn('model_reasoning_effort="high"',args)
            return result
    def test_observe_with_null_box_and_zero_motion_is_valid(self):
        self.assertEqual(self.infer(dict(VALID)),VALID)
    def test_move_at_one_and_five_mm_bounds_is_valid(self):
        for distance in [1,5]:
            with self.subTest(distance=distance):
                a={**VALID,'step':'move','direction':'left','distance_mm':distance}
                self.assertEqual(self.infer(a),a)
    def test_move_requires_non_none_direction(self):
        with self.assertRaisesRegex(ValueError,'bounded movement'):
            self.infer({**VALID,'step':'move','distance_mm':3})
    def test_move_below_one_mm_is_invalid(self):
        with self.assertRaisesRegex(ValueError,'bounded movement'):
            self.infer({**VALID,'step':'move','direction':'up','distance_mm':.5})
    def test_unknown_direction_is_schema_invalid(self):
        with self.assertRaisesRegex(BrainTransportError,'資料格式'):
            self.infer({**VALID,'step':'move','direction':'diagonal','distance_mm':3})
    def test_nonmove_cannot_include_direction_or_distance(self):
        for update in [{'direction':'left'},{'distance_mm':1}]:
            with self.subTest(update=update),self.assertRaisesRegex(ValueError,'Non-movement'):
                self.infer({**VALID,**update})
    def test_box_must_fit_current_frame(self):
        for box in [{'x':.9,'y':.1,'w':.2,'h':.2},{'x':.1,'y':.9,'w':.2,'h':.2}]:
            with self.subTest(box=box),self.assertRaisesRegex(ValueError,'current object box'):
                self.infer({**VALID,'bbox':box})
    def test_valid_edge_box_is_accepted(self):
        a={**VALID,'bbox':{'x':.5,'y':.5,'w':.5,'h':.5}}
        self.assertEqual(self.infer(a),a)
    def test_observer_empty_or_mixed_frames_never_launch_process(self):
        for frames in [[],[{'source':'observer','path':'never-read.png'}],self.frames+[{'source':'observer','path':'never-read.png'}]]:
            with self.subTest(frames=frames),patch('server.brain.subprocess.Popen') as popen:
                with self.assertRaisesRegex(ValueError,'only accepts wrist'):
                    self.brain.grasp_advice('test',frames,{'stage':'offline'})
                popen.assert_not_called()
    def test_missing_extra_and_wrongly_typed_fields_are_schema_invalid(self):
        missing=dict(VALID);del missing['clear_path']
        for answer in [missing,{**VALID,'motor_goal':1234},{**VALID,'confidence':'unknown'},
                       {**VALID,'distance_mm':6},{**VALID,'step':'execute'},
                       {**VALID,'bbox':{'x':0,'y':0,'w':0,'h':.1}}]:
            with self.subTest(answer=answer),self.assertRaisesRegex(BrainTransportError,'資料格式'):self.infer(answer)

if __name__=='__main__':unittest.main(verbosity=2)
