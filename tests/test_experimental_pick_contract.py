"""Experimental pick API gates and bounded wrist-history contracts.

TestClient is never entered as a lifespan context. No camera/serial/network
operation is allowed; each API gate must reject before submitting a job.
"""
import copy, importlib, tempfile, threading, time, types, unittest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from server.grasp import WristGrasp


def pick_action(**changes):
    return {'kind':'pick','label':'試著抓起來','reason':'synthetic test','target':'',
            'available':True,'unavailable_reason':'',**changes}


def object_record(name, safe, actions=None):
    return {'name':name,'description':'synthetic object','safe_to_touch':safe,
            'actions':copy.deepcopy(actions or [])}


class ExperimentalPickAPIContracts(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('server.app')
        self.client=TestClient(self.module.app)  # Deliberately no `with` / lifespan.
        self.addCleanup(self.client.close)
        self.headers={'x-lerobot-token':self.module.TOKEN}
        for owner,name in ((self.module.robot,'connect'),(self.module.camera,'start'),
                           (self.module.grasp,'run_program')):
            guard=patch.object(owner,name,side_effect=AssertionError('hardware/job call from offline gate test'))
            guard.start();self.addCleanup(guard.stop)
        scene_patch=patch.object(self.module,'scenes',{})
        scene_patch.start();self.addCleanup(scene_patch.stop)

    def test_disabled_flag_blocks_cached_available_pick_action_before_submission(self):
        m=self.module
        m.scenes['cached-enabled']={'id':'cached-enabled','created_at':time.time()-10,
            'objects':[object_record('synthetic tube',True,[pick_action()])]}
        with patch.dict(m.CONFIG,{'experimental_pick_enabled':False}),patch.object(m,'submit') as submit:
            response=self.client.post('/api/actions',headers=self.headers,
                json={'scene_id':'cached-enabled','object_index':0,'action_index':0})
        self.assertEqual(response.status_code,409)
        self.assertIn('未開啟',response.json()['detail'])
        submit.assert_not_called();m.grasp.run_program.assert_not_called()
        self.assertTrue(m.scenes['cached-enabled']['objects'][0]['actions'][0]['available'],
                        'Server flag must override even a cached available action')

    def test_disabled_flag_blocks_direct_grasp_attempt_before_submission(self):
        m=self.module
        with patch.dict(m.CONFIG,{'experimental_pick_enabled':False}),patch.object(m,'submit') as submit:
            response=self.client.post('/api/grasp-attempt',headers=self.headers,json={'query':'synthetic tube'})
        self.assertEqual(response.status_code,409)
        self.assertIn('未開啟',response.json()['detail'])
        submit.assert_not_called();m.grasp.run_program.assert_not_called()

    def enrich(self,objects,enabled):
        scene={'query':'synthetic','summary':'test','empty':False,'selected_index':0,
               'uncertainty':'','objects':objects}
        frames=[{'id':'synthetic-wrist','source':'wrist','captured_at':time.time(),
                 'camera_generation':'synthetic-generation','path':'never-read.jpg'}]
        with patch.dict(self.module.CONFIG,{'experimental_pick_enabled':enabled}):
            return self.module.enrich(scene,frames)

    def test_enabled_enrichment_adds_pick_only_to_safe_objects(self):
        result=self.enrich([object_record('safe',True),object_record('unsafe',False)],True)
        safe,unsafe=result['objects']
        picks=[a for a in safe['actions'] if a['kind']=='pick']
        self.assertEqual(len(picks),1);self.assertTrue(picks[0]['available'])
        self.assertEqual([a for a in unsafe['actions'] if a['kind']=='pick'],[])
        self.assertIs(unsafe['safe_to_touch'],False)
        self.assertTrue(any(a['kind']=='inspect' and a['available'] for a in unsafe['actions']))

    def test_enabled_enrichment_never_overrides_existing_unsafe_pick(self):
        result=self.enrich([object_record('unsafe',False,[pick_action()])],True)
        unsafe=result['objects'][0];picks=[a for a in unsafe['actions'] if a['kind']=='pick']
        self.assertEqual(len(picks),1);self.assertFalse(picks[0]['available'])
        self.assertTrue(picks[0]['unavailable_reason']);self.assertIs(unsafe['safe_to_touch'],False)
        cached=self.module.scenes[result['id']]['objects'][0]
        self.assertFalse(next(a for a in cached['actions'] if a['kind']=='pick')['available'])

    def test_enabled_enrichment_does_not_duplicate_an_existing_safe_pick(self):
        result=self.enrich([object_record('safe',True,[pick_action(available=False)])],True)
        picks=[a for a in result['objects'][0]['actions'] if a['kind']=='pick']
        self.assertEqual(len(picks),1);self.assertTrue(picks[0]['available'])

    def test_disabled_enrichment_adds_no_pick_and_disables_existing_pick(self):
        result=self.enrich([object_record('safe without pick',True),
                            object_record('safe cached pick',True,[pick_action()])],False)
        first,second=result['objects']
        self.assertFalse(any(a['kind']=='pick' for a in first['actions']))
        self.assertFalse(next(a for a in second['actions'] if a['kind']=='pick')['available'])


class WristAdviceHistoryContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.robot=types.SimpleNamespace(begin=Mock())
        self.brain=types.SimpleNamespace(grasp_advice=Mock(return_value={'step':'observe'}))
        self.grasp=WristGrasp(self.robot,self.brain,Mock(),self.temp.name)
        self.grasp.record=Mock()
        no_images=patch('server.grasp.unchanged_wrist',return_value={})
        no_images.start();self.addCleanup(no_images.stop)

    def frame(self,name,generation='current'):
        return {'id':name,'source':'wrist','camera_generation':generation,
                'path':'unused-synthetic.jpg','joints':{'gripper':1857}}

    def test_history_excludes_previous_camera_generation(self):
        start=self.frame('start');current=self.frame('current');fresh=self.frame('fresh')
        self.grasp.attempt_start=start
        self.grasp.latest={'before':self.frame('old','previous'),'after':self.frame('old-after','previous')}
        self.grasp.capture=Mock(side_effect=[current,fresh])
        self.grasp.advice('synthetic',lambda message:None,threading.Event())
        views=self.brain.grasp_advice.call_args.args[1]
        self.assertEqual([f['id'] for f in views],['start','current'])
        self.assertTrue(all(f['source']=='wrist' and f['camera_generation']=='current' for f in views))
        self.assertEqual(self.brain.grasp_advice.call_args.args[2]['frame_roles'],['attempt_start','current'])

    def test_attempt_generation_change_blocks_advice_before_brain(self):
        self.grasp.attempt_start=self.frame('start','previous')
        self.grasp.capture=Mock(return_value=self.frame('current'))
        with self.assertRaisesRegex(RuntimeError,'重新連接'):
            self.grasp.advice('synthetic',lambda message:None,threading.Event())
        self.brain.grasp_advice.assert_not_called()


class SnapshotOwnershipContracts(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('server.app')
        for name,value in (('jobs',{}),('scenes',{}),('frames',{}),('active',None),
                           ('snapshot_owner',threading.local())):
            replacement=patch.object(self.module,name,value)
            replacement.start();self.addCleanup(replacement.stop)

    def test_snap_pins_worker_owner_but_not_unowned_preview(self):
        m=self.module;owner={'id':'worker','status':'running','frame_pins':set()}
        m.jobs['worker']=owner;m.active='worker'
        owned={'id':'owned','source':'wrist','path':'unused-owned.jpg'}
        preview={'id':'preview','source':'wrist','path':'unused-preview.jpg'}
        with patch.object(m.camera,'snapshot',side_effect=[owned,preview]):
            m.snapshot_owner.job=owner
            self.assertEqual(m.snap(),owned)
            del m.snapshot_owner.job
            # A running global job must not adopt an unrelated preview capture.
            self.assertEqual(m.snap(),preview)
        self.assertEqual(owner['frame_pins'],{'owned'})
        self.assertEqual(set(m.frames),{'owned','preview'})

    def test_failed_submit_clears_worker_snapshot_owner(self):
        m=self.module;seen=[]
        robot=types.SimpleNamespace(stop=threading.Event(),pwm_owner=None)
        executor=Mock();executor.submit.side_effect=lambda run:run()
        def failure(job):
            self.assertIs(m.snapshot_owner.job,job)
            seen.append(job)
            raise RuntimeError('synthetic worker failure')
        with patch.object(m,'robot',robot),patch.object(m,'executor',executor):
            result=m.submit('grasp-attempt',failure)
        executor.submit.assert_called_once()
        self.assertEqual(len(seen),1)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['message'],'synthetic worker failure')
        self.assertFalse(hasattr(m.snapshot_owner,'job'))

    def test_preview_eviction_keeps_completed_grasp_result_images(self):
        m=self.module
        m.frames={str(i):{'id':str(i),'path':f'synthetic-{i}.jpg'} for i in range(203)}
        m.scenes['result']={'frames':[],'grasp':{'pre_lift_frame':m.frames['0'],'held_frame':m.frames['1']}}
        m.jobs['finished']={'kind':'grasp-attempt','status':'completed','engineering_result':
            {'pre_lift_frame':m.frames['2'],'held_frame':m.frames['3']}}
        with patch.object(m.camera,'snapshot',return_value={'id':'new','path':'synthetic-new.jpg'}),patch.object(m,'Path') as paths:
            m.snap()
        self.assertEqual(len(m.frames),200)
        self.assertTrue({'0','1','2','3'}<=set(m.frames))
        self.assertEqual([call.args[0] for call in paths.call_args_list],
                         [f'synthetic-{i}.jpg' for i in range(4,8)])

    def test_preview_preserves_completed_observation_evidence_after_failed_tasks(self):
        m=self.module
        m.frames={str(i):{'id':str(i),'source':'wrist','path':f'synthetic-{i}.jpg'} for i in range(203)}
        for i,kind in enumerate(('grasp-attempt','object-program','command')):
            receipt={'observations':[{'completed':True,'after':m.frames[str(i)]}]}
            m.jobs[str(i)]={'kind':kind,'status':'failed'}
            if kind=='command':m.jobs[str(i)]['command_result']={'steps':[{'receipt':receipt}]}
            else:m.jobs[str(i)]['engineering_result']=receipt
        with patch.object(m.camera,'snapshot',return_value={'id':'new','path':'synthetic-new.jpg'}),patch.object(m,'Path'):
            m.snap()
        self.assertEqual(len(m.frames),200)
        self.assertTrue({'0','1','2'}<=set(m.frames))
        self.assertNotIn('3',m.frames)


if __name__=='__main__':unittest.main(verbosity=2)
