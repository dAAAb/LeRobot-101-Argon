"""Read-only engineering artifacts, with no lifespan, camera or motor access."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class EngineeringSceneAPITests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('server.app')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)
        self.directory = self.runtime / 'sim-scene' / 'current'
        self.directory.mkdir(parents=True)
        self.frame_id = 'offline-snapshot-A'
        (self.directory / 'scene.json').write_text(json.dumps({'frame': {'id': self.frame_id}}))
        runtime = patch.object(self.module, 'RUNTIME', self.runtime)
        runtime.start(); self.addCleanup(runtime.stop)
        # Deliberately do not enter TestClient's lifespan context. Entire
        # controller/perception owners are traps; these GETs need only disk IO.
        self.owners = {}
        for name in ('robot', 'camera', 'brain', 'grasp', 'executor', 'tactician'):
            fake = Mock(name=name)
            guard = patch.object(self.module, name, fake)
            guard.start(); self.addCleanup(guard.stop)
            self.owners[name] = fake
        submit = patch.object(self.module, 'submit', side_effect=AssertionError('No job may be submitted'))
        self.submit = submit.start(); self.addCleanup(submit.stop)
        self.client = TestClient(self.module.app)
        self.addCleanup(self.client.close)

    def url(self, name, frame_id=None):
        return '/api/engineering/scene/' + name + '?frame_id=' + (frame_id or self.frame_id)

    def tearDown(self):
        self.submit.assert_not_called()
        for name, owner in self.owners.items():
            self.assertEqual(owner.mock_calls, [], f'Engineering artifact touched {name}')

    def test_only_named_artifacts_are_read_with_correct_media_type_and_no_store(self):
        files = {'scene.json': ('application/json', json.dumps({'frame': {'id': self.frame_id}}).encode()),
                 'scene.glb': ('model/gltf-binary', b'glTF-offline-test'),
                 'scene.xml': ('application/xml', b'<mujoco/>'),
                 'verification.json': ('application/json', b'{"passed":true}'),
                 'depth.json': ('application/json', b'{"status":"insufficient"}'),
                 'wrist.jpg': ('image/jpeg', b'jpeg-offline-test')}
        for name, (media, content) in files.items():
            (self.directory / name).write_bytes(content)
            with self.subTest(name=name):
                response = self.client.get(self.url(name))
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, content)
                self.assertEqual(response.headers['content-type'].split(';')[0], media)
                self.assertEqual(response.headers['cache-control'], 'no-store')
                self.assertEqual((self.directory / name).read_bytes(), content)

    def test_missing_artifact_directory_and_file_return_404_without_capture_or_export(self):
        (self.directory / 'scene.json').unlink()
        response = self.client.get(self.url('wrist.jpg'))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(list(self.directory.iterdir()), [])
        self.directory.rmdir()
        response = self.client.get('/api/engineering/scene/scene.json')
        self.assertEqual(response.status_code, 404)
        self.assertFalse(self.directory.exists())

    def test_extra_files_are_not_exposed_even_inside_current_directory(self):
        for name in ('private.json', 'scene.json.bak', 'follower-calibration.json', 'ASSET_NOTICE.md'):
            (self.directory / name).write_text('not an API artifact')
            with self.subTest(name=name):
                self.assertEqual(self.client.get('/api/engineering/scene/' + name).status_code, 404)

    def test_encoded_path_traversal_and_nested_paths_are_rejected(self):
        (self.runtime / 'secret.json').write_text('never expose')
        for suffix in ('%2e%2e%2fsecret.json', '%2e%2e%5csecret.json',
                       '%252e%252e%252fsecret.json', 'nested/scene.json',
                       '%2fetc%2fpasswd', 'scene.json%00'):
            with self.subTest(suffix=suffix):
                response = self.client.get('/api/engineering/scene/' + suffix)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn('never expose', response.text)

    def test_named_artifact_symlink_cannot_escape_snapshot_directory(self):
        outside = self.runtime / 'outside.json'; outside.write_text('outside marker')
        (self.directory / 'scene.json').unlink()
        (self.directory / 'scene.json').symlink_to(outside)
        response = self.client.get('/api/engineering/scene/scene.json')
        self.assertEqual(response.status_code, 404)
        self.assertNotIn('outside marker', response.text)
        (self.directory / 'scene.json').unlink()
        (self.directory / 'scene.json').mkdir()
        self.assertEqual(self.client.get('/api/engineering/scene/scene.json').status_code, 404)

    def test_endpoint_has_no_write_or_motion_method_even_with_local_session(self):
        path = self.directory / 'scene.json'; path.write_bytes(b'original snapshot')
        for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            with self.subTest(method=method):
                response = self.client.request(method, '/api/engineering/scene/scene.json',
                    headers={'x-lerobot-token': self.module.TOKEN}, json={'action': 'move'})
                self.assertEqual(response.status_code, 405)
                self.assertEqual(path.read_bytes(), b'original snapshot')

    def test_same_origin_and_trusted_host_rules_also_protect_engineering_artifacts(self):
        (self.directory / 'scene.json').write_text('{}')
        endpoint = '/api/engineering/scene/scene.json'
        self.assertEqual(self.client.get(endpoint, headers={'Origin': 'https://external.example'}).status_code, 403)
        self.assertEqual(self.client.get(endpoint, headers={'Host': 'external.example'}).status_code, 400)
        self.assertEqual(self.client.get(endpoint, headers={'Origin': 'http://localhost:3100'}).status_code, 200)

    def test_non_manifest_artifacts_require_loaded_frame_id_and_reject_changed_snapshot(self):
        for name in ('scene.glb', 'scene.xml', 'verification.json', 'depth.json', 'wrist.jpg'):
            (self.directory / name).write_bytes(b'current artifact')
            with self.subTest(name=name):
                self.assertEqual(self.client.get('/api/engineering/scene/' + name).status_code, 400)
                response = self.client.get(self.url(name, 'older-snapshot'))
                self.assertEqual(response.status_code, 409)
                self.assertIn('重新載入', response.json()['detail'])
                self.assertEqual(self.client.get(self.url(name)).content, b'current artifact')
        self.assertEqual(self.client.get('/api/engineering/scene/scene.json').status_code, 200)
        self.assertEqual(self.client.get(self.url('scene.json', 'older-snapshot')).status_code, 409)

    def test_directory_rename_between_manifest_and_artifact_cannot_mix_versions(self):
        (self.directory / 'scene.glb').write_bytes(b'old bound geometry')
        real_open = os.open
        swapped = False
        def concurrent_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if path == 'scene.glb' and 'dir_fd' in kwargs and not swapped:
                swapped = True
                self.directory.rename(self.directory.with_name('previous'))
                self.directory.mkdir()
                (self.directory / 'scene.json').write_text(json.dumps({'frame': {'id': 'new-snapshot'}}))
                (self.directory / 'scene.glb').write_bytes(b'new geometry')
            return real_open(path, flags, *args, **kwargs)
        with patch.object(self.module.os, 'open', side_effect=concurrent_open):
            response = self.client.get(self.url('scene.glb'))
        self.assertTrue(swapped)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b'old bound geometry')
        self.assertEqual(self.client.get(self.url('scene.glb')).status_code, 409)
        self.assertEqual(self.client.get(self.url('scene.glb', 'new-snapshot')).content, b'new geometry')

    def test_oversized_artifact_is_rejected_before_reading_large_body(self):
        with (self.directory / 'scene.glb').open('wb') as handle:
            handle.truncate(64 * 1024 * 1024 + 1)
        response = self.client.get(self.url('scene.glb'))
        self.assertEqual(response.status_code, 413)
        self.assertLess(len(response.content), 1000)

    def test_history_selects_a_complete_bundle_without_using_current_artifacts(self):
        old=self.directory.parent/'saved-older';old.mkdir()
        (old/'scene.json').write_text(json.dumps({'frame':{'id':'old','captured_at':123}}))
        (old/'scene.glb').write_bytes(b'old geometry')
        (old/'depth.json').write_text('{}')
        listing=self.client.get('/api/engineering/scenes').json()['snapshots']
        self.assertEqual(listing,[{'snapshot_id':'saved-older','frame_id':'old','captured_at':123,'has_depth':True}])
        response=self.client.get('/api/engineering/scene/scene.glb?frame_id=old&snapshot_id=saved-older')
        self.assertEqual(response.content,b'old geometry')
        self.assertEqual(self.client.get('/api/engineering/scene/scene.glb?frame_id=new&snapshot_id=saved-older').status_code,409)
        self.assertEqual(self.client.get('/api/engineering/scene/scene.json?snapshot_id=../outside').status_code,404)

    def test_capture_requires_session_and_idle_owner_then_publishes_only_its_new_frame(self):
        robot=self.owners['robot'];robot.moving=False;robot.pwm_owner=None
        self.owners['grasp'].state='idle'
        headers={'x-lerobot-token':self.module.TOKEN}
        with patch.object(self.module,'active',None), patch('server.engineering_snapshot.capture_snapshot',return_value={'id':'new'}) as capture, patch('server.engineering_snapshot.publish_snapshot',return_value={'frame_id':'new','depth_included':False}) as publish:
            self.assertEqual(self.client.post('/api/engineering/scene-capture',json={}).status_code,403)
            capture.assert_not_called()
            robot.moving=True
            self.assertEqual(self.client.post('/api/engineering/scene-capture',headers=headers,json={}).status_code,409)
            capture.assert_not_called()
            robot.moving=False
            response=self.client.post('/api/engineering/scene-capture',headers=headers,json={})
            self.assertEqual(response.status_code,200)
            self.assertFalse(response.json()['depth_included'])
            capture.assert_called_once_with(robot,self.owners['camera'])
            publish.assert_called_once_with(self.runtime,{'id':'new'})

    def test_capture_failure_keeps_saved_scene_and_releases_capture_lock(self):
        self.owners['robot'].moving=False;self.owners['robot'].pwm_owner=None
        self.owners['grasp'].state='idle'
        before=(self.directory/'scene.json').read_bytes()
        with patch.object(self.module,'active',None), patch('server.engineering_snapshot.capture_snapshot',side_effect=RuntimeError('image changed')), patch('server.engineering_snapshot.publish_snapshot') as publish:
            response=self.client.post('/api/engineering/scene-capture',headers={'x-lerobot-token':self.module.TOKEN},json={})
        self.assertEqual(response.status_code,503)
        publish.assert_not_called()
        self.assertEqual((self.directory/'scene.json').read_bytes(),before)
        self.assertFalse(self.module.engineering_capture_lock.locked())


if __name__ == '__main__':
    unittest.main()
