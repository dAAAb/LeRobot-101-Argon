import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from server import create_app


class FakeEngine:
    device = 'cpu'

    def __init__(self, device='cpu'):
        self.device = 'cpu' if device == 'auto' else device
        self.calls = []

    def generate(self, prompt, seed):
        self.calls.append((prompt, seed))
        return {'format': 'humanml3d-22', 'fps': 20, 'joints': [[[0, 1, 0]] * 22] * 4,
                'prompt': prompt, 'seed': seed, 'model': 'test-only'}


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.loaded = []
        def factory(device):
            self.loaded.append(device)
            if device == 'mlx':
                raise RuntimeError('Test load failure')
            return self.engine if device == 'auto' else FakeEngine(device)
        self.client = TestClient(create_app(factory, lambda: ['cpu', 'mps', 'mlx'], 'auto'),
                                 base_url='http://127.0.0.1:17905')
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_loaded_health_and_motion_contract(self):
        health = self.client.get('/health')
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()['ready'])
        self.assertEqual(health.json()['max_frames'], 196)
        self.assertEqual(health.json()['available_devices'], ['cpu', 'mps', 'mlx'])
        self.assertEqual(health.json()['selected_device'], 'auto')
        response = self.client.post('/v1/motions/generate', json={'prompt': ' wave ', 'seed': 42})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.engine.calls, [('wave', 42)])
        self.assertEqual(response.headers['cache-control'], 'no-store')

    def test_bounded_typed_input_and_no_arbitrary_options(self):
        for body in [{'prompt': ''}, {'prompt': '  '}, {'prompt': 'a' * 513},
                     {'prompt': 1}, {'prompt': 'wave', 'seed': True},
                     {'prompt': 'wave', 'seed': -1}, {'prompt': 'wave', 'seed': 0.5},
                     {'prompt': 'wave', 'seed': '42'}, {'prompt': 'wave', 'seed': 2147483648},
                     {'prompt': 'wave', 'model': 'https://example.com/model'},
                     {'prompt': 'wave', 'duration': 999}]:
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/v1/motions/generate', json=body).status_code, 422)
        self.assertEqual(self.engine.calls, [])

    def test_declared_and_streamed_request_limit(self):
        route = '/v1/motions/generate'
        self.assertEqual(self.client.post(route, content=b'x' * 8193).status_code, 413)
        response = self.client.post(route, content=iter([b'x' * 4096, b'x' * 4097]),
                                    headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.engine.calls, [])

    def test_local_origins_hosts_and_cors(self):
        self.assertEqual(self.client.post('/v1/motions/generate', json={'prompt': 'wave'},
                                         headers={'Origin': 'https://example.com'}).status_code, 403)
        self.assertEqual(self.client.get('/health', headers={'Host': '127.0.0.1.example.com'}).status_code, 400)
        response = self.client.options('/v1/motions/generate', headers={
            'Origin': 'http://127.0.0.1:17900', 'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['access-control-allow-origin'], 'http://127.0.0.1:17900')

    def test_model_failure_is_not_a_fake_motion(self):
        def fail(*_args):
            raise RuntimeError('The model did not produce motion tokens')
        self.engine.generate = fail
        response = self.client.post('/v1/motions/generate', json={'prompt': 'wave'})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('joints', response.json())

    def test_busy_engine_rejects_a_second_generation(self):
        started = threading.Event()
        release = threading.Event()
        original = self.engine.generate

        def block(prompt, seed):
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError('Test synchronization timed out')
            return original(prompt, seed)

        self.engine.generate = block
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.client.post, '/v1/motions/generate', json={'prompt': 'wave'})
            self.assertTrue(started.wait(timeout=3))
            try:
                response = self.client.post('/v1/motions/generate', json={'prompt': 'dance'})
                self.assertEqual(response.status_code, 409)
                self.assertTrue(self.client.get('/health').json()['busy'])
                self.assertEqual(self.client.post('/v1/runtime', json={'device': 'mps'}).status_code, 409)
            finally:
                release.set()
            self.assertEqual(pending.result().status_code, 200)
        self.assertEqual(len(self.engine.calls), 1)

    def test_runtime_switch_is_atomic_and_validated(self):
        for body in [{'device': 'cuda'}, {'device': '../engine'}, {'device': 1},
                     {'device': 'mps', 'model': 'https://example.com'}]:
            self.assertEqual(self.client.post('/v1/runtime', json=body).status_code, 422)
        self.assertEqual(self.loaded, ['auto'])
        response = self.client.post('/v1/runtime', json={'device': 'mps'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['device'], 'mps')
        self.assertEqual(response.json()['selected_device'], 'mps')
        self.assertFalse(response.json()['busy'])
        self.assertGreaterEqual(response.json()['load_seconds'], 0)
        self.client.post('/v1/runtime', json={'device': 'mps'})
        self.assertEqual(self.loaded, ['auto', 'mps'])
        self.assertEqual(self.client.post('/v1/runtime', json={'device': 'mlx'}).status_code, 422)
        self.assertEqual(self.client.get('/health').json()['device'], 'mps')
        self.assertEqual(self.client.post('/v1/motions/generate', json={'prompt': 'wave'}).status_code, 200)

    def test_runtime_load_blocks_generation_and_other_switches(self):
        started, release = threading.Event(), threading.Event()
        def factory(device):
            if device == 'mps':
                started.set()
                if not release.wait(5):
                    raise RuntimeError('Test synchronization timed out')
            return FakeEngine(device)
        with TestClient(create_app(factory, lambda: ['cpu', 'mps'], 'cpu'),
                        base_url='http://127.0.0.1:17905') as client:
            self.assertEqual(client.post('/v1/runtime', json={'device': 'mlx'}).status_code, 422)
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(client.post, '/v1/runtime', json={'device': 'mps'})
                self.assertTrue(started.wait(3))
                try:
                    self.assertEqual(client.post('/v1/motions/generate', json={'prompt': 'wave'}).status_code, 409)
                    self.assertEqual(client.post('/v1/runtime', json={'device': 'cpu'}).status_code, 409)
                    self.assertEqual(client.get('/health').json()['device'], 'cpu')
                finally:
                    release.set()
                self.assertEqual(pending.result().status_code, 200)


if __name__ == '__main__':
    unittest.main()
