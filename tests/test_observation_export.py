"""Archived observation geometry must preserve real holds and reject dispatch."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from scripts.export_observation_candidates import export_candidates, held_goals
from server.active_observation import execute_observation
from server.local_cartesian import ALL


class ObservationExportTests(unittest.TestCase):
    def setUp(self):
        self.raw = dict(zip(ALL, [2156, 931, 3040, 2251, 820, 1857]))
        self.goals = {**self.raw, 'shoulder_pan': 2158, 'wrist_flex': 2238, 'wrist_roll': 814}
        self.frame = {'source': 'wrist', 'camera_generation': 'archived-camera',
                      'joints': self.raw, 'capture_monotonic': 123., 'path': 'archived-wrist.jpg'}
        self.diagnostics = {'joints': {n: {'position': self.raw[n], 'goal': self.goals[n]} for n in ALL}}

    def test_explicit_diagnostics_do_not_substitute_measured_positions_for_goals(self):
        self.assertEqual(held_goals(self.diagnostics, self.frame), self.goals)
        self.diagnostics['joints']['wrist_flex']['position'] += 1
        with self.assertRaisesRegex(ValueError, 'do not match'):
            held_goals(self.diagnostics, self.frame)
        with self.assertRaises(ValueError):
            held_goals({'joints': {n: {'position': self.raw[n]} for n in ALL}}, self.frame)

    def test_export_preserves_history_and_removes_both_execution_seals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, goals, output, plain = [root / name for name in ('frame.json', 'goals.json', 'output.json', 'list.json')]
            frame.write_text(json.dumps(self.frame)); goals.write_text(json.dumps(self.diagnostics))
            artifact = export_candidates(frame, goals, output, list_output=plain)
            self.assertEqual(artifact['source_frame'], self.frame)
            self.assertEqual(artifact['held_goals'], self.goals)
            self.assertFalse(artifact['dispatch_authorized'])
            self.assertTrue(artifact['execution_contract_seals_removed'])
            self.assertGreater(artifact['candidate_generation_seconds'], 0)
            self.assertGreaterEqual(artifact['total_generation_seconds'], artifact['candidate_generation_seconds'])
            self.assertEqual(json.loads(plain.read_text()), artifact['candidates'])
            self.assertGreaterEqual(len(artifact['candidates']), 2)
            for candidate in artifact['candidates']:
                self.assertEqual(candidate['reference']['capture_monotonic'], 123.)
                self.assertNotIn('contract_sha256', candidate)
                if candidate['segment'] is not None:
                    self.assertNotIn('contract_sha256', candidate['segment'])
                    self.assertEqual(candidate['segment']['held_goals'], self.goals)
                with self.assertRaisesRegex(ValueError, '候選已改變'):
                    execute_observation(SimpleNamespace(), candidate, self.frame,
                                        threading.Event(), lambda _: None, verdict={})


if __name__ == '__main__':
    unittest.main()
