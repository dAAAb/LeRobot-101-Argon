"""Export offline observation geometry from a saved wrist frame and held goals.

No hardware or inference clients are constructed. Output has no executable
contract seals: live operation must regenerate and assess a fresh candidate.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.active_observation import build_observation_candidates, _pose
from server.local_cartesian import ALL, default_model


def held_goals(data, frame):
    """Accept explicit goals or acknowledged diagnostics, never infer a hold."""
    if not isinstance(data, dict):
        raise ValueError('Held goals must be an explicit JSON object')
    if 'joints' in data:
        entries = data['joints']
        if not isinstance(entries, dict) or set(entries) != set(ALL):
            raise ValueError('Diagnostics must include all six joints')
        positions = _pose({n: entries[n].get('position') for n in ALL})
        if positions != frame['joints']:
            raise ValueError('Diagnostic positions do not match the saved frame')
        return _pose({n: entries[n].get('goal') for n in ALL})
    return _pose(data.get('goals', data))


def export_candidates(frame_path, goals_path, output_path, *, list_output=None):
    started = time.perf_counter()
    frame_path, goals_path, output_path = map(Path, (frame_path, goals_path, output_path))
    original = json.loads(frame_path.read_text())
    raw = _pose(original.get('joints'))
    if original.get('source') != 'wrist' or not original.get('camera_generation'):
        raise ValueError('An archived wrist-frame record is required')
    held = held_goals(json.loads(goals_path.read_text()), original)
    model = default_model()
    grasp = SimpleNamespace(model=model, state='idle',
                            robot=SimpleNamespace(goals=held, pwm_owner=None))
    # The pure builder's freshness precondition is represented synthetically
    # for replay only. Restore the actual archived timestamp in output, and
    # remove seals so geometry replay cannot be dispatched as a live contract.
    geometry_frame = {**original, 'joints': raw, 'capture_monotonic': time.monotonic()}
    build_started = time.perf_counter()
    candidates = build_observation_candidates(grasp, geometry_frame)
    build_seconds = time.perf_counter() - build_started
    for candidate in candidates:
        candidate.pop('contract_sha256', None)
        candidate['reference']['capture_monotonic'] = original['capture_monotonic']
        if candidate.get('segment') is not None:
            candidate['segment'].pop('contract_sha256', None)
    artifact = {
        'source': 'offline-archived-wrist-geometry', 'dispatch_authorized': False,
        'execution_contract_seals_removed': True, 'source_frame_path': str(frame_path.resolve()),
        'source_goals_path': str(goals_path.resolve()), 'source_frame': original,
        'held_goals': held, 'candidates': candidates,
        'candidate_generation_seconds': build_seconds,
        'total_generation_seconds': time.perf_counter() - started,
        'limitations': ['No current visual clearance or pose freshness is asserted.',
                        'Geometry uses the CAD camera prior, not calibrated object depth.',
                        'Regenerate and assess a live candidate before physical execution.'],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))
    if list_output is not None:
        list_path = Path(list_output)
        list_path.parent.mkdir(parents=True, exist_ok=True)
        list_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
    return artifact


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--frame', required=True)
    parser.add_argument('--goals', required=True, help='Six-axis goals or acknowledged joint diagnostics JSON')
    parser.add_argument('--output', required=True, help='Provenance-rich geometry artifact')
    parser.add_argument('--list-output', help='Optional plain candidate list for export_sim_scene.py')
    args = parser.parse_args(argv)
    artifact = export_candidates(args.frame, args.goals, args.output, list_output=args.list_output)
    print(json.dumps({'output': args.output, 'candidates': [c['id'] for c in artifact['candidates']],
                      'candidate_generation_seconds': artifact['candidate_generation_seconds'],
                      'total_generation_seconds': artifact['total_generation_seconds'],
                      'dispatch_authorized': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
