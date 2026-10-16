#!/usr/bin/env python3
"""Fit saved wrist sessions and register an INACTIVE candidate; no devices."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.calibration_store import CalibrationSessionStore
from server.geometry import Kinematics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-profile', type=Path, required=True)
    parser.add_argument('--target-session', required=True)
    parser.add_argument('--source', action='append', required=True, metavar='SESSION_ID:SOLVE_ID',
                        help='Repeat for each session; preserves this saved report\'s train/heldout split')
    parser.add_argument('--urdf', type=Path, default=ROOT / 'assets/so101_follower_cam.urdf')
    parser.add_argument('--motor-calibration', type=Path, default=ROOT / 'config/follower-calibration.json')
    args = parser.parse_args()
    try:
        store, sources, splits = CalibrationSessionStore(ROOT), {}, {}
        for source in args.source:
            sid, solve_id = source.split(':')
            if sid in sources:
                raise ValueError('A source session may only be supplied once')
            # Store validates the solve ID and journal membership before fitting.
            session = store._load(sid)
            if not any(s['solve_id'] == solve_id for s in session['solves']):
                raise ValueError('Source solve does not belong to its recorded session')
            report = json.loads((store._directory(sid) / 'solves' / solve_id / 'report.json').read_text())
            splits[sid] = {n: report['validation'][n] for n in ('training_frame_ids', 'heldout_frame_ids')}
            sources[sid] = solve_id
        model = SimpleNamespace(urdf_path=args.urdf, fk=Kinematics(args.urdf),
            calibration=json.loads(args.motor_calibration.read_text()))
        result = store.solve_multi_session(args.target_session, model, splits=splits,
            split_source_solve_ids=sources, parent_profile_path=args.parent_profile)
        print(json.dumps(dict(session_id=result['session_id'], status=result['status'],
            last_solve=result['last_solve'], activation_performed=False), ensure_ascii=False, indent=2))
        return 0 if result['last_solve']['status'] == 'validated' else 2
    except (ValueError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
