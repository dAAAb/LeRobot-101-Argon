r"""Validate a new fixed-board session with frozen wrist camera parameters.

Example (offline, no controller or hardware access):
  python scripts/extend_workspace_calibration.py \
    --profile config/my-validated-camera.json \
    --session .runtime/calibration-sessions/<id>/session.json \
    --output .runtime/calibration-extensions/new-region

At least six distinct settled poses are needed. Alternate pose families are
held out by default; repeat --heldout-frame-id to declare a split explicitly.
The output directory must be new. Rejection writes report.json only; success
also writes candidate-profile.json. Neither result activates any profile.
Fixed-optics per-view PnP quality and 15 mm / 3 degree metric consistency are
checked separately; chain pixel residuals remain diagnostic. A passing profile
retains measured base/relative-pose uncertainty and is not grasp clearance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.board_calibration import BoardSpec
from server.calibration_extension import validate_workspace_extension
from server.geometry import Kinematics
from server.spatial_runtime import geometry_revision


def _json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError(f'Missing or oversized JSON input: {path}')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    # Python's JSON decoder accepts non-standard NaN; reject it in all inputs.
    json.dumps(value, allow_nan=False)
    return value


def read_session(path):
    """Verify saved image bytes and session identity without changing the journal."""
    path = Path(path).resolve()
    session = _json(path)
    if (session.get('board_static_declared') is not True
            or session.get('status') not in ('collecting', 'solved')
            or not isinstance(session.get('session_id'), str) or not session['session_id']):
        raise ValueError('Session must declare one fixed board and be collecting or solved')
    spec = session['board_spec']
    board = BoardSpec(columns=spec['columns'], rows=spec['rows'], square_size_m=spec['square_size_m'])
    if session.get('board_fingerprint') != board.fingerprint or spec != board.to_dict():
        raise ValueError('Session board specification/fingerprint is inconsistent')
    frames = deepcopy(session['frames'])
    if not isinstance(frames, list) or not 6 <= len(frames) <= 48:
        raise ValueError('Session must contain 6–48 saved wrist observations')
    for frame in frames:
        if (frame.get('source') != 'wrist' or frame.get('board_session_id') != session['session_id']
                or frame.get('camera_id') != session['camera_id']
                or frame.get('camera_generation') != session['camera_generation']
                or frame.get('image_size') != session['image_size']):
            raise ValueError('Captured wrist frame does not match this board/camera session')
        for key in ('mount_revision', 'geometry_revision'):
            if frame.get(key, session[key]) != session[key]:
                raise ValueError(f'Captured frame has mismatched {key}')
            frame[key] = session[key]
        image_path = Path(frame['path'])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        image_path = image_path.resolve()
        if (not image_path.is_relative_to(path.parent / 'frames') or not image_path.is_file()
                or not 0 < image_path.stat().st_size <= 16 * 1024 * 1024):
            raise ValueError('Saved wrist image must belong to this session frames directory')
        data = image_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != frame.get('image_sha256'):
            raise ValueError('Saved wrist image content has changed')
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or list(image.shape[1::-1]) != session['image_size']:
            raise ValueError('Saved wrist image cannot be decoded at the recorded resolution')
        frame['path'] = str(image_path)
    return session, frames, board


def extend_saved_session(profile_path, session_path, output, *, urdf_path,
                         motor_calibration_path, heldout_frame_ids=None):
    """Produce an immutable report/candidate pair from files only."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Use a new output directory; existing evidence is never overwritten')
    profile = _json(profile_path)
    session, frames, board = read_session(session_path)
    model = SimpleNamespace(urdf_path=Path(urdf_path), calibration=_json(motor_calibration_path),
                            fk=Kinematics(urdf_path))
    revision = geometry_revision(model)
    if session['geometry_revision'] != revision:
        raise ValueError('Session geometry revision does not match the supplied URDF/FK/motor calibration')
    indices = None
    if heldout_frame_ids is not None:
        ids = [f['id'] for f in frames]
        if len(set(heldout_frame_ids)) != len(heldout_frame_ids) or any(i not in ids for i in heldout_frame_ids):
            raise ValueError('Each heldout frame ID must be unique and present in the session')
        indices = [ids.index(i) for i in heldout_frame_ids]
    report = validate_workspace_extension(frames, model, profile,
        camera_id=session['camera_id'], mount_revision=session['mount_revision'],
        geometry_revision=revision, board_session_id=session['session_id'], board=board,
        holdout_indices=indices)
    report['input_files'] = {name: {'path': str(Path(path).resolve()),
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for name, path in
        (('parent_profile', profile_path), ('board_session', session_path),
         ('urdf', urdf_path), ('motor_calibration', motor_calibration_path))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.workspace-validation-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'result'
        staged.mkdir()
        (staged / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        if report['status'] == 'validated':
            (staged / 'candidate-profile.json').write_text(json.dumps(report['calibration_profile'],
                ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        staged.rename(output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--profile', type=Path, required=True, help='Existing validated wrist-only profile')
    parser.add_argument('--session', type=Path, required=True, help='New fixed-board session.json')
    parser.add_argument('--output', type=Path, required=True, help='New report/candidate directory')
    parser.add_argument('--urdf', type=Path, default=ROOT / 'assets/so101_follower_cam.urdf')
    parser.add_argument('--motor-calibration', type=Path, default=ROOT / 'config/follower-calibration.json')
    parser.add_argument('--heldout-frame-id', action='append', help='Repeat for at least three distinct poses')
    args = parser.parse_args(argv)
    try:
        report = extend_saved_session(args.profile, args.session, args.output,
            urdf_path=args.urdf, motor_calibration_path=args.motor_calibration,
            heldout_frame_ids=args.heldout_frame_id)
    except (ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
        print(f'Workspace validation rejected: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'status': report['status'], 'output': str(args.output.resolve()),
        'candidate_profile': str((args.output / 'candidate-profile.json').resolve())
                             if report['calibration_profile'] is not None else None,
        'blockers': report['blockers'], 'activation_performed': False}, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'validated' else 2


if __name__ == '__main__':
    raise SystemExit(main())
