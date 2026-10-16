"""Offline MuJoCo/GLB engineering export from an explicit saved wrist frame."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from server.sim_scene import export_scene


def validated_depth(depth, frame):
    """Keep saved depth bound to this exact wrist exposure and camera session.

    Axial sensitivity intervals are preserved as supplied, never promoted to
    isotropic 3D confidence balls or an inferred object collision volume.
    """
    if not isinstance(depth, dict) or depth.get('source') != 'wrist-only':
        raise ValueError('Depth evidence must come from wrist-only observations')
    identity = frame.get('id'); generation = frame.get('camera_generation')
    provenance = depth.get('provenance')
    if (not isinstance(identity, str) or not identity
            or depth.get('current_frame_id') != identity):
        raise ValueError('Depth current_frame_id does not match the scene frame')
    if (not isinstance(generation, str) or not generation or not isinstance(provenance, dict)
            or provenance.get('camera_generation') != generation
            or provenance.get('source', 'wrist-only') != 'wrist-only'):
        raise ValueError('Depth camera generation does not match the scene frame')
    # JSON itself accepts NaN in Python; the engineering bundle must not.
    json.dumps(depth, allow_nan=False)
    if depth.get('status') not in ('estimated', 'insufficient'):
        raise ValueError('Unknown depth evidence status')
    points = depth.get('points', [])
    if not isinstance(points, list) or len(points) > 64:
        raise ValueError('Depth evidence supports at most 64 visible-surface points')
    def number(value, lo=-3., hi=3.):
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Depth coordinates and uncertainty must be finite and bounded')
        return float(value)
    for point in points:
        if (not isinstance(point, dict) or not isinstance(point.get('position_base_m'), list)
                or len(point['position_base_m']) != 3):
            raise ValueError('Depth points require three robot-base coordinates')
        for value in point['position_base_m']:
            number(value)
    if depth['status'] == 'estimated':
        distance = depth.get('distance'); uncertainty = depth.get('uncertainty')
        if not isinstance(distance, dict) or not isinstance(uncertainty, dict):
            raise ValueError('Estimated depth requires distance and axial uncertainty')
        median = number(distance.get('camera_optical_axis_m'), .02, 2.)
        number(distance.get('gripper_forward_axis_m'))
        interval = uncertainty.get('camera_depth_interval_m')
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError('Depth uncertainty requires the original axial interval')
        lo, hi = [number(v, 0., 100.) for v in interval]
        number(uncertainty.get('depth_error_bound_m'), 0., 100.)
        if not lo <= median <= hi:
            raise ValueError('Depth estimate is outside its supplied axial uncertainty interval')
    elif depth.get('distance') is not None or points:
        raise ValueError('Insufficient evidence cannot publish estimated depth or points')
    result = deepcopy(depth)
    result.update(engineering_only=True, production_eligible=False, motion_authorized=False,
                  exported_uncertainty_representation='original_axial_interval_not_isotropic_3d_bound')
    return result


def validated_wrist_jpeg(frame):
    """Read a real, decodable JPEG from the recorded path; never capture one."""
    recorded = frame.get('path')
    if recorded is None or recorded == '':
        return None  # Geometry-only archives remain useful and explicit.
    if not isinstance(recorded, str):
        raise ValueError('Wrist image path must name an actual saved JPEG')
    path = Path(recorded)
    if not path.is_file() or not 0 < path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError('Wrist image is missing or exceeds the export size limit')
    data = path.read_bytes()
    if not data.startswith(b'\xff\xd8\xff'):
        raise ValueError('Recorded wrist image is not a JPEG')
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[0] * decoded.shape[1] > 24_000_000:
        raise ValueError('Recorded wrist JPEG is invalid or too large')
    image_size = frame.get('image_size')
    if image_size is not None and list(decoded.shape[1::-1]) != image_size:
        raise ValueError('Recorded wrist image dimensions do not match the frame metadata')
    return data


def export_bundle(destination, frame, *, targets=None, candidates=None, depth=None):
    """Validate optional evidence, then publish one complete offline artifact set."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Use a new scene directory')
    accepted_depth = validated_depth(depth, frame) if depth is not None else None
    wrist_jpeg = validated_wrist_jpeg(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bundle-', dir=destination.parent) as temporary:
        stage = Path(temporary) / 'scene'
        state = export_scene(stage, frame, targets=targets, candidates=candidates, project_root=ROOT)
        if wrist_jpeg is not None:
            (stage / 'wrist.jpg').write_bytes(wrist_jpeg)
            state['artifacts']['wrist_image'] = 'wrist.jpg'
        if accepted_depth is not None:
            (stage / 'depth.json').write_text(json.dumps(accepted_depth, ensure_ascii=False, allow_nan=False, indent=2))
            state['artifacts']['depth'] = 'depth.json'
        (stage / 'scene.json').write_text(json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2))
        stage.rename(destination)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', required=True, type=Path, help='Saved wrist observation JSON, including six measured joints')
    parser.add_argument('--output', required=True, type=Path, help='New artifact directory')
    parser.add_argument('--targets', type=Path, help='Optional explicit target evidence JSON list')
    parser.add_argument('--candidates', type=Path, help='Optional already-planned candidate JSON list')
    parser.add_argument('--depth', type=Path, help='Optional wrist-only active-depth JSON bound to the same frame and camera')
    args = parser.parse_args(argv)
    read = lambda path: json.loads(path.read_text()) if path else None
    state = export_bundle(args.output, read(args.frame), targets=read(args.targets),
                          candidates=read(args.candidates), depth=read(args.depth))
    print(json.dumps({'output': str(args.output.resolve()), 'engineering_only': True,
                      'joint_pose': state['frame']['joints'], 'verification': state['verification'],
                      'targets_with_position': sum(t['position_m'] is not None for t in state['targets'])},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
