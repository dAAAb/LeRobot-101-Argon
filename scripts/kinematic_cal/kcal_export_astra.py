"""Export the fitted kinematic model as Astra spatial/jaw profiles. No hardware access.

Writes, next to the existing profiles in config/calibration/:
  kinematic-<model stem>-profile.json    spatial profile (K, D, hand-eye, tilted table plane, measured uncertainty)
  jaw-<sha256>-profile.json              the active jaw profile re-bound to the new geometry/camera revision
bound to the geometry revision Astra has with `kinematic_model_global: true`. Activate them together with that
flag (the v2 workspace-calibration artifacts and their regression fixtures assume the offset-free FK).

  python kcal_export_astra.py [--model config/calibration/kinematic-model-20261002.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import kcal_model as km
from kcal_common import ASTRA, BODY, OBJ, inv
from server.local_cartesian import default_model
from server.spatial_contract import WristCalibration
from server.spatial_runtime import geometry_revision

IMAGE = (1920, 1080)


def heldout_camera_errors(sessions):
    """Camera pose from the model chain vs camera pose implied by PnP on the fitted board, held-out views only."""
    model, report = km.solve(sessions, save=False)
    views = [v for v in km.load_views(sessions) if v['split'] in ('heldout', 'check')
             and (v['name'] + '@' + v['placement'][-6:]) not in report['rejected']]
    _, C = km.predict(views, model['K'], model['D'], model['X'], model['boards'], model['offsets_deg'])
    trans, rot = [], []
    for v, Cv in zip(views, C):
        ok, rvec, tvec = cv2.solvePnP(OBJ, np.array(v['corners']), model['K'], model['D'])
        T_cb = np.eye(4)
        T_cb[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cb[:3, 3] = tvec.ravel()
        C_pnp = model['boards'][v['placement']] @ inv(T_cb)
        d = inv(Cv) @ C_pnp
        trans.append(float(np.linalg.norm(d[:3, 3])))
        rot.append(float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))))
    return np.array(trans), np.array(rot), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='config/calibration/kinematic-model-20261002.json')
    ap.add_argument('--jaw', default='config/calibration/jaw-994751be84ebe980031025af4e1ed570adcb6ade55c3dae00fc24c2c3ade9cb8-profile.json')
    args = ap.parse_args()
    config = json.loads((ASTRA / 'config/robot.json').read_text())
    m = json.loads((ASTRA / args.model).read_text())
    # The revision Astra will have with `kinematic_model_global: true` (offsets in every FK).
    runtime = default_model(ASTRA, offsets={n: v for n, v in m['offsets_ticks'].items() if v})
    revision = geometry_revision(runtime)
    sessions = [km.COMMITTED_SESSIONS / s for s in m['sessions']]
    trans, rot, report = heldout_camera_errors(sessions)
    val = m['validation']
    tilt = np.array(m['table_tilt_T'])
    plane = np.eye(4)
    plane[:3, :3] = tilt[:3, :3]
    plane[:3, 3] = tilt[:3, :3] @ np.array([0.0, 0.0, m['table_z_m']])
    cal = json.loads((ASTRA / 'config/follower-calibration.json').read_text())
    stem = Path(args.model).stem
    validation_id = 'kinematic-' + hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:40]
    profile = {
        'schema_version': 1, 'source': 'wrist-only', 'calibration_revision': stem,
        'camera_id': config['wrist_camera_id'], 'mount_revision': config.get('camera_mount_revision', 'existing-mount-unverified'),
        'geometry_revision': revision, 'native_image_size': list(IMAGE), 'image_size': list(IMAGE),
        'camera_matrix': m['camera_matrix'], 'distortion': m['distortion'], 'distortion_model': 'opencv-brown',
        'handeye': m['handeye_gripper_link_T_optical'],
        'validated_components': ['intrinsics', 'handeye', 'robot_geometry'],
        'validation_id': validation_id, 'validation_kind': 'independent-heldout',
        'validation_scope': {
            'kind': 'kinematic-model-joint-ranges',
            'note': 'Joint offsets fitted from three checkerboard placements; bounds are the onboard motor ranges. '
                    'Leave-one-placement-out extrapolation error up to ~10 mm (see docs/KINEMATIC_CALIBRATION_2026-10-01.md).',
            'joint_raw_bounds': {n: [cal[n]['range_min'], cal[n]['range_max']] for n in BODY},
            'board_plane_base': plane.tolist(),
        },
        'board_session_id': stem,
        'metric_calibrated': True, 'needs_more_views': False, 'metric_accuracy': 'measured-heldout',
        'grasp_ready': False,
        'uncertainty_interpretation': (
            'Held-out views of three board placements: board-corner 3D error max (base position); camera pose from '
            'the FK chain vs PnP on the fitted board, max translation and rotation (relative). Includes PnP noise.'),
        'relative_translation_uncertainty_m': round(float(trans.max()), 5),
        'base_position_uncertainty_m': round(val['board_corner_max_mm'] / 1000, 5),
        'relative_rotation_uncertainty_deg': round(float(rot.max()), 4),
        'focal_relative_uncertainty': 0.005,
        'kinematic_model': {'path': args.model, 'offsets_ticks': m['offsets_ticks'], 'sessions': m['sessions'],
                            'heldout_board_corner_mm': {'median': val['board_corner_median_mm'], 'max': val['board_corner_max_mm']},
                            'heldout_camera_translation_mm': {'median': round(float(np.median(trans)) * 1000, 2), 'max': round(float(trans.max()) * 1000, 2)},
                            'heldout_camera_rotation_deg': {'median': round(float(np.median(rot)), 3), 'max': round(float(rot.max()), 3)},
                            'table_tilt_deg': m['table_tilt_deg'], 'table_z_m': m['table_z_m']},
    }
    out = ASTRA / 'config/calibration' / f'{stem}-profile.json'
    out.write_text(json.dumps(profile, indent=1))
    calibration = WristCalibration.from_mapping(profile, native_image_size=IMAGE, camera_id=profile['camera_id'],
                                                geometry_revision=revision, cad_handeye=np.eye(4),
                                                mount_revision=profile['mount_revision'])
    if not calibration.metric_calibrated:
        raise SystemExit('exported profile is not accepted as metric-calibrated')

    old_path = ASTRA / args.jaw
    jaw = json.loads(old_path.read_text())
    jaw['restamp_provenance'] = {
        'parent_profile': old_path.name, 'parent_geometry_revision': jaw['geometry_revision'],
        'parent_camera_calibration_fingerprint': jaw.get('model_assisted', {}).get('camera_calibration_fingerprint'),
        'reason': 'joint offsets and a new wrist camera model; pad samples NOT re-measured (CAD/silhouette values carried over)'}
    jaw['geometry_revision'] = revision
    if 'model_assisted' in jaw:
        jaw['model_assisted']['camera_calibration_fingerprint'] = calibration.fingerprint
    blob = json.dumps(jaw, indent=1).encode()
    jaw_out = ASTRA / 'config/calibration' / f'jaw-{hashlib.sha256(blob).hexdigest()}-profile.json'
    jaw_out.write_bytes(blob)
    print(json.dumps({'spatial_calibration_profile': str(out.relative_to(ASTRA)),
                      'jaw_calibration_profile': str(jaw_out.relative_to(ASTRA)),
                      'geometry_revision': revision,
                      'uncertainty': {k: profile[k] for k in ('relative_translation_uncertainty_m', 'base_position_uncertainty_m',
                                                              'relative_rotation_uncertainty_deg')},
                      'heldout': profile['kinematic_model']}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
