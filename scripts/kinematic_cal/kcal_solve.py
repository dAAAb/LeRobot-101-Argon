"""Offline kinematic calibration from a kcal_run.py session. No hardware access.

Unknowns (model C): software joint-zero offsets of shoulder_lift, elbow_flex and
wrist_flex; camera hand-eye X (gripper_link -> optical); board pose in base.
shoulder_pan offset is exactly absorbed by the board yaw, wrist_roll offset by X,
so both stay zero. Intrinsics come from the training views only (no FK involved).
Every reported validation number uses views that took no part in any fit.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from kcal_common import BODY, OBJ, ROOT, BatchFK, inv, load_v2_camera, project

TPD = 4096 / 360.0                   # ticks per degree
OFFSET_JOINTS = ['shoulder_lift', 'elbow_flex', 'wrist_flex']
bfk = BatchFK()


def make_T(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:6]
    return T


def split_T(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def chain(views, K, D, X, Tbb, offsets_deg):
    raw = {n: np.array([v['joints'][n] for v in views], float) for n in BODY}
    offs = {n: offsets_deg.get(n, 0.0) * TPD for n in BODY}
    C = bfk(raw, offs) @ X
    Pb = np.c_[OBJ, np.ones(len(OBJ))] @ Tbb.T
    Pc = np.einsum('vij,kj->vki', inv(C), Pb)[..., :3]
    uv, _ = project(K, D, Pc)
    return uv, C


def residuals(views, K, D, X, Tbb, offsets_deg):
    uv, _ = chain(views, K, D, X, Tbb, offsets_deg)
    return (uv - np.array([v['corners'] for v in views])).reshape(-1)


def fit(views, K, D, X0, Tbb0, fit_offsets, fit_handeye=True):
    def unpack(p):
        k = 0
        offs = {}
        if fit_offsets:
            offs = dict(zip(OFFSET_JOINTS, p[:3]))
            k = 3
        X = make_T(p[k:k + 6]) if fit_handeye else X0
        k += 6 if fit_handeye else 0
        return offs, X, make_T(p[k:k + 6])

    def objective(p):
        offs, X, Tbb = unpack(p)
        return residuals(views, K, D, X, Tbb, offs)

    p0 = np.r_[np.zeros(3) if fit_offsets else [], split_T(X0) if fit_handeye else [], split_T(Tbb0)]
    sol = least_squares(objective, p0, loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=400)
    offs, X, Tbb = unpack(sol.x)
    J = sol.jac
    r = sol.fun
    dof = max(1, len(r) - len(sol.x))
    sigma2 = float(r @ r) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(sol.x), np.nan)
    return {'offsets_deg': offs, 'offsets_std_deg': dict(zip(OFFSET_JOINTS, std[:3])) if fit_offsets else {},
            'X': X, 'Tbb': Tbb, 'success': bool(sol.success), 'cost': float(sol.cost)}


def evaluate(views, K, D, X, Tbb, offs):
    uv, C = chain(views, K, D, X, Tbb, offs)
    corners = np.array([v['corners'] for v in views])
    err = np.linalg.norm(uv - corners, axis=-1)
    # 3D: chain-predicted camera pose vs. that view's own PnP camera pose (both relative to the board)
    trans, rot = [], []
    for v, Cv in zip(views, C):
        ok, rvec, tvec = cv2.solvePnP(OBJ, np.array(v['corners']), K, D, flags=cv2.SOLVEPNP_IPPE)
        rvec, tvec = cv2.solvePnPRefineLM(OBJ, np.array(v['corners']), K, D, rvec, tvec)
        T_cb = np.eye(4)
        T_cb[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cb[:3, 3] = tvec.ravel()
        T_cb_chain = inv(Cv) @ Tbb
        # board corner displacement in metres, camera frame
        P = np.c_[OBJ, np.ones(len(OBJ))]
        d = (P @ T_cb_chain.T)[:, :3] - (P @ T_cb.T)[:, :3]
        trans.append(float(np.linalg.norm(d, axis=1).max()))
        dR = T_cb_chain[:3, :3] @ T_cb[:3, :3].T
        rot.append(float(np.degrees(Rotation.from_matrix(dR).magnitude())))
    up = -Tbb[:3, 2]
    return {'rms_px': float(np.sqrt(np.mean(err ** 2))), 'p95_px': float(np.percentile(err, 95)),
            'max_px': float(err.max()), 'per_view_rms_px': [round(float(np.sqrt(np.mean(e ** 2))), 2) for e in err],
            'board_corner_max_mm': round(1000 * max(trans), 2), 'board_corner_median_mm': round(1000 * float(np.median(trans)), 2),
            'rotation_max_deg': round(max(rot), 3),
            'table_tilt_deg': round(float(np.degrees(np.arccos(np.clip(up[2], -1, 1)))), 3),
            'board_origin_z_mm': round(1000 * float(Tbb[2, 3]), 2)}


def intrinsics(views):
    obj = [OBJ.astype(np.float32)] * len(views)
    img = [np.array(v['corners'], np.float32) for v in views]
    K0, D0, _ = load_v2_camera()
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, (1920, 1080), K0.copy(), np.zeros(5), flags=flags)
    return K, D.ravel(), rms


def main():
    session = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT.parents[1] / 'config/calibration/kinematic-sessions/session-20261001-173458'
    views = [json.loads(l) for l in open(session / 'views.jsonl') if json.loads(l)['detected']]
    # Resolve the checkerboard's 180-degree ordering ambiguity with the plan's predicted corners.
    # The colour test can be fooled at grazing angles; the two candidates differ by a whole board
    # (~300 px) while the v2-model prediction error is ~70-150 px, so this choice is unambiguous.
    plan = json.loads((session / 'plan.json').read_text())
    predicted = {f"view_{p['index']:02d}": np.array(p['predicted_corners']) for p in plan['views']}
    for v in views:
        if v['name'] not in predicted:
            continue
        c = np.array(v['corners'])
        flipped = c.reshape(6, 9, 2)[::-1, ::-1].reshape(-1, 2)
        d_keep = float(np.linalg.norm(c - predicted[v['name']], axis=1).mean())
        d_flip = float(np.linalg.norm(flipped - predicted[v['name']], axis=1).mean())
        if d_flip + 100 < d_keep:
            print(f"{v['name']}: corner order flipped 180 deg (prediction distance {d_keep:.0f} -> {d_flip:.0f} px)")
            v['corners'] = flipped.tolist()
            v['orientation_fixed'] = True
        elif not d_keep + 100 < d_flip:
            raise SystemExit(f"{v['name']}: ambiguous orientation ({d_keep:.0f} vs {d_flip:.0f} px)")
    train = [v for v in views if v['split'] == 'train']
    held = [v for v in views if v['split'] == 'heldout']
    check = [v for v in views if v['split'] == 'check']
    print(f'session {session.name}: {len(train)} train, {len(held)} held-out, {len(check)} start/return checks')

    K2, D2, X2 = load_v2_camera()
    K, D, rms = intrinsics(train)
    Kall, Dall, rms_all = intrinsics(train + held + check)
    print(f'intrinsics (train only): fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f} '
          f'dist={np.round(D[:4], 4).tolist()} rms={rms:.3f}px   [v2: fx={K2[0,0]:.1f} fy={K2[1,1]:.1f} cx={K2[0,2]:.1f} cy={K2[1,2]:.1f}]')

    # board pose initialisation: start view through the v2 chain
    start = check[0] if check else train[0]
    _, C0 = chain([start], K2, D2, X2, np.eye(4), {})
    ok, rvec, tvec = cv2.solvePnP(OBJ, np.array(start['corners']), K2, D2)
    T_cb = np.eye(4)
    T_cb[:3, :3] = cv2.Rodrigues(rvec)[0]
    T_cb[:3, 3] = tvec.ravel()
    Tbb0 = C0[0] @ T_cb

    # Detection sanity: a view whose chain residual is far above the rest (e.g. a 9x6 window that
    # locked onto a spurious extra row of junctions) is a correspondence failure, not model error.
    probe = fit(train, K, D, X2, Tbb0, fit_offsets=True)
    rejected = []
    for group in (train, held, check):
        uv, _ = chain(group, K, D, probe['X'], probe['Tbb'], probe['offsets_deg'])
        rms_v = np.sqrt(np.mean(np.sum((uv - np.array([v['corners'] for v in group])) ** 2, -1), 1))
        limit = max(20.0, 4 * float(np.median(rms_v)))
        rejected += [(v['name'], round(float(r), 1)) for v, r in zip(group, rms_v) if r > limit]
    if rejected:
        print('rejected as detection failures (view, rms px):', rejected)
        names = {n for n, _ in rejected}
        train = [v for v in train if v['name'] not in names]
        held = [v for v in held if v['name'] not in names]
        check = [v for v in check if v['name'] not in names]
        views = [v for v in views if v['name'] not in names]

    results = {}
    # A: Astra v2 exactly as published (its K, D, X); only the board pose is re-fitted
    a = fit(train, K2, D2, X2, Tbb0, fit_offsets=False, fit_handeye=False)
    results['A_astra_v2_as_is'] = (a, K2, D2)
    # B: hand-eye + board re-fitted on this wider data, no joint offsets (Astra's model class)
    b = fit(train, K, D, X2, Tbb0, fit_offsets=False)
    results['B_handeye_only'] = (b, K, D)
    # C: + software joint-zero offsets
    c = fit(train, K, D, X2, Tbb0, fit_offsets=True)
    results['C_handeye_plus_joint_offsets'] = (c, K, D)

    report = {'session': session.name, 'created': time.strftime('%Y-%m-%d %H:%M:%S'),
              'train': [v['name'] for v in train], 'heldout': [v['name'] for v in held], 'models': {}}
    print(f"\n{'model':32}{'train rms':>10}{'HELD-OUT rms':>13}{'p95':>7}{'max':>7}{'board err max':>14}{'rot max':>8}{'table tilt':>11}{'board z':>9}")
    for name, (r, Km, Dm) in results.items():
        tr = evaluate(train, Km, Dm, r['X'], r['Tbb'], r['offsets_deg'])
        he = evaluate(held, Km, Dm, r['X'], r['Tbb'], r['offsets_deg'])
        ck = evaluate(check, Km, Dm, r['X'], r['Tbb'], r['offsets_deg']) if check else None
        report['models'][name] = {'train': tr, 'heldout': he, 'check': ck,
                                  'offsets_deg': {k: round(v, 3) for k, v in r['offsets_deg'].items()},
                                  'offsets_std_deg': {k: round(float(v), 3) for k, v in r['offsets_std_deg'].items()},
                                  'handeye': r['X'].tolist(), 'board_base': r['Tbb'].tolist(),
                                  'camera_matrix': Km.tolist(), 'distortion': list(map(float, Dm))}
        print(f"{name:32}{tr['rms_px']:>9.2f}px{he['rms_px']:>11.2f}px{he['p95_px']:>7.1f}{he['max_px']:>7.1f}"
              f"{he['board_corner_max_mm']:>11.1f} mm{he['rotation_max_deg']:>7.2f}°{tr['table_tilt_deg']:>9.2f}°{tr['board_origin_z_mm']:>7.1f}mm")
    off = report['models']['C_handeye_plus_joint_offsets']
    print('\njoint-zero offsets (model C):', {k: f"{v:+.2f}° ± {off['offsets_std_deg'][k]:.2f}° ({v * TPD:+.0f} ticks)" for k, v in off['offsets_deg'].items()})
    print('held-out per-view rms px  B:', report['models']['B_handeye_only']['heldout']['per_view_rms_px'])
    print('held-out per-view rms px  C:', off['heldout']['per_view_rms_px'])

    # final profile: refit model C on ALL views with all-view intrinsics (validation numbers above stay train/held-out)
    final = fit(train + held + check, Kall, Dall, c['X'], c['Tbb'], fit_offsets=True)
    hashes = {v['image']: hashlib.sha256((session / v['image']).read_bytes()).hexdigest() for v in views if (session / v['image']).exists()}
    profile = {'schema': 'claude-kinematic-cal/1', 'created': report['created'], 'session': session.name,
               'model': 'Astra URDF FK, angle=(raw + offset_ticks - 2047)*2pi/4096; offsets are SOFTWARE ONLY (EEPROM untouched)',
               'joint_offsets_ticks': {n: round(final['offsets_deg'].get(n, 0.0) * TPD, 2) for n in BODY},
               'joint_offsets_deg': {n: round(final['offsets_deg'].get(n, 0.0), 4) for n in BODY},
               'unobservable_fixed_zero': ['shoulder_pan (absorbed by board yaw)', 'wrist_roll (absorbed by hand-eye)'],
               'camera_matrix': Kall.tolist(), 'distortion': list(map(float, Dall)), 'intrinsics_rms_px': rms_all,
               'handeye_gripper_link_T_optical': final['X'].tolist(), 'board_base': final['Tbb'].tolist(),
               'validation': {k: report['models'][k]['heldout'] for k in report['models']},
               'image_sha256': hashes}
    out = ROOT / ('analysis-' + session.name)
    out.mkdir(exist_ok=True)
    (out / 'kinematic-profile.json').write_text(json.dumps(profile, indent=1))
    (out / 'solve-report.json').write_text(json.dumps(report, indent=1))
    print('\nfinal (all views) offsets ticks:', profile['joint_offsets_ticks'])
    print('saved', out / 'kinematic-profile.json')


if __name__ == '__main__':
    main()
