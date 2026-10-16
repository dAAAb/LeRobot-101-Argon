"""Multi-placement kinematic calibration with a shared, level table plane. No hardware access.

Unknowns: software joint-zero offsets (shoulder_lift, elbow_flex, wrist_flex), hand-eye X,
one table height z (all boards lie on the table the robot base stands on, so the board
normal is the base +z axis), and per placement (x, y, yaw). Intrinsics come from training
views only. Validation numbers always use views that took no part in any fit.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from kcal_common import BODY, OBJ, ROOT, BatchFK, inv, load_v2_camera, project

TPD = 4096 / 360.0
OFFSET_JOINTS = ['shoulder_lift', 'elbow_flex', 'wrist_flex']
MODEL_PATH = ROOT / 'kinematic-model.json'            # where a new solve is written (git-ignored)
COMMITTED_MODEL = ROOT.parents[1] / 'config/calibration/kinematic-model-20261002.json'
COMMITTED_SESSIONS = ROOT.parents[1] / 'config/calibration/kinematic-sessions'
bfk = BatchFK()


def make_T(p):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:6]
    return T


def split_T(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def board_on_table(x, y, yaw, z):
    """Board frame: origin at the first inner corner, z axis into the table."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', yaw).as_matrix() @ np.diag([1., -1., -1.])
    T[:3, 3] = [x, y, z]
    return T


def camera_poses(joints_list, X, offsets_deg):
    raw = {n: np.array([j[n] for j in joints_list], float) for n in BODY}
    offs = {n: offsets_deg.get(n, 0.0) * TPD for n in BODY}
    return bfk(raw, offs) @ X


def predict(views, K, D, X, boards, offsets_deg):
    C = camera_poses([v['joints'] for v in views], X, offsets_deg)
    out = []
    for v, Cv in zip(views, C):
        Pb = np.c_[OBJ, np.ones(len(OBJ))] @ boards[v['placement']].T
        Pc = (Pb @ inv(Cv).T)[:, :3]
        out.append(project(K, D, Pc)[0])
    return np.array(out), C


# --------------------------------------------------------------------------- data
def load_views(session_dirs):
    views = []
    for sd in session_dirs:
        sd = Path(sd)
        plan = json.loads((sd / 'plan.json').read_text()) if (sd / 'plan.json').exists() else {'views': []}
        predicted = {f"view_{p['index']:02d}": np.array(p['predicted_corners']) for p in plan['views']}
        for line in open(sd / 'views.jsonl'):
            v = json.loads(line)
            if not v['detected'] or v.get('ambiguous_pattern'):
                continue
            v['placement'] = sd.name
            v['session_dir'] = str(sd)
            if v['name'] in predicted:   # resolve the 180-degree ordering ambiguity
                c = np.array(v['corners'])
                f = c.reshape(6, 9, 2)[::-1, ::-1].reshape(-1, 2)
                dk = float(np.linalg.norm(c - predicted[v['name']], axis=1).mean())
                df = float(np.linalg.norm(f - predicted[v['name']], axis=1).mean())
                if df + 100 < dk:
                    v['corners'] = f.tolist()
                    v['orientation_fixed'] = True
            views.append(v)
    return views


def intrinsics(views, K0):
    obj = [OBJ.astype(np.float32)] * len(views)
    img = [np.array(v['corners'], np.float32) for v in views]
    # k1,k2,k3,p1,p2: the wide-angle lens needs k3 (held-out PnP max 2.0 -> 0.7 px). The rational
    # model fits slightly better but folds near the image corners, so it is not used.
    rms, K, D, _, _ = cv2.calibrateCamera(obj, img, (1920, 1080), K0.copy(), np.zeros(5),
                                          flags=cv2.CALIB_USE_INTRINSIC_GUESS)
    D = D.ravel()
    k1, k2, p1, p2, k3 = D[:5]
    rmax = np.hypot(max(K[0, 2], 1920 - K[0, 2]) / K[0, 0], max(K[1, 2], 1080 - K[1, 2]) / K[1, 1])
    r = np.linspace(0, 1.15 * rmax, 2000)
    if not np.all(np.diff(r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)) > 0):
        raise ValueError('distortion model is not monotonic over the image; refusing it')
    return K, D, rms


def initial_boards(views, K, D, X, offsets):
    """Per placement: average the board pose implied by each view's PnP through the chain."""
    boards = {}
    for pl in sorted({v['placement'] for v in views}):
        poses = []
        for v in [w for w in views if w['placement'] == pl]:
            ok, rvec, tvec = cv2.solvePnP(OBJ, np.array(v['corners']), K, D)
            T_cb = np.eye(4)
            T_cb[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cb[:3, 3] = tvec.ravel()
            poses.append(camera_poses([v['joints']], X, offsets)[0] @ T_cb)
        t = np.mean([P[:3, 3] for P in poses], axis=0)
        yaw = float(np.median([np.arctan2(P[1, 0], P[0, 0]) for P in poses]))
        boards[pl] = (t[0], t[1], yaw, t[2])
    return boards


# --------------------------------------------------------------------------- fit
def fit(train, K, D, X0, offsets0, boards0, fit_offsets=True):
    placements = sorted(boards0)

    def unpack(p):
        k = 0
        offs = dict(zip(OFFSET_JOINTS, p[:3])) if fit_offsets else dict(offsets0)
        k = 3 if fit_offsets else 0
        X = make_T(p[k:k + 6])
        k += 6
        z = p[k]
        k += 1
        tilt = np.eye(4)                       # one shared table plane, allowed to tilt w.r.t. the base
        tilt[:3, :3] = Rotation.from_rotvec([p[k], p[k + 1], 0.0]).as_matrix()
        k += 2
        boards = {}
        for pl in placements:
            x, y, yaw = p[k:k + 3]
            k += 3
            boards[pl] = tilt @ board_on_table(x, y, yaw, z)
        return offs, X, boards, (z, tilt)

    z0 = float(np.mean([b[3] for b in boards0.values()]))
    p0 = np.r_[[offsets0.get(n, 0.0) for n in OFFSET_JOINTS] if fit_offsets else [], split_T(X0), z0, 0.0, 0.0,
               np.concatenate([boards0[pl][:3] for pl in placements])]
    corners = np.array([v['corners'] for v in train])

    def objective(p):
        offs, X, boards, _ = unpack(p)
        uv, _ = predict(train, K, D, X, boards, offs)
        return (uv - corners).reshape(-1)

    sol = least_squares(objective, p0, loss='soft_l1', f_scale=1.0, x_scale='jac', max_nfev=800)
    offs, X, boards, (z, tilt) = unpack(sol.x)
    r = sol.fun
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * float(r @ r) / max(1, len(r) - len(sol.x))
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {'offsets_deg': offs, 'offsets_std_deg': dict(zip(OFFSET_JOINTS, std[:3])) if fit_offsets else {},
            'X': X, 'boards': boards, 'table_z': float(z), 'table_tilt': tilt,
            'table_normal': tilt[:3, 2].tolist(), 'success': bool(sol.success)}


def evaluate(views, K, D, model):
    if not views:
        return None
    uv, C = predict(views, K, D, model['X'], model['boards'], model['offsets_deg'])
    err = np.linalg.norm(uv - np.array([v['corners'] for v in views]), axis=-1)
    corner_mm = []
    for v, Cv in zip(views, C):
        ok, rvec, tvec = cv2.solvePnP(OBJ, np.array(v['corners']), K, D)
        T_cb = np.eye(4)
        T_cb[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cb[:3, 3] = tvec.ravel()
        P = np.c_[OBJ, np.ones(len(OBJ))]
        d = (P @ (inv(Cv) @ model['boards'][v['placement']]).T)[:, :3] - (P @ T_cb.T)[:, :3]
        corner_mm.append(1000 * float(np.linalg.norm(d, axis=1).max()))
    return {'rms_px': round(float(np.sqrt(np.mean(err ** 2))), 2), 'p95_px': round(float(np.percentile(err, 95)), 2),
            'max_px': round(float(err.max()), 2), 'board_corner_max_mm': round(max(corner_mm), 2),
            'board_corner_median_mm': round(float(np.median(corner_mm)), 2),
            'per_view': {v['name'] + '@' + v['placement'][-6:]: round(float(np.sqrt(np.mean(e ** 2))), 2) for v, e in zip(views, err)}}


def solve(session_dirs, save=True):
    views = load_views(session_dirs)
    train = [v for v in views if v['split'] == 'train']
    held = [v for v in views if v['split'] in ('heldout', 'check')]
    K2, D2, X2 = load_v2_camera()
    K, D, rms = intrinsics(train, K2)
    boards0 = initial_boards(train, K, D, X2, {})
    model = fit(train, K, D, X2, {}, boards0)
    # reject correspondence failures (e.g. a 9x6 window locked onto spurious junctions)
    rejected = []
    for group in (train, held):
        ev = evaluate(group, K, D, model)
        if not ev:
            continue
        vals = np.array(list(ev['per_view'].values()))
        limit = max(20.0, 4 * float(np.median(vals)))
        rejected += [k for k, r in ev['per_view'].items() if r > limit]
    if rejected:
        keep = lambda v: (v['name'] + '@' + v['placement'][-6:]) not in rejected  # noqa: E731
        train, held = [v for v in train if keep(v)], [v for v in held if keep(v)]
        K, D, rms = intrinsics(train, K2)
        model = fit(train, K, D, model['X'], model['offsets_deg'], initial_boards(train, K, D, model['X'], model['offsets_deg']))
    model.update(K=K, D=D, intrinsics_rms=rms)
    report = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'sessions': [Path(s).name for s in session_dirs],
              'n_train': len(train), 'n_validation': len(held), 'rejected': rejected,
              'offsets_deg': {k: round(v, 3) for k, v in model['offsets_deg'].items()},
              'offsets_std_deg': {k: round(float(v), 3) for k, v in model['offsets_std_deg'].items()},
              'table_z_mm': round(1000 * model['table_z'], 2),
              'table_tilt_deg': round(float(np.degrees(np.arccos(np.clip(model['table_tilt'][2, 2], -1, 1)))), 3),
              'train': evaluate(train, K, D, model), 'validation': evaluate(held, K, D, model)}
    if save:
        payload = {'schema': 'claude-kinematic-model/1', **{k: report[k] for k in ('created', 'sessions', 'offsets_deg', 'offsets_std_deg', 'table_z_mm')},
                   'note': 'angle = (raw + offset_ticks - 2047)*2pi/4096 in Astra URDF; offsets are software only, EEPROM untouched',
                   'offsets_ticks': {n: round(model['offsets_deg'].get(n, 0.0) * TPD, 2) for n in BODY},
                   'camera_matrix': K.tolist(), 'distortion': list(map(float, D)), 'intrinsics_rms_px': rms,
                   'handeye_gripper_link_T_optical': model['X'].tolist(), 'table_z_m': model['table_z'],
                   'table_tilt_T': model['table_tilt'].tolist(), 'table_tilt_deg': report['table_tilt_deg'],
                   'table_plane_note': 'table = table_tilt_T applied to the plane z = table_z_m (base frame)',
                   'boards_base': {k: v.tolist() for k, v in model['boards'].items()},
                   'validation': report['validation'], 'train_fit': report['train'], 'rejected': rejected}
        MODEL_PATH.write_text(json.dumps(payload, indent=1))
    return model, report


def load_model(path=None):
    """Newest local solve if present, else the committed 2026-10-01 model."""
    path = Path(path) if path else (MODEL_PATH if MODEL_PATH.exists() else COMMITTED_MODEL)
    m = json.loads(path.read_text())
    return {'K': np.array(m['camera_matrix']), 'D': np.array(m['distortion']),
            'X': np.array(m['handeye_gripper_link_T_optical']), 'offsets_ticks': m['offsets_ticks'],
            'table_z': m['table_z_m'], 'raw': m}


if __name__ == '__main__':
    def resolve(s):
        for base in (Path.cwd(), ROOT, COMMITTED_SESSIONS):
            if (base / s / 'views.jsonl').exists():
                return base / s
        raise SystemExit(f'session not found: {s}')
    sessions = [resolve(s) for s in sys.argv[1:]] or sorted(p for p in COMMITTED_SESSIONS.glob('session-2026*') if not p.name.endswith('-test'))
    model, report = solve(sessions)
    print(json.dumps({k: report[k] for k in ('sessions', 'n_train', 'n_validation', 'rejected', 'offsets_deg', 'offsets_std_deg', 'table_z_mm', 'table_tilt_deg')}, ensure_ascii=False))
    for k in ('train', 'validation'):
        r = report[k]
        print(f"{k:10} rms {r['rms_px']} px  p95 {r['p95_px']}  max {r['max_px']}  board corner max {r['board_corner_max_mm']} mm (median {r['board_corner_median_mm']})")
    print('saved', MODEL_PATH)
