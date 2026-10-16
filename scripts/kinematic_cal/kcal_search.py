"""Find the checkerboard at a new placement with the wrist camera. MOVES THE ARM.

Look-poses come from the current kinematic model (kinematic-model.json). Every segment
is screened for table clearance and executed with kcal_run's guarded mover; the arm
stays at the pose that found the board so that kcal_plan2.py can plan from there.
"""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from kcal_common import BODY, ROOT, BatchFK, Clearance, inv, project, board_pose_cam
from kcal_model import load_model
from kcal_run import CAL, LIMIT_MARGIN, Abort, Runner, start_recorder, stop_recorder

JAW_BOXES = [(280, 620, 860, 1080), (1330, 650, 1680, 1080)]
CLEARANCE = 0.050
TARGETS = [(0.39, -0.05), (0.37, -0.02), (0.41, -0.07), (0.38, 0.02), (0.36, -0.09)]


def with_offsets(q, offs):
    out = {n: float(q[n]) + offs.get(n, 0.0) for n in BODY}
    out['gripper'] = 2047
    return out


def pose_clearance(clr, q, offs):
    h = clr.min_height(with_offsets(q, offs), np.array([0., 0., 1.]), 0.0)
    return min(v for k, v in h.items() if k != 'shoulder_link')


def path_clearance(clr, a, b, offs):
    steps = max(2, int(np.ceil(max(abs(a[n] - b[n]) for n in BODY) / 6)))
    return min(pose_clearance(clr, {n: a[n] + (b[n] - a[n]) * s for n in BODY}, offs) for s in np.linspace(0, 1, steps + 1))


def path_ok(clr, a, b, offs):
    """Clear by CLEARANCE, or never lower than where the arm already is (start pose)."""
    need = min(CLEARANCE, pose_clearance(clr, a, offs) - 0.002)
    return pose_clearance(clr, b, offs) >= CLEARANCE and path_clearance(clr, a, b, offs) >= need


def look_candidates(model, target, q0, rng, n=300_000, half=(0.07, 0.10), dist_range=(0.26, 0.42)):
    lim = {m: (CAL[m]['range_min'] + LIMIT_MARGIN + 20, CAL[m]['range_max'] - LIMIT_MARGIN - 20) for m in BODY}
    box = {'shoulder_pan': (q0['shoulder_pan'] - 700, q0['shoulder_pan'] + 700), 'shoulder_lift': (900, 2700),
           'elbow_flex': (1500, 2960), 'wrist_flex': (850, 2650),
           'wrist_roll': (q0['wrist_roll'] - 300, q0['wrist_roll'] + 300)}
    S = {m: np.round(rng.uniform(max(lim[m][0], box[m][0]), min(lim[m][1], box[m][1]), n)) for m in BODY}
    C = BatchFK()(S, model['offsets_ticks']) @ model['X']
    z = model['table_z']
    hx, hy = half                     # board extent incl. outer squares: long side along base y
    rect = [(dx, dy) for dx in (-hx, 0, hx) for dy in (-hy, 0, hy) if (dx, dy) != (0, 0)]
    pts = np.r_[[[target[0], target[1], z]], [[target[0] + dx, target[1] + dy, z] for dx, dy in rect]]
    Pc = np.einsum('nij,kj->nki', inv(C), np.c_[pts, np.ones(len(pts))])[..., :3]
    uv, depth = project(model['K'], model['D'], Pc)
    ok = (depth > 0.1).all(1)
    ok &= (uv[:, 0, 0] > 640) & (uv[:, 0, 0] < 1280) & (uv[:, 0, 1] > 220) & (uv[:, 0, 1] < 480)
    ok &= ((uv[..., 0] > 60) & (uv[..., 0] < 1860) & (uv[..., 1] > 40) & (uv[..., 1] < 1040)).all(1)
    for x0, y0, x1, y1 in JAW_BOXES:
        ok &= ~((uv[..., 0] > x0 - 30) & (uv[..., 0] < x1 + 30) & (uv[..., 1] > y0 - 30) & (uv[..., 1] < y1 + 30)).any(1)
    dist = np.linalg.norm(C[:, :3, 3] - pts[0], axis=1)
    ok &= (dist > dist_range[0]) & (dist < dist_range[1])
    idx = np.flatnonzero(ok)
    motion = np.max(np.abs(np.stack([S[m][idx] - q0[m] for m in BODY], 1)), axis=1)
    return S, idx[np.argsort(motion)], motion


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Search the board around table points (base frame, metres).')
    ap.add_argument('--target', nargs=2, type=float, action='append', metavar=('X', 'Y'),
                    help='candidate board centre; repeat for a search list (default: the 2026-10-01 front placement list)')
    args = ap.parse_args()
    targets = [tuple(t) for t in args.target] if args.target else TARGETS
    model = load_model()
    offs = model['offsets_ticks']
    rng = np.random.default_rng(3)
    out = ROOT / ('search-' + time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir()
    runner = Runner(out)
    clr = Clearance()
    proc = None
    result = {'status': 'not_found', 'attempts': []}
    try:
        runner.health()
        goals = runner.arm.goals()
        current = {n: goals[n] for n in BODY}
        proc, live = start_recorder(out)
        for k, target in enumerate(targets):
            S, order, motion = look_candidates(model, target, current, rng, n=400_000, half=(0.06, 0.09), dist_range=(0.26, 0.50))
            chosen = None
            for i in order[:80]:
                q = {m: int(S[m][i]) for m in BODY}
                if path_ok(clr, current, q, offs):
                    chosen = q
                    break
            print(f'target {target}: {len(order)} candidate look-poses; chosen {chosen}', flush=True)
            if chosen is None:
                result['attempts'].append({'target': target, 'reason': 'no clear look-pose'})
                continue
            runner.move(chosen, f'search_{k}')
            runner.settle()
            path, joints, span = runner.capture(live, f'search_{k}')
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            board = board_pose_cam(gray, model['K'], model['D'])
            attempt = {'target': target, 'pose': chosen, 'image': path.name, 'detected': board is not None,
                       'ambiguous': bool(board and board['ambiguous'])}
            result['attempts'].append(attempt)
            current = chosen
            if board is None or board['ambiguous']:
                print('  board not usable here', attempt, flush=True)
                continue
            raw = {n: joints[n] for n in BODY}
            G = BatchFK()({n: np.array([raw[n]]) for n in BODY}, offs)[0]
            T_bb = G @ model['X'] @ board['T_cam_board']
            tilt = float(np.degrees(np.arccos(abs(T_bb[2, 2]))))
            dz = float(T_bb[2, 3] - model['table_z'])
            centre = (T_bb @ np.array([0.08, 0.05, 0, 1]))[:3]
            print(f'  FOUND: board centre {np.round(centre * 1000, 1)} mm, tilt {tilt:.2f} deg, height vs table {dz * 1000:+.1f} mm', flush=True)
            if tilt > 5 or abs(dz) > 0.025:
                attempt['rejected'] = 'not on the table plane'
                continue
            result.update(status='found', T_base_board=T_bb.tolist(), centre_mm=(centre * 1000).round(1).tolist(),
                          tilt_deg=tilt, height_vs_table_mm=dz * 1000, search_pose=raw, image=path.name)
            break
    except Abort as exc:
        result.update(status='aborted', reason=str(exc), held=runner.hold())
        print('ABORTED:', exc, flush=True)
    finally:
        if proc is not None:
            stop_recorder(proc)
        (out / 'search.json').write_text(json.dumps(result, indent=1))
        if result['status'] == 'found':
            (ROOT / 'placement.json').write_text(json.dumps({**result, 'search_dir': out.name}, indent=1))
        runner.log.close()
        runner.arm.close()
    print(json.dumps({'status': result['status'], 'dir': out.name, 'centre_mm': result.get('centre_mm')}))


if __name__ == '__main__':
    main()
