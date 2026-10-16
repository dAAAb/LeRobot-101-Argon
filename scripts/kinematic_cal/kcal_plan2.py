"""Plan diverse, table-safe views of a located board (placement.json) with the calibrated model.

Reads the arm (positions/goals/health) and never moves it. Output plan.json has the same
format as kcal_plan.py, so kcal_run.py executes it unchanged.
  python kcal_plan2.py                       # uses placement.json and the live arm state
  python kcal_plan2.py --synthetic X Y YAW --start '{"shoulder_pan":...}'   # offline check
"""
from __future__ import annotations

import argparse
import json
import time

import cv2
import numpy as np

from kcal_common import BODY, BOARD_OUTLINE, IMAGE_SIZE, OBJ, ROOT, Arm, BatchFK, Clearance, inv, project
from kcal_model import board_on_table, load_model
from kcal_run import CAL, LIMIT_MARGIN
from kcal_search import path_clearance, pose_clearance

N_VIEWS = 14
CLEARANCE = 0.050
V_PEAK = 100.0
MARGIN_PX = 50
JAW_BOXES = [(280, 620, 860, 1080), (1330, 650, 1680, 1080)]


def seg_time(a, b):
    return max(1.0, 1.875 * max(abs(a[n] - b[n]) for n in BODY) / V_PEAK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', nargs=3, type=float, metavar=('X', 'Y', 'YAW'))
    ap.add_argument('--start')
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    model = load_model()
    offs = model['offsets_ticks']
    K, D, X = model['K'], model['D'], model['X']
    if args.synthetic:
        T_bb = board_on_table(args.synthetic[0], args.synthetic[1], args.synthetic[2], model['table_z'])
    else:
        T_bb = np.array(json.loads((ROOT / 'placement.json').read_text())['T_base_board'])
    if args.start:
        q0 = json.loads(args.start)
        g0, health = dict(q0), None
    else:
        arm = Arm()
        try:
            q0, g0, health = arm.positions(), arm.goals(), arm.health()
        finally:
            arm.close()
        if any(h['status'] or h['torque'] != 1 for h in health.values()):
            raise SystemExit(f'motor health not nominal: {health}')

    lim = {m: (CAL[m]['range_min'] + LIMIT_MARGIN + 20, CAL[m]['range_max'] - LIMIT_MARGIN - 20) for m in BODY}
    half = {'shoulder_pan': 350, 'shoulder_lift': 450, 'elbow_flex': 350, 'wrist_flex': 450, 'wrist_roll': 300}
    # inner corners need MARGIN_PX; the half-square ring beyond them only needs to stay in frame
    ring = np.array([[-0.010, -0.010, 0], [0.170, -0.010, 0], [0.170, 0.110, 0], [-0.010, 0.110, 0]])
    Pb = np.c_[np.r_[OBJ, ring], np.ones(len(OBJ) + 4)] @ T_bb.T
    centre = (T_bb @ np.array([0.08, 0.05, 0, 1]))[:3]
    W, H = IMAGE_SIZE
    Ss, uvs, dists, views, spans = [], [], [], [], []
    for _ in range(5):
        S = {m: np.round(rng.uniform(max(lim[m][0], q0[m] - half[m]), min(lim[m][1], q0[m] + half[m]), 200_000)) for m in BODY}
        C = BatchFK()(S, offs) @ X
        uv, z = project(K, D, np.einsum('nij,kj->nki', inv(C), Pb)[..., :3])
        ok = (z > 0.08).all(1)
        ok &= ((uv[:, :54, 0] > MARGIN_PX) & (uv[:, :54, 0] < W - MARGIN_PX) & (uv[:, :54, 1] > MARGIN_PX) & (uv[:, :54, 1] < H - MARGIN_PX)).all(1)
        ok &= ((uv[:, 54:, 0] > 5) & (uv[:, 54:, 0] < W - 5) & (uv[:, 54:, 1] > 5) & (uv[:, 54:, 1] < H - 5)).all(1)
        for x0, y0, x1, y1 in JAW_BOXES:
            ok &= ~((uv[..., 0] > x0 - 20) & (uv[..., 0] < x1 + 20) & (uv[..., 1] > y0 - 20) & (uv[..., 1] < y1 + 20)).any(1)
        dist = np.linalg.norm(C[:, :3, 3] - centre, axis=1)
        view = np.degrees(np.arccos(np.clip(np.einsum('ni,i->n', C[:, :3, 2], -np.array([0., 0., 1.])), -1, 1)))
        span = np.linalg.norm(uv[:, :54].max(1) - uv[:, :54].min(1), axis=1)
        ok &= (dist > 0.14) & (dist < 0.45) & (view > 5) & (view < 65) & (span > 420)
        sel = np.flatnonzero(ok)
        Ss.append({m: S[m][sel] for m in BODY}); uvs.append(uv[sel]); dists.append(dist[sel]); views.append(view[sel]); spans.append(span[sel])
    S = {m: np.concatenate([s[m] for s in Ss]) for m in BODY}
    uv, dist, view, span = np.concatenate(uvs), np.concatenate(dists), np.concatenate(views), np.concatenate(spans)
    idx = np.arange(len(dist))
    rng.shuffle(idx)
    clr = Clearance()
    good = []
    for i in idx[:6000]:
        q = {m: int(S[m][i]) for m in BODY}
        c = pose_clearance(clr, q, offs)
        if c >= CLEARANCE:
            good.append((i, q, c))
    print(f'{len(idx)} visible samples; {len(good)} keep {CLEARANCE*1000:.0f} mm clearance')
    if len(good) < N_VIEWS:
        raise SystemExit('not enough safe views for this placement')

    lo = {m: min(q[m] for _, q, _ in good) for m in BODY}
    hi = {m: max(q[m] for _, q, _ in good) for m in BODY}
    F = np.array([[(q[m] - lo[m]) / max(1, hi[m] - lo[m]) for m in BODY] for _, q, _ in good])
    dmin = np.linalg.norm(F - np.array([(q0[m] - lo[m]) / max(1, hi[m] - lo[m]) for m in BODY]), axis=1)
    picked = []
    for _ in range(N_VIEWS):
        j = int(np.argmax(dmin))
        picked.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(F - F[j], axis=1))
    start = {m: g0[m] for m in BODY}
    order, cur, left = [], start, list(picked)
    while left:
        j = min(left, key=lambda k: seg_time(cur, good[k][1]))
        order.append(j)
        left.remove(j)
        cur = good[j][1]
    waypoints = [start] + [good[j][1] for j in order] + [start]
    start_clear = pose_clearance(clr, start, offs)
    segments = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        worst = path_clearance(clr, a, b, offs)
        need = min(CLEARANCE, start_clear - 0.002) if (a is start or b is start) else CLEARANCE
        segments.append({'min_clearance_mm': round(worst * 1000, 1), 'required_mm': round(need * 1000, 1),
                         'seconds': round(seg_time(a, b), 2)})
    bad = [k for k, s in enumerate(segments) if s['min_clearance_mm'] < s['required_mm']]
    views = []
    for k, j in enumerate(order):
        i, q, c = good[j]
        views.append({'index': k + 1, 'split': 'heldout' if (k + 1) % 3 == 0 else 'train',
                      'target': {m: q[m] for m in BODY}, 'pose_clearance_mm': round(c * 1000, 1),
                      'camera_distance_m': round(float(dist[i]), 3), 'view_angle_deg': round(float(view[i]), 1),
                      'board_span_px': round(float(span[i])), 'predicted_corners': uv[i, :54].round(1).tolist()})
    plan = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'model': 'kinematic-model.json',
            'start_positions': q0, 'start_goals': g0, 'health': health,
            'board_centre_mm': (centre * 1000).round(1).tolist(), 'views': views, 'segments': segments,
            'segments_below_required_clearance': bad, 'total_motion_seconds': round(sum(s['seconds'] for s in segments), 1)}
    out = ROOT / ('plan-synthetic.json' if args.synthetic else 'plan.json')
    out.write_text(json.dumps(plan, indent=1))
    tiles = []
    for v in views:
        tile = np.full((270, 480, 3), 30, np.uint8)
        for x0, y0, x1, y1 in JAW_BOXES:
            cv2.rectangle(tile, (x0 // 4, y0 // 4), (x1 // 4, y1 // 4), (90, 90, 90), -1)
        for p in (np.array(v['predicted_corners']) / 4).astype(int):
            cv2.circle(tile, tuple(p), 2, (0, 220, 255), -1)
        cv2.putText(tile, f"#{v['index']} {v['split']} {v['camera_distance_m']*100:.0f}cm {v['view_angle_deg']:.0f}deg",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out.with_suffix('.jpg')), np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]))
    print(json.dumps({k: plan[k] for k in ('board_centre_mm', 'total_motion_seconds', 'segments_below_required_clearance')}))
    print('joint excursion from start (ticks):', {m: (min(v['target'][m] for v in views) - q0[m], max(v['target'][m] for v in views) - q0[m]) for m in BODY})
    print('min clearance per segment (mm):', [s['min_clearance_mm'] for s in segments])


if __name__ == '__main__':
    main()
