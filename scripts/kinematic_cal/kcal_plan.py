"""Plan diverse, table-safe checkerboard views. Reads the arm and camera; never moves anything."""
from __future__ import annotations

import json
import sys
import time

import cv2
import numpy as np

from kcal_common import (BODY, BOARD_OUTLINE, IMAGE_SIZE, OBJ, ROOT, Arm, BatchFK, Clearance, board_pose_cam,
                         inv, load_v2_camera, project, snapshot)

N_VIEWS = 24
CLEARANCE_POSE = 0.050          # m, every moving link above both the board plane and base z=0
CLEARANCE_PATH = 0.045
V_PEAK = 100.0                  # ticks/s peak joint speed (min-jerk profile)
MARGIN_PX = 120
JAW_BOXES = [(280, 620, 860, 1080), (1330, 650, 1680, 1080)]   # x0, y0, x1, y1 in the 1920x1080 image
SPAN = {'shoulder_pan': 330, 'wrist_roll': 300}
BOX = {'shoulder_lift': (1300, 2650), 'elbow_flex': (1900, 2960), 'wrist_flex': (900, 2500)}
EXEMPT = {'shoulder_link'}       # only rotates about the vertical pan axis; height never changes


def seg_time(a, b):
    return max(1.0, 1.875 * max(abs(a[n] - b[n]) for n in BODY) / V_PEAK)


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
    K, D, X = load_v2_camera()
    arm = Arm()
    try:
        q0 = arm.positions()
        g0 = arm.goals()
        health = arm.health()
    finally:
        arm.close()
    if any(h['status'] or h['torque'] != 1 for h in health.values()):
        raise SystemExit(f'motor health not nominal: {health}')
    img = snapshot(ROOT / 'plan-reference.jpg')
    board = board_pose_cam(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), K, D)
    if board is None:
        raise SystemExit('board not visible from the current pose')
    bfk = BatchFK()
    G0 = bfk({n: np.array([q0[n]]) for n in BODY})[0]
    T_bb = G0 @ X @ board['T_cam_board']
    up = -T_bb[:3, 2]
    d = float(up @ T_bb[:3, 3])
    planes = [(up, d), (np.array([0., 0., 1.]), 0.0)]

    # ---- sample
    N = 400_000
    S = {}
    for n in BODY:
        if n in SPAN:
            S[n] = rng.uniform(q0[n] - SPAN[n], q0[n] + SPAN[n], N)
        else:
            S[n] = rng.uniform(*BOX[n], N)
    S = {n: np.round(v).astype(int) for n, v in S.items()}
    G = bfk(S)
    C = G @ X                                          # base_T_camera
    Pb = np.c_[np.r_[OBJ, BOARD_OUTLINE], np.ones(len(OBJ) + 4)] @ T_bb.T          # board points in base
    Pc = np.einsum('nij,kj->nki', inv(C), Pb)[..., :3]
    uv, z = project(K, D, Pc)
    W, H = IMAGE_SIZE
    ok = (z > 0.08).all(1)
    ok &= ((uv[..., 0] > MARGIN_PX) & (uv[..., 0] < W - MARGIN_PX) & (uv[..., 1] > MARGIN_PX) & (uv[..., 1] < H - MARGIN_PX)).all(1)
    for x0, y0, x1, y1 in JAW_BOXES:
        inside = (uv[..., 0] > x0 - 60) & (uv[..., 0] < x1 + 60) & (uv[..., 1] > y0 - 60) & (uv[..., 1] < y1 + 60)
        ok &= ~inside.any(1)
    centre = (T_bb @ np.array([0.08, 0.05, 0, 1]))[:3]
    dist = np.linalg.norm(C[:, :3, 3] - centre, axis=1)
    view = np.degrees(np.arccos(np.clip(np.einsum('ni,i->n', C[:, :3, 2], -up), -1, 1)))
    span_px = np.linalg.norm(uv[:, :54].max(1) - uv[:, :54].min(1), axis=1)
    ok &= (dist > 0.16) & (dist < 0.42) & (view > 8) & (view < 60) & (span_px > 450)
    idx = np.flatnonzero(ok)
    print(f'{len(idx)} / {N} samples see the whole board clear of the jaws')

    # ---- table clearance (exact meshes) on a random subset of visible samples
    clr = Clearance()

    def clearance(q):
        worst = 1e9
        for n_, d_ in planes:
            h = clr.min_height(q, n_, d_)
            worst = min(worst, min(v for k, v in h.items() if k not in EXEMPT))
        return worst

    rng.shuffle(idx)
    good = []
    for i in idx[:8000]:
        q = {n: int(S[n][i]) for n in BODY}
        q['gripper'] = q0['gripper']
        c = clearance(q)
        if c >= CLEARANCE_POSE:
            good.append((i, q, c))
    print(f'{len(good)} of the first {min(8000, len(idx))} visible samples keep {CLEARANCE_POSE*1000:.0f} mm table clearance')

    # ---- farthest-point selection in normalised joint space (observability of joint offsets)
    lo = {n: min(q[n] for _, q, _ in good) for n in BODY}
    hi = {n: max(q[n] for _, q, _ in good) for n in BODY}
    F = np.array([[(q[n] - lo[n]) / max(1, hi[n] - lo[n]) for n in BODY] for _, q, _ in good])
    f0 = np.array([(q0[n] - lo[n]) / max(1, hi[n] - lo[n]) for n in BODY])
    picked = []
    dmin = np.linalg.norm(F - f0, axis=1)
    for _ in range(N_VIEWS):
        j = int(np.argmax(dmin))
        picked.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(F - F[j], axis=1))

    # ---- order: greedy nearest neighbour by segment time, starting and ending at the current goal pose
    start = {n: g0[n] for n in BODY}
    order, cur, left = [], start, list(picked)
    while left:
        j = min(left, key=lambda k: seg_time(cur, good[k][1]))
        order.append(j)
        left.remove(j)
        cur = good[j][1]

    # ---- path clearance along straight joint-space segments
    waypoints = [start] + [{n: good[j][1][n] for n in BODY} for j in order] + [start]
    segments = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        steps = max(2, int(np.ceil(max(abs(a[n] - b[n]) for n in BODY) / 6)))
        worst = 1e9
        for s in np.linspace(0, 1, steps + 1):
            q = {n: a[n] + (b[n] - a[n]) * s for n in BODY}
            q['gripper'] = q0['gripper']
            worst = min(worst, clearance(q))
        segments.append({'from': a, 'to': b, 'min_clearance_mm': round(worst * 1000, 1), 'seconds': round(seg_time(a, b), 2)})
    bad = [i for i, s in enumerate(segments) if s['min_clearance_mm'] < CLEARANCE_PATH * 1000]

    views = []
    for k, j in enumerate(order):
        i, q, c = good[j]
        views.append({'index': k + 1, 'split': 'heldout' if (k + 1) % 3 == 0 else 'train', 'target': {n: q[n] for n in BODY},
                      'pose_clearance_mm': round(c * 1000, 1), 'camera_distance_m': round(float(dist[i]), 3),
                      'view_angle_deg': round(float(view[i]), 1), 'board_span_px': round(float(span_px[i])),
                      'predicted_corners': uv[i, :54].round(1).tolist()})
    plan = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'start_positions': q0, 'start_goals': g0, 'health': health,
            'board_origin_base_mm': (T_bb[:3, 3] * 1000).round(1).tolist(),
            'board_tilt_deg': float(np.degrees(np.arccos(up[2]))), 'reference_pnp_rms_px': board['pnp_rms_px'],
            'views': views, 'segments': segments, 'segments_below_path_clearance': bad,
            'total_motion_seconds': round(sum(s['seconds'] for s in segments), 1),
            'limits': {'pose_clearance_mm': CLEARANCE_POSE * 1000, 'path_clearance_mm': CLEARANCE_PATH * 1000,
                       'peak_ticks_per_s': V_PEAK}}
    (ROOT / 'plan.json').write_text(json.dumps(plan, indent=1))

    # ---- contact sheet of predicted views for visual self-check
    tiles = []
    for v in views:
        tile = np.full((270, 480, 3), 30, np.uint8)
        for x0, y0, x1, y1 in JAW_BOXES:
            cv2.rectangle(tile, (x0 // 4, y0 // 4), (x1 // 4, y1 // 4), (90, 90, 90), -1)
        pts = (np.array(v['predicted_corners']) / 4).astype(int)
        for p in pts:
            cv2.circle(tile, tuple(p), 2, (0, 220, 255), -1)
        cv2.putText(tile, f"#{v['index']} {v['split']} {v['camera_distance_m']*100:.0f}cm {v['view_angle_deg']:.0f}deg",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)])
    cv2.imwrite(str(ROOT / 'plan-preview.jpg'), sheet)

    print(json.dumps({k: plan[k] for k in ('board_origin_base_mm', 'board_tilt_deg', 'total_motion_seconds', 'segments_below_path_clearance')}))
    print('start', {n: q0[n] for n in BODY})
    print(f"{'#':>3} {'split':8}" + ''.join(f'{n[:11]:>12}' for n in BODY) + '  clear_mm  dist_cm  angle')
    for v in views:
        print(f"{v['index']:>3} {v['split']:8}" + ''.join(f"{v['target'][n] - q0[n]:>+12d}" for n in BODY)
              + f"  {v['pose_clearance_mm']:8.1f}  {v['camera_distance_m']*100:7.1f}  {v['view_angle_deg']:5.1f}")
    print('min path clearance per segment (mm):', [s['min_clearance_mm'] for s in segments])
    print('joint excursion from start (ticks):', {n: (min(v['target'][n] for v in views) - q0[n], max(v['target'][n] for v in views) - q0[n]) for n in BODY})


if __name__ == '__main__':
    main()
