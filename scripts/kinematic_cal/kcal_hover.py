"""Acceptance test: park the jaw tips a given height above one checkerboard corner. MOVES THE ARM (unless --plan-only).

1. Locate the board from a fresh wrist image at the current pose (see, then reach).
2. IK (calibrated model) for the TCP = midpoint of the two CAD jaw-tip lows, approach as vertical as reachable.
3. Descend through waypoints (default 80/40/25/15/10 mm). After each settle, measured joints -> FK height;
   the final waypoint is corrected closed-loop for servo steady-state error (max 3 corrections).
4. Evidence at the hover: measured-FK gap, wrist-camera PnP gap (independent of joint offsets/sag), D435 frame.

Guards: everything in kcal_run.Runner (tracking, load, status, temperature, limits, STOP file) plus
real-jaw-angle mesh clearance on every segment and an abort if the measured TCP gap drops below FLOOR.
The gripper is never commanded.
  python kcal_hover.py --plan-only            # locate + plan, print, write hover-plan.json; no motion
  python kcal_hover.py                        # execute the plan written by --plan-only (re-checks drift)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from kcal_common import BODY, OBJ, PATTERN, ROOT, Arm, BatchFK, board_pose_cam, snapshot
from kcal_model import load_model
from kcal_run import CAL, LIMIT_MARGIN, Abort, Runner, start_recorder, stop_recorder
from server.local_cartesian import default_model

TIP_A = np.array([-0.0107, -0.0002, -0.1044])   # fixed jaw lowest region (gripper_link, CAD)
WAYPOINTS_MM = [80, 40, 25, 15]
FLOOR = 0.005                                     # abort if the measured TCP gap is ever below this
MESH_FLOOR = 0.003                                # real-jaw mesh must stay this far above the table plane
MAX_TILT = 40.0                                   # deg, approach axis vs table normal
PLAN = ROOT / 'hover-plan.json'
D435 = 'Intel(R) RealSense(TM) Depth Camera 435i Depth'


class Geometry:
    def __init__(self, model, gripper_raw):
        self.m, self.offs, self.fk = model, model['offsets_ticks'], BatchFK()
        tilt = np.array(model['raw']['table_tilt_T'])
        self.n, self.d = tilt[:3, 2], float(model['table_z'])
        self.cad = default_model()
        ang = (gripper_raw - 2047) * 2 * np.pi / 4096
        c, s = np.cos(ang), np.sin(ang)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        jo = self.cad.jaw_origin
        self.jaw_pts = (self.cad.jaw @ Rz.T) @ jo[:3, :3].T + jo[:3, 3]      # moving jaw in gripper_link
        tip_b = self.jaw_pts[self.jaw_pts[:, 2] < self.jaw_pts[:, 2].min() + 0.001].mean(0)
        self.tip_a, self.tip_b = TIP_A, tip_b
        self.tcp_local = (TIP_A + tip_b) / 2
        self.gripper_raw = gripper_raw

    def G(self, q):
        return self.fk({n: np.array([float(q[n])]) for n in BODY}, self.offs)[0]

    def tcp(self, q):
        G = self.G(q)
        return G[:3, :3] @ self.tcp_local + G[:3, 3], -G[:3, 2]

    def mesh_min(self, q):
        """Lowest robot point above the table plane with the REAL jaw angle (base/shoulder exempt)."""
        full = {n: float(q[n]) + self.offs.get(n, 0.0) for n in BODY}
        full['gripper'] = self.gripper_raw
        T = self.cad.fk.transforms(full)
        low = {}
        for link, v in self.cad.vertices.items():
            if link in ('base_link', 'shoulder_link'):
                continue
            pts = v @ T[link][:3, :3].T + T[link][:3, 3]
            low[link] = float((pts @ self.n).min() - self.d)
        J = self.jaw_pts @ T['gripper_link'][:3, :3].T + T['gripper_link'][:3, 3]
        low['moving_jaw'] = float((J @ self.n).min() - self.d)
        return min(low.values()), min(low, key=low.get)


def limits():
    return {n: (CAL[n]['range_min'] + LIMIT_MARGIN + 20, CAL[n]['range_max'] - LIMIT_MARGIN - 20) for n in BODY}


def ik(geo, target, q_ref):
    lim = limits()
    lo, hi = np.array([lim[n][0] for n in BODY]), np.array([lim[n][1] for n in BODY])

    def res(x):
        q = dict(zip(BODY, x))
        p, a = geo.tcp(q)
        G = geo.G(q)
        level = geo.n @ (G[:3, :3] @ (geo.tip_a - geo.tip_b)) * 1000     # both jaw tips at the same height
        tilt = np.degrees(np.arccos(np.clip(a @ -geo.n, -1, 1)))
        return np.r_[(p - target) * 1000, level, 0.15 * tilt, 3.0 * max(0.0, tilt - MAX_TILT),
                     0.002 * (x - np.array([q_ref[n] for n in BODY]))]

    best = None
    for dpan in (0, -150, 150):
        x0 = np.clip(np.array([q_ref[n] + (dpan if n == 'shoulder_pan' else 0) for n in BODY], float), lo, hi)
        sol = least_squares(res, x0, bounds=(lo, hi), x_scale=100.0)
        if best is None or sol.cost < best.cost:
            best = sol
    q = {n: float(v) for n, v in zip(BODY, best.x)}
    p, a = geo.tcp(q)
    return q, float(np.linalg.norm(p - target) * 1000), float(np.degrees(np.arccos(np.clip(a @ -geo.n, -1, 1))))


def seg_check(geo, a, b, steps=40):
    worst = (9.9, '')
    for s in np.linspace(0, 1, steps + 1):
        q = {n: a[n] + (b[n] - a[n]) * s for n in BODY}
        h = geo.mesh_min(q)
        if h[0] < worst[0]:
            worst = h
    return worst


def locate_board(model, q, path):
    img = snapshot(path)
    b = board_pose_cam(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), model['K'], model['D'])
    if b is None or b['ambiguous']:
        raise SystemExit('board not usable from the current pose; move to a pose that sees the whole board')
    G = BatchFK()({n: np.array([float(q[n])]) for n in BODY}, model['offsets_ticks'])[0]
    return G @ model['X'] @ b['T_cam_board'], b['pnp_rms_px']


def gap_of(geo, q, corner):
    p, _ = geo.tcp(q)
    return float(geo.n @ (p - corner)), float(np.linalg.norm((p - corner) - geo.n * (geo.n @ (p - corner))))


def plan(args):
    model = load_model()
    arm = Arm()
    try:
        pos, goals, health = arm.positions(), arm.goals(), arm.health()
    finally:
        arm.close()
    if any(h['status'] or h['torque'] != 1 for h in health.values()):
        raise SystemExit(f'motor health not nominal: {health}')
    geo = Geometry(model, pos['gripper'])
    T_bb, pnp = locate_board(model, pos, ROOT / 'hover-locate.jpg')
    corners = (np.c_[OBJ, np.ones(len(OBJ))] @ T_bb.T)[:, :3]
    start = {n: goals[n] for n in BODY}
    if args.corner:
        idx = [args.corner[1] * PATTERN[0] + args.corner[0]]
    else:
        idx = range(len(OBJ))
    options = []
    for i in idx:
        q, err, tilt = ik(geo, corners[i] + geo.n * args.height / 1000, start)
        options.append((tilt + 5 * err, i, q, err, tilt))
    options.sort(key=lambda o: o[0])
    _, i, q_final, err, tilt = options[0]
    corner = corners[i]
    wps, prev = [], start
    for h in [w for w in WAYPOINTS_MM if w > args.height] + [args.height]:
        q, e, t = ik(geo, corner + geo.n * h / 1000, q_final if h < 30 else prev)
        q = {n: int(round(v)) for n, v in q.items()}
        low, link = seg_check(geo, prev, q)
        wps.append({'height_mm': h, 'target': q, 'ik_err_mm': round(e, 2), 'tilt_deg': round(t, 1),
                    'segment_min_mesh_mm': round(low * 1000, 1), 'segment_min_link': link,
                    'max_joint_step_ticks': int(max(abs(q[n] - prev[n]) for n in BODY))})
        prev = q
    corner_vs_table = float(geo.n @ corner - geo.d)
    G = geo.G(wps[-1]['target'])
    tip_dh = float(geo.n @ (G[:3, :3] @ (geo.tip_a - geo.tip_b)))
    out = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'start_positions': pos, 'start_goals': start,
           'board_T_base': T_bb.tolist(), 'locate_pnp_px': round(pnp, 3), 'corner_index': [int(i % PATTERN[0]), int(i // PATTERN[0])],
           'corner_base_mm': (corner * 1000).round(1).tolist(), 'tcp_local_mm': (geo.tcp_local * 1000).round(2).tolist(),
           'gripper_raw': pos['gripper'], 'height_mm': args.height, 'waypoints': wps,
           'corner_vs_table_plane_mm': round(corner_vs_table * 1000, 2), 'final_tip_height_difference_mm': round(tip_dh * 1000, 2)}
    PLAN.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('corner_index', 'corner_base_mm', 'tcp_local_mm', 'locate_pnp_px', 'corner_vs_table_plane_mm', 'final_tip_height_difference_mm')}))
    for w in wps:
        print(f"  -> {w['height_mm']:5.1f} mm  tilt {w['tilt_deg']:4.1f} deg  ik_err {w['ik_err_mm']} mm  "
              f"max step {w['max_joint_step_ticks']:4d} t  segment lowest mesh {w['segment_min_mesh_mm']:6.1f} mm ({w['segment_min_link']})  {w['target']}")
    bad = [w for w in wps if w['segment_min_mesh_mm'] < MESH_FLOOR * 1000 or w['ik_err_mm'] > 1.0]
    if bad:
        print('PLAN REJECTED: clearance or IK error out of bounds')


def d435_frame(path):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'avfoundation', '-pixel_format', 'uyvy422',
                    '-framerate', '30', '-video_size', '640x480', '-i', f'{D435}:none', '-ss', '1', '-frames:v', '1',
                    '-y', str(path)], timeout=30)


def execute(args):
    p = json.loads(PLAN.read_text())
    model = load_model()
    session = ROOT / ('hover-' + time.strftime('%Y%m%d-%H%M%S'))
    session.mkdir()
    (session / 'hover-plan.json').write_text(json.dumps(p, indent=1))
    runner = Runner(session)
    geo = Geometry(model, p['gripper_raw'])
    T_bb = np.array(p['board_T_base'])
    i = p['corner_index'][1] * PATTERN[0] + p['corner_index'][0]
    corner = (T_bb @ np.r_[OBJ[i], 1])[:3]
    summary = {'session': session.name, 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': []}
    proc = None
    try:
        pos, goals = runner.arm.positions(), runner.arm.goals()
        drift = max(abs(pos[n] - p['start_positions'][n]) for n in BODY)
        if drift > 15 or abs(pos['gripper'] - p['gripper_raw']) > 15:
            raise Abort(f'arm moved {drift} ticks since planning; re-plan first')
        runner.health()
        proc, live = start_recorder(session)
        final = p['waypoints'][-1]
        bias = {n: 0 for n in BODY}           # servo steady-state error (command - measured), fed forward
        for k, w in enumerate(p['waypoints']):
            target = {n: w['target'][n] + bias[n] for n in BODY}
            tries = 4 if w is final else 1
            for t in range(tries):
                move = runner.move(target, f"wp{k}_{w['height_mm']}mm_{t}")
                settled, settle_s, stable = runner.settle()
                gap, lateral = gap_of(geo, settled, corner)
                low, link = geo.mesh_min(settled)
                rec = {'waypoint_mm': w['height_mm'], 'try': t, 'commanded': target, 'measured': {n: settled[n] for n in BODY},
                       'measured_gap_mm': round(gap * 1000, 2), 'lateral_mm': round(lateral * 1000, 2),
                       'mesh_min_mm': round(low * 1000, 1), 'mesh_min_link': link, 'settled': stable, 'move': move}
                summary['steps'].append(rec)
                print(f"wp {w['height_mm']:5.1f} mm try {t}: measured gap {gap*1000:6.2f} mm  lateral {lateral*1000:5.2f} mm  "
                      f"lowest mesh {low*1000:5.1f} mm ({link})  settled={stable}", flush=True)
                bias = {n: int(round(target[n] - settled[n])) for n in BODY}
                rec['bias_ticks'] = bias
                if gap < FLOOR or low < MESH_FLOOR:
                    raise Abort(f'measured gap {gap*1000:.1f} mm / mesh {low*1000:.1f} mm below floor')
                if w is not final or abs(gap * 1000 - w['height_mm']) <= 1.0 and lateral <= 0.0015:
                    break
                # closed-loop: re-solve IK for the residual, applied to the command (servo steady-state error)
                want = corner + geo.n * w['height_mm'] / 1000
                q_meas_target, _, _ = ik(geo, want, settled)
                target = {n: int(round(target[n] + (q_meas_target[n] - settled[n]))) for n in BODY}
                if max(abs(target[n] - w['target'][n]) for n in BODY) > 60:
                    raise Abort('correction larger than 60 ticks; stopping')
        path, joints, span = runner.capture(live, 'hover_wrist')
        d435_frame(session / 'hover_d435.jpg')
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        b = board_pose_cam(gray, model['K'], model['D'])
        ev = {'image': path.name, 'joint_span': span}
        gap, lateral = gap_of(geo, joints, corner)
        ev.update(fk_gap_mm=round(gap * 1000, 2), fk_lateral_mm=round(lateral * 1000, 2))
        if b is not None:
            G = geo.G(joints)
            X = model['X']
            T_cb = b['T_cam_board']
            c_cam = (T_cb @ np.r_[OBJ[i], 1])[:3]
            tcp_cam = (np.linalg.inv(X) @ np.r_[geo.tcp_local, 1])[:3]
            n_cam = (np.linalg.inv(G @ X)[:3, :3] @ geo.n)
            v = tcp_cam - c_cam
            ev.update(camera_gap_mm=round(float(n_cam @ v) * 1000, 2),
                      camera_lateral_mm=round(float(np.linalg.norm(v - n_cam * (n_cam @ v))) * 1000, 2),
                      camera_pnp_px=round(b['pnp_rms_px'], 3), camera_ambiguous=bool(b['ambiguous']))
        summary['evidence'] = ev
        print('EVIDENCE', json.dumps(ev), flush=True)
        if not args.no_return:
            for k, w in reversed(list(enumerate(p['waypoints'][:-1]))):
                runner.move({n: w['target'][n] + bias[n] for n in BODY}, f'up_{w["height_mm"]}mm')
            runner.move(p['start_goals'], 'return')
            runner.settle()
        summary['status'] = 'completed'
    except Abort as exc:
        summary.update(status='aborted', reason=str(exc), held=runner.hold())
        print('ABORTED:', exc, '-> holding', summary['held'], flush=True)
    except Exception as exc:
        summary.update(status='error', reason=repr(exc), held=runner.hold())
        print('ERROR:', repr(exc), '-> holding', summary['held'], flush=True)
    finally:
        if proc is not None:
            stop_recorder(proc)
        summary['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        (session / 'summary.json').write_text(json.dumps(summary, indent=1))
        runner.log.close()
        runner.arm.close()
    print(json.dumps({'status': summary['status'], 'reason': summary.get('reason'), 'session': session.name}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--height', type=float, default=10.0, help='mm above the corner')
    ap.add_argument('--corner', nargs=2, type=int, metavar=('COL', 'ROW'), help='inner corner (0..8, 0..5); default: easiest')
    ap.add_argument('--no-return', action='store_true', help='stay at the hover pose')
    a = ap.parse_args()
    plan(a) if a.plan_only else execute(a)
