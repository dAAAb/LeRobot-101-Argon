"""Execute the approved calibration plan produced by kcal_plan.py.

Writes ONLY Goal_Position (42) of the five body joints, min-jerk interpolated at
50 Hz. Gripper, torque enable, EEPROM, gains and speed/acceleration registers are
never written. Any abort -> goals := present positions (hold) and exit.
Stop at any time: `touch STOP` in this folder, Ctrl-C, or kill the process (the
servos then keep the last commanded goal, which is always within one 20 ms step).
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import time

import cv2
import numpy as np
from scservo_sdk import COMM_SUCCESS, SCS_HIBYTE, SCS_LOBYTE, GroupSyncWrite

from kcal_common import ASTRA, BODY, CAMERA, IDS, ROOT, Arm, board_pose_cam, load_v2_camera

CAL = json.loads((ASTRA / 'config/follower-calibration.json').read_text())
LIMIT_MARGIN = 40        # ticks inside the onboard calibrated range
TRACK_LIMIT = 120        # ticks |command - present| while moving
LOAD_LIMIT = 750         # 0.1% PWM, 3 consecutive samples (0.3 s)
TEMP_LIMIT = 55          # deg C
GAP_LIMIT = 90           # ticks |goal - present| before starting a segment
RATE_HZ = 50
V_PEAK = 100.0           # ticks/s
STOP_FILE = ROOT / 'STOP'


class Abort(RuntimeError):
    pass


class Runner:
    def __init__(self, session):
        self.session = session
        self.arm = Arm()
        self.writer = GroupSyncWrite(self.arm.port, self.arm.ph, 42, 2)
        self.log = open(session / 'motion.jsonl', 'a')
        self.stop_requested = False
        signal.signal(signal.SIGINT, self._signal)
        signal.signal(signal.SIGTERM, self._signal)

    def _signal(self, *_):
        self.stop_requested = True

    def check_stop(self):
        if self.stop_requested or STOP_FILE.exists():
            raise Abort('stop requested')

    def write_goals(self, goals, check_limits=True):
        self.writer.clearParam()
        for n in BODY:
            v = int(round(goals[n]))
            lo, hi = (CAL[n]['range_min'] + LIMIT_MARGIN, CAL[n]['range_max'] - LIMIT_MARGIN) if check_limits else (0, 4095)
            if not lo <= v <= hi:
                raise Abort(f'{n} goal {v} outside [{lo}, {hi}]')
            self.writer.addParam(IDS[n], [SCS_LOBYTE(v), SCS_HIBYTE(v)])
        if self.writer.txPacket() != COMM_SUCCESS:
            raise Abort('sync write failed')

    def hold(self):
        try:
            pos = self.arm.positions()
            self.write_goals({n: pos[n] for n in BODY}, check_limits=False)
            return {n: pos[n] for n in BODY}
        except Exception as exc:  # report, never mask the original abort
            return {'error': str(exc)}

    def health(self):
        for n in BODY:
            status, torque = self.arm.read1(n, 65), self.arm.read1(n, 40)
            if status:
                raise Abort(f'{n} status {status:#04x}')
            if torque != 1:
                raise Abort(f'{n} torque is off (not re-enabling)')

    def move(self, target, label):
        self.check_stop()
        self.health()
        goals, pos = self.arm.goals(), self.arm.positions()
        start = {n: goals[n] for n in BODY}
        gaps = {n: start[n] - pos[n] for n in BODY if abs(start[n] - pos[n]) > GAP_LIMIT}
        if gaps:
            raise Abort(f'goal/position gap before move: {gaps}')
        for n in BODY:  # validate the endpoint before any write
            lo, hi = CAL[n]['range_min'] + LIMIT_MARGIN, CAL[n]['range_max'] - LIMIT_MARGIN
            if not lo <= target[n] <= hi:
                raise Abort(f'{n} target {target[n]} outside [{lo}, {hi}]')
        duration = max(1.0, 1.875 * max(abs(target[n] - start[n]) for n in BODY) / V_PEAK)
        t0, k = time.monotonic(), 0
        strikes = {n: 0 for n in BODY}
        peak = {n: 0.0 for n in BODY}
        while True:
            self.check_stop()
            s = min(1.0, (time.monotonic() - t0) / duration)
            m = s ** 3 * (10 - 15 * s + 6 * s * s)
            cmd = {n: start[n] + (target[n] - start[n]) * m for n in BODY}
            self.write_goals(cmd)
            pos = self.arm.positions()
            err = {n: cmd[n] - pos[n] for n in BODY}
            for n in BODY:
                peak[n] = max(peak[n], abs(err[n]))
            bad = {n: round(err[n]) for n in BODY if abs(err[n]) > TRACK_LIMIT}
            if bad:
                raise Abort(f'tracking error {bad}')
            rec = {'label': label, 't': round(time.monotonic() - t0, 4), 'cmd': {n: round(cmd[n], 1) for n in BODY},
                   'pos': {n: pos[n] for n in BODY}}
            if k % 5 == 0:
                loads = self.arm.loads()
                rec['load'] = {n: loads[n] for n in BODY}
                for n in BODY:
                    strikes[n] = strikes[n] + 1 if abs(loads[n]) > LOAD_LIMIT else 0
                if any(v >= 3 for v in strikes.values()):
                    raise Abort(f'sustained load {rec["load"]}')
            if k % 12 == 0:
                self.health()
            if k % 50 == 0:
                for n in BODY:
                    temp = self.arm.read1(n, 63)
                    if temp > TEMP_LIMIT:
                        raise Abort(f'{n} temperature {temp} C')
            self.log.write(json.dumps(rec) + '\n')
            if s >= 1:
                break
            k += 1
            delay = t0 + k / RATE_HZ - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return {'seconds': round(duration, 2), 'peak_tracking_ticks': {n: round(v, 1) for n, v in peak.items()}}

    def settle(self, timeout=4.0):
        hist, t0 = [], time.monotonic()
        while True:
            self.check_stop()
            hist.append((time.monotonic(), self.arm.positions()))
            recent = [p for t, p in hist if t >= hist[-1][0] - 0.5]
            if (hist[-1][0] - hist[0][0] >= 0.5 and len(recent) >= 8 and
                    all(max(p[n] for p in recent) - min(p[n] for p in recent) <= 2 for n in BODY)):
                return hist[-1][1], round(time.monotonic() - t0, 2), True
            if time.monotonic() - t0 > timeout:
                return hist[-1][1], round(timeout, 2), False
            time.sleep(0.04)

    def capture(self, live, name):
        after_wall = time.time()
        before = [self.arm.positions() for _ in range(3)]
        data, deadline = None, time.time() + 4
        while time.time() < deadline:
            self.check_stop()
            try:
                if live.stat().st_mtime > after_wall + 0.35:
                    blob = live.read_bytes()
                    if blob[-2:] == b'\xff\xd9' and cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_GRAYSCALE) is not None:
                        data = blob
                        break
            except FileNotFoundError:
                pass
            time.sleep(0.05)
        if data is None:
            raise Abort('no fresh camera frame')
        after = [self.arm.positions() for _ in range(3)]
        samples = before + after
        span = max(max(s[n] for s in samples) - min(s[n] for s in samples) for n in BODY)
        path = self.session / f'{name}.jpg'
        path.write_bytes(data)
        joints = {n: float(np.mean([s[n] for s in samples])) for n in IDS}
        return path, joints, span


def start_recorder(session):
    live = session / 'live.jpg'
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'avfoundation', '-pixel_format', 'nv12', '-framerate', '30',
           '-video_size', '1920x1080', '-i', f'{CAMERA}:none',
           '-map', '0:v', '-vf', 'fps=6', '-q:v', '2', '-update', '1', '-y', str(live),
           '-map', '0:v', '-vf', 'fps=30', '-c:v', 'h264_videotoolbox', '-b:v', '8M', '-y', str(session / 'wrist.mp4')]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=open(session / 'ffmpeg.log', 'w'))
    t0 = time.time()
    while time.time() - t0 < 12:
        if live.exists() and live.stat().st_size > 20000:
            return proc, live
        if proc.poll() is not None:
            raise RuntimeError('camera recorder exited; see ffmpeg.log')
        time.sleep(0.1)
    stop_recorder(proc)
    raise RuntimeError('camera recorder produced no frames')


def stop_recorder(proc):
    if proc.poll() is not None:
        return
    try:
        proc.stdin.write(b'q')
        proc.stdin.flush()
        proc.wait(timeout=6)
        return
    except Exception:
        pass
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true', help='only a +30 tick wrist-roll move and back')
    ap.add_argument('--dry-run', action='store_true', help='validate plan and print schedule; no writes, no camera')
    args = ap.parse_args()
    plan = json.loads((ROOT / 'plan.json').read_text())
    K, D, _ = load_v2_camera()

    if args.dry_run:
        arm = Arm()
        try:
            pos, goals = arm.positions(), arm.goals()
        finally:
            arm.close()
        drift = {n: pos[n] - plan['start_positions'][n] for n in BODY}
        print('drift since planning (ticks):', drift)
        prev, total = {n: goals[n] for n in BODY}, 0.0
        for v in plan['views'] + [{'index': 'return', 'target': {n: goals[n] for n in BODY}}]:
            t = v['target']
            for n in BODY:
                lo, hi = CAL[n]['range_min'] + LIMIT_MARGIN, CAL[n]['range_max'] - LIMIT_MARGIN
                assert lo <= t[n] <= hi, (v['index'], n, t[n])
            dur = max(1.0, 1.875 * max(abs(t[n] - prev[n]) for n in BODY) / V_PEAK)
            total += dur + 1.5
            prev = t
        print(f'all {len(plan["views"])} targets inside limits; estimated run time {total/60:.1f} min')
        return

    STOP_FILE.unlink(missing_ok=True)
    session = ROOT / ('session-' + time.strftime('%Y%m%d-%H%M%S') + ('-test' if args.test else ''))
    session.mkdir()
    (session / 'plan.json').write_text((ROOT / 'plan.json').read_text())
    runner = Runner(session)
    summary = {'session': session.name, 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'views': []}
    proc = None
    try:
        pos, goals = runner.arm.positions(), runner.arm.goals()
        drift = max(abs(pos[n] - plan['start_positions'][n]) for n in BODY)
        if drift > 15:
            raise Abort(f'arm moved {drift} ticks since planning; re-plan first')
        runner.health()
        start_goals = {n: goals[n] for n in BODY}
        summary['start_goals'] = start_goals
        if args.test:
            seq = [('test_roll_plus30', {**start_goals, 'wrist_roll': start_goals['wrist_roll'] + 30}, 'test'),
                   ('test_return', start_goals, 'test')]
        else:
            seq = [(f"view_{v['index']:02d}", v['target'], v['split']) for v in plan['views']] + [('view_99_return', start_goals, 'check')]
        proc, live = start_recorder(session)
        items = [('view_00_start', None, 'check')] + seq
        for name, target, split in items:
            move = runner.move(target, name) if target is not None else None
            settled, settle_s, stable = runner.settle()
            path, joints, span = runner.capture(live, name)
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            board = board_pose_cam(gray, K, D)
            rec = {'name': name, 'split': split, 'image': path.name, 'joints': joints, 'capture_joint_span': span,
                   'settle_s': settle_s, 'settled': stable, 'move': move,
                   'target': target, 'steady_error': {n: round(target[n] - joints[n], 1) for n in BODY} if target else None,
                   'detected': board is not None}
            if board is not None:
                rec.update(corners=board['corners'].round(3).tolist(), pnp_rms_px=round(board['pnp_rms_px'], 3),
                           ordering=board['ordering'], ambiguous_pattern=bool(board['ambiguous']))
            summary['views'].append(rec)
            with open(session / 'views.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"{name:16} settled={stable!s:5} {settle_s:4.1f}s  span={span}  board={'yes' if board else 'NO '}"
                  + (f" pnp={board['pnp_rms_px']:.2f}px{' AMBIGUOUS' if board['ambiguous'] else ''}" if board else '')
                  + (f"  peak_track={max(move['peak_tracking_ticks'].values()):.0f}t" if move else ''), flush=True)
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
    print(json.dumps({'status': summary['status'], 'reason': summary.get('reason'), 'session': session.name,
                      'detected': sum(v['detected'] for v in summary['views']), 'captured': len(summary['views'])}))


if __name__ == '__main__':
    main()
