"""Comprehensive robot action logger & Intel RealSense D435 observer recorder.

Records every planning and execution step of kinematic grasping with:
- Full joint positions, commanded goals, tracking errors, and sag compensation
- Gripper telemetry (position, goal, signed PWM load, current mA, temperature)
- Synchronized first-person (wrist camera) and third-person (Intel RealSense D435) snapshots
- Continuous background D435 video (.mp4) + synchronized joint trajectory (.jsonl) during execution

Safety invariant: D435 capture ONLY opens OpenCV camera index 0 (verified 640x480)
and never touches the wrist camera index (1920x1080) or acquires robot.io serial lock
from the background frame grabber.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import cv2


class D435Observer:
    """Thread-safe zero-latency frame grabber & video recorder for Intel RealSense D435 (index 0, 640x480)."""

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._lock = threading.Lock()
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._latest_frame = None
        self._latest_ts: float = 0.0
        self._latest_seq: int = 0
        self._video_writer: cv2.VideoWriter | None = None
        self._stream_log_path: Path | None = None
        self._robot_ref = None
        self._disabled = False

    def start(self, run_dir: Path | None = None, robot=None) -> bool:
        with self._lock:
            if self._disabled or os.environ.get('PYTEST_CURRENT_TEST'):
                return False
            self._robot_ref = robot
            if self._cap is None or not self._cap.isOpened():
                try:
                    cap = cv2.VideoCapture(self.device_index)
                    if not cap.isOpened():
                        cap.release()
                        return False
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # Wrist camera and Mac FaceTime camera are 1920x1080; D435 RGB/IR stream on macOS is 640x480.
                    if (w, h) != (640, 480):
                        cap.release()
                        self._disabled = True
                        return False
                    self._cap = cap
                except Exception:
                    return False

            if run_dir is not None:
                run_dir.mkdir(parents=True, exist_ok=True)
                video_path = run_dir / 'd435_stream.mp4'
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'avc1'), 5.0, (640, 480))
                if not writer.isOpened():
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (640, 480))
                self._video_writer = writer
                self._stream_log_path = run_dir / 'd435_stream.jsonl'

            if not self._running:
                self._running = True
                self._thread = threading.Thread(target=self._loop, name='d435-observer', daemon=True)
                self._thread.start()
            return True

    def _loop(self) -> None:
        last_record_ts = 0.0
        # Warm up auto-exposure on start
        for _ in range(4):
            with self._lock:
                cap = self._cap
            if cap is None:
                break
            try:
                cap.read()
            except Exception:
                break
            time.sleep(0.03)

        while True:
            with self._lock:
                if not self._running or self._cap is None:
                    break
                cap = self._cap
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            now = time.time()
            if ok and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = now
                    self._latest_seq += 1
                    seq = self._latest_seq
                    writer = self._video_writer
                    log_path = self._stream_log_path
                    robot = self._robot_ref
                if writer is not None and (now - last_record_ts) >= 0.19:
                    last_record_ts = now
                    try:
                        writer.write(frame)
                        if log_path is not None:
                            pos = dict(getattr(robot, 'position', {}) or {}) if robot is not None else {}
                            goals = dict(getattr(robot, 'goals', {}) or {}) if robot is not None else {}
                            bias = dict(getattr(robot, 'bias', {}) or {}) if robot is not None else {}
                            entry = {
                                'time': now,
                                'd435_seq': seq,
                                'position': pos,
                                'goals': goals,
                                'bias': bias,
                            }
                            with log_path.open('a', encoding='utf-8') as f:
                                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    except Exception:
                        pass
            time.sleep(0.04)

    def save_snapshot(self, target_path: Path) -> dict[str, Any] | None:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            ts = self._latest_ts
            seq = self._latest_seq
        if frame is None:
            # Wait briefly for first frame if just started
            for _ in range(8):
                time.sleep(0.05)
                with self._lock:
                    if self._latest_frame is not None:
                        frame = self._latest_frame.copy()
                        ts = self._latest_ts
                        seq = self._latest_seq
                        break
        if frame is None:
            return None
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target_path), frame)
            return {
                'path': str(target_path),
                'captured_at': ts,
                'sequence': seq,
                'size': [int(frame.shape[1]), int(frame.shape[0])],
            }
        except Exception:
            return None

    def stop_recording(self) -> None:
        with self._lock:
            self._running = False
            writer = self._video_writer
            self._video_writer = None
            self._stream_log_path = None
            self._robot_ref = None
            cap = self._cap
            self._cap = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


_GLOBAL_D435 = D435Observer(device_index=0)


class ActionRunLogger:
    """Structured per-run logger writing synchronized wrist + D435 images and motor telemetry."""

    def __init__(self, runtime_dir: Path, robot, run_id: str, kind: str = 'grasp'):
        self.runtime_dir = Path(runtime_dir)
        self.robot = robot
        self.run_id = run_id
        self.kind = kind
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.runtime_dir / 'action-logs' / f'{stamp}_{kind}_{run_id}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = self.run_dir / 'events.jsonl'
        self.seq = 0
        self.d435_active = _GLOBAL_D435.start(run_dir=self.run_dir, robot=robot)

    def record(
        self,
        step_name: str,
        payload: dict[str, Any] | None = None,
        *,
        wrist_frame: dict[str, Any] | None = None,
        capture_d435: bool = True,
        gripper_telemetry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.seq += 1
        idx_str = f'{self.seq:02d}_{step_name}'
        now = time.time()
        pos = dict(getattr(self.robot, 'position', {}) or {})
        goals = dict(getattr(self.robot, 'goals', {}) or {})
        bias = dict(getattr(self.robot, 'bias', {}) or {})

        wrist_saved = None
        if isinstance(wrist_frame, dict) and wrist_frame.get('path'):
            src = Path(wrist_frame['path'])
            if src.is_file():
                dst = self.run_dir / f'{idx_str}_wrist.jpg'
                try:
                    shutil.copy2(src, dst)
                    wrist_saved = {
                        'id': wrist_frame.get('id'),
                        'path': str(dst),
                        'joints': wrist_frame.get('joints'),
                        'captured_at': wrist_frame.get('captured_at'),
                    }
                except Exception:
                    pass

        d435_saved = None
        if capture_d435 and self.d435_active:
            dst_d435 = self.run_dir / f'{idx_str}_d435.jpg'
            d435_saved = _GLOBAL_D435.save_snapshot(dst_d435)

        entry: dict[str, Any] = {
            'time': now,
            'iso_time': dt.datetime.fromtimestamp(now).isoformat(timespec='milliseconds'),
            'run_id': self.run_id,
            'seq': self.seq,
            'step': step_name,
            'robot_position': pos,
            'robot_goals': goals,
            'robot_bias': bias,
        }
        if gripper_telemetry is not None:
            entry['gripper_telemetry'] = {
                k: v for k, v in gripper_telemetry.items() if k != 'body_position'
            }
        if wrist_saved is not None:
            entry['wrist_image'] = wrist_saved
        if d435_saved is not None:
            entry['d435_image'] = d435_saved
        if payload:
            entry.update(payload)

        try:
            with self.events_file.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=float) + '\n')
        except Exception:
            pass
        return entry

    def close(self) -> None:
        try:
            _GLOBAL_D435.stop_recording()
        except Exception:
            pass
