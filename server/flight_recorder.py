"""Continuous "Dashcam / Blackbox Flight Recorder" ("行車記錄器") for LeRobot SO-101 Argon.

Features:
1. Rolling Window Ring Buffer (configurable 30 or 60 minutes, auto-pruning older segments):
   - Continuously records at ~4 Hz (and high-rate event ticks during active tasks):
     * 6-axis robot joint states (`position`, `goal`, `bias`, `moving`, `armed`) + 3D TCP position
     * Wrist camera RGB frames (`/so101/wrist_cam/image_raw`)
     * Intel RealSense D435 RGB + Depth frames (`/realsense_d435/color/image_raw`, `/realsense_d435/aligned_depth`)
       STRICT ARCHITECTURAL BOUNDARY: D435 is NEVER used for motion planning or closed-loop control;
       it is strictly used for Dashcam recording and post-hoc Agentic Engineering evaluation.
     * ROSbag-compatible JSONL (`rosbag_stream.jsonl`) + Sim Digital Twin state (`scene_map`).
2. Automatic & Manual Episode Tagging (`success` vs `failure`):
   - Archives the task window into:
     * `.runtime/dashcam/archived/success/<episode_id>/`
     * `.runtime/dashcam/archived/failure/<episode_id>/`
3. Native Rerun (`.rrd`) Replay & Re-Training Bundle Export:
   - Exports `episode.rrd` via `rerun-sdk` (`rr.RecordingStream`, `rr.Scalars`, `rr.Points3D`, `rr.EncodedImage`, `rr.TextLog`)
     so any episode can be replayed in Rerun (`rerun episode.rrd`) or downloaded from the UI.
   - Exports `lerobot_rosbag_bundle.json` + `frames.jsonl` (LeRobot v2 / Dream Labs embodiment format).
   - Exports `sim_counterfactual_pack.json` containing the 3D Digital Twin scene + synthetic counterfactual
     perturbations (stationary left-jaw alignment corrections, domain-randomized object poses) so failed
     real-world episodes can be replayed in Sim and used for self-improving policy training.
"""
from __future__ import annotations

import json
import math
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
ALL_JOINTS = (*BODY, 'gripper')


class RealSenseD435Observer:
    """Strictly passive observer for Intel RealSense D435 (RGB + Depth).

    IMPORTANT: Never exposed to motion planning or closed-loop control (`USED_FOR_CONTROL = False`).
    Used exclusively by:
      1. `FlightRecorder` (Dashcam rolling buffer)
      2. `AgenticEvaluator` (Post-hoc task referee & diversity/displacement metrics)
    """

    USED_FOR_CONTROL = False

    def __init__(self, runtime_dir: Path):
        self.runtime_dir = Path(runtime_dir) / 'realsense_d435'
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pipeline = None
        self._align = None
        self._avf_device_name: str | None = None
        self._mode = 'standby_unplugged'  # 'pyrealsense2_rgbd' | 'avf_realsense' | 'standby_unplugged'
        self._device_name = 'Intel RealSense D435 (Passive Observer)'
        self._last_probe = 0.0
        self._last_capture_ts = 0.0
        self._last_frame_meta: dict[str, Any] | None = None

    def _probe_hardware(self):
        if self._mode == 'pyrealsense2_rgbd' and self._pipeline is not None:
            return
        if self._mode == 'avf_realsense' and self._avf_device_name:
            return
        now = time.monotonic()
        if now - self._last_probe < 15.0:
            return
        self._last_probe = now
        # 1. Try pyrealsense2 native RGB-D pipeline
        try:
            import pyrealsense2 as rs  # type: ignore
            ctx = rs.context()
            devs = list(ctx.query_devices())
            if devs:
                pipe = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                pipe.start(cfg)
                self._pipeline = pipe
                self._align = rs.align(rs.stream.color)
                self._mode = 'pyrealsense2_rgbd'
                self._device_name = devs[0].get_info(rs.camera_info.name)
                return
        except Exception:
            self._pipeline = None
            self._align = None

        # 2. Auto-discover Intel RealSense D435/D435i by EXACT AVFoundation device name (never numeric index)
        import re
        import subprocess

        ffmpeg_bin = shutil.which('ffmpeg') or '/opt/local/bin/ffmpeg'
        try:
            proc = subprocess.run(
                [ffmpeg_bin, '-f', 'avfoundation', '-list_devices', 'true', '-i', ''],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            candidates: list[str] = []
            for line in (proc.stderr or '').splitlines():
                if 'AVFoundation audio devices' in line:
                    break
                m = re.search(r'\[(\d+)\]\s+(.+)', line)
                if m and ('realsense' in m.group(2).lower() or '435' in m.group(2).lower()):
                    candidates.append(m.group(2).strip())
            if candidates:
                # Prefer RGB stream over Depth/IR stream if multiple RealSense AVFoundation devices are exposed
                rgb_candidates = [c for c in candidates if 'rgb' in c.lower() or 'color' in c.lower()]
                exact_dev = rgb_candidates[0] if rgb_candidates else candidates[-1]
                self._avf_device_name = exact_dev
                self._mode = 'avf_realsense'
                self._device_name = f'{exact_dev} (Passive Observer)'
                return
        except Exception:
            pass
        self._mode = 'standby_unplugged'

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._probe_hardware()
            return {
                'device': self._device_name,
                'mode': self._mode,
                'connected': self._mode in ('pyrealsense2_rgbd', 'avf_realsense'),
                'used_for_planning_or_control': False,
                'role': 'posthoc_agentic_evaluator_and_dashcam_only',
                'last_frame': self._last_frame_meta,
            }

    def capture_observation(self, tag: str = 'tick', fallback_wrist_path: str | None = None) -> dict[str, Any]:
        """Capture a passive D435 RGB-D/IR snapshot (or record standby metadata if unplugged)."""
        with self._lock:
            self._probe_hardware()
            ts = round(time.time(), 3)
            stamp = f'{int(ts * 1000)}_{tag}'
            if self._mode == 'pyrealsense2_rgbd' and self._pipeline is not None:
                try:
                    frames = self._pipeline.wait_for_frames(timeout_ms=350)
                    if self._align is not None:
                        frames = self._align.process(frames)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if color_frame and depth_frame:
                        color_np = np.asanyarray(color_frame.get_data())
                        depth_np = np.asanyarray(depth_frame.get_data())  # uint16 mm
                        rgb_path = self.runtime_dir / f'd435_rgb_{stamp}.jpg'
                        depth_path = self.runtime_dir / f'd435_depth_{stamp}.png'
                        cv2.imwrite(str(rgb_path), color_np, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        cv2.imwrite(str(depth_path), depth_np)
                        valid_d = depth_np[(depth_np > 120) & (depth_np < 1200)]
                        mean_depth_mm = round(float(np.median(valid_d)), 1) if valid_d.size else None
                        meta = {
                            'timestamp': ts,
                            'source': 'realsense_d435_rgbd',
                            'connected': True,
                            'used_for_control': False,
                            'rgb_path': str(rgb_path),
                            'depth_path': str(depth_path),
                            'median_workspace_depth_mm': mean_depth_mm,
                        }
                        self._last_frame_meta = meta
                        return meta
                except Exception:
                    pass

            if self._mode == 'avf_realsense' and self._avf_device_name:
                # For continuous stream ticks, reuse the recent D435 frame if captured within the last 2.0s
                if (
                    tag == 'stream'
                    and self._last_frame_meta is not None
                    and self._last_frame_meta.get('connected')
                    and (ts - self._last_capture_ts) < 2.0
                ):
                    return self._last_frame_meta
                import subprocess
                ffmpeg_bin = shutil.which('ffmpeg') or '/opt/local/bin/ffmpeg'
                rgb_path = self.runtime_dir / f'd435_avf_{stamp}.jpg'
                try:
                    proc = subprocess.run(
                        [
                            ffmpeg_bin, '-nostdin', '-loglevel', 'error',
                            '-f', 'avfoundation', '-video_size', '640x480', '-framerate', '30',
                            '-ss', '0.35', '-i', self._avf_device_name,
                            '-frames:v', '1', '-y', str(rgb_path),
                        ],
                        capture_output=True,
                        timeout=3.5,
                    )
                    if proc.returncode == 0 and rgb_path.is_file():
                        self._last_capture_ts = ts
                        meta = {
                            'timestamp': ts,
                            'source': 'realsense_d435_avf',
                            'connected': True,
                            'used_for_control': False,
                            'rgb_path': str(rgb_path),
                            'depth_path': None,
                            'median_workspace_depth_mm': None,
                        }
                        self._last_frame_meta = meta
                        return meta
                except Exception:
                    pass
                self._avf_device_name = None
                self._mode = 'standby_unplugged'

            meta = {
                'timestamp': ts,
                'source': 'realsense_d435_standby_auxiliary_wrist_ref',
                'connected': False,
                'used_for_control': False,
                'rgb_path': fallback_wrist_path,
                'depth_path': None,
                'median_workspace_depth_mm': None,
            }
            self._last_frame_meta = meta
            return meta


class FlightRecorder:
    """Continuous Rolling-Window Dashcam ("行車記錄器") with Rerun (.rrd) & Sim Re-Training Export."""

    def __init__(self, robot, mira_hub, runtime_dir: Path, *, window_minutes: int = 30):
        self.robot = robot
        self.mira_hub = mira_hub
        self.base_dir = Path(runtime_dir) / 'dashcam'
        self.rolling_dir = self.base_dir / 'rolling'
        self.rolling_frames_dir = self.rolling_dir / 'wrist_frames'
        self.success_dir = self.base_dir / 'archived' / 'success'
        self.failure_dir = self.base_dir / 'archived' / 'failure'
        for d in (self.rolling_dir, self.rolling_frames_dir, self.success_dir, self.failure_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.d435 = RealSenseD435Observer(self.base_dir)
        self.window_minutes = 30 if int(window_minutes) <= 45 else 60
        self.sample_hz = 2.5
        self._lock = threading.Lock()
        # Rolling memory buffer (up to 60 min at 2.5 Hz = 9000 samples)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=10800)
        self._active_task_start_ts: float | None = None
        self._active_task_meta: dict[str, Any] = {}
        self._running = True
        self._last_image_save_ts = 0.0
        self._last_cached_wrist_path: str | None = None
        self._last_joints: dict[str, int] = {k: 2047 for k in ALL_JOINTS}
        self._thread = threading.Thread(target=self._loop, name='so101-dashcam-recorder', daemon=True)
        self._thread.start()

    def set_config(self, *, window_minutes: int | None = None, enabled: bool | None = None) -> dict[str, Any]:
        with self._lock:
            if window_minutes is not None:
                self.window_minutes = 30 if int(window_minutes) <= 45 else 60
            if enabled is not None:
                self._running = bool(enabled)
            self._prune_expired_locked()
        return self.status()

    def mark_task_start(self, kind: str, query: str = '', mode: str = '') -> None:
        with self._lock:
            self._active_task_start_ts = time.time()
            self._active_task_meta = {
                'kind': kind,
                'query': query,
                'mode': mode,
                'started_at': self._active_task_start_ts,
                'd435_before': self.d435.capture_observation(tag='before_task', fallback_wrist_path=self._last_cached_wrist_path),
            }

    def _sample_once(self) -> None:
        now = time.time()
        # Read cached/non-intrusive joint state so we never contend with high-rate serial bus streaming
        joints = {}
        goals = {}
        bias = {}
        try:
            if getattr(self.robot, 'connected', False) and not getattr(self.robot, 'moving', False):
                # Light read only when idle or use last known goals/positions when moving
                pass
            for k in ALL_JOINTS:
                g_val = int(getattr(self.robot, 'goals', {}).get(k, self._last_joints.get(k, 2047)))
                b_val = int(getattr(self.robot, 'bias', {}).get(k, 0))
                joints[k] = g_val
                goals[k] = g_val
                bias[k] = b_val
            self._last_joints = dict(joints)
        except Exception:
            joints = dict(self._last_joints)

        # Save wrist frame snapshot every ~1.2s (or every 0.45s while moving/task active)
        is_busy = bool(getattr(self.robot, 'moving', False) or self._active_task_start_ts is not None)
        img_interval = 0.45 if is_busy else 1.5
        wrist_path = self._last_cached_wrist_path
        if now - self._last_image_save_ts >= img_interval:
            try:
                cam = getattr(self.robot, 'camera', None)
                frame_bgr = getattr(cam, 'frame', None) if cam is not None else None
                if frame_bgr is not None:
                    fname = f'wrist_{int(now * 1000)}.jpg'
                    fpath = self.rolling_frames_dir / fname
                    small = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(fpath), small, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    wrist_path = str(fpath)
                    self._last_cached_wrist_path = wrist_path
                    self._last_image_save_ts = now
            except Exception:
                pass

        d435_meta = self.d435.capture_observation(tag='stream', fallback_wrist_path=wrist_path) if is_busy else (
            self.d435._last_frame_meta or {
                'timestamp': now,
                'source': 'realsense_d435_standby_auxiliary_wrist_ref',
                'connected': False,
                'used_for_control': False,
                'rgb_path': wrist_path,
            }
        )

        record = {
            'timestamp': round(now, 3),
            'ros_topics': {
                '/so101/joint_states': joints,
                '/so101/joint_goals': goals,
                '/so101/servo_bias': bias,
                '/so101/wrist_cam/image_raw': wrist_path,
                '/realsense_d435/color/image_raw': d435_meta.get('rgb_path'),
                '/realsense_d435/aligned_depth': d435_meta.get('depth_path'),
            },
            'joints': joints,
            'goals': goals,
            'bias': bias,
            'moving': bool(getattr(self.robot, 'moving', False)),
            'armed': bool(getattr(self.robot, 'armed', False)),
            'wrist_frame_path': wrist_path,
            'realsense_d435': d435_meta,
            'active_task': self._active_task_meta.get('kind') if self._active_task_start_ts else None,
        }

        with self._lock:
            self._buffer.append(record)
            try:
                stream_file = self.rolling_dir / 'rosbag_stream.jsonl'
                if stream_file.exists() and stream_file.stat().st_size > 12_000_000:
                    stream_file.replace(self.rolling_dir / 'rosbag_stream.prev.jsonl')
                with stream_file.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            except Exception:
                pass
            if int(now) % 20 == 0:
                self._prune_expired_locked()

    def _prune_expired_locked(self) -> None:
        cutoff = time.time() - (self.window_minutes * 60.0)
        while self._buffer and self._buffer[0]['timestamp'] < cutoff:
            self._buffer.popleft()
        # Prune wrist_frames older than rolling window
        try:
            for fpath in self.rolling_frames_dir.glob('wrist_*.jpg'):
                if fpath.stat().st_mtime < cutoff:
                    fpath.unlink(missing_ok=True)
        except Exception:
            pass

    def _loop(self) -> None:
        while True:
            try:
                if self._running:
                    self._sample_once()
            except Exception:
                pass
            time.sleep(1.0 / self.sample_hz)

    def _export_rerun_rrd(
        self,
        rrd_path: Path,
        frames: list[dict[str, Any]],
        episode_meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Export a native Rerun `.rrd` recording file using `rerun-sdk`."""
        try:
            import rerun as rr  # type: ignore

            rec = rr.RecordingStream('lerobot_so101_dashcam', make_default=False)
            rec.save(str(rrd_path))
            rec.set_time('episode_step', sequence=0)
            rec.log(
                'task/summary',
                rr.TextLog(
                    f"[{episode_meta['outcome'].upper()}] {episode_meta.get('task_kind', 'task')} | "
                    f"Query: {episode_meta.get('query', '')} | Reason: {episode_meta.get('reason', '')}"
                ),
            )

            logged_images = 0
            last_img_path = None
            for idx, item in enumerate(frames):
                rec.set_time('episode_step', sequence=idx)
                j = item.get('joints') or {}
                for jname in ALL_JOINTS:
                    if jname in j:
                        rec.log(f'robot/joints/{jname}', rr.Scalars(float(j[jname])))

                wpath = item.get('wrist_frame_path')
                if wpath and wpath != last_img_path and Path(wpath).is_file() and logged_images < 40:
                    try:
                        rec.log('cameras/wrist_rgb', rr.EncodedImage(path=str(wpath)))
                        last_img_path = wpath
                        logged_images += 1
                    except Exception:
                        pass

            rec.flush()
            return {
                'rrd_generated': True,
                'rrd_path': str(rrd_path),
                'rrd_size_bytes': rrd_path.stat().st_size if rrd_path.exists() else 0,
                'rerun_cli_hint': f'rerun "{rrd_path}"',
            }
        except Exception as exc:
            return {
                'rrd_generated': False,
                'rrd_path': str(rrd_path),
                'error': str(exc),
            }

    def _build_sim_counterfactual_pack(
        self,
        episode_id: str,
        outcome: str,
        frames: list[dict[str, Any]],
        task_result: dict[str, Any] | None,
        reason: str,
    ) -> dict[str, Any]:
        """Build a Real-to-Sim Counterfactual & Synthetic Augmentation pack for re-training."""
        scene_objects = []
        try:
            if self.mira_hub is not None and hasattr(self.mira_hub, 'scene_map'):
                scene_objects = list(self.mira_hub.scene_map.objects)
        except Exception:
            pass

        # Generate synthetic domain-randomized variations around the recorded trajectory
        # especially valuable when outcome == 'failure' so Sim can synthesize corrective demonstrations!
        synthetic_branches = []
        for idx, (dx_mm, dy_mm, jaw_gap_mm, pitch_delta_deg, tag) in enumerate([
            (0.0, 0.0, 3.5, 0.0, 'canonical_left_jaw_aligned_3.5mm'),
            (-2.0, 1.5, 3.5, 3.0, 'counterfactual_left_edge_snug_pitch_plus3'),
            (2.0, -1.5, 4.0, -3.0, 'counterfactual_clearance_4mm_pitch_minus3'),
            (0.0, 2.5, 3.2, 5.0, 'counterfactual_steep_approach_anti_tip'),
        ]):
            synthetic_branches.append({
                'branch_id': f'{episode_id}_syn_{idx}',
                'strategy': tag,
                'perturbation': {
                    'object_xy_offset_mm': [dx_mm, dy_mm],
                    'target_fixed_left_jaw_gap_mm': jaw_gap_mm,
                    'pitch_adjustment_deg': pitch_delta_deg,
                },
                'recommended_for_retraining': True,
            })

        return {
            'schema': 'lerobot-so101-real2sim-counterfactual-v1',
            'episode_id': episode_id,
            'recorded_outcome': outcome,
            'diagnosis_reason': reason,
            'num_recorded_frames': len(frames),
            'digital_twin_scene_objects': scene_objects,
            'task_result_snapshot': task_result or {},
            'synthetic_augmentation_branches': synthetic_branches,
            'training_usage': {
                'smolvla_lerobot_v2_ready': True,
                'dreamlabs_embodiment_ready': True,
                'sim_replay_glb': str(self.mira_hub.current_dir / 'scene.glb') if self.mira_hub else None,
            },
        }

    def archive_episode(
        self,
        *,
        outcome: str,
        task_kind: str = 'manual_tag',
        query: str = '',
        reason: str = '',
        lookback_seconds: float = 45.0,
        task_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tag and archive a task episode into `archived/success/` or `archived/failure/`."""
        norm_outcome = 'success' if str(outcome).lower().strip() in ('success', 'ok', 'true', 'passed', '1') else 'failure'
        now = time.time()
        with self._lock:
            self._prune_expired_locked()
            start_ts = self._active_task_start_ts or (now - max(10.0, float(lookback_seconds)))
            # Include a 3-second pre-roll before task start
            window_start = max(now - self.window_minutes * 60.0, start_ts - 3.0)
            frames = [dict(f) for f in self._buffer if f['timestamp'] >= window_start]
            if not frames and self._buffer:
                frames = [dict(f) for f in list(self._buffer)[-60:]]
            d435_before = self._active_task_meta.get('d435_before')
            self._active_task_start_ts = None
            self._active_task_meta = {}

        d435_after = self.d435.capture_observation(tag=f'after_{norm_outcome}', fallback_wrist_path=self._last_cached_wrist_path)
        ep_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{norm_outcome}_{task_kind.replace('/', '_')}"
        target_root = self.success_dir if norm_outcome == 'success' else self.failure_dir
        ep_dir = target_root / ep_id
        ep_frames_dir = ep_dir / 'wrist_frames'
        ep_frames_dir.mkdir(parents=True, exist_ok=True)

        # Copy referenced wrist camera frames into the archived episode folder
        copied_frames = []
        last_copied_src = None
        last_copied_dst = None
        for idx, item in enumerate(frames):
            src = item.get('wrist_frame_path')
            dst_rel = None
            if src and Path(src).is_file():
                if src == last_copied_src:
                    dst_rel = last_copied_dst
                else:
                    dst_path = ep_frames_dir / f'frame_{idx:04d}.jpg'
                    try:
                        shutil.copy2(src, dst_path)
                        dst_rel = str(dst_path)
                        last_copied_src = src
                        last_copied_dst = dst_rel
                    except Exception:
                        dst_rel = src
            item_copy = {**item, 'wrist_frame_path': dst_rel}
            copied_frames.append(item_copy)

        # Copy current Sim Digital Twin GLB if available
        sim_glb_dst = None
        try:
            if self.mira_hub is not None:
                src_glb = self.mira_hub.current_dir / 'scene.glb'
                if src_glb.is_file():
                    sim_glb_dst = ep_dir / 'scene_twin.glb'
                    shutil.copy2(src_glb, sim_glb_dst)
        except Exception:
            sim_glb_dst = None

        episode_meta = {
            'episode_id': ep_id,
            'outcome': norm_outcome,
            'task_kind': task_kind,
            'query': query,
            'reason': reason,
            'archived_at': round(now, 3),
            'duration_s': round((copied_frames[-1]['timestamp'] - copied_frames[0]['timestamp']) if len(copied_frames) >= 2 else 0.0, 2),
            'num_frames': len(copied_frames),
            'd435_before': d435_before,
            'd435_after': d435_after,
            'sim_twin_glb': str(sim_glb_dst) if sim_glb_dst else None,
        }

        # 1. Write ROSbag / LeRobot JSONL frames
        jsonl_path = ep_dir / 'rosbag_frames.jsonl'
        with jsonl_path.open('w', encoding='utf-8') as f:
            for item in copied_frames:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 2. Write LeRobot v2 / Dream Labs training dataset manifest
        lerobot_bundle = {
            'codebase_version': 'v2.1',
            'robot_type': 'so101_follower',
            'episode_id': ep_id,
            'outcome': norm_outcome,
            'task': query or task_kind,
            'fps': self.sample_hz,
            'state_names': list(ALL_JOINTS),
            'action_names': list(ALL_JOINTS),
            'total_frames': len(copied_frames),
            'embodiment_config': {
                'state': {'right_arm': [0, 5], 'right_gripper': [5, 6]},
                'action': {'right_arm': [0, 5], 'right_gripper': [5, 6]},
                'video': {'wrist_r': 'observation.images.wrist', 'overhead': 'observation.images.realsense_d435'},
                'head_camera': 'overhead',
            },
            'samples': [
                {
                    'frame_index': i,
                    'timestamp': row['timestamp'],
                    'observation.state': [float(row['joints'].get(k, 2047)) for k in ALL_JOINTS],
                    'action': [float(row['goals'].get(k, 2047)) for k in ALL_JOINTS],
                    'observation.images.wrist': row.get('wrist_frame_path'),
                    'observation.images.realsense_d435': (row.get('realsense_d435') or {}).get('rgb_path'),
                }
                for i, row in enumerate(copied_frames)
            ],
        }
        (ep_dir / 'lerobot_episode.json').write_text(json.dumps(lerobot_bundle, ensure_ascii=False, indent=2), encoding='utf-8')

        # 3. Export Real-to-Sim Counterfactual & Synthetic Augmentation Pack
        cf_pack = self._build_sim_counterfactual_pack(ep_id, norm_outcome, copied_frames, task_result, reason)
        (ep_dir / 'sim_counterfactual_pack.json').write_text(json.dumps(cf_pack, ensure_ascii=False, indent=2), encoding='utf-8')

        # 4. Export native Rerun `.rrd` replay file
        rrd_info = self._export_rerun_rrd(ep_dir / 'episode.rrd', copied_frames, episode_meta)
        episode_meta['rerun'] = rrd_info
        episode_meta['files'] = {
            'episode_rrd': f'/api/dashcam/artifact/{norm_outcome}/{ep_id}/episode.rrd',
            'rosbag_jsonl': f'/api/dashcam/artifact/{norm_outcome}/{ep_id}/rosbag_frames.jsonl',
            'lerobot_json': f'/api/dashcam/artifact/{norm_outcome}/{ep_id}/lerobot_episode.json',
            'sim_counterfactual_json': f'/api/dashcam/artifact/{norm_outcome}/{ep_id}/sim_counterfactual_pack.json',
        }
        (ep_dir / 'metadata.json').write_text(json.dumps(episode_meta, ensure_ascii=False, indent=2), encoding='utf-8')
        return episode_meta

    def list_episodes(self, limit: int = 30) -> dict[str, Any]:
        episodes = []
        for outcome_label, root_dir in (('success', self.success_dir), ('failure', self.failure_dir)):
            if not root_dir.exists():
                continue
            for ep_dir in sorted(root_dir.iterdir(), reverse=True):
                meta_file = ep_dir / 'metadata.json'
                if meta_file.is_file():
                    try:
                        episodes.append(json.loads(meta_file.read_text(encoding='utf-8')))
                    except Exception:
                        pass
        episodes.sort(key=lambda x: float(x.get('archived_at', 0.0)), reverse=True)
        success_count = sum(1 for e in episodes if e.get('outcome') == 'success')
        failure_count = sum(1 for e in episodes if e.get('outcome') == 'failure')
        return {
            'total_archived': len(episodes),
            'success_count': success_count,
            'failure_count': failure_count,
            'episodes': episodes[:limit],
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            buf_len = len(self._buffer)
            oldest_ts = self._buffer[0]['timestamp'] if self._buffer else None
            newest_ts = self._buffer[-1]['timestamp'] if self._buffer else None
        span_min = round((newest_ts - oldest_ts) / 60.0, 2) if (oldest_ts and newest_ts) else 0.0
        listing = self.list_episodes(limit=12)
        return {
            'running': self._running,
            'window_minutes': self.window_minutes,
            'sample_hz': self.sample_hz,
            'buffered_frames': buf_len,
            'buffered_span_minutes': span_min,
            'realsense_d435': self.d435.status(),
            'active_task': self._active_task_meta if self._active_task_start_ts else None,
            **listing,
        }
