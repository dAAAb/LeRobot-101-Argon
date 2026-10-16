"""Read-only encoder/image pairing and publication of engineering snapshots.

Never arm, hold, connect, or command the robot, including on failure. These
captures are saved observations, not continuous live state or measured depth.
"""
from __future__ import annotations

import json
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from .local_cartesian import ALL

ROOT = Path(__file__).resolve().parents[1]


def _export_snapshot_bundle(destination, frame):
    if all(importlib.util.find_spec(name) is not None for name in ('mujoco', 'trimesh')):
        from scripts.export_sim_scene import export_bundle
        state = export_bundle(destination, frame)
    else:
        # Reuse the project's existing offline engineering environment. Never
        # install packages in, or switch, the live controller's interpreter.
        worker = ROOT / '.runtime/mujoco-venv/bin/python'
        if not worker.is_file() or not os.access(worker, os.X_OK):
            raise RuntimeError('工程匯出環境尚未備妥；原快照仍保留')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.capture-export-', dir=destination.parent) as temporary:
            source = Path(temporary) / 'frame.json'
            source.write_text(json.dumps(frame, ensure_ascii=False, allow_nan=False))
            try:
                subprocess.run([str(worker), '-B', str(ROOT / 'scripts/export_sim_scene.py'),
                    '--frame', str(source), '--output', str(destination)],
                    cwd=ROOT, check=True, timeout=60, capture_output=True, text=True)
            except (subprocess.SubprocessError, OSError) as exc:
                raise RuntimeError('工程場景匯出未完成；原快照仍保留') from exc
        state = json.loads((destination / 'scene.json').read_text())
    # A worker result is not allowed to attach unrelated observations/depth.
    exported = state.get('frame', {})
    if (any(exported.get(key) != frame.get(key) for key in ('id', 'source', 'camera_generation', 'joints'))
            or (destination / 'depth.json').exists()
            or any(not (destination / name).is_file() for name in ('scene.glb', 'scene.xml', 'verification.json', 'wrist.jpg'))):
        raise RuntimeError('工程匯出與本次影像不一致；原快照仍保留')
    return state


def capture_snapshot(robot, camera):
    if not robot.connected or robot.moving or robot.pwm_owner is not None:
        raise RuntimeError('請先連接手臂並等待目前工作結束，再擷取姿勢')
    generation = camera.generation
    if not isinstance(generation, str) or not generation:
        raise RuntimeError('尚未取得腕部相機身分')
    started = time.monotonic()
    samples = []
    for _ in range(6):
        before = time.monotonic()
        raw = robot.read()
        after = time.monotonic()
        if set(raw) != set(ALL) or any(type(v) is not int or not 0 <= v <= 4095 for v in raw.values()):
            raise RuntimeError('無法取得完整的馬達位置')
        samples.append((before, after, dict(raw)))
        time.sleep(.12)
    window_end = samples[-1][1]
    frame = None
    while time.monotonic() - started < 3:
        if robot.moving or camera.generation != generation:
            raise RuntimeError('擷取期間手臂或相機狀態改變，請重新擷取')
        stamp = camera.latest_stamp()
        captured = stamp.get('frame_capture_monotonic')
        if (stamp.get('camera_source') != 'wrist' or type(captured) not in (int, float)
                or not math.isfinite(captured) or captured > time.monotonic()):
            raise RuntimeError('腕部影像時刻無效')
        if captured >= window_end:
            frame = camera.snapshot()
            break
        time.sleep(.04)
    if frame is None:
        raise RuntimeError('沒有與目前姿勢同步的新腕部影像')
    before = time.monotonic(); raw = robot.read(); after = time.monotonic()
    if set(raw) != set(ALL) or any(type(v) is not int or not 0 <= v <= 4095 for v in raw.values()):
        raise RuntimeError('無法取得完整的馬達位置')
    samples.append((before, after, dict(raw)))
    captured = frame.get('capture_monotonic')
    if (frame.get('source') != 'wrist' or frame.get('camera_generation') != generation
            or camera.generation != generation or robot.moving
            or type(captured) not in (int, float) or not math.isfinite(captured)
            or not window_end <= captured <= before or not 0 <= after - captured < 1):
        raise RuntimeError('腕部影像與馬達位置無法配對，請重新擷取')
    span = max(max(s[2][n] for s in samples) - min(s[2][n] for s in samples) for n in ALL)
    if span > 2:
        raise RuntimeError('手臂尚未停穩，沒有更新工程快照')
    bracket = samples[-2]
    frame = {**frame, 'joints': {n: round((bracket[2][n] + raw[n]) / 2) for n in ALL},
             'timestamp_basis': 'fresh wrist JPEG bracketed by read-only settled encoder samples',
             'joint_capture_span_ticks': span, 'settled_duration_s': window_end - samples[0][1],
             'joint_observation_bracket_monotonic': [bracket[0], after]}
    return frame


def publish_snapshot(runtime, frame):
    """Keep old bundles intact and atomically select a complete depth-free one."""
    root = Path(runtime) / 'sim-scene'
    identity = 'capture-' + uuid.uuid4().hex
    destination = root / identity
    state = _export_snapshot_bundle(destination, frame)  # Never copy an older depth.json.
    state['capture_kind'] = 'current_pose_without_depth'
    state['depth_status'] = 'not_measured_for_this_snapshot'
    (destination / 'scene.json').write_text(json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2))
    current = root / 'current'
    temporary = root / ('.current-' + uuid.uuid4().hex)
    temporary.symlink_to(identity, target_is_directory=True)
    try:
        # Older exports used a real /current directory. Preserve it too.
        if current.exists() and not current.is_symlink():
            current.rename(root / ('archive-' + uuid.uuid4().hex))
        temporary.replace(current)
    finally:
        temporary.unlink(missing_ok=True)
    return {'snapshot_id': identity, 'frame_id': frame['id'],
            'captured_at': frame.get('captured_at'), 'depth_included': False}
