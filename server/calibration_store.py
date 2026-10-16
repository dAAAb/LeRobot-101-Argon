"""Local, immutable wrist calibration captures; no devices and no activation.

The caller captures settled wrist images and declares that the printed board
has remained fixed. This store never moves the arm or infers that a moved board
is still the old reference. Calibration reports are candidates, never selected
as the production profile by this module.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

import cv2
import numpy as np

from .geometry import observation_families


_JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
_FRAME_FIELDS = ('id', 'source', 'camera_id', 'camera_generation', 'sequence', 'capture_monotonic',
                 'captured_at', 'image_size', 'joints', 'timestamp_basis', 'joint_capture_span_ticks',
                 'settled_duration_s', 'settled_joint_window', 'joint_observation_bracket_monotonic',
                 'image_wait_joint_span_ticks', 'image_joint_skew_s')


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise ValueError(f'{name} 需要明確的識別名稱')
    return value


def _validate_frame(frame):
    if not isinstance(frame, dict) or frame.get('source') != 'wrist':
        raise ValueError('校準只接受腕部相機，不接受 D435 或外部相機')
    for name in ('id', 'camera_id', 'camera_generation'):
        _text(frame.get(name), name)
    if type(frame.get('sequence')) is not int or frame['sequence'] <= 0:
        raise ValueError('校準影像缺少有效的相機序號')
    stamp = frame.get('capture_monotonic')
    if not _finite(stamp) or stamp < 0:
        raise ValueError('校準影像缺少有效的取像時間')
    raw = frame.get('joints')
    if (not isinstance(raw, dict) or set(raw) != set(_JOINTS)
            or any(type(raw[n]) is not int or not 0 <= raw[n] <= 4095 for n in _JOINTS)):
        raise ValueError('校準影像需要六個馬達的實測位置')
    bracket = frame.get('joint_observation_bracket_monotonic')
    if (not isinstance(bracket, (list, tuple)) or len(bracket) != 2
            or not all(_finite(v) for v in bracket) or not bracket[0] <= stamp <= bracket[1]):
        raise ValueError('校準影像必須位於實測馬達時間區間內')
    spans = [frame.get('joint_capture_span_ticks'), frame.get('image_wait_joint_span_ticks')]
    if any(type(v) is not int or not 0 <= v <= 2 for v in spans):
        raise ValueError('請等手臂停穩後再拍攝校準影像')
    window = frame.get('settled_joint_window')
    if not isinstance(window, list) or len(window) < 2:
        raise ValueError('校準需要停穩後連續讀取的馬達位置')
    previous = -math.inf
    for sample in window:
        start, end = sample.get('read_start_monotonic'), sample.get('read_end_monotonic')
        joints = sample.get('joints')
        if (not _finite(start) or not _finite(end) or not previous <= start <= end <= bracket[0]
                or not isinstance(joints, dict) or set(joints) != set(_JOINTS)
                or any(type(joints[n]) is not int or abs(joints[n] - raw[n]) > 2 for n in _JOINTS)):
            raise ValueError('停穩紀錄與影像中的實測馬達位置不一致')
        previous = end
    if any(max(s['joints'][n] for s in window) - min(s['joints'][n] for s in window) > 2 for n in _JOINTS):
        raise ValueError('馬達位置仍在改變，請等手臂停穩')


def _origin_fingerprint(record):
    return hashlib.sha256(json.dumps({k: v for k, v in record.items() if k != 'sha256'},
        ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class CalibrationSessionStore:
    """Session journal rooted under the supplied repository's .runtime folder."""

    def __init__(self, root):
        self.root = Path(root).resolve() / '.runtime' / 'calibration-sessions'
        self._lock = threading.RLock()

    def _directory(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch('[0-9a-f]{32}', session_id):
            raise ValueError('無效的校準工作階段')
        directory = self.root / session_id
        if directory.is_symlink():
            raise ValueError('校準工作階段不能是符號連結')
        return directory

    def _load(self, session_id):
        path = self._directory(session_id) / 'session.json'
        if not path.is_file():
            raise ValueError('找不到這個校準工作階段')
        session = json.loads(path.read_text())
        if session.get('session_id') != session_id:
            raise ValueError('校準工作階段資料不一致')
        return session

    def _save(self, session):
        directory = self._directory(session['session_id'])
        session['updated_at'] = time.time()
        temporary = directory / ('session-' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(session, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(directory / 'session.json')

    @staticmethod
    def _summary(session):
        frames = session['frames']
        coverages = [f['board_detection']['coverage_fraction'] for f in frames]
        centers = []
        for f in frames:
            size = np.asarray(f['image_size'], float)
            centers.append((np.mean(f['board_detection']['corners_px'], axis=0) / size).tolist())
        return {'session_id': session['session_id'], 'status': session['status'],
            'camera_id': session['camera_id'], 'camera_generation': session['camera_generation'],
            'mount_revision': session['mount_revision'], 'geometry_revision': session['geometry_revision'],
            'board_spec': deepcopy(session['board_spec']), 'board_static_declared': session['board_static_declared'],
            'image_size': deepcopy(session['image_size']),
            'collected_view_count': len(frames),
            'distinct_pose_count': len(observation_families(frames)) if frames else 0,
            'minimum_distinct_views_for_solve': 12,
            'coverage': {'minimum_board_fraction': min(coverages) if coverages else None,
                         'maximum_board_fraction': max(coverages) if coverages else None,
                         'board_centers_normalized': centers},
            'last_detection': deepcopy(session.get('last_detection')),
            'last_solve': deepcopy(session['solves'][-1]) if session['solves'] else None,
            'created_at': session['created_at'], 'updated_at': session['updated_at'],
            'active_profile_changed': False,
            'message': session.get('message', '')}

    def start(self, *, camera_id, mount_revision, geometry_revision, board_static):
        """Start a new fixed-board session; never imports older captures."""
        from .board_calibration import BoardSpec
        if board_static is not True:
            raise ValueError('請先固定校準板；板子移動後必須開始新的校準工作階段')
        values = {k: _text(v, k) for k, v in dict(camera_id=camera_id,
                   mount_revision=mount_revision, geometry_revision=geometry_revision).items()}
        with self._lock:
            session_id = uuid.uuid4().hex
            directory = self._directory(session_id)
            directory.mkdir(parents=True, exist_ok=False)
            (directory / 'frames').mkdir()
            (directory / 'solves').mkdir()
            now = time.time()
            session = {'schema_version': 1, 'session_id': session_id, **values,
                'status': 'collecting', 'camera_generation': None, 'image_size': None,
                'board_static_declared': True, 'board_spec': BoardSpec().to_dict(),
                'board_fingerprint': BoardSpec().fingerprint, 'frames': [], 'solves': [],
                'created_at': now, 'updated_at': now,
                'message': '校準板請保持固定；移動手臂到不同視角後再拍攝。'}
            self._save(session)
            return self._summary(session)

    def status(self, session_id):
        with self._lock:
            return self._summary(self._load(session_id))

    def list_sessions(self):
        with self._lock:
            if not self.root.is_dir():
                return []
            sessions = [self._summary(self._load(p.name)) for p in self.root.iterdir()
                        if p.is_dir() and not p.is_symlink() and re.fullmatch('[0-9a-f]{32}', p.name)
                        and (p / 'session.json').is_file()]
            return sorted(sessions, key=lambda s: s['created_at'], reverse=True)

    def motion_origin(self, session_id, frame):
        """Remember the first measured pose; later calls never move this origin.

        This is a persistence boundary, not a camera or robot operation. The
        trusted caller supplies an actual settled wrist snapshot. For migration
        it may supply the original saved planning frame, including its full
        motor timing evidence, before asking with the current frame. No JPEG is
        read and the frame is not counted as a board calibration sample.

        A camera reconnect is compatible only while no board images have been
        accepted. The returned generation follows the current frame; the saved
        original joints and their original generation remain unchanged. Once
        board images exist, their camera generation must continue to match.
        """
        with self._lock:
            session = self._load(session_id)
            if (session['status'] in ('invalidated', 'solving')
                    or session['board_static_declared'] is not True):
                raise ValueError('這個校準工作階段目前不能規劃取景')
            _validate_frame(frame)
            if frame['camera_id'] != session['camera_id']:
                raise ValueError('取景起點必須來自工作階段指定的腕部相機')
            if frame.get('board_session_id', session_id) != session_id:
                raise ValueError('不能混用不同校準工作階段的取景紀錄')
            if session['frames'] and (session['camera_generation'] != frame['camera_generation']
                    or any(f['camera_generation'] != frame['camera_generation'] for f in session['frames'])):
                raise ValueError('腕部相機已重新連接，請開始新的校準工作階段')
            path = self._directory(session_id) / 'motion-origin.json'
            if path.is_symlink():
                raise ValueError('校準取景起點不能是符號連結')
            if not path.exists():
                record = {'schema_version': 1, 'session_id': session_id,
                    'camera_id': session['camera_id'], 'mount_revision': session['mount_revision'],
                    'geometry_revision': session['geometry_revision'],
                    'frame': {k: deepcopy(frame[k]) for k in _FRAME_FIELDS if k in frame},
                    'recorded_at': time.time()}
                record['sha256'] = _origin_fingerprint(record)
                # Publish a complete file without replacing an origin written
                # concurrently by another store instance or backend process.
                temporary = path.with_name('motion-origin-' + uuid.uuid4().hex + '.tmp')
                try:
                    with temporary.open('x') as handle:
                        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        os.link(temporary, path)
                    except FileExistsError:
                        pass
                finally:
                    temporary.unlink(missing_ok=True)
            if path.is_symlink():
                raise ValueError('校準取景起點不能是符號連結')
            try:
                record = json.loads(path.read_text())
                if (not isinstance(record, dict) or record.get('schema_version') != 1
                        or record.get('sha256') != _origin_fingerprint(record)
                        or any(record.get(k) != session[k] for k in
                               ('session_id', 'camera_id', 'mount_revision', 'geometry_revision'))):
                    raise ValueError('校準取景起點的工作階段或幾何版本不一致')
                original = record['frame']
                _validate_frame(original)
                if original['camera_id'] != session['camera_id']:
                    raise ValueError('校準取景起點的腕部相機不一致')
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError('校準取景起點紀錄不完整，不能以目前姿勢取代') from exc
            return {'generation': frame['camera_generation'], 'joints': deepcopy(original['joints']),
                'frame_id': original['id'], 'origin_camera_generation': original['camera_generation'],
                'geometry_revision': record['geometry_revision']}

    def capture(self, session_id, frame, *, board_static):
        """Copy one supplied settled image and its bound motor readings."""
        from .board_calibration import BoardSpec
        from .calibration_detection_worker import detect_board_isolated
        with self._lock:
            session = self._load(session_id)
            if session['status'] in ('invalidated', 'solving'):
                raise ValueError('這個校準工作階段目前不能加入照片')
            if board_static is not True:
                session.update(status='invalidated', board_static_declared=False,
                               message='校準板已移動；請固定後開始新的工作階段，不能混用舊照片。')
                self._save(session)
                raise ValueError(session['message'])
            _validate_frame(frame)
            if frame['camera_id'] != session['camera_id']:
                raise ValueError('這張照片不是工作階段指定的腕部相機')
            if frame.get('board_session_id', session_id) != session_id:
                raise ValueError('不能混用不同校準工作階段的照片')
            if session['camera_generation'] not in (None, frame['camera_generation']):
                raise ValueError('腕部相機已重新連接，請開始新的校準工作階段')
            if session['frames']:
                last = session['frames'][-1]
                if (frame['sequence'] <= last['sequence'] or frame['capture_monotonic'] <= last['capture_monotonic']
                        or any(f['id'] == frame['id'] for f in session['frames'])):
                    raise ValueError('這張影像已加入過，或比目前的校準影像更舊')
            if len(session['frames']) >= 48:
                raise ValueError('這個工作階段已有 48 張照片，請先計算校準結果')
            source = frame.get('path')
            if not isinstance(source, str) or not Path(source).is_file():
                raise ValueError('找不到腕部影像檔案')
            jpeg = Path(source).read_bytes()
            image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('腕部影像無法解碼')
            size = list(image.shape[1::-1])
            if list(frame.get('image_size', [])) != size:
                raise ValueError('腕部影像尺寸與拍攝紀錄不符')
            if session['image_size'] not in (None, size):
                raise ValueError('影像尺寸已改變，請重新開始校準')
            clean = {k: deepcopy(frame[k]) for k in _FRAME_FIELDS if k in frame}
            clean.update(board_session_id=session_id, image_size=size)
            detection = detect_board_isolated(jpeg, frame=clean, board=BoardSpec())
            session['last_detection'] = {k: deepcopy(v) for k, v in detection.items() if k != 'corners_px'}
            if not detection.get('found'):
                session['message'] = detection.get('reason') or '還沒看清完整校準板，這張不加入計算。'
                self._save(session)
                return {**self._summary(session), 'accepted': False}
            if detection.get('board_fingerprint') != session['board_fingerprint']:
                raise ValueError('校準板規格與工作階段不符')
            name = f'view-{len(session["frames"]) + 1:03d}.jpg'
            path = self._directory(session_id) / 'frames' / name
            with path.open('xb') as handle:
                handle.write(jpeg)
            clean.update(path=str(path), image_sha256=hashlib.sha256(jpeg).hexdigest(),
                         board_detection=deepcopy(detection))
            session['frames'].append(clean)
            session.update(camera_generation=frame['camera_generation'], image_size=size,
                           status='collecting', message='已加入這個視角；校準板請繼續保持固定。')
            self._save(session)
            return {**self._summary(session), 'accepted': True, 'frame_id': clean['id']}

    def solve_multi_session(self, session_id, model, *, splits, split_source_solve_ids,
                            parent_profile_path, cancel=None):
        """Offline shared-camera solve; register an inactive candidate normally.

        ``splits`` maps existing session IDs to explicit training/heldout frame
        IDs. Source journals and images are rechecked before publication. No
        caller-supplied report or validation flag can enter this path.
        """
        from .board_calibration import _hash
        from .calibration_multi_session import fit_multi_session_calibration
        parent_path = Path(parent_profile_path).resolve()
        parent_bytes = parent_path.read_bytes()
        parent = json.loads(parent_bytes)
        artifact = dict(path=str(parent_path), sha256=hashlib.sha256(parent_bytes).hexdigest())
        with self._lock:
            if session_id not in splits or set(split_source_solve_ids) != set(splits):
                raise ValueError('候選所屬工作階段必須列入原本的分組。')
            sessions = [self._load(sid) for sid in splits]
            split_sources = {}
            for source in sessions:
                if source['status'] in ('invalidated', 'solving'):
                    raise ValueError('來源工作階段目前不能計算校準。')
                sid, solve_id = source['session_id'], split_source_solve_ids[source['session_id']]
                if (not isinstance(solve_id, str) or not re.fullmatch('[0-9a-f]{32}', solve_id)
                        or not any(s['solve_id'] == solve_id for s in source['solves'])):
                    raise ValueError('分組必須來自已保存的工作階段求解報告。')
                split_path = self._directory(sid) / 'solves' / solve_id / 'report.json'
                split_bytes = split_path.read_bytes()
                old_report = json.loads(split_bytes)
                prior_split = {n: old_report['validation'][n]
                               for n in ('training_frame_ids', 'heldout_frame_ids')}
                if (old_report.get('board_session_id') != sid or old_report.get('source') != 'wrist-only'
                        or splits[sid] != prior_split):
                    raise ValueError('不能改動原報告的 training/heldout 分組。')
                split_sources[sid] = dict(solve_id=solve_id, path=str(split_path),
                    sha256=hashlib.sha256(split_bytes).hexdigest(), **prior_split)
                for frame in source['frames']:
                    if not Path(frame['path']).resolve().is_relative_to(self._directory(source['session_id']) / 'frames'):
                        raise ValueError('來源影像不屬於這個校準工作階段。')
            if cancel is not None and cancel.is_set():
                raise RuntimeError('校準計算已停止')
            target = deepcopy(next(s for s in sessions if s['session_id'] == session_id))
            previous_status = target['status']
            target['status'] = 'solving'
            self._save(target)
            snapshots = {s['session_id']: _hash(target if s['session_id'] == session_id else s) for s in sessions}
        try:
            report = fit_multi_session_calibration(sessions, model, parent, splits=splits,
                board_session_id=session_id, parent_artifact=artifact, split_sources=split_sources)
            if cancel is not None and cancel.is_set():
                raise RuntimeError('校準計算已停止')
            with self._lock:
                if (parent_path.read_bytes() != parent_bytes
                        or any(_hash(self._load(sid)) != digest for sid, digest in snapshots.items())
                        or any(hashlib.sha256(Path(s['path']).read_bytes()).hexdigest() != s['sha256']
                               for s in split_sources.values())):
                    raise ValueError('校準來源在求解期間改變；沒有保存候選。')
                for source in sessions:
                    for frame in source['frames']:
                        if hashlib.sha256(Path(frame['path']).read_bytes()).hexdigest() != frame['image_sha256']:
                            raise ValueError('影像在求解期間改變；沒有保存候選。')
                solve_id = uuid.uuid4().hex
                directory = self._directory(session_id) / 'solves' / solve_id
                directory.mkdir(exist_ok=False)
                report_path = directory / 'report.json'
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
                candidate_path = None
                if report['calibration_profile'] is not None:
                    candidate_path = directory / 'candidate-profile.json'
                    candidate_path.write_text(json.dumps(report['calibration_profile'],
                        ensure_ascii=False, indent=2, allow_nan=False))
                target['solves'].append(dict(solve_id=solve_id, status=report['status'],
                    report_path=str(report_path), candidate_profile_path=str(candidate_path) if candidate_path else None,
                    view_count=sum(len(s['frames']) for s in sessions), created_at=time.time(),
                    target_view_count=len(target['frames']),
                    source_session_view_counts={s['session_id']: len(s['frames']) for s in sessions},
                    validation=deepcopy(report['validation']), blockers=deepcopy(report['blockers']),
                    source_session_ids=list(splits), active_profile_changed=False))
                target.update(status='solved' if report['status'] == 'validated' else 'collecting',
                    message='多工作階段校準候選已保存，尚未套用。' if candidate_path else '校準未通過，請查看驗證報告。')
                self._save(target)
                return self._summary(target)
        except Exception:
            with self._lock:
                # Never overwrite a concurrent capture or explicit invalidation.
                if _hash(self._load(session_id)) == snapshots[session_id]:
                    target.update(status=previous_status, message='計算未完成，原校準與照片已保留。')
                    self._save(target)
            raise

    def solve(self, session_id, model, *, cancel=None, frozen_profile=None):
        """Write an immutable candidate/report pair, never the active config."""
        from .board_calibration import BoardSpec, fit_board_calibration
        with self._lock:
            session = self._load(session_id)
            if session['status'] in ('invalidated', 'solving') or session['board_static_declared'] is not True:
                raise ValueError('這個工作階段目前不能計算校準')
            if cancel is not None and cancel.is_set():
                raise RuntimeError('校準計算已停止')
            if not session['frames']:
                raise ValueError('請先拍攝校準板')
            for frame in session['frames']:
                if hashlib.sha256(Path(frame['path']).read_bytes()).hexdigest() != frame['image_sha256']:
                    raise ValueError('校準影像內容已改變，請開始新的工作階段')
            session['status'] = 'solving'
            self._save(session)
        try:
            arguments = dict(camera_id=session['camera_id'], mount_revision=session['mount_revision'],
                geometry_revision=session['geometry_revision'], board_session_id=session_id, board=BoardSpec())
            if frozen_profile is None:
                report = fit_board_calibration(deepcopy(session['frames']), model, **arguments)
            else:
                from .calibration_extension import validate_workspace_extension
                report = validate_workspace_extension(deepcopy(session['frames']), model,
                    deepcopy(frozen_profile), **arguments)
            if cancel is not None and cancel.is_set():
                raise RuntimeError('校準計算已停止')
            with self._lock:
                result_id = uuid.uuid4().hex
                directory = self._directory(session_id) / 'solves' / result_id
                directory.mkdir(exist_ok=False)
                report_path = directory / 'report.json'
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
                candidate = report.get('calibration_profile')
                candidate_path = None
                if candidate is not None:
                    candidate_path = directory / 'candidate-profile.json'
                    candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False))
                session['solves'].append({'solve_id': result_id, 'status': report['status'],
                    'report_path': str(report_path), 'candidate_profile_path': str(candidate_path) if candidate_path else None,
                    'view_count': len(session['frames']), 'created_at': time.time(),
                    'validation': deepcopy(report.get('validation')), 'blockers': deepcopy(report.get('blockers', [])),
                    'active_profile_changed': False})
                session.update(status='solved' if report['status'] == 'validated' else 'collecting',
                    message='校準候選已保存，尚未套用。' if candidate is not None else '目前照片仍不足以通過校準，請查看需要補拍的條件。')
                self._save(session)
                return self._summary(session)
        except Exception:
            with self._lock:
                session.update(status='collecting', message='校準計算未完成，原有照片已保留。')
                self._save(session)
            raise
