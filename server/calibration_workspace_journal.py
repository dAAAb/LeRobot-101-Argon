"""Durable progress for a finite engineering calibration excursion.

Never changes the origin or dispatches hardware. An interrupted station stays
unresolved across a service restart; a new HTTP request cannot silently replay
it or advance to the next station.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import uuid


class WorkspaceJournal:
    def __init__(self, root):
        self.root = Path(root) / '.runtime' / 'calibration-workspace'
        self.lock = threading.RLock()

    def _path(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch('[0-9a-f]{32}', session_id):
            raise ValueError('無效的校準工作階段')
        path = self.root / (session_id + '.json')
        if self.root.is_symlink() or path.is_symlink():
            raise ValueError('校準紀錄不能是符號連結')
        return path

    def load(self, session_id):
        with self.lock:
            path = self._path(session_id)
            if not path.exists():
                return None
            result = json.loads(path.read_text())
            if result.get('session_id') != session_id:
                raise ValueError('校準紀錄的工作階段不符')
            return result

    def _save(self, record):
        path = self._path(record['session_id'])
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def create(self, session_id, protocol):
        with self.lock:
            if self.load(session_id) is not None:
                raise ValueError('這批校準已有固定起點與路徑，不能重新建立以累加移動範圍')
            record = dict(session_id=session_id, protocol=deepcopy(protocol),
                          completed_station_ids=[], receipts=[], captures=[], inflight=None)
            self._save(record)
            return record

    def fork_remaining(self, parent_session_id, new_protocol, *, failed_candidate, failure_evidence):
        """Publish an explicitly reviewed remaining route in a fresh session.

        The parent interruption and accepted photos remain unchanged. This is
        an offline checkpoint operation, never a retry endpoint or dispatch.
        A child may start at the measured stop but inherits the global origin.
        """
        from .calibration_workspace_motion import _validate, _hash, _camera_generation, observation
        with self.lock:
            parent = self.load(parent_session_id)
            if parent is None or not isinstance(parent.get('inflight'), dict):
                raise ValueError('需要保留原本尚未完成的工程移動紀錄')
            previous = parent['protocol']
            _validate(previous, 'engineering_workspace_calibration')
            _validate(new_protocol, 'engineering_workspace_calibration')
            child_id = new_protocol.get('session_id')
            self._path(child_id)
            if child_id == parent_session_id or self.load(child_id) is not None:
                raise ValueError('剩餘路徑必須發布到新的空白工作階段')
            checkpoint = new_protocol.get('replan')
            completed = parent['completed_station_ids']
            prefix = [s['station_id'] for s in previous['stations'][:len(completed)]]
            if (not isinstance(checkpoint, dict) or completed != prefix
                    or len(completed) >= len(previous['stations'])
                    or checkpoint.get('parent_record_sha256') != _hash(parent)
                    or checkpoint.get('parent_protocol_sha256') != previous['contract_sha256']
                    or checkpoint.get('parent_session_id') != parent_session_id
                    or checkpoint.get('completed_station_ids') != completed
                    or checkpoint.get('root_original_origin') != previous['origin']
                    or checkpoint.get('interrupted_station_id') != parent['inflight']['station_id']
                    or checkpoint.get('interrupted_candidate_id') != parent['inflight']['candidate_id']
                    or parent['inflight']['station_id'] != previous['stations'][len(completed)]['station_id']
                    or checkpoint.get('checkpoint_start_joints') != new_protocol['reference']['joints']
                    or checkpoint.get('checkpoint_frame') != new_protocol['reference']
                    or checkpoint.get('checkpoint_held_goals') != new_protocol['initial_held_goals']):
                raise ValueError('剩餘路徑必須明確綁定中斷點、完成前段與新實測起點')
            frozen = ('origin', 'camera_id', 'mount_revision', 'camera_profile', 'camera_profile_sha256',
                'geometry_revision', 'origin_limit_profile', 'joint_delta_bounds_from_original_origin',
                'maximum_tip_distance_from_original_origin_mm', 'maximum_segment_ticks',
                'maximum_segment_tip_mm', 'maximum_segment_seconds')
            if any(k not in previous or k not in new_protocol or previous[k] != new_protocol[k] for k in frozen):
                raise ValueError('衍生路徑不能更換相機、校準、原始起點或擴大總範圍')
            camera_binding = dict(source_camera_generation=_camera_generation(previous, parent.get('camera_rebind')),
                checkpoint_camera_generation=new_protocol['reference']['camera_generation'],
                camera_id=previous['camera_id'], mount_revision=previous['mount_revision'],
                camera_profile_sha256=previous['camera_profile_sha256'])
            if any(checkpoint.get(key) != expected for key, expected in camera_binding.items()):
                raise ValueError('檢查點的相機來源、重新連線版本或校準識別不符')
            mapping, children = checkpoint.get('source_station_mapping'), new_protocol.get('stations')
            if (not isinstance(mapping, list) or not isinstance(children, list) or not mapping or len(mapping) != len(children)
                    or len({s.get('station_id') for s in children if isinstance(s, dict)}) != len(children)):
                raise ValueError('衍生路徑缺少每一站對應原始剩餘路徑的紀錄')
            source_index, prelude = len(completed), False
            for index, (link, station) in enumerate(zip(mapping, children)):
                if source_index >= len(previous['stations']):
                    raise ValueError('衍生路徑新增了原計畫之外的目標')
                source = previous['stations'][source_index]
                active = source.get('planned_active_joints', source['nominal_segment']['active_joints'])
                if (not isinstance(link, dict) or not isinstance(station, dict)
                        or link.get('station_id') != station.get('station_id')
                        or link.get('source_station_id') != source['station_id']
                        or link.get('source_station_index') != source_index
                        or link.get('original_target_joints') != source['target_joints']
                        or station.get('station_index') != index
                        or link.get('planned_active_joints') != station.get('planned_active_joints')):
                    raise ValueError('衍生路徑的站點對照或原始目標不符')
                if link.get('phase') == 'elbow_prelude':
                    if (prelude or 'elbow_flex' not in active or len(active) < 2
                            or station['planned_active_joints'] != ['elbow_flex']
                            or station['target_joints']['elbow_flex'] != source['target_joints']['elbow_flex']
                            or station.get('capture_required') is not False
                            or station.get('capture_id') is not None or station.get('split') is not None):
                        raise ValueError('拆出的肘前置動作不能改目標或額外取像')
                    prelude = True
                elif link.get('phase') == 'source_target':
                    expected = [n for n in active if n != 'elbow_flex'] if prelude else active
                    if (station.get('target_joints') != source['target_joints']
                            or station.get('planned_active_joints') != expected
                            or any(station.get(k) != source.get(k) for k in ('capture_id', 'capture_required', 'split'))
                            or ('elbow_flex' in active and len(active) > 1 and not prelude)):
                        raise ValueError('衍生路徑必須保留每個原始目標與分組，混合肘動作必須先拆開')
                    source_index += 1
                    prelude = False
                else:
                    raise ValueError('衍生路徑包含未知階段')
            if prelude or source_index != len(previous['stations']):
                raise ValueError('衍生路徑漏掉了原始剩餘目標')
            if not isinstance(failed_candidate, dict):
                raise ValueError('必須保存完整的中斷候選')
            candidate = {k: deepcopy(v) for k, v in failed_candidate.items() if k not in ('candidate_id', 'frame')}
            _validate(candidate, 'engineering_workspace_station')
            if (failed_candidate.get('candidate_id') != parent['inflight']['candidate_id']
                    or candidate.get('station_id') != parent['inflight']['station_id']
                    or candidate.get('station_index') != len(completed)
                    or candidate.get('session_id') != parent_session_id
                    or candidate.get('protocol_sha256') != previous['contract_sha256']
                    or candidate.get('origin') != previous['origin']
                    or not isinstance(failure_evidence, dict)
                    or not isinstance(failure_evidence.get('error'), str) or not failure_evidence['error'].strip()
                    or type(failure_evidence.get('time')) not in (int, float)
                    or not math.isfinite(failure_evidence['time']) or failure_evidence['time'] <= 0):
                raise ValueError('中斷候選與失敗證據不屬於原路徑')
            observation._pose(failure_evidence.get('actual'))
            observation._pose(failure_evidence.get('goals'))
            evidence = dict(parent_record=deepcopy(parent), failed_candidate=deepcopy(failed_candidate),
                            failure_evidence=deepcopy(failure_evidence))
            self._archive_checkpoint(parent)
            self._archive_checkpoint(evidence)
            record = dict(session_id=child_id, protocol=deepcopy(new_protocol),
                completed_station_ids=[], receipts=[], captures=[], inflight=None,
                derived_from_session_id=parent_session_id,
                derived_from_protocol_sha256=previous['contract_sha256'],
                checkpoint_evidence_sha256=_hash(evidence), checkpoint_evidence=evidence)
            self._save(record)
            return record

    def _archive_checkpoint(self, value):
        """Publish a complete immutable checkpoint without replacing history."""
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        revisions = self.root / 'revisions'
        archive = revisions / (digest + '.json')
        if revisions.is_symlink() or archive.is_symlink():
            raise ValueError('校準 checkpoint 不能是符號連結')
        revisions.mkdir(parents=True, exist_ok=True)
        if archive.exists():
            if json.loads(archive.read_text()) != value:
                raise ValueError('既有校準 checkpoint 不一致，不可覆寫')
            return
        temporary = revisions / ('.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x') as stream:
                stream.write(canonical); stream.flush(); os.fsync(stream.fileno())
            temporary.chmod(0o444)
            try:
                os.link(temporary, archive)
            except FileExistsError:
                if archive.is_symlink() or json.loads(archive.read_text()) != value:
                    raise ValueError('既有校準 checkpoint 不一致，不可覆寫')
        finally:
            temporary.unlink(missing_ok=True)

    def publish_addendum(self, session_id, new_protocol):
        """Offline, once-only four-view wrist addendum to a completed survey.

        The image session and its original bounds continue unchanged. The
        entire prior movement/capture journal is archived before publishing
        fresh per-addendum progress. This never rearms or dispatches motion.
        """
        from .calibration_workspace_motion import _validate, _hash, _camera_generation, observation
        with self.lock:
            old = self.load(session_id)
            if old is None:
                raise ValueError('找不到已完成的工程取景紀錄')
            previous = old['protocol']
            _validate(previous, 'engineering_workspace_calibration')
            _validate(new_protocol, 'engineering_workspace_calibration')
            if (old.get('inflight') is not None or not previous.get('stations')
                    or old.get('completed_station_ids') != [s['station_id'] for s in previous['stations']]
                    or 'previous_complete_record' in old or 'addendum' in previous):
                raise ValueError('只可對完整完成、沒有中斷的取景追加一次四張腕部觀察')
            metadata = new_protocol.get('addendum')
            checkpoint = new_protocol.get('replan')
            if (new_protocol.get('session_id') != session_id or not isinstance(metadata, dict)
                    or metadata.get('kind') != 'completed_survey_wrist_addendum'
                    or metadata.get('parent_session_id') != session_id
                    or metadata.get('parent_record_sha256') != _hash(old)
                    or metadata.get('parent_protocol_sha256') != previous['contract_sha256']
                    or not isinstance(checkpoint, dict)
                    or checkpoint.get('checkpoint_start_joints') != new_protocol['reference']['joints']
                    or checkpoint.get('checkpoint_frame') != new_protocol['reference']
                    or checkpoint.get('checkpoint_held_goals') != new_protocol['initial_held_goals']):
                raise ValueError('補取像計畫必須綁定完整前段紀錄與新的實測起點')
            frozen = ('origin', 'camera_id', 'mount_revision', 'camera_profile', 'camera_profile_sha256',
                'geometry_revision', 'origin_limit_profile', 'joint_delta_bounds_from_original_origin',
                'maximum_tip_distance_from_original_origin_mm', 'maximum_segment_ticks',
                'maximum_segment_tip_mm', 'maximum_segment_seconds')
            if (any(k not in previous or k not in new_protocol or previous[k] != new_protocol[k] for k in frozen)
                    or new_protocol['reference']['camera_generation'] !=
                        _camera_generation(previous, old.get('camera_rebind'))):
                raise ValueError('補取像不能重設原點、改變相機版本或擴大原始範圍')
            raw = observation._pose(new_protocol['reference']['joints'])
            observation._pose(new_protocol['initial_held_goals'])
            stations = new_protocol.get('stations')
            old_ids = {s['station_id'] for s in previous['stations']}
            old_captures = {s.get('capture_id') for s in previous['stations']}
            if (not isinstance(stations, list) or len(stations) != 4
                    or new_protocol.get('capture_count') != 4
                    or any(not isinstance(s, dict) for s in stations)
                    or len({s.get('station_id') for s in stations}) != 4
                    or len({s.get('capture_id') for s in stations}) != 4
                    or sorted(s.get('split', '') for s in stations) != ['heldout', 'heldout', 'training', 'training']):
                raise ValueError('補取像限四個不同腕部姿勢，分為兩張 training 與兩張 heldout')
            for index, station in enumerate(stations):
                active = station.get('planned_active_joints')
                target = observation._pose(station.get('target_joints'))
                if (station.get('station_index') != index
                        or not isinstance(station.get('station_id'), str) or not station['station_id']
                        or station['station_id'] in old_ids
                        or not isinstance(station.get('capture_id'), str) or not station['capture_id']
                        or station['capture_id'] in old_captures or station.get('capture_required') is not True
                        or not isinstance(active, list) or not active or len(set(active)) != len(active)
                        or any(n not in ('wrist_flex', 'wrist_roll') for n in active)
                        or any(target[n] != raw[n] for n in raw if n not in ('wrist_flex', 'wrist_roll'))):
                    raise ValueError('補取像只可新增腕部動作，其他關節保持 checkpoint 的實測位置')
            self._archive_checkpoint(old)
            record = dict(session_id=session_id, protocol=deepcopy(new_protocol),
                completed_station_ids=[], receipts=[], captures=[], inflight=None,
                previous_complete_record=deepcopy(old), previous_complete_record_sha256=_hash(old))
            self._save(record)
            return record

    def recover(self, session_id, new_protocol):
        """Explicit offline first-station recovery, with one final settling retry.

        This does not capture, move, retry a station, or reset the calibration
        session's motion origin. The caller must construct and review a newly
        screened finite protocol from the actual stopped pose. HTTP endpoints
        deliberately do not invoke this method automatically.
        """
        def canonical(value):
            return json.dumps(value, sort_keys=True, ensure_ascii=False,
                              separators=(',', ':'), allow_nan=False)

        def protocol_hash(protocol):
            if (not isinstance(protocol, dict) or protocol.get('version') != 1
                    or protocol.get('kind') != 'engineering_workspace_calibration'
                    or protocol.get('session_id') != session_id):
                raise ValueError('恢復計畫不是這批校準的完整協定')
            claimed = protocol.get('contract_sha256')
            actual = hashlib.sha256(canonical({k: v for k, v in protocol.items()
                                              if k != 'contract_sha256'}).encode()).hexdigest()
            if claimed != actual:
                raise ValueError('恢復計畫或原始協定的內容已改變')
            return actual

        with self.lock:
            old = self.load(session_id)
            if (old is None or not isinstance(old.get('inflight'), dict)
                    or old.get('completed_station_ids') != []):
                raise ValueError('只可明確恢復第一站中斷、尚無已完成姿勢的校準計畫')
            previous = old.get('protocol')
            parent_hash = protocol_hash(previous)
            protocol_hash(new_protocol)
            stations = previous.get('stations')
            failed_station = old['inflight'].get('station_id')
            if (not isinstance(stations, list) or not stations
                    or not isinstance(stations[0], dict)
                    or failed_station != stations[0].get('station_id')):
                raise ValueError('未完成紀錄必須對應原始協定的第一站')
            recovery = new_protocol.get('recovery')
            if (not isinstance(recovery, dict)
                    or recovery.get('parent_protocol_sha256') != parent_hash
                    or recovery.get('parent_record_sha256') != hashlib.sha256(canonical(old).encode()).hexdigest()
                    or recovery.get('failed_station_id') != failed_station
                    or 'origin' not in previous or 'origin' not in new_protocol
                    or canonical(new_protocol['origin']) != canonical(previous['origin'])):
                raise ValueError('恢復計畫必須保留原始起點並明確連回這次失敗的第一站')
            frozen = ('camera_profile_sha256', 'geometry_revision', 'origin_limit_profile',
                'joint_delta_bounds_from_original_origin', 'maximum_tip_distance_from_original_origin_mm',
                'maximum_segment_ticks', 'maximum_segment_tip_mm', 'maximum_segment_seconds')
            if any(key not in previous or key not in new_protocol
                   or canonical(previous[key]) != canonical(new_protocol[key]) for key in frozen):
                raise ValueError('恢復計畫不能更換校準版本或放大原始工程動作範圍')

            prior_recovery = previous.get('recovery')
            attempt = recovery.get('attempt', 1)
            if 'recovery' not in previous and old.get('recovered_from') is None:
                if (recovery.get('kind') != 'first_station_single_elbow_prelude'
                        or type(attempt) is not int or attempt != 1):
                    raise ValueError('第一次恢復只允許已核對的單肘前置動作')
                ancestry = []
            elif (isinstance(prior_recovery, dict)
                    and prior_recovery.get('kind') == 'first_station_single_elbow_prelude'
                    and type(prior_recovery.get('attempt', 1)) is int
                    and prior_recovery.get('attempt', 1) == 1
                    and old.get('recovered_from') == prior_recovery.get('parent_protocol_sha256')
                    and isinstance(old.get('recovered_from'), str)
                    and re.fullmatch('[0-9a-f]{64}', old['recovered_from'])):
                if (recovery.get('kind') != 'first_station_settling_retry'
                        or type(attempt) is not int or attempt != 2
                        or recovery.get('root_original_protocol_sha256') != old['recovered_from']):
                    raise ValueError('第二次只允許同一目標的最後一次到位等待驗證，不能建立無限重試鏈')
                ancestry = old.get('recovered_from_chain', [old['recovered_from']])
                if ancestry != [old['recovered_from']]:
                    raise ValueError('恢復歷史鏈不符，不能追加重試')
                # This last validation attempt may rebuild the screened paths
                # from a newly measured stopped pose. It may not add a stage,
                # replace the elbow endpoint, or change any later target.
                following = new_protocol.get('stations')
                unchanged = ('station_id', 'station_index', 'capture_id',
                             'split', 'capture_required')
                if (not isinstance(following, list) or len(following) != len(stations)
                        or any(not isinstance(before, dict) or not isinstance(after, dict)
                            or any(key not in before or key not in after
                                   or canonical(before[key]) != canonical(after[key]) for key in unchanged)
                            for before, after in zip(stations, following))):
                    raise ValueError('到位等待驗證必須保留原單肘目標、全部後續姿勢與取像分組')
                # Holding joints can drift by a tick after the first failure.
                # Bind them to the NEW measured pose so this remains an
                # elbow-only command, never a return-to-old-hold correction.
                names = {'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                         'wrist_flex', 'wrist_roll', 'gripper'}
                start = recovery.get('recovery_start')
                evidence = recovery.get('failure_evidence')
                actual = evidence.get('actual') if isinstance(evidence, dict) else None
                original_target = stations[0].get('target_joints')
                target = following[0].get('target_joints')
                poses = (start, actual, original_target, target)
                if (any(not isinstance(pose, dict) or set(pose) != names
                        or any(type(value) is not int for value in pose.values()) for pose in poses)
                        or any(abs(start[name] - actual[name]) > (4 if name == 'gripper' else 12)
                               for name in names)
                        or target != {**start, 'elbow_flex': original_target['elbow_flex']}
                        or any('target_joints' not in before or 'target_joints' not in after
                               or canonical(before['target_joints']) != canonical(after['target_joints'])
                               for before, after in zip(stations[1:], following[1:]))):
                    raise ValueError('到位等待驗證只可更新保持軸的實測值，原肘目標與後續目標必須不變')
            else:
                raise ValueError('這批校準不允許第三次恢復或無限重試鏈')

            # Publish the complete old record without overwriting an existing
            # revision. A successful archive followed by a failed current-file
            # save is retryable only when that immutable archive still agrees.
            revisions = self.root / 'revisions'
            archive = revisions / (parent_hash + '.json')
            if revisions.is_symlink() or archive.is_symlink():
                raise ValueError('校準恢復歷史不能是符號連結')
            revisions.mkdir(parents=True, exist_ok=True)
            text = canonical(old)
            if archive.exists():
                if canonical(json.loads(archive.read_text())) != text:
                    raise ValueError('既有校準恢復歷史內容不符，不可覆寫')
            else:
                temporary = revisions / ('.' + uuid.uuid4().hex + '.tmp')
                try:
                    with temporary.open('x') as stream:
                        stream.write(text)
                        stream.flush()
                        os.fsync(stream.fileno())
                    temporary.chmod(0o444)
                    try:
                        os.link(temporary, archive)
                    except FileExistsError:
                        if archive.is_symlink() or canonical(json.loads(archive.read_text())) != text:
                            raise ValueError('既有校準恢復歷史內容不符，不可覆寫')
                finally:
                    temporary.unlink(missing_ok=True)
            record = dict(session_id=session_id, protocol=deepcopy(new_protocol),
                completed_station_ids=[], receipts=[], captures=[], inflight=None,
                recovered_from=parent_hash, recovered_from_chain=[*ancestry, parent_hash],
                recovery_attempt=attempt)
            self._save(record)
            return record

    def begin(self, session_id, station_id, candidate_id):
        with self.lock:
            record = self.load(session_id)
            if record is None or record['inflight'] is not None:
                raise ValueError('尚有未核對完成的校準移動，不能重播或接續下一站')
            if station_id in record['completed_station_ids']:
                raise ValueError('這個校準姿勢已完成，不能重播舊路徑')
            record['inflight'] = dict(station_id=station_id, candidate_id=candidate_id)
            self._save(record)

    def resolve_interrupted_observation(self, session_id, reconciliation):
        """Offline evidence reconciliation; never rearm, replay or move.

        A separate numerical builder checks the failed single-elbow segment,
        endpoint telemetry and fresh stable measured pose. Preserve its full
        evidence and the old journal before accepting that measured pose as
        the first completed observation, explicitly not a reached endpoint.
        """
        from .calibration_workspace_motion import _validate, _hash, _camera_generation
        with self.lock:
            old = self.load(session_id)
            _validate(reconciliation, 'engineering_workspace_interrupted_observation')
            if (old is None or not isinstance(old.get('inflight'), dict)
                    or old.get('completed_station_ids') != [] or old.get('receipts') != []
                    or old.get('captures') != [] or old.get('interrupted_observations')):
                raise ValueError('只可離線核對尚未取像、第一站中斷的實測觀察，不能重播或跳站')
            protocol = old['protocol']
            _validate(protocol, 'engineering_workspace_calibration')
            first = protocol['stations'][0]
            receipt = reconciliation.get('receipt')
            if (reconciliation.get('parent_record_sha256') != _hash(old)
                    or reconciliation.get('protocol_sha256') != protocol['contract_sha256']
                    or reconciliation.get('station_id') != first['station_id']
                    or old['inflight'].get('station_id') != first['station_id']
                    or reconciliation.get('candidate_id') != old['inflight'].get('candidate_id')
                    or not isinstance(receipt, dict)
                    or receipt.get('completed') is not True
                    or receipt.get('completion') != 'stable_partial_observation'
                    or receipt.get('endpoint_reached') is not False
                    or receipt.get('motors_commanded') is not False
                    or receipt.get('protocol_sha256') != protocol['contract_sha256']
                    or receipt.get('station_id') != first['station_id']):
                raise ValueError('離線觀察證據必須連回這次中斷，且不能宣稱已到達原目標')
            rebind = reconciliation.get('camera_rebind')
            generation = _camera_generation(protocol, rebind)
            after = receipt.get('after')
            if (not isinstance(after, dict) or after.get('source') != 'wrist' or not after.get('id')
                    or after.get('camera_generation') != generation):
                raise ValueError('離線觀察缺少符合相機版本的新實測影像')
            for key in ('failure_evidence', 'telemetry', 'failed_candidate'):
                if not reconciliation.get(key):
                    raise ValueError('離線核對必須保留原始失敗、馬達紀錄與候選')
            # Two immutable artifacts: the full interrupted journal and the
            # numerical evidence. Either existing differing artifact rejects.
            revisions = self.root / 'revisions'
            if revisions.is_symlink():
                raise ValueError('校準恢復歷史不能是符號連結')
            revisions.mkdir(parents=True, exist_ok=True)
            for value in (old, reconciliation):
                archive = revisions / (_hash(value) + '.json')
                if archive.is_symlink():
                    raise ValueError('校準恢復歷史不能是符號連結')
                if archive.exists():
                    if json.loads(archive.read_text()) != value:
                        raise ValueError('既有校準觀察歷史內容不符，不可覆寫')
                    continue
                temporary = revisions / ('.' + uuid.uuid4().hex + '.tmp')
                try:
                    with temporary.open('x') as stream:
                        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
                        stream.flush(); os.fsync(stream.fileno())
                    temporary.chmod(0o444)
                    try:
                        os.link(temporary, archive)
                    except FileExistsError:
                        if archive.is_symlink() or json.loads(archive.read_text()) != value:
                            raise ValueError('既有校準觀察歷史內容不符，不可覆寫')
                finally:
                    temporary.unlink(missing_ok=True)
            record = deepcopy(old)
            record.update(inflight=None, completed_station_ids=[first['station_id']],
                receipts=[deepcopy(receipt)], interrupted_observations=[deepcopy(reconciliation)])
            if rebind is not None:
                record['camera_rebind'] = deepcopy(rebind)
            self._save(record)
            return record

    def moved(self, session_id, station_id, receipt):
        with self.lock:
            record = self.load(session_id)
            if record is None or (record['inflight'] or {}).get('station_id') != station_id:
                raise ValueError('校準移動回報與正在執行的姿勢不符')
            if receipt.get('completed') is not True:
                raise ValueError('沒有完成實際移動，不能推進校準紀錄')
            record['receipts'].append(deepcopy(receipt))
            record['completed_station_ids'].append(station_id)
            record['inflight'] = None
            self._save(record)

    def captured(self, session_id, station_id, result):
        with self.lock:
            record = self.load(session_id)
            if record is None or station_id not in record['completed_station_ids']:
                raise ValueError('拍攝結果沒有對應已完成的姿勢')
            record['captures'].append(dict(station_id=station_id, result=deepcopy(result)))
            self._save(record)
