from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path


DEFAULT_RANDOM_QUERIES = [
    '往左找到 toothpick container，把它拿起來再放回去',
    '找到桌面上的發泡錠圓筒（effervescent tube），拿起來再放回去',
    '夾起牙籤罐（toothpick container）再放回原位',
    '往右看有沒有圓筒罐子，找到後拿起來再放回去',
]


def _is_token_or_quota_exhausted(err_text: str) -> bool:
    t = (err_text or '').lower()
    return any(
        k in t
        for k in (
            'resource_exhausted',
            'quota',
            'rate limit',
            'insufficient_quota',
            'credit balance',
            'token',
            'billing',
            '429',
        )
    )


class LoopEngineer:
    """Agentic Engineering closed-loop self-diagnosis, auto-retry, and autonomous optimization loop."""

    def __init__(self, root: Path, runtime: Path, robot, brain, kinematic, *, submit_job=None, is_busy=None, cancel_active=None):
        self.root = Path(root)
        self.runtime = Path(runtime)
        self.robot = robot
        self.brain = brain
        self.kinematic = kinematic
        self.submit_job = submit_job
        self.is_busy = is_busy
        self.cancel_active = cancel_active
        self.lock = threading.Lock()
        self.loop_cancel = threading.Event()
        self.loop_thread: threading.Thread | None = None

        self.config_path = self.runtime / 'loop-engineer-config.json'
        self.history_path = self.runtime / 'loop-engineer-history.jsonl'
        self.config = self._load_config()
        self.last_attempt: dict | None = None
        self.loop_state = {
            'running': False,
            'mode': self.config.get('mode', 'custom'),
            'custom_query': self.config.get('custom_query', '往左找到 toothpick container，把它拿起來再放回去'),
            'duration_hours': self.config.get('duration_hours', 1.0),
            'started_at': None,
            'ends_at': None,
            'iterations': 0,
            'successes': 0,
            'failures': 0,
            'current_query': None,
            'last_diagnosis': None,
            'stop_reason': None,
        }

    def _load_config(self) -> dict:
        defaults = {
            'auto_retry_enabled': True,
            'max_retries_per_grasp': 1,
            'mode': 'custom',
            'custom_query': '往左找到 toothpick container，把它拿起來再放回去',
            'random_queries': list(DEFAULT_RANDOM_QUERIES),
            'duration_hours': 1.0,  # None or 0 means unlimited (until token exhausted)
            'cooldown_seconds': 4.0,
        }
        if self.config_path.exists():
            try:
                loaded = json.loads(self.config_path.read_text())
                if isinstance(loaded, dict):
                    defaults.update(loaded)
            except Exception:
                pass
        return defaults

    def update_config(self, patch: dict) -> dict:
        with self.lock:
            for k, v in (patch or {}).items():
                if k == 'auto_retry_enabled':
                    self.config[k] = bool(v)
                elif k == 'max_retries_per_grasp':
                    self.config[k] = int(max(0, min(3, int(v))))
                elif k == 'mode' and v in ('custom', 'random'):
                    self.config[k] = v
                    self.loop_state['mode'] = v
                elif k == 'custom_query' and isinstance(v, str):
                    self.config[k] = v.strip() or self.config['custom_query']
                    self.loop_state['custom_query'] = self.config[k]
                elif k == 'duration_hours':
                    self.config[k] = None if v in (None, 0, '0', 'unlimited', 'infinite') else float(v)
                    self.loop_state['duration_hours'] = self.config[k]
                elif k == 'cooldown_seconds':
                    self.config[k] = float(max(1.0, min(60.0, float(v))))
                elif k == 'random_queries' and isinstance(v, list) and v:
                    self.config[k] = [str(x).strip() for x in v if str(x).strip()]
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(self.config, ensure_ascii=False, indent=2))
            return dict(self.config)

    def _append_history(self, entry: dict):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open('a') as f:
            f.write(json.dumps({'time': time.time(), **entry}, ensure_ascii=False, default=float) + '\n')

    def recent_history(self, limit: int = 12) -> list[dict]:
        if not self.history_path.exists():
            return []
        try:
            lines = self.history_path.read_text().splitlines()[-limit:]
            return [json.loads(line) for line in reversed(lines) if line.strip()]
        except Exception:
            return []

    def _find_action_dirs(self, plan_id: str | None, action_log_dir: str | None = None) -> list[Path]:
        dirs: list[Path] = []
        if action_log_dir:
            p = Path(action_log_dir)
            if p.exists():
                dirs.append(p)
        base = self.runtime / 'action-logs'
        if plan_id and base.exists():
            for d in sorted(base.glob(f'*{plan_id}*')):
                if d.is_dir() and d not in dirs:
                    dirs.append(d)
        return dirs

    def diagnose_attempt(
        self,
        *,
        query: str,
        plan_summary: dict | None,
        exec_result: dict | None,
        error_text: str | None = None,
        cancel: threading.Event | None = None,
    ) -> dict:
        """Inspect recorded action logs, wrist images, and Intel RealSense D435 frames to diagnose a failed grasp."""
        cancel = cancel or threading.Event()
        plan_id = (exec_result or {}).get('plan_id') or (plan_summary or {}).get('plan_id')
        action_dir = (exec_result or {}).get('action_log_dir') or (plan_summary or {}).get('action_log_dir')
        dirs = self._find_action_dirs(plan_id, action_dir)

        wrist_frames = []
        d435_path = None
        events = []
        for d in dirs:
            ev_file = d / 'events.jsonl'
            if ev_file.exists():
                for line in ev_file.read_text().splitlines():
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            pass
            for wf in sorted(d.glob('*_wrist.jpg')):
                wrist_frames.append({'path': str(wf), 'id': wf.stem})
            for df in sorted(d.glob('*_d435.jpg')):
                if any(k in df.name for k in ('grasp', 'before_close', 'missed', 'error', 'plan')):
                    d435_path = str(df)

        obj = (exec_result or {}).get('object') or (plan_summary or {}).get('object') or {}
        grasp_info = (exec_result or {}).get('grasp') or (plan_summary or {}).get('grasp') or {}
        cur_tuning = self.kinematic.get_tuning(query)
        missed_empty = bool((exec_result or {}).get('missed_empty_close')) or ('空夾' in (error_text or '')) or ('抓空' in (error_text or ''))

        telemetry = {
            'query': query,
            'plan_id': plan_id,
            'object_measured': {
                'centre': obj.get('centre'),
                'radius_m': obj.get('radius'),
                'height_m': obj.get('height'),
                'posture': obj.get('posture', 'upright'),
                'lying_flat': obj.get('lying_flat', False),
            },
            'planned_grasp': grasp_info,
            'close': (exec_result or {}).get('close'),
            'lift': (exec_result or {}).get('lift'),
            'verification': (exec_result or {}).get('verification'),
            'missed_empty_close': missed_empty,
            'error': error_text or (exec_result or {}).get('error'),
            'current_tuning': cur_tuning,
        }

        vlm_diag = None
        if getattr(self.brain, 'available', False) and wrist_frames and not cancel.is_set():
            try:
                vlm_diag = self.brain.diagnose_grasp_failure(
                    query,
                    wrist_frames[-2:],
                    d435_path,
                    telemetry,
                    cancel,
                )
            except Exception:
                vlm_diag = None

        patch: dict = {}
        height_m = float(grasp_info.get('height_m') or 0.055)
        is_toothpick = self.kinematic._target_key(query) == 'toothpick_container'

        if vlm_diag:
            failure_mode = vlm_diag.get('failure_mode', 'other')
            posture = vlm_diag.get('object_posture', 'upright')
            summary_msg = vlm_diag.get('reason') or f'VLM 診斷：{failure_mode}'
            if posture == 'lying_flat':
                patch['force_posture'] = 'lying_flat'
            elif posture == 'upright' and cur_tuning.get('force_posture') == 'lying_flat':
                patch['force_posture'] = None
            h_delta = float(vlm_diag.get('height_offset_delta_mm') or 0.0)
            d_delta = float(vlm_diag.get('depth_offset_delta_mm') or 0.0)
            l_delta = float(vlm_diag.get('lateral_offset_delta_mm') or 0.0)
            if abs(h_delta) > 0.5:
                patch['height_offset_mm'] = float(cur_tuning.get('height_offset_mm', 0.0)) + h_delta
            if abs(d_delta) > 0.5:
                patch['depth_offset_mm'] = float(cur_tuning.get('depth_offset_mm', 0.0)) + d_delta
            if abs(l_delta) > 0.5:
                patch['lateral_offset_mm'] = float(cur_tuning.get('lateral_offset_mm', 0.0)) + l_delta
            if vlm_diag.get('increase_squeeze'):
                patch['extra_squeeze_ticks'] = int(cur_tuning.get('extra_squeeze_ticks', 0)) + 10
        else:
            failure_mode = 'missed_high_or_tipped' if missed_empty else 'shallow_or_slipped'
            summary_msg = '根據動作日誌與夾爪遙測自動分析'

        # Deterministic kinematic safeguard on top of VLM diagnosis:
        if missed_empty:
            if is_toothpick and height_m >= 0.045 and not obj.get('lying_flat'):
                patch['force_posture'] = 'lying_flat'
                patch['height_offset_mm'] = min(float(patch.get('height_offset_mm', 0.0)), -4.0)
                patch['depth_offset_mm'] = max(float(patch.get('depth_offset_mm', 0.0)), 6.0)
                summary_msg = (
                    f'偵測到抓空（原規劃夾取高度 {height_m * 1000:.0f} mm 過高，牙籤罐可能已倒下平放），'
                    '已自動切換為橫躺平放模型並降低夾取高度、加深夾取深度'
                )
            else:
                patch.setdefault('height_offset_mm', float(cur_tuning.get('height_offset_mm', 0.0)) - 10.0)
                patch.setdefault('depth_offset_mm', float(cur_tuning.get('depth_offset_mm', 0.0)) + 7.0)
                if not vlm_diag:
                    summary_msg = '偵測到合爪抓空，自動降低夾取高度 (-10 mm) 並往前加深 (+7 mm)'
        elif not (exec_result or {}).get('grasp_confirmed', False):
            patch.setdefault('depth_offset_mm', float(cur_tuning.get('depth_offset_mm', 0.0)) + 5.0)
            patch.setdefault('height_offset_mm', float(cur_tuning.get('height_offset_mm', 0.0)) - 4.0)
            patch.setdefault('extra_squeeze_ticks', int(cur_tuning.get('extra_squeeze_ticks', 0)) + 10)
            if not vlm_diag:
                summary_msg = '夾爪接觸後未能穩固提起物件，自動加深夾取深度 (+5 mm)、降低高度 (-4 mm) 並增加夾緊行程 (+10 ticks)'

        updated_tuning = self.kinematic.update_tuning(query, patch) if patch else cur_tuning
        diag = {
            'plan_id': plan_id,
            'query': query,
            'failure_mode': failure_mode,
            'summary': summary_msg,
            'vlm_diagnosis': vlm_diag,
            'applied_patch': patch,
            'updated_tuning': updated_tuning,
            'action_log_dirs': [str(d) for d in dirs],
            'd435_snapshot': d435_path,
        }
        self.loop_state['last_diagnosis'] = diag
        self._append_history({'type': 'diagnosis', **diag})
        return diag

    def execute_grasp_with_retry(
        self,
        query: str,
        report,
        cancel: threading.Event,
        *,
        direction_hint: str | None = None,
        return_object: bool = True,
        auto_retry: bool | None = None,
        max_retries: int | None = None,
    ) -> dict:
        """Run plan + execute, and if the grasp misses or fails, self-diagnose from logs/video and auto-retry."""
        if auto_retry is None:
            auto_retry = bool(self.config.get('auto_retry_enabled', True))
        if max_retries is None:
            max_retries = int(self.config.get('max_retries_per_grasp', 1)) if auto_retry else 0

        attempts = []
        last_plan = None
        last_exec = None
        last_diag = None

        for attempt_idx in range(max_retries + 1):
            if cancel.is_set():
                raise RuntimeError('已停止')
            if attempt_idx > 0:
                report(f'🔄 自動重試（第 {attempt_idx}/{max_retries} 次）：重新確認桌面環境與物件姿態…')
            plan = self.kinematic.plan(query, report, cancel, direction_hint=direction_hint)
            last_plan = plan
            if not plan.get('executable'):
                prob_str = '；'.join(plan.get('problems') or ['無法找到安全夾取姿勢'])
                diag = self.diagnose_attempt(
                    query=query,
                    plan_summary=plan,
                    exec_result=None,
                    error_text=prob_str,
                    cancel=cancel,
                )
                last_diag = diag
                attempts.append({'attempt': attempt_idx + 1, 'plan': plan, 'execution': None, 'diagnosis': diag})
                if attempt_idx < max_retries and not cancel.is_set():
                    report(f"🧠 Loop Engineer 自動診斷：{diag['summary']} → 正在自動重試…")
                    continue
                self.last_attempt = {
                    'query': query,
                    'direction_hint': direction_hint,
                    'return_object': return_object,
                    'plan': plan,
                    'execution': None,
                    'diagnosis': diag,
                    'success': False,
                    'time': time.time(),
                }
                return {'plan': plan, 'execution': None, 'diagnosis': diag, 'attempts': attempts}

            ex = self.kinematic.execute(plan['plan_id'], report, cancel, return_object=return_object)
            last_exec = ex
            if ex.get('grasp_confirmed'):
                # Successful grasp! If lying_flat was forced temporarily, keep working offsets but record success.
                attempt_rec = {'attempt': attempt_idx + 1, 'plan': plan, 'execution': ex, 'diagnosis': last_diag}
                attempts.append(attempt_rec)
                self.last_attempt = {
                    'query': query,
                    'direction_hint': direction_hint,
                    'return_object': return_object,
                    'plan': plan,
                    'execution': ex,
                    'diagnosis': last_diag,
                    'success': True,
                    'time': time.time(),
                }
                self._append_history({
                    'type': 'grasp_attempt',
                    'query': query,
                    'plan_id': plan['plan_id'],
                    'success': True,
                    'attempts_count': len(attempts),
                    'tuning': plan.get('tuning'),
                })
                return {'plan': plan, 'execution': ex, 'diagnosis': last_diag, 'attempts': attempts}

            # Grasp was not confirmed (e.g., missed_empty_close or slipped on lift)
            diag = self.diagnose_attempt(
                query=query,
                plan_summary=plan,
                exec_result=ex,
                error_text=ex.get('error') or '未夾起物件',
                cancel=cancel,
            )
            last_diag = diag
            attempts.append({'attempt': attempt_idx + 1, 'plan': plan, 'execution': ex, 'diagnosis': diag})
            self._append_history({
                'type': 'grasp_attempt',
                'query': query,
                'plan_id': plan['plan_id'],
                'success': False,
                'attempts_count': len(attempts),
                'diagnosis': diag['summary'],
            })
            if attempt_idx < max_retries and not cancel.is_set():
                report(f"🧠 Loop Engineer 自動診斷：{diag['summary']} → 自動調整參數並重試（{attempt_idx + 1}/{max_retries}）")
                continue

        self.last_attempt = {
            'query': query,
            'direction_hint': direction_hint,
            'return_object': return_object,
            'plan': last_plan,
            'execution': last_exec,
            'diagnosis': last_diag,
            'success': False,
            'time': time.time(),
        }
        return {'plan': last_plan, 'execution': last_exec, 'diagnosis': last_diag, 'attempts': attempts}

    # ------------------------------------------------------------------ Autonomous Loop Engineer
    def start_loop(
        self,
        *,
        mode: str | None = None,
        custom_query: str | None = None,
        duration_hours: float | None = -1,
    ) -> dict:
        patch = {}
        if mode in ('custom', 'random'):
            patch['mode'] = mode
        if custom_query is not None and custom_query.strip():
            patch['custom_query'] = custom_query.strip()
        if duration_hours != -1:
            patch['duration_hours'] = duration_hours
        if patch:
            self.update_config(patch)

        with self.lock:
            if self.loop_thread is not None and self.loop_thread.is_alive():
                return self.status()
            self.loop_cancel.clear()
            now = time.time()
            dur = self.config.get('duration_hours')
            ends_at = (now + float(dur) * 3600.0) if dur and float(dur) > 0 else None
            self.loop_state.update(
                running=True,
                mode=self.config.get('mode', 'custom'),
                custom_query=self.config.get('custom_query'),
                duration_hours=dur,
                started_at=now,
                ends_at=ends_at,
                iterations=0,
                successes=0,
                failures=0,
                current_query=None,
                stop_reason=None,
            )
            self.loop_thread = threading.Thread(target=self._loop_worker, daemon=True)
            self.loop_thread.start()
            return self.status()

    def stop_loop(self, reason: str = '使用者手動停止 Loop Engineer') -> dict:
        self.loop_cancel.set()
        self.loop_state['stop_reason'] = reason
        self.loop_state['running'] = False
        if self.cancel_active is not None:
            try:
                self.cancel_active()
            except Exception:
                pass
        return self.status()

    def _pick_next_query(self) -> str:
        if self.config.get('mode') == 'random':
            candidates = self.config.get('random_queries') or DEFAULT_RANDOM_QUERIES
            return random.choice(candidates)
        return self.config.get('custom_query') or DEFAULT_RANDOM_QUERIES[0]

    def _loop_worker(self):
        try:
            while not self.loop_cancel.is_set():
                ends_at = self.loop_state.get('ends_at')
                if ends_at is not None and time.time() >= float(ends_at):
                    self.loop_state['stop_reason'] = '已達到設定的自動優化運作時間'
                    break

                # Wait until job runner is free if another task is running
                if self.is_busy is not None and self.is_busy():
                    if self.loop_cancel.wait(1.0):
                        break
                    continue

                query = self._pick_next_query()
                self.loop_state['current_query'] = query
                self.loop_state['iterations'] = int(self.loop_state.get('iterations', 0)) + 1
                iter_num = self.loop_state['iterations']

                try:
                    if self.submit_job is not None:
                        done = threading.Event()
                        run_err: list[str] = []

                        def job_fn(j, report_fn):
                            try:
                                if self.robot.stop.is_set():
                                    self.robot.resume()
                                report_fn(f'🤖 Loop Engineer 第 {iter_num} 輪測試：{query}', 'planning')
                                out = self.execute_grasp_with_retry(
                                    query,
                                    lambda m: report_fn(m, 'moving'),
                                    j['cancel'],
                                    return_object=True,
                                    auto_retry=True,
                                    max_retries=max(1, int(self.config.get('max_retries_per_grasp', 1))),
                                )
                                ex = out.get('execution') or {}
                                j['engineering_result'] = {
                                    **ex,
                                    'task': 'grasp',
                                    'plan': out.get('plan'),
                                    'diagnosis': out.get('diagnosis'),
                                    'attempts': out.get('attempts'),
                                }
                                ok = bool(ex.get('grasp_confirmed'))
                                if ok:
                                    self.loop_state['successes'] = int(self.loop_state.get('successes', 0)) + 1
                                    report_fn(f'✅ Loop Engineer 第 {iter_num} 輪成功夾起並放回！', 'done')
                                else:
                                    self.loop_state['failures'] = int(self.loop_state.get('failures', 0)) + 1
                                    diag_msg = (out.get('diagnosis') or {}).get('summary') or '未夾起'
                                    report_fn(f'🛠️ Loop Engineer 第 {iter_num} 輪未成功，已記錄優化：{diag_msg}', 'done')
                            except Exception as e:
                                run_err.append(str(e))
                                self.loop_state['failures'] = int(self.loop_state.get('failures', 0)) + 1
                                raise
                            finally:
                                done.set()

                        self.submit_job(job_fn)
                        while not done.wait(0.5):
                            if self.loop_cancel.is_set():
                                if self.cancel_active is not None:
                                    self.cancel_active()
                                break
                        if run_err and _is_token_or_quota_exhausted(run_err[0]):
                            self.loop_state['stop_reason'] = f'API Token / 額度已用盡，自動停止：{run_err[0]}'
                            break
                    else:
                        out = self.execute_grasp_with_retry(
                            query,
                            lambda _m: None,
                            self.loop_cancel,
                            return_object=True,
                            auto_retry=True,
                        )
                        if (out.get('execution') or {}).get('grasp_confirmed'):
                            self.loop_state['successes'] = int(self.loop_state.get('successes', 0)) + 1
                        else:
                            self.loop_state['failures'] = int(self.loop_state.get('failures', 0)) + 1
                except Exception as exc:
                    if _is_token_or_quota_exhausted(str(exc)):
                        self.loop_state['stop_reason'] = f'API Token / 額度已用盡，自動停止：{exc}'
                        break

                cooldown = float(self.config.get('cooldown_seconds', 4.0))
                if self.loop_cancel.wait(cooldown):
                    break
        finally:
            self.loop_state['running'] = False
            self.loop_state['current_query'] = None

    def status(self) -> dict:
        tunings = {}
        tpath = self.kinematic._tuning_path()
        if tpath.exists():
            try:
                tunings = json.loads(tpath.read_text())
            except Exception:
                tunings = {}
        now = time.time()
        ends_at = self.loop_state.get('ends_at')
        remaining_s = max(0.0, float(ends_at) - now) if (self.loop_state.get('running') and ends_at) else None
        return {
            'config': dict(self.config),
            'loop': {
                **self.loop_state,
                'remaining_seconds': remaining_s,
            },
            'last_attempt': self.last_attempt,
            'tunings': tunings,
            'recent_history': self.recent_history(10),
        }
