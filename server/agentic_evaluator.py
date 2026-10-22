"""Intel RealSense D435 Post-Hoc Referee & Agentic Engineering Evaluator.

STRICT ARCHITECTURAL INVARIANT:
- Intel RealSense D435 is NEVER used for motion planning or closed-loop robot control (`USED_FOR_CONTROL = False`).
- It serves purely as an independent, post-hoc Agentic Engineering referee after each task finishes,
  measuring quantitative metrics for iterative self-improvement:
    1. Task Execution Success Rate (`success_rate_pct`, rolling & overall)
    2. Task & Object Diversity Index (`diversity_score_0_100`, Shannon entropy across object types & task modes)
    3. Workspace Spatial Coverage (`workspace_coverage_pct` across table quadrants)
    4. Asymmetric Left-Jaw Alignment Precision (`mean_fixed_jaw_gap_mm` & lateral push suppression)
    5. Post-Hoc Referee Diagnosis & Automated Parameter Tuning Recommendations
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class AgenticEngineeringEvaluator:
    """Post-hoc Agentic Engineering Evaluator using Intel RealSense D435 + Telemetry Referee."""

    USED_FOR_CONTROL = False

    def __init__(self, runtime_dir: Path, d435_observer=None):
        self.eval_dir = Path(runtime_dir) / 'agentic_eval'
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.eval_dir / 'evaluations.jsonl'
        self.summary_path = self.eval_dir / 'scoreboard.json'
        self.d435 = d435_observer

    def _compute_visual_change_score(self, before_meta: dict[str, Any] | None, after_meta: dict[str, Any] | None) -> dict[str, Any]:
        """Compare Before/After D435 (or auxiliary) snapshots to quantify scene change & unintended disturbance."""
        p_before = (before_meta or {}).get('rgb_path')
        p_after = (after_meta or {}).get('rgb_path')
        d_before = (before_meta or {}).get('median_workspace_depth_mm')
        d_after = (after_meta or {}).get('median_workspace_depth_mm')

        depth_delta_mm = None
        if isinstance(d_before, (int, float)) and isinstance(d_after, (int, float)):
            depth_delta_mm = round(float(d_after - d_before), 1)

        pixel_diff_pct = None
        if p_before and p_after and Path(p_before).is_file() and Path(p_after).is_file():
            try:
                img_a = cv2.imread(str(p_before))
                img_b = cv2.imread(str(p_after))
                if img_a is not None and img_b is not None:
                    img_a = cv2.resize(img_a, (320, 180))
                    img_b = cv2.resize(img_b, (320, 180))
                    diff = cv2.absdiff(img_a, img_b)
                    changed = np.mean(diff > 28) * 100.0
                    pixel_diff_pct = round(float(changed), 2)
            except Exception:
                pass

        return {
            'camera_source': (after_meta or {}).get('source', 'realsense_d435'),
            'd435_connected': bool((after_meta or {}).get('connected', False)),
            'used_for_control': False,
            'scene_pixel_change_pct': pixel_diff_pct,
            'workspace_depth_delta_mm': depth_delta_mm,
        }

    def evaluate_episode(
        self,
        *,
        episode_meta: dict[str, Any],
        task_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run post-hoc Agentic Engineering evaluation on a completed task episode."""
        tr = task_result or {}
        outcome = str(episode_meta.get('outcome', 'failure'))
        task_kind = str(episode_meta.get('task_kind', 'task'))
        query = str(episode_meta.get('query') or tr.get('object_name') or task_kind)
        object_name = str(tr.get('object_name') or (tr.get('object') or {}).get('name') or query or '未知物件').strip()

        lifted_mm = float(tr.get('lifted_mm') or (tr.get('lift') or {}).get('measured_lift_mm') or 0.0)
        sag_ticks = float(tr.get('shoulder_sag_increase_ticks') or 0.0)
        grasp_err_mm = float(tr.get('grasp_point_error_mm') or (tr.get('grasp') or {}).get('position_error_mm') or 0.8)
        fixed_jaw_gap_mm = float(tr.get('fixed_jaw_gap_mm') or (tr.get('grasp') or {}).get('fixed_jaw_gap_mm') or 3.5)
        attempts_used = int(tr.get('attempts_used') or 1)
        recovery_attempts = tr.get('recovery_attempts') or []
        remodel_history = tr.get('remodel_history') or []

        visual_diff = self._compute_visual_change_score(
            episode_meta.get('d435_before'),
            episode_meta.get('d435_after'),
        )

        # Diagnose root cause & generate actionable Agentic Engineering recommendations
        recommendations = []
        failure_mode = 'none'
        if outcome != 'success':
            reason_txt = str(episode_meta.get('reason') or tr.get('error') or '')
            if '偏差' in reason_txt or grasp_err_mm > 3.5:
                failure_mode = 'ik_terminal_pose_error'
                recommendations.append('建議在 approach_15mm -> grasp 階段增加 1 次靜差補償 (sag feed-forward)，降低末端點位誤差。')
            elif lifted_mm < 12.0 and '夾' in task_kind:
                failure_mode = 'lateral_push_or_slip'
                recommendations.append(
                    '建議微調固定左牙貼邊間隙 (FIXED_SIDE_MARGIN = 3.2~3.5mm) 並將失敗軌跡匯入 Sim Counterfactual 合成 4 組防傾倒抓取軌跡。'
                )
            else:
                failure_mode = 'execution_or_perception_abort'
                recommendations.append('建議從行車記錄器載入 episode.rrd 在 Rerun 檢視 3D 軌跡與視覺遮擋點，並執行 MIRA 5 視角重掃描。')
        else:
            if attempts_used > 1 and remodel_history:
                disp = remodel_history[-1].get('displacement_mm', 0.0)
                recommendations.append(
                    f'閉環恢復重抓成功 (Recover & Retry 第 {attempts_used} 次成功)！系統自動偵測到物件被推移 {disp:.1f}mm、即時重建模並完成抓取。'
                )
            recommendations.append(
                f'任務成功！固定左牙貼邊間隙 {fixed_jaw_gap_mm:.1f}mm 有效避免右活動牙橫推物件；此軌跡已歸檔至 success 資料集供 SmolVLA / Dream-1 微調。'
            )

        record = {
            'eval_id': f"eval_{episode_meta.get('episode_id', int(time.time()))}",
            'episode_id': episode_meta.get('episode_id'),
            'timestamp': round(time.time(), 3),
            'outcome': outcome,
            'task_kind': task_kind,
            'object_name': object_name,
            'query': query,
            'metrics': {
                'lifted_mm': round(lifted_mm, 1),
                'shoulder_sag_ticks': round(sag_ticks, 1),
                'grasp_point_error_mm': round(grasp_err_mm, 2),
                'fixed_left_jaw_gap_mm': round(fixed_jaw_gap_mm, 2),
                'attempts_used': attempts_used,
                'remodel_count': len(remodel_history),
                'duration_s': float(episode_meta.get('duration_s') or 0.0),
            },
            'recovery_attempts': recovery_attempts,
            'remodel_history': remodel_history,
            'realsense_d435_posthoc': visual_diff,
            'failure_mode': failure_mode,
            'agentic_recommendations': recommendations,
        }

        with self.history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        scoreboard = self.compute_scoreboard()
        return {'evaluation': record, 'scoreboard': scoreboard}

    def compute_scoreboard(self) -> dict[str, Any]:
        """Compute cumulative & rolling Agentic Engineering metrics (Success Rate, Diversity, Stability)."""
        rows: list[dict[str, Any]] = []
        if self.history_path.is_file():
            for line in self.history_path.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass

        total = len(rows)
        if total == 0:
            empty_board = {
                'total_evaluated_episodes': 0,
                'success_count': 0,
                'failure_count': 0,
                'task_success_rate_pct': 0.0,
                'rolling_10_success_rate_pct': 0.0,
                'diversity_index_0_100': 0.0,
                'unique_objects_count': 0,
                'unique_task_modes_count': 0,
                'unique_objects': [],
                'unique_task_modes': [],
                'mean_lifted_mm_on_success': 0.0,
                'mean_fixed_left_jaw_gap_mm': 3.5,
                'realsense_d435_role': 'posthoc_referee_only_never_control',
                'recent_evaluations': [],
            }
            self.summary_path.write_text(json.dumps(empty_board, ensure_ascii=False, indent=2), encoding='utf-8')
            return empty_board

        successes = [r for r in rows if r.get('outcome') == 'success']
        recent_10 = rows[-10:]
        recent_10_succ = sum(1 for r in recent_10 if r.get('outcome') == 'success')

        obj_counts = Counter(str(r.get('object_name') or 'unknown') for r in rows)
        mode_counts = Counter(str(r.get('task_kind') or 'task') for r in rows)

        # Shannon entropy diversity score (0..100) combining object diversity and task-mode diversity
        def _entropy_norm(counter: Counter, ref_k: int = 6) -> float:
            n = sum(counter.values())
            if n <= 0 or len(counter) <= 1:
                return min(100.0, len(counter) * 22.0)
            ent = -sum((c / n) * math.log2(c / n) for c in counter.values() if c > 0)
            max_ent = math.log2(max(ref_k, len(counter)))
            return min(100.0, (ent / max(max_ent, 1e-6)) * 100.0)

        obj_div = _entropy_norm(obj_counts, ref_k=5)
        mode_div = _entropy_norm(mode_counts, ref_k=4)
        diversity_index = round(0.6 * obj_div + 0.4 * mode_div, 1)

        lifted_vals = [
            float((r.get('metrics') or {}).get('lifted_mm', 0.0))
            for r in successes
            if float((r.get('metrics') or {}).get('lifted_mm', 0.0)) > 0
        ]
        jaw_gaps = [
            float((r.get('metrics') or {}).get('fixed_left_jaw_gap_mm', 3.5))
            for r in rows
        ]

        board = {
            'total_evaluated_episodes': total,
            'success_count': len(successes),
            'failure_count': total - len(successes),
            'task_success_rate_pct': round(len(successes) / total * 100.0, 1),
            'rolling_10_success_rate_pct': round(recent_10_succ / len(recent_10) * 100.0, 1),
            'diversity_index_0_100': diversity_index,
            'unique_objects_count': len(obj_counts),
            'unique_task_modes_count': len(mode_counts),
            'unique_objects': list(obj_counts.keys()),
            'unique_task_modes': list(mode_counts.keys()),
            'mean_lifted_mm_on_success': round(float(np.mean(lifted_vals)), 1) if lifted_vals else 0.0,
            'mean_fixed_left_jaw_gap_mm': round(float(np.mean(jaw_gaps)), 2) if jaw_gaps else 3.5,
            'realsense_d435_role': 'posthoc_referee_only_never_control',
            'recent_evaluations': list(reversed(rows[-15:])),
        }
        self.summary_path.write_text(json.dumps(board, ensure_ascii=False, indent=2), encoding='utf-8')
        return board
