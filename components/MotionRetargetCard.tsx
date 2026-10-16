'use client';

import React, { useEffect, useRef, useState } from 'react';
import {
  ArrowUpRight,
  Check,
  Hand,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';

export interface MotionArmSelection {
  selected_arm: 'right' | 'left';
  selection_reason: string;
  right_wrist_path_m: number;
  left_wrist_path_m: number;
  right_score: number;
  left_score: number;
}

export interface MotionWaypoint {
  frame_index: number;
  time_s: number;
  joints: Record<string, number>;
  tip_m: [number, number, number];
  clearance_mm: number;
}

export interface MotionGPTPlan {
  id: string;
  query: string;
  title_zh: string;
  summary_zh: string;
  english_prompt: string;
  planning_ms: number;
  skeleton_format: string;
  fps: number;
  num_frames: number;
  duration_s: number;
  engine: string;
  arm_selection: MotionArmSelection;
  human_edges: [number, number][];
  active_arm_chain: [number, number, number];
  preview_frame_indices: number[];
  skeleton_frames: number[][][];
  so101_frames: Record<string, number>[];
  tip_trajectory_m: [number, number, number][];
  waypoints: MotionWaypoint[];
  joint_ranges_deg: Record<string, [number, number]>;
  min_table_clearance_mm: number;
  table_clearance_passed: boolean;
}

export interface MotionGPTReceipt {
  completed?: boolean;
  motion_id?: string;
  title_zh?: string;
  english_prompt?: string;
  engine?: string;
  selected_arm?: string;
  executed_waypoints?: number;
  min_table_clearance_mm?: number;
  measured_roll_span_deg?: number;
  measured_pan_span_deg?: number;
  measured_lift_span_deg?: number;
}

export const QUICK_MOTION_CHIPS = [
  { id: 'right-wave', query: '揮揮手', label: '👋 揮揮手' },
  { id: 'stretch', query: '伸懶腰', label: '🙆‍♂️ 伸懶腰' },
  { id: 'boxing', query: '打一套拳擊', label: '🥊 打一套拳' },
  { id: 'dance', query: '開心跳舞律動', label: '🕺 開心律動' },
  { id: 'bow', query: '鞠躬敬禮', label: '🙇 鞠躬致意' },
  { id: 'cheer', query: '舉手歡呼', label: '🙌 舉手歡呼' },
  { id: 'nod', query: '點頭說好', label: '👍 點頭回應' },
  { id: 'beckon', query: '招手過來', label: '🫲 招手過來' },
];

const JOINT_LABELS: Record<string, string> = {
  shoulder_pan: '底座旋轉 (Pan)',
  shoulder_lift: '肩部俯仰 (Lift)',
  elbow_flex: '肘部伸展 (Elbow)',
  wrist_flex: '手腕俯仰 (Pitch)',
  wrist_roll: '手腕翻轉 (Roll)',
};

export function MotionRetargetCard({
  plan,
  receipt,
  preferredArm,
  onChangeArm,
  onSelectQuery,
  onExecute,
  canExecute,
  executeReason,
  executing,
}: {
  plan: MotionGPTPlan;
  receipt?: MotionGPTReceipt | null;
  preferredArm: 'auto' | 'right' | 'left';
  onChangeArm: (arm: 'auto' | 'right' | 'left') => void;
  onSelectQuery: (query: string) => void;
  onExecute: () => void;
  canExecute: boolean;
  executeReason: string;
  executing: boolean;
}) {
  const [playing, setPlaying] = useState(true);
  const [frameIdx, setFrameIdx] = useState(0);
  const humanCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const robotCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const totalPreviewFrames = plan.skeleton_frames?.length || 1;

  useEffect(() => {
    setFrameIdx(0);
    setPlaying(true);
  }, [plan.id, plan.query, plan.arm_selection?.selected_arm]);

  useEffect(() => {
    if (!playing || totalPreviewFrames <= 1) return;
    const intervalMs = Math.max(45, Math.round((plan.duration_s * 1000) / totalPreviewFrames));
    const timer = setInterval(() => {
      setFrameIdx((prev) => (prev + 1) % totalPreviewFrames);
    }, intervalMs);
    return () => clearInterval(timer);
  }, [playing, totalPreviewFrames, plan.duration_s]);

  // Draw Human 22-joint 3D skeleton with active single-arm highlighted
  useEffect(() => {
    const canvas = humanCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Subtle background grid
    ctx.fillStyle = '#f8fafd';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#e8eef5';
    ctx.lineWidth = 1;
    for (let x = 24; x < w; x += 28) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 24; y < h; y += 28) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const joints = plan.skeleton_frames?.[frameIdx] || plan.skeleton_frames?.[0];
    if (!joints || joints.length < 22) return;

    const pelvis = joints[0] || [0, 0.9, 0];
    const scale = h * 0.46;
    const cx = w * 0.5;
    const cy = h * 0.58;

    // Slight 3D oblique projection so both frontal X/Y and forward Z depth are visible
    const project = (pt: number[]): [number, number] => {
      const dx = (pt[0] - pelvis[0]) + 0.22 * (pt[2] - pelvis[2]);
      const dy = (pt[1] - pelvis[1]) - 0.12 * (pt[2] - pelvis[2]);
      return [cx - dx * scale, cy - dy * scale];
    };

    const [shIdx, elIdx, wrIdx] = plan.active_arm_chain || [17, 19, 21];
    const activeSet = new Set<number>([shIdx, elIdx, wrIdx]);

    // Draw wrist trail across all preview frames
    ctx.beginPath();
    plan.skeleton_frames.forEach((f, idx) => {
      const [px, py] = project(f[wrIdx]);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.strokeStyle = 'rgba(26, 115, 232, 0.28)';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Draw skeleton bones
    for (const [a, b] of plan.human_edges || []) {
      if (!joints[a] || !joints[b]) continue;
      const [x1, y1] = project(joints[a]);
      const [x2, y2] = project(joints[b]);
      const isActiveArm = activeSet.has(a) && activeSet.has(b);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = isActiveArm ? '#1a73e8' : '#9aa0a6';
      ctx.lineWidth = isActiveArm ? 4.5 : 2;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // Draw joints
    joints.forEach((pt, idx) => {
      const [px, py] = project(pt);
      const isActive = activeSet.has(idx);
      ctx.beginPath();
      ctx.arc(px, py, isActive ? 5.2 : 2.8, 0, Math.PI * 2);
      ctx.fillStyle = idx === wrIdx ? '#ea4335' : isActive ? '#1a73e8' : '#5f6368';
      ctx.fill();
    });

    // Caption inside canvas
    ctx.fillStyle = '#3c4043';
    ctx.font = '600 12px Arial, sans-serif';
    ctx.fillText(
      `HumanML3D 22 關節 · 擷取${plan.arm_selection?.selected_arm === 'left' ? '左手' : '右手'}`,
      12,
      20,
    );
  }, [plan, frameIdx]);

  // Draw SO-101 5-DOF retargeted arm & 3D trajectory above table plane
  useEffect(() => {
    const canvas = robotCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = '#f8fafd';
    ctx.fillRect(0, 0, w, h);

    const tableY = h - 34;
    const safetyY = tableY - 26;

    // Draw calibrated table surface & >=55mm safety plane
    ctx.fillStyle = '#eef2f6';
    ctx.fillRect(0, tableY, w, h - tableY);
    ctx.strokeStyle = '#bdc1c6';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, tableY);
    ctx.lineTo(w, tableY);
    ctx.stroke();

    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#34a853';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(12, safetyY);
    ctx.lineTo(w - 12, safetyY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#137333';
    ctx.font = '11px Arial, sans-serif';
    ctx.fillText(`3D 桌面安全淨空線 (最低實測 ${plan.min_table_clearance_mm.toFixed(1)} mm)`, 12, tableY - 8);

    const currentJoints = plan.so101_frames?.[frameIdx] || plan.so101_frames?.[0];
    if (!currentJoints) return;

    // Schematic kinematic chain of SO-101 in side-oblique view
    const baseX = w * 0.28;
    const baseY = tableY - 8;
    const l1 = 68;
    const l2 = 64;
    const l3 = 44;

    const panRad = ((currentJoints.shoulder_pan - 2028) * Math.PI) / 2048;
    const liftRad = ((currentJoints.shoulder_lift - 1520) * Math.PI) / 2048;
    const elbowRad = ((currentJoints.elbow_flex - 2220) * Math.PI) / 2048;
    const wristRad = ((currentJoints.wrist_flex - 2060) * Math.PI) / 2048;

    const a1 = -Math.PI / 2 + 0.35 + liftRad * 0.9;
    const elbowX = baseX + Math.cos(a1) * l1 + Math.sin(panRad) * 18;
    const elbowY = baseY + Math.sin(a1) * l1;

    const a2 = a1 + 0.72 + elbowRad * 0.85;
    const wristX = elbowX + Math.cos(a2) * l2 + Math.sin(panRad) * 26;
    const wristY = elbowY + Math.sin(a2) * l2;

    const a3 = a2 + wristRad * 0.8;
    const tipX = wristX + Math.cos(a3) * l3 + Math.sin(panRad) * 14;
    const tipY = Math.min(safetyY - 4, wristY + Math.sin(a3) * l3);

    // Draw tip trajectory across all frames
    ctx.beginPath();
    plan.so101_frames.forEach((jf, idx) => {
      const pR = ((jf.shoulder_pan - 2028) * Math.PI) / 2048;
      const lR = ((jf.shoulder_lift - 1520) * Math.PI) / 2048;
      const eR = ((jf.elbow_flex - 2220) * Math.PI) / 2048;
      const wR = ((jf.wrist_flex - 2060) * Math.PI) / 2048;
      const ang1 = -Math.PI / 2 + 0.35 + lR * 0.9;
      const ex = baseX + Math.cos(ang1) * l1 + Math.sin(pR) * 18;
      const ey = baseY + Math.sin(ang1) * l1;
      const ang2 = ang1 + 0.72 + eR * 0.85;
      const wx = ex + Math.cos(ang2) * l2 + Math.sin(pR) * 26;
      const wy = ey + Math.sin(ang2) * l2;
      const ang3 = ang2 + wR * 0.8;
      const tx = wx + Math.cos(ang3) * l3 + Math.sin(pR) * 14;
      const ty = Math.min(safetyY - 4, wy + Math.sin(ang3) * l3);
      if (idx === 0) ctx.moveTo(tx, ty);
      else ctx.lineTo(tx, ty);
    });
    ctx.strokeStyle = 'rgba(52, 168, 83, 0.35)';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Base pedestal
    ctx.fillStyle = '#3c4043';
    ctx.fillRect(baseX - 18, baseY - 6, 36, 14);

    // Arm links
    const segments: [[number, number], [number, number], string, number][] = [
      [[baseX, baseY - 4], [elbowX, elbowY], '#1a73e8', 7],
      [[elbowX, elbowY], [wristX, wristY], '#4285f4', 5.5],
      [[wristX, wristY], [tipX, tipY], '#34a853', 4],
    ];
    for (const [[x1, y1], [x2, y2], color, width] of segments) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // Joints & gripper roll indicator
    for (const [jx, jy, col] of [
      [baseX, baseY - 4, '#202124'],
      [elbowX, elbowY, '#1a73e8'],
      [wristX, wristY, '#fbbc04'],
      [tipX, tipY, '#ea4335'],
    ] as [number, number, string][]) {
      ctx.beginPath();
      ctx.arc(jx, jy, 5, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
    }

    const tipM = plan.tip_trajectory_m?.[frameIdx] || [0.26, 0.0, 0.36];
    ctx.fillStyle = '#3c4043';
    ctx.font = '600 12px Arial, sans-serif';
    ctx.fillText(
      `LeRobot SO-101 5軸重映射 · 末端 (${(tipM[0] * 100).toFixed(0)}, ${(tipM[1] * 100).toFixed(0)}, ${(tipM[2] * 100).toFixed(0)}) cm`,
      12,
      20,
    );
  }, [plan, frameIdx]);

  const currentJoints = plan.so101_frames?.[frameIdx] || plan.so101_frames?.[0] || {};

  return (
    <section
      className="suggestion motiongpt-card"
      aria-label="MotionGPT 單手動作重映射"
      style={{ marginBottom: 28 }}
    >
      <div className="sponsored" style={{ justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Sparkles size={15} />
          <strong>MotionGPT 單手動作重映射</strong>
          <span>· AIRI HumanML3D → LeRobot SO-101</span>
        </div>
        <span className="motion-engine-badge">
          <ShieldCheck size={14} />
          淨空 {plan.min_table_clearance_mm.toFixed(1)} mm · {plan.engine}
        </span>
      </div>

      <div className="motion-header-row">
        <div>
          <h2 style={{ margin: '6px 0 4px' }}>{plan.title_zh}</h2>
          <p style={{ margin: 0, color: '#5f6368', fontSize: 14 }}>{plan.summary_zh}</p>
          <p className="motion-prompt-line">
            <span>T5 Motion Prompt:</span> <code>{plan.english_prompt}</code>
          </p>
        </div>

        <div className="arm-selector" role="group" aria-label="選擇映射的人類單手">
          {(['auto', 'right', 'left'] as const).map((arm) => (
            <button
              key={arm}
              type="button"
              className={preferredArm === arm ? 'active' : ''}
              onClick={() => onChangeArm(arm)}
            >
              {arm === 'auto' ? '自動慣用手' : arm === 'right' ? '右手 Right' : '左手 Left'}
            </button>
          ))}
        </div>
      </div>

      <div className="motion-Preset-row">
        {QUICK_MOTION_CHIPS.map((chip) => (
          <button
            key={chip.id}
            type="button"
            className={'motion-chip ' + (plan.id === chip.id ? 'active' : '')}
            onClick={() => onSelectQuery(chip.query)}
          >
            {chip.label}
          </button>
        ))}
      </div>

      <div className="motion-canvases">
        <div className="canvas-box">
          <canvas ref={humanCanvasRef} width={320} height={220} />
          <div className="canvas-foot">
            <span>
              {plan.arm_selection.selection_reason}（右腕路徑 {plan.arm_selection.right_wrist_path_m}m / 左腕{' '}
              {plan.arm_selection.left_wrist_path_m}m）
            </span>
          </div>
        </div>
        <div className="canvas-box">
          <canvas ref={robotCanvasRef} width={320} height={220} />
          <div className="canvas-foot">
            <span>
              SO-101 關節即時預覽 · Pan {currentJoints.shoulder_pan} / Lift {currentJoints.shoulder_lift} / Roll{' '}
              {currentJoints.wrist_roll}
            </span>
          </div>
        </div>
      </div>

      <div className="motion-scrubber">
        <button
          type="button"
          className="scrub-btn"
          onClick={() => setPlaying((p) => !p)}
          aria-label={playing ? '暫停預覽' : '播放預覽'}
        >
          {playing ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0, totalPreviewFrames - 1)}
          value={frameIdx}
          onChange={(e) => {
            setPlaying(false);
            setFrameIdx(Number(e.target.value));
          }}
          aria-label="動作時間軸"
        />
        <span className="scrub-time">
          Frame {plan.preview_frame_indices?.[frameIdx] ?? frameIdx} / {plan.num_frames} · {plan.duration_s}s ({plan.fps}{' '}
          FPS)
        </span>
      </div>

      <div className="motion-joint-grid">
        {Object.entries(plan.joint_ranges_deg || {}).map(([joint, [minDeg, maxDeg]]) => (
          <div key={joint} className="joint-stat">
            <small>{JOINT_LABELS[joint] || joint}</small>
            <strong>
              {minDeg > 0 ? `+${minDeg.toFixed(1)}` : minDeg.toFixed(1)}° ~{' '}
              {maxDeg > 0 ? `+${maxDeg.toFixed(1)}` : maxDeg.toFixed(1)}°
            </strong>
          </div>
        ))}
      </div>

      <div className="action-pills" style={{ marginTop: 16, alignItems: 'center' }}>
        <button
          type="button"
          disabled={!canExecute || executing}
          title={executeReason}
          onClick={onExecute}
          style={{ background: canExecute ? '#1a73e8' : undefined, color: canExecute ? '#fff' : undefined }}
        >
          {executing ? <LoaderCircle size={16} className="spin" /> : <Hand size={16} />}
          <span>在 LeRobot SO-101 執行此單手動作 ({plan.waypoints.length} 個平滑路徑點)</span>
          {canExecute && <ArrowUpRight size={15} />}
        </button>
        <button
          type="button"
          onClick={() => {
            setFrameIdx(0);
            setPlaying(true);
          }}
        >
          <RotateCcw size={15} />
          <span>重播 3D 預覽</span>
        </button>
      </div>

      {!canExecute && <p className="calibration-note">實體執行待命：{executeReason}（上方 3D 重映射預覽可隨時互動操作）</p>}

      {receipt?.completed && (
        <div className="performed" style={{ marginTop: 14 }}>
          <Check size={18} />
          <div>
            <strong>已完成單手重映射動作：{receipt.title_zh || plan.title_zh}</strong>
            <p style={{ margin: '4px 0 0', fontSize: 13 }}>
              執行 {receipt.executed_waypoints} 個路徑點 · 底座擺幅 {receipt.measured_pan_span_deg}° · 肩部抬升{' '}
              {receipt.measured_lift_span_deg}° · 手腕翻轉 {receipt.measured_roll_span_deg}° · 桌面最小淨空{' '}
              {receipt.min_table_clearance_mm} mm
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
