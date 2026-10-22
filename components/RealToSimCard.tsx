'use client';
/* oxlint-disable next/no-img-element -- Live MIRA BEV maps must bypass image caching. */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import {
  Box,
  Check,
  Compass,
  Eye,
  Layers,
  LoaderCircle,
  MessageSquare,
  Pause,
  Play,
  RotateCcw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react';
import type { ThinkingLevel } from '@/components/ER2SpatialCard';

export interface SceneObjectPhysics {
  estimated_mass_g: number;
  material: string;
  friction_coeff: number;
  center_of_mass_z_ratio: number;
  rigidity: string;
  fragile: boolean;
  stackable: boolean;
}

export interface SceneObjectRecord {
  id: string;
  name: string;
  description: string;
  confidence: number;
  centre: [number, number, number];
  radius: number;
  bounding_radius: number;
  semi_major_m: number;
  semi_minor_m: number;
  height: number;
  yaw_deg: number;
  grasp_yaw_deg: number;
  shape_primitive: string;
  color_rgba: [number, number, number, number];
  crop_path?: string | null;
  bev_point: [number, number];
  mesh_glb_path?: string | null;
  observation_count?: number;
  physics?: SceneObjectPhysics;
}

export interface FreeSpaceSlot {
  slot_label: string;
  reference_object_id: string | null;
  reference_object_name: string;
  direction: string;
  position_m: [number, number, number];
  bev_point: [number, number];
}

export interface SpatialRelation {
  from_id: string;
  from_name: string;
  to_id: string;
  to_name: string;
  dx_mm: number;
  dy_mm: number;
  distance_mm: number;
  direction_zh: string;
}

export interface PoseStreamFrame {
  step: number;
  time_s: number;
  phase: string;
  joints: Record<string, number>;
  tcp_xyz: [number, number, number];
  table_clearance_mm: number;
  table_link: string;
  target_clearance_mm: number | null;
}

export interface TwinSimulationReport {
  passed: boolean;
  problems: string[];
  duration_s: number;
  num_frames: number;
  fps: number;
  min_table_clearance_mm: number;
  min_target_clearance_mm: number | null;
  min_neighbor_clearance_mm: number | null;
  worst_neighbor_name: string;
  physics_verification: {
    object_mass_g: number;
    max_friction_payload_g: number;
    narrow_axis_diameter_mm: number;
    jaw_aperture_mm: number;
    fit_margin_mm: number;
    stability_score: number;
  };
  pose_stream: PoseStreamFrame[];
}

export interface RealToSimPlan {
  plan_id: string;
  mode: 'grasp' | 'pick_and_place';
  instruction: string;
  object: SceneObjectRecord;
  place_target_m: [number, number, number] | null;
  place_label_zh: string;
  executable: boolean;
  problems: string[];
  simulation: TwinSimulationReport;
  pose_stream: PoseStreamFrame[];
}

export interface RealToSimStatus {
  scene_map: {
    object_count: number;
    objects: SceneObjectRecord[];
    relations: SpatialRelation[];
    free_space_slots: FreeSpaceSlot[];
    last_bev?: {
      width_px: number;
      height_px: number;
      resolution_mm_per_px: number;
      coverage_ratio: number;
      views_fused: number;
      synthesis_ms: number;
    } | null;
    last_summary_zh: string;
    last_updated: number;
  };
  active_plan: RealToSimPlan | null;
  cloud_providers: {
    fal_ai_configured: boolean;
    world_labs_configured: boolean;
    er2_configured: boolean;
  };
  qa_history: {
    id: string;
    question: string;
    answer_zh: string;
    latency_ms: number;
  }[];
  worldlabs_last?: {
    status: string;
    operation_id: string;
    model?: string;
  } | null;
  dashcam?: {
    running: boolean;
    window_minutes: number;
    sample_hz: number;
    buffered_frames: number;
    buffered_span_minutes: number;
    total_archived: number;
    success_count: number;
    failure_count: number;
    realsense_d435: {
      device: string;
      mode: string;
      connected: boolean;
      used_for_planning_or_control: boolean;
      role: string;
    };
    episodes: {
      episode_id: string;
      outcome: 'success' | 'failure';
      task_kind: string;
      query: string;
      reason: string;
      archived_at: number;
      duration_s: number;
      num_frames: number;
      rerun?: {
        rrd_generated: boolean;
        rrd_size_bytes?: number;
        rerun_cli_hint?: string;
      };
      files?: {
        episode_rrd: string;
        rosbag_jsonl: string;
        lerobot_json: string;
        sim_counterfactual_json: string;
      };
    }[];
  };
  agentic_eval?: {
    total_evaluated_episodes: number;
    success_count: number;
    failure_count: number;
    task_success_rate_pct: number;
    rolling_10_success_rate_pct: number;
    diversity_index_0_100: number;
    unique_objects_count: number;
    unique_task_modes_count: number;
    unique_objects: string[];
    unique_task_modes: string[];
    mean_lifted_mm_on_success: number;
    mean_fixed_left_jaw_gap_mm: number;
    realsense_d435_role: string;
    recent_evaluations: {
      eval_id: string;
      episode_id: string;
      outcome: 'success' | 'failure';
      task_kind: string;
      object_name: string;
      failure_mode: string;
      agentic_recommendations: string[];
      metrics: {
        lifted_mm: number;
        shoulder_sag_ticks: number;
        grasp_point_error_mm: number;
        fixed_left_jaw_gap_mm: number;
      };
    }[];
  };
  vla?: {
    dreamlabs: {
      api_key_configured: boolean;
      api_key_masked: string | boolean;
      sdk_installed: boolean;
      active_model: string;
      available_models: { id: string; name: string; description_zh: string }[];
    };
    smolvla: {
      sdk_installed: boolean;
      active_model: string;
      available_models: {
        id: string;
        name: string;
        tasks_zh: string;
        zero_shot_note: string;
        downloads: number;
      }[];
    };
    last_prediction?: Record<string, unknown> | null;
  };
}

interface RealToSimCardProps {
  request: <T>(path: string, body?: unknown) => Promise<T>;
  busy: boolean;
  thinkingLevel: ThinkingLevel;
  onJobStarted?: (job: unknown) => void;
}

function TwinViewport3D({
  revision,
  activeFrame,
}: {
  revision: number;
  activeFrame: PoseStreamFrame | null;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const tcpMarkerRef = useRef<THREE.Mesh | null>(null);
  const invalidateRef = useRef<(() => void) | null>(null);
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    if (tcpMarkerRef.current && activeFrame?.tcp_xyz) {
      tcpMarkerRef.current.position.set(
        activeFrame.tcp_xyz[0],
        activeFrame.tcp_xyz[1],
        activeFrame.tcp_xyz[2],
      );
      tcpMarkerRef.current.visible = true;
      invalidateRef.current?.();
    }
  }, [activeFrame]);

  useEffect(() => {
    const element = hostRef.current;
    if (!element) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      queueMicrotask(() => setLoadError('此瀏覽器未啟用 WebGL，請切換至 MIRA 正交俯視圖。'));
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x0f172a, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    element.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 2.2));
    const dirLight = new THREE.DirectionalLight(0xffffff, 2.6);
    dirLight.position.set(0.8, -0.7, 1.4);
    scene.add(dirLight);

    const camera = new THREE.PerspectiveCamera(42, 1, 0.005, 10);
    camera.up.set(0, 0, 1);
    camera.position.set(0.52, -0.42, 0.38);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0.22, 0.0, 0.08);
    controls.enableDamping = true;
    controls.minDistance = 0.14;
    controls.maxDistance = 1.8;

    const grid = new THREE.GridHelper(0.72, 18, 0x38bdf8, 0x1e293b);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(0.22, 0, 0.001);
    scene.add(grid);

    const axes = new THREE.AxesHelper(0.08);
    axes.position.set(0, 0, 0.003);
    scene.add(axes);

    // Live animated TCP playback sphere (`steer-a-robot` style pose indicator)
    const tcpGeo = new THREE.SphereGeometry(0.009, 20, 20);
    const tcpMat = new THREE.MeshStandardMaterial({
      color: 0x22c55e,
      emissive: 0x15803d,
      emissiveIntensity: 0.7,
      roughness: 0.2,
    });
    const tcpMesh = new THREE.Mesh(tcpGeo, tcpMat);
    tcpMesh.visible = false;
    scene.add(tcpMesh);
    tcpMarkerRef.current = tcpMesh;

    let active = true;
    let animId: number | null = null;
    const invalidate = () => {
      if (active && animId === null) {
        animId = requestAnimationFrame(render);
      }
    };
    function render() {
      animId = null;
      if (!active) return;
      const moved = controls.update();
      renderer.render(scene, camera);
      if (moved) invalidate();
    }
    invalidateRef.current = invalidate;
    controls.addEventListener('change', invalidate);

    const abort = new AbortController();
    fetch(`/api/real-to-sim/artifact/scene.glb?v=${revision}`, {
      signal: abort.signal,
      cache: 'no-store',
    })
      .then(async (res) => {
        if (!res.ok) throw new Error('請先點擊「MIRA 掃描建模」以生成 3D 數位孿生');
        return new GLTFLoader().parseAsync(await res.arrayBuffer(), '');
      })
      .then((gltf) => {
        if (!active) return;
        setLoadError('');
        scene.add(gltf.scene);
        invalidate();
      })
      .catch((err) => {
        if (active && err?.name !== 'AbortError') {
          setLoadError(err instanceof Error ? err.message : '尚未建立 3D 數位孿生');
        }
      });

    const ro = new ResizeObserver(() => {
      const w = Math.max(element.clientWidth, 200);
      const h = Math.max(element.clientHeight, 220);
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      invalidate();
    });
    ro.observe(element);
    invalidate();

    return () => {
      active = false;
      tcpMarkerRef.current = null;
      abort.abort();
      if (animId !== null) cancelAnimationFrame(animId);
      controls.removeEventListener('change', invalidate);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [revision]);

  return (
    <div className="relative h-[320px] w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
      <div ref={hostRef} className="h-full w-full" />
      <div className="pointer-events-none absolute top-2.5 left-3 flex flex-wrap items-center gap-1.5">
        <span className="rounded-md bg-slate-900/85 px-2 py-0.5 text-[11px] font-medium text-sky-300 backdrop-blur">
          Agentview 3D 數位孿生 (可拖曳旋轉)
        </span>
        {activeFrame && (
          <span className="rounded-md bg-emerald-950/90 px-2 py-0.5 text-[11px] font-medium text-emerald-300 backdrop-blur">
            預演階段: {activeFrame.phase} · 桌面淨空 {activeFrame.table_clearance_mm} mm
          </span>
        )}
      </div>
      {loadError && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 p-4 text-center text-xs text-slate-300">
          {loadError}
        </div>
      )}
    </div>
  );
}

export function RealToSimCard({
  request,
  busy,
  thinkingLevel,
  onJobStarted,
}: RealToSimCardProps) {
  const [status, setStatus] = useState<RealToSimStatus | null>(null);
  const [viewTab, setViewTab] = useState<'twin3d' | 'bev'>('twin3d');
  const [revision, setRevision] = useState<number>(1);
  const [scanning, setScanning] = useState(false);
  const [planning, setPlanning] = useState(false);
  const [reconstructingId, setReconstructingId] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  // Task steering & Pick-and-Place state
  const [selectedObjectId, setSelectedObjectId] = useState<string>('');
  const [selectedPlaceSlot, setSelectedPlaceSlot] = useState<string>('');
  const [instruction, setInstruction] = useState<string>('');
  const [qaInput, setQaInput] = useState<string>('');

  // VLA (Dream Labs & SmolVLA) & Dashcam state
  const [vlaEngine, setVlaEngine] = useState<'smolvla' | 'dreamlabs'>('smolvla');
  const [vlaCheckpoint, setVlaCheckpoint] = useState<string>('lerobot/smolvla_base');
  const [vlaPrompt, setVlaPrompt] = useState<string>('pick up the white tube on the table');
  const [vlaRunning, setVlaRunning] = useState<boolean>(false);
  const [vlaResult, setVlaResult] = useState<Record<string, unknown> | null>(null);
  const [archivingOutcome, setArchivingOutcome] = useState<'success' | 'failure' | null>(null);

  // 25 Hz Mental Simulation playback state (`steer-a-robot` player style)
  const [playingSim, setPlayingSim] = useState(false);
  const [simStepIdx, setSimStepIdx] = useState(0);
  const wasBusyRef = useRef(busy);

  const refreshStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/real-to-sim/status', { cache: 'no-store' });
      if (!res.ok) return;
      const data = (await res.json()) as RealToSimStatus;
      setStatus(data);
      setSelectedObjectId((prev) => prev || (data.scene_map.objects[0]?.id ?? ''));
    } catch {
      // Ignore background fetch errors
    }
  }, []);

  useEffect(() => {
    if (wasBusyRef.current && !busy) {
      void refreshStatus().then(() => {
        setRevision((r) => r + 1);
      });
    }
    wasBusyRef.current = busy;
  }, [busy, refreshStatus]);

  useEffect(() => {
    let alive = true;
    fetch('/api/real-to-sim/status', { cache: 'no-store' })
      .then((res) => (res.ok ? (res.json() as Promise<RealToSimStatus>) : null))
      .then((data) => {
        if (!alive || !data) return;
        setStatus(data);
        setSelectedObjectId((prev) => prev || (data.scene_map.objects[0]?.id ?? ''));
      })
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, []);

  const activePlan = status?.active_plan ?? null;
  const poseStream = activePlan?.pose_stream ?? [];

  useEffect(() => {
    if (!playingSim || poseStream.length === 0) return;
    const timer = window.setInterval(() => {
      setSimStepIdx((prev) => {
        if (prev + 1 >= poseStream.length) {
          setPlayingSim(false);
          return 0;
        }
        return prev + 1;
      });
    }, 40); // 25 Hz playback
    return () => window.clearInterval(timer);
  }, [playingSim, poseStream.length]);

  const handleScanBev = async (mode: 'instant' | 'multiview') => {
    if (scanning || busy) return;
    setScanning(true);
    setErrorMsg('');
    try {
      const data = await request<Record<string, unknown>>('/api/real-to-sim/scan-bev', {
        mode,
        query: instruction.trim(),
        thinking_level: thinkingLevel,
      });
      if (mode === 'multiview' && data && 'id' in data) {
        onJobStarted?.(data);
      } else {
        await refreshStatus();
        setRevision((r) => r + 1);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'MIRA 掃描建模失敗');
    } finally {
      setScanning(false);
    }
  };

  const handlePlanTask = async () => {
    if (planning || busy) return;
    setPlanning(true);
    setErrorMsg('');
    try {
      const targetObj = status?.scene_map.objects.find((o) => o.id === selectedObjectId);
      const autoInstruction =
        instruction.trim() ||
        (selectedPlaceSlot && targetObj
          ? `把${targetObj.name}放到${selectedPlaceSlot}`
          : targetObj
            ? `夾起${targetObj.name}`
            : '夾起桌面物件');

      await request<RealToSimPlan>('/api/real-to-sim/plan', {
        instruction: autoInstruction,
        place_slot_label: selectedPlaceSlot || null,
        target_object_id: selectedObjectId || null,
        thinking_level: thinkingLevel,
      });
      await refreshStatus();
      setRevision((r) => r + 1);
      setSimStepIdx(0);
      setPlayingSim(true);
      setViewTab('twin3d');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : '沙盒預演規劃失敗');
    } finally {
      setPlanning(false);
    }
  };

  const handleExecutePlan = async () => {
    if (!activePlan || busy) return;
    setErrorMsg('');
    try {
      const job = await request<Record<string, unknown>>('/api/real-to-sim/execute', {
        plan_id: activePlan.plan_id,
        return_object_if_grasp_only: true,
      });
      onJobStarted?.(job);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : '啟動實體執行失敗');
    }
  };

  const handleReconstruct3D = async (target: 'object' | 'world', objectId?: string) => {
    if (reconstructingId) return;
    const tag = target === 'world' ? 'world' : objectId || 'obj';
    setReconstructingId(tag);
    setErrorMsg('');
    try {
      await request('/api/real-to-sim/reconstruct-3d', {
        target,
        object_id: objectId || selectedObjectId || null,
      });
      await refreshStatus();
      setRevision((r) => r + 1);
      setViewTab('twin3d');
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : '雲端 3D 生成失敗');
    } finally {
      setReconstructingId(null);
    }
  };

  const handleAskScene = async () => {
    if (!qaInput.trim() || asking) return;
    setAsking(true);
    setErrorMsg('');
    try {
      await request('/api/real-to-sim/ask', {
        question: qaInput.trim(),
        thinking_level: thinkingLevel,
      });
      setQaInput('');
      await refreshStatus();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : '場景問答失敗');
    } finally {
      setAsking(false);
    }
  };

  const handleSetDashcamWindow = async (minutes: 30 | 60) => {
    try {
      await request('/api/dashcam/config', { window_minutes: minutes });
      await refreshStatus();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : '更新行車記錄器視窗失敗');
    }
  };

  const handleArchiveEpisode = async (outcome: 'success' | 'failure') => {
    if (archivingOutcome) return;
    setArchivingOutcome(outcome);
    setErrorMsg('');
    try {
      await request('/api/dashcam/archive', {
        outcome,
        task_kind: 'manual_user_tag',
        query: instruction.trim() || '桌面操作任務',
        reason: outcome === 'success' ? '使用者手動標記任務成功' : '使用者手動標記任務失敗（供 Sim 反事實再訓練）',
        lookback_seconds: 60,
      });
      await refreshStatus();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : '歸檔行車記錄器片段失敗');
    } finally {
      setArchivingOutcome(null);
    }
  };

  const handleRunVlaZeroShot = async (executeOnRobot: boolean) => {
    if (vlaRunning || busy) return;
    setVlaRunning(true);
    setErrorMsg('');
    try {
      const ckpt =
        vlaEngine === 'dreamlabs'
          ? vlaCheckpoint.startsWith('dreamlabs/')
            ? vlaCheckpoint
            : 'dreamlabs/dream-1-beta'
          : vlaCheckpoint.startsWith('dreamlabs/')
            ? 'lerobot/smolvla_base'
            : vlaCheckpoint;
      const res = await request<Record<string, unknown>>('/api/vla/predict', {
        engine: vlaEngine,
        prompt: vlaPrompt.trim() || 'pick up the target object on the table',
        checkpoint: ckpt,
        execute_on_robot: executeOnRobot,
      });
      if (executeOnRobot && res && 'id' in res) {
        onJobStarted?.(res);
      } else {
        setVlaResult(res);
      }
      await refreshStatus();
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'VLA 模型推論失敗');
    } finally {
      setVlaRunning(false);
    }
  };

  const objects = status?.scene_map.objects ?? [];
  const freeSlots = status?.scene_map.free_space_slots ?? [];
  const relations = status?.scene_map.relations ?? [];
  const currentPoseFrame =
    poseStream.length > 0 ? poseStream[Math.min(simStepIdx, poseStream.length - 1)] : null;

  return (
    <section
      className="mt-4 rounded-2xl border border-indigo-500/25 bg-gradient-to-b from-slate-900/95 to-slate-950/95 p-4 text-slate-100 shadow-lg"
      aria-label="MIRA Real-to-Sim-to-Real 數位孿生工作台"
    >
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
        <div className="flex items-center gap-2">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-300">
            <Layers className="h-4 w-4" />
          </span>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold tracking-tight text-white">
                MIRA × Steer-a-Robot × SAM-3D 真實環境數位孿生沙盒
              </h3>
              <span className="rounded-full bg-indigo-500/20 px-2 py-0.5 text-[10px] font-medium text-indigo-300">
                Real-to-Sim-to-Real
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              多視角正交俯視圖 (1px=0.5mm) · 持久 3D 場景圖 · 出手前物理沙盒預演 · FAL / World Labs 3D 重建
            </p>
          </div>
        </div>

        {/* Cloud Provider Status Badges */}
        <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
          <span
            className={`rounded-md px-2 py-0.5 font-medium ${
              status?.cloud_providers.er2_configured
                ? 'bg-emerald-500/15 text-emerald-300'
                : 'bg-slate-800 text-slate-400'
            }`}
          >
            ER 2 空間腦
          </span>
          <span
            className={`rounded-md px-2 py-0.5 font-medium ${
              status?.cloud_providers.fal_ai_configured
                ? 'bg-sky-500/15 text-sky-300'
                : 'bg-slate-800 text-slate-400'
            }`}
          >
            FAL.ai 3D (.glb)
          </span>
          <span
            className={`rounded-md px-2 py-0.5 font-medium ${
              status?.cloud_providers.world_labs_configured
                ? 'bg-purple-500/15 text-purple-300'
                : 'bg-slate-800 text-slate-400'
            }`}
          >
            World Labs Marble
          </span>
        </div>
      </div>

      {/* Top Action Bar: MIRA Scan + View Switcher */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={scanning || busy}
            onClick={() => void handleScanBev('instant')}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white shadow-sm transition hover:bg-indigo-500 disabled:opacity-50"
          >
            {scanning ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Eye className="h-3.5 w-3.5" />}
            MIRA 單視角即時建模 (&lt;1s)
          </button>
          <button
            type="button"
            disabled={scanning || busy}
            onClick={() => void handleScanBev('multiview')}
            className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-400/35 bg-indigo-950/60 px-3 py-1.5 text-xs font-medium text-indigo-200 transition hover:bg-indigo-900/70 disabled:opacity-50"
          >
            <Compass className="h-3.5 w-3.5" />
            MIRA 5 視角主動環視建模
          </button>
          {status?.cloud_providers.world_labs_configured && (
            <button
              type="button"
              disabled={Boolean(reconstructingId) || busy}
              onClick={() => void handleReconstruct3D('world')}
              className="inline-flex items-center gap-1.5 rounded-lg border border-purple-400/35 bg-purple-950/50 px-2.5 py-1.5 text-xs font-medium text-purple-200 transition hover:bg-purple-900/60 disabled:opacity-50"
            >
              {reconstructingId === 'world' ? (
                <LoaderCircle className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Sparkles className="h-3.5 w-3.5" />
              )}
              World Labs 3D 環境生成
            </button>
          )}
        </div>

        {/* Viewport Toggle */}
        <div className="inline-flex rounded-lg border border-slate-800 bg-slate-900 p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setViewTab('twin3d')}
            className={`rounded-md px-2.5 py-1 font-medium transition ${
              viewTab === 'twin3d' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Agentview 3D 數位孿生
          </button>
          <button
            type="button"
            onClick={() => setViewTab('bev')}
            className={`rounded-md px-2.5 py-1 font-medium transition ${
              viewTab === 'bev' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            MIRA 正交俯視圖 (BEV)
          </button>
        </div>
      </div>

      {(scanning || planning || reconstructingId || asking) && (
        <div className="mt-2.5 flex items-center gap-2 rounded-lg border border-indigo-500/40 bg-indigo-950/60 px-3 py-2 text-xs text-indigo-200">
          <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin text-indigo-400" />
          <span>
            {scanning
              ? 'Argon ER 2 正在雙視角 (手腕 + MIRA 正交俯視圖) 深度推理空間與物理屬性…'
              : planning
                ? '正在數位孿生沙盒中計算 6-DoF 軌跡並執行 25Hz 碰撞與穩定度預演…'
                : reconstructingId === 'world'
                  ? 'World Labs Marble 正在啟動雲端 3D 環境生成…'
                  : reconstructingId
                    ? 'FAL.ai 正在雲端重建 1024×1024 材質 3D 網格 (.glb，約需 15–35 秒)，完成後將自動載入數位孿生…'
                    : 'Argon ER 2 正在根據即時畫面與 3D 場景圖回答您的問題…'}
          </span>
        </div>
      )}

      {status?.worldlabs_last && (
        <div className="mt-2 flex items-center justify-between rounded-lg border border-purple-500/35 bg-purple-950/40 px-3 py-1.5 text-[11px] text-purple-200">
          <span>
            ✨ World Labs Marble 3D 環境生成已受理（模型：{status.worldlabs_last.model || 'marble-1.0-draft'} · ID:{' '}
            {status.worldlabs_last.operation_id}）
          </span>
          <span className="rounded bg-purple-500/20 px-1.5 py-0.5 text-[10px] font-medium text-purple-300">
            {status.worldlabs_last.status}
          </span>
        </div>
      )}

      {errorMsg && (
        <div className="mt-2.5 rounded-lg border border-rose-500/40 bg-rose-950/60 px-3 py-2 text-xs text-rose-200">
          {errorMsg}
        </div>
      )}

      {/* Main Split Grid: Viewport on Left, Scene Graph & Steering on Right */}
      <div className="mt-3 grid grid-cols-1 gap-4 lg:grid-cols-12">
        {/* Left 7 cols: 3D Twin Viewport or MIRA Orthographic BEV */}
        <div className="lg:col-span-7">
          {viewTab === 'twin3d' ? (
            <TwinViewport3D revision={revision} activeFrame={currentPoseFrame} />
          ) : (
            <div className="relative flex h-[320px] w-full items-center justify-center overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
              <img
                src={`/api/real-to-sim/artifact/bev_grid.jpg?v=${revision}`}
                alt="MIRA 正交俯視圖 (1px = 0.5mm)"
                className="h-full w-auto object-contain"
              />
              <div className="pointer-events-none absolute top-2.5 left-3 rounded-md bg-slate-900/85 px-2 py-0.5 text-[11px] font-medium text-emerald-300 backdrop-blur">
                MIRA 正交俯視圖 (1 px = 0.5 mm · 已剔除夾爪遮擋)
              </div>
            </div>
          )}

          {/* `steer-a-robot` style 25 Hz Mental Simulation Playback Scrubber */}
          {activePlan && poseStream.length > 0 && (
            <div className="mt-2.5 rounded-xl border border-slate-800 bg-slate-900/90 p-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setPlayingSim((p) => !p)}
                    className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 text-white hover:bg-indigo-500"
                    title={playingSim ? '暫停沙盒預演' : '播放沙盒預演'}
                  >
                    {playingSim ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setPlayingSim(false);
                      setSimStepIdx(0);
                    }}
                    className="inline-flex h-7 w-7 items-center justify-center rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800"
                    title="重頭預演"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                  </button>
                  <span className="font-medium text-slate-200">
                    腦中物理沙盒預演 ({currentPoseFrame?.time_s.toFixed(2)}s / {activePlan.simulation.duration_s}s)
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[11px]">
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-sky-300">
                    最低桌面淨空: {activePlan.simulation.min_table_clearance_mm} mm
                  </span>
                  <span className="rounded bg-slate-800 px-2 py-0.5 text-emerald-300">
                    物理穩定度: {Math.round(activePlan.simulation.physics_verification.stability_score * 100)}%
                  </span>
                </div>
              </div>
              <input
                type="range"
                min={0}
                max={Math.max(0, poseStream.length - 1)}
                value={simStepIdx}
                onChange={(e) => {
                  setPlayingSim(false);
                  setSimStepIdx(Number(e.target.value));
                }}
                className="mt-2 h-1.5 w-full cursor-pointer accent-indigo-500"
              />
            </div>
          )}
        </div>

        {/* Right 5 cols: Persistent 3D Scene Graph + Zero-Shot Pick & Place Steering */}
        <div className="flex flex-col justify-between gap-3 lg:col-span-5">
          {/* Scene Summary & Objects List */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/75 p-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-200">
                持久 3D 場景圖 ({objects.length} 個物件)
              </span>
              {status?.scene_map.last_bev && (
                <span className="text-[10px] text-slate-400">
                  融合 {status.scene_map.last_bev.views_fused} 視角 · {status.scene_map.last_bev.synthesis_ms} ms
                </span>
              )}
            </div>
            {status?.scene_map.last_summary_zh && (
              <p className="mt-1 text-[11px] leading-relaxed text-slate-300">
                {status.scene_map.last_summary_zh}
              </p>
            )}

            {objects.length === 0 ? (
              <p className="mt-2 text-xs text-slate-400">
                尚未建立桌面 3D 物件記憶。請點擊上方「MIRA 單視角即時建模」或「5 視角主動環視建模」。
              </p>
            ) : (
              <div className="mt-2 max-h-[145px] space-y-1.5 overflow-y-auto pr-1">
                {objects.map((obj) => {
                  const selected = obj.id === selectedObjectId;
                  return (
                    <div
                      key={obj.id}
                      className={`flex items-center justify-between gap-2 rounded-lg border px-2.5 py-1.5 text-xs transition ${
                        selected
                          ? 'border-indigo-500 bg-indigo-950/60 text-white'
                          : 'border-slate-800 bg-slate-950/60 text-slate-300 hover:border-slate-700'
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => setSelectedObjectId(obj.id)}
                        className="min-w-0 flex-1 text-left"
                      >
                        <div className="flex items-center gap-1.5">
                          <span
                            className="inline-block h-2.5 w-2.5 rounded-full"
                            style={{
                              backgroundColor: `rgb(${Math.round(obj.color_rgba[0] * 255)}, ${Math.round(
                                obj.color_rgba[1] * 255,
                              )}, ${Math.round(obj.color_rgba[2] * 255)})`,
                            }}
                          />
                          <span className="truncate font-medium">{obj.name}</span>
                          <span className="text-[10px] text-slate-400">
                            ({Math.round(obj.centre[0] * 1000)}, {Math.round(obj.centre[1] * 1000)} mm)
                          </span>
                        </div>
                        <div className="mt-0.5 text-[10px] text-slate-400">
                          尺寸 {Math.round(obj.semi_major_m * 2000)}×{Math.round(obj.semi_minor_m * 2000)}×
                          {Math.round(obj.height * 1000)}mm · 偏航 {obj.yaw_deg}° · 預估{' '}
                          {obj.physics?.estimated_mass_g ?? 120}g
                        </div>
                      </button>

                      {status?.cloud_providers.fal_ai_configured && (
                        <button
                          type="button"
                          disabled={Boolean(reconstructingId)}
                          onClick={(e) => {
                            e.stopPropagation();
                            void handleReconstruct3D('object', obj.id);
                          }}
                          className={`shrink-0 rounded-md border px-2 py-1 text-[10px] font-medium transition disabled:opacity-50 ${
                            obj.mesh_glb_path
                              ? 'border-emerald-500/40 bg-emerald-950/60 text-emerald-200 hover:bg-emerald-900/70'
                              : 'border-sky-500/40 bg-sky-950/60 text-sky-200 hover:bg-sky-900/70'
                          }`}
                          title="使用 FAL.ai 生成公制對齊的 3D .glb 網格"
                        >
                          {reconstructingId === obj.id ? (
                            <span className="inline-flex items-center gap-1">
                              <LoaderCircle className="h-3 w-3 animate-spin" />
                              生成中…
                            </span>
                          ) : obj.mesh_glb_path ? (
                            '✓ 已套用 3D 網格'
                          ) : (
                            'FAL 3D 網格'
                          )}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}

            {/* Spatial Relations Pills */}
            {relations.length > 0 && (
              <div className="mt-2 border-t border-slate-800/80 pt-1.5 text-[10px] text-slate-400">
                空間關係：{relations.slice(0, 2).map((r) => r.direction_zh).join('；')}
              </div>
            )}
          </div>

          {/* Zero-Shot Pick & Place / Autosteer Controls */}
          <div className="rounded-xl border border-slate-800 bg-slate-900/75 p-3">
            <label htmlFor="r2s-instruction-input" className="block text-xs font-semibold text-slate-200">
              零樣本夾取 / 放置引導 (Steer-a-Robot 閉環操控)
            </label>

            {/* Free-Space Placement Slot Chips */}
            {freeSlots.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                <button
                  type="button"
                  onClick={() => setSelectedPlaceSlot('')}
                  className={`rounded-md px-2 py-0.5 text-[10px] font-medium transition ${
                    selectedPlaceSlot === ''
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  僅原位夾取驗證
                </button>
                {freeSlots.slice(0, 6).map((slot) => (
                  <button
                    key={slot.slot_label}
                    type="button"
                    onClick={() => setSelectedPlaceSlot(slot.slot_label)}
                    className={`rounded-md px-2 py-0.5 text-[10px] font-medium transition ${
                      selectedPlaceSlot === slot.slot_label
                        ? 'bg-amber-500 text-slate-950'
                        : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                    }`}
                  >
                    放置：{slot.slot_label}
                  </button>
                ))}
              </div>
            )}

            <input
              id="r2s-instruction-input"
              type="text"
              value={instruction}
              onChange={(e) => setInstruction(e.target.value)}
              placeholder="輸入自然語言指令，例如：把綠色品客放到藍色方塊右邊..."
              className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-2.5 py-1.5 text-xs text-white placeholder-slate-500 focus:border-indigo-500 focus:outline-none"
            />

            <div className="mt-2 flex flex-wrap items-center gap-2">
              <button
                type="button"
                disabled={planning || busy}
                onClick={() => void handlePlanTask()}
                className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-sky-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-sky-500 disabled:opacity-50"
              >
                {planning ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Box className="h-3.5 w-3.5" />}
                1. 腦中沙盒預演規劃
              </button>

              {activePlan && activePlan.executable && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => void handleExecutePlan()}
                  className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:bg-emerald-500 disabled:opacity-50"
                >
                  <ShieldCheck className="h-3.5 w-3.5" />
                  2. 預演通過・實體執行
                </button>
              )}
            </div>

            {activePlan && !activePlan.executable && activePlan.problems.length > 0 && (
              <div className="mt-2 rounded-lg border border-amber-500/40 bg-amber-950/50 px-2.5 py-1.5 text-[11px] text-amber-200">
                沙盒攔截風險：{activePlan.problems.join('；')}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom Bar: `steer-a-robot` style Live Scene Q&A */}
      <div className="mt-3 rounded-xl border border-slate-800 bg-slate-900/60 p-2.5">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-4 w-4 shrink-0 text-indigo-400" />
          <input
            type="text"
            value={qaInput}
            onChange={(e) => setQaInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                void handleAskScene();
              }
            }}
            placeholder="即時場景問答 (Live Scene Q&A)：問問 Argon 桌上有什麼、距離幾毫米、哪個物體最重或適合放哪裡..."
            className="flex-1 bg-transparent text-xs text-white placeholder-slate-400 focus:outline-none"
          />
          <button
            type="button"
            disabled={asking || !qaInput.trim()}
            onClick={() => void handleAskScene()}
            className="inline-flex items-center gap-1 rounded-lg bg-slate-800 px-2.5 py-1 text-xs font-medium text-indigo-300 hover:bg-slate-700 disabled:opacity-50"
          >
            {asking ? <LoaderCircle className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
            提問
          </button>
        </div>
        {status?.qa_history && status.qa_history.length > 0 && (
          <div className="mt-2 border-t border-slate-800/80 pt-2 text-xs text-slate-200">
            <span className="font-medium text-indigo-300">
              Q: {status.qa_history[status.qa_history.length - 1].question}
            </span>
            <p className="mt-0.5 text-[11px] leading-relaxed text-slate-300">
              {status.qa_history[status.qa_history.length - 1].answer_zh}
            </p>
          </div>
        )}
      </div>

      {/* VLA Engine Hub (Dream Labs Dream-1 & SmolVLA SO-101 Multi-Task) + Dashcam + Agentic Evaluator */}
      <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-3">
        {/* 1. Dream Labs & SmolVLA Zero-Shot / Multi-Task Hub */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-indigo-300">
              🧠 VLA 策略模型 (Dream Labs / SmolVLA)
            </span>
            <div className="inline-flex rounded-md border border-slate-700 bg-slate-950 p-0.5 text-[10px]">
              <button
                type="button"
                onClick={() => {
                  setVlaEngine('smolvla');
                  setVlaCheckpoint('lerobot/smolvla_base');
                }}
                className={`rounded px-2 py-0.5 font-medium ${
                  vlaEngine === 'smolvla' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                }`}
              >
                SmolVLA (9 模型)
              </button>
              <button
                type="button"
                onClick={() => {
                  setVlaEngine('dreamlabs');
                  setVlaCheckpoint('dreamlabs/dream-1-beta');
                }}
                className={`rounded px-2 py-0.5 font-medium ${
                  vlaEngine === 'dreamlabs' ? 'bg-indigo-600 text-white' : 'text-slate-400'
                }`}
              >
                Dream Labs (Dream-1)
              </button>
            </div>
          </div>

          <select
            aria-label="選擇 VLA 模型權重"
            value={vlaCheckpoint}
            onChange={(e) => setVlaCheckpoint(e.target.value)}
            className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-slate-200"
          >
            {vlaEngine === 'smolvla'
              ? (status?.vla?.smolvla.available_models ?? [
                  { id: 'lerobot/smolvla_base', name: 'SmolVLA Base (官方 450M 基礎模型)', tasks_zh: 'Zero-shot 基礎模型' },
                  { id: 'hjkso1406/smolvla-so101-4tasks-aug-v2', name: 'SmolVLA SO-101 四合一日常任務', tasks_zh: '抓取/放置/整理' },
                  { id: 'BC2605/so101-smolvla-rack-to-cup-box-anticoll', name: 'SmolVLA SO-101 置架取物入杯/盒 (防碰撞)', tasks_zh: '防碰撞收納' },
                  { id: 'khanhnd61/smolvla-so101-multi-task-gguf', name: 'SmolVLA SO-101 Multi-Task', tasks_zh: '多任務搬移' },
                  { id: 'Sa74ll/smolvla_so101_pickandplace', name: 'SmolVLA SO-101 Pick & Place', tasks_zh: '通用抓放' },
                ]).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name} — {m.tasks_zh}
                  </option>
                ))
              : (status?.vla?.dreamlabs.available_models ?? [
                  { id: 'dreamlabs/dream-1-beta', name: 'Dream Labs Dream-1 Beta (48步 EEF 策略)' },
                  { id: 'dreamlabs/dream-1-beta-flash', name: 'Dream Labs Dream-1 Beta Flash (高速版)' },
                ]).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
          </select>

          <input
            type="text"
            value={vlaPrompt}
            onChange={(e) => setVlaPrompt(e.target.value)}
            placeholder="輸入英文或中文任務指令 (例如: pick up the white tube)..."
            className="mt-1.5 w-full rounded-lg border border-slate-700 bg-slate-950 px-2 py-1 text-[11px] text-white"
          />

          <div className="mt-2 flex gap-1.5">
            <button
              type="button"
              disabled={vlaRunning || busy}
              onClick={() => void handleRunVlaZeroShot(false)}
              className="flex-1 rounded-lg bg-indigo-600/90 px-2.5 py-1 text-[11px] font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
            >
              {vlaRunning ? '推論中…' : 'Zero-Shot 預測軌跡'}
            </button>
            <button
              type="button"
              disabled={vlaRunning || busy}
              onClick={() => void handleRunVlaZeroShot(true)}
              className="flex-1 rounded-lg border border-emerald-500/40 bg-emerald-950/60 px-2.5 py-1 text-[11px] font-medium text-emerald-200 hover:bg-emerald-900/70 disabled:opacity-50"
            >
              護欄保護實體執行
            </button>
          </div>

          {vlaResult && (
            <div className="mt-2 rounded-lg border border-slate-800 bg-slate-950/90 p-2 text-[10px] text-slate-300">
              <div className="font-semibold text-indigo-300">
                {String(vlaResult.engine).toUpperCase()} · {String(vlaResult.checkpoint)} ({String(vlaResult.elapsed_s)}s)
              </div>
              {vlaResult.zero_shot_analysis_zh ? (
                <p className="mt-1 leading-relaxed text-slate-400">{String(vlaResult.zero_shot_analysis_zh)}</p>
              ) : (
                <p className="mt-1 text-slate-400">
                  已產生 {String(vlaResult.chunk_horizon ?? 48)} 步 3D EEF 動作序列，首步 EEF XYZ:{' '}
                  {JSON.stringify(vlaResult.first_eef_xyz_m)}
                </p>
              )}
            </div>
          )}
        </div>

        {/* 2. Continuous Dashcam Flight Recorder ("行車記錄器" + Rerun .rrd + Sim Re-Training) */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-sky-300">
              📹 行車記錄器 (ROSbag / Rerun .rrd / Sim 訓練)
            </span>
            <div className="inline-flex rounded-md border border-slate-700 bg-slate-950 p-0.5 text-[10px]">
              <button
                type="button"
                onClick={() => void handleSetDashcamWindow(30)}
                className={`rounded px-2 py-0.5 font-medium ${
                  (status?.dashcam?.window_minutes ?? 30) <= 30 ? 'bg-sky-600 text-white' : 'text-slate-400'
                }`}
              >
                30 分鐘視窗
              </button>
              <button
                type="button"
                onClick={() => void handleSetDashcamWindow(60)}
                className={`rounded px-2 py-0.5 font-medium ${
                  (status?.dashcam?.window_minutes ?? 30) > 30 ? 'bg-sky-600 text-white' : 'text-slate-400'
                }`}
              >
                60 分鐘視窗
              </button>
            </div>
          </div>

          <p className="mt-1 text-[10px] text-slate-400">
            滑動緩衝：{status?.dashcam?.buffered_frames ?? 0} 幀 ({status?.dashcam?.buffered_span_minutes ?? 0} 分鐘) · 已歸檔：
            <span className="text-emerald-300"> 成功 {status?.dashcam?.success_count ?? 0}</span> /
            <span className="text-rose-300"> 失敗 {status?.dashcam?.failure_count ?? 0}</span>
          </p>

          <div className="mt-2 flex gap-1.5">
            <button
              type="button"
              disabled={Boolean(archivingOutcome)}
              onClick={() => void handleArchiveEpisode('success')}
              className="flex-1 rounded-lg bg-emerald-600/85 px-2 py-1 text-[11px] font-medium text-white hover:bg-emerald-500 disabled:opacity-50"
            >
              ✓ 標記成功並歸檔
            </button>
            <button
              type="button"
              disabled={Boolean(archivingOutcome)}
              onClick={() => void handleArchiveEpisode('failure')}
              className="flex-1 rounded-lg bg-rose-600/85 px-2 py-1 text-[11px] font-medium text-white hover:bg-rose-500 disabled:opacity-50"
            >
              ✕ 標記失敗 (Sim 再訓練)
            </button>
          </div>

          {status?.dashcam?.episodes && status.dashcam.episodes.length > 0 && (
            <div className="mt-2 max-h-28 space-y-1 overflow-y-auto pr-1 text-[10px]">
              {status.dashcam.episodes.slice(0, 4).map((ep) => (
                <div
                  key={ep.episode_id}
                  className="flex items-center justify-between rounded border border-slate-800 bg-slate-950/80 px-2 py-1"
                >
                  <span className={ep.outcome === 'success' ? 'text-emerald-300' : 'text-rose-300'}>
                    [{ep.outcome === 'success' ? '成功' : '失敗'}] {ep.query || ep.task_kind} ({ep.num_frames}幀)
                  </span>
                  {ep.files && (
                    <span className="flex gap-1.5">
                      <a
                        href={ep.files.episode_rrd}
                        download
                        className="text-sky-400 underline hover:text-sky-300"
                        title="下載 Rerun .rrd 錄影檔"
                      >
                        .rrd
                      </a>
                      <a
                        href={ep.files.sim_counterfactual_json}
                        download
                        className="text-indigo-400 underline hover:text-indigo-300"
                        title="下載 Sim 反事實與合成訓練包"
                      >
                        Sim包
                      </a>
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 3. Intel RealSense D435 Post-Hoc Referee & Agentic Engineering Metrics */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
          <div className="flex items-center justify-between gap-2">
            <span className="text-xs font-semibold text-emerald-300">
              📐 Agentic Engineering 事後評估 (D435 裁判)
            </span>
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[9px] text-emerald-300">
              不介入控制迴路
            </span>
          </div>

          <div className="mt-2 grid grid-cols-3 gap-1.5 text-center">
            <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-1.5">
              <div className="text-[10px] text-slate-400">任務成功率</div>
              <div className="text-sm font-bold text-emerald-300">
                {status?.agentic_eval?.task_success_rate_pct ?? 0}%
              </div>
            </div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-1.5">
              <div className="text-[10px] text-slate-400">多樣性指標</div>
              <div className="text-sm font-bold text-sky-300">
                {status?.agentic_eval?.diversity_index_0_100 ?? 0}
              </div>
            </div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-1.5">
              <div className="text-[10px] text-slate-400">左固定牙貼邊</div>
              <div className="text-sm font-bold text-amber-300">
                {status?.agentic_eval?.mean_fixed_left_jaw_gap_mm ?? 3.5}mm
              </div>
            </div>
          </div>

          <p className="mt-1.5 text-[10px] text-slate-400">
            已涵蓋 {status?.agentic_eval?.unique_objects_count ?? 0} 種物件 ·{' '}
            {status?.agentic_eval?.unique_task_modes_count ?? 0} 種任務模式 · D435 狀態：
            {status?.dashcam?.realsense_d435?.connected ? '已連線 RGB-D' : '待命 (自動以輔助快照比對)'}
          </p>

          {status?.agentic_eval?.recent_evaluations &&
            status.agentic_eval.recent_evaluations.length > 0 && (
              <div className="mt-1.5 rounded border border-slate-800 bg-slate-950/90 px-2 py-1 text-[10px] text-slate-300">
                <span className="font-medium text-emerald-300">最新自我改進建議：</span>
                {status.agentic_eval.recent_evaluations[0].agentic_recommendations?.[0] ?? ''}
              </div>
            )}
        </div>
      </div>
    </section>
  );
}
