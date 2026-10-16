'use client';
/* oxlint-disable next/no-img-element -- Local evidence images must bypass image transformation and caching. */

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { showJawEnvelope, snapshotEvidencePresentation, visibleExtentPresentation, type VisibleExtent } from '../../lib/engineering-display';
import { ControllerRequestError, controllerRequest } from '../../lib/client-status';
import './scene.css';

type Pose = { position_m: [number, number, number]; quaternion_xyzw: [number, number, number, number] };
type Snapshot = {
  exported_at: number;
  frame: { id: string; camera_generation: string; captured_at?: number; joints: Record<string, number> };
  camera: { optical_pose: Pose };
  candidates: { id: string; label: string; camera_pose: Pose }[];
  targets: { position_m: number[] | null }[];
  robot?: { jaw?: { angle_calibrated: boolean; is_actual_jaw_pose: boolean; rendering: string } };
  verification: { passed: boolean; mujoco_version?: string };
};
type Depth = { status: string; reason: string; metric_calibrated: boolean;
  current_frame_id?: string;
  roi_frame_id?: string;
  frame_ids?: string[];
  provenance?: { camera_generation?: string; camera_model?: string };
  camera_transform_base_m?: number[][];
  maximum_camera_translation_mm?: number;
  distance?: { camera_optical_axis_m: number; gripper_forward_axis_m: number } | null;
  uncertainty?: { camera_depth_interval_m: number[]; base_position_error_bound_m?: number; gripper_axis_error_bound_m?: number };
  visible_extent?: VisibleExtent | null;
  points?: { position_base_m: [number, number, number] }[];
};
const RELOAD_SCENE = '場景快照已更新，請按「重新載入快照」。';
const artifactUrl = (name: string, frameId: string, revision: number, snapshotId = '') =>
  `/api/engineering/scene/${name}?frame_id=${encodeURIComponent(frameId)}&v=${revision}${snapshotId ? `&snapshot_id=${encodeURIComponent(snapshotId)}` : ''}`;
type SavedSnapshot = { snapshot_id: string; frame_id: string; captured_at?: number; has_depth: boolean };

function View({ snapshot, revision, depth, envelopeVisible, snapshotId }: {
  snapshot: Snapshot; revision: number; depth: Depth | null; envelopeVisible: boolean; snapshotId: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const model = useRef<THREE.Object3D | null>(null);
  const showEnvelope = useRef(envelopeVisible);
  const invalidateView = useRef<(() => void) | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    showEnvelope.current = envelopeVisible;
    if (model.current) showJawEnvelope(model.current, envelopeVisible);
    invalidateView.current?.();
  }, [envelopeVisible]);
  useEffect(() => {
    const element = host.current;
    if (!element) return;
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
    catch { queueMicrotask(() => setError('這個瀏覽器未提供 WebGL，仍可查看右側資料與下載場景。')); return; }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0xf4f7fb);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1;
    element.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.add(new THREE.HemisphereLight(0xffffff, 0xa8adb5, 2));
    const light = new THREE.DirectionalLight(0xffffff, 2.5); light.position.set(1, -.6, 1.5); scene.add(light);
    const camera = new THREE.PerspectiveCamera(42, 1, .002, 10);
    camera.up.set(0, 0, 1); camera.position.set(.44, -.4, .36);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(.08, 0, .14); controls.enableDamping = true; controls.minDistance = .12; controls.maxDistance = 2;
    const grid = new THREE.GridHelper(.8, 16, 0xb8c4d2, 0xd8dfe8); grid.rotation.x = Math.PI / 2; grid.position.z = .001; scene.add(grid);
    const axes = new THREE.AxesHelper(.1); axes.position.z = .004; scene.add(axes);
    const origin = new THREE.Vector3(...snapshot.camera.optical_pose.position_m);
    function cameraMark(pose: Pose, color: number, length: number) {
      const p = new THREE.Vector3(...pose.position_m);
      const q = new THREE.Quaternion(...pose.quaternion_xyzw);
      const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(q);
      scene.add(new THREE.ArrowHelper(forward, p, length, color, .012, .006));
      return p;
    }
    cameraMark(snapshot.camera.optical_pose, 0x12a46b, .13);
    if (depth?.provenance?.camera_model === 'same-attempt-refined-prior' && depth.camera_transform_base_m) {
      const rows = depth.camera_transform_base_m;
      const matrix = new THREE.Matrix4().fromArray(rows.flat()).transpose();
      const q = new THREE.Quaternion().setFromRotationMatrix(matrix);
      const p = new THREE.Vector3().setFromMatrixPosition(matrix);
      cameraMark({ position_m: p.toArray(), quaternion_xyzw: q.toArray() }, 0xf29900, .13);
    }
    // FOV is deliberately a display prior; it is not a measured camera model.
    const frustum = new THREE.PerspectiveCamera(45, 16 / 9, .008, .16);
    frustum.position.copy(origin);
    frustum.quaternion.copy(new THREE.Quaternion(...snapshot.camera.optical_pose.quaternion_xyzw)
      .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI)));
    frustum.updateMatrixWorld(); scene.add(new THREE.CameraHelper(frustum));
    snapshot.candidates.forEach(candidate => {
      const p = cameraMark(candidate.camera_pose, 0x4285f4, .05);
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([origin, p]), new THREE.LineBasicMaterial({ color: 0x4285f4 })));
    });
    if (depth?.status === 'estimated' && depth.points?.length) {
      const vertices = depth.points.flatMap(p => p.position_base_m);
      const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0xf29900, size: .004 })));
    }
    const disposeObject = (object: THREE.Object3D) => object.traverse(node => {
      const mesh = node as THREE.Mesh;
      mesh.geometry?.dispose();
      const materials = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
      materials.forEach(material => material.dispose());
    });
    let active = true;
    let animation: number | null = null;
    function invalidate() {
      if (active && animation === null) animation = requestAnimationFrame(render);
    }
    function render() {
      animation = null;
      if (!active) return;
      const changed = controls.update();
      renderer.render(scene, camera);
      // OrbitControls emits changes while damping settles. Stop scheduling
      // once the view is still; static CAD does not need a permanent 60fps loop.
      if (changed) invalidate();
    }
    invalidateView.current = invalidate;
    controls.addEventListener('change', invalidate);
    const controller = new AbortController();
    fetch(artifactUrl('scene.glb', snapshot.frame.id, revision, snapshotId), { signal: controller.signal, cache: 'no-store' })
      .then(async response => {
        if (response.status === 409) throw new Error(RELOAD_SCENE);
        if (!response.ok) throw new Error('三維模型未載入，請重新載入場景。');
        return new GLTFLoader().parseAsync(await response.arrayBuffer(), '');
      }).then(gltf => {
        if (active) {
          showJawEnvelope(gltf.scene, showEnvelope.current);
          model.current = gltf.scene;
          scene.add(gltf.scene);
          invalidate();
        } else disposeObject(gltf.scene);
      })
      .catch(error => { if (active) setError(error instanceof Error ? error.message : '三維模型未載入，請重新載入場景。'); });
    const resize = new ResizeObserver(() => {
      const w = element.clientWidth, h = element.clientHeight;
      renderer.setSize(w, h); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
      invalidate();
    }); resize.observe(element);
    invalidate();
    return () => { active = false; model.current = null; controller.abort();
      if (animation !== null) cancelAnimationFrame(animation);
      if (invalidateView.current === invalidate) invalidateView.current = null;
      controls.removeEventListener('change', invalidate); resize.disconnect(); controls.dispose();
      disposeObject(scene); renderer.dispose(); renderer.domElement.remove(); };
  }, [snapshot, revision, depth, snapshotId]);
  return <div className="sim-canvas" ref={host} aria-label="可拖曳旋轉的 LeRobot 三維場景">{error && <p className="sim-render-error">{error}</p>}</div>;
}

export default function EngineeringScene() {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [depth, setDepth] = useState<Depth | null>(null);
  const [wristImage, setWristImage] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [revision, setRevision] = useState(0);
  const [envelopeVisible, setEnvelopeVisible] = useState(false);
  const [cloudVisible, setCloudVisible] = useState(false);
  const [savedSnapshots, setSavedSnapshots] = useState<SavedSnapshot[]>([]);
  const [snapshotId, setSnapshotId] = useState('');
  const [capturing, setCapturing] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const token = useRef('');
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 30000); return () => clearInterval(timer); }, []);
  async function captureCurrent() {
    setCapturing(true); setError('');
    try {
      const result = await controllerRequest<{ snapshot_id: string }>('/api/engineering/scene-capture', {}, token, fetch, 75000);
      setSnapshot(null); setDepth(null); setCloudVisible(false); setWristImage(null);
      setSnapshotId(result.snapshot_id); setRevision(v => v + 1); setNow(Date.now());
    } catch (error) {
      setError(error instanceof ControllerRequestError && error.status === 0
        ? '暫時無法確認快照是否已匯出。這項擷取不會移動手臂；請稍後按「重新載入快照」查看結果，先不要重複擷取。'
        : error instanceof Error ? error.message : '快照擷取失敗，原快照仍保留。');
    }
    finally { setCapturing(false); }
  }
  useEffect(() => {
    const controller = new AbortController();
    let imageUrl: string | null = null;
    const request = (url: string) => fetch(url, { signal: controller.signal, cache: 'no-store' });
    async function load() {
      const manifest = await request(`/api/engineering/scene/scene.json${snapshotId ? `?snapshot_id=${encodeURIComponent(snapshotId)}` : ''}`);
      if (!manifest.ok) throw new Error('目前還沒有匯出的場景。請先完成一次工程觀察。');
      const state = await manifest.json() as Snapshot;
      const [evidence, image] = await Promise.all([
        request(artifactUrl('depth.json', state.frame.id, revision, snapshotId)).then(async response => {
          if (response.status === 409) throw new Error(RELOAD_SCENE);
          return response.ok ? response.json() as Promise<Depth> : null;
        }),
        request(artifactUrl('wrist.jpg', state.frame.id, revision, snapshotId)).then(async response => {
          if (response.status === 409) throw new Error(RELOAD_SCENE);
          return response.ok ? response.blob() : null;
        }),
      ]);
      if (controller.signal.aborted) return;
      imageUrl = image ? URL.createObjectURL(image) : null;
      setSnapshot(state);
      setWristImage(imageUrl);
      const matches = evidence?.current_frame_id === state.frame.id && evidence?.provenance?.camera_generation === state.frame.camera_generation;
      setDepth(!evidence || matches ? evidence : { status: 'insufficient', metric_calibrated: false,
        reason: '這份距離資料不屬於目前快照，已隱藏；請匯出同一次觀察的資料。' });
      setCloudVisible(false);
    }
    request('/api/engineering/scenes').then(response => response.json() as Promise<{ snapshots: SavedSnapshot[] }>).then(data => {
      if (!controller.signal.aborted && Array.isArray(data.snapshots)) setSavedSnapshots(data.snapshots);
    }).catch(() => {});
    load()
      .catch(e => { if (!controller.signal.aborted) setError(String(e.message || e)); });
    return () => { controller.abort(); if (imageUrl) URL.revokeObjectURL(imageUrl); };
  }, [revision, snapshotId]);
  const mm = (m: number) => `${Math.round(m * 1000)} mm`;
  const visibleExtent = snapshot ? visibleExtentPresentation(depth, snapshot.frame.id) : null;
  const estimatedDepth = cloudVisible && depth?.status === 'estimated' ? depth : null;
  const evidenceState = snapshotEvidencePresentation(snapshot?.frame.captured_at, depth?.metric_calibrated, now);
  return <main className="sim-page">
    <header className="sim-header"><div><Link href="/" className="sim-brand">Le<span>Robot</span></Link><span className="sim-eyebrow">工程觀察室</span></div>
      <div className="sim-header-actions"><Link href="/engineering/calibration" className="sim-calibration-link">相機與手臂校準</Link>
      <button disabled={capturing} onClick={captureCurrent}>{capturing ? '正在擷取…' : '擷取目前姿勢'}</button>
      <button onClick={() => { setError(''); setSnapshot(null); setDepth(null); setWristImage(null); setSnapshotId(''); setRevision(v => v + 1); }}>重新載入快照</button></div></header>
    <div className="sim-heading"><div><p className="sim-eyebrow">已保存的工程觀察</p><h1>影像與姿勢，對照同一次拍攝。</h1>
      <p>擷取只讀取腕部影像與馬達位置；新快照不沿用先前的物件點雲。</p></div><span className="sim-badge">僅供工程驗證 · 不會移動手臂</span></div>
    {error && <p role="alert" className="sim-error">{error}</p>}
    <div className="sim-snapshot-tools"><label>查看保存快照 <select value={snapshotId} onChange={event => {
      setSnapshot(null); setDepth(null); setCloudVisible(false); setWristImage(null); setError(''); setSnapshotId(event.target.value);
    }}><option value="">最近匯出的快照</option>{savedSnapshots.map(saved => <option key={saved.snapshot_id} value={saved.snapshot_id}>
      {snapshotEvidencePresentation(saved.captured_at, false).capturedLabel}{saved.has_depth ? ' · 附歷史估計' : ' · 姿勢與影像'}
    </option>)}</select></label></div>
    {snapshot && <div className="sim-snapshot-notice" aria-live="polite"><strong>{evidenceState.historical ? '歷史快照' : '剛擷取的快照'} · {evidenceState.capturedLabel}</strong>
      <p>這是拍攝當時的姿勢，不是即時畫面。綠色相機使用 CAD 安裝位置，尚非實測校準位置。</p>
      {depth?.status === 'estimated' ? <p>{evidenceState.calibrated ? '這份深度有校準來源，仍只屬於本次拍攝。' : '這份歷史點雲未完成實測校準，預設隱藏；不能代表目前物件位置或夾爪間隙。'}</p>
        : <p>這份快照沒有同次物件深度；只顯示腕部影像與手臂模型。</p>}</div>}
    {snapshot ? <div className="sim-grid">
      <section className="sim-model"><View snapshot={snapshot} revision={revision} depth={cloudVisible ? depth : null} envelopeVisible={envelopeVisible} snapshotId={snapshotId} />
        <div className="sim-legend"><span>🟢 快照的 CAD 相機</span><span>🔵 候選視角</span>{cloudVisible && <span>🟠 本快照的表面估計</span>}<span>格線 5 cm</span></div>
        {depth?.status === 'estimated' && <label className="sim-overlay-toggle"><input type="checkbox" checked={cloudVisible} onChange={event => setCloudVisible(event.target.checked)} />顯示此快照的{evidenceState.calibrated ? '' : '未校準'}點雲<span>只與同次姿勢對照</span></label>}
        <label className="sim-overlay-toggle"><input type="checkbox" checked={envelopeVisible}
          onChange={event => setEnvelopeVisible(event.target.checked)} />顯示夾爪可能活動範圍<span>橘色線框 · 不是實際零件</span></label>
        <p className="sim-caption">拖曳旋轉 · 滾輪縮放。視野框使用示意角度。{snapshot.robot?.jaw?.rendering === 'full_sweep_envelope'
          ? '這份舊快照尚未包含活動夾爪模型；需重新匯出才能看到完整夾爪。'
          : '實心模型呈現兩側夾爪；活動側角度尚未校準，開口僅供示意。'}</p>
      </section>
      <aside className="sim-panels">
        <section className="sim-card"><p className="sim-eyebrow">腕部相機 · 同一次快照</p>
          {wristImage ? <img src={wristImage} alt="用來重建此場景的腕部相機畫面" /> : <p className="sim-caption">這份快照沒有附上腕部照片。</p>}
          <p className="sim-caption">{snapshot.frame.captured_at ? new Date(snapshot.frame.captured_at * 1000).toLocaleString('zh-TW') : '已儲存的觀察'} · 非即時影像</p>
        </section>
        <section className="sim-card"><p className="sim-eyebrow">距離線索</p>
          <h2>{estimatedDepth?.distance ? `拍攝時的估計深度 ${mm(estimatedDepth.distance.camera_optical_axis_m)}` : '沒有顯示物件距離'}</h2>
          {estimatedDepth?.uncertainty && <p>估計範圍 {estimatedDepth.uncertainty.camera_depth_interval_m.map(mm).join(' — ')}</p>}
          <p>{estimatedDepth?.reason || '此處只顯示同一次拍攝的資料。查看歷史估計時，請先勾選點雲。'}</p>
          {estimatedDepth?.uncertainty?.base_position_error_bound_m !== undefined && <p>機器人座標的位置誤差估計 ±{mm(estimatedDepth.uncertainty.base_position_error_bound_m)}；這不是已驗證的安全間隙。</p>}
          {estimatedDepth?.distance && <p>固定夾墊前方的估計值 {mm(estimatedDepth.distance.gripper_forward_axis_m)}，不代表可前進距離。</p>}
          {estimatedDepth && visibleExtent && <div className="sim-extent"><p>{visibleExtent.earlierReference ? '參考影像中的可見投影範圍約 ' : '可見投影範圍約 '}<strong>{mm(visibleExtent.widthM)} × {mm(visibleExtent.heightM)}</strong>（寬 × 高）</p>
            {visibleExtent.cropped && <p className="sim-caption">物件部分在畫面外，尚非完整尺寸。</p>}
            <p className="sim-caption">這是畫面中可見部分的投影範圍，不代表直徑或完整高度。</p></div>}
          {estimatedDepth?.maximum_camera_translation_mm !== undefined && <p className="sim-stat">歷史觀察基線 <strong>{estimatedDepth.maximum_camera_translation_mm.toFixed(1)} mm</strong></p>}
          <p className="sim-caption">此 CAD 場景的相機位置不是實測物件深度。顯示的距離指拍攝時的可見表面，不是夾爪還要前進多少。</p>
          {estimatedDepth?.provenance?.camera_model === 'same-attempt-refined-prior' && <p className="sim-caption">橘色箭頭使用這次影像擬合的未校準相機先驗；綠色箭頭保留 CAD 位置。兩者不一致時，點雲相對夾爪的位置仍可能有偏差。</p>}
        </section>
        <section className="sim-card"><p className="sim-eyebrow">模型核對</p><h2>{snapshot.verification.passed ? 'MuJoCo 姿勢一致' : '姿勢核對未通過'}</h2>
          <p>MuJoCo {snapshot.verification.mujoco_version || ''} 與原有運動學計算交叉核對。這驗證模型換算，尚不代表實機精度或抓取成功。</p>
          <div className="sim-downloads"><a href={artifactUrl('scene.glb', snapshot.frame.id, revision, snapshotId)} download>下載 3D 模型</a><a href={artifactUrl('scene.json', snapshot.frame.id, revision, snapshotId)} download>場景資料</a><a href={artifactUrl('verification.json', snapshot.frame.id, revision, snapshotId)} download>核對結果</a></div>
        </section>
        <details className="sim-card"><summary>實測關節位置</summary><dl>{Object.entries(snapshot.frame.joints).map(([joint, raw]) => <div key={joint}><dt>{joint}</dt><dd>{raw}</dd></div>)}</dl></details>
      </aside>
    </div> : !error && <p className="sim-loading">正在載入工程場景…</p>}
    <footer>D435 與手機照片只供旁觀驗證；正式操作的感知資料仍來自腕部相機。</footer>
  </main>;
}
