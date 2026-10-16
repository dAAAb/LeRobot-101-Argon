'use client';
/* oxlint-disable next/no-img-element -- The wrist stream is a live local camera endpoint. */

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { ControllerRequestError, controllerRequest } from '../../../lib/client-status';
import './calibration.css';

type Detection = { found?: boolean; reason?: string; coverage_fraction?: number };
type Solve = { solve_id: string; status: string; view_count?: number;
  target_view_count?: number; source_session_ids?: string[]; source_session_view_counts?: Record<string, number>;
  blockers: string[]; validation?: {
  accepted_views?: number; independent_pose_groups?: number;
  heldout_max_translation_m?: number; heldout_max_rotation_deg?: number;
  heldout_max_reprojection_p95_px?: number;
}; candidate_profile_path?: string | null };
type Session = {
  session_id: string; status: string; collected_view_count: number; distinct_pose_count: number;
  minimum_distinct_views_for_solve: number; board_static_declared: boolean;
  created_at: number; image_size: number[] | null; message: string;
  board_spec: { columns: number; rows: number; square_size_m: number };
  coverage: { minimum_board_fraction: number | null; maximum_board_fraction: number | null };
  last_detection: Detection | null; last_solve: Solve | null; accepted?: boolean;
};
type Mode = 'legacy' | 'shadow' | 'v2';
type SpatialState = { spatial_backend?: Mode; calibration_selected?: boolean; jaw_calibration_selected?: boolean };
type CalibrationState = { spatial: SpatialState | null; sessions: Session[]; board_url: string };
type Job = { id: string; status: string; message: string; calibration_result?: Session };
const BOARD_URL = '/calibration/lerobot-a4-calibration-board.pdf';
const percent = (value: number | null | undefined) => typeof value === 'number' ? `${Math.round(value * 100)}%` : '—';
const errorMessage = (cause: unknown) => cause instanceof ControllerRequestError && cause.status === 404
  ? '校準服務或工作階段尚未就緒，請稍後更新狀態。'
  : cause instanceof Error ? cause.message : '暫時無法更新校準狀態。';

function calibrationResultPresentation(session: Session | null) {
  const solve = session?.last_solve;
  if (!session || !solve) return null;
  const photoCount = Number.isSafeInteger(solve.view_count) && solve.view_count! > 0 ? solve.view_count! : null;
  const multiSession = 'target_view_count' in solve || 'source_session_ids' in solve || 'source_session_view_counts' in solve;
  const sourceIds = solve.source_session_ids;
  const sourceCounts = solve.source_session_view_counts;
  // A combined solve records both its total input and the selected batch's snapshot.
  // Missing or inconsistent provenance must not accidentally look like a single-batch result.
  const validSources = multiSession && photoCount !== null && Array.isArray(sourceIds) && sourceIds.length >= 2
    && sourceIds.every(id => typeof id === 'string' && id.length > 0)
    && new Set(sourceIds).size === sourceIds.length && sourceIds.includes(session.session_id)
    && sourceCounts !== null && typeof sourceCounts === 'object' && !Array.isArray(sourceCounts)
    && Object.keys(sourceCounts).length === sourceIds.length
    && sourceIds.every(id => Object.hasOwn(sourceCounts, id) && Number.isSafeInteger(sourceCounts[id]) && sourceCounts[id] > 0)
    && Number.isSafeInteger(solve.target_view_count) && solve.target_view_count! > 0
    && sourceCounts[session.session_id] === solve.target_view_count
    && sourceIds.reduce((sum, id) => sum + sourceCounts[id], 0) === photoCount;
  const targetPhotoCount = multiSession ? (validSources ? solve.target_view_count! : null) : photoCount;
  const matchesPhotos = targetPhotoCount !== null && targetPhotoCount === session.collected_view_count;
  const canApply = matchesPhotos && session.status === 'solved' && solve.status === 'validated' && !!solve.solve_id;
  const historical = !matchesPhotos || (solve.status === 'validated' && session.status !== 'solved');
  let message = validSources
    ? `這份結果使用 ${sourceIds!.length} 批共 ${photoCount} 張照片，其中這批 ${targetPhotoCount} 張。`
    : `這份結果使用 ${photoCount} 張照片。`;
  if (session.status === 'invalidated') message = '這批照片已失效，請開始新一批校準；舊結果不能套用。';
  else if (session.status === 'solving') message = '正在用目前的照片重新計算；下方先保留上次結果。';
  else if (multiSession && !validSources) message = '上次跨批次結果的照片紀錄不完整，請重新計算後再套用。';
  else if (photoCount === null) message = '上次結果沒有記錄照片數，請重新計算後再套用。';
  else if (!matchesPhotos) message = validSources
    ? `上次跨批次計算使用這批的 ${targetPhotoCount} 張；這批目前已收錄 ${session.collected_view_count} 張。請重新計算，才能套用最新結果。`
    : `上次計算用了 ${photoCount} 張；目前已收錄 ${session.collected_view_count} 張。請重新計算，才能套用最新結果。`;
  else if (historical) message = `上次計算用了 ${photoCount} 張，但這批照片尚未完成最新驗證；請重新計算後再套用。`;
  return { photoCount, canApply, historical, message };
}

function CalibrationSolveResult({ session, busy, onActivate }: {
  session: Session | null; busy: boolean; onActivate: () => void;
}) {
  const solve = session?.last_solve;
  const result = calibrationResultPresentation(session);
  if (!solve || !result) return null;
  const metricLabel = result.historical ? `上次${result.photoCount === null ? '' : ` ${result.photoCount} 張`}的結果` : '保留視角';
  return <div className={`cal-solve ${result.canApply ? 'is-validated' : ''}`}>
    <h3>{result.historical ? '上次的校準結果，需要更新。' : solve.status === 'validated' ? '驗證通過，校準候選已保存。' : solve.status === 'insufficient' ? '還需要補一些有效視角。' : '這批照片尚未通過驗證。'}</h3>
    <p role={result.historical ? 'status' : undefined}>{result.message}</p>
    {!!solve.blockers?.length && <ul>{solve.blockers.map((blocker, index) => <li key={`${index}-${blocker}`}>{blocker}</li>)}</ul>}
    {typeof solve.validation?.heldout_max_translation_m === 'number' && <p>{metricLabel}位置誤差：最高 {(solve.validation.heldout_max_translation_m * 1000).toFixed(1)} mm</p>}
    {typeof solve.validation?.heldout_max_rotation_deg === 'number' && <p>{metricLabel}角度誤差：最高 {solve.validation.heldout_max_rotation_deg.toFixed(1)}°</p>}
    <p className="cal-caption">這份結果只適用於已測過的工作範圍，不代表已完成夾爪接觸校準。</p>
    {solve.status === 'validated' && <button type="button" className="cal-button cal-button-secondary"
      disabled={busy || !result.canApply} onClick={onActivate}>套用已驗證的校準</button>}
  </div>;
}

export default function CalibrationPage() {
  const token = useRef('');
  const operation = useRef(false);
  const mounted = useRef(true);
  const [data, setData] = useState<CalibrationState | null>(null);
  const [selected, setSelected] = useState('');
  const [printChecked, setPrintChecked] = useState(false);
  const [boardFixed, setBoardFixed] = useState(false);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [preview, setPreview] = useState(false);
  const [modeChoice, setModeChoice] = useState<Mode>('legacy');

  const request = useCallback(<T,>(path: string, body?: unknown) => controllerRequest<T>(path, body, token), []);
  const refresh = useCallback(async () => {
    const next = await request<CalibrationState>('/api/engineering/calibration');
    if (mounted.current) {
      setData(next);
      if (next.spatial?.spatial_backend) setModeChoice(next.spatial.spatial_backend);
      setSelected(current => next.sessions.some(session => session.session_id === current)
        ? current : next.sessions[0]?.session_id || '');
    }
    return next;
  }, [request]);
  const incorporate = useCallback((session: Session) => {
    setData(current => ({ spatial: current?.spatial ?? null, board_url: current?.board_url || BOARD_URL,
      sessions: [session, ...(current?.sessions || []).filter(value => value.session_id !== session.session_id)] }));
    setSelected(session.session_id);
    setNotice(session.message);
  }, []);

  useEffect(() => {
    mounted.current = true;
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      refresh().catch(cause => { if (active) setError(errorMessage(cause)); })
        .finally(() => { if (active) setLoading(false); });
    });
    return () => { active = false; mounted.current = false; };
  }, [refresh]);

  const jobId = job?.status === 'running' ? job.id : '';
  useEffect(() => {
    if (!jobId) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const next = await request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`);
        if (!active) return;
        setJob(next);
        if (next.status !== 'running') {
          if (next.calibration_result) incorporate(next.calibration_result);
          if (next.status === 'failed' || next.status === 'cancelled') setError(next.message || '這次校準工作未完成');
          await refresh();
          return;
        }
      } catch (cause) {
        if (!active) return;
        setError(errorMessage(cause));
        if (cause instanceof ControllerRequestError && cause.status === 404) {
          setJob({ id: jobId, status: 'failed', message: '這個工作已不在控制器中，請更新校準狀態。' });
          return;
        }
      }
      if (active) timer = setTimeout(poll, 1200);
    }
    timer = setTimeout(poll, 400);
    return () => { active = false; clearTimeout(timer); };
  }, [jobId, request, refresh, incorporate]);

  const session = data?.sessions.find(value => value.session_id === selected) || null;
  const busy = sending || !!jobId;
  const usable = !!session && session.status !== 'invalidated' && session.status !== 'solving';
  const minimum = session?.minimum_distinct_views_for_solve || 12;
  const ready = usable && (session?.distinct_pose_count || 0) >= minimum;
  const extensionReady = usable && !!data?.spatial?.calibration_selected && (session?.distinct_pose_count || 0) >= 6;
  const solve = session?.last_solve;
  const found = session?.last_detection?.found;
  const corners = found ? (session?.board_spec.columns || 9) * (session?.board_spec.rows || 6) : 0;

  async function act(kind: 'start' | 'capture' | 'solve' | 'extend') {
    if (operation.current || busy) return;
    operation.current = true;
    setSending(true); setError(''); setNotice('');
    try {
      if (kind === 'start') {
        const next = await request<Session>('/api/engineering/calibration/sessions',
          { board_static: boardFixed, print_scale_checked: printChecked });
        if (mounted.current) incorporate(next);
      } else if (session) {
        const next = await request<Job>(`/api/engineering/calibration/sessions/${encodeURIComponent(session.session_id)}/${kind}`,
          kind === 'capture' ? { board_static: boardFixed } : {});
        if (mounted.current) {
          setJob(next);
          if (next.status !== 'running') {
            if (next.calibration_result) incorporate(next.calibration_result);
            if (next.status === 'failed' || next.status === 'cancelled') setError(next.message);
            await refresh();
          }
        }
      }
    } catch (cause) {
      if (mounted.current) setError(errorMessage(cause));
    } finally {
      operation.current = false;
      if (mounted.current) setSending(false);
    }
  }

  async function configure(kind: 'mode' | 'activate') {
    if (operation.current || busy) return;
    if (kind === 'activate' && !calibrationResultPresentation(session)?.canApply) return;
    operation.current = true;
    setSending(true); setError(''); setNotice('');
    try {
      if (kind === 'mode') {
        await request('/api/engineering/calibration/mode', { mode: modeChoice });
      } else {
        await request(`/api/engineering/calibration/sessions/${encodeURIComponent(session!.session_id)}/activate`,
          { solve_id: solve!.solve_id });
      }
      await refresh();
      if (mounted.current) setNotice(kind === 'mode' ? '操作模式已更新；下次操作會使用這個模式。' : '已選用通過驗證的相機校準。抓取時會再核對夾爪模型是否與這個版本相符。');
    } catch (cause) {
      if (mounted.current) setError(errorMessage(cause));
    } finally {
      operation.current = false;
      if (mounted.current) setSending(false);
    }
  }

  return <main className="cal-page">
    <header className="cal-header"><Link href="/" className="cal-brand">Le<span>Robot</span></Link>
      <Link href="/engineering">返回工程觀察室 <span aria-hidden="true">↗</span></Link></header>
    <div className="cal-title"><p className="cal-eyebrow">腕部相機 · 空間校準</p>
      <h1>把眼睛與手對準。</h1>
      <p>讓同一張影像裡的物件，對應到手臂實際能到達的位置。</p>
      <span className="cal-chip">只拍攝與計算，不會移動手臂</span>
    </div>

    {error && <div className="cal-alert" role="alert"><p>{error}</p>
      <button type="button" onClick={() => { setError(''); refresh().catch(cause => setError(errorMessage(cause))); }}>更新狀態</button></div>}
    {notice && !error && <output className="cal-notice">{notice}</output>}

    <div className="cal-grid">
      <section className="cal-card cal-prepare" aria-labelledby="prepare-title">
        <p className="cal-eyebrow">01 · 準備</p><h2 id="prepare-title">一張固定的 A4 棋盤。</h2>
        <a className="cal-download" href={data?.board_url || BOARD_URL} target="_blank" rel="noreferrer">開啟可列印的 A4 棋盤 <span aria-hidden="true">↗</span></a>
        <ol className="cal-instructions">
          <li>以 <strong>100%／實際大小</strong>列印，關閉「符合頁面」。用尺確認紙上的 <strong>10 cm 核對線</strong>確實長 10 cm。</li>
          <li>把紙平整固定在桌面上；完整棋盤需要在腕部相機內。從第一張到最後一張，<strong>棋盤都不要移動</strong>。</li>
          <li>用現有控制方式換手臂視角，停穩後再拍。需要不同位置與角度，連拍同一個姿勢不會增加有效視角。</li>
        </ol>
        <label className="cal-check"><input type="checkbox" checked={printChecked} disabled={busy} onChange={event => setPrintChecked(event.target.checked)} />
          <span>我已核對列印比例與 10 cm 長度</span></label>
        <label className="cal-check"><input type="checkbox" checked={boardFixed} disabled={busy} onChange={event => setBoardFixed(event.target.checked)} />
          <span>這批照片的棋盤位置保持固定</span></label>
        <button type="button" className="cal-button" disabled={busy || !printChecked || !boardFixed || loading}
          onClick={() => act('start')}>{session ? '開始新一批校準' : '開始校準'}</button>
        <p className="cal-caption">棋盤、相機安裝位置或解析度改變後，請開始新一批；不要接著使用舊照片。</p>
      </section>

      <section className="cal-card" aria-labelledby="capture-title">
        <div className="cal-section-top"><div><p className="cal-eyebrow">02 · 收集視角</p><h2 id="capture-title">讓相機從不同位置看棋盤。</h2></div>
          <button type="button" className="cal-text-button" onClick={() => setPreview(value => !value)}>{preview ? '關閉預覽' : '腕部相機預覽'}</button></div>
        {preview && <div className="cal-preview"><img src="/api/camera/stream" alt="目前腕部相機的即時畫面，拍攝按鈕不會移動手臂" /><span>腕部相機 · 即時預覽</span></div>}
        {loading ? <p className="cal-empty">正在讀取校準工作階段…</p> : !session ? <p className="cal-empty">完成左側準備後，就可以開始收集照片。</p> : <>
          <label className="cal-session-label" htmlFor="cal-session">目前照片批次</label>
          <select id="cal-session" value={selected} disabled={busy} onChange={event => { setSelected(event.target.value); setNotice(''); }}>
            {data?.sessions.map(value => <option key={value.session_id} value={value.session_id}>
              {new Date(value.created_at * 1000).toLocaleString('zh-TW')} · {value.collected_view_count} 張{value.status === 'invalidated' ? ' · 已失效' : ''}
            </option>)}
          </select>
          <div className="cal-metrics"><div><strong>{session.distinct_pose_count}<small> / {minimum}</small></strong><span>不同姿勢</span></div>
            <div><strong>{session.collected_view_count}</strong><span>已收錄照片</span></div>
            <div><strong>{corners || '—'}<small>{found ? ' / 54' : ''}</small></strong><span>最近一次角點</span></div></div>
          <progress max={minimum} value={Math.min(session.distinct_pose_count, minimum)} aria-label="不同校準視角收集進度" />
          <p className="cal-caption">至少 {minimum} 個不同姿勢；計算時還會檢查角度差異，並用保留的照片獨立驗證。</p>
          {session.last_detection && <p className={`cal-detection ${found ? 'is-found' : ''}`}>
            <span aria-hidden="true">{found ? '✓' : '○'}</span> {session.last_detection.reason || (found ? '已找到完整棋盤' : '尚未找到完整棋盤')}
            {found && <span> · 覆蓋畫面 {percent(session.last_detection.coverage_fraction)}</span>}</p>}
          {session.status === 'invalidated' && <p className="cal-alert">這批照片的棋盤已移動，請開始新一批校準。</p>}
          <div className="cal-actions"><button type="button" className="cal-button" disabled={busy || !usable || !boardFixed}
            onClick={() => act('capture')}>拍攝目前視角</button><span>只拍照，不移動手臂</span></div>
        </>}
      </section>

      <section className="cal-card cal-results" aria-labelledby="solve-title">
        <div><p className="cal-eyebrow">03 · 核對</p><h2 id="solve-title">用沒參與計算的照片，檢查是否真的對準。</h2>
          <p>計算相機內參與安裝位置，再核對手臂姿勢與影像是否一致。夾爪開口與接觸位置使用另一份模型。</p></div>
        <button type="button" className="cal-button cal-button-secondary" disabled={busy || !ready || !boardFixed} onClick={() => act('solve')}>計算並驗證校準</button>
        {data?.spatial?.calibration_selected && <div className="cal-actions">
          <button type="button" className="cal-button cal-button-secondary" disabled={busy || !extensionReady || !boardFixed}
            onClick={() => act('extend')}>驗證新的手臂姿勢</button>
          <span>已有相機校準時，至少 6 個不同姿勢即可檢查新工作範圍；實測誤差會保留。</span>
        </div>}
        {busy && <output className="cal-working"><span aria-hidden="true" className="cal-spinner" />{jobId ? job?.message || '正在處理校準工作…' : '正在送出…'}</output>}
        <CalibrationSolveResult session={session} busy={busy} onActivate={() => configure('activate')} />
        <p className="cal-save-note">夾爪模型：{!data ? '正在讀取設定…' : data.spatial?.jaw_calibration_selected
          ? '已選用。抓取時會核對相機版本、目前開口與模型誤差；這不代表已經抓取成功。'
          : '尚未選用。完成空夾爪的不同開口取樣與模型核對後，就能計算接觸位置。'}</p>
        <p className="cal-save-note">求解只會保存報告與候選設定，<strong>不會自動啟用</strong>，也不會改寫馬達內的校準。</p>
      </section>

      <section className="cal-card cal-results" aria-labelledby="mode-title">
        <div><p className="cal-eyebrow">工程設定 · 可回退</p><h2 id="mode-title">分開試用新版，也能切回舊版。</h2>
          <p>切換模式或套用校準前，請先停止手臂並處理好夾爪中的物件。設定切換本身不會移動手臂。</p></div>
        <div className="cal-mode-control"><label className="cal-session-label" htmlFor="cal-mode">操作模式</label>
          <select id="cal-mode" value={modeChoice} disabled={busy || !data} onChange={event => setModeChoice(event.target.value as Mode)}>
            <option value="legacy">舊版 · 回退到原有流程</option>
            <option value="shadow">工程觀察 · 操作仍使用舊版</option>
            <option value="v2">新版 · 本機空間規劃</option>
          </select>
          <button type="button" className="cal-button cal-button-secondary" disabled={busy || !data || modeChoice === data.spatial?.spatial_backend}
            onClick={() => configure('mode')}>套用模式</button>
        </div>
        <p className="cal-save-note">新版需要已驗證的座標校準；尚未準備好時會指出缺少什麼，避免反覆換角度。切回舊版不會刪除校準照片與報告。</p>
      </section>
    </div>
    <footer>校準與正式操作使用腕部相機。D435 仍只作工程旁觀驗證。</footer>
  </main>;
}
