'use client';
/* oxlint-disable next/no-img-element -- Live wrist snapshots must bypass image optimization and its cache. */
import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import {
  controllerRequest,
  readControllerPoll,
  progressPresentation,
} from '@/lib/client-status';
import {
  Plus,
  Mic,
  Camera,
  Search,
  Sparkles,
  CircleStop,
  X,
  ArrowUpRight,
  Check,
  LoaderCircle,
  Hand,
  ChevronRight,
  RotateCcw,
  ImagePlus,
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import {
  MotionRetargetCard,
  QUICK_MOTION_CHIPS,
  type MotionGPTPlan,
  type MotionGPTReceipt,
} from '@/components/MotionRetargetCard';
import {
  ER2SpatialCard,
  type ER2SpatialResult,
  type ThinkingLevel,
} from '@/components/ER2SpatialCard';
import { RealToSimCard } from '@/components/RealToSimCard';
type Frame = { id: string; url: string; captured_at: number; source?: string };
type GraspResult = {
  task?: string;
  completed?: boolean;
  place_confirmed?: boolean;
  stack_confirmed?: boolean;
  grasp_confirmed?: boolean;
  released?: boolean;
  state?: string;
  held_frame?: Frame;
  observations?: { completed?: boolean; after?: Frame }[];
};
type Action = {
  kind: string;
  label: string;
  reason: string;
  target: string;
  available: boolean;
  unavailable_reason: string;
};
type Item = {
  name: string;
  description: string;
  confidence: number;
  bbox: { x: number; y: number; w: number; h: number };
  frame_index: number;
  matches_query: boolean;
  safe_to_touch: boolean;
  actions: Action[];
};
type Scene = {
  id: string;
  query: string;
  summary: string;
  objects: Item[];
  selected_index: number;
  empty: boolean;
  uncertainty: string;
  frames: Frame[];
  elapsed: number;
  notice?: string;
  notice_kind?: string;
  performed?: string;
  grasp?: GraspResult;
  observation?: {
    mode: string;
    view_count: number;
    metric_calibrated: boolean;
  };
  details?: {
    visible: boolean;
    object_name: string;
    summary: string;
    attributes: string[];
    visible_text: string[];
    unknowns: string[];
    frame: Frame;
  };
  spatial?: {
    status: string;
    reason?: string;
    stability?: string;
    note?: string;
    objects: {
      name: string;
      status: string;
      distance_m?: number;
      projected_width_m?: number;
      projected_height_m?: number;
    }[];
  };
};
type CommandStep = {
  skill: string;
  label: string;
  direction?: string;
  distance_mm?: number;
  target?: string;
};
type ActionSuggestion = {
  kind: 'action' | 'ambiguous';
  query: string;
  instruction: string;
  title: string;
  summary: string;
  label: string;
  availability: 'available' | 'experimental' | 'unsupported' | 'clarify';
  reason: string;
};
type CommandResult = {
  plan?: {
    status: string;
    summary: string;
    steps: CommandStep[];
    unsupported_reason?: string;
  };
  completed: boolean;
  status: 'running' | 'completed' | 'blocked' | 'unsupported' | 'cancelled';
  steps: { skill: string; completed: boolean; reason?: string; receipt?: unknown }[];
  reason?: string;
};
type Job = {
  id: string;
  kind: string;
  status: string;
  stage: string;
  message: string;
  result: Scene | null;
  engineering_result?: unknown;
  manual_wave_requested?: boolean;
  manual_wave_phase?: 'assessing' | 'waving';
  command_result?: CommandResult;
  action_suggestion?: ActionSuggestion;
  motiongpt_plan?: MotionGPTPlan;
  motiongpt_result?: MotionGPTReceipt;
  events: { message: string; stage: string }[];
};
type Status = {
  camera: {
    connected: boolean;
    name: string;
    error: string;
    source?: string;
    age_ms?: number | null;
    image_quality?: { usable?: boolean };
  };
  robot: {
    connected: boolean;
    moving: boolean;
    stopped: boolean;
    error: string;
  };
  brain: {
    model: string;
    engine?: 'astra' | 'argon' | 'dreamlabs' | 'smolvla';
    thinking_level?: ThinkingLevel;
    dreamlabs_model?: string;
    smolvla_model?: string;
    dreamlabs_configured?: boolean;
    reasoning_effort?: string;
    api_key_configured?: boolean;
    codex_available?: boolean;
    transport?: string;
    available: boolean;
    verified: boolean;
    latency: number | null;
    active?: {
      id: string;
      schema: string;
      phase: 'starting' | 'sending' | 'connecting' | 'reconnecting' | 'reasoning' | 'validating';
      elapsed: number;
    } | null;
    error_kind?: string | null;
    last_call?: unknown;
  };
  er2?: {
    model: string;
    thinking_level?: ThinkingLevel;
    sdk_installed: boolean;
    api_key_configured: boolean;
    available: boolean;
    verified: boolean;
    last_latency_ms?: number | null;
  };
  job: Job | null;
  capabilities?: { pick?: boolean; place?: boolean; stack?: boolean; command?: boolean; motiongpt?: boolean; er2?: boolean };
  tactics?: { provider: string; available: boolean; model: string };
  grasp?: { state: string; low_power_owner: boolean };
};
type RecognitionResult = { isFinal: boolean; 0: { transcript: string } };
type Recognition = {
  lang: string;
  interimResults: boolean;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: { error: string }) => void) | null;
  onresult: ((event: { results: ArrayLike<RecognitionResult> }) => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
};
type RecognitionWindow = Window & {
  SpeechRecognition?: new () => Recognition;
  webkitSpeechRecognition?: new () => Recognition;
};
type ModelDocument = Document & {
  modelContext?: {
    registerTool(
      tool: {
        name: string;
        description: string;
        inputSchema: object;
        annotations: object;
        execute(input: unknown): Promise<unknown>;
      },
      options: { signal: AbortSignal },
    ): unknown;
  };
};
type StopResult = { held: boolean; failures: Record<string, string> };
type UploadResult = { id: string; url: string };

function actionPresentation(
  action: Pick<Action, 'kind' | 'label' | 'reason' | 'unavailable_reason'> & Partial<Pick<Action, 'target'>>,
  item: Item | null,
  status: Status | null,
  busy: boolean,
  connectionIssue: string,
) {
  const pick = action.kind === 'pick';
  const placement = ['place', 'stack'].includes(action.kind);
  const manipulation = pick || placement;
  const inspect = action.kind === 'inspect';
  const supported = ['inspect', 'pick', 'place', 'stack', 'wave'].includes(action.kind);
  const holding = ['holding', 'holding_unverified', 'releasing'].includes(
    status?.grasp?.state || '',
  );
  let reason = action.reason;
  // Old scenes only supply a name. Availability comes from the live controller;
  // every supported action reacquires its own evidence on the server.
  let enabled = supported;
  if (!supported) {
    enabled = false;
    reason = action.unavailable_reason || '這個動作目前尚未開放';
  } else if (
    busy ||
    status?.robot?.moving ||
    status?.job?.status === 'running'
  ) {
    enabled = false;
    reason = '請先等目前工作結束。';
  } else if (connectionIssue || !status?.camera?.connected) {
    enabled = false;
    reason = '請先確認控制器與腕部相機的連線。';
  } else if (!inspect && status.robot.stopped) {
    enabled = false;
    reason = '手臂已停止，恢復後才能執行動作。';
  } else if (!inspect && (holding || status.grasp?.low_power_owner)) {
    enabled = false;
    reason = '夾爪正在保持物件，請先完成目前操作。';
  } else if (manipulation && status.capabilities?.[action.kind as 'pick' | 'place' | 'stack'] !== true) {
    enabled = false;
    reason = pick ? '實驗抓取尚未開放，目前可以搜尋和觀察物件。' : '這項實驗物件操作尚未開放，目前可以查看細節。';
  } else if (manipulation && !item?.safe_to_touch) {
    enabled = false;
    reason =
      '目前影像還無法確認這個物件適合抓取。可先查看細節，確認物件與周圍情況。';
  } else if (placement && !action.target?.trim()) {
    enabled = false;
    reason = '請在搜尋框寫出要放到哪裡，例如「把圓筒放到盒子上」。';
  } else if (action.kind !== 'wave' && !item?.name.trim()) {
    enabled = false;
    reason = '請先選擇要查看的物件。';
  }
  return {
    enabled,
    label: manipulation ? `${action.label} · 實驗` : action.label,
    reason,
  };
}

function resultActions(item: Item): Action[] {
  if (
    !item.matches_query ||
    item.actions.some((action) => action.kind === 'pick')
  )
    return item.actions;
  return [
    ...item.actions,
    {
      kind: 'pick',
      label: '試著抓起來',
      target: '',
      reason: '先安排整段路徑，再連續對準，並在合爪與抬起時確認結果',
      available: false,
      unavailable_reason: '',
    },
  ];
}

function sceneNotice(scene: Scene | null, job: Job | null) {
  if (!scene?.notice) return null;
  const recognitionOnly = scene.notice_kind === 'recognition_only' ||
    scene.notice === '已認出你點的物件；抓取位置尚未通過精度驗證。';
  if (!recognitionOnly) return scene.notice;
  // A retained recognition result is not a verdict on the current operation.
  // Its progress, failure reason and receipt are rendered from the actual job.
  if (job && ['grasp-attempt', 'object-program', 'command'].includes(job.kind)) return null;
  return '已認出物件，尚未嘗試抓取。可選擇下方的「嘗試輕夾」。';
}

function ResultActionButtons({
  item,
  status,
  busy,
  connectionIssue,
  onAction,
  inline = false,
}: {
  item: Item;
  status: Status | null;
  busy: boolean;
  connectionIssue: string;
  onAction: (kind: string, item: Item, action: Action) => void;
  inline?: boolean;
}) {
  const actions = resultActions(item);
  const pick = actions.find((action) => action.kind === 'pick');
  const picking =
    pick && actionPresentation(pick, item, status, busy, connectionIssue);
  const missingDestination = actions.some((action) =>
    ['place', 'stack'].includes(action.kind) && !action.target?.trim());
  return (
    <>
      <div className={inline ? 'inline-actions' : 'action-pills'}>
        {actions
          .filter(
            (action) =>
              !inline ||
              action.available ||
              ['inspect', 'pick', 'place', 'stack'].includes(action.kind),
          )
          .map((action, index) => {
            const view = actionPresentation(
              action,
              item,
              status,
              busy,
              connectionIssue,
            );
            return (
              <button
                key={index}
                disabled={!view.enabled}
                title={view.reason}
                className={!inline && !view.enabled ? 'unavailable' : ''}
                onClick={() => onAction(action.kind, item, action)}
              >
                {!inline &&
                  (action.kind === 'inspect' ? (
                    <Search size={16} />
                  ) : (
                    <Hand size={16} />
                  ))}
                <span>{view.label}</span>
                {inline ? (
                  <ChevronRight size={13} />
                ) : (
                  view.enabled && <ArrowUpRight size={15} />
                )}
              </button>
            );
          })}
      </div>
      {missingDestination && (
        <p className="calibration-note">請在搜尋框寫出要放到哪裡，例如「把圓筒放到盒子上」。</p>
      )}
      {picking && !picking.enabled && (
        <p className="calibration-note">抓取暫不可用：{picking.reason}</p>
      )}
    </>
  );
}

function observationSummary(observation: Scene['observation']) {
  return observation?.mode === 'wrist_roll_arc' &&
    Number.isInteger(observation.view_count) &&
    observation.view_count > 1
    ? `已從 ${observation.view_count} 個視角觀察`
    : '腕部相機實際觀察';
}

function ResultStats({ scene }: { scene: Scene }) {
  return (
    <p className="result-stats">
      找到{' '}
      {
        scene.objects.filter((object) => object.matches_query || !scene.query)
          .length
      }{' '}
      個相關物件 · {scene.elapsed} 秒 · {observationSummary(scene.observation)}
    </p>
  );
}

function actionRequest(kind: string, query = '', destination = '') {
  if (!['inspect', 'pick', 'place', 'stack', 'wave'].includes(kind))
    throw new Error('這個動作目前尚未開放');
  const name = kind === 'wave' ? '' : query.trim();
  if (kind !== 'wave' && !name) throw new Error('請先選擇物件');
  const placement = ['place', 'stack'].includes(kind);
  const target = destination.trim();
  if (placement && !target)
    throw new Error('請在搜尋框寫出要放到哪裡，例如「把圓筒放到盒子上」。');
  const instruction = placement
    ? `把「${name}」${kind === 'stack' ? '疊放在' : '放到'}「${target}」${kind === 'stack' ? '上面' : ''}`
    : name;
  return {
    path: '/api/object-action',
    body: { kind, query: instruction },
  };
}

function commandRequest(instruction: string) {
  const value = instruction.trim();
  if (!value || value.length > 500)
    throw new Error('請輸入 1–500 字的動作指令');
  return { path: '/api/command', body: { instruction: value } };
}

function commandPresentation(
  instruction: string,
  status: Status | null,
  busy: boolean,
  connectionIssue: string,
) {
  let reason = '';
  if (busy || status?.robot?.moving || status?.job?.status === 'running')
    reason = '請先等目前工作結束。';
  else if (connectionIssue || !status?.camera?.connected)
    reason = '請先確認控制器與腕部相機的連線。';
  else if (status.robot.stopped)
    reason = '手臂已停止，恢復後才能執行動作。';
  else if ((status.grasp?.low_power_owner || ['holding', 'holding_unverified', 'releasing'].includes(status.grasp?.state || ''))
    && !/^(?:(?:放開|釋放)(?:夾爪|物件|手上的物件)?|release(?: the object)?)[。！!]?$/i.test(instruction.trim()))
    reason = '夾爪正在保持物件，請先完成目前操作或搜尋「放開」。';
  else if (status.capabilities?.command !== true)
    reason = '動作指令尚未開放。';
  else if (!instruction.trim())
    reason = '請先搜尋想做的事。';
  else if (instruction.trim().length > 500)
    reason = '請將動作指令縮短至 500 字以內。';
  return { enabled: !reason, reason: reason || '先安排整段動作，再連續執行，並在關鍵步驟核對結果。' };
}

function searchActionSuggestion(job: Job | null) {
  return job?.kind === 'search' && job.status === 'completed'
    ? job.action_suggestion || null : null;
}

function suggestionPresentation(
  suggestion: ActionSuggestion,
  status: Status | null,
  busy: boolean,
  connectionIssue: string,
) {
  const actionable = suggestion.kind === 'action'
    && ['available', 'experimental'].includes(suggestion.availability);
  if (!actionable) return {
    actionable: false, enabled: false,
    reason: suggestion.reason || (suggestion.kind === 'ambiguous'
      ? '再多說一點，你想找什麼或做什麼？' : '這個動作目前還做不到。'),
  };
  return { actionable: true,
    ...commandPresentation(suggestion.instruction, status, busy, connectionIssue) };
}

function ActionSuggestionCard({ suggestion, status, busy, connectionIssue, onExecute }: {
  suggestion: ActionSuggestion;
  status: Status | null;
  busy: boolean;
  connectionIssue: string;
  onExecute: () => void;
}) {
  const view = suggestionPresentation(suggestion, status, busy, connectionIssue);
  return (
    <article className="suggestion" aria-label="搜尋推薦動作" style={{ marginBottom: 28 }}>
      <div className="sponsored">
        <Sparkles size={15} />
        <strong>推薦動作</strong>
        <span>· 由好奇心贊助</span>
      </div>
      <h2>{suggestion.title || '讓我來幫忙。'}</h2>
      {suggestion.summary && <p>{suggestion.summary}</p>}
      {view.actionable && (
        <div className="action-pills">
          <button type="button" disabled={!view.enabled} title={view.reason} onClick={onExecute}>
            <Hand size={16} />
            {suggestion.label || '就這麼做'}
            <ArrowUpRight size={15} />
          </button>
        </div>
      )}
      <p className="calibration-note">
        {view.enabled ? '點一下，我會先看看眼前，再開始動作。' : view.reason}
      </p>
    </article>
  );
}

function commandOutcome(job: Job | null, stopped = false) {
  if (job?.kind !== 'command') return null;
  const receipt = job.command_result;
  if (!receipt && job.status === 'running' && !stopped) return null;
  const planned = receipt?.plan?.steps || [];
  const recorded = receipt?.steps || [];
  const rows = planned.map((step, index) => ({
    label: step.label,
    verified: recorded[index]?.skill === step.skill && recorded[index]?.completed === true,
    attempted: recorded[index]?.skill === step.skill,
  }));
  const cancelled = stopped || job.status === 'cancelled' || receipt?.status === 'cancelled';
  const confirmed = !cancelled && receipt?.plan?.status !== 'unsupported'
    && job.status === 'completed' && receipt?.status === 'completed'
    && receipt.completed === true && planned.length > 0 && recorded.length === planned.length
    && rows.every((row) => row.verified);
  const running = !cancelled && job.status === 'running' && receipt?.status === 'running';
  const reobserving = running && job.stage === 'reobserving';
  const unsupported = receipt?.status === 'unsupported' || receipt?.plan?.status === 'unsupported';
  const title = cancelled ? '動作指令已停止'
    : confirmed ? '動作指令已完成'
    : unsupported ? '這個指令目前還做不到'
    : reobserving ? '換個角度看清楚，再接著操作'
    : running ? '正在逐步執行'
    : receipt?.status === 'blocked' || job.status === 'failed' ? '動作指令尚未完成'
    : '動作結果仍待確認';
  return {
    confirmed, running, reobserving, title, rows,
    summary: receipt?.plan?.summary || '',
    reason: confirmed ? '' : receipt?.reason || receipt?.plan?.unsupported_reason
      || (cancelled ? '後續步驟已停止；已驗證的步驟保留在下方。'
        : reobserving ? '正在重新觀察同一個目標，接著判斷如何繼續原本的指令。'
          : running ? '每一步都會核對實際結果，再決定是否繼續。'
          : job.message || '尚未收到完整的步驟驗證結果。'),
  };
}

function CommandReceipt({ job, stopped = false, suppressReason = false }: {
  job: Job | null;
  stopped?: boolean;
  suppressReason?: boolean;
}) {
  const outcome = commandOutcome(job, stopped);
  if (!outcome) return null;
  // A primitive can move and then fail before returning a receipt. Its missing
  // completion record does not prove that the first pending step never started.
  const firstPending = outcome.rows.findIndex((row) => !row.verified);
  return (
    <section className="suggestion" aria-label="動作指令結果" aria-live="polite"
      style={{ marginBottom: 24, borderColor: outcome.confirmed ? '#ceead6' : undefined }}>
      <div className="sponsored">
        {outcome.confirmed ? <Check size={16} /> : <Hand size={16} />}
        <strong>動作清單</strong>
        <span>· 按實際結果核對</span>
      </div>
      <h2 style={{ color: outcome.confirmed ? '#137333' : '#202124' }}>{outcome.title}</h2>
      {outcome.summary && <p>{outcome.summary}</p>}
      {outcome.rows.length > 0 && (
        <ol style={{ listStyle: 'none', padding: 0, margin: '20px 0 0' }}>
          {outcome.rows.map((row, index) => (
            <li key={index} style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 0', borderTop: '1px solid #edf0f3' }}>
              <span aria-hidden="true" style={{ color: '#70757a', minWidth: 16 }}>{index + 1}</span>
              <span style={{ flex: 1 }}>{row.label}</span>
              <span style={{ color: row.verified ? '#137333' : '#70757a', fontSize: 13, whiteSpace: 'nowrap' }}>
                {row.verified ? '已驗證' : outcome.running && index === firstPending
                  ? outcome.reobserving ? '換角度觀察中' : '執行中'
                  : row.attempted || index === firstPending ? '未確認完成' : '尚未執行'}
              </span>
            </li>
          ))}
        </ol>
      )}
      {!suppressReason && outcome.reason && (serialConnectionFailure(outcome.reason) || motorCommunicationFailure(outcome.reason) || motionCorrectionFailure(outcome.reason)
        ? <ErrorNotice message={outcome.reason} kind="command" />
        : <p style={{ marginTop: 16, marginBottom: 0 }}>{outcome.reason}</p>)}
    </section>
  );
}

function retireMissingJob(job: Job | null, missingId: string): Job | null {
  return job?.id === missingId
    ? { ...job, status: 'unavailable', message: '先前工作紀錄已無法取得' }
    : job;
}

function reconcilePolledJobError(
  message: string,
  owner: { jobId: string; message: string } | null,
  current: Job | null,
) {
  if (!owner || !current || message !== owner.message) return message;
  return current.id !== owner.jobId || current.status === 'completed' ? '' : message;
}

function waveOutcome(job: Job | null, stopped = false) {
  if (stopped || job?.kind !== 'wave' || job.status !== 'completed')
    return null;
  const receipt = job.engineering_result as {
    wave_completed?: boolean;
    cycles?: number;
    measured_roll_range_deg?: number[];
  } | null;
  if (receipt?.wave_completed !== true)
    return {
      confirmed: false,
      title: '揮手結果仍待確認',
      description: '工作已結束，但尚未收到完整的揮手結果。',
    };
  const range = receipt.measured_roll_range_deg;
  const cycles = receipt.cycles;
  const measurements = [];
  if (Number.isInteger(cycles) && (cycles || 0) > 0)
    measurements.push(`完成 ${cycles} 次揮手`);
  if (
    Array.isArray(range) &&
    range.length === 2 &&
    range.every(Number.isFinite) &&
    range[0] <= range[1]
  )
    measurements.push(
      `手腕實測角度 ${range[0].toFixed(1)}° 至 ${range[1].toFixed(1)}°`,
    );
  return {
    confirmed: true,
    title: job.message || '揮手動作已完成',
    description: measurements.join(' · '),
  };
}

function retainWaveReceipt(previous: Job | null, current: Job | null) {
  return current &&
    current.id !== previous?.id &&
    waveOutcome(current)?.confirmed
    ? current
    : previous;
}

function manualWaveAvailable(
  job: Job | null,
  status: Status | null,
  sending: boolean,
  connectionIssue: string,
  stopped: boolean,
) {
  return Boolean(
    job?.kind === 'wave' && job.status === 'running' &&
    status?.job?.id === job.id && status.job.status === 'running' &&
    status.robot.stopped === false && !stopped && !sending && !connectionIssue &&
    !job.manual_wave_requested && !status.job.manual_wave_requested &&
    job.manual_wave_phase !== 'waving' && status.job.manual_wave_phase !== 'waving',
  );
}

function acknowledgeManualWave(current: Job | null, response: Job, requestedId: string) {
  if (response.id !== requestedId || response.kind !== 'wave' || !response.manual_wave_requested ||
      current?.id !== requestedId || current.kind !== 'wave' || current.status !== 'running')
    return current;
  // A POST acknowledges intent. Polling remains authoritative for motion and
  // completion; a slower acknowledgement must not replace a newer snapshot.
  return { ...current, manual_wave_requested: true };
}

function ManualWaveStep({ enabled, onRequest }: { enabled: boolean; onRequest: () => void }) {
  return (
    <button type="button" className="wave-step-link" disabled={!enabled}
      title="強制執行『揮手』" aria-label="強制執行『揮手』" onClick={onRequest}>
      揮手
    </button>
  );
}

function cameraPreviewNeedsLook(status: Status | null) {
  const camera = status?.camera;
  // The controller reports freshness and image usability even while this
  // dialog is closed. Preserve an observation's framing when opening it.
  const usableCurrentView = Boolean(
    camera?.connected && camera.source === 'wrist' && !camera.error &&
    typeof camera.age_ms === 'number' && Number.isFinite(camera.age_ms) &&
    camera.age_ms >= 0 && camera.age_ms < 1000 &&
    camera.image_quality?.usable === true,
  );
  return Boolean(
    status &&
    !usableCurrentView &&
    !status.robot.stopped &&
    !status.grasp?.low_power_owner &&
    !['holding', 'holding_unverified', 'releasing'].includes(
      status.grasp?.state || '',
    ),
  );
}

function inferenceProgressDescription(
  job: Job | null,
  status: Status | null,
  showProgress: boolean,
  stopped: boolean,
) {
  const active = status?.brain.active;
  if (!showProgress || stopped || status?.robot.stopped !== false ||
      job?.status !== 'running' || status?.job?.status !== 'running' ||
      status.job.id !== job.id || !active?.id) return null;
  const brainName = status?.brain?.model?.includes('gemini') ? 'Argon ER 2' : (status?.brain?.model ? 'Argon' : 'Astra');
  const descriptions: Record<string, string> = {
    starting: `正在準備 ${brainName}…`,
    sending: active.schema === 'SearchIntent' ? `正在傳送指令給 ${brainName}…` : `正在傳送畫面與指令給 ${brainName}…`,
    connecting: `等待 ${brainName} 回應…`,
    reconnecting: `連線中斷，正在重新連接 ${brainName}…`,
    reasoning: ['SearchIntent', 'SkillPlan'].includes(active.schema)
      ? `${brainName} 正在理解指令…` : `${brainName} 正在看畫面…`,
    validating: `正在整理 ${brainName} 的回應…`,
  };
  const description = descriptions[active.phase];
  if (!description) return null;
  // Show measured server time only, without a client-side countdown or ETA.
  return Number.isFinite(active.elapsed) && active.elapsed >= 0
    ? `${description}（已等待 ${Math.floor(active.elapsed)} 秒）`
    : description;
}

function jobPresentation(job: Job | null, graspState = '') {
  if (job?.status === 'running' && job.stage === 'reobserving' &&
      ['grasp-attempt', 'object-program', 'command'].includes(job.kind)) {
    return {
      title: '換個角度看清楚，再接著操作',
      description: job.message || '正在從新的腕部視角觀察同一個目標，接著會重新安排操作路徑。',
      steps: ['換個角度', '接著操作', '核對'],
      current: 0,
    };
  }
  if (job?.kind === 'search' && job.stage === 'understanding') {
    return {
      title: '正在理解你的搜尋',
      description: '先了解你想找什麼，或想讓我做什麼。',
      steps: ['理解', '整理結果'],
      current: 0,
    };
  }
  if (job?.kind === 'command') {
    return {
      title: job.message || '正在理解你的動作指令',
      description: '先安排整段動作，再連續執行；在抓住、放下等關鍵步驟核對結果。',
      steps: ['規劃', '執行', '核對'],
      current: ['observing', 'verifying', 'verify', 'checking', 'done'].includes(job.stage) ? 2
        : ['moving', 'executing'].includes(job.stage) ? 1 : 0,
    };
  }
  if (job?.kind === 'object-program') {
    const current = ['verifying', 'checking', 'verify', 'done'].includes(job.stage) ? 2
      : ['moving', 'executing'].includes(job.stage) ? 1 : 0;
    return {
      title: job.message || '正在安排物件操作',
      description: current === 0
        ? '先安排物件到目的地的整段路徑。'
        : current === 1
          ? '依照計畫連續移動，合爪或放開前會核對物件狀態。'
          : '正在確認是否真的抓住、放穩；動作結束不代表操作成功。',
      steps: ['規劃', '執行', '核對'],
      current,
    };
  }
  if (job?.kind === 'wave') {
    return {
      title: job.message || '準備揮揮手',
      description: job.manual_wave_requested
        ? '依你的指示，略過障礙物判斷；保持目前位置後原地揮手。'
        : '先看手邊空間；需要時抬高或退開，再向你揮手。',
      steps: ['準備', '揮手', '完成'],
      current: job.stage === 'moving' ? 1 : 0,
    };
  }
  if (job?.kind === 'inspect') {
    return {
      title: job.message || '正在查看物件',
      description: '用最新腕部影像查看外觀、文字與可確認的細節。',
      steps: ['拍照', '查看', '詳細資訊'],
      current: job.stage === 'thinking' ? 1 : 0,
    };
  }
  if (job?.kind === 'grasp-attempt') {
    const planning = ['starting', 'planning'].includes(job.stage);
    const current =
      graspState === 'closing'
        ? 1
        : [
              'contact_candidate',
              'verifying',
              'holding',
              'holding_unverified',
            ].includes(graspState)
          ? 2
          : 0;
    return {
      title: job.message || '準備嘗試抓取',
      description:
        current === 0 && planning
          ? '正在重新觀察、確認抓取位置並檢查接近路徑；通過檢查後才會接近與合爪。'
          : current === 0
          ? '先安排整段接近與抓取路徑，連續移動後在合爪、抬起時核對。'
          : current === 1
            ? '正在輕合夾爪，接著確認物件是否真的被拿起。'
            : '正在核對物件是否離開桌面；完成前不代表已經抓住。',
      steps: ['對準', '輕合', '確認'],
      current,
    };
  }
  if (job?.kind === 'grasp-release') {
    return {
      title: job.message || '準備釋放物件',
      description: '確認物件已有支撐，張開夾爪後再核對是否已釋放。',
      steps: ['確認支撐', '張開', '確認釋放'],
      current: graspState === 'releasing' ? 1 : 0,
    };
  }
  if (job?.kind === 'home') {
    return {
      title: job.message || '正在將手臂歸回正前方預設原位',
      description: '以連續平滑軌跡將機械臂歸回正前方標準原位 (shoulder_pan = 2030)。',
      steps: ['準備', '歸位', '完成'],
      current: job.stage === 'done' ? 2 : 1,
    };
  }
  if (job?.kind === 'motiongpt') {
    return {
      title: job.message || '正在執行單手自然動作',
      description: '依 AIRI MotionGPT HumanML3D 單手骨架重映射軌跡，以 1:1 真人速度連續執行。',
      steps: ['重映射', '執行', '完成'],
      current: job.stage === 'done' ? 2 : job.stage === 'moving' ? 1 : 0,
    };
  }
  const current =
    job?.stage === 'moving'
      ? 2
      : ['thinking', 'measuring'].includes(job?.stage || '')
        ? 1
        : 0;
  return {
    title: job?.message || '準備看看眼前的世界',
    description:
      job?.stage === 'measuring'
        ? '以實際關節角度與不同視角推估距離，不需要人工輸入尺寸。'
        : job?.stage === 'thinking'
          ? '腕部影像已送交 Argon ER 2，正在理解物件與操作方式。'
          : job?.stage === 'moving'
            ? 'LeRobot 正在執行這次動作。'
            : 'LeRobot 正在用自己的視角觀察。',
    steps: ['觀察', '理解', '動手'],
    current,
  };
}

function graspOutcome(
  job: Job | null,
  stopped = false,
): {
  confirmed: boolean;
  title: string;
  description: string;
  frame?: Frame;
} | null {
  if (
    stopped ||
    !job ||
    job.status !== 'completed' ||
    !['grasp-attempt', 'grasp-release', 'object-program'].includes(job.kind)
  )
    return null;
  const raw = job.result?.grasp ?? job.engineering_result;
  const result =
    raw && typeof raw === 'object' && !Array.isArray(raw)
      ? (raw as GraspResult)
      : null;
  if (job.kind === 'object-program') {
    const stack = result?.task === 'stack';
    const placement = result?.task === 'place';
    const confirmed = result?.completed === true &&
      (stack ? result.stack_confirmed === true : placement && result.place_confirmed === true);
    return {
      confirmed,
      title: confirmed ? (stack ? '已確認堆疊完成' : '已確認物件放到目的地') : '物件操作結果仍待確認',
      description: confirmed
        ? '腕部影像已確認物件由指定目的地穩定支撐，且夾爪已空。'
        : '工作已結束，尚未確認完整操作成功；請先查看物件與夾爪狀態。',
    };
  }
  if (job.kind === 'grasp-release') {
    return result?.released === true
      ? {
          confirmed: true,
          title: '物件已釋放',
          description: '已確認夾爪張開，並完成釋放檢查。',
        }
      : {
          confirmed: false,
          title: '釋放結果仍待確認',
          description: '工作已結束，請先確認物件與夾爪的狀態。',
        };
  }
  if (result?.grasp_confirmed === true) {
    return {
      confirmed: true,
      title: '已拿起物件',
      description: '腕部影像已確認物件離桌；目前保持姿勢，等待下一步。',
      frame: result.held_frame,
    };
  }
  return {
    confirmed: false,
    title:
      result?.grasp_confirmed === false
        ? '尚未確認拿起物件'
        : '抓取結果仍待確認',
    description:
      result?.state === 'holding_unverified'
        ? '夾爪已輕合，但還不能確認物件已離開桌面。目前保持姿勢，等待下一步。'
        : '工作已結束，請先確認物件與夾爪的狀態。',
    frame: result?.held_frame,
  };
}
function observationEvidence(job: Job | null, currentJob: Job | null): {
  frame: Frame;
  description: string;
} | null {
  if (!job?.id || currentJob?.id !== job.id ||
      !['grasp-attempt', 'object-program', 'command'].includes(job.kind) ||
      !['running', 'completed', 'failed', 'cancelled'].includes(job.status)) return null;
  // Use the receipt attached to this job, never a retained inspection scene.
  // A retained scene can still contain a previous operation's grasp result.
  const raw = job.kind === 'command'
    ? [...(job.command_result?.steps || [])].reverse().map((step) => step.receipt)
      .find((value) => value && typeof value === 'object' && !Array.isArray(value) &&
        Array.isArray((value as GraspResult).observations))
    : job.engineering_result;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const receipt = raw as GraspResult;
  if (!Array.isArray(receipt.observations)) return null;
  const last = [...receipt.observations].reverse().find((item) => item?.completed === true);
  const frame = last?.after;
  if (!frame || frame.source !== 'wrist' || typeof frame.id !== 'string' ||
      !/^[a-zA-Z0-9_-]+$/.test(frame.id) || frame.url !== `/api/frames/${frame.id}` ||
      typeof frame.captured_at !== 'number' || !Number.isFinite(frame.captured_at) ||
      frame.captured_at <= 0) return null;
  const confirmed = job.status === 'completed' && receipt.completed === true &&
    (job.kind !== 'command' || job.command_result?.completed === true) &&
    (receipt.task === 'stack' ? receipt.stack_confirmed === true
      : receipt.task === 'place' ? receipt.place_confirmed === true
        : receipt.task === 'grasp' && receipt.grasp_confirmed === true);
  return {
    frame,
    description: confirmed ? '這是操作前換角度時的觀察紀錄。'
      : '已重新觀察；物件操作尚未完成',
  };
}
function ObservationEvidenceCard({ evidence }: {
  evidence: NonNullable<ReturnType<typeof observationEvidence>>;
}) {
  return (
    <section className="suggestion" aria-label="換角度後的畫面"
      style={{ marginBottom: 24, padding: 20 }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 18 }}>
        <img src={evidence.frame.url} alt="換角度後的腕部實際影像"
          style={{ display: 'block', width: 180, maxWidth: '100%', borderRadius: 10 }} />
        <div>
          <div className="sponsored"><Camera size={15} /><strong>換角度後的畫面</strong></div>
          <p style={{ margin: '8px 0', fontSize: 14 }}>{evidence.description}</p>
          <span style={{ fontSize: 12, color: '#70757a' }}>
            {new Date(evidence.frame.captured_at * 1000).toLocaleTimeString('zh-TW')} · 腕部相機
          </span>
        </div>
      </div>
    </section>
  );
}
function ObjectDetails({
  details,
  earlier = false,
}: {
  details: NonNullable<Scene['details']>;
  earlier?: boolean;
}) {
  return (
    <section
      className="suggestion"
      aria-label="物件詳細資訊"
      aria-live="polite"
      style={{ marginBottom: 24 }}
    >
      <div className="sponsored">
        <Camera size={15} />
        <strong>詳細資訊</strong>
        <span>· {earlier ? '先前的詳細觀察' : '最新腕部觀察'}</span>
      </div>
      <h2>
        {details.visible ? details.object_name : '這次畫面中沒有確認到目標'}
      </h2>
      <p>{details.summary}</p>
      {details.frame?.url && (
        <figure style={{ margin: '16px 0' }}>
          <img
            src={details.frame.url}
            alt={`${details.object_name}的${earlier ? '先前' : '本次'}腕部觀察影像`}
            style={{
              display: 'block',
              width: '100%',
              maxWidth: 460,
              borderRadius: 12,
            }}
          />
          <figcaption style={{ fontSize: 12, color: '#70757a', marginTop: 6 }}>
            {new Date(details.frame.captured_at * 1000).toLocaleTimeString(
              'zh-TW',
            )}{' '}
            · 腕部相機
          </figcaption>
        </figure>
      )}
      {details.attributes.length > 0 && (
        <>
          <h3>看得見的特徵</h3>
          <ul>
            {details.attributes.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </>
      )}
      {details.visible_text.length > 0 && (
        <>
          <h3>辨識到的文字</h3>
          <ul>
            {details.visible_text.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </>
      )}
      {details.unknowns.length > 0 && (
        <>
          <h3>還不能確定</h3>
          <ul>
            {details.unknowns.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function waveErrorPresentation(message: string, kind?: string) {
  const clearancePrefix = /^(揮腕空間尚未確認|揮手空間尚未確認|抬手路徑尚未確認)：/;
  const clearanceWords = /淨空|旋轉範圍|揮手範圍|掃掠|掃過|外物|外部物件|遮擋|遮住|阻擋|相交|分離|足夠空間/;
  const operationalFailure = /讀取失敗|寫入失敗|連線失敗|畫面中斷|影像中斷|相機.*(?:中斷|斷線|重新連接)|未到達目標位置|幅度不足|回程不足/;
  if (!clearancePrefix.test(message) &&
      !(kind === 'wave' && clearanceWords.test(message) && !operationalFailure.test(message)))
    return null;
  const raising = message.startsWith('抬手路徑尚未確認：');
  const motion = raising ? '抬手' : '揮手';
  const reason = message.replace(clearancePrefix, '');
  if (/找不到.*(?:退讓|側移)路線/.test(reason))
    return {
      title: '還沒找到能避開物件的路線，先停在這裡。',
      detail: '這次尚未完成揮手，可以查看相機畫面了解目前位置。',
    };
  const human = reason.split(/[。；;，,\n]/).some((clause) =>
    /人手|人的手|人員|人體|人物|動物|寵物|有人/.test(clause) &&
    !/(?:未見|未看到|未看見|沒有(?:看到|看見|發現)?|無可見的?|無明顯的?|無)(?:任何|其他)?(?:人手|人的手|人員|人體|人物|動物|寵物|人)/.test(clause),
  );
  if (human)
    return {
      title: `還不能確認人或寵物已避開${motion}範圍，我先停下來。`,
      detail: '請讓附近的人與寵物離開手臂活動範圍，再試一次。',
    };
  if (/找不到合適的抬手路徑|未提供抬升候選|沒有.*抬手路徑/.test(reason))
    return {
      title: '還沒找到能避開物件的抬手路線，先停在這裡。',
      detail: '這次尚未完成揮手，可以查看相機畫面了解目前位置。',
    };
  const nearbyObject = /圓筒|瓶子|瓶身|杯子|盒子|物件|障礙物/.test(reason) &&
    /貼近|緊鄰|靠近|較近|很近|接近|遮|擋|相交|重疊|交疊|碰|間隔|距離|分離|外側|旁/.test(reason);
  if (nearbyObject)
    return {
      title: `${/圓筒/.test(reason) ? '圓筒' : '旁邊的物件'}可能擋到${motion}，我先停下來。`,
      detail: '目前還不能確認有足夠空間，這次尚未完成動作。',
    };
  return {
    title: `還看不清楚${raising ? '抬手的路徑' : '揮手的空間'}，我先停下來。`,
    detail: '目前還不能確認空間足夠，可以先查看相機畫面。',
  };
}

function motorCommunicationFailure(message: string) {
  return /馬達(?:讀取|寫入)失敗/.test(message);
}

function serialConnectionFailure(message: string) {
  return message.trim() === 'device reports readiness to read but returned no data (device disconnected or multiple access on port?)';
}

function motionCorrectionFailure(message: string) {
  return /^位置修正中止：(shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll): (no progress after four corrections:|correction would cross safe joint limit)/.exec(message);
}

function wristEndpointFailure(message: string) {
  const match = /^未到達目標位置，已保持姿勢：\{\s*(['"])wrist_roll\1\s*:\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\}$/.exec(message);
  if (!match) return null;
  const ticks = Number(match[2]);
  return Number.isFinite(ticks) ? (Math.abs(ticks) * (360 / 4096)).toFixed(2) : null;
}

function ErrorNotice({ message, kind }: { message: string; kind?: string }) {
  if (message.includes('校準') && (message.includes('相機') || message.includes('夾墊') || message.includes('座標')))
    return <div className="error" role="alert">
      <p>{message}</p>
      <Link href="/engineering/calibration">開啟校準頁與可列印棋盤格 →</Link>
    </div>;
  if (serialConnectionFailure(message))
    return (
      <div className="error motion-error" role="alert">
        <strong>手臂的 USB 連線中斷，或通訊埠被其他程式占用。</strong>
        <p>請確認 USB 連線，並關閉其他正在連接手臂的控制程式後再試。</p>
        <details>
          <summary>查看原因</summary>
          <code>{message}</code>
        </details>
      </div>
    );
  if (motorCommunicationFailure(message))
    return (
      <div className="error motion-error" role="alert">
        <strong>手臂通訊暫時中斷，已停止動作。</strong>
        <details>
          <summary>查看原因</summary>
          <code>{message}</code>
        </details>
      </div>
    );
  const wristErrorDegrees = wristEndpointFailure(message);
  if (wristErrorDegrees !== null)
    return (
      <div className="error motion-error" role="alert">
        <strong>手腕尚未到位，已先停止。</strong>
        <p>距離目標約 {wristErrorDegrees}°；這次{kind === 'wave' ? '揮手' : '動作'}尚未完成。</p>
        <details>
          <summary>查看原因</summary>
          <code>{message}</code>
        </details>
      </div>
    );
  const stopped = motionCorrectionFailure(message);
  const joints: Record<string, string> = {
    shoulder_pan: '底座',
    shoulder_lift: '肩部',
    elbow_flex: '肘部',
    wrist_flex: '手腕俯仰',
    wrist_roll: '手腕旋轉',
  };
  const wave = stopped ? null : waveErrorPresentation(message, kind);
  if (wave)
    return (
      <div className="error motion-error" role="alert">
        <strong>{wave.title}</strong>
        <p>{wave.detail}</p>
        <details>
          <summary>查看原因</summary>
          <p>{message}</p>
        </details>
      </div>
    );
  if (!stopped)
    return (
      <div className="error" role="alert">
        {message}
      </div>
    );
  return (
    <div className="error motion-error" role="alert">
      <strong>{joints[stopped[1]]}移動未達預期，已停止動作</strong>
      {stopped[2].startsWith('no progress') ? (
        <p>連續四次位置修正仍沒有足夠進展。可能與負載、摩擦或阻擋有關，目前尚未確認原因。</p>
      ) : (
        <p>位置仍有誤差，但下一次修正會超出這一步已檢查的範圍，需要重新評估路徑與補償。這不代表馬達已碰到實體行程極限。</p>
      )}
      <p>
        請先確認手臂周圍沒有卡住。確認後可按「恢復」；恢復只解除停止，不會自動重試抓取。
      </p>
      <details>
        <summary>技術細節</summary>
        <code>{message}</code>
      </details>
    </div>
  );
}
function Logo({ small = false }: { small?: boolean }) {
  return (
    <span className={'logo ' + (small ? 'small' : '')} aria-label="LeRobot">
      <b>L</b>
      <b>e</b>
      <b>R</b>
      <b>o</b>
      <b>b</b>
      <b>o</b>
      <b>t</b>
    </span>
  );
}
function Thumb({ item, scene }: { item: Item; scene: Scene }) {
  const f = scene.frames[item.frame_index];
  return (
    <div className="objectthumb">
      <img src={f?.url} alt={item.name} />
      <div
        className="box"
        style={{
          left: `${item.bbox.x * 100}%`,
          top: `${item.bbox.y * 100}%`,
          width: `${item.bbox.w * 100}%`,
          height: `${item.bbox.h * 100}%`,
        }}
      />
    </div>
  );
}
export default function Home() {
  const [q, setQ] = useState(''),
    [status, setStatus] = useState<Status | null>(null),
    [scene, setScene] = useState<Scene | null>(null),
    [job, setJob] = useState<Job | null>(null),
    [lastWave, setLastWave] = useState<Job | null>(null),
    [resultMode, setResultMode] = useState(false),
    [error, setErrorState] = useState(''),
    [connectionIssue, setConnectionIssue] = useState(''),
    [tab, setTab] = useState('all');
  const [cameraOpen, setCameraOpen] = useState(false),
    [cameraFrame, setCameraFrame] = useState<Frame | null>(null),
    [cameraPoint, setCameraPoint] = useState<{ x: number; y: number } | null>(
      null,
    ),
    [settings, setSettings] = useState(false),
    [voiceOpen, setVoiceOpen] = useState(false),
    [listening, setListening] = useState(false),
    [voiceError, setVoiceError] = useState('');
  const [reference, setReference] = useState<{
      id: string;
      url: string;
      name: string;
    } | null>(null),
    [sending, setSending] = useState(false),
    [manualWaveSending, setManualWaveSending] = useState(false),
    [stopSending, setStopSending] = useState(false),
    [holdVerified, setHoldVerified] = useState<boolean | null>(null),
    [stopAcknowledgedJob, setStopAcknowledgedJob] = useState(''),
    [now, setNow] = useState(0);
  const token = useRef(''),
    fileRef = useRef<HTMLInputElement>(null),
    queryRef = useRef<HTMLInputElement>(null),
    activeJob = useRef(''),
    polledError = useRef<{ jobId: string; message: string } | null>(null),
    pollRevision = useRef(0),
    manualWavePending = useRef(''),
    stopPending = useRef(false),
    lastFinished = useRef(''),
    recognition = useRef<Recognition | null>(null),
    currentFrame = useRef<Frame | null>(null),
    searchRef = useRef<(value: string) => Promise<Job | null>>(
      async () => null,
    );
  const setError = useCallback((value: string | ((previous: string) => string)) => {
    // Transport, stop and camera errors are not owned by a polled job.
    polledError.current = null;
    setErrorState(value);
  }, []);
  const { busy, showProgress, stopped } = progressPresentation(
    sending,
    job,
    status?.job,
    Boolean(connectionIssue),
    stopAcknowledgedJob,
  );
  const request = useCallback(
    async <T,>(path: string, body?: unknown): Promise<T> =>
      controllerRequest<T>(path, body, token),
    [],
  );
  const accept = useCallback((j: Job) => {
    pollRevision.current += 1;
    setStopAcknowledgedJob('');
    setConnectionIssue('');
    setJob(j);
    setLastWave((previous) => retainWaveReceipt(previous, j));
    activeJob.current = j.id;
    sessionStorage.setItem('lerobot-active-job', j.id);
  }, []);
  useEffect(() => {
    let alive = true;
    activeJob.current ||= sessionStorage.getItem('lerobot-active-job') || '';
    let pending = false;
    async function poll() {
      if (pending) return;
      pending = true;
      const revision = pollRevision.current;
      const isCurrent = () => alive && pollRevision.current === revision;
      try {
        const snapshot = await readControllerPoll<Job, Status>(
          request,
          activeJob.current,
          isCurrent,
        );
        if (snapshot.kind === 'stale') return;
        const { status: s, current, activeId, missingId } = snapshot;
        activeJob.current = activeId;
        if (activeId) sessionStorage.setItem('lerobot-active-job', activeId);
        else sessionStorage.removeItem('lerobot-active-job');
        setStatus(s);
        setConnectionIssue('');
        setNow(Date.now());
        if (missingId) {
          setStopAcknowledgedJob('');
          setHoldVerified(null);
          // A 404 means the job record is gone, not that motion completed or stopped.
          // Keep any partial scene so a controller restart does not erase what was seen.
          setJob((previous) => retireMissingJob(previous, missingId));
          setError(
            '先前工作紀錄已無法取得，已保留搜尋結果。點選物件或動作時會重新觀察。',
          );
        }
        if (current && current.id === activeJob.current) {
          const errorOwner = polledError.current;
          if (errorOwner && (errorOwner.jobId !== current.id || current.status === 'completed')) {
            polledError.current = null;
            setErrorState((previous) => reconcilePolledJobError(previous, errorOwner, current));
          }
          setJob(current);
          setLastWave((previous) => retainWaveReceipt(previous, current));
          if (
            ['grasp-attempt', 'grasp-release', 'object-program', 'wave', 'inspect', 'command'].includes(
              current.kind,
            ) || current.action_suggestion
          )
            setResultMode(true);
          if (current.status === 'running' && current.result) {
            setScene(current.result);
            setResultMode(true);
          }
          if (
            current.status === 'running' &&
            !['look', 'wave'].includes(current.kind)
          )
            setResultMode(true);
          if (
            current.status !== 'running' &&
            lastFinished.current !== current.id
          ) {
            lastFinished.current = current.id;
            if (current.result) {
              setScene(current.result);
              setQ(current.result.query);
              setResultMode(true);
              if (current.kind === 'point') {
                const idx = current.result.selected_index;
                if (idx >= 0) setQ(current.result.objects[idx].name);
                setCameraOpen(false);
              }
            } else if (current.action_suggestion) {
              setScene(null);
              setQ(current.action_suggestion.query);
            }
            if (current.status === 'failed') {
              polledError.current = { jobId: current.id, message: current.message };
              setErrorState(current.message);
            }
          }
        }
        if (s.robot.stopped && s.robot.error && current?.status !== 'running') {
          polledError.current = current?.status === 'failed' && current.message === s.robot.error
            ? { jobId: current.id, message: s.robot.error } : null;
          setErrorState(s.robot.error);
        }
      } catch {
        if (isCurrent()) {
          setStatus(null);
          setConnectionIssue(
            '暫時無法更新進度，正在重新連線。已保留目前的畫面與結果。',
          );
        }
      } finally {
        pending = false;
      }
    }
    void poll();
    const i = setInterval(poll, 650);
    return () => {
      alive = false;
      clearInterval(i);
    };
  }, [request, setError]);
  useEffect(() => {
    if (!cameraOpen || cameraPoint) return;
    let alive = true;
    let pending = false;
    async function update() {
      if (pending) return;
      pending = true;
      try {
        const f = await request<Frame>('/api/camera/snapshot', {});
        const im = new Image();
        im.onload = () => {
          if (alive) {
            currentFrame.current = f;
            setCameraFrame(f);
          }
        };
        im.src = f.url;
      } catch (e) {
        if (alive) {
          setError((e as Error).message);
          currentFrame.current = null;
          setCameraFrame(null);
        }
      } finally {
        pending = false;
      }
    }
    void update();
    const t = setInterval(update, 500);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [cameraOpen, cameraPoint, request, setError]);
  useEffect(
    () => () => {
      recognition.current?.abort();
    },
    [],
  );
  async function search(mode = 'search', value = q): Promise<Job | null> {
    if (busy) return null;
    if (mode === 'search' && !value.trim() && !reference) {
      queryRef.current?.focus();
      return null;
    }
    setError('');
    setSending(true);
    setResultMode(true);
    setTab('all');
    setScene(null);
    try {
      const accepted = await request<Job>('/api/search', {
        query: value,
        mode,
        reference_id: reference?.id ?? null,
      });
      accept(accepted);
      return accepted;
    } catch (e) {
      setError((e as Error).message);
      return null;
    } finally {
      setSending(false);
    }
  }
  async function executeCommand() {
    const suggestion = searchActionSuggestion(job);
    if (!suggestion || !suggestionPresentation(suggestion, status, busy, connectionIssue).enabled) return;
    setError('');
    setSending(true);
    setResultMode(true);
    setTab('all');
    try {
      const submission = commandRequest(suggestion.instruction);
      accept(await request<Job>(submission.path, submission.body));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  }
  useEffect(() => {
    searchRef.current = async (value) => {
      setQ(value);
      return search('search', value);
    };
  });
  useEffect(() => {
    const context = (document as ModelDocument).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    Promise.resolve(
      context.registerTool(
        {
          name: 'search_with_robot',
          description:
            'Search for an object or an action. Action requests return suggestions; this tool never executes suggested actions. Object searches may use the wrist camera and move the arm to observe; they do not grasp objects.',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', minLength: 1, maxLength: 500 },
            },
            required: ['query'],
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: true },
          execute: async (input: unknown) => {
            const query =
              input && typeof input === 'object' && 'query' in input
                ? input.query
                : null;
            if (
              typeof query !== 'string' ||
              !query.trim() ||
              query.length > 500
            )
              throw new Error('A query of 1–500 characters is required');
            const accepted = await searchRef.current(query);
            if (!accepted)
              throw new Error(
                'Search was not accepted; inspect the current robot status.',
              );
            return {
              status: accepted.status,
              job_id: accepted.id,
              query,
            };
          },
        },
        { signal: lifecycle.signal },
      ),
    ).catch(() => {});
    return () => lifecycle.abort();
  }, []);
  async function openCamera() {
    if (busy) return;
    setCameraPoint(null);
    setCameraOpen(true);
    setError('');
    // Reuse a fresh view instead of resetting a carefully chosen wrist angle.
    // Stopped/owned grippers can also preview images without moving.
    if (!cameraPreviewNeedsLook(status)) return;
    try {
      accept(await request<Job>('/api/look', {}));
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function pointAt(e: React.MouseEvent<HTMLButtonElement>) {
    const im = e.currentTarget.querySelector('img');
    if (!im) return;
    if (
      busy ||
      !currentFrame.current ||
      !status?.camera.connected ||
      Date.now() / 1000 - currentFrame.current.captured_at > 2
    )
      return;
    const r = im.getBoundingClientRect();
    const point = {
      x: e.detail === 0 ? 0.5 : (e.clientX - r.left) / r.width,
      y: e.detail === 0 ? 0.5 : (e.clientY - r.top) / r.height,
    };
    setCameraPoint(point);
    setSending(true);
    try {
      accept(
        await request<Job>('/api/search', {
          query: '',
          mode: 'point',
          frame_id: im.currentSrc.split('/').pop(),
          ...point,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
      setCameraPoint(null);
    } finally {
      setSending(false);
    }
  }
  async function upload(file: File) {
    setError('');
    if (file.size > 8_000_000) {
      setError('請選擇 8 MB 以下的照片');
      return;
    }
    setSending(true);
    try {
      const data = await new Promise<string>((resolve, reject) => {
        const r = new FileReader();
        r.onload = () => resolve(r.result as string);
        r.onerror = reject;
        r.readAsDataURL(file);
      });
      const result = await request<UploadResult>('/api/upload', {
        image: data,
      });
      setReference({ id: result.id, url: result.url || data, name: file.name });
      queryRef.current?.focus();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  }
  async function objectAction(
    kind: string,
    item: Item | null = null,
    proposed?: Action,
  ) {
    const definition = proposed || {
      kind,
      label: kind === 'wave' ? '揮揮手' : '查看物件',
      reason: '',
      unavailable_reason: '',
    };
    if (
      !actionPresentation(definition, item, status, busy, connectionIssue)
        .enabled
    )
      return;
    setError('');
    setSending(true);
    setResultMode(true);
    try {
      const submission = actionRequest(kind, item?.name, proposed?.target);
      accept(await request<Job>(submission.path, submission.body));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  }
  async function stop() {
    const targetId = activeJob.current || job?.id || status?.job?.id || '';
    pollRevision.current += 1;
    stopPending.current = true;
    setStopSending(true);
    setError('');
    try {
      const result = await request<StopResult>('/api/stop', {});
      pollRevision.current += 1;
      setStopAcknowledgedJob(targetId);
      setHoldVerified(result.held === true);
      if (!result.held)
        setError(
          '已停止發送動作，但無法確認所有關節保持姿勢：' +
            Object.keys(result.failures || {}).join('、'),
        );
      setJob((j) =>
        j
          ? {
              ...j,
              status: 'cancelled',
              message: result.held
                ? '已停止，保持目前姿勢'
                : '已停止發送動作，請確認手臂狀態',
            }
          : j,
      );
      setStatus((s) =>
        s ? { ...s, robot: { ...s.robot, stopped: true } } : s,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      stopPending.current = false;
      setStopSending(false);
    }
  }
  async function forceWave() {
    if (manualWavePending.current || stopPending.current ||
        !manualWaveAvailable(job, status, sending, connectionIssue, stopped) ||
        activeJob.current !== job?.id) return;
    const requestedId = job.id;
    manualWavePending.current = requestedId;
    const revision = ++pollRevision.current;
    setManualWaveSending(true);
    setError('');
    const isCurrent = () => revision === pollRevision.current && activeJob.current === requestedId;
    try {
      const response = await request<Job>(`/api/jobs/${encodeURIComponent(requestedId)}/force-wave`, {});
      if (!isCurrent()) return;
      pollRevision.current += 1;
      setJob((current) => acknowledgeManualWave(current, response, requestedId));
    } catch (e) {
      if (isCurrent()) setError((e as Error).message);
    } finally {
      if (manualWavePending.current === requestedId) {
        manualWavePending.current = '';
        setManualWaveSending(false);
      }
    }
  }
  async function resume() {
    const revision = ++pollRevision.current;
    try {
      await request('/api/resume', {});
      if (revision !== pollRevision.current) return;
      pollRevision.current += 1;
      setHoldVerified(null);
      setError('');
      setStatus((s) => s ? { ...s, robot: { ...s.robot, stopped: false, error: '' } } : s);
    } catch (e) {
      if (revision === pollRevision.current) setError((e as Error).message);
    }
  }
  function voice() {
    setVoiceOpen(true);
    setVoiceError('');
    const SR =
      (window as RecognitionWindow).SpeechRecognition ||
      (window as RecognitionWindow).webkitSpeechRecognition;
    if (!SR) {
      setVoiceError(
        '此瀏覽器未提供語音辨識。請用 Safari 或 Chrome 開啟，或直接輸入指令。',
      );
      return;
    }
    recognition.current?.abort();
    const r = new SR();
    recognition.current = r;
    r.lang = 'zh-TW';
    r.interimResults = true;
    r.onstart = () => setListening(true);
    r.onend = () => setListening(false);
    r.onerror = (e) => {
      setListening(false);
      setVoiceError(
        e.error === 'not-allowed'
          ? '請在瀏覽器允許麥克風存取。'
          : '暫時無法辨識語音，請再試一次。',
      );
    };
    r.onresult = (e) => {
      const text = Array.from(e.results)
        .map((v) => v[0].transcript)
        .join('');
      setQ(text);
      if (e.results[e.results.length - 1].isFinal) {
        r.stop();
        setVoiceOpen(false);
        void searchRef.current(text);
      }
    };
    try {
      r.start();
    } catch {
      setListening(false);
      setVoiceError('麥克風暫時無法啟動，請再試一次。');
    }
  }
  const [motionPlan, setMotionPlan] = useState<MotionGPTPlan | null>(null);
  const [motionReceipt, setMotionReceipt] = useState<MotionGPTReceipt | null>(null);
  const [preferredArm, setPreferredArm] = useState<'auto' | 'right' | 'left'>('auto');
  const [er2Result, setEr2Result] = useState<ER2SpatialResult | null>(null);
  const [er2Busy, setEr2Busy] = useState(false);
  const [geminiKeyInput, setGeminiKeyInput] = useState('');
  const [brainConfigSaving, setBrainConfigSaving] = useState(false);
  const [brainConfigSavedNote, setBrainConfigSavedNote] = useState('');
  const lastSyncedMotionJobId = useRef('');
  const lastSyncedMotionReceiptJobId = useRef('');
  const lastSyncedSuggestionJobId = useRef('');
  const dismissedSuggestionJobId = useRef('');

  const activeEngine: 'astra' | 'argon' =
    status?.brain?.engine ||
    (status?.brain?.model?.startsWith('gemini') ? 'argon' : 'astra');
  const activeThinkingLevel: ThinkingLevel =
    (status?.brain?.thinking_level as ThinkingLevel | undefined) || 'low';

  async function updateBrainConfig(payload: {
    engine?: 'astra' | 'argon' | 'dreamlabs' | 'smolvla';
    thinking_level?: ThinkingLevel;
    gemini_api_key?: string;
    dreamlabs_api_key?: string;
    dreamlabs_model?: string;
    smolvla_model?: string;
  }) {
    setBrainConfigSaving(true);
    setBrainConfigSavedNote('');
    setError('');
    try {
      const res = await request<{
        brain: Status['brain'];
        er2: Status['er2'];
      }>('/api/brain/config', payload);
      setStatus((prev) =>
        prev ? { ...prev, brain: res.brain, er2: res.er2 } : prev,
      );
      if (payload.gemini_api_key !== undefined) {
        setGeminiKeyInput('');
        setBrainConfigSavedNote(
          payload.gemini_api_key.trim()
            ? '已儲存 GEMINI_API_KEY 至 .env.local 並立即生效'
            : '已清除 GEMINI_API_KEY（自動切換為 Astra / Codex 備援）',
        );
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBrainConfigSaving(false);
    }
  }

  const previewMotion = useCallback(
    async (queryValue: string, armChoice: 'auto' | 'right' | 'left' = preferredArm) => {
      if (job?.id) {
        lastSyncedMotionJobId.current = job.id;
        lastSyncedMotionReceiptJobId.current = job.id;
        dismissedSuggestionJobId.current = job.id;
      }
      setMotionReceipt(null);
      try {
        const plan = await request<MotionGPTPlan>('/api/motiongpt/plan', {
          query: queryValue.trim() || '揮揮手',
          preferred_arm: armChoice,
        });
        setMotionPlan(plan);
        setMotionReceipt(null);
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [job?.id, preferredArm, request, setError],
  );

  async function openMotionChip(queryValue: string) {
    setQ(queryValue);
    setResultMode(true);
    setTab('all');
    setScene(null);
    setError('');
    await previewMotion(queryValue, preferredArm);
  }

  async function changePreferredArm(armChoice: 'auto' | 'right' | 'left') {
    setPreferredArm(armChoice);
    if (motionPlan || resultMode) {
      await previewMotion(motionPlan?.query || q || '揮揮手', armChoice);
    }
  }

  async function executeMotionPlan() {
    if (!motionPlan || busy) return;
    setError('');
    setSending(true);
    setResultMode(true);
    try {
      const accepted = await request<Job>('/api/motiongpt/execute', {
        query: motionPlan.query || q || '揮揮手',
        preferred_arm: preferredArm,
      });
      lastSyncedMotionJobId.current = accepted.id;
      dismissedSuggestionJobId.current = accepted.id;
      accept(accepted);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  }

  async function runER2Spatial(thinkingLevel: ThinkingLevel) {
    if (busy || er2Busy) return;
    setEr2Busy(true);
    setError('');
    try {
      const res = await request<ER2SpatialResult>('/api/er2/spatial', {
        query: q,
        thinking_level: thinkingLevel,
      });
      setEr2Result(res);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setEr2Busy(false);
    }
  }

  useEffect(() => {
    if (job?.id && (job.kind === 'motiongpt' || job.kind === 'search' || job.kind === 'command')) {
      if (job.motiongpt_plan && lastSyncedMotionJobId.current !== job.id) {
        lastSyncedMotionJobId.current = job.id;
        setMotionPlan(job.motiongpt_plan);
      }
      if (job.motiongpt_result && lastSyncedMotionReceiptJobId.current !== job.id) {
        lastSyncedMotionReceiptJobId.current = job.id;
        setMotionReceipt(job.motiongpt_result);
      }
    }
  }, [job]);

  useEffect(() => {
    const instruction = job?.action_suggestion?.instruction?.trim();
    if (!resultMode || !job?.id || !instruction || lastSyncedSuggestionJobId.current === job.id) return;
    lastSyncedSuggestionJobId.current = job.id;
    if (/(揮手|揮揮手|打招呼|伸懶腰|伸展|拳|跳|舞|律動|鞠躬|敬禮|歡呼|點頭|招手|左手|右手|單手|wave|stretch|box|punch|dance|bow|cheer|nod|beckon)/i.test(instruction)) {
      void previewMotion(instruction, preferredArm);
    }
  }, [resultMode, job?.id, job?.action_suggestion?.instruction, preferredArm, previewMotion]);

  const selected =
    scene && scene.selected_index >= 0
      ? scene.objects[scene.selected_index]
      : null;
  const spatialObject = scene?.spatial?.objects.find(
    (o) => o.name === selected?.name && o.status === 'estimated',
  );
  const progress = jobPresentation(job, status?.grasp?.state);
  const outcome = busy
    ? null
    : graspOutcome(job, stopped || !!status?.robot.stopped);
  const currentWaved = busy
    ? null
    : waveOutcome(job, stopped || !!status?.robot.stopped);
  const waved = currentWaved || waveOutcome(lastWave);
  const historicalWave = !currentWaved && Boolean(waved);
  const commanded = commandOutcome(job, stopped);
  const rawSuggested = searchActionSuggestion(job);
  const suggested =
    rawSuggested &&
    dismissedSuggestionJobId.current !== job?.id &&
    (!motionPlan || rawSuggested.query.trim() === motionPlan.query.trim())
      ? rawSuggested
      : null;
  const reobserved = observationEvidence(job, status?.job || null);
  const notice = sceneNotice(scene, job);
  const waveAction = {
    kind: 'wave',
    label: '揮揮手',
    reason: '讓 LeRobot 跟你打聲招呼',
    unavailable_reason: '',
  };
  const wavePresentation = actionPresentation(
    waveAction,
    null,
    status,
    busy,
    connectionIssue,
  );
  const inspectAction = {
    kind: 'inspect',
    label: '查看物件',
    reason: '重新拍照，查看眼前物件的詳細資訊',
    unavailable_reason: '',
  };
  const holdingObject = ['holding', 'holding_unverified'].includes(
    status?.grasp?.state || '',
  );
  const motionCanExecute = Boolean(
    status &&
      !busy &&
      !connectionIssue &&
      !status.robot.stopped &&
      !status.robot.moving &&
      !holdingObject &&
      !status.grasp?.low_power_owner,
  );
  const homeCanExecute = Boolean(
    status &&
      !sending &&
      !connectionIssue &&
      !status.robot.stopped &&
      !status.robot.moving &&
      !holdingObject &&
      !status.grasp?.low_power_owner,
  );
  const motionExecuteReason = connectionIssue
    ? '請先確認本機控制器連線'
    : !status
      ? '等待本機控制器連線'
      : status.robot.stopped
        ? '手臂已停止，按「恢復」後即可執行實體動作'
        : holdingObject || status.grasp?.low_power_owner
          ? '夾爪正在保持物件，請先完成目前操作'
          : busy
            ? '請先等目前工作結束'
            : '通過 3D 桌面淨空檢查，可在實體手臂執行';
  const statusLabel = connectionIssue
    ? '連線待確認'
    : !status
      ? '本機控制器未連線'
      : status.robot.stopped
        ? '已停止'
        : !status.camera.connected
          ? '腕部相機未連線'
          : status.robot.moving
            ? '正在動動手'
            : busy
              ? job?.stage === 'reobserving' ? '正在換角度觀察'
                : job?.kind === 'grasp-attempt'
                  ? '正在嘗試抓取'
                  : job?.kind === 'object-program' ? '正在操作物件'
                    : job?.kind === 'command' ? '正在執行指令'
                    : job?.kind === 'motiongpt' ? '正在執行單手動作'
                    : job?.kind === 'home' ? '正在歸回原位'
                    : job?.stage === 'understanding' ? '正在理解' : '正在看看'
              : holdingObject
                ? '夾爪正在保持'
                : '準備好了';
  async function goHome() {
    if (!homeCanExecute) return;
    setError('');
    setSending(true);
    setResultMode(true);
    setTab('all');
    setMotionPlan(null);
    setMotionReceipt(null);
    setScene(null);
    try {
      const accepted = await request<Job>('/api/robot/home', {});
      lastSyncedMotionJobId.current = accepted.id;
      lastSyncedMotionReceiptJobId.current = accepted.id;
      dismissedSuggestionJobId.current = accepted.id;
      accept(accepted);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  }
  const searchBar = (
    <form
      className="searchbox"
      onSubmit={(e) => {
        e.preventDefault();
        void search();
      }}
    >
      <button
        type="button"
        className="icon"
        aria-label="上傳參考照片"
        title="用照片找物件"
        disabled={busy}
        onClick={() => fileRef.current?.click()}
      >
        <Plus />
      </button>
      {reference && (
        <button
          type="button"
          className="reference"
          aria-label="移除參考照片"
          onClick={() => setReference(null)}
        >
          <img src={reference.url} alt="參考照片" />
          <X size={11} />
        </button>
      )}
      <input
        ref={queryRef}
        aria-label="搜尋物件或輸入動作指令"
        placeholder={
          reference ? '找找照片裡的東西' : '世界就在手邊。想找什麼，或做什麼單手動作？'
        }
        maxLength={500}
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      {q && (
        <button
          className="icon clear"
          type="button"
          aria-label="清除搜尋"
          onClick={() => {
            setQ('');
            queryRef.current?.focus();
          }}
        >
          <X size={19} />
        </button>
      )}
      <button
        type="button"
        className="icon mic"
        aria-label="語音搜尋"
        title="用說的"
        onClick={voice}
        disabled={busy}
      >
        <Mic size={22} />
      </button>
      <button
        type="button"
        className="icon lens"
        aria-label="開啟腕部相機"
        title="用相機搜尋"
        onClick={openCamera}
        disabled={busy}
      >
        <Camera size={23} />
      </button>
      {resultMode && (
        <button className="icon mic" aria-label="搜尋" disabled={busy}>
          <Search size={22} />
        </button>
      )}
    </form>
  );
  return (
    <div className={'site ' + (resultMode ? 'results-page' : '')}>
      <header className="top">
        {resultMode ? (
          <button
            className="brand-home"
            aria-label="回到首頁"
            onClick={() => {
              setResultMode(false);
              pollRevision.current += 1;
              activeJob.current = '';
              sessionStorage.removeItem('lerobot-active-job');
              setScene(null);
              setError('');
            }}
          >
            <Logo small />
          </button>
        ) : (
          <span>LeRobot 101 Argon</span>
        )}
        {resultMode && <div className="header-search">{searchBar}</div>}
        <div className="top-right">
          <button
            className="connection"
            style={{ whiteSpace: 'nowrap', color: '#137333' }}
            disabled={!homeCanExecute}
            title="將機械臂歸回正前方預設原位 (shoulder_pan=2030)"
            onClick={() => void goHome()}
          >
            {showProgress && job?.kind === 'home' ? (
              <LoaderCircle size={18} className="spin" />
            ) : (
              <RotateCcw size={17} />
            )}
            歸回原位
          </button>
          <button
            className="connection"
            style={{ whiteSpace: 'nowrap', color: '#1a73e8' }}
            disabled={!wavePresentation.enabled}
            title={wavePresentation.reason}
            onClick={() => objectAction('wave')}
          >
            {showProgress && job?.kind === 'wave' ? (
              <LoaderCircle size={18} className="spin" />
            ) : (
              <Hand size={18} />
            )}
            揮揮手
          </button>
          <button
            className="connection"
            style={{ whiteSpace: 'nowrap', color: '#5f6368' }}
            title="點擊切換 Argon / Astra / Dream Labs / SmolVLA 推論引擎與 Thinking Level"
            onClick={() => setSettings(true)}
          >
            <Sparkles size={15} />
            {activeEngine === 'argon'
              ? status?.brain?.api_key_configured
                ? 'Argon ER 2'
                : 'Argon (Astra 備援)'
              : activeEngine === 'dreamlabs'
                ? 'Dream Labs (Dream-1)'
                : activeEngine === 'smolvla'
                  ? 'SmolVLA (SO-101)'
                  : 'Astra'}{' '}
            · Thinking: {activeThinkingLevel}
          </button>
          <button className="connection" onClick={() => setSettings(true)}>
            <i
              className={
                status?.camera.connected && !status.robot.stopped
                  ? 'online'
                  : ''
              }
            />
            {statusLabel}
          </button>
          <button
            className={'icon stop ' + (status?.robot.stopped ? 'stopped' : '')}
            onClick={stop}
            aria-label="立即停止機械臂"
            title="立即停止，保持姿勢"
          >
            <CircleStop size={22} />
          </button>
          <button
            className="avatar"
            aria-label="連線資訊"
            onClick={() => setSettings(true)}
          >
            {activeEngine === 'argon' ? 'Ar' : 'As'}
          </button>
        </div>
      </header>
      <input
        hidden
        ref={fileRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/heic,image/heif"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void upload(f);
          e.target.value = '';
        }}
      />
      {!resultMode ? (
        <main className="home">
          <div className="brand">
            <Logo />
            <span className="astra">
              {activeEngine === 'argon'
                ? 'Argon'
                : activeEngine === 'dreamlabs'
                  ? 'Dream-1'
                  : activeEngine === 'smolvla'
                    ? 'SmolVLA'
                    : 'Astra'}
            </span>
          </div>
          {searchBar}
          <div className="searchbuttons" style={{ flexWrap: 'wrap' }}>
            <button disabled={busy} onClick={() => search()}>
              {sending ? <LoaderCircle className="spin" size={16} /> : null}
              LeRobot 搜尋
            </button>
            <button disabled={busy} onClick={() => search('lucky', '')}>
              <Sparkles size={16} />
              好手氣
            </button>
          </div>
          <div className="home-motion-strip" aria-label="單手自然動作與推論引擎快速選擇">
            <div className="home-motion-label" style={{ marginBottom: 8, flexWrap: 'wrap', gap: 10 }}>
              <span>推論與動作模型：</span>
              <div className="arm-selector" role="group" aria-label="視覺推論引擎選擇">
                <button
                  type="button"
                  className={activeEngine === 'argon' ? 'active' : ''}
                  disabled={brainConfigSaving}
                  onClick={() => void updateBrainConfig({ engine: 'argon' })}
                >
                  Argon (Gemini ER 2)
                </button>
                <button
                  type="button"
                  className={activeEngine === 'astra' ? 'active' : ''}
                  disabled={brainConfigSaving}
                  onClick={() => void updateBrainConfig({ engine: 'astra' })}
                >
                  Astra (Codex)
                </button>
                <button
                  type="button"
                  className={activeEngine === 'dreamlabs' ? 'active' : ''}
                  disabled={brainConfigSaving}
                  onClick={() => void updateBrainConfig({ engine: 'dreamlabs' })}
                >
                  Dream Labs (Dream-1)
                </button>
                <button
                  type="button"
                  className={activeEngine === 'smolvla' ? 'active' : ''}
                  disabled={brainConfigSaving}
                  onClick={() => void updateBrainConfig({ engine: 'smolvla' })}
                >
                  SmolVLA (SO-101)
                </button>
              </div>
              <span style={{ marginLeft: 4 }}>Thinking：</span>
              <div className="arm-selector" role="group" aria-label="思考深度選擇">
                {(['none', 'low', 'medium', 'high'] as const).map((lvl) => (
                  <button
                    key={lvl}
                    type="button"
                    className={activeThinkingLevel === lvl ? 'active' : ''}
                    disabled={brainConfigSaving}
                    onClick={() => void updateBrainConfig({ thinking_level: lvl })}
                  >
                    {lvl}
                  </button>
                ))}
              </div>
            </div>
            <div className="home-motion-label">
              <span>人手單臂重映射 (AIRI MotionGPT → SO-101)：</span>
              <div className="arm-selector" role="group" aria-label="預設單手選擇">
                {(['auto', 'right', 'left'] as const).map((arm) => (
                  <button
                    key={arm}
                    type="button"
                    className={preferredArm === arm ? 'active' : ''}
                    onClick={() => void changePreferredArm(arm)}
                  >
                    {arm === 'auto' ? '自動慣用手' : arm === 'right' ? '右手' : '左手'}
                  </button>
                ))}
              </div>
            </div>
            <div className="motion-Preset-row">
              {QUICK_MOTION_CHIPS.map((chip) => (
                <button
                  key={chip.id}
                  type="button"
                  className="motion-chip"
                  disabled={busy}
                  onClick={() => void openMotionChip(chip.query)}
                >
                  {chip.label}
                </button>
              ))}
              <button
                type="button"
                className="motion-chip"
                style={{ borderColor: 'rgba(99, 102, 241, 0.5)', background: 'rgba(79, 70, 229, 0.12)', fontWeight: 600 }}
                disabled={busy}
                onClick={() => {
                  setResultMode(true);
                  setTab('r2s');
                }}
              >
                ✨ MIRA × Steer-a-Robot 3D 數位孿生沙盒
              </button>
            </div>
          </div>
          <p className="offered">
            讓好奇心，動手做。<span>由 Gemini 4 Argon · Gemini Robotics ER 2 · MIRA Real-to-Sim · MotionGPT 驅動</span>
          </p>
          {reference && (
            <p className="reference-hint">
              <ImagePlus size={15} /> 已加入照片，按搜尋看看它在不在眼前。
            </p>
          )}
          {connectionIssue && (
            <output className="notice">{connectionIssue}</output>
          )}
          {error && <ErrorNotice message={error} kind={job?.kind} />}
        </main>
      ) : (
        <>
          <nav className="result-tabs" aria-label="搜尋結果類型">
            <button
              className={tab === 'all' ? 'active' : ''}
              onClick={() => setTab('all')}
            >
              全部
            </button>
            <button
              className={tab === 'r2s' ? 'active' : ''}
              onClick={() => setTab('r2s')}
            >
              Real-to-Sim 3D 沙盒
            </button>
            <button
              className={tab === 'motion' ? 'active' : ''}
              onClick={() => {
                setTab('motion');
                if (!motionPlan) void previewMotion(q || '揮揮手', preferredArm);
              }}
            >
              單手動作 (MotionGPT)
            </button>
            <button
              className={tab === 'er2' ? 'active' : ''}
              onClick={() => setTab('er2')}
            >
              ER 2 空間推理
            </button>
            <button
              className={tab === 'images' ? 'active' : ''}
              onClick={() => setTab('images')}
            >
              眼前影像
            </button>
            <span>搜尋範圍：手邊的世界與自然動作</span>
          </nav>
          <main className="results">
            {connectionIssue && (
              <output className="notice">{connectionIssue}</output>
            )}
            {error && <ErrorNotice message={error} kind={job?.kind} />}
            {showProgress && (
              <section className="progress" aria-live="polite">
                <div className="thinking-icon">
                  <Sparkles size={24} />
                </div>
                <div>
                  <h2>{progress.title}</h2>
                  <p>{inferenceProgressDescription(job, status, showProgress,
                    stopped || stopSending || Boolean(connectionIssue)) || progress.description}</p>
                  <div className="steps">
                    {progress.steps.map((step, index) => (
                      <span
                        key={step}
                        className={
                          index < progress.current
                            ? 'done'
                            : index === progress.current
                              ? 'current'
                              : ''
                        }
                      >
                        {index > 0 && <span aria-hidden="true"> → </span>}
                        {job?.kind === 'wave' && step === '揮手' ? (
                          <ManualWaveStep
                            enabled={manualWaveAvailable(job, status, sending || manualWaveSending,
                              connectionIssue, stopped || stopSending)}
                            onRequest={forceWave}
                          />
                        ) : step}
                      </span>
                    ))}
                  </div>
                </div>
                <button className="cancel" onClick={stop}>
                  停止
                </button>
              </section>
            )}
            {reobserved && <ObservationEvidenceCard evidence={reobserved} />}
            {(stopped || (!busy && job?.status === 'cancelled')) && (
              <div className="notice">
                {holdVerified === true
                  ? '已停止，保持目前姿勢。'
                  : '已停止發送動作。'}
              </div>
            )}
            {(tab === 'r2s' || tab === 'all') && (
              <RealToSimCard
                request={request}
                busy={busy}
                thinkingLevel={activeThinkingLevel}
                onJobStarted={(newJob) => {
                  if (newJob && typeof newJob === 'object' && 'id' in newJob) {
                    accept(newJob as Job);
                  }
                }}
              />
            )}
            {(tab === 'all' || tab === 'motion') && motionPlan && (
              <MotionRetargetCard
                plan={motionPlan}
                receipt={motionReceipt}
                preferredArm={preferredArm}
                onChangeArm={(arm) => void changePreferredArm(arm)}
                onSelectQuery={(val) => {
                  setQ(val);
                  void previewMotion(val, preferredArm);
                }}
                onExecute={() => void executeMotionPlan()}
                canExecute={motionCanExecute}
                executeReason={motionExecuteReason}
                executing={busy && job?.kind === 'motiongpt'}
              />
            )}
            {(tab === 'er2' || (tab === 'all' && er2Result)) && (
              <ER2SpatialCard
                available={Boolean(status?.er2?.available)}
                cameraConnected={Boolean(status?.camera.connected)}
                busy={busy || er2Busy}
                query={q}
                thinkingLevel={activeThinkingLevel}
                onChangeThinkingLevel={(lvl) => void updateBrainConfig({ thinking_level: lvl })}
                onOpenSettings={() => setSettings(true)}
                onRunER2={(lvl) => void runER2Spatial(lvl)}
                result={er2Result}
                onPickObject={(name) => {
                  setQ(name);
                  void objectAction('pick', {
                    name,
                    description: '',
                    confidence: 0.9,
                    bbox: { x: 0.4, y: 0.4, w: 0.2, h: 0.2 },
                    frame_index: 0,
                    matches_query: true,
                    safe_to_touch: true,
                    actions: [],
                  });
                }}
              />
            )}
            {suggested && (
              <ActionSuggestionCard suggestion={suggested} status={status} busy={busy}
                connectionIssue={connectionIssue} onExecute={executeCommand} />
            )}
            <CommandReceipt job={job} stopped={stopped}
              suppressReason={!!error && error === commanded?.reason} />
            {job?.kind === 'home' && !busy && job.status === 'completed' && (
              <output className="performed">
                <Check size={18} />
                <div>
                  <strong>已歸回正前方預設原位 (shoulder_pan = 2030)</strong>
                  <p>
                    機械臂已平滑歸回正前方標準原位，可開始下一個單手動作或物件搜尋。
                  </p>
                </div>
              </output>
            )}
            {outcome && (
              <section
                aria-live="polite"
                className={outcome.confirmed ? 'performed' : 'notice'}
              >
                {outcome.confirmed ? <Check size={18} /> : <Hand size={18} />}
                <div>
                  <strong>{outcome.title}</strong>
                  <p>{outcome.description}</p>
                  {outcome.frame?.url && (
                    <img
                      src={outcome.frame.url}
                      alt="抓取後的腕部實際影像"
                      style={{
                        display: 'block',
                        maxWidth: 'min(100%, 320px)',
                        borderRadius: 12,
                        marginTop: 12,
                      }}
                    />
                  )}
                </div>
              </section>
            )}
            {waved && (
              <output className={waved.confirmed ? 'performed' : 'notice'}>
                {waved.confirmed ? <Check size={18} /> : <Hand size={18} />}
                <div>
                  <strong>
                    {historicalWave
                      ? `上次揮手記錄：${waved.title}`
                      : waved.title}
                  </strong>
                  <p>{waved.description}</p>
                </div>
              </output>
            )}
            {holdingObject && !outcome && !showProgress && (
              <div className="notice">
                夾爪可能仍有物件，請先確認並完成目前操作。
              </div>
            )}
            {scene && (
              <>
                <ResultStats scene={scene} />
                {scene.spatial && (
                  <div className="spatial-note">
                    <span>立體觀察</span>
                    {spatialObject
                      ? `距腕部鏡頭約 ${Math.round((spatialObject.distance_m || 0) * 100)} cm · 可見寬度約 ${((spatialObject.projected_width_m || 0) * 100).toFixed(1)} cm（${scene.spatial.stability === 'uncertain' ? '定位尚未穩定' : '實驗估計值'}）`
                      : '有效視差還不足，暫時無法可靠估計尺寸。'}
                  </div>
                )}
                {notice && <div className="notice">{notice}</div>}
                {scene.details && <ObjectDetails details={scene.details}
                  earlier={Boolean(reobserved && scene.details.frame?.id !== reobserved.frame.id)} />}
                {scene.performed &&
                  !scene.grasp &&
                  job?.kind !== 'grasp-attempt' &&
                  job?.kind !== 'object-program' &&
                  job?.kind !== 'wave' &&
                  job?.status === 'completed' &&
                  job.result?.id === scene.id && (
                    <div className="performed">
                      <Check size={18} />
                      {scene.performed}
                    </div>
                  )}
                {tab === 'images' ? (
                  <div className="frames-grid">
                    {scene.frames.map((f, i) => (
                      <figure key={f.id}>
                        <img src={f.url} alt={`腕部觀察視角 ${i + 1}`} />
                        <figcaption>
                          視角 {i + 1} ·{' '}
                          {new Date(f.captured_at * 1000).toLocaleTimeString(
                            'zh-TW',
                          )}
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                ) : (
                  <div className="results-layout">
                    <section className="result-list">
                      {selected && (
                        <article className="suggestion">
                          <div className="sponsored">
                            <Sparkles size={15} />
                            <strong>推薦動作</strong>
                            <span>· 由好奇心贊助</span>
                          </div>
                          <h2>{selected.name}，可以這樣玩。</h2>
                          <p>{selected.description}</p>
                          <ResultActionButtons
                            item={selected}
                            status={status}
                            busy={busy}
                            connectionIssue={connectionIssue}
                            onAction={objectAction}
                          />
                          {selected.actions.some(
                            (a) => !['pick', 'place', 'stack', 'inspect', 'wave'].includes(a.kind),
                          ) && (
                            <p className="calibration-note">
                              {selected.actions.some(
                                (a) => a.kind === 'pick',
                              ) && status?.capabilities?.pick
                                ? '物件操作為實驗功能；尚未開放的動作仍在實作與驗證中。'
                                : '未開放的動作仍在實作與驗證中，目前可以搜尋和觀察物件。'}
                            </p>
                          )}
                        </article>
                      )}
                      {scene.empty ||
                      (scene.query &&
                        !scene.objects.some((o) => o.matches_query)) ? (
                        <div className="empty-result">
                          <Search size={31} />
                          <h2>
                            {scene.empty
                              ? '眼前暫時沒有合適的小物件。'
                              : '這次還沒找到你說的東西。'}
                          </h2>
                          <p>{scene.summary}</p>
                          <button
                            onClick={() => search('search', q)}
                            disabled={busy}
                          >
                            <RotateCcw size={16} />
                            再看一次
                          </button>
                        </div>
                      ) : null}
                      {scene.objects.map((o, oi) => (
                        <article className="organic" key={oi}>
                          <div className="source">
                            <span className="source-icon">
                              <Camera size={17} />
                            </span>
                            <div>
                              <span>你眼前的世界</span>
                              <small>lerobot / 桌面 / {o.name}</small>
                            </div>
                            <span className="match">
                              {o.matches_query ? '符合搜尋' : '也在附近'}
                            </span>
                          </div>
                          <h2>
                            <button
                              onClick={() => objectAction('inspect', o)}
                              disabled={
                                !actionPresentation(
                                  inspectAction,
                                  o,
                                  status,
                                  busy,
                                  connectionIssue,
                                ).enabled
                              }
                              title={
                                actionPresentation(
                                  inspectAction,
                                  o,
                                  status,
                                  busy,
                                  connectionIssue,
                                ).reason
                              }
                            >
                              {o.name}
                              <ArrowUpRight size={16} />
                            </button>
                          </h2>
                          <p>{o.description}</p>
                          <ResultActionButtons
                            item={o}
                            status={status}
                            busy={busy}
                            connectionIssue={connectionIssue}
                            onAction={objectAction}
                            inline
                          />
                        </article>
                      ))}
                      {scene.uncertainty && (
                        <p className="uncertainty">{scene.uncertainty}</p>
                      )}
                    </section>
                    {selected && (
                      <aside className="knowledge">
                        <Thumb item={selected} scene={scene} />
                        <div>
                          <h2>{selected.name}</h2>
                          <p>{selected.description}</p>
                          <dl>
                            <div>
                              <dt>找到的位置</dt>
                              <dd>腕部視角 {selected.frame_index + 1}</dd>
                            </div>
                            <div>
                              <dt>辨識把握</dt>
                              <dd>{Math.round(selected.confidence * 100)}%</dd>
                            </div>
                          </dl>
                          <button onClick={openCamera} disabled={busy}>
                            <Camera size={16} />
                            看看 LeRobot 看見什麼
                          </button>
                        </div>
                      </aside>
                    )}
                  </div>
                )}
              </>
            )}
            {!busy &&
              !scene &&
              !error &&
              !outcome &&
              !waved &&
              !commanded &&
              !suggested &&
              !holdingObject && (
                <div className="empty-result">
                  <Search size={30} />
                  <p>輸入一個物件，開始探索。</p>
                </div>
              )}
          </main>
        </>
      )}
      {status?.robot.stopped && (
        <div className="stop-banner">
          <CircleStop size={18} />
          已停止機械臂<button onClick={resume}>恢復</button>
        </div>
      )}
      <footer>
        <div>台灣 · 你的桌面</div>
        <div>
          <span>一隻手臂，一個全新的搜尋方式。</span>
          <button onClick={() => setSettings(true)}>
            只有腕部相機 · 本機連接
          </button>
        </div>
      </footer>
      <Dialog
        open={cameraOpen}
        onOpenChange={(v) => {
          setCameraOpen(v);
          if (!v) setCameraPoint(null);
        }}
      >
        <DialogContent className="camera-dialog">
          <DialogTitle>借你我的眼睛</DialogTitle>
          <DialogDescription>
            點選畫面裡的物件，讓 LeRobot 認識它。
          </DialogDescription>
          <div className="camera-stage">
            {cameraFrame ? (
              <>
                <button
                  className="camera-target"
                  onClick={pointAt}
                  aria-label="選取畫面中的物件；鍵盤選取中央"
                >
                  <img src={cameraFrame.url} alt="LeRobot 腕部即時視角" />
                </button>
                {cameraPoint && (
                  <span
                    className="point"
                    style={{
                      left: `${cameraPoint.x * 100}%`,
                      top: `${cameraPoint.y * 100}%`,
                    }}
                  />
                )}
                <span className="live">
                  <i />
                  {cameraPoint
                    ? '已選取'
                    : status?.camera.connected &&
                        cameraFrame &&
                        now / 1000 - cameraFrame.captured_at < 2
                      ? 'WRIST · LIVE'
                      : 'WRIST · 畫面中斷'}
                </span>
              </>
            ) : (
              <div className="camera-wait">
                <Camera size={34} />
                <p>{status?.camera.error || '正在開啟腕部相機…'}</p>
              </div>
            )}
          </div>
          <div className="camera-foot">
            <span>
              {busy
                ? job?.message || '正在辨識…'
                : '點一下，發現眼前的小東西。'}
            </span>
            {busy ? (
              <button className="cancel" onClick={stop}>
                停止
              </button>
            ) : (
              <button onClick={() => setCameraPoint(null)}>
                <RotateCcw size={16} />
                重新看
              </button>
            )}
          </div>
          {error && <ErrorNotice message={error} kind={job?.kind} />}
          <p className="calibration-note">
            辨識完成後會提供操作建議；選擇抓取後會重新觀察、定位並檢查路徑。
          </p>
        </DialogContent>
      </Dialog>
      <Dialog open={settings} onOpenChange={setSettings}>
        <DialogContent className="settings-dialog">
          <DialogTitle>LeRobot {activeEngine === 'argon' ? 'Argon' : 'Astra'}，準備好了嗎？</DialogTitle>
          <DialogDescription>你的手臂、單手自然動作重映射與具身空間推理。</DialogDescription>
          <dl className="connection-list">
            <div>
              <dt>推論引擎</dt>
              <dd>
                <div className="arm-selector" role="group" aria-label="設定視窗推論引擎選擇" style={{ flexWrap: 'wrap' }}>
                  <button
                    type="button"
                    className={activeEngine === 'argon' ? 'active' : ''}
                    disabled={brainConfigSaving}
                    onClick={() => void updateBrainConfig({ engine: 'argon' })}
                  >
                    Argon (Gemini ER 2)
                  </button>
                  <button
                    type="button"
                    className={activeEngine === 'astra' ? 'active' : ''}
                    disabled={brainConfigSaving}
                    onClick={() => void updateBrainConfig({ engine: 'astra' })}
                  >
                    Astra (Codex)
                  </button>
                  <button
                    type="button"
                    className={activeEngine === 'dreamlabs' ? 'active' : ''}
                    disabled={brainConfigSaving}
                    onClick={() => void updateBrainConfig({ engine: 'dreamlabs' })}
                  >
                    Dream Labs (Dream-1)
                  </button>
                  <button
                    type="button"
                    className={activeEngine === 'smolvla' ? 'active' : ''}
                    disabled={brainConfigSaving}
                    onClick={() => void updateBrainConfig({ engine: 'smolvla' })}
                  >
                    SmolVLA (SO-101)
                  </button>
                </div>
              </dd>
            </div>
            <div>
              <dt>Thinking Level</dt>
              <dd>
                <div className="arm-selector" role="group" aria-label="設定視窗思考深度選擇">
                  {(['none', 'low', 'medium', 'high'] as const).map((lvl) => (
                    <button
                      key={lvl}
                      type="button"
                      className={activeThinkingLevel === lvl ? 'active' : ''}
                      disabled={brainConfigSaving}
                      onClick={() => void updateBrainConfig({ thinking_level: lvl })}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>
              </dd>
            </div>
            <div>
              <dt>腕部相機</dt>
              <dd>
                {status?.camera.connected
                  ? '已連接 icspring camera'
                  : '等待連接'}
              </dd>
            </div>
            <div>
              <dt>Follower</dt>
              <dd>
                {status?.robot.connected ? '已讀取馬達校準' : '首次動作時連接'}
              </dd>
            </div>
            <div>
              <dt>空間核心</dt>
              <dd>Gemini 4 Argon · 3D 運動學校準 (2.8 mm)</dd>
            </div>
            <div>
              <dt>Robotics ER 2</dt>
              <dd>
                {status?.er2?.available
                  ? `已就緒 (${status.er2.model})`
                  : '待設定 GEMINI_API_KEY（目前自動使用 Astra 備援）'}
              </dd>
            </div>
            <div>
              <dt>單手自然動作</dt>
              <dd>AIRI MotionGPT (HumanML3D 22關節 → SO-101 5軸)</dd>
            </div>
            {status?.tactics && (
              <div>
                <dt>動作選擇</dt>
                <dd>{status.tactics.available ? `Jev · ${status.tactics.model}` : 'Jev 尚未啟用'}</dd>
              </div>
            )}
            <div>
              <dt>視覺推論後援</dt>
              <dd>
                {status?.brain.verified
                  ? `已實際完成視覺推論 (${status?.brain.transport || 'ready'})`
                  : status?.brain.available
                    ? `可用 (${status?.brain.transport === 'google-genai' ? 'Gemini ER 2 API' : '本機 Codex / Astra'})`
                    : '等待本機控制器'}
              </dd>
            </div>
          </dl>
          <div style={{ marginTop: 12, padding: '10px 12px', borderRadius: 10, background: '#f8f9fa', border: '1px solid #e8eaed' }}>
            <label htmlFor="gemini-api-key-input" style={{ display: 'block', fontSize: 12, fontWeight: 600, color: '#3c4043', marginBottom: 6 }}>
              Gemini API Key (<code>GEMINI_API_KEY</code> · 儲存至 <code>.env.local</code>)
            </label>
            <div style={{ display: 'flex', gap: 8 }}>
              <input
                id="gemini-api-key-input"
                type="password"
                placeholder={status?.brain?.api_key_configured ? '已設定 GEMINI_API_KEY（輸入新金鑰可覆寫）' : '貼上 AIza... 金鑰以啟用雲端 Gemini ER 2'}
                value={geminiKeyInput}
                onChange={(e) => setGeminiKeyInput(e.target.value)}
                style={{ flex: 1, padding: '6px 10px', borderRadius: 8, border: '1px solid #dadce0', fontSize: 13 }}
              />
              <button
                type="button"
                className="motion-chip"
                disabled={brainConfigSaving || !geminiKeyInput.trim()}
                onClick={() => void updateBrainConfig({ gemini_api_key: geminiKeyInput.trim(), engine: 'argon' })}
              >
                儲存並啟用 Argon
              </button>
            </div>
            {brainConfigSavedNote && (
              <p style={{ margin: '6px 0 0', fontSize: 12, color: '#137333' }}>{brainConfigSavedNote}</p>
            )}
          </div>
          <p className="settings-note">
            相機、三位置棋盤格運動學標定與 MotionGPT 單手動作重映射在本機執行；支援 Gemini Robotics ER 2 (<code>gemini-robotics-er-2-preview</code>) 與 Astra (<code>gpt-6-astra</code>) 雙引擎切換與 Thinking Level (<code>none</code> / <code>low</code> / <code>medium</code> / <code>high</code>) 調整。
          </p>
          {status?.camera.error && (
            <p className="error">{status.camera.error}</p>
          )}
          {status?.robot.error && <ErrorNotice message={status.robot.error} />}
        </DialogContent>
      </Dialog>
      <Dialog
        open={voiceOpen}
        onOpenChange={(v) => {
          setVoiceOpen(v);
          if (!v) recognition.current?.abort();
        }}
      >
        <DialogContent className="voice-dialog">
          <DialogTitle>
            {listening ? '我在聽。' : '用說的，也可以。'}
          </DialogTitle>
          <DialogDescription>
            說出想找的物件，或想讓 LeRobot 做的事。
          </DialogDescription>
          <button
            className={'voice-orb ' + (listening ? 'listening' : '')}
            onClick={voice}
            aria-label="開始語音辨識"
          >
            <Mic size={36} />
          </button>
          <p className="transcript">{q || '「幫我找桌上的瓶子」'}</p>
          {voiceError && <p className="error">{voiceError}</p>}
          <p className="calibration-note">語音由瀏覽器提供的辨識服務處理。</p>
        </DialogContent>
      </Dialog>
    </div>
  );
}
