'use client';

import { useState } from 'react';
import { RotateCcw, Sparkles, Play, Square, Shuffle, Check, Wrench } from 'lucide-react';

export type LoopEngineerDiagnosis = {
  plan_id?: string;
  query?: string;
  failure_mode?: string;
  summary?: string;
  applied_patch?: Record<string, unknown>;
  updated_tuning?: Record<string, unknown>;
};

export type LoopEngineerStatus = {
  config?: {
    auto_retry_enabled?: boolean;
    max_retries_per_grasp?: number;
    mode?: 'custom' | 'random';
    custom_query?: string;
    duration_hours?: number | null;
    cooldown_seconds?: number;
  };
  loop?: {
    running?: boolean;
    mode?: 'custom' | 'random';
    custom_query?: string;
    duration_hours?: number | null;
    started_at?: number | null;
    ends_at?: number | null;
    remaining_seconds?: number | null;
    iterations?: number;
    successes?: number;
    failures?: number;
    current_query?: string | null;
    last_diagnosis?: LoopEngineerDiagnosis | null;
    stop_reason?: string | null;
  };
  last_attempt?: {
    query?: string;
    success?: boolean;
    time?: number;
    diagnosis?: LoopEngineerDiagnosis | null;
  } | null;
  tunings?: Record<
    string,
    {
      force_posture?: string;
      height_offset_mm?: number;
      depth_offset_mm?: number;
      lateral_offset_mm?: number;
      extra_squeeze_ticks?: number;
    }
  >;
};

type Props = {
  loopEngineer: LoopEngineerStatus | null | undefined;
  currentQuery: string;
  busy: boolean;
  onUpdateConfig: (patch: Record<string, unknown>) => Promise<void>;
  onManualRetry: (query?: string, diagnoseFirst?: boolean) => Promise<void>;
  onStartLoop: (params: {
    mode: 'custom' | 'random';
    custom_query: string;
    duration_hours: number | null;
  }) => Promise<void>;
  onStopLoop: () => Promise<void>;
  onResetTuning: () => Promise<void>;
};

const DURATION_OPTIONS: { label: string; value: number | null }[] = [
  { label: '1 小時', value: 1 },
  { label: '4 小時', value: 4 },
  { label: '12 小時', value: 12 },
  { label: '1 天 (24h)', value: 24 },
  { label: '3 天 (72h)', value: 72 },
  { label: '不限制（直到 Token 用完）', value: null },
];

function formatRemaining(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return '不限時間（直到 Token 用完）';
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h > 0) return `剩餘 ${h} 小時 ${m} 分`;
  return `剩餘 ${m} 分 ${s % 60} 秒`;
}

export function LoopEngineerCard({
  loopEngineer,
  currentQuery,
  busy,
  onUpdateConfig,
  onManualRetry,
  onStartLoop,
  onStopLoop,
  onResetTuning,
}: Props) {
  const cfg = loopEngineer?.config || {};
  const lp = loopEngineer?.loop || {};
  const lastAttempt = loopEngineer?.last_attempt;
  const tunings = loopEngineer?.tunings || {};
  const tuningEntries = Object.entries(tunings);

  const [customAction, setCustomAction] = useState<string>(
    cfg.custom_query || '往左找到 toothpick container，把它拿起來再放回去',
  );
  const [randomMode, setRandomMode] = useState<boolean>(cfg.mode === 'random');
  const [durationHours, setDurationHours] = useState<number | null>(
    cfg.duration_hours === undefined ? 1 : cfg.duration_hours,
  );
  const [submitting, setSubmitting] = useState(false);

  const autoRetry = cfg.auto_retry_enabled ?? true;
  const isLoopRunning = Boolean(lp.running);
  const lastFailed = Boolean(lastAttempt && lastAttempt.success === false);
  const retryQuery = (lastAttempt?.query || currentQuery || customAction).trim();
  const latestDiag = lp.last_diagnosis || lastAttempt?.diagnosis;

  return (
    <div className="mt-3 rounded-2xl border border-stone-200/90 bg-white/95 p-3.5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-xl bg-amber-50 text-amber-700 border border-amber-200/80">
            <Wrench className="h-3.5 w-3.5" />
          </span>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-stone-900">
                Agentic Loop Engineer 自動優化與重試
              </span>
              {isLoopRunning && (
                <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-700 border border-emerald-200">
                  <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  自動優化運作中（第 {lp.iterations || 1} 輪）
                </span>
              )}
            </div>
            <p className="text-[11px] text-stone-500">
              每次夾取皆先確認桌面環境與物件是否倒下，並記錄 Action Log + D435 錄影供自我診斷修正
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-xl border border-stone-200 bg-stone-50/80 px-2.5 py-1.5 text-xs text-stone-700 hover:bg-stone-100">
            <input
              type="checkbox"
              checked={autoRetry}
              onChange={async (e) => {
                await onUpdateConfig({ auto_retry_enabled: e.target.checked });
              }}
              className="h-3.5 w-3.5 rounded border-stone-300 text-stone-900 focus:ring-0"
            />
            <span>沒抓到自動重試</span>
          </label>

          <button
            type="button"
            disabled={busy || submitting || !retryQuery}
            onClick={async () => {
              setSubmitting(true);
              try {
                await onManualRetry(retryQuery, true);
              } finally {
                setSubmitting(false);
              }
            }}
            className={`inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-medium transition ${
              lastFailed
                ? 'bg-amber-600 text-white hover:bg-amber-700 shadow-xs'
                : 'border border-stone-200 bg-white text-stone-700 hover:bg-stone-50'
            } disabled:opacity-45`}
            title={retryQuery ? `重新檢測環境並重試：${retryQuery}` : '請先輸入或執行一個動作'}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>{lastFailed ? '手動重試（AI 診斷修正）' : '手動重試上次動作'}</span>
          </button>
        </div>
      </div>

      {/* Latest Self-Diagnosis Banner */}
      {latestDiag?.summary && (
        <div className="mt-2.5 rounded-xl border border-amber-200/80 bg-amber-50/70 px-3 py-2 text-xs text-amber-900 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-start gap-1.5">
            <Sparkles className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600" />
            <div>
              <span className="font-semibold">自我診斷修正：</span>
              <span>{latestDiag.summary}</span>
            </div>
          </div>
          {tuningEntries.length > 0 && (
            <button
              type="button"
              onClick={onResetTuning}
              className="text-[11px] font-medium text-amber-700 underline hover:text-amber-900"
            >
              重設微調參數
            </button>
          )}
        </div>
      )}

      {/* Loop Engineer Controls */}
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-12 items-center border-t border-stone-100 pt-2.5">
        <div className="sm:col-span-5 flex items-center gap-2">
          <label className="inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-xl border border-stone-200 bg-stone-50 px-2.5 py-1.5 text-xs text-stone-700 hover:bg-stone-100">
            <input
              type="checkbox"
              checked={randomMode}
              onChange={(e) => {
                const next = e.target.checked;
                setRandomMode(next);
                void onUpdateConfig({ mode: next ? 'random' : 'custom' });
              }}
              className="h-3.5 w-3.5 rounded border-stone-300 text-stone-900"
            />
            <Shuffle className="h-3 w-3 text-stone-500" />
            <span>隨機測試</span>
          </label>

          <input
            type="text"
            disabled={randomMode || isLoopRunning}
            value={randomMode ? '隨機抽選桌面物件夾取並自我修正' : customAction}
            onChange={(e) => setCustomAction(e.target.value)}
            onBlur={() => {
              if (!randomMode && customAction.trim()) {
                void onUpdateConfig({ custom_query: customAction.trim() });
              }
            }}
            placeholder="輸入自然語言動作（例如：往左找到牙籤罐拿起來再放回去）"
            className="w-full min-w-0 rounded-xl border border-stone-200 bg-stone-50/60 px-2.5 py-1.5 text-xs text-stone-800 placeholder:text-stone-400 disabled:opacity-60"
          />
        </div>

        <div className="sm:col-span-4 flex items-center gap-1.5">
          <span className="shrink-0 text-[11px] text-stone-500">運作時間：</span>
          <select
            disabled={isLoopRunning}
            value={durationHours === null ? 'unlimited' : String(durationHours)}
            onChange={(e) => {
              const val = e.target.value === 'unlimited' ? null : Number(e.target.value);
              setDurationHours(val);
              void onUpdateConfig({ duration_hours: val });
            }}
            className="w-full rounded-xl border border-stone-200 bg-stone-50/80 px-2.5 py-1.5 text-xs text-stone-800 disabled:opacity-60"
          >
            {DURATION_OPTIONS.map((opt) => (
              <option
                key={opt.value === null ? 'unlimited' : String(opt.value)}
                value={opt.value === null ? 'unlimited' : String(opt.value)}
              >
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div className="sm:col-span-3 flex items-center justify-end gap-2">
          {!isLoopRunning ? (
            <button
              type="button"
              disabled={submitting}
              onClick={async () => {
                setSubmitting(true);
                try {
                  await onStartLoop({
                    mode: randomMode ? 'random' : 'custom',
                    custom_query: (customAction || currentQuery || '夾起牙籤罐再放回原位').trim(),
                    duration_hours: durationHours,
                  });
                } finally {
                  setSubmitting(false);
                }
              }}
              className="inline-flex w-full sm:w-auto items-center justify-center gap-1.5 rounded-xl bg-stone-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-stone-800 disabled:opacity-45"
            >
              <Play className="h-3.5 w-3.5" />
              <span>啟動自動優化</span>
            </button>
          ) : (
            <button
              type="button"
              disabled={submitting}
              onClick={async () => {
                setSubmitting(true);
                try {
                  await onStopLoop();
                } finally {
                  setSubmitting(false);
                }
              }}
              className="inline-flex w-full sm:w-auto items-center justify-center gap-1.5 rounded-xl bg-rose-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-rose-700 disabled:opacity-45"
            >
              <Square className="h-3.5 w-3.5" />
              <span>停止自動優化</span>
            </button>
          )}
        </div>
      </div>

      {/* Live Loop Stats & Learned Offsets */}
      {(isLoopRunning || (lp.iterations || 0) > 0 || tuningEntries.length > 0) && (
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-stone-100 pt-2 text-[11px] text-stone-600">
          <div className="flex flex-wrap items-center gap-3">
            <span>
              累計測試：<strong className="text-stone-900">{lp.iterations || 0}</strong> 次
            </span>
            <span className="inline-flex items-center gap-1 text-emerald-700">
              <Check className="h-3 w-3" /> 成功：{lp.successes || 0}
            </span>
            <span className="text-amber-700">自動診斷修正：{lp.failures || 0}</span>
            {isLoopRunning && <span>{formatRemaining(lp.remaining_seconds)}</span>}
            {!isLoopRunning && lp.stop_reason && (
              <span className="text-stone-500">（{lp.stop_reason}）</span>
            )}
          </div>

          {tuningEntries.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              {tuningEntries.map(([key, t]) => (
                <span
                  key={key}
                  className="inline-flex items-center gap-1 rounded-lg bg-stone-100 px-2 py-0.5 text-[10px] text-stone-700"
                >
                  <strong>{key}</strong>:
                  {t.force_posture ? ` 姿態=${t.force_posture}` : ''}
                  {t.height_offset_mm ? ` 高${t.height_offset_mm > 0 ? '+' : ''}${t.height_offset_mm}mm` : ''}
                  {t.depth_offset_mm ? ` 深${t.depth_offset_mm > 0 ? '+' : ''}${t.depth_offset_mm}mm` : ''}
                  {t.extra_squeeze_ticks ? ` 夾力+${t.extra_squeeze_ticks}` : ''}
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
