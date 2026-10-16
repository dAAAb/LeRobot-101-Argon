'use client';

import React, { useState } from 'react';
import { Camera, Crosshair, LoaderCircle, Sparkles } from 'lucide-react';

export interface ER2Item {
  label: string;
  point_1000: [number, number];
  box_2d_1000: [number, number, number, number];
  point_norm: { x: number; y: number };
  bbox_norm: { x: number; y: number; w: number; h: number };
  confidence: number;
  description: string;
  safe_to_touch: boolean;
  matches_query: boolean;
  metric_3d?: {
    center_m: [number, number, number];
    radius_mm: number;
    height_mm: number;
  };
}

export interface ER2SpatialResult {
  model: string;
  thinking_level: string;
  elapsed_ms: number;
  image_size: [number, number];
  summary: string;
  uncertainty: string;
  items: ER2Item[];
  frame?: { id: string; url: string; captured_at: number };
}

export function ER2SpatialCard({
  available,
  cameraConnected,
  busy,
  query,
  onRunER2,
  result,
  onPickObject,
}: {
  available: boolean;
  cameraConnected: boolean;
  busy: boolean;
  query: string;
  onRunER2: (thinkingLevel: 'none' | 'low' | 'high') => void;
  result: ER2SpatialResult | null;
  onPickObject?: (name: string) => void;
}) {
  const [thinkingLevel, setThinkingLevel] = useState<'none' | 'low' | 'high'>('low');

  return (
    <section
      className="suggestion er2-card"
      aria-label="Gemini Robotics ER 2 具身空間推理"
      style={{ marginBottom: 24 }}
    >
      <div className="sponsored" style={{ justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Crosshair size={15} />
          <strong>Gemini Robotics ER 2 空間點選與 3D 定位</strong>
          <span>· gemini-robotics-er-2-preview (0–1000 座標 + 3D 桌面射線)</span>
        </div>
        <div className="arm-selector" role="group" aria-label="ER 2 思考深度">
          {(['none', 'low', 'high'] as const).map((lvl) => (
            <button
              key={lvl}
              type="button"
              className={thinkingLevel === lvl ? 'active' : ''}
              onClick={() => setThinkingLevel(lvl)}
            >
              Thinking: {lvl}
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 12, marginTop: 8 }}>
        <p style={{ margin: 0, fontSize: 14, color: '#5f6368' }}>
          以 Google Gemini Robotics ER 2 標定腕部畫面中的 2D 抓取點 <code>[y, x]</code>、邊界框 <code>[ymin, xmin, ymax, xmax]</code>，並結合三位置棋盤格運動學標定換算桌面 3D 座標。
        </p>
        <button
          type="button"
          className="er2-trigger-btn"
          disabled={!available || !cameraConnected || busy}
          onClick={() => onRunER2(thinkingLevel)}
          title={
            !available
              ? '請先設定 GEMINI_API_KEY 以啟用 Gemini Robotics ER 2'
              : !cameraConnected
                ? '請先連接腕部相機'
                : '執行 ER 2 空間點選與 3D 定位'
          }
        >
          {busy ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
          <span>執行 ER 2 空間標定{query.trim() ? `：「${query.trim()}」` : ''}</span>
        </button>
      </div>

      {!available && (
        <p className="calibration-note" style={{ marginTop: 8 }}>
          提示：在 <code>.env.local</code> 或環境變數設定 <code>GEMINI_API_KEY</code> 後，即可直接呼叫{' '}
          <code>gemini-robotics-er-2-preview</code> 進行即時空間點選。
        </p>
      )}

      {result && (
        <div className="er2-result-grid" style={{ marginTop: 14 }}>
          {result.frame?.url && (
            <div className="er2-image-wrap">
              <img src={result.frame.url} alt="ER 2 空間標定腕部影像" />
              {result.items.map((it, idx) => (
                <React.Fragment key={idx}>
                  <div
                    className="er2-box"
                    style={{
                      left: `${it.bbox_norm.x * 100}%`,
                      top: `${it.bbox_norm.y * 100}%`,
                      width: `${it.bbox_norm.w * 100}%`,
                      height: `${it.bbox_norm.h * 100}%`,
                    }}
                  >
                    <span className="er2-tag">{it.label}</span>
                  </div>
                  <span
                    className="er2-point"
                    style={{
                      left: `${it.point_norm.x * 100}%`,
                      top: `${it.point_norm.y * 100}%`,
                    }}
                    title={`${it.label} [${it.point_1000[0]}, ${it.point_1000[1]}]`}
                  />
                </React.Fragment>
              ))}
            </div>
          )}
          <div className="er2-items-list">
            <p style={{ margin: '0 0 8px', fontWeight: 600 }}>
              {result.summary} <small style={{ color: '#70757a', fontWeight: 400 }}>({result.elapsed_ms} ms)</small>
            </p>
            {result.items.map((it, idx) => (
              <div key={idx} className="er2-item-row">
                <div>
                  <strong>{it.label}</strong>{' '}
                  <small style={{ color: '#1a73e8' }}>
                    point=[{it.point_1000[0]}, {it.point_1000[1]}] · box=[{it.box_2d_1000.join(', ')}]
                  </small>
                  <p style={{ margin: '3px 0', fontSize: 13, color: '#5f6368' }}>{it.description}</p>
                  {it.metric_3d && (
                    <small style={{ color: '#137333', display: 'block' }}>
                      3D 桌面座標: X={(it.metric_3d.center_m[0] * 100).toFixed(1)}cm, Y=
                      {(it.metric_3d.center_m[1] * 100).toFixed(1)}cm, Z={(it.metric_3d.center_m[2] * 100).toFixed(1)}cm ·
                      半徑 {it.metric_3d.radius_mm}mm · 高 {it.metric_3d.height_mm}mm
                    </small>
                  )}
                </div>
                {onPickObject && it.safe_to_touch && (
                  <button type="button" className="motion-chip" onClick={() => onPickObject(it.label)}>
                    抓取此物件
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
