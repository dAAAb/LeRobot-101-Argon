/** Render the actual calibration result card and exercise its request guard. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import ts from 'typescript';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

const source = readFileSync(new URL('../app/engineering/calibration/page.tsx', import.meta.url), 'utf8');
const ast = ts.createSourceFile('page.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
const names = ['calibrationResultPresentation', 'CalibrationSolveResult', 'configure'];
const declarations = names.map(name => {
  let found;
  function visit(node) {
    if (ts.isFunctionDeclaration(node) && node.name?.text === name) found = node;
    if (!found) ts.forEachChild(node, visit);
  }
  visit(ast);
  assert.ok(found, `Missing actual calibration-page function ${name}`);
  return found.getText(ast);
});
const script = ts.transpileModule(declarations.join('\n') + `\nmodule.exports={${names.join(',')}};`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
}).outputText;

function fixture(changes = {}, solveChanges = {}) {
  return { session_id: 'session-A', status: 'solved', collected_view_count: 16,
    last_solve: { solve_id: 'solve-A', status: 'validated', view_count: 16, blockers: [],
      validation: { heldout_max_translation_m: .0005, heldout_max_rotation_deg: .12 }, ...solveChanges }, ...changes };
}

function page(session) {
  const requests = [];
  const context = { module: { exports: {} }, exports: {}, require: createRequire(import.meta.url),
    session, solve: session?.last_solve, busy: false, operation: { current: false },
    mounted: { current: false }, setSending() {}, setError() {}, setNotice() {},
    request: async (...args) => { requests.push(args); }, refresh: async () => {} };
  vm.runInNewContext(script, context);
  const api = context.module.exports;
  return { ...api, requests, html: busy => renderToStaticMarkup(createElement(api.CalibrationSolveResult,
    { session, busy: !!busy, onActivate() {} })) };
}

test('a 12-photo result cannot appear current or be applied after four new photos', async () => {
  const session = fixture({ status: 'collecting' }, { view_count: 12 });
  const view = page(session);
  const html = view.html();
  assert.match(html, /上次計算用了 12 張；目前已收錄 16 張/);
  assert.match(html, /請重新計算/);
  assert.match(html, /上次 12 張的結果位置誤差：最高 0\.5 mm/);
  assert.doesNotMatch(html, /is-validated/);
  assert.match(html, /<button[^>]*disabled=""[^>]*>套用已驗證的校準/);
  await view.configure('activate');
  assert.equal(view.requests.length, 0);
});

test('only the successful result for all current photos can be applied', async () => {
  const session = fixture();
  const view = page(session);
  assert.match(view.html(), /這份結果使用 16 張照片/);
  assert.match(view.html(), /is-validated/);
  assert.doesNotMatch(view.html(), /disabled=/);
  assert.match(view.html(true), /disabled=""/);
  await view.configure('activate');
  assert.equal(view.requests.length, 1);
  assert.equal(view.requests[0][0], '/api/engineering/calibration/sessions/session-A/activate');
  assert.equal(view.requests[0][1].solve_id, 'solve-A');
});

test('matching counts alone cannot enable an unfinished, invalidated, or unidentified result', async () => {
  const sessions = [
    ...['collecting', 'solving', 'invalidated'].map(status => fixture({ status })),
    fixture({}, { view_count: undefined }), fixture({}, { view_count: NaN }),
    fixture({}, { view_count: '16' }), fixture({}, { view_count: 12 }),
    fixture({}, { solve_id: '' }), fixture({}, { status: 'rejected' }),
  ];
  for (const session of sessions) {
    const view = page(session);
    assert.equal(view.calibrationResultPresentation(session).canApply, false);
    if (session.last_solve.status === 'validated') assert.match(view.html(), /disabled=""/);
    await view.configure('activate');
    assert.equal(view.requests.length, 0);
  }
  assert.equal(page(null).html(), '');
});

function combinedFixture(changes = {}, solveChanges = {}) {
  return fixture({ collected_view_count: 6, ...changes }, {
    view_count: 22, target_view_count: 6, source_session_ids: ['session-B', 'session-A'],
    source_session_view_counts: { 'session-B': 16, 'session-A': 6 }, ...solveChanges,
  });
}

test('a combined solve uses the target batch snapshot while displaying its complete input', async () => {
  const session = combinedFixture();
  const view = page(session);
  assert.match(view.html(), /這份結果使用 2 批共 22 張照片，其中這批 6 張/);
  assert.match(view.html(), /is-validated/);
  assert.doesNotMatch(view.html(), /disabled=/);
  assert.match(view.html(true), /disabled=""/);
  await view.configure('activate');
  assert.equal(view.requests.length, 1);
  assert.equal(view.requests[0][0], '/api/engineering/calibration/sessions/session-A/activate');
  assert.equal(view.requests[0][1].solve_id, 'solve-A');
});

test('adding a target photo invalidates a combined solve even when its other source is unchanged', async () => {
  // Also protects against a delayed status refresh that still says "solved".
  for (const status of ['collecting', 'solved']) {
    const session = combinedFixture({ collected_view_count: 7, status });
    const view = page(session);
    assert.match(view.html(), /上次跨批次計算使用這批的 6 張；這批目前已收錄 7 張/);
    assert.doesNotMatch(view.html(), /is-validated/);
    assert.match(view.html(), /disabled=""/);
    await view.configure('activate');
    assert.equal(view.requests.length, 0);
  }
});

test('missing or malformed combined provenance never enables apply', async () => {
  const changes = [
    { target_view_count: undefined }, { source_session_ids: undefined },
    { source_session_view_counts: undefined }, { source_session_view_counts: null },
    { source_session_ids: ['session-A', 'session-A'] },
    { source_session_ids: ['session-B', 'session-C'] },
    { source_session_ids: 'session-A' }, { source_session_ids: [null, 'session-A'] },
    { source_session_view_counts: { 'session-B': 15, 'session-A': 6 } },
    { source_session_view_counts: { 'session-B': 16, 'session-A': 7 } },
    { source_session_view_counts: { 'session-B': 16, 'session-A': 6, extra: 1 } },
    { source_session_view_counts: { 'session-B': '16', 'session-A': 6 } },
    { target_view_count: '6' }, { target_view_count: NaN }, { view_count: undefined },
    // A total coincidentally equal to the current batch must not mask broken provenance.
    { view_count: 6, source_session_view_counts: undefined },
  ];
  for (const solveChanges of changes) {
    const session = combinedFixture({}, solveChanges);
    const view = page(session);
    assert.equal(view.calibrationResultPresentation(session).canApply, false);
    assert.match(view.html(), /跨批次結果的照片紀錄不完整/);
    assert.match(view.html(), /disabled=""/);
    await view.configure('activate');
    assert.equal(view.requests.length, 0);
  }
});
