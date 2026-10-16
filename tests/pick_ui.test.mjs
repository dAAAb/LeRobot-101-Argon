/** Exercise the page's pure presentation logic without DOM, motors or network.
 * TypeScript extracts the actual helpers; no duplicate implementation is tested.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { createRequire } from 'node:module';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { Camera, Search, Hand, Check, ChevronRight, ArrowUpRight, Sparkles } from 'lucide-react';
import {
  ControllerRequestError,
  controllerRequest,
  readControllerPoll,
  progressPresentation,
} from '../lib/client-status.ts';

const source = readFileSync(
  new URL('../app/page.tsx', import.meta.url),
  'utf8',
);
const ast = ts.createSourceFile(
  'page.tsx',
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const names = [
  'actionPresentation',
  'jobPresentation',
  'inferenceProgressDescription',
  'graspOutcome',
  'observationEvidence',
  'ObservationEvidenceCard',
  'retireMissingJob',
  'reconcilePolledJobError',
  'poll',
  'waveOutcome',
  'retainWaveReceipt',
  'manualWaveAvailable',
  'acknowledgeManualWave',
  'ManualWaveStep',
  'forceWave',
  'stop',
  'resultActions',
  'sceneNotice',
  'ResultActionButtons',
  'observationSummary',
  'ResultStats',
  'cameraPreviewNeedsLook',
  'openCamera',
  'ObjectDetails',
  'actionRequest',
  'objectAction',
  'commandRequest',
  'commandPresentation',
  'searchActionSuggestion',
  'suggestionPresentation',
  'ActionSuggestionCard',
  'commandOutcome',
  'CommandReceipt',
  'executeCommand',
  'search',
  'ErrorNotice',
  'motorCommunicationFailure',
  'serialConnectionFailure',
  'motionCorrectionFailure',
  'wristEndpointFailure',
  'waveErrorPresentation',
  'resume',
];
const declarations = names.map((name) => {
  let node;
  function visit(candidate) {
    if (ts.isFunctionDeclaration(candidate) && candidate.name?.text === name)
      node = candidate;
    if (!node) ts.forEachChild(candidate, visit);
  }
  visit(ast);
  assert.ok(node, `Missing page helper ${name}`);
  return node.getText(ast);
});
const javascript = ts.transpileModule(
  declarations.join('\n') + `\nmodule.exports={${names.join(',')}};`,
  {
    compilerOptions: {
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
    },
  },
).outputText;
const context = {
  module: { exports: {} },
  exports: {},
  require: createRequire(import.meta.url),
  Camera,
  Search,
  Hand,
  Check,
  ChevronRight,
  ArrowUpRight,
  Sparkles,
  Link: ({children, ...props}) => createElement('a', props, children),
};
vm.runInNewContext(javascript, context);
const {
  ErrorNotice,
  actionPresentation,
  jobPresentation,
  inferenceProgressDescription,
  graspOutcome,
  observationEvidence,
  ObservationEvidenceCard,
  actionRequest,
  retireMissingJob,
  reconcilePolledJobError,
  waveOutcome,
  retainWaveReceipt,
  manualWaveAvailable,
  acknowledgeManualWave,
  ManualWaveStep,
  resultActions,
  sceneNotice,
  ResultActionButtons,
  observationSummary,
  ResultStats,
  ObjectDetails,
  cameraPreviewNeedsLook,
  openCamera,
  commandRequest,
  commandPresentation,
  searchActionSuggestion,
  suggestionPresentation,
  ActionSuggestionCard,
  commandOutcome,
  CommandReceipt,
} = context.module.exports;

const runningWave = (changes = {}) => ({
  id: 'wave/one', kind: 'wave', status: 'running', stage: 'thinking',
  message: '看一下手邊空間', result: null, events: [], ...changes,
});
const waveStatus = (job = runningWave()) => ({
  robot: { stopped: false }, camera: { connected: true }, job,
});

test('inference progress distinguishes connection, reasoning and validation using actual server elapsed time', () => {
  const job = runningWave();
  for (const [phase, text] of [
    ['starting', '正在準備 Astra…'],
    ['sending', '正在傳送畫面與指令給 Astra…'],
    ['connecting', '等待 Astra 回應…'],
    ['reconnecting', '連線中斷，正在重新連接 Astra…'],
    ['reasoning', 'Astra 正在看畫面…'],
    ['validating', '正在整理 Astra 的回應…'],
  ]) {
    const status = {
      ...waveStatus(job),
      brain: { active: { id: 'inference-1', schema: 'clearance', phase, elapsed: 17.9 } },
    };
    assert.equal(inferenceProgressDescription(job, status, true, false), `${text}（已等待 17 秒）`);
    status.brain.active.elapsed = 0;
    assert.equal(inferenceProgressDescription(job, status, true, false), `${text}（已等待 0 秒）`);
    for (const elapsed of [undefined, -1, NaN, Infinity, '17']) {
      status.brain.active.elapsed = elapsed;
      assert.equal(inferenceProgressDescription(job, status, true, false), text);
    }
  }
  for (const schema of ['SearchIntent', 'SkillPlan']) {
    const status = { ...waveStatus(job),
      brain: { active: { id: 'inference-2', schema, phase: 'reasoning', elapsed: 3 } } };
    assert.equal(inferenceProgressDescription(job, status, true, false),
      'Astra 正在理解指令…（已等待 3 秒）');
    if (schema === 'SearchIntent') {
      status.brain.active.phase = 'sending';
      assert.equal(inferenceProgressDescription(job, status, true, false),
        '正在傳送指令給 Astra…（已等待 3 秒）');
    }
  }
});

test('inference status cannot appear for absent, finished, stopped or mismatched jobs', () => {
  const job = runningWave();
  const status = {
    ...waveStatus(job),
    brain: { active: { id: 'inference-1', schema: 'clearance', phase: 'reasoning', elapsed: 8 } },
  };
  for (const [currentJob, currentStatus, visible, stopped] of [
    [job, status, false, false],
    [job, status, true, true],
    [null, status, true, false],
    [job, null, true, false],
    [job, { ...status, job: null }, true, false],
    [job, { ...status, robot: { stopped: true } }, true, false],
    [job, { ...status, job: runningWave({ id: 'another-job' }) }, true, false],
    [job, { ...status, brain: {} }, true, false],
    [job, { ...status, brain: { active: null } }, true, false],
    [job, { ...status, brain: { active: { ...status.brain.active, id: '' } } }, true, false],
    [job, { ...status, brain: { active: { ...status.brain.active, phase: 'unknown' } } }, true, false],
    ...['completed', 'failed', 'cancelled'].flatMap((state) => [
      [runningWave({ status: state }), status, true, false],
      [job, { ...status, job: runningWave({ status: state }) }, true, false],
    ]),
  ]) assert.equal(inferenceProgressDescription(currentJob, currentStatus, visible, stopped), null);
});

test('manual wave is limited to this active running wave before waving starts', () => {
  const current = runningWave();
  assert.equal(manualWaveAvailable(current, waveStatus(current), false, '', false), true);
  for (const [job, status, sending, issue, stopped] of [
    [null, waveStatus(), false, '', false],
    [current, null, false, '', false],
    [current, waveStatus(), true, '', false],
    [current, waveStatus(), false, 'offline', false],
    [current, waveStatus(), false, '', true],
    [current, { ...waveStatus(), robot: { stopped: true } }, false, '', false],
    [current, { ...waveStatus(), job: null }, false, '', false],
    [current, waveStatus(runningWave({ id: 'newer' })), false, '', false],
    [current, waveStatus(runningWave({ status: 'completed' })), false, '', false],
    [current, waveStatus(runningWave({ manual_wave_requested: true })), false, '', false],
    [current, waveStatus(runningWave({ manual_wave_phase: 'waving' })), false, '', false],
    ...[{ kind: 'command' }, { status: 'failed' }, { status: 'cancelled' },
      { manual_wave_requested: true }, { manual_wave_phase: 'waving' }].map((change) => {
      const job = runningWave(change);
      return [job, waveStatus(job), false, '', false];
    }),
  ]) assert.equal(manualWaveAvailable(job, status, sending, issue, stopped), false);
});

test('manual wave step is a keyboard-operable action with the requested native hover title', () => {
  const html = renderToStaticMarkup(createElement(ManualWaveStep, {enabled: true, onRequest() {}}));
  assert.match(html, /<button type="button" class="wave-step-link"/);
  assert.match(html, /title="強制執行『揮手』"/);
  assert.match(html, /aria-label="強制執行『揮手』"/);
  assert.match(html, />揮手<\/button>/);
  assert.doesNotMatch(html, /disabled|href=|role="link"/);
  const disabled = renderToStaticMarkup(createElement(ManualWaveStep, {enabled: false, onRequest() {}}));
  assert.match(disabled, /disabled=""/);
});

test('manual acknowledgement retains latest progress and cannot revive a finished or different job', () => {
  const response = runningWave({manual_wave_requested: true});
  const latest = runningWave({stage: 'moving', message: '最新一步', manual_wave_phase: 'waving'});
  const accepted = acknowledgeManualWave(latest, response, latest.id);
  assert.equal(accepted.manual_wave_requested, true);
  assert.equal(accepted.message, '最新一步');
  assert.equal(accepted.manual_wave_phase, 'waving');
  for (const current of [null, runningWave({id: 'next'}), runningWave({status: 'completed'}),
    runningWave({status: 'cancelled'}), runningWave({kind: 'command'})]) {
    assert.equal(acknowledgeManualWave(current, response, response.id), current);
  }
  for (const badResponse of [runningWave(), runningWave({id: 'wrong', manual_wave_requested: true}),
    runningWave({kind: 'command', manual_wave_requested: true})]) {
    assert.equal(acknowledgeManualWave(latest, badResponse, latest.id), latest);
  }
});

function prepareManualWaveHandler() {
  context.job = runningWave(); context.status = waveStatus(context.job);
  context.sending = false; context.connectionIssue = ''; context.stopped = false;
  context.activeJob = {current: context.job.id}; context.pollRevision = {current: 0};
  context.manualWavePending = {current: ''}; context.stopPending = {current: false};
  context.setManualWaveSending = (value) => { context.manualWaveSending = value; };
  context.setStopSending = (value) => { context.stopSending = value; };
  context.setError = (value) => { context.error = value; };
  context.setJob = (update) => { context.job = update(context.job); };
  context.setStatus = (update) => { context.status = update(context.status); };
  context.setStopAcknowledgedJob = (id) => { context.stopAcknowledgedJob = id; };
  context.setHoldVerified = () => {};
  context.accept = () => { throw new Error('Manual override must not accept a new job'); };
}

test('manual request targets one job once and reports acceptance only after the server responds', async () => {
  prepareManualWaveHandler();
  const calls = []; let resolve;
  context.request = (path, body) => {
    calls.push({path, body}); return new Promise((done) => {resolve = done;});
  };
  const pending = context.module.exports.forceWave();
  await context.module.exports.forceWave();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].path, '/api/jobs/wave%2Fone/force-wave');
  assert.equal(JSON.stringify(calls[0].body), '{}');
  assert.equal(context.job.manual_wave_requested, undefined);
  assert.doesNotMatch(jobPresentation(context.job).description, /略過障礙物/);
  resolve(runningWave({manual_wave_requested: true}));
  await pending;
  assert.equal(context.job.manual_wave_requested, true);
  assert.equal(jobPresentation(context.job).description, '依你的指示，略過障礙物判斷；保持目前位置後原地揮手。');
  assert.equal(waveOutcome(context.job), null);
  assert.equal(context.manualWaveSending, false);
  assert.equal(context.manualWavePending.current, '');
  await context.module.exports.forceWave();
  assert.equal(calls.length, 1, 'an acknowledged request is not submitted again');
});

test('a stop wins over a late manual acknowledgement without restoring motion or clearing stop', async () => {
  prepareManualWaveHandler();
  let resolveManual;
  context.request = async (path) => path === '/api/stop'
    ? {held: true, failures: {}}
    : new Promise((resolve) => {resolveManual = resolve;});
  const pending = context.module.exports.forceWave();
  await context.module.exports.stop();
  assert.equal(context.job.status, 'cancelled');
  resolveManual(runningWave({manual_wave_requested: true}));
  await pending;
  assert.equal(context.job.status, 'cancelled');
  assert.equal(context.job.manual_wave_requested, undefined);
  assert.equal(context.status.robot.stopped, true);
  assert.equal(context.stopAcknowledgedJob, 'wave/one');
});

test('late manual responses never replace a newer job or terminal polling result', async () => {
  for (const newer of [runningWave({id: 'wave/two'}), runningWave({status: 'completed', stage: 'done'})]) {
    prepareManualWaveHandler(); let resolve;
    context.request = () => new Promise((done) => {resolve = done;});
    const pending = context.module.exports.forceWave();
    context.job = newer; context.activeJob.current = newer.id;
    if (newer.id !== 'wave/one') context.pollRevision.current += 1;
    resolve(runningWave({manual_wave_requested: true})); await pending;
    assert.equal(context.job, newer);
  }
});

test('a rejected manual request does not claim acceptance and old errors cannot replace a newer state', async () => {
  prepareManualWaveHandler();
  context.request = async () => {throw new Error('已開始揮手');};
  await context.module.exports.forceWave();
  assert.equal(context.error, '已開始揮手');
  assert.equal(context.job.manual_wave_requested, undefined);
  assert.equal(context.manualWavePending.current, '');
  prepareManualWaveHandler(); let reject;
  context.request = () => new Promise((_resolve, fail) => {reject = fail;});
  const pending = context.module.exports.forceWave();
  context.pollRevision.current += 1; context.error = 'newer message';
  reject(new Error('old error')); await pending;
  assert.equal(context.error, 'newer message');
});

test('suggestion execution is trimmed, bounded and uses its own endpoint', () => {
  assert.deepEqual(JSON.parse(JSON.stringify(commandRequest(' 先抬高1公分，再揮揮手 '))), {
    path: '/api/command', body: {instruction: '先抬高1公分，再揮揮手'},
  });
  assert.throws(() => commandRequest('  '), /1–500/);
  assert.throws(() => commandRequest('動'.repeat(501)), /1–500/);
});

test('command access uses live capability and permits an explicit release while holding', () => {
  const ready = {camera: {connected: true}, robot: {stopped: false, moving: false},
    capabilities: {command: true}, grasp: {state: 'idle'}};
  assert.equal(commandPresentation('揮手', ready, false, '').enabled, true);
  assert.equal(commandPresentation('放開', {...ready, grasp: {state: 'holding', low_power_owner: true}}, false, '').enabled, true);
  for (const [state, busy, issue, instruction] of [
    [null, false, '', '揮手'], [ready, true, '', '揮手'], [ready, false, '連線中斷', '揮手'],
    [{...ready, capabilities: {}}, false, '', '揮手'],
    [{...ready, robot: {stopped: true}}, false, '', '揮手'],
    [{...ready, job: {status: 'running'}}, false, '', '揮手'],
    [{...ready, camera: {connected: false}}, false, '', '揮手'], [ready, false, '', ' '],
  ]) assert.equal(commandPresentation(instruction, state, busy, issue).enabled, false);
});

const actionSuggestion = (changes = {}) => ({
  kind: 'action', query: '先抬高1公分，再揮揮手', instruction: '先抬高1公分，再揮揮手',
  title: '抬高一點，揮揮手。', summary: '先抬高 1 公分，再跟你打招呼。', label: '就這麼做',
  availability: 'experimental', reason: '', ...changes,
});
const suggestionJob = (changes = {}) => ({
  id: 'suggestion-1', kind: 'search', status: 'completed', stage: 'done', result: null,
  action_suggestion: actionSuggestion(), ...changes,
});

test('homepage has only search and lucky buttons with no separate action entry or example', () => {
  let buttons;
  function visit(node) {
    if (ts.isJsxElement(node) && node.openingElement.attributes.properties.some((attr) =>
      ts.isJsxAttribute(attr) && attr.name.text === 'className'
      && ts.isStringLiteral(attr.initializer) && attr.initializer.text === 'searchbuttons')) buttons = node;
    ts.forEachChild(node, visit);
  }
  visit(ast);
  assert.ok(buttons);
  const children = buttons.children.filter((child) => ts.isJsxElement(child) || ts.isJsxSelfClosingElement(child));
  assert.equal(children.length, 2);
  assert.ok(children.every((child) => ts.isJsxElement(child) && child.openingElement.tagName.getText(ast) === 'button'));
  assert.match(buttons.getText(ast), /LeRobot 搜尋/);
  assert.match(buttons.getText(ast), /好手氣/);
  assert.doesNotMatch(source, /command-hint|先抬高1公分，再揮揮手|<CommandButton/);
});

test('only completed search suggestions are actionable results, not ordinary scenes or unfinished jobs', () => {
  assert.equal(searchActionSuggestion(suggestionJob()).instruction, actionSuggestion().instruction);
  for (const job of [null, {kind: 'search', status: 'completed', result: {objects: []}},
    suggestionJob({status: 'running'}), suggestionJob({status: 'cancelled'}), suggestionJob({kind: 'command'})]) {
    assert.equal(searchActionSuggestion(job), null);
  }
});

test('suggestion card shows intent without executing and keeps unavailable execution gates visible', () => {
  const ready = {camera: {connected: true}, robot: {stopped: false}, capabilities: {command: true}};
  let executions = 0;
  const render = (status) => renderToStaticMarkup(createElement(ActionSuggestionCard, {
    suggestion: actionSuggestion(), status, busy: false, connectionIssue: '', onExecute() { executions += 1; },
  }));
  const html = render(ready);
  assert.match(html, /搜尋推薦動作/); assert.match(html, /抬高一點，揮揮手。/);
  assert.match(html, /type="button"/); assert.match(html, /就這麼做/);
  assert.match(html, /點一下，我會先看看眼前/);
  assert.doesNotMatch(html, /disabled|動作指令已完成|class="performed"/);
  assert.equal(executions, 0);
  const stopped = render({...ready, robot: {stopped: true}});
  assert.match(stopped, /disabled/); assert.match(stopped, /手臂已停止/);
  const offline = render({...ready, camera: {connected: false}});
  assert.match(offline, /disabled/); assert.match(offline, /腕部相機的連線/);
});

test('unsupported and ambiguous suggestions show a friendly reason without an execution button', () => {
  for (const suggestion of [actionSuggestion({availability: 'unsupported', reason: '目前還不能疊物件。'}),
    actionSuggestion({kind: 'ambiguous', availability: 'clarify', instruction: '', reason: '你想找杯子，還是拿起杯子？'})]) {
    const html = renderToStaticMarkup(createElement(ActionSuggestionCard, {
      suggestion: {...suggestion, title: '<script>hi</script>'}, status: null, busy: false,
      connectionIssue: '', onExecute() { throw new Error('must never execute'); },
    }));
    assert.match(html, /&lt;script&gt;/);
    assert.ok(html.includes(suggestion.reason));
    assert.doesNotMatch(html, /<button|<script>|動作指令已完成/);
    assert.equal(suggestionPresentation(suggestion, null, false, '').enabled, false);
  }
});

test('search only requests a suggestion; clicking submits its instruction even if search text changed', async () => {
  const instruction = '先抬高1公分，再揮揮手';
  const requests = [];
  context.q = instruction; context.busy = false; context.reference = null; context.connectionIssue = '';
  context.job = null;
  context.status = {camera: {connected: true}, robot: {stopped: false}, capabilities: {command: true}};
  for (const name of ['setError', 'setSending', 'setResultMode', 'setTab', 'setScene', 'accept']) context[name] = () => {};
  context.request = async (path, body) => {requests.push({path, body}); return {id: 'accepted'};};
  await context.module.exports.search();
  assert.equal(requests[0].path, '/api/search');
  assert.equal(requests[0].body.query, instruction);
  assert.equal(requests[0].body.mode, 'search');
  await context.module.exports.executeCommand();
  assert.equal(requests.length, 1, 'there is no command entry without a returned suggestion');
  context.job = suggestionJob(); context.q = '現在只是打入另一個搜尋';
  await context.module.exports.executeCommand();
  assert.equal(requests[1].path, '/api/command');
  assert.equal(requests[1].body.instruction, instruction);
  context.busy = true;
  await context.module.exports.executeCommand();
  assert.equal(requests.length, 2);
});

test('stopped or camera-offline users can search for suggestions but cannot execute them', async () => {
  const requests = [];
  context.q = '揮揮手'; context.busy = false; context.reference = null; context.connectionIssue = '';
  context.job = suggestionJob();
  for (const name of ['setError', 'setSending', 'setResultMode', 'setTab', 'setScene', 'accept']) context[name] = () => {};
  context.request = async (path, body) => {requests.push({path, body}); return suggestionJob();};
  for (const state of [
    {camera: {connected: true}, robot: {stopped: true}, capabilities: {command: true}},
    {camera: {connected: false}, robot: {stopped: false}, capabilities: {command: true}},
  ]) {
    context.status = state;
    await context.module.exports.search();
    await context.module.exports.executeCommand();
  }
  assert.deepEqual(requests.map(({path}) => path), ['/api/search', '/api/search']);
});

const commandJob = (changes = {}) => ({
  id: 'command-1', kind: 'command', status: 'completed', stage: 'done', message: '', result: null,
  command_result: {
    completed: true, status: 'completed',
    plan: {status: 'ready', summary: '抬高後揮手', steps: [
      {skill: 'raise', label: '抬高 1 公分'}, {skill: 'wave', label: '揮揮手'},
    ]},
    steps: [{skill: 'raise', completed: true}, {skill: 'wave', completed: true}],
  }, ...changes,
});

test('command success requires all corresponding primitive postconditions, not a finished job alone', () => {
  assert.equal(commandOutcome(commandJob()).confirmed, true);
  for (const mutation of [
    (j) => {j.command_result.completed = false;},
    (j) => {j.command_result.steps[1].completed = false;},
    (j) => {j.command_result.steps[1].skill = 'look';},
    (j) => {j.command_result.steps.pop();},
    (j) => {j.command_result.plan.steps = []; j.command_result.steps = [];},
    (j) => {j.command_result.plan.status = 'unsupported';},
    (j) => {delete j.command_result;},
    (j) => {j.status = 'failed';},
    (j) => {j.command_result.status = 'blocked';},
  ]) {
    const current = commandJob(); mutation(current);
    assert.equal(commandOutcome(current).confirmed, false);
  }
  assert.equal(commandOutcome(commandJob(), true).confirmed, false);
  assert.equal(commandOutcome(commandJob({kind: 'search'})), null);
});

test('blocked command keeps verified partial progress and never presents overall success', () => {
  const current = commandJob({status: 'failed'});
  current.command_result.status = 'blocked'; current.command_result.completed = false;
  current.command_result.steps[1].completed = false;
  current.command_result.reason = '圓筒可能擋到揮手，我先停下來。';
  const html = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
  assert.match(html, /動作指令尚未完成/);
  assert.match(html, /已驗證/); assert.match(html, /未確認完成/);
  assert.match(html, /圓筒可能擋到揮手/);
  assert.doesNotMatch(html, /動作指令已完成|class="performed"|<h2 style="color:#137333"/);
});

test('a failed primitive without a returned receipt is not described as never started', () => {
  const current = commandJob({status: 'failed'});
  current.command_result.status = 'blocked'; current.command_result.completed = false;
  current.command_result.steps.pop();
  current.command_result.plan.steps.push({skill: 'observe', label: '最後看一看'});
  const html = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
  const rows = html.match(/<li\b.*?<\/li>/g);
  assert.match(rows[0], /已驗證/);
  assert.match(rows[1], /揮揮手.*未確認完成/);
  assert.doesNotMatch(rows[1], /尚未執行/);
  assert.match(rows[2], /最後看一看.*尚未執行/);
});

test('unsupported commands expose the reason and escape plan text', () => {
  const current = commandJob();
  current.command_result = {status: 'unsupported', completed: false, steps: [],
    plan: {status: 'unsupported', summary: '<script>unsupported</script>', steps: [],
      unsupported_reason: '目前不能把 A 放在 B 上面。'}};
  const html = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
  assert.match(html, /這個指令目前還做不到/); assert.match(html, /目前不能把 A 放在 B 上面/);
  assert.match(html, /&lt;script&gt;/); assert.doesNotMatch(html, /<script>|已驗證|動作指令已完成/);
});

test('command progress and cancellation retain accurate per-step labels', () => {
  for (const [stage, current] of [['planning', 0], ['moving', 1], ['observing', 2]]) {
    const progress = jobPresentation(commandJob({status: 'running', stage}));
    assert.equal(progress.current, current); assert.equal(progress.steps.join('/'), '規劃/執行/核對');
  }
  const current = commandJob({status: 'running'});
  current.command_result.completed = false; current.command_result.status = 'running';
  current.command_result.steps = [{skill: 'raise', completed: true}];
  const running = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
  assert.match(running, /正在逐步執行/); assert.match(running, /已驗證/); assert.match(running, /執行中/);
  const stopped = renderToStaticMarkup(createElement(CommandReceipt, {job: current, stopped: true}));
  assert.match(stopped, /動作指令已停止/); assert.match(stopped, /未確認完成/);
  assert.doesNotMatch(stopped, /執行中|動作指令已完成/);
});

test('stalled motion explains uncertainty and keeps the original evidence in closed details', () => {
  const message = '位置修正中止：elbow_flex: no progress after four corrections: movement=0.00, error reduction=0.00 ticks';
  const html = renderToStaticMarkup(createElement(ErrorNotice, {message}));
  assert.match(html, /肘部移動未達預期/);
  assert.match(html, /目前尚未確認原因/);
  assert.match(html, /不會自動重試抓取/);
  assert.match(html, /<details><summary>技術細節<\/summary><code>/);
  assert.ok(html.includes(message));
  assert.doesNotMatch(html, /<button|<details open/);
});

test('other errors keep their original meaning and are HTML escaped', () => {
  const html = renderToStaticMarkup(createElement(ErrorNotice, {message: '相機中斷 <script>'}));
  assert.match(html, /相機中斷 &lt;script&gt;/);
  assert.doesNotMatch(html, /摩擦|恢復|<details>/);
});

test('adopting a new job or its completion retires only an owned stale polling error', () => {
  const message = "未到達目標位置，已保持姿勢：{'wrist_roll': -9}";
  const owner = {jobId: 'old-wave', message};
  assert.equal(reconcilePolledJobError(message, owner, {id: 'new-wave', status: 'running'}), '');
  assert.equal(reconcilePolledJobError(message, owner, {id: 'new-wave', status: 'completed'}), '');
  assert.equal(reconcilePolledJobError(message, owner, {id: 'old-wave', status: 'completed'}), '');
  for (const current of [null, {id: 'old-wave', status: 'failed'}, {id: 'old-wave', status: 'cancelled'}])
    assert.equal(reconcilePolledJobError(message, owner, current), message);
  const complete = {id: 'new-wave', status: 'completed'};
  for (const unrelated of ['請求逾時，暫時無法確認是否已受理。請先查看工作狀態。', '已停止發送動作，但無法確認所有關節保持姿勢', '腕部相機畫面已過期']) {
    assert.equal(reconcilePolledJobError(unrelated, owner, complete), unrelated);
    assert.equal(reconcilePolledJobError(unrelated, null, complete), unrelated);
  }
  // An explicit operation can report the same text. Without job ownership it remains visible.
  assert.equal(reconcilePolledJobError(message, null, complete), message);
});

test('actual polling replaces an old wave failure when it adopts an external wave, preserving unrelated errors', async () => {
  let displayedError = '';
  let serverJob = {id: 'old-wave', kind: 'wave', status: 'failed', message: 'old wrist error'};
  Object.assign(context, {
    pending: false, alive: true, pollRevision: {current: 1}, activeJob: {current: 'old-wave'},
    polledError: {current: null}, lastFinished: {current: ''}, readControllerPoll,
    sessionStorage: {setItem() {}, removeItem() {}},
    setStatus() {}, setConnectionIssue() {}, setNow() {}, setStopAcknowledgedJob() {},
    setHoldVerified() {}, setJob() {}, setLastWave() {}, setResultMode() {}, setScene() {}, setQ() {},
    setCameraOpen() {},
    setErrorState(value) { displayedError = typeof value === 'function' ? value(displayedError) : value; },
    request: async (path) => {
      assert.equal(path, '/api/status');
      return {job: serverJob, robot: {stopped: false, error: ''}};
    },
  });
  await context.module.exports.poll();
  assert.equal(displayedError, 'old wrist error');
  assert.equal(context.polledError.current.jobId, 'old-wave');
  serverJob = {id: 'new-wave', kind: 'wave', status: 'running', message: 'waving'};
  await context.module.exports.poll();
  assert.equal(context.activeJob.current, 'new-wave');
  assert.equal(displayedError, '');
  serverJob = {...serverJob, status: 'completed', message: 'finished'};
  await context.module.exports.poll();
  assert.equal(displayedError, '');
  displayedError = '停止請求逾時，尚未確認'; context.polledError.current = null;
  await context.module.exports.poll();
  assert.equal(displayedError, '停止請求逾時，尚未確認');
  serverJob = {id: 'third-wave', kind: 'wave', status: 'running', message: 'waving'};
  await context.module.exports.poll();
  serverJob = {...serverJob, status: 'failed', message: 'current wrist error'};
  await context.module.exports.poll();
  assert.equal(displayedError, 'current wrist error');
});

test('a single wrist endpoint residual is explained in degrees without claiming a completed wave', () => {
  for (const [message, angle] of [
    ["未到達目標位置，已保持姿勢：{'wrist_roll': -5}", '0.44'],
    ['未到達目標位置，已保持姿勢：{"wrist_roll": 8}', '0.70'],
    ["未到達目標位置，已保持姿勢：{ 'wrist_roll' : 1.25e1 }", '1.10'],
  ]) {
    const html = renderToStaticMarkup(createElement(ErrorNotice, {message, kind: 'wave'}));
    const visible = html.split('<details>')[0];
    assert.match(visible, /手腕尚未到位，已先停止。/);
    assert.ok(visible.includes(`距離目標約 ${angle}°；這次揮手尚未完成。`));
    assert.doesNotMatch(visible, /wrist_roll|保持姿勢|擋到|障礙|極限|已完成|已揮手/);
    assert.match(html, /<details><summary>查看原因<\/summary><code>/);
    assert.doesNotMatch(html, /<details open|<button/);
    assert.ok(html.includes(message.replaceAll("'", '&#x27;').replaceAll('"', '&quot;')));
  }
  const general = renderToStaticMarkup(createElement(ErrorNotice, {
    message: "未到達目標位置，已保持姿勢：{'wrist_roll': -5}", kind: 'inspect',
  }));
  assert.match(general, /這次動作尚未完成。/);
  assert.doesNotMatch(general, /揮手/);
});

test('wrist endpoint formatting leaves other joints and malformed diagnostics unchanged', () => {
  for (const residual of [
    "{'elbow_flex': -5}", "{'wrist_roll': -5, 'elbow_flex': 2}",
    "{'wrist_roll': NaN}", "{'wrist_roll': Infinity}", "{'wrist_roll': 1e999}",
    "{'wrist_roll': true}", "{'wrist_roll': '-5'}", "{'wrist_roll': 05}",
    "{'wrist_roll\": -5}", "{'wrist_roll': -5} <script>",
  ]) {
    const message = `未到達目標位置，已保持姿勢：${residual}`;
    const html = renderToStaticMarkup(createElement(ErrorNotice, {message, kind: 'wave'}));
    assert.match(html, /未到達目標位置，已保持姿勢/);
    assert.doesNotMatch(html, /距離目標約|手腕尚未到位|查看原因|<script>/);
  }
});

test('serial zero-data failure explains both possible causes without claiming a known owner or verified hold', () => {
  const message = 'device reports readiness to read but returned no data (device disconnected or multiple access on port?)';
  const current = commandJob({status: 'failed'});
  current.command_result.status = 'blocked'; current.command_result.completed = false;
  current.command_result.steps = []; current.command_result.reason = message;
  for (const element of [createElement(ErrorNotice, {message, kind: 'wave'}),
    createElement(CommandReceipt, {job: current})]) {
    const html = renderToStaticMarkup(element);
    const visible = html.split('<details>')[0];
    assert.match(visible, /USB 連線中斷，或通訊埠被其他程式占用/);
    assert.match(visible, /確認 USB 連線/);
    assert.match(visible, /關閉其他正在連接手臂的控制程式後再試/);
    assert.doesNotMatch(visible, /device reports|LeRobot-101-OS|已確認|已保持|已停止動作|動作指令已完成/);
    assert.match(html, /<details><summary>查看原因<\/summary><code>/);
    assert.ok(html.includes(message));
    assert.equal(html.split(message).length - 1, 1);
    assert.doesNotMatch(html, /<details open|<button/);
  }
  for (const unrelated of ['camera device disconnected', 'multiple access on port?',
    'device reports readiness to read but returned no data']) {
    const html = renderToStaticMarkup(createElement(ErrorNotice, {message: unrelated}));
    assert.ok(html.includes(unrelated));
    assert.doesNotMatch(html, /USB|其他程式|<details>/);
  }
});

test('motor read and write failures show plain language with closed, escaped technical details', () => {
  for (const message of ['wrist_flex 馬達讀取失敗 (-6/0)',
    'gripper 馬達寫入失敗 (-1/0) <script>raw</script>', '馬達讀取失敗，無法核對旋轉範圍']) {
    const html = renderToStaticMarkup(createElement(ErrorNotice, {message, kind: 'wave'}));
    const visible = html.split('<details>')[0];
    assert.match(visible, /手臂通訊暫時中斷，已停止動作。/);
    assert.doesNotMatch(visible, /wrist_flex|gripper|-6\/-?0|-1\/0|讀取失敗|寫入失敗|保持|已抓住|可能擋到/);
    assert.match(html, /<details><summary>查看原因<\/summary><code>/);
    assert.doesNotMatch(html, /<details open|<script>|<button/);
  }
  const unrelated = renderToStaticMarkup(createElement(ErrorNotice, {message: '相機讀取失敗'}));
  assert.match(unrelated, /相機讀取失敗/);
  assert.doesNotMatch(unrelated, /手臂通訊|已停止動作/);
});

test('command failure receipt reuses the motor notice without exposing raw codes or claiming a hold', () => {
  const current = commandJob({status: 'failed'});
  current.command_result.status = 'blocked'; current.command_result.completed = false;
  current.command_result.steps = [];
  current.command_result.reason = 'wrist_flex 馬達讀取失敗 (-6/0)';
  const html = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
  const visible = html.split('<details>')[0];
  assert.match(visible, /動作指令尚未完成/);
  assert.match(visible, /手臂通訊暫時中斷，已停止動作。/);
  assert.doesNotMatch(visible, /wrist_flex|-6\/0|馬達讀取失敗|保持|已驗證|動作指令已完成/);
  assert.match(html, /<summary>查看原因<\/summary><code>wrist_flex 馬達讀取失敗 \(-6\/0\)/);
  assert.equal(html.split('wrist_flex').length - 1, 1);
});

test('command correction failures use the existing plain explanation and hide raw diagnostics', () => {
  for (const [message, expected] of [
    ['位置修正中止：elbow_flex: no progress after four corrections: movement=0.00, error reduction=0.00 ticks', '連續四次位置修正仍沒有足夠進展'],
    ['位置修正中止：wrist_flex: correction would cross safe joint limit', '這一步已檢查的範圍'],
  ]) {
    const current = commandJob({status: 'failed'});
    current.command_result.status = 'blocked'; current.command_result.completed = false;
    current.command_result.steps = []; current.command_result.reason = message;
    const html = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
    const visible = html.split('<details>')[0];
    assert.ok(visible.includes(expected));
    assert.match(visible, /動作指令尚未完成/);
    assert.doesNotMatch(visible, /elbow_flex|wrist_flex|no progress|correction would|ticks|動作指令已完成|已驗證/);
    assert.match(html, /<details><summary>技術細節<\/summary><code>/);
    assert.ok(html.includes(message));
    assert.equal(html.split(message).length - 1, 1);
    assert.doesNotMatch(html, /<details open/);
  }
  const current = commandJob({status: 'failed'});
  current.command_result.status = 'blocked'; current.command_result.completed = false;
  current.command_result.reason = '物件還沒有可靠支撐，先不放開';
  const ordinary = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
  assert.match(ordinary, /物件還沒有可靠支撐，先不放開/);
  assert.doesNotMatch(ordinary, /<details>|位置修正|手臂通訊/);
});

test('results suppress only the command reason already shown by the main error notice', () => {
  let receipt;
  function visit(node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(ast) === 'CommandReceipt') receipt = node;
    ts.forEachChild(node, visit);
  }
  visit(ast);
  assert.ok(receipt);
  const renderReceipt = ts.transpileModule(`receiptElement = (${receipt.getText(ast)});`, {
    compilerOptions: {target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX},
  }).outputText;
  const reason = '位置修正中止：elbow_flex: no progress after four corrections: movement=0.00 ticks';
  const current = commandJob({status: 'failed'});
  current.command_result.status = 'blocked'; current.command_result.completed = false;
  current.command_result.steps[1].completed = false;
  current.command_result.reason = reason;
  context.job = current; context.stopped = false; context.commanded = commandOutcome(current);
  for (const error of [reason, '', 'wrist_flex 馬達讀取失敗 (-6/0)']) {
    context.error = error;
    vm.runInNewContext(`{${renderReceipt}}`, context);
    const receiptHtml = renderToStaticMarkup(context.receiptElement);
    const pageHtml = (error ? renderToStaticMarkup(createElement(ErrorNotice, {message: error, kind: 'command'})) : '') + receiptHtml;
    assert.equal(receiptHtml.includes(reason), error !== reason);
    assert.equal(pageHtml.split(reason).length - 1, 1);
    if (error && error !== reason) assert.ok(pageHtml.includes(error));
    assert.match(receiptHtml, /動作指令尚未完成/);
    assert.match(receiptHtml, /抬高 1 公分.*已驗證/);
    assert.match(receiptHtml, /揮揮手.*未確認完成/);
    assert.doesNotMatch(receiptHtml, /動作指令已完成/);
  }
});

test('a cylinder wave refusal is short by default with the full reason collapsed', () => {
  const message = '揮腕空間尚未確認：雖未見人手，但圓筒靠近夾爪外側，單張腕部照片無法確認整個旋轉範圍都與物件分離。';
  const html = renderToStaticMarkup(createElement(ErrorNotice, { message }));
  const visible = html.split('<details>')[0];
  assert.match(visible, /圓筒可能擋到揮手，我先停下來。/);
  assert.doesNotMatch(visible, /單張腕部照片|人或寵物|正在抬|已抬|自動重試/);
  assert.ok(html.includes(message));
  assert.match(html, /<details><summary>查看原因<\/summary>/);
  assert.doesNotMatch(html, /<details open|<button/);
});

test('wave clearances distinguish people, other objects, and unresolved visibility', () => {
  for (const [message, expected] of [
    ['揮手空間尚未確認：有人手伸入旋轉範圍，旁邊也有圓筒。', '還不能確認人或寵物已避開揮手範圍'],
    ['揮手空間尚未確認：盒子擋住外側夾爪的旋轉範圍。', '旁邊的物件可能擋到揮手'],
    ['揮手空間尚未確認：看不清楚完整路徑，無法判斷。', '還看不清楚揮手的空間'],
    ['揮腕空間尚未確認：夾爪手指外側看不清楚。', '還看不清楚揮手的空間'],
  ]) {
    const html = renderToStaticMarkup(createElement(ErrorNotice, { message }));
    assert.ok(html.split('<details>')[0].includes(expected));
    assert.ok(html.includes(message));
  }
});

test('an unconfirmed raise path does not imply raising occurred or will happen automatically', () => {
  const message = '抬手路徑尚未確認：上方物件可能與夾爪相交。';
  const html = renderToStaticMarkup(createElement(ErrorNotice, { message }));
  const visible = html.split('<details>')[0];
  assert.match(visible, /旁邊的物件可能擋到抬手，我先停下來。/);
  assert.doesNotMatch(visible, /已抬|正在抬|我會抬|自動|已揮手/);
});

test('legacy wave reasons use wave context, preserve transport errors, and escape model text', () => {
  const message = '圓筒遮住旋轉範圍 <script>alert(1)</script>';
  const html = renderToStaticMarkup(createElement(ErrorNotice, { message, kind: 'wave' }));
  assert.match(html.split('<details>')[0], /圓筒可能擋到揮手/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<script>/);
  for (const failure of ['腕部畫面中斷，已停止移動', '腕部相機斷線，無法確認揮手範圍']) {
    const camera = renderToStaticMarkup(createElement(ErrorNotice, { message: failure, kind: 'wave' }));
    assert.ok(camera.includes(failure));
    assert.doesNotMatch(camera, /可能擋到|查看原因/);
  }
  const unrelated = renderToStaticMarkup(createElement(ErrorNotice, { message, kind: 'inspect' }));
  assert.doesNotMatch(unrelated, /可能擋到揮手|<details>/);
});

test('a screened correction boundary is not described as a mechanical end stop', () => {
  const html = renderToStaticMarkup(createElement(ErrorNotice, {message: '位置修正中止：elbow_flex: correction would cross safe joint limit'}));
  assert.match(html, /這一步已檢查的範圍/);
  assert.match(html, /不代表馬達已碰到實體行程極限/);
  assert.doesNotMatch(html, /連續四次/);
});

test('resume invalidates old polls and updates acknowledged local stop state', async () => {
  let release;
  context.pollRevision = {current: 7};
  context.request = () => new Promise((resolve) => {release = resolve;});
  let error = 'old fault', state = {robot:{stopped:true,error:'old fault'}};
  context.setError = (value) => {error = value;};
  context.setStatus = (update) => {state = update(state);};
  context.setHoldVerified = () => {};
  const pending = context.module.exports.resume();
  assert.equal(context.pollRevision.current, 8);
  release({stopped:false}); await pending;
  assert.equal(context.pollRevision.current, 9);
  assert.equal(error, '');
  assert.equal(state.robot.stopped, false);
  assert.equal(state.robot.error, '');
});

test('a newer stop supersedes a late resume response', async () => {
  let release;
  context.pollRevision = {current: 1};
  context.request = () => new Promise((resolve) => {release = resolve;});
  context.setError = context.setStatus = context.setHoldVerified = () => assert.fail('stale response must not update state');
  const pending = context.module.exports.resume();
  context.pollRevision.current += 1;
  release({stopped:false}); await pending;
});

const pick = {
  kind: 'pick',
  label: '抓起來',
  reason: '輕輕拿起',
  available: true,
  unavailable_reason: '',
};
const item = {
  name: '目標圓筒',
  safe_to_touch: true,
  matches_query: true,
  actions: [],
};
const status = (enabled = true) => ({
  capabilities: { pick: enabled },
  camera: { connected: true },
  robot: { stopped: false, moving: false },
  job: null,
  grasp: { state: 'idle', low_power_owner: false },
});
const job = (overrides = {}) => ({
  id: 'synthetic-job',
  kind: 'grasp-attempt',
  status: 'completed',
  stage: 'done',
  message: '',
  result: null,
  ...overrides,
});

test('point recognition states localization has not been performed, without inventing a failed precision check', () => {
  for (const scene of [
    {notice: '已認出你點的物件；抓取位置尚未通過精度驗證。'},
    {notice_kind: 'recognition_only', notice: 'new recognition message'},
  ]) {
    const message = sceneNotice(scene, job({kind: 'point'}));
    assert.match(message, /尚未嘗試抓取/);
    assert.match(message, /可選擇下方的「嘗試輕夾」/);
    assert.doesNotMatch(message, /尚未通過|驗證失敗|校正中/);
  }
  assert.equal(sceneNotice(null, null), null);
  assert.equal(sceneNotice({}, null), null);
});

test('a retained recognition banner yields to the actual operation progress or failure, preserving other scene limits', () => {
  const scene = {notice_kind: 'recognition_only', notice: 'recognition only'};
  for (const kind of ['grasp-attempt', 'object-program', 'command']) {
    for (const status of ['running', 'failed', 'cancelled', 'completed']) {
      assert.equal(sceneNotice(scene, job({kind, status})), null);
      const limited = {notice: '已完成加寬觀察；抓取仍須通過獨立精度驗證。'};
      assert.equal(sceneNotice(limited, job({kind, status})), limited.notice);
    }
  }
  const failed = job({status: 'failed', stage: 'error', message: '手眼校準檔無法對應目前相機'});
  const html = renderToStaticMarkup(createElement(ErrorNotice, {message: failed.message, kind: failed.kind}));
  assert.match(html, /手眼校準檔無法對應目前相機/);
  assert.match(html, /href="\/engineering\/calibration"/);
  assert.equal(graspOutcome(failed), null);
});

test('grasp localization progress follows the controller planning phase without claiming calibration or success', () => {
  for (const stage of ['starting', 'planning']) {
    const running = job({status: 'running', stage, message: '正在計算物件距離'});
    const view = jobPresentation(running, 'approaching');
    assert.equal(view.title, running.message);
    assert.match(view.description, /重新觀察、確認抓取位置/);
    assert.doesNotMatch(view.description, /校正中|已通過|已拿起/);
    assert.equal(graspOutcome(running), null);
  }
  assert.doesNotMatch(jobPresentation(job({status: 'running', stage: 'moving'}), 'approaching').description,
    /正在重新觀察/);
});

test('pick uses live capability instead of expired scene availability; client never enables a flag', () => {
  for (const s of [null, {}, status(false)])
    assert.equal(actionPresentation(pick, item, s, false, '').enabled, false);
  assert.equal(
    actionPresentation({ ...pick, available: false }, item, status(), false, '')
      .enabled,
    true,
  );
  assert.equal(
    actionPresentation(pick, item, status(), false, '').enabled,
    true,
  );
  assert.equal(
    actionPresentation(pick, item, status(), false, '').label,
    '抓起來 · 實驗',
  );
});

test('pick stays disabled while busy, disconnected, stopped, or already holding', () => {
  assert.equal(
    actionPresentation(pick, item, status(), true, '').enabled,
    false,
  );
  assert.equal(
    actionPresentation(pick, item, status(), false, 'lost connection').enabled,
    false,
  );
  for (const state of ['holding', 'holding_unverified', 'releasing']) {
    const s = status();
    s.grasp.state = state;
    assert.equal(actionPresentation(pick, item, s, false, '').enabled, false);
  }
  for (const mutate of [
    (s) => {
      s.camera.connected = false;
    },
    (s) => {
      s.robot.stopped = true;
    },
  ]) {
    const s = status();
    mutate(s);
    assert.equal(actionPresentation(pick, item, s, false, '').enabled, false);
  }
});

test('pick cannot override object eligibility or start a new task during an owned jaw session', () => {
  assert.equal(
    actionPresentation(
      pick,
      { ...item, safe_to_touch: false },
      status(),
      false,
      '',
    ).enabled,
    false,
  );
  const s = status();
  s.grasp = { state: 'approaching', low_power_owner: true };
  assert.equal(actionPresentation(pick, item, s, false, '').enabled, false);
});

test('disabled pick does not disable existing observation action', () => {
  const observe = { ...pick, kind: 'inspect', label: '再看仔細一點' };
  assert.equal(
    actionPresentation(observe, item, status(false), false, '').enabled,
    true,
  );
  assert.equal(
    actionPresentation(observe, item, status(false), false, '').label,
    observe.label,
  );
});

test('matching inspection-only object keeps a visible disabled pick with an explanation', () => {
  const object = {
    ...item,
    safe_to_touch: false,
    actions: [{ ...pick, kind: 'inspect', label: '查看細節' }],
  };
  for (const inline of [false, true]) {
    const html = renderToStaticMarkup(
      createElement(ResultActionButtons, {
        item: object,
        status: status(),
        busy: false,
        connectionIssue: '',
        inline,
        onAction: () => assert.fail('rendering must never execute an action'),
      }),
    );
    const buttons = html.match(/<button\b[^>]*>[\s\S]*?<\/button>/g);
    const picking = buttons.find((button) => button.includes('試著抓起來'));
    assert.ok(picking, html);
    assert.match(picking, /disabled=""/);
    assert.ok(html.includes('抓取暫不可用：目前影像還無法確認'));
    assert.equal((html.match(/試著抓起來/g) || []).length, 1);
  }
  assert.equal(
    object.actions.length,
    1,
    'UI must not mutate the model recommendation',
  );
});

test('default pick affordance preserves live capability and safety checks without duplicating a recommendation', () => {
  const actions = resultActions(item);
  assert.equal(actions.length, 1);
  assert.equal(actions[0].kind, 'pick');
  assert.equal(
    actionPresentation(actions[0], item, status(false), false, '').enabled,
    false,
  );
  assert.equal(
    actionPresentation(
      actions[0],
      { ...item, safe_to_touch: false },
      status(),
      false,
      '',
    ).enabled,
    false,
  );
  assert.equal(
    actionPresentation(actions[0], item, status(), false, '').enabled,
    true,
  );
  const recommended = { ...item, actions: [pick] };
  assert.equal(resultActions(recommended), recommended.actions);
  assert.equal(resultActions({ ...item, matches_query: false }).length, 0);
});

test('search result reports captured views without presenting an uncalibrated baseline as object distance', () => {
  const scene = {
    query: '圓筒',
    objects: [item],
    elapsed: 22,
    observation: {
      mode: 'wrist_roll_arc',
      view_count: 7,
      metric_calibrated: false,
      camera_baseline_model_mm: 50.2,
      measured_roll_range_deg: [-20, 20],
      not_external_metrology: true,
    },
  };
  const html = renderToStaticMarkup(createElement(ResultStats, { scene }));
  assert.ok(html.includes('已從 7 個視角觀察'));
  assert.ok(html.includes('22'));
  assert.equal(html.includes('50.2'), false);
  assert.equal(html.includes('mm'), false);
  assert.equal(html.includes('已校準'), false);
  for (const observation of [
    undefined,
    { mode: 'wrist_roll_arc', view_count: 1 },
    { mode: 'wrist_roll_arc', view_count: 2.5 },
    { mode: 'unknown', view_count: 7 },
  ])
    assert.equal(observationSummary(observation), '腕部相機實際觀察');
});

test('grasp progress uses actual stage and never calls in-progress verification a success', () => {
  const running = job({ status: 'running', message: '實際控制器訊息' });
  for (const [state, current] of [
    ['approaching', 0],
    ['closing', 1],
    ['verifying', 2],
  ]) {
    const view = jobPresentation(running, state);
    assert.equal(view.current, current);
    assert.equal(view.steps.join('/'), '對準/輕合/確認');
    assert.equal(view.title, '實際控制器訊息');
  }
  assert.match(
    jobPresentation(running, 'verifying').description,
    /不代表已經抓住/,
  );
});

test('fresh search does not falsely mark observation already done', () => {
  assert.equal(
    jobPresentation(job({ kind: 'search', stage: 'starting' })).current,
    0,
  );
  assert.equal(
    jobPresentation(job({ kind: 'search', stage: 'thinking' })).current,
    1,
  );
});

test('semantic search progress does not claim camera observation or motion before classification', () => {
  const view = jobPresentation(job({kind: 'search', stage: 'understanding', message: '先理解你的意思'}));
  assert.equal(view.title, '正在理解你的搜尋');
  assert.equal(view.steps.join('/'), '理解/整理結果');
  assert.equal(view.current, 0);
  assert.doesNotMatch(JSON.stringify(view), /相機|影像|視角|觀察|動手|正在執行/);
  const observing = jobPresentation(job({kind: 'search', stage: 'observing'}));
  assert.equal(observing.steps.join('/'), '觀察/理解/動手');
  assert.match(observing.description, /自己的視角觀察/);
});

test('browser search tool explains suggestions versus object observations without promising a scan', () => {
  let tool;
  function visit(node) {
    if (ts.isObjectLiteralExpression(node) && node.properties.some((property) =>
      ts.isPropertyAssignment(property) && property.name.getText(ast) === 'name'
      && ts.isStringLiteral(property.initializer) && property.initializer.text === 'search_with_robot')) tool = node;
    ts.forEachChild(node, visit);
  }
  visit(ast);
  assert.ok(tool);
  const description = tool.properties.find((property) =>
    ts.isPropertyAssignment(property) && property.name.getText(ast) === 'description').initializer.text;
  assert.match(description, /Action requests return suggestions/);
  assert.match(description, /never executes suggested actions/);
  assert.match(description, /Object searches may use the wrist camera and move the arm/);
  assert.doesNotMatch(description, /makes a small scan/);
});

test('running, failed, cancelled, or missing jobs cannot render a successful grasp', () => {
  for (const state of ['running', 'failed', 'cancelled', 'unavailable'])
    assert.equal(
      graspOutcome(
        job({ status: state, result: { grasp: { grasp_confirmed: true } } }),
      ),
      null,
    );
  assert.equal(graspOutcome(null), null);
});

test('completed job without an explicit grasp result remains unconfirmed', () => {
  const outcome = graspOutcome(job());
  assert.equal(outcome.confirmed, false);
  assert.match(outcome.title, /待確認/);
  assert.equal(
    graspOutcome(
      job({ engineering_result: { arbitrary_engineering_report: true } }),
    ).confirmed,
    false,
  );
});

test('unverified low-power hold gets a neutral result, not a green success', () => {
  const outcome = graspOutcome(
    job({
      result: {
        grasp: { grasp_confirmed: false, state: 'holding_unverified' },
      },
    }),
  );
  assert.equal(outcome.confirmed, false);
  assert.equal(outcome.title, '尚未確認拿起物件');
  assert.match(outcome.description, /保持姿勢/);
});

test('explicit confirmed grasp exposes the actual held wrist frame', () => {
  const frame = {
    id: 'synthetic-frame',
    url: '/api/frames/synthetic-frame',
    captured_at: 10,
  };
  const outcome = graspOutcome(
    job({
      result: {
        grasp: { grasp_confirmed: true, state: 'holding', held_frame: frame },
      },
    }),
  );
  assert.equal(outcome.confirmed, true);
  assert.equal(outcome.frame.url, frame.url);
});

test('restored direct grasp jobs use engineering_result without inventing a search scene', () => {
  const direct = job({
    engineering_result: { grasp_confirmed: false, state: 'holding_unverified' },
  });
  assert.equal(graspOutcome(direct).confirmed, false);
  assert.equal(direct.result, null);
  assert.equal(
    graspOutcome(
      job({ engineering_result: { grasp_confirmed: true, state: 'holding' } }),
    ).confirmed,
    true,
  );
});

test('stop acknowledgement prevents an old confirmed receipt from reappearing as live success', () => {
  const completed = job({ result: { grasp: { grasp_confirmed: true } } });
  assert.equal(graspOutcome(completed, true), null);
});

test('release completion needs explicit released truth and keeps its own wording', () => {
  const released = job({
    kind: 'grasp-release',
    engineering_result: { released: true },
  });
  assert.equal(graspOutcome(released).title, '物件已釋放');
  assert.equal(graspOutcome(job({ kind: 'grasp-release' })).confirmed, false);
  assert.equal(
    graspOutcome(
      job({ kind: 'search', result: { grasp: { grasp_confirmed: true } } }),
    ),
    null,
  );
});

test('all named result actions acquire fresh evidence without a scene id or coordinates', () => {
  for (const kind of ['inspect', 'pick']) {
    const request = actionRequest(kind, '  目標圓筒  ');
    assert.equal(request.path, '/api/object-action');
    assert.deepEqual(JSON.parse(JSON.stringify(request.body)), {
      kind,
      query: item.name,
    });
  }
  assert.deepEqual(
    JSON.parse(JSON.stringify(actionRequest('wave', 'ignored old query'))),
    {
      path: '/api/object-action',
      body: { kind: 'wave', query: '' },
    },
  );
  assert.throws(() => actionRequest('place', 'cup'), /搜尋框.*放到哪裡/);
  assert.throws(() => actionRequest('dance', 'cup'), /尚未開放/);
  assert.throws(() => actionRequest('pick', '  '), /選擇物件/);
});

test('controller restart retires the lost running job and permits fresh result actions with retained results', async () => {
  const prior = job({
    id: 'lost',
    status: 'running',
    result: { id: 'expired', objects: [item] },
  });
  const fresh = status();
  const snapshot = await readControllerPoll(
    async (path) => {
      if (path === '/api/status') return fresh;
      throw new ControllerRequestError(404, 'record was lost');
    },
    prior.id,
    () => true,
  );
  const retired = retireMissingJob(prior, snapshot.missingId);
  assert.equal(retired.status, 'unavailable');
  assert.equal(retired.result, prior.result);
  const view = progressPresentation(
    false,
    retired,
    snapshot.status.job,
    false,
    '',
  );
  assert.equal(view.busy, false);
  assert.equal(view.showProgress, false);
  for (const kind of ['pick', 'inspect']) {
    const staleAction = {
      ...pick,
      kind,
      available: false,
      unavailable_reason: 'old scene unavailable',
    };
    assert.equal(
      actionPresentation(staleAction, item, fresh, view.busy, '').enabled,
      true,
    );
    assert.equal(actionRequest(kind, item.name).path, '/api/object-action');
  }
  assert.equal(retireMissingJob(prior, 'different-job'), prior);
});

test('camera-only details remain available while stopped or holding, but wave and pick do not', () => {
  for (const graspState of ['holding', 'holding_unverified', 'releasing']) {
    const s = status();
    s.grasp.state = graspState;
    s.robot.stopped = true;
    assert.equal(
      actionPresentation(
        { ...pick, kind: 'inspect', available: false },
        item,
        s,
        false,
        '',
      ).enabled,
      true,
    );
    for (const kind of ['wave', 'pick'])
      assert.equal(
        actionPresentation({ ...pick, kind }, item, s, false, '').enabled,
        false,
      );
  }
  for (const kind of ['inspect', 'pick', 'wave']) {
    assert.equal(
      actionPresentation({ ...pick, kind }, item, status(), true, '').enabled,
      false,
    );
    assert.equal(
      actionPresentation({ ...pick, kind }, item, status(), false, 'offline')
        .enabled,
      false,
    );
    assert.equal(
      actionPresentation(
        { ...pick, kind },
        item,
        { ...status(), camera: { connected: false } },
        false,
        '',
      ).enabled,
      false,
    );
  }
  assert.equal(
    actionPresentation({ ...pick, kind: 'wave' }, null, status(), false, '')
      .enabled,
    true,
  );
});

test('explicit actions submit once and an uncertain mutation never automatically retries', async () => {
  const calls = [];
  const submission = actionRequest('pick', item.name);
  await assert.rejects(
    controllerRequest(
      submission.path,
      submission.body,
      { current: 'mock' },
      async (path, init) => {
        calls.push({ path, body: JSON.parse(init.body) });
        return new Promise(() => {});
      },
      15,
    ),
    /無法確認是否已受理/,
  );
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0], {
    path: '/api/object-action',
    body: { kind: 'pick', query: item.name },
  });
});

test('wave requires explicit completion evidence and displays measured angle range and cycles', () => {
  const running = job({
    kind: 'wave',
    status: 'running',
    stage: 'moving',
    message: '正在擺動手腕',
  });
  assert.equal(jobPresentation(running).title, running.message);
  assert.equal(jobPresentation(running).current, 1);
  for (const status of ['running', 'failed', 'cancelled', 'unavailable'])
    assert.equal(waveOutcome({ ...running, status }), null);
  const done = {
    ...running,
    status: 'completed',
    message: '已完成揮手',
    engineering_result: {
      wave_completed: true,
      cycles: 2,
      measured_roll_range_deg: [-5.2, 5.1],
    },
  };
  assert.equal(waveOutcome(done).confirmed, true);
  assert.equal(waveOutcome(done).title, '已完成揮手');
  assert.equal(
    waveOutcome(done).description,
    '完成 2 次揮手 · 手腕實測角度 -5.2° 至 5.1°',
  );
  for (const receipt of [
    null,
    {},
    { wave_completed: false },
    { wave_completed: 'true' },
  ]) {
    const unconfirmed = waveOutcome({ ...done, engineering_result: receipt });
    assert.equal(unconfirmed.confirmed, false);
    assert.equal(unconfirmed.title, '揮手結果仍待確認');
  }
  assert.equal(
    waveOutcome({
      ...done,
      engineering_result: {
        wave_completed: true,
        cycles: -1,
        measured_roll_range_deg: [5, -5],
      },
    }).description,
    '',
  );
  assert.equal(waveOutcome(done, true), null);
  assert.equal(waveOutcome(job()), null);
  assert.equal(graspOutcome(done), null);
});

test('verified wave receipt survives later jobs and a missing old record without treating them as wave success', () => {
  const done = job({
    kind: 'wave',
    engineering_result: {
      wave_completed: true,
      cycles: 2,
      measured_roll_range_deg: [-5, 5],
    },
  });
  const saved = retainWaveReceipt(null, done);
  assert.equal(saved, done);
  for (const later of [
    job({ id: 'inspection', kind: 'inspect' }),
    job({ id: 'new-wave', kind: 'wave', status: 'running' }),
    job({ id: 'new-wave', kind: 'wave', status: 'failed' }),
    retireMissingJob(done, done.id),
  ])
    assert.equal(retainWaveReceipt(saved, later), saved);
  assert.equal(retainWaveReceipt(null, job({ kind: 'wave' })), null);
  assert.equal(
    retainWaveReceipt(saved, { ...done }),
    saved,
    'Repeated polls retain one receipt',
  );
  assert.equal(waveOutcome(saved).confirmed, true);
});

test('opening a stopped or owned camera preview never submits a look motion', async () => {
  const cases = [
    null,
    { ...status(), robot: { stopped: true } },
    ...['approaching', 'holding', 'holding_unverified', 'releasing'].map(
      (state) => ({ ...status(), grasp: { state, low_power_owner: true } }),
    ),
    ...['holding', 'holding_unverified', 'releasing'].map(
      (state) => ({ ...status(), grasp: { state, low_power_owner: false } }),
    ),
  ];
  for (const s of cases) {
    const calls = [];
    Object.assign(context, {
      busy: false,
      status: s,
      setCameraPoint: (value) => calls.push(['point', value]),
      setCameraOpen: (value) => calls.push(['open', value]),
      setError: (value) => calls.push(['error', value]),
      request: async (path) => {
        calls.push(['POST', path]);
        return { id: 'look' };
      },
      accept: (value) => calls.push(['accepted', value]),
    });
    await openCamera();
    assert.ok(calls.some(([kind, value]) => kind === 'open' && value === true));
    assert.equal(
      calls.some(([kind]) => kind === 'POST' || kind === 'accepted'),
      false,
    );
  }
  const calls = [];
  Object.assign(context, {
    status: status(),
    request: async (path) => {
      calls.push(path);
      return { id: 'look' };
    },
    accept: () => {},
  });
  await openCamera();
  assert.deepEqual(calls, ['/api/look']);
});

test('opening preview preserves a fresh usable wrist view even before the dialog has been opened', async () => {
  const calls = [];
  const fresh = { ...status(), camera: { connected: true, source: 'wrist', error: '',
    age_ms: 250, image_quality: { usable: true } } };
  Object.assign(context, {
    busy: false,
    status: fresh,
    setCameraPoint: (value) => calls.push(['point', value]),
    setCameraOpen: (value) => calls.push(['open', value]),
    setError: (value) => calls.push(['error', value]),
    request: async (path) => { calls.push(['POST', path]); return { id: 'look' }; },
    accept: (value) => calls.push(['accepted', value]),
  });
  await openCamera();
  assert.deepEqual(calls, [['point', null], ['open', true], ['error', '']]);
  assert.equal(cameraPreviewNeedsLook(fresh), false);
});

test('initial look remains available when the current wrist view is missing, stale or unusable', () => {
  const freshCamera = { connected: true, source: 'wrist', error: '',
    age_ms: 250, image_quality: { usable: true } };
  for (const change of [
    { connected: false }, { source: 'engineering' }, { source: undefined },
    { age_ms: null }, { age_ms: -1 }, { age_ms: 1000 }, { age_ms: NaN },
    { age_ms: Infinity }, { image_quality: { usable: false } },
    { image_quality: undefined }, { error: 'camera disconnected' },
  ]) {
    assert.equal(cameraPreviewNeedsLook({ ...status(), camera: { ...freshCamera, ...change } }), true);
  }
  assert.equal(cameraPreviewNeedsLook({ ...status(), camera: { ...freshCamera, age_ms: 999 } }), false);
});

test('details render the new wrist frame, evidence, uncertainty and escaped object text', () => {
  const details = {
    visible: true,
    object_name: '圓筒',
    summary: '白色圓筒，目前可見部分標籤。',
    attributes: ['白色瓶身'],
    visible_text: ['<script>untrusted</script>'],
    unknowns: ['內容物無法確認'],
    frame: { id: 'fresh', url: '/api/frames/fresh', captured_at: 10 },
  };
  const html = renderToStaticMarkup(createElement(ObjectDetails, { details }));
  for (const text of [
    '物件詳細資訊',
    '最新腕部觀察',
    '/api/frames/fresh',
    '白色瓶身',
    '內容物無法確認',
    '&lt;script&gt;untrusted&lt;/script&gt;',
  ])
    assert.ok(html.includes(text), text);
  assert.equal(html.includes('<script>'), false);
  const absent = renderToStaticMarkup(
    createElement(ObjectDetails, {
      details: { ...details, visible: false, attributes: [], visible_text: [] },
    }),
  );
  assert.ok(absent.includes('這次畫面中沒有確認到目標'));
  assert.equal(absent.includes('看得見的特徵'), false);
});

const observedFrame = (id = 'new-angle') => ({
  id, url: `/api/frames/${id}`, captured_at: 100, source: 'wrist',
});
const observedJob = (changes = {}) => job({
  id: 'current-object-task', kind: 'grasp-attempt', status: 'failed',
  engineering_result: {task: 'grasp', completed: false, grasp_confirmed: false,
    observations: [{completed: true, after: observedFrame()}]},
  ...changes,
});

test('partial object operation renders its latest completed wrist observation without claiming success', () => {
  for (const kind of ['grasp-attempt', 'object-program']) {
    const current = observedJob({kind});
    current.engineering_result.observations.unshift({completed: true, after: observedFrame('older-angle')});
    current.engineering_result.observations.push({completed: false, after: observedFrame('unfinished-angle')});
    const evidence = observationEvidence(current, current);
    assert.equal(evidence.frame.id, 'new-angle');
    assert.equal(evidence.description, '已重新觀察；物件操作尚未完成');
    const html = renderToStaticMarkup(createElement(ObservationEvidenceCard, {evidence}));
    assert.match(html, /換角度後的畫面/);
    assert.match(html, /\/api\/frames\/new-angle/);
    assert.match(html, /已重新觀察；物件操作尚未完成/);
    assert.match(html, /腕部相機/);
    assert.doesNotMatch(html, /已拿起物件|已確認堆疊完成|older-angle|unfinished-angle/);
  }
});

test('observation card is limited to completed observations owned by the current job', () => {
  const current = observedJob();
  for (const [shown, authoritative] of [
    [null, current], [current, null],
    [current, {...current, id: 'newer-task'}],
    [{...current, kind: 'search'}, current],
    [{...current, status: 'unavailable'}, current],
    [{...current, engineering_result: undefined, result: {grasp: current.engineering_result}}, current],
    [{...current, engineering_result: {}}, current],
    [{...current, engineering_result: {observations: []}}, current],
    [{...current, engineering_result: {observations: [{completed: false, after: observedFrame()}]}}, current],
    [{...current, engineering_result: {observations: [{after: observedFrame()}]}}, current],
  ]) assert.equal(observationEvidence(shown, authoritative), null);
});

test('observation evidence rejects other cameras and nonlocal or mismatched frame URLs', () => {
  for (const change of [
    {source: 'engineering-observer'}, {source: 'd435'}, {source: undefined},
    {url: 'javascript:alert(1)'}, {url: 'https://example.test/frame'},
    {url: '//example.test/frame'}, {url: '/api/frames/a-different-frame'},
    {url: '/api/frames/new-angle?external=true'}, {id: '../frame'},
    {captured_at: NaN}, {captured_at: Infinity}, {captured_at: '100'},
  ]) {
    const current = observedJob();
    current.engineering_result.observations[0].after = {...observedFrame(), ...change};
    assert.equal(observationEvidence(current, current), null, JSON.stringify(change));
  }
});

test('observation card remains a timestamped observation after verified task completion', () => {
  for (const task of ['grasp', 'place', 'stack']) {
    const current = observedJob({status: 'completed', kind: task === 'grasp' ? 'grasp-attempt' : 'object-program'});
    Object.assign(current.engineering_result, {task, completed: true, [`${task}_confirmed`]: true});
    assert.equal(observationEvidence(current, current).description, '這是操作前換角度時的觀察紀錄。');
    current.engineering_result[`${task}_confirmed`] = false;
    assert.equal(observationEvidence(current, current).description, '已重新觀察；物件操作尚未完成');
  }
});

test('same command partial step exposes observation evidence without relying on an unrelated latest result', () => {
  const receipt = observedJob().engineering_result;
  const current = observedJob({kind: 'command', engineering_result: undefined,
    command_result: {completed: false, status: 'blocked', steps: [
      {skill: 'observe', completed: true},
      {skill: 'grasp', completed: false, receipt},
    ]}});
  assert.equal(observationEvidence(current, current).frame.id, 'new-angle');
  assert.equal(observationEvidence(current, current).description, '已重新觀察；物件操作尚未完成');
  current.command_result.steps = [{skill: 'grasp', completed: false}];
  current.engineering_result = receipt;
  assert.equal(observationEvidence(current, current), null);
});

test('an older detail summary is explicitly labeled earlier and retains its own image timestamp', () => {
  const details = {visible: true, object_name: '圓筒', summary: '先前夾爪擋住圓筒底部。',
    attributes: [], visible_text: [], unknowns: [], frame: observedFrame('old-inspection')};
  const html = renderToStaticMarkup(createElement(ObjectDetails, {details, earlier: true}));
  assert.match(html, /先前的詳細觀察/);
  assert.match(html, /先前夾爪擋住圓筒底部/);
  assert.match(html, /\/api\/frames\/old-inspection/);
  assert.match(html, /腕部相機/);
  assert.doesNotMatch(html, /最新腕部觀察|\/api\/frames\/new-angle/);
});

test('active observation evidence renders before retained inspection details', () => {
  const card = source.indexOf('{reobserved && <ObservationEvidenceCard');
  const details = source.indexOf('{scene.details && <ObjectDetails');
  assert.ok(card > source.indexOf('<main className="results">') && card < details);
  assert.match(source, /earlier=\{Boolean\(reobserved && scene\.details\.frame\?\.id !== reobserved\.frame\.id\)\}/);
});


test('place and stack preserve both named objects and the recommended destination on click', async () => {
  for (const kind of ['place', 'stack']) {
    const destination = '藍色盒子的左上方';
    const request = actionRequest(kind, ' 白黃圓筒 ', ` ${destination} `);
    assert.equal(request.path, '/api/object-action');
    assert.equal(request.body.kind, kind);
    assert.match(request.body.query, /白黃圓筒/);
    assert.match(request.body.query, /藍色盒子的左上方/);
    if (kind === 'stack') assert.match(request.body.query, /疊放在.*上面$/);
    for (const missing of ['', '  ', undefined])
      assert.throws(() => actionRequest(kind, '白黃圓筒', missing), /搜尋框.*放到哪裡/);
  }
  const sent = [];
  Object.assign(context, {
    busy: false, connectionIssue: '', q: 'another query that must not replace destination',
    status: {...status(), capabilities: {pick: true, place: true, stack: true}},
    setError() {}, setSending() {}, setResultMode() {}, accept() {},
    request: async (path, body) => {sent.push({path, body}); return {id: 'object-program'};},
  });
  await context.module.exports.objectAction('stack', item, {...pick, kind: 'stack', target: '紅色積木'});
  assert.equal(sent.length, 1);
  assert.deepEqual(JSON.parse(JSON.stringify(sent[0])), {
    path: '/api/object-action', body: {kind: 'stack', query: '把「目標圓筒」疊放在「紅色積木」上面'},
  });
  await context.module.exports.objectAction('place', item, {...pick, kind: 'place', target: ''});
  assert.equal(sent.length, 1, 'an absent destination never submits a mutation');
});

test('place and stack need live capability, an eligible source, a destination and an idle unheld robot', () => {
  for (const kind of ['place', 'stack']) {
    const action = {...pick, kind, target: '盒子'};
    const ready = {...status(), capabilities: {pick: true, place: true, stack: true}};
    assert.equal(actionPresentation(action, item, ready, false, '').enabled, true);
    assert.match(actionPresentation(action, item, ready, false, '').label, /實驗/);
    for (const change of [
      {capabilities: {pick: true}}, {robot: {...ready.robot, stopped: true}},
      {robot: {...ready.robot, moving: true}}, {job: {status: 'running'}},
      {grasp: {state: 'holding', low_power_owner: true}},
      {grasp: {state: 'approaching', low_power_owner: true}},
    ]) assert.equal(actionPresentation(action, item, {...ready, ...change}, false, '').enabled, false);
    assert.equal(actionPresentation(action, item, ready, true, '').enabled, false);
    assert.equal(actionPresentation(action, item, ready, false, 'offline').enabled, false);
    assert.equal(actionPresentation(action, {...item, safe_to_touch: false}, ready, false, '').enabled, false);
    const missing = actionPresentation({...action, target: ' '}, item, ready, false, '');
    assert.equal(missing.enabled, false);
    assert.match(missing.reason, /搜尋框.*放到哪裡/);
  }
  const holding = {...status(), capabilities: {command: true}, grasp: {state: 'holding', low_power_owner: true}};
  assert.equal(commandPresentation('放開', holding, false, '').enabled, true);
  for (const instruction of ['揮手', '拿起積木', '放開然後揮手'])
    assert.equal(commandPresentation(instruction, holding, false, '').enabled, false);
  for (const kind of ['stacking', 'place_and_wave', 'execute', '__proto__', '']) {
    assert.throws(() => actionRequest(kind, '物件', '盒子'), /尚未開放/);
    assert.equal(actionPresentation({...pick, kind, target: '盒子'}, item, status(), false, '').enabled, false);
  }
});

test('missing placement destination stays visible with a search-box instruction', () => {
  const selected = {...item, actions: [{...pick, kind: 'stack', label: '疊上去', target: ''}]};
  const html = renderToStaticMarkup(createElement(ResultActionButtons, {
    item: selected, status: {...status(), capabilities: {pick: true, stack: true}},
    busy: false, connectionIssue: '', onAction() {}, inline: true,
  }));
  assert.match(html, /疊上去 · 實驗/);
  assert.match(html, /請在搜尋框寫出要放到哪裡/);
  assert.match(html.match(/<button[^>]*>[^]*?疊上去[^]*?<\/button>/)?.[0] || '', /disabled=""/);
});

test('object-program progress shows actual planning, execution and verification without a success claim', () => {
  for (const [stage, current] of [['planning', 0], ['moving', 1], ['verifying', 2]]) {
    const running = job({kind: 'object-program', stage, status: 'running', message: '目前步驟'});
    const view = jobPresentation(running);
    assert.equal(view.current, current);
    assert.equal(view.steps.join('/'), '規劃/執行/核對');
    assert.equal(view.title, '目前步驟');
    assert.equal(graspOutcome(running), null);
  }
  assert.match(jobPresentation(job({kind: 'object-program', stage: 'verifying'})).description, /不代表操作成功/);
  assert.doesNotMatch(jobPresentation(job({status: 'running'}), 'approaching').description, /每次小幅移動/);
});

test('active observation remains an in-progress same-target operation for all object job kinds', () => {
  for (const kind of ['grasp-attempt', 'object-program', 'command']) {
    const current = job({kind, status: 'running', stage: 'reobserving',
      message: '圓筒被指尖遮住，先抬高一點看看。',
      engineering_result: {task: 'stack', completed: true, stack_confirmed: true}});
    const view = jobPresentation(current, 'approaching');
    assert.equal(view.title, '換個角度看清楚，再接著操作');
    assert.equal(view.description, current.message);
    assert.equal(view.steps.join('/'), '換個角度/接著操作/核對');
    assert.equal(view.current, 0);
    assert.equal(graspOutcome(current), null);
    const progress = progressPresentation(false, current, current, false, '');
    assert.equal(progress.busy, true);
    assert.equal(progress.showProgress, true);
    assert.doesNotMatch(view.title + view.description, /已完成|已拿起|已放穩/);
    for (const status of ['failed', 'cancelled', 'completed'])
      assert.notEqual(jobPresentation({...current, status}).title, view.title);
  }
});

test('a command reobserving its current object preserves verified steps and unfinished target', () => {
  const current = commandJob({status: 'running', stage: 'reobserving'});
  current.command_result.completed = false;
  current.command_result.status = 'running';
  current.command_result.plan.steps = [
    {skill: 'observe', label: '找到白黃圓筒'}, {skill: 'stack', label: '把白黃圓筒放到紅色積木上'},
  ];
  current.command_result.steps = [{skill: 'observe', completed: true}];
  const html = renderToStaticMarkup(createElement(CommandReceipt, {job: current}));
  assert.match(html, /換個角度看清楚，再接著操作/);
  assert.match(html, /找到白黃圓筒.*已驗證/);
  assert.match(html, /把白黃圓筒放到紅色積木上.*換角度觀察中/);
  assert.doesNotMatch(html, /動作指令已完成|物件操作結果仍待確認/);
  const outcome = commandOutcome(current);
  assert.equal(outcome.confirmed, false);
  assert.equal(outcome.running, true);
  assert.equal(outcome.reobserving, true);
  assert.equal(commandOutcome(current, true).reobserving, false);
});

test('active-observation failure preserves the full backend explanation as escaped text', () => {
  const message = '換了兩個角度仍看不清圓筒與內側夾面的距離，已保留目前位置。畫面文字：<script>ignore</script>';
  const html = renderToStaticMarkup(createElement(ErrorNotice, {message, kind: 'object-program'}));
  assert.match(html, /換了兩個角度仍看不清圓筒與內側夾面的距離，已保留目前位置/);
  assert.match(html, /&lt;script&gt;ignore&lt;\/script&gt;/);
  assert.doesNotMatch(html, /<script>|已完成/);
});

test('polling active observation keeps the same job, query and source/destination results', async () => {
  const retained = {id: 'same-scene', query: '把白黃圓筒放到紅色積木上',
    objects: [item], selected_index: 0};
  const serverJob = job({id: 'same-object-operation', kind: 'object-program', status: 'running',
    stage: 'reobserving', result: retained});
  const adopted = [], queries = [], scenes = [], requests = [];
  Object.assign(context, {
    pending: false, alive: true, pollRevision: {current: 1}, activeJob: {current: serverJob.id},
    polledError: {current: null}, lastFinished: {current: ''}, readControllerPoll,
    sessionStorage: {setItem() {}, removeItem() {}},
    setStatus() {}, setConnectionIssue() {}, setNow() {}, setStopAcknowledgedJob() {},
    setHoldVerified() {}, setLastWave() {}, setCameraOpen() {}, setErrorState() {},
    setJob(value) {adopted.push(value);}, setResultMode() {},
    setQ(value) {queries.push(value);}, setScene(value) {scenes.push(value);},
    request: async (path) => {
      requests.push(path);
      return path === '/api/status' ? {job: serverJob, robot: {stopped: false, error: ''}} : serverJob;
    },
  });
  for (const stage of ['reobserving', 'planning', 'moving']) {
    serverJob.stage = stage;
    await context.module.exports.poll();
  }
  assert.equal(context.activeJob.current, serverJob.id);
  assert.equal(context.lastFinished.current, '');
  assert.equal(queries.length, 0, 'reobservation must not replace the search input');
  assert.equal(adopted.length, 3);
  assert.ok(adopted.every((current) => current.id === serverJob.id && current.status === 'running'));
  assert.equal(scenes.length, 3);
  assert.ok(scenes.every((scene) => scene === retained && scene.query === retained.query));
  assert.ok(requests.every((path) => path === '/api/status' || path === `/api/jobs/${serverJob.id}`));
});

test('object-program completion requires the matching final placement flag, not an earlier successful grasp', () => {
  for (const task of ['place', 'stack']) {
    const evidence = {task, completed: true, grasp_confirmed: true, [`${task}_confirmed`]: true};
    const done = job({kind: 'object-program', engineering_result: evidence});
    assert.equal(graspOutcome(done).confirmed, true);
    assert.doesNotMatch(graspOutcome(done).title, /已拿起/);
    for (const changed of [
      {}, {...evidence, completed: false}, {...evidence, [`${task}_confirmed`]: false},
      {...evidence, task: 'unknown'}, {task, completed: true, grasp_confirmed: true},
    ]) assert.equal(graspOutcome({...done, engineering_result: changed}).confirmed, false);
    for (const state of ['running', 'failed', 'cancelled'])
      assert.equal(graspOutcome({...done, status: state}), null);
    assert.equal(graspOutcome(done, true), null);
  }
});

test('polling adopts an externally started object program and opens its result without inventing a scene', async () => {
  let adopted = null;
  const modes = [];
  const serverJob = job({id: 'external-program', kind: 'object-program', status: 'running',
    engineering_result: {task: 'stack', completed: true, stack_confirmed: true}});
  Object.assign(context, {
    pending: false, alive: true, pollRevision: {current: 1}, activeJob: {current: null},
    polledError: {current: null}, lastFinished: {current: ''}, readControllerPoll,
    sessionStorage: {setItem() {}, removeItem() {}},
    setStatus() {}, setConnectionIssue() {}, setNow() {}, setStopAcknowledgedJob() {},
    setHoldVerified() {}, setLastWave() {}, setQ() {}, setCameraOpen() {}, setErrorState() {},
    setJob(value) {adopted = value;}, setResultMode(value) {modes.push(value);},
    setScene() {assert.fail('a motor-only program must not manufacture a search scene');},
    request: async () => ({job: serverJob, robot: {stopped: false, error: ''}}),
  });
  await context.module.exports.poll();
  assert.equal(context.activeJob.current, 'external-program');
  assert.equal(adopted.kind, 'object-program');
  assert.ok(modes.includes(true));
  assert.equal(graspOutcome(adopted), null);
  serverJob.status = 'completed';
  await context.module.exports.poll();
  assert.equal(graspOutcome(adopted).confirmed, true);
});
