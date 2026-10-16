import test from 'node:test';
import assert from 'node:assert/strict';
import { ControllerRequestError, controllerRequest, readControllerPoll, progressPresentation } from '../lib/client-status.ts';
const json = (data, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'content-type': 'application/json' } });
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return { promise, resolve, reject }; };

test('running job reaches done and no longer displays work', async () => {
  const partial = { id: 'scene', objects: [{ name: 'cup' }] };
  const result = await readControllerPoll(async () => ({ job: { id: 'a', status: 'done', result: partial } }), 'a', () => true);
  assert.equal(result.kind, 'snapshot'); assert.equal(result.current.result, partial);
  assert.deepEqual(progressPresentation(false, result.current, result.status.job, false, ''), { busy: false, showProgress: false, stopped: false });
});
test('saved-job lookup restores the same partial result', async () => {
  const partial = { id: 'scene', objects: [{ name: 'cup' }] }; const calls = [];
  const result = await readControllerPoll(async path => { calls.push(path); return path === '/api/status' ? { job: null } : { id: 'a', status: 'running', result: partial }; }, 'a', () => true);
  assert.deepEqual(calls, ['/api/status', '/api/jobs/a']); assert.equal(result.current.result, partial); assert.equal(result.activeId, 'a');
});
test('lookup 500/network failures never become missing-job decisions', async () => {
  for (const failure of [new ControllerRequestError(500, 'server temporarily unavailable'), new TypeError('network failure')]) {
    await assert.rejects(readControllerPoll(async path => { if (path === '/api/status') return { job: null }; throw failure; }, 'a', () => true), e => e === failure);
  }
});
test('confirmed 404 reports unavailable record without claiming completion or stop', async () => {
  const result = await readControllerPoll(async path => { if (path === '/api/status') return { job: null }; throw new ControllerRequestError(404, 'missing'); }, 'old', () => true);
  assert.equal(result.missingId, 'old'); assert.equal(result.activeId, ''); assert.equal(result.current, null);
  assert.equal('completed' in result, false); assert.equal('stopped' in result, false);
});
test('late old status cannot replace a job accepted while polling', async () => {
  const wait = deferred(); let current = true; const calls = [];
  const polling = readControllerPoll(async path => { calls.push(path); return wait.promise; }, 'old', () => current);
  current = false; wait.resolve({ job: { id: 'old', status: 'running' } });
  assert.deepEqual(await polling, { kind: 'stale' }); assert.deepEqual(calls, ['/api/status']);
});
test('late lookup error cannot erase a newly accepted job or its scene', async () => {
  const wait = deferred(); let current = true;
  const polling = readControllerPoll(async path => path === '/api/status' ? { job: null } : wait.promise, 'old', () => current);
  await Promise.resolve(); current = false; wait.reject(new ControllerRequestError(404, 'missing old job'));
  assert.deepEqual(await polling, { kind: 'stale' });
});
test('hanging fetch is bounded, aborted, and subsequent read can recover', async () => {
  let signal; const token = { current: '' };
  await assert.rejects(controllerRequest('/api/status', undefined, token, async (_p, init) => { signal = init.signal; return new Promise(() => {}); }, 15), /連線逾時/);
  assert.equal(signal.aborted, true);
  const result = await controllerRequest('/api/status', undefined, token, async () => json({ job: { id: 'a', status: 'done' } }), 50);
  assert.equal(result.job.status, 'done');
});
test('headers received but hanging JSON body is also bounded', async () => {
  await assert.rejects(controllerRequest('/api/status', undefined, { current: '' }, async () => ({ ok: true, json: () => new Promise(() => {}) }), 15), /連線逾時/);
});
test('uncertain POST is never automatically retried', async () => {
  const calls = [];
  await assert.rejects(controllerRequest('/api/search', { query: 'cup' }, { current: 'synthetic-token' }, async path => { calls.push(path); return new Promise(() => {}); }, 15), /無法確認是否已受理/);
  assert.deepEqual(calls, ['/api/search']);
});
test('explicit 403 refreshes token then retries once', async () => {
  const calls = []; const token = { current: 'expired-synthetic-token' };
  const result = await controllerRequest('/api/search', { query: 'cup' }, token, async (path, init) => {
    calls.push([path, init.headers?.['x-lerobot-token']]);
    if (path === '/api/session') return json({ token: 'new-synthetic-token' });
    return calls.length === 1 ? json({ detail: 'expired' }, 403) : json({ id: 'accepted', status: 'running' });
  }, 50);
  assert.equal(result.id, 'accepted'); assert.deepEqual(calls.map(x => x[0]), ['/api/search', '/api/session', '/api/search']);
  assert.equal(calls[2][1], 'new-synthetic-token');
});
test('500 mutation failure is not retried or mistaken for token expiry', async () => {
  let calls = 0;
  await assert.rejects(controllerRequest('/api/search', {}, { current: 'synthetic-token' }, async () => { calls++; return json({ detail: 'unavailable' }, 500); }, 50), e => e.status === 500);
  assert.equal(calls, 1);
});
test('lost connection removes observation spinner while retaining blocked controls and job identity', () => {
  const job = { id: 'a', status: 'running', result: { id: 'partial' } };
  const result = progressPresentation(false, job, null, true, '');
  assert.deepEqual(result, { busy: true, showProgress: false, stopped: false }); assert.equal(job.status, 'running'); assert.equal(job.result.id, 'partial');
});
test('acknowledged stop does not revive observation animation during backend cancellation cleanup', () => {
  const job = { id: 'a', status: 'running' };
  assert.deepEqual(progressPresentation(false, job, job, false, 'a'), { busy: true, showProgress: false, stopped: true });
  const next = { id: 'b', status: 'running' };
  assert.deepEqual(progressPresentation(false, next, next, false, 'a'), { busy: true, showProgress: true, stopped: false });
});
