// 推理引擎生命周期：GPU 重置错误识别、串行队列。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  abortError,
  createInflightGate,
  createSerialQueue,
  GPU_RESET_MESSAGE,
  gpuResetError,
  isGpuResetError,
} from '../web/src/engines/lifecycle.js';

test('isGpuResetError：识别 mapAsync / Instance reference 等 GPU 重置错误', () => {
  const e = new Error(
    "Failed to execute 'mapAsync' on 'GPUBuffer': A valid external Instance reference no longer exists.",
  );
  e.name = 'AbortError'; // 原始错误的名字就是 AbortError，必须仍能识别
  assert.equal(isGpuResetError(e), true);
  assert.equal(isGpuResetError(new Error('The WebGPU device was lost while loading the model.')), true);
  assert.equal(isGpuResetError(new Error('GPU process crashed, device lost')), true);
});

test('isGpuResetError：普通错误与用户中止不匹配', () => {
  const abort = new Error('The operation was aborted.');
  abort.name = 'AbortError';
  assert.equal(isGpuResetError(abort), false);
  assert.equal(isGpuResetError(new Error('Prompt tokens exceed context window size')), false);
  assert.equal(isGpuResetError(new Error('Failed to fetch')), false);
  assert.equal(isGpuResetError(null), false);
  assert.equal(isGpuResetError(undefined), false);
});

test('gpuResetError：统一的错误名与提示，且自身不会再被识别为重置错误', () => {
  const e = gpuResetError();
  assert.equal(e.name, 'GpuResetError');
  assert.equal(e.message, GPU_RESET_MESSAGE);
  assert.equal(e.message, 'GPU 资源已重置，请重新发送。');
  assert.notEqual(e.name, 'AbortError');
  assert.equal(isGpuResetError(e), false);
});

test('createSerialQueue：任务按提交顺序一个接一个执行', async () => {
  const run = createSerialQueue();
  const log = [];
  const slow = (tag, ms) => async () => {
    log.push(`start ${tag}`);
    await new Promise((r) => setTimeout(r, ms));
    log.push(`end ${tag}`);
    return tag;
  };
  const results = await Promise.all([run(slow('a', 30)), run(slow('b', 5)), run(slow('c', 1))]);
  assert.deepEqual(results, ['a', 'b', 'c']);
  assert.deepEqual(log, ['start a', 'end a', 'start b', 'end b', 'start c', 'end c']);
});

test('createSerialQueue：前一个任务失败不影响后一个任务', async () => {
  const run = createSerialQueue();
  const failed = run(async () => {
    throw new Error('boom');
  });
  await assert.rejects(failed, /boom/);
  assert.equal(await run(async () => 'ok'), 'ok');
});

test('abortError：名字为 AbortError，界面会把它当作用户停止，且不被识别为 GPU 重置', () => {
  const e = abortError();
  assert.equal(e.name, 'AbortError');
  assert.equal(isGpuResetError(e), false);
});

test('createInflightGate：生成结束后 pending 清空，等待者立即继续', async () => {
  const g = createInflightGate();
  assert.equal(g.pending, null);
  const token = g.begin();
  assert.ok(g.pending);
  let resumed = false;
  const waiter = g.pending.then(() => {
    resumed = true;
  });
  g.end(token);
  await waiter;
  assert.equal(resumed, true);
  assert.equal(g.pending, null);
});

test('createInflightGate：引擎被丢弃（releaseAll）时，即使生成从未结束，等待者也能继续', async () => {
  const g = createInflightGate();
  g.begin(); // 模拟：worker 被终止，生成的 Promise 永远不会 end
  const waiter = g.pending;
  g.releaseAll();
  await waiter; // 不能挂起
  assert.equal(g.pending, null);
});

test('createInflightGate：先被丢弃的生成，之后再 end 不会误清除新一轮的生成', () => {
  const g = createInflightGate();
  const old = g.begin();
  g.releaseAll();
  const fresh = g.begin();
  g.end(old); // 旧令牌结束：不应清除新令牌
  assert.equal(g.pending === null, false);
  g.end(fresh);
  assert.equal(g.pending, null);
});
