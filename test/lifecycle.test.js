// 推理引擎生命周期：GPU 重置错误识别、串行队列。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  abortError,
  createSerialQueue,
  createStallGuard,
  GPU_RESET_MESSAGE,
  gpuResetError,
  isGpuResetError,
  isTimeoutError,
  iterateWithTimeout,
  TimeoutError,
  withTimeout,
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

test('withTimeout：及时完成则返回结果，超时则 reject TimeoutError', async () => {
  assert.equal(await withTimeout(Promise.resolve(7), 50, '超时'), 7);
  await assert.rejects(withTimeout(new Promise(() => {}), 10, '太慢了'), (err) => {
    assert.ok(err instanceof TimeoutError);
    assert.equal(isTimeoutError(err), true);
    assert.equal(err.message, '太慢了');
    return true;
  });
});

test('abortError：名字为 AbortError，界面会把它当作用户停止', () => {
  const e = abortError();
  assert.equal(e.name, 'AbortError');
  assert.equal(isTimeoutError(e), false);
  assert.equal(isGpuResetError(e), false);
});

test('createStallGuard：长时间没有 touch 才触发，只触发一次', async () => {
  let fired = 0;
  const g = createStallGuard({ stallMs: 30, checkMs: 5, message: '卡住', onStall: () => fired++ });
  await new Promise((r) => setTimeout(r, 80));
  g.stop();
  assert.equal(fired, 1);
  await new Promise((r) => setTimeout(r, 40));
  assert.equal(fired, 1);
});

test('createStallGuard：持续 touch 时不触发；stop 之后不触发', async () => {
  let fired = 0;
  const g = createStallGuard({ stallMs: 40, checkMs: 5, message: '卡住', onStall: () => fired++ });
  for (let i = 0; i < 10; i++) {
    g.touch();
    await new Promise((r) => setTimeout(r, 10));
  }
  assert.equal(fired, 0);
  g.stop();
  await new Promise((r) => setTimeout(r, 80));
  assert.equal(fired, 0);
});

test('iterateWithTimeout：正常输出全部内容', async () => {
  async function* gen() {
    yield 'a';
    yield 'b';
  }
  const out = [];
  for await (const v of iterateWithTimeout(gen(), { firstMs: 50, idleMs: 50, firstMessage: 'x', idleMessage: 'y' })) {
    out.push(v);
  }
  assert.deepEqual(out, ['a', 'b']);
});

test('iterateWithTimeout：第一块卡住时使用 firstMessage，并先调用 onTimeout', async () => {
  async function* stalled() {
    await new Promise(() => {});
    yield 'never';
  }
  let timedOut = null;
  await assert.rejects(
    (async () => {
      for await (const _ of iterateWithTimeout(stalled(), {
        firstMs: 20,
        idleMs: 1000,
        firstMessage: '首块超时',
        idleMessage: '中途超时',
        onTimeout: (e) => {
          timedOut = e;
        },
      })) {
        // 不应该收到任何输出
      }
    })(),
    (err) => err.message === '首块超时' && isTimeoutError(err),
  );
  assert.ok(timedOut);
});

test('iterateWithTimeout：中途卡住时使用 idleMessage', async () => {
  async function* partly() {
    yield 'first';
    await new Promise(() => {});
  }
  const got = [];
  await assert.rejects(
    (async () => {
      for await (const v of iterateWithTimeout(partly(), {
        firstMs: 1000,
        idleMs: 20,
        firstMessage: '首块超时',
        idleMessage: '中途超时',
      })) {
        got.push(v);
      }
    })(),
    (err) => err.message === '中途超时' && isTimeoutError(err),
  );
  assert.deepEqual(got, ['first']);
});
