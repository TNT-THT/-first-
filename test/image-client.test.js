// 生图客户端：worker 出错时的提示、运行库目录随请求传递、出错后的清理（使用假 worker，不需要浏览器）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createImageClient, workerErrorMessage } from '../web/src/image/client.js';

function fakeWorkerFactory() {
  const made = [];
  const create = () => {
    const w = {
      sent: [],
      terminated: false,
      postMessage(m) { this.sent.push(m); },
      terminate() { this.terminated = true; },
    };
    made.push(w);
    return w;
  };
  return { create, made };
}

test('worker 出错信息：空 message 时仍说明位置与排查方法', () => {
  const msg = workerErrorMessage({ message: '', filename: 'http://h/assets/sd15-worker.js', lineno: 12 }, 'http://h/assets/sd15-worker.js');
  assert.match(msg, /生图 worker 出错/);
  assert.match(msg, /位置 http:\/\/h\/assets\/sd15-worker\.js:12/);
  assert.match(msg, /脚本 http:\/\/h\/assets\/sd15-worker\.js/);
  assert.match(msg, /浏览器控制台/);
});

test('worker 出错信息：有 message 时原样包含', () => {
  const msg = workerErrorMessage({ message: 'boom' });
  assert.match(msg, /：boom/);
});

test('generate 把运行库目录与模型大小随消息传给 worker', async () => {
  const f = fakeWorkerFactory();
  const client = createImageClient({ createWorker: f.create });
  const p = client.generate({ prompt: 'x', steps: 4 }, { urls: { a: 'u' }, sizes: { a: 3 }, wasmBase: 'http://h/ort/' });
  const sent = f.made[0].sent[0];
  assert.equal(sent.type, 'generate');
  assert.equal(sent.wasmBase, 'http://h/ort/');
  assert.deepEqual(sent.sizes, { a: 3 });
  // 让请求结束，避免悬空的 promise
  f.made[0].onmessage({ data: { id: sent.id, type: 'result', width: 1, height: 1, rgba: new Uint8Array(4) } });
  const res = await p;
  assert.equal(res.width, 1);
});

test('worker 出错：挂起的请求以带诊断信息的错误结束，worker 被终止', async () => {
  const f = fakeWorkerFactory();
  const client = createImageClient({ createWorker: f.create, scriptUrl: 'http://h/sd15-worker.js' });
  const p = client.generate({ prompt: 'x' }, { urls: {}, sizes: {}, wasmBase: 'http://h/ort/' });
  f.made[0].onerror({ message: '', filename: 'http://h/sd15-worker.js', lineno: 1 });
  await assert.rejects(p, (err) => {
    assert.match(err.message, /生图 worker 出错/);
    assert.notEqual(err.message, '生图 worker 出错');
    return true;
  });
  assert.equal(f.made[0].terminated, true);
  assert.equal(client.running, false);
});

test('worker 消息无法读取：请求失败且 worker 被终止', async () => {
  const f = fakeWorkerFactory();
  const client = createImageClient({ createWorker: f.create });
  const p = client.generate({ prompt: 'x' }, { urls: {}, sizes: {}, wasmBase: 'http://h/ort/' });
  f.made[0].onmessageerror({});
  await assert.rejects(p, /无法读取其返回的消息/);
  assert.equal(f.made[0].terminated, true);
});

test('load（预加载）：只发送 load 消息（不含提示词），收到 loaded 后结束；进度回调随 id 送达', async () => {
  const f = fakeWorkerFactory();
  const client = createImageClient({ createWorker: f.create });
  const progress = [];
  const p = client.load({ urls: { textEncoder: 'u' }, sizes: { textEncoder: 3 }, wasmBase: 'http://h/ort/', onProgress: (m) => progress.push(m.message) });
  const w = f.made[0];
  const sent = w.sent[0];
  assert.equal(sent.type, 'load');
  assert.equal(sent.request, undefined);
  w.onmessage({ data: { id: sent.id, type: 'progress', message: '读取中' } });
  w.onmessage({ data: { id: sent.id, type: 'loaded' } });
  assert.deepEqual(await p, { id: sent.id, type: 'loaded' });
  assert.deepEqual(progress, ['读取中']);
});

test('load 被 dispose（切换模型、离开生图模式）时以 AbortError 结束，不留下待处理请求', async () => {
  const f = fakeWorkerFactory();
  const client = createImageClient({ createWorker: f.create });
  const p = client.load({ urls: { textEncoder: 'u' }, sizes: {} });
  assert.equal(client.running, true);
  client.dispose();
  await assert.rejects(p, (err) => err.name === 'AbortError');
  assert.equal(client.running, false);
  assert.equal(f.made[0].terminated, true);
});
