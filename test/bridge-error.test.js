import test from 'node:test';
import assert from 'node:assert/strict';
import { bridgeErrorMessage, bridgeErrorReport, bridgeModelDiagnostics, mapBridgeParams, startLocalModelBridge } from '../web/src/local-model-bridge.js';
import { assertModelRecord } from '../web/src/engines/webllm.js';

test('桥接错误说明：保留错误类型与真实原因，并带上模型 ID', () => {
  const gpu = new Error('device lost: GPUBuffer mapAsync failed');
  gpu.name = 'GpuResetError';
  assert.equal(
    bridgeErrorMessage(gpu, 'Qwen2.5-3B-Instruct-q4f16_1-MLC'),
    '本机模型生成失败（模型 Qwen2.5-3B-Instruct-q4f16_1-MLC）：GpuResetError: device lost: GPUBuffer mapAsync failed',
  );

  // DOMException 等带 name/message 的非 Error 对象
  assert.match(bridgeErrorMessage({ name: 'AbortError', message: '已停止' }, 'm1'), /AbortError: 已停止/);
  // 字符串错误
  assert.match(bridgeErrorMessage('模型尚未加载', 'm1'), /模型 m1.*模型尚未加载/);
  // 没有 message 的普通对象：序列化内容，而不是只剩一句通用提示
  assert.match(bridgeErrorMessage({ code: 'OOM', detail: 'out of memory' }, 'm1'), /OOM/);
  // 完全没有信息的错误也要落到可读文本
  assert.match(bridgeErrorMessage(undefined, 'm1'), /本机模型生成失败（模型 m1）：/);
  // 超长原因会被截断，避免把整段堆栈塞进 API 响应
  const long = bridgeErrorMessage(new Error('x'.repeat(5000)), 'm1');
  assert.ok(long.length < 800);
});

test('桥接生成参数映射：支持的参数转成引擎选项，非法值明确报错而不是静默丢弃', () => {
  assert.deepEqual(mapBridgeParams(null), {});
  assert.deepEqual(mapBridgeParams({}), {});
  assert.deepEqual(
    mapBridgeParams({ top_p: 0.9, stop: '<end>', frequency_penalty: 0.5, seed: 7, response_format: { type: 'json_object' } }),
    { topP: 0.9, stop: ['<end>'], frequencyPenalty: 0.5, seed: 7, responseFormat: { type: 'json_object' } },
  );
  assert.throws(() => mapBridgeParams({ top_p: 3 }), /top_p 不合法/);
  assert.throws(() => mapBridgeParams({ frequency_penalty: -9 }), /frequency_penalty 不合法/);
  assert.throws(() => mapBridgeParams({ seed: 1.5 }), /seed 不合法/);
  assert.throws(() => mapBridgeParams({ response_format: { type: 'json_schema' } }), /仅支持 response_format/);
});

test('桥接结构化错误报告：阶段、cause、嵌套错误与怪对象都不能把真实原因弄丢', () => {
  const cause = Object.assign(new Error('VRAM allocation failed'), { code: 'OOM' });
  const outer = Object.assign(new Error('engine create failed'), { cause });
  const report = bridgeErrorReport(outer, {
    model: 'Huihui-Qwen3.5-4B-abliterated-q4f16_1-MLC-local',
    phase: 'load-model',
    readyNote: '输出自检未能完成（health failed）',
    deltaCount: 0,
    elapsedMs: 123,
    modelDiagnostics: { kind: 'local-import', originId: 'Huihui-Qwen3.5-4B-abliterated-q4f16_1-MLC' },
  });
  assert.equal(report.phase, 'load-model');
  assert.equal(report.protocolVersion, 2);
  assert.match(report.message, /阶段 加载模型/);
  assert.match(report.message, /engine create failed/);
  assert.match(report.message, /原因：VRAM allocation failed/);
  assert.match(report.message, /本地导入模型/);
  assert.match(report.message, /加载自检/);

  // WebLLM worker 经常把错误压成 { error: { message } } 或只带 code 的对象
  assert.match(bridgeErrorReport({ error: { name: 'MLCError', message: 'shader compile failed' } }, { model: 'm' }).message, /MLCError: shader compile failed/);
  assert.match(bridgeErrorReport({ code: 'quota', detail: 'cache full' }, { model: 'm' }).message, /code=quota/);

  // 循环对象与带抛错 getter 的对象：序列化函数本身绝不能再抛错，否则网页发不出 error 帧
  const cyclic = {};
  cyclic.self = cyclic;
  assert.match(bridgeErrorReport(cyclic, { model: 'm' }).message, /没有文字说明|self=/);
  const hostile = {};
  Object.defineProperty(hostile, 'message', { get() { throw new Error('getter exploded'); } });
  assert.doesNotThrow(() => bridgeErrorReport(hostile, { model: 'm' }));
  assert.match(bridgeErrorReport(hostile, { model: 'm' }).message, /没有文字说明|无法读取/);
  assert.match(bridgeErrorReport(42n, { model: 'm' }).message, /42n/);
});

test('桥接模型诊断与模型记录校验：本地导入来源可见，缺记录时明确报错', () => {
  const local = bridgeModelDiagnostics({ models: [], extraRecords: [] }, 'not-exist');
  assert.equal(local.kind, 'unknown');
  assert.throws(() => assertModelRecord(null, 'm1'), /找不到模型记录：m1/);
  assert.throws(() => assertModelRecord({ model: '', model_lib: 'x.wasm' }, 'm1'), /缺少权重地址/);
  assert.throws(() => assertModelRecord({ model: 'https://example.com/m', model_lib: '' }, 'm1'), /缺少运行库地址/);
  assert.equal(assertModelRecord({ model: 'https://example.com/m', model_lib: 'x.wasm' }, 'm1').model, 'https://example.com/m');
});

class FakeBridgeSocket {
  static OPEN = 1;
  static instances = [];

  constructor(url) {
    this.url = url;
    this.readyState = FakeBridgeSocket.OPEN;
    this.sent = [];
    FakeBridgeSocket.instances.push(this);
    queueMicrotask(() => this.onopen?.());
  }

  send(data) {
    this.sent.push(JSON.parse(data));
  }

  close() {
    this.readyState = 3;
    this.onclose?.({ code: 1000 });
  }

  receive(message) {
    this.onmessage?.({ data: JSON.stringify(message) });
  }
}

async function waitForFrame(socket, type, timeoutMs = 2000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const frame = socket.sent.find((item) => item.type === type);
    if (frame) return frame;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  throw new Error(`等待 ${type} 帧超时；已收到：${JSON.stringify(socket.sent)}`);
}

async function runFakeBridgeScenario(t, backend) {
  const previousWebSocket = globalThis.WebSocket;
  const previousLocation = globalThis.location;
  FakeBridgeSocket.instances = [];
  globalThis.WebSocket = FakeBridgeSocket;
  globalThis.location = { protocol: 'http:', host: '127.0.0.1:3000', origin: 'http://127.0.0.1:3000' };
  const bridge = startLocalModelBridge({
    webllmCfg: { models: [{ id: 'fake-model', label: 'Fake Model' }] },
    backendFactory: () => backend,
  });
  t.after(() => {
    bridge.stop();
    globalThis.WebSocket = previousWebSocket;
    globalThis.location = previousLocation;
  });
  const socket = FakeBridgeSocket.instances[0];
  await waitForFrame(socket, 'ready');
  return socket;
}

test('网页桥接全链路模拟：加载失败、生成失败、零输出都会带阶段回传结构化错误', async (t) => {
  const scenarios = [
    {
      name: '加载阶段 worker 字符串错误',
      backend: {
        setModel() {},
        async ensureReady() { throw 'Error: Failed to fetch model shard'; },
        async *stream() {},
      },
      phase: 'load-model',
      pattern: /Failed to fetch model shard/,
    },
    {
      name: '生成阶段嵌套对象错误',
      backend: {
        setModel() {},
        async ensureReady() { return { healthy: false, note: '输出自检异常（empty）' }; },
        async *stream() { throw { error: { name: 'MLCError', message: 'shader compile failed' } }; },
      },
      phase: 'generate',
      pattern: /MLCError: shader compile failed/,
    },
    {
      name: '生成正常结束但零输出',
      backend: {
        setModel() {},
        async ensureReady() { return { healthy: true }; },
        async *stream() {},
      },
      phase: 'generate',
      pattern: /EmptyModelOutputError: 模型生成结束但没有输出任何内容/,
    },
  ];

  for (const scenario of scenarios) {
    await t.test(scenario.name, async (st) => {
      const socket = await runFakeBridgeScenario(st, scenario.backend);
      socket.receive({ type: 'request', id: `req-${scenario.phase}`, payload: { model: 'fake-model', messages: [{ role: 'user', content: '你好' }] } });
      const frame = await waitForFrame(socket, 'error');
      assert.equal(frame.error.phase, scenario.phase);
      assert.equal(frame.error.protocolVersion, 2);
      assert.match(frame.message, scenario.pattern);
    });
  }
});

test('网页桥接全链路模拟：已输出片段后失败，错误报告带 deltaCount', async (t) => {
  const socket = await runFakeBridgeScenario(t, {
    setModel() {},
    async ensureReady() { return { healthy: true }; },
    async *stream() {
      yield '你好';
      const err = new Error('device lost');
      err.name = 'GpuResetError';
      throw err;
    },
  });
  socket.receive({ type: 'request', id: 'req-delta', payload: { model: 'fake-model', messages: [{ role: 'user', content: '你好' }] } });
  const delta = await waitForFrame(socket, 'delta');
  assert.equal(delta.content, '你好');
  const frame = await waitForFrame(socket, 'error');
  assert.equal(frame.error.deltaCount, 1);
  assert.match(frame.message, /GpuResetError: device lost/);
  assert.match(frame.message, /失败前已收到 1 个输出片段/);
});

test('网页桥接全链路模拟：本地导入模型缓存不完整时在加载前明确报错', async (t) => {
  const previousLocalStorage = globalThis.localStorage;
  const previousCaches = globalThis.caches;
  const store = new Map();
  const localEntry = {
    id: 'local-model',
    label: 'Local Model（导入）',
    originId: 'Huihui-Qwen3.5-4B-abliterated-q4f16_1-MLC',
    record: {
      model_id: 'local-model',
      model: 'https://local.oj-ai.invalid/models/local-model',
      model_lib: 'https://example.com/Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm',
    },
  };
  store.set('oj-ai.localModels', JSON.stringify([localEntry]));
  globalThis.localStorage = {
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => store.set(key, String(value)),
    removeItem: (key) => store.delete(key),
  };
  const emptyCache = { keys: async () => [], match: async () => null, put: async () => {}, delete: async () => false };
  globalThis.caches = { open: async () => emptyCache };
  t.after(() => {
    globalThis.localStorage = previousLocalStorage;
    globalThis.caches = previousCaches;
  });

  let ensureCalled = false;
  const socket = await runFakeBridgeScenario(t, {
    setModel() {},
    async ensureReady() { ensureCalled = true; },
    async *stream() {},
  });
  socket.receive({ type: 'request', id: 'req-local', payload: { model: 'local-model', messages: [{ role: 'user', content: '你好' }] } });
  const frame = await waitForFrame(socket, 'error');
  assert.equal(frame.error.phase, 'check-local-model');
  assert.match(frame.message, /模型尚未完整缓存/);
  assert.match(frame.message, /本地导入模型/);
  assert.equal(ensureCalled, false, '缓存不完整时不应启动 GPU 加载');
});

function setupLocalModelGlobals(t, cacheStorage) {
  const previousLocalStorage = globalThis.localStorage;
  const previousCaches = globalThis.caches;
  const store = new Map();
  const localEntry = {
    id: 'local-model',
    label: 'Local Model（导入）',
    originId: 'Huihui-Qwen3.5-4B-abliterated-q4f16_1-MLC',
    record: {
      model_id: 'local-model',
      model: 'https://local.oj-ai.invalid/models/local-model',
      model_lib: 'https://example.com/Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm',
    },
  };
  store.set('oj-ai.localModels', JSON.stringify([localEntry]));
  globalThis.localStorage = {
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => store.set(key, String(value)),
    removeItem: (key) => store.delete(key),
  };
  globalThis.caches = cacheStorage;
  t.after(() => {
    globalThis.localStorage = previousLocalStorage;
    globalThis.caches = previousCaches;
  });
}

test('网页桥接：本地模型缓存完整性同一登记只全量检查一次，后续请求不再枚举缓存', async (t) => {
  const base = 'https://local.oj-ai.invalid/models/local-model/resolve/main/';
  const wasmUrl = 'https://example.com/Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm';
  let keysCalls = 0;
  const completeCache = (scope) => ({
    keys: async () => {
      keysCalls += 1;
      if (scope === 'webllm/wasm') return [{ url: wasmUrl }];
      return [
        { url: `${base}tensor-cache.json` },
        { url: `${base}tokenizer.json` },
        { url: `${base}mlc-chat-config.json` },
      ];
    },
    match: async () => ({ json: async () => ({ records: [] }) }),
    put: async () => {},
    delete: async () => false,
  });
  setupLocalModelGlobals(t, { open: async (scope) => completeCache(scope) });

  const socket = await runFakeBridgeScenario(t, {
    setModel() {},
    async ensureReady() { return { healthy: true }; },
    async *stream() { yield '你好'; },
  });
  for (const id of ['req-a', 'req-b']) {
    const sentBefore = socket.sent.length;
    socket.receive({ type: 'request', id, payload: { model: 'local-model', messages: [{ role: 'user', content: '你好' }] } });
    const deadline = Date.now() + 2000;
    let done = null;
    while (Date.now() < deadline && !done) {
      done = socket.sent.slice(sentBefore).find((f) => f.type === 'complete' && f.id === id);
      if (!done) await new Promise((resolve) => setTimeout(resolve, 5));
    }
    assert.ok(done, `请求 ${id} 应完成`);
  }
  assert.equal(keysCalls, 3, '两次请求只应做一次三作用域的缓存枚举');
});

test('网页桥接：引擎已就绪时跳过本地缓存检查，缓存为空也不误报', async (t) => {
  const emptyCache = { keys: async () => [], match: async () => null, put: async () => {}, delete: async () => false };
  setupLocalModelGlobals(t, { open: async () => emptyCache });
  const socket = await runFakeBridgeScenario(t, {
    setModel() {},
    isReady: () => true,
    async ensureReady() { return { healthy: true }; },
    async *stream() { yield '你好'; },
  });
  socket.receive({ type: 'request', id: 'req-ready', payload: { model: 'local-model', messages: [{ role: 'user', content: '你好' }] } });
  const frame = await waitForFrame(socket, 'complete');
  assert.equal(frame.id, 'req-ready');
});

test('foldSystemMessages：调用方自带的 system 折叠进开头唯一一条，不再触发模板顺序错误', async () => {
  const { foldSystemMessages } = await import('../web/src/engines/backend.js');
  // 典型触发场景：本站 system + 软件自带 system + user（原样发给 WebLLM 会抛 SystemMessageOrderError）
  const a = foldSystemMessages('本站系统提示', [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: '你好' },
  ]);
  assert.equal(a.rest.length, 1);
  assert.equal(a.rest[0].role, 'user');
  assert.ok(a.systemContent.startsWith('本站系统提示'));
  assert.ok(a.systemContent.includes('You are a helpful assistant.'));

  // system 夹在历史中间、多条 system：全部折叠，其余消息顺序不变
  const b = foldSystemMessages('S0', [
    { role: 'user', content: 'u1' },
    { role: 'system', content: 'S1' },
    { role: 'assistant', content: 'a1' },
    { role: 'system', content: 'S2' },
    { role: 'user', content: 'u2' },
  ]);
  assert.deepEqual(b.rest.map((m) => m.content), ['u1', 'a1', 'u2']);
  assert.ok(b.systemContent.indexOf('S1') < b.systemContent.indexOf('S2'));

  // 已包含的片段不重复添加；空白 system 忽略；无 system 时原样返回
  const c = foldSystemMessages('S0\n\nS1', [
    { role: 'system', content: 'S1' },
    { role: 'system', content: '   ' },
    { role: 'user', content: 'hi' },
  ]);
  assert.equal(c.systemContent, 'S0\n\nS1');
  const d = foldSystemMessages('S0', [{ role: 'user', content: 'hi' }]);
  assert.equal(d.systemContent, 'S0');
  assert.equal(d.rest.length, 1);
});
