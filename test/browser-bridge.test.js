import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createServer, request as httpRequest } from 'node:http';
import { execFile } from 'node:child_process';
import WebSocket from 'ws';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { BrowserModelBridge } from '../server/browser-model-bridge.js';

const PASSWORD = 'correct-horse-battery';

let lastFakeBridge = null;

function fakeBridge({ connected = true, models = [{ id: 'test-model', label: 'Test Model' }], silent = false } = {}) {
  const bridge = {
    models,
    connected,
    silent,
    lastPayload: null,
    status() {
      return { connected: this.connected, models: this.models, queued: 0 };
    },
    async run(payload, { onDelta } = {}) {
      this.lastPayload = payload;
      if (!this.silent) {
        onDelta?.('你好');
        onDelta?.('，世界');
      }
      return { finishReason: 'stop' };
    },
  };
  lastFakeBridge = bridge;
  return bridge;
}

async function setup(t, { withBridge = true, bridgeConnected = true, bridgeModels, bridgeSilent = false } = {}) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-ai-bridge-'));
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
  });
  const app = createApp(cfg, { browserBridge: withBridge ? fakeBridge({ connected: bridgeConnected, models: bridgeModels, silent: bridgeSilent }) : null });
  const server = await new Promise((resolve) => {
    const srv = app.listen(0, '127.0.0.1', () => resolve(srv));
  });
  t.after(async () => {
    await new Promise((r) => server.close(r));
    await rm(dir, { recursive: true, force: true });
  });
  return `http://127.0.0.1:${server.address().port}`;
}

test('桥接模式：/api/config 暴露 localBrowserBridge，状态接口返回模型列表', async (t) => {
  const base = await setup(t);
  const cfgRes = await fetch(`${base}/api/config`);
  const cfg = await cfgRes.json();
  assert.equal(cfg.localBrowserBridge, true);

  const statusRes = await fetch(`${base}/api/local-model-bridge/status`);
  const status = await statusRes.json();
  assert.equal(status.connected, true);
  assert.deepEqual(status.models, [{ id: 'test-model', label: 'Test Model' }]);
});

test('桥接模式：网页可自助申请 API Key，且申请的 Key 可调用 /v1', async (t) => {
  const base = await setup(t);
  const applyRes = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: '网页自助申请' }),
  });
  assert.equal(applyRes.status, 201);
  const created = await applyRes.json();
  assert.ok(created.key.startsWith('ojai_'));

  const modelsRes = await fetch(`${base}/v1/models`, {
    headers: { authorization: `Bearer ${created.key}` },
  });
  assert.equal(modelsRes.status, 200);
  const models = await modelsRes.json();
  assert.equal(models.data[0].id, 'test-model');

  const chatRes = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${created.key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(chatRes.status, 200);
  const chat = await chatRes.json();
  assert.equal(chat.object, 'chat.completion');
  assert.equal(chat.choices[0].message.content, '你好，世界');
  // 调用方未指定时不再用服务端默认值压住本机参数：temperature/maxTokens 不传，只给硬上限
  assert.equal(lastFakeBridge.lastPayload.temperature, undefined);
  assert.equal(lastFakeBridge.lastPayload.maxTokens, undefined);
  assert.equal(lastFakeBridge.lastPayload.hardMaxTokens, 8192);
});

test('桥接模式：CSP 放行 WebAssembly 与模型下载域名，网页模型才能加载', async (t) => {
  const base = await setup(t);
  const res = await fetch(`${base}/healthz`);
  const csp = res.headers.get('content-security-policy') || '';
  assert.match(csp, /'wasm-unsafe-eval'/);
  assert.match(csp, /https:\/\/huggingface\.co/);
  assert.match(csp, /https:\/\/\*\.hf\.co/);
});

test('桥接模式：网页未连接时聊天请求返回 503', async (t) => {
  const base = await setup(t, { bridgeConnected: false });
  const applyRes = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: 'offline-test' }),
  });
  const { key } = await applyRes.json();

  const chatRes = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(chatRes.status, 503);
  const chat = await chatRes.json();
  assert.match(chat.error.message, /尚未连接/);
});

test('桥接模式：流式请求返回 SSE 片段', async (t) => {
  const base = await setup(t);
  const applyRes = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: 'stream-test' }),
  });
  const { key } = await applyRes.json();

  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: '你好' }], stream: true }),
  });
  assert.equal(res.status, 200);
  const text = await res.text();
  assert.match(text, /data: \{/);
  assert.match(text, /你好/);
  assert.match(text, /\[DONE\]/);
});

test('真实 WebSocket 桥接：网页客户端上报模型后 API 请求经桥接返回', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-ai-bridge-ws-'));
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
  });
  const bridge = new BrowserModelBridge();
  const server = createServer(createApp(cfg, { browserBridge: bridge }));
  bridge.attach(server);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => {
    await new Promise((r) => server.close(r));
    await rm(dir, { recursive: true, force: true });
  });

  const ws = new WebSocket(`ws://127.0.0.1:${server.address().port}/api/local-model-bridge`, {
    headers: { origin: base },
  });
  t.after(() => ws.close());
  await new Promise((resolve, reject) => {
    ws.once('open', resolve);
    ws.once('error', reject);
  });
  ws.send(JSON.stringify({ type: 'ready', models: [{ id: 'ws-model', label: 'WS Model' }] }));
  // 等待服务端记录模型列表
  await new Promise((resolve) => setTimeout(resolve, 100));
  assert.equal(bridge.status().connected, true);
  assert.equal(bridge.status().models[0].id, 'ws-model');

  ws.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') {
      ws.send(JSON.stringify({ type: 'delta', id: msg.id, content: '桥接' }));
      ws.send(JSON.stringify({ type: 'delta', id: msg.id, content: '成功' }));
      ws.send(JSON.stringify({ type: 'complete', id: msg.id, finishReason: 'stop' }));
    }
  });

  const applyRes = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: 'ws-test' }),
  });
  const { key } = await applyRes.json();

  const chatRes = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'ws-model', messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(chatRes.status, 200);
  const chat = await chatRes.json();
  assert.equal(chat.choices[0].message.content, '桥接成功');
  ws.close();
});

test('真实 WebSocket 桥接：网页端回传的生成失败原因要原样到达 API 调用方', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-ai-bridge-'));
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
  });
  const bridge = new BrowserModelBridge();
  const server = createServer(createApp(cfg, { browserBridge: bridge }));
  bridge.attach(server);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => {
    await new Promise((r) => server.close(r));
    await rm(dir, { recursive: true, force: true });
  });

  const ws = new WebSocket(`ws://127.0.0.1:${server.address().port}/api/local-model-bridge`, {
    headers: { origin: base },
  });
  t.after(() => ws.close());
  await new Promise((resolve, reject) => {
    ws.once('open', resolve);
    ws.once('error', reject);
  });
  ws.send(JSON.stringify({ type: 'ready', models: [{ id: 'ws-model', label: 'WS Model' }] }));
  await new Promise((resolve) => setTimeout(resolve, 100));
  ws.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') {
      ws.send(JSON.stringify({
        type: 'error',
        id: msg.id,
        message: '本机模型生成失败（模型 ws-model）：GpuResetError: device lost',
      }));
    }
  });

  const applyRes = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: 'ws-error-test' }),
  });
  const { key } = await applyRes.json();
  const chatRes = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'ws-model', messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(chatRes.status, 503);
  const data = await chatRes.json();
  assert.match(data.error.message, /GpuResetError: device lost/, '调用方应看到网页端的真实失败原因，而不是通用提示');
  ws.close();
});

test('桥接模式：网页自身聊天 /api/chat 也经桥接返回流式回答', async (t) => {
  const base = await setup(t);
  // 入口登录已删除：网页聊天免登录
  const chatRes = await fetch(`${base}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(chatRes.status, 200);
  const text = await chatRes.text();
  assert.match(text, /data: \{/);
  assert.match(text, /你好/);
  assert.match(text, /\[DONE\]/);
  assert.equal(lastFakeBridge.lastPayload.model, 'test-model');
});

test('桥接模式：网页聊天与 API 可各自选择不同模型，互不影响', async (t) => {
  const base = await setup(t, {
    bridgeModels: [
      { id: 'test-model', label: 'Test Model' },
      { id: 'second-model', label: 'Second Model' },
    ],
  });
  const webChat = (body) => fetch(`${base}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });

  // 网页聊天指定第二个模型
  const picked = await webChat({ messages: [{ role: 'user', content: '你好' }], model: 'second-model' });
  assert.equal(picked.status, 200);
  await picked.text();
  assert.equal(lastFakeBridge.lastPayload.model, 'second-model');

  // API 用 Key 指定第一个模型
  const applyRes = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: 'pick-model' }),
  });
  const { key } = await applyRes.json();
  // API 的模型列表包含全部可用模型（含导入 AI 在桥接侧上报的条目），逐个都可调用
  const modelsRes = await fetch(`${base}/v1/models`, { headers: { authorization: `Bearer ${key}` } });
  assert.equal(modelsRes.status, 200);
  const modelsData = await modelsRes.json();
  assert.deepEqual(modelsData.data.map((m) => m.id), ['test-model', 'second-model']);
  const apiRes = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'second-model', messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(apiRes.status, 200);
  await apiRes.json();
  assert.equal(lastFakeBridge.lastPayload.model, 'second-model');

  // 网页聊天不带 model 时仍回到默认第一个，不被 API 上次的选择带偏
  const fallback = await webChat({ messages: [{ role: 'user', content: '你好' }] });
  assert.equal(fallback.status, 200);
  await fallback.text();
  assert.equal(lastFakeBridge.lastPayload.model, 'test-model');

  // 指定不存在的模型：明确 404，而不是默默换一个模型生成
  const bogus = await webChat({ messages: [{ role: 'user', content: '你好' }], model: 'no-such-model' });
  assert.equal(bogus.status, 404);
});

test('网页聊天 streamChat：带 model 时逐条请求指定模型，不带时不发送 model 字段', async () => {
  const { streamChat } = await import('../web/src/api.js');
  const sse = 'data: {"choices":[{"delta":{"content":"A"}}]}\n\ndata: {"choices":[{"delta":{"content":"B"}}]}\n\ndata: [DONE]\n\n';
  const realFetch = globalThis.fetch;
  let captured = null;
  globalThis.fetch = async (_url, init) => {
    captured = JSON.parse(init.body);
    return {
      ok: true,
      body: new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(sse));
          controller.close();
        },
      }),
    };
  };
  try {
    const parts = [];
    for await (const delta of streamChat({ messages: [{ role: 'user', content: 'hi' }], model: 'second-model' })) parts.push(delta);
    assert.equal(parts.join(''), 'AB');
    assert.equal(captured.model, 'second-model');
    for await (const delta of streamChat({ messages: [{ role: 'user', content: 'hi' }] })) parts.push(delta);
    assert.ok(!('model' in captured), '未指定模型时不应发送 model 字段');
  } finally {
    globalThis.fetch = realFetch;
  }
});

test('非桥接模式：自助申请与桥接状态接口返回 404', async (t) => {
  const base = await setup(t, { withBridge: false });
  const cfg = await (await fetch(`${base}/api/config`)).json();
  assert.equal(cfg.localBrowserBridge, false);

  const applyRes = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: 'x' }),
  });
  assert.equal(applyRes.status, 404);

  const statusRes = await fetch(`${base}/api/local-model-bridge/status`);
  assert.equal(statusRes.status, 404);
});

// ---- 安全与健壮性修复回归 ----

function rawRequest(base, { method = 'POST', path: reqPath, headers = {}, body = '' }) {
  const url = new URL(base);
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      { hostname: url.hostname, port: url.port, path: reqPath, method, headers },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, text: Buffer.concat(chunks).toString('utf8') }));
      },
    );
    req.on('error', reject);
    req.end(body);
  });
}

async function applyKey(base) {
  const res = await fetch(`${base}/api/api-keys/apply`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ label: 'hardening-test' }),
  });
  assert.equal(res.status, 201);
  return (await res.json()).key;
}

test('桥接模式：申请 API Key 只允许同源，跨源网站（哪怕在白名单里）不能替用户申请', async (t) => {
  const base = await setup(t);
  const cross = await rawRequest(base, {
    path: '/api/api-keys/apply',
    headers: { 'content-type': 'application/json', origin: 'https://pd.geneluo.fun' },
    body: JSON.stringify({ label: 'cross-site' }),
  });
  assert.equal(cross.status, 403);
  assert.match(cross.text, /只允许在本机网页中申请/);

  const sameOrigin = await rawRequest(base, {
    path: '/api/api-keys/apply',
    headers: { 'content-type': 'application/json', origin: base },
    body: JSON.stringify({ label: 'same-site' }),
  });
  assert.equal(sameOrigin.status, 201);

  // 跨域许可只发给 /v1：白名单来源对 /api/* 预检拿不到 Access-Control-Allow-Origin
  const apiPreflight = await rawRequest(base, {
    method: 'OPTIONS',
    path: '/api/api-keys/apply',
    headers: { origin: 'https://pd.geneluo.fun', 'access-control-request-method': 'POST' },
  });
  assert.equal(apiPreflight.headers['access-control-allow-origin'], undefined);
  const v1Preflight = await rawRequest(base, {
    method: 'OPTIONS',
    path: '/v1/models',
    headers: { origin: 'https://pd.geneluo.fun', 'access-control-request-method': 'GET' },
  });
  assert.ok(v1Preflight.headers['access-control-allow-origin'], '/v1 预检应获得跨域许可');
});

test('桥接模式：不支持的生成参数明确拒绝（400），支持的参数与显式采样值透传', async (t) => {
  const base = await setup(t);
  const key = await applyKey(base);

  const bad = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'test-model',
      messages: [{ role: 'user', content: '你好' }],
      n: 2,
    }),
  });
  assert.equal(bad.status, 400);
  assert.match((await bad.json()).error.message, /不支持生成参数：n/);

  const ok = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'test-model',
      messages: [{ role: 'user', content: '你好' }],
      temperature: 0.7,
      max_tokens: 128,
      top_p: 0.9,
      stop: ['<end>'],
    }),
  });
  assert.equal(ok.status, 200);
  await ok.json();
  assert.equal(lastFakeBridge.lastPayload.temperature, 0.7, '调用方显式温度应透传');
  assert.equal(lastFakeBridge.lastPayload.maxTokens, 128);
  assert.equal(lastFakeBridge.lastPayload.hardMaxTokens, 128);
  assert.equal(lastFakeBridge.lastPayload.params.top_p, 0.9);
  assert.deepEqual(lastFakeBridge.lastPayload.params.stop, ['<end>']);
  assert.ok(lastFakeBridge.lastPayload.systemPrompt, '服务端系统提示词应随桥接请求下发');
});

async function setupRealBridge(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-ai-bridge-hardening-'));
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
  });
  const bridge = new BrowserModelBridge();
  const server = createServer(createApp(cfg, { browserBridge: bridge }));
  bridge.attach(server);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  const sockets = new Set();
  t.after(async () => {
    // 先终止所有桥接连接再关服务，避免 server.close 等连接拖住测试收尾
    for (const ws of sockets) {
      try { ws.terminate(); } catch {}
    }
    await new Promise((r) => server.close(r));
    await rm(dir, { recursive: true, force: true });
  });
  let connectCount = 0;
  const connect = async ({ protocolVersion = 0, models = [{ id: 'ws-model', label: 'WS Model' }] } = {}) => {
    connectCount += 1;
    const tabId = `tab-${protocolVersion}-${connectCount}`;
    const ws = new WebSocket(`ws://127.0.0.1:${server.address().port}/api/local-model-bridge`, {
      headers: { origin: base },
    });
    sockets.add(ws);
    ws.once('close', () => sockets.delete(ws));
    await new Promise((resolve, reject) => {
      ws.once('open', resolve);
      ws.once('error', reject);
    });
    ws.send(JSON.stringify({
      type: 'ready',
      ...(protocolVersion ? { protocolVersion, clientVersion: `bridge-v${protocolVersion}`, tabId } : {}),
      models,
    }));
    await new Promise((resolve) => setTimeout(resolve, 50));
    return ws;
  };
  return { bridge, base, connect };
}

test('真实 WebSocket 桥接：畸形消息（null）不能打崩服务，之后请求照常', async (t) => {
  const { bridge, connect } = await setupRealBridge(t);
  const ws = await connect();
  t.after(() => ws.close());
  ws.send('null');
  ws.send('{"type":123}');
  ws.send('not-json');
  await new Promise((resolve) => setTimeout(resolve, 100));
  assert.equal(bridge.status().connected, true, '畸形消息后连接应保持可用');

  const done = new Promise((resolve, reject) => {
    ws.on('message', (raw) => {
      const msg = JSON.parse(String(raw));
      if (msg.type === 'request') {
        ws.send(JSON.stringify({ type: 'complete', id: msg.id, finishReason: 'stop' }));
        resolve();
      }
    });
    bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'hi' }] }, {}).then(
      (v) => assert.equal(v.finishReason, 'stop'),
      reject,
    );
  });
  await done;
  ws.close();
});

test('真实 WebSocket 桥接：连接被新网页接管时，在途请求立即失败而不是挂到超时', async (t) => {
  const { bridge, connect } = await setupRealBridge(t);
  const ws1 = await connect();
  t.after(() => { try { ws1.close(); } catch {} });

  const requestSeen = new Promise((resolve) => {
    ws1.on('message', (raw) => {
      const msg = JSON.parse(String(raw));
      if (msg.type === 'request') resolve(msg.id); // 故意不应答，让请求保持在途
    });
  });
  const pending = bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'hi' }] }, {});
  // 立刻挂上处理函数：接管发生在 await 期间，晚挂会被判为未处理的拒绝
  const settled = pending.then(() => 'resolved', (err) => err);
  await requestSeen;

  const ws2 = await connect(); // 新标签页接管连接
  t.after(() => { try { ws2.close(); } catch {} });

  const outcome = await Promise.race([
    settled,
    new Promise((resolve) => setTimeout(() => resolve('still-pending'), 2000)),
  ]);
  assert.notEqual(outcome, 'still-pending', '在途请求不能挂到 10 分钟超时');
  assert.notEqual(outcome, 'resolved');
  assert.match(outcome.message, /接管/);
  assert.equal(outcome.status, 503);
});

test('真实 WebSocket 桥接：旧版网页只回通用文案时，服务端明说它是旧版网页并给出处理办法', async (t) => {
  const { bridge, connect } = await setupRealBridge(t);
  const ws = await connect();
  t.after(() => ws.close());
  ws.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') {
      ws.send(JSON.stringify({ type: 'error', id: msg.id, message: '本机模型生成失败' }));
    }
  });
  await assert.rejects(
    bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'hi' }] }, {}),
    /未上报诊断信息的旧版模型网页.*关闭所有旧的服务网页标签页/,
  );
  assert.equal(bridge.status().clientProtocolVersion, 0);
  assert.match(bridge.status().lastError.message, /旧版模型网页/);
  ws.close();
});

test('桥接模式绑定非回环地址时服务拒绝启动（除非显式 OJ_ALLOW_PUBLIC_BRIDGE=1）', async () => {
  const env = {
    ...process.env,
    LOCAL_BROWSER_BRIDGE: '1',
    HOST: '0.0.0.0',
    PORT: '0',
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(tmpdir(), 'oj-bridge-keys.json'),
    API_SETTINGS_FILE: path.join(tmpdir(), 'oj-bridge-settings.json'),
  };
  const result = await new Promise((resolve) => {
    execFile(process.execPath, ['server/index.js'], { cwd: path.resolve('.'), env, timeout: 15000 }, (err, stdout, stderr) => {
      resolve({ code: err?.code ?? 0, stdout, stderr });
    });
  });
  assert.notEqual(result.code, 0);
  assert.match(result.stderr, /只能绑定本机回环地址/);
});

test('桥接模式：模型零输出不能算成功（/v1 非流式返回 502）', async (t) => {
  const base = await setup(t, { bridgeSilent: true });
  const key = await applyKey(base);
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(res.status, 502);
  assert.match((await res.json()).error.message, /没有输出任何内容/);
});

test('真实 WebSocket 桥接：v2 网页的结构化错误原样保留，并进入 status.lastError', async (t) => {
  const { bridge, connect } = await setupRealBridge(t);
  const ws = await connect({ protocolVersion: 2 });
  t.after(() => ws.close());
  ws.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') {
      ws.send(JSON.stringify({
        type: 'error',
        id: msg.id,
        message: '本机模型生成失败（模型 ws-model，阶段 加载模型）：Error: VRAM allocation failed；原因：OOM',
        error: {
          message: '本机模型生成失败（模型 ws-model，阶段 加载模型）：Error: VRAM allocation failed；原因：OOM',
          model: 'ws-model',
          phase: 'load-model',
          phaseLabel: '加载模型',
          name: 'Error',
          code: 'OOM',
          detail: 'VRAM allocation failed',
          protocolVersion: 2,
        },
      }));
    }
  });
  await assert.rejects(
    bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'hi' }] }, {}),
    (err) => {
      assert.equal(err.status, 503);
      assert.equal(err.bridgeReport.phase, 'load-model');
      assert.equal(err.bridgeReport.code, 'OOM');
      return true;
    },
  );
  const status = bridge.status();
  assert.equal(status.clientProtocolVersion, 2);
  assert.equal(status.lastError.phase, 'load-model');
  assert.equal(status.lastError.code, 'OOM');
  ws.close();
});

test('真实 WebSocket 桥接：新标签接管不关旧标签，旧标签可在新标签关闭后自动恢复提供', async (t) => {
  const { bridge, connect } = await setupRealBridge(t);
  const ws1 = await connect({ protocolVersion: 2 });
  t.after(() => { try { ws1.close(); } catch {} });
  let respondOnWs1 = false;
  const firstRequestSeen = new Promise((resolve) => {
    ws1.on('message', (raw) => {
      const msg = JSON.parse(String(raw));
      if (msg.type !== 'request') return;
      resolve(msg.id);
      if (respondOnWs1) ws1.send(JSON.stringify({ type: 'complete', id: msg.id, finishReason: 'stop' }));
    });
  });
  const pending = bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'hi' }] }, {});
  const settled = pending.then(() => 'resolved', (err) => err);
  await firstRequestSeen;

  const ws2 = await connect({ protocolVersion: 2 });
  t.after(() => { try { ws2.close(); } catch {} });
  const outcome = await Promise.race([
    settled,
    new Promise((resolve) => setTimeout(() => resolve('still-pending'), 2000)),
  ]);
  assert.notEqual(outcome, 'still-pending');
  assert.match(outcome.message, /接管/);
  assert.equal(ws1.readyState, WebSocket.OPEN, '被接管的旧标签不应被关掉，否则它会重连抢占');

  ws2.close();
  await new Promise((resolve) => setTimeout(resolve, 100));
  respondOnWs1 = true;
  const result = await bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'again' }] }, {});
  assert.equal(result.finishReason, 'stop');
});

test('真实 WebSocket 桥接：旧版标签页不能抢走新版标签页，新版关闭后才由旧版兜底', async (t) => {
  const { bridge, connect } = await setupRealBridge(t);
  const wsNew = await connect({ protocolVersion: 2 });
  t.after(() => { try { wsNew.close(); } catch {} });
  let newHits = 0;
  wsNew.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') {
      newHits += 1;
      wsNew.send(JSON.stringify({ type: 'complete', id: msg.id, finishReason: 'stop' }));
    }
  });
  const wsOld = await connect();
  t.after(() => { try { wsOld.close(); } catch {} });
  let oldHits = 0;
  wsOld.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') {
      oldHits += 1;
      wsOld.send(JSON.stringify({ type: 'complete', id: msg.id, finishReason: 'stop' }));
    }
  });

  await bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'hi' }] }, {});
  assert.equal(newHits, 1);
  assert.equal(oldHits, 0, '旧版标签页后连上也不能成为提供方');

  wsNew.close();
  await new Promise((resolve) => setTimeout(resolve, 100));
  // 旧版标签页此前已被停放（连接被关），它按自身退避重连后才能兜底
  const wsOld2 = await connect();
  t.after(() => { try { wsOld2.close(); } catch {} });
  let old2Hits = 0;
  wsOld2.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') {
      old2Hits += 1;
      wsOld2.send(JSON.stringify({ type: 'complete', id: msg.id, finishReason: 'stop' }));
    }
  });
  await bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'fallback' }] }, {});
  assert.equal(old2Hits, 1, '新版关闭、旧版重连后才由旧版兜底提供');
});

test('消息校验：OpenAI 富文本分段 content 归一化为纯文本，图片段明确拒绝', async () => {
  const { validateMessages } = await import('../server/chat.js');
  const out = validateMessages([
    { role: 'system', content: [{ type: 'text', text: '规则' }] },
    { role: 'assistant', content: null },
    { role: 'user', content: [{ type: 'text', text: '你好' }, { type: 'input_text', text: '，请讲题' }] },
  ], { allowSystem: true });
  assert.deepEqual(out, [
    { role: 'system', content: '规则' },
    { role: 'user', content: '你好，请讲题' },
  ]);
  assert.throws(
    () => validateMessages([{ role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,x' } }] }], { allowSystem: true }),
    /仅支持文本消息/,
  );
  assert.throws(
    () => validateMessages([{ role: 'user', content: [{ type: 'text', text: '  ' }] }], { allowSystem: true }),
    /内容为空/,
  );
});

test('桥接模式：/v1 忽略 user/stream_options/metadata/n=1 等元数据字段，真正的生成参数照常拒绝', async (t) => {
  const base = await setup(t);
  const key = await applyKey(base);
  const ok = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'test-model',
      user: 'tester',
      stream_options: { include_usage: true },
      metadata: { scene: 'unit' },
      n: 1,
      messages: [{ role: 'user', content: [{ type: 'text', text: '你好' }] }],
    }),
  });
  assert.equal(ok.status, 200);
  assert.equal((await ok.json()).choices[0].message.content, '你好，世界');
  assert.deepEqual(lastFakeBridge.lastPayload.messages, [{ role: 'user', content: '你好' }]);
  assert.equal(lastFakeBridge.lastPayload.params.user, undefined);
  assert.equal(lastFakeBridge.lastPayload.params.stream_options, undefined);

  const bad = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: `Bearer ${key}`, 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'test-model',
      logprobs: true,
      messages: [{ role: 'user', content: '你好' }],
    }),
  });
  assert.equal(bad.status, 400);
  assert.match((await bad.json()).error.message, /不支持生成参数：logprobs/);
});

test('真实 WebSocket 桥接：新版在场时旧版标签页会被停放，新版关闭后旧版才可兜底', async (t) => {
  const { bridge, base, connect } = await setupRealBridge(t);
  const wsNew = await connect({ protocolVersion: 2 });
  wsNew.on('message', (raw) => {
    const msg = JSON.parse(String(raw));
    if (msg.type === 'request') wsNew.send(JSON.stringify({ type: 'complete', id: msg.id, finishReason: 'stop' }));
  });
  // 手动连旧版：先挂上 close 监听再发 ready，避免错过服务端的停放关闭事件
  const wsOld = new WebSocket(`${base.replace('http:', 'ws:')}/api/local-model-bridge`, {
    headers: { origin: base },
  });
  t.after(() => { try { wsOld.terminate(); } catch {} });
  const oldClosed = new Promise((resolve) => wsOld.once('close', (closeCode) => resolve(closeCode)));
  await new Promise((resolve, reject) => {
    wsOld.once('open', resolve);
    wsOld.once('error', reject);
  });
  wsOld.send(JSON.stringify({ type: 'ready', models: [{ id: 'ws-model', label: 'WS Model' }] }));
  const code = await Promise.race([oldClosed, new Promise((resolve) => setTimeout(() => resolve('timeout'), 2000))]);
  assert.equal(code, 1012, '旧版备用连接应被 1012 停放');
  assert.equal(bridge.status().latestProtocolVersion, 2);

  // 请求仍由新版提供
  const result = await bridge.run({ model: 'ws-model', messages: [{ role: 'user', content: 'hi' }] }, {});
  assert.equal(result.finishReason, 'stop');

  // 新版关闭后，重新连上的旧版可以兜底（哪怕它只回通用错误文案，也由服务端标注来源）
  wsNew.close();
  await new Promise((resolve) => setTimeout(resolve, 60));
  const wsOld2 = await connect();
  t.after(() => { try { wsOld2.close(); } catch {} });
  assert.equal(bridge.status().clientProtocolVersion, 0);
});
