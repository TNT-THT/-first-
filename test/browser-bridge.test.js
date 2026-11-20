import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createServer } from 'node:http';
import WebSocket from 'ws';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { BrowserModelBridge } from '../server/browser-model-bridge.js';

const PASSWORD = 'correct-horse-battery';

let lastFakeBridge = null;

function fakeBridge({ connected = true, models = [{ id: 'test-model', label: 'Test Model' }] } = {}) {
  const bridge = {
    models,
    connected,
    lastPayload: null,
    status() {
      return { connected: this.connected, models: this.models, queued: 0 };
    },
    async run(payload, { onDelta } = {}) {
      this.lastPayload = payload;
      onDelta?.('你好');
      onDelta?.('，世界');
      return { finishReason: 'stop' };
    },
  };
  lastFakeBridge = bridge;
  return bridge;
}

async function setup(t, { withBridge = true, bridgeConnected = true, bridgeModels } = {}) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-ai-bridge-'));
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
  });
  const app = createApp(cfg, { browserBridge: withBridge ? fakeBridge({ connected: bridgeConnected, models: bridgeModels }) : null });
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
  assert.equal(lastFakeBridge.lastPayload.temperature, 0.3);
  assert.equal(lastFakeBridge.lastPayload.maxTokens, 8192);
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

test('桥接模式：网页自身聊天 /api/chat 也经桥接返回流式回答', async (t) => {
  const base = await setup(t);
  const loginRes = await fetch(`${base}/api/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password: PASSWORD }),
  });
  assert.equal(loginRes.status, 200);
  const { token } = await loginRes.json();

  const chatRes = await fetch(`${base}/api/chat`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
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
  const loginRes = await fetch(`${base}/api/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password: PASSWORD }),
  });
  const { token } = await loginRes.json();
  const webChat = (body) => fetch(`${base}/api/chat`, {
    method: 'POST',
    headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
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
