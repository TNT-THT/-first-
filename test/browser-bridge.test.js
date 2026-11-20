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

function fakeBridge({ connected = true } = {}) {
  const bridge = {
    models: [{ id: 'test-model', label: 'Test Model' }],
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

async function setup(t, { withBridge = true, bridgeConnected = true } = {}) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-ai-bridge-'));
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
  });
  const app = createApp(cfg, { browserBridge: withBridge ? fakeBridge({ connected: bridgeConnected }) : null });
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
