// API 用量统计接口：仅管理员可见；/v1 成功（流式/非流式）记 Token，失败记状态，清空生效。
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';

const PASSWORD = 'correct-horse-battery';
const ADMIN = 'manager-secret-test-password';

function fakeBridge({ fail = false } = {}) {
  return {
    fail,
    status() {
      return { connected: true, models: [{ id: 'test-model', label: 'Test Model' }], queued: 0 };
    },
    async run(_payload, { onDelta } = {}) {
      if (this.fail) throw new Error('网页模型生成失败（测试）');
      onDelta?.('你好');
      onDelta?.('，世界');
      return { finishReason: 'stop' };
    },
  };
}

async function setup(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-usage-api-'));
  const bridge = fakeBridge();
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: ADMIN,
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
    USAGE_STATS_FILE: path.join(dir, 'usage.json'),
  });
  const app = createApp(cfg, { browserBridge: bridge });
  const server = await new Promise((resolve) => {
    const srv = app.listen(0, '127.0.0.1', () => resolve(srv));
  });
  t.after(async () => {
    await new Promise((r) => server.close(r));
    await rm(dir, { recursive: true, force: true });
  });
  return { base: `http://127.0.0.1:${server.address().port}`, bridge };
}

async function adminToken(base) {
  const res = await fetch(`${base}/api/admin/api-keys/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password: ADMIN }),
  });
  assert.equal(res.status, 200);
  return (await res.json()).token;
}

async function getStats(base, token) {
  const res = await fetch(`${base}/api/admin/usage-stats`, {
    headers: token ? { authorization: `Bearer ${token}` } : {},
  });
  return { status: res.status, body: res.status === 200 ? await res.json() : null };
}

async function chat(base, token, { stream = false } = {}) {
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` },
    body: JSON.stringify({ model: 'test-model', stream, messages: [{ role: 'user', content: '你好' }] }),
  });
  const text = await res.text();
  return { status: res.status, text };
}

test('未登录或非管理员不能查看 API 用量', async (t) => {
  const { base } = await setup(t);
  assert.equal((await getStats(base, '')).status, 401);
  assert.equal((await getStats(base, 'not-a-token')).status, 401);
});

test('/v1 非流式与流式成功调用都记入 API 用量（口径与响应 usage 一致）', async (t) => {
  const { base } = await setup(t);
  const token = await adminToken(base);

  const one = await chat(base, token);
  assert.equal(one.status, 200);
  const payload = JSON.parse(one.text);
  assert.equal(payload.usage.total_tokens, payload.usage.prompt_tokens + payload.usage.completion_tokens);

  const two = await chat(base, token, { stream: true });
  assert.equal(two.status, 200);
  assert.match(two.text, /\[DONE\]/);

  const { status, body } = await getStats(base, token);
  assert.equal(status, 200);
  assert.equal(body.source, 'api');
  const model = body.models.find((m) => m.id === 'test-model');
  assert.ok(model, '应有 test-model 的统计行');
  assert.equal(model.calls, 2);
  assert.equal(model.errors, 0);
  // 每次：提示词 “你好”（2 字 → ceil(2/4)=1），输出 “你好，世界”（5 字 → ceil(5/4)=2）
  assert.equal(model.promptTokens, payload.usage.prompt_tokens * 2);
  assert.equal(model.completionTokens, payload.usage.completion_tokens * 2);
  assert.equal(body.totalCalls, 2);
  assert.equal(body.recent.length, 2);
  assert.equal(body.recent[0].status, 'ok');
});

test('/v1 生成失败记一次失败（不计 Token），清空后归零', async (t) => {
  const { base, bridge } = await setup(t);
  const token = await adminToken(base);

  bridge.fail = true;
  const failed = await chat(base, token);
  assert.equal(failed.status, 500);
  bridge.fail = false;

  let { body } = await getStats(base, token);
  let model = body.models.find((m) => m.id === 'test-model');
  assert.equal(model.calls, 1);
  assert.equal(model.errors, 1);
  assert.equal(model.totalTokens, 0, '失败不计 Token');

  const del = await fetch(`${base}/api/admin/usage-stats`, {
    method: 'DELETE',
    headers: { authorization: `Bearer ${token}` },
  });
  assert.equal(del.status, 200);
  assert.equal((await del.json()).totalCalls, 0);

  ({ body } = await getStats(base, token));
  assert.equal(body.totalCalls, 0);
  assert.deepEqual(body.models, []);
});
