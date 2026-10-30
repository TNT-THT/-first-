import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { startMockLLM } from './mock-llm.js';

const PASSWORD = 'correct-horse-battery';
let llm;
let server;
let base;
let cfg;

async function req(path, { method = 'GET', token, body, headers = {} } = {}) {
  const h = { ...headers };
  if (token) h.authorization = `Bearer ${token}`;
  if (body !== undefined) h['content-type'] = 'application/json';
  return fetch(base + path, { method, headers: h, body: body === undefined ? undefined : JSON.stringify(body) });
}

async function login() {
  const r = await req('/api/login', { method: 'POST', body: { password: PASSWORD } });
  assert.equal(r.status, 200);
  return (await r.json()).token;
}

before(async () => {
  llm = await startMockLLM();
  cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    LLM_BASE_URL: llm.url,
    LLM_API_KEY: 'sk-test-123',
    LLM_MODEL: 'qwen-test',
    FRAME_ANCESTORS: '*',
  });
  const app = createApp(cfg);
  await new Promise((resolve) => {
    server = app.listen(0, '127.0.0.1', resolve);
  });
  base = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await new Promise((r) => server.close(r));
  await llm.close();
});

test('响应头：允许嵌入且不包含 X-Frame-Options', async () => {
  const r = await req('/healthz');
  assert.equal(r.headers.get('x-frame-options'), null);
  const csp = r.headers.get('content-security-policy');
  assert.match(csp, /frame-ancestors \*/);
  assert.equal(r.headers.get('x-content-type-options'), 'nosniff');
});

test('CSP frame-ancestors 可通过环境变量收紧', async () => {
  const restricted = createApp(loadConfig({ APP_PASSWORD: PASSWORD, FRAME_ANCESTORS: "'self' https://hydro.ac" }));
  const s = await new Promise((r) => {
    const srv = restricted.listen(0, '127.0.0.1', () => r(srv));
  });
  const r = await fetch(`http://127.0.0.1:${s.address().port}/healthz`);
  assert.match(r.headers.get('content-security-policy'), /frame-ancestors 'self' https:\/\/hydro\.ac/);
  await new Promise((r2) => s.close(r2));
});

test('CSP 的 connect-src 默认允许免费的联网资料检索来源，可用 CSP_CONNECT_SRC 覆盖', async () => {
  const r = await req('/healthz');
  const csp = r.headers.get('content-security-policy');
  assert.match(csp, /connect-src 'self' https:\/\/\*\.wikipedia\.org/);
  assert.match(csp, /https:\/\/api\.stackexchange\.com/);
  assert.match(csp, /https:\/\/hn\.algolia\.com/);

  const custom = createApp(
    loadConfig({ APP_PASSWORD: PASSWORD, CSP_CONNECT_SRC: 'https://searx.example.com' }),
  );
  const s = await new Promise((resolve) => {
    const srv = custom.listen(0, '127.0.0.1', () => resolve(srv));
  });
  const r2 = await fetch(`http://127.0.0.1:${s.address().port}/healthz`);
  const csp2 = r2.headers.get('content-security-policy');
  assert.match(csp2, /connect-src 'self' https:\/\/searx\.example\.com/);
  assert.equal(csp2.includes('wikipedia'), false);
  await new Promise((r3) => s.close(r3));
});

test('/api/config 公开且不泄露密钥', async () => {
  const r = await req('/api/config');
  const text = await r.text();
  assert.equal(r.status, 200);
  assert.ok(!text.includes('sk-test-123'));
  assert.ok(!text.includes(PASSWORD));
  const data = JSON.parse(text);
  assert.equal(data.model, 'qwen-test');
  assert.ok(Array.isArray(data.allowedParentOrigins));
});

test('登录：错误密码 401，正确密码返回令牌', async () => {
  const bad = await req('/api/login', { method: 'POST', body: { password: 'wrong' } });
  assert.equal(bad.status, 401);
  const token = await login();
  assert.match(token, /^v1\./);
  const ok = await req('/api/session', { token });
  assert.equal(ok.status, 200);
});

test('未登录或伪造令牌无法访问 /api/chat', async () => {
  const body = { messages: [{ role: 'user', content: '题目' }] };
  assert.equal((await req('/api/chat', { method: 'POST', body })).status, 401);
  assert.equal((await req('/api/chat', { method: 'POST', body, token: 'v1.1.deadbeef' })).status, 401);
  const token = await login();
  const tampered = token.replace(/.$/, (c) => (c === '0' ? '1' : '0'));
  assert.equal((await req('/api/chat', { method: 'POST', body, token: tampered })).status, 401);
});

test('/api/chat 流式透传上游结果，并带上服务端密钥与系统提示', async () => {
  const token = await login();
  const before = llm.requests.length;
  const r = await req('/api/chat', {
    method: 'POST',
    token,
    body: { messages: [{ role: 'user', content: '两数之和怎么做？' }] },
  });
  assert.equal(r.status, 200);
  assert.match(r.headers.get('content-type'), /text\/event-stream/);
  assert.equal(r.headers.get('x-accel-buffering'), 'no');
  const text = await r.text();
  const deltas = text
    .split('\n')
    .filter((l) => l.startsWith('data:') && !l.includes('[DONE]'))
    .map((l) => JSON.parse(l.slice(5)).choices[0].delta.content);
  assert.equal(deltas.join(''), '你好，世界$O(n)$');
  assert.ok(text.includes('[DONE]'));

  const upstream = llm.requests[before];
  assert.equal(upstream.headers.authorization, 'Bearer sk-test-123');
  assert.equal(upstream.body.model, 'qwen-test');
  assert.equal(upstream.body.stream, true);
  assert.equal(upstream.body.messages[0].role, 'system');
  assert.match(upstream.body.messages[0].content, /外部资料.*不可信/);
  assert.equal(upstream.body.messages[1].content, '两数之和怎么做？');
});

test('上游报错时返回 502，且不把上游原始信息（含密钥）回显给浏览器', async () => {
  const token = await login();
  const r = await req('/api/chat', {
    method: 'POST',
    token,
    body: { messages: [{ role: 'user', content: '__ERROR__' }] },
  });
  const text = await r.text();
  assert.equal(r.status, 502);
  assert.match(text, /模型服务返回错误/);
  assert.ok(!text.includes('LEAKED'));
});

test('请求体校验：空数组、非法角色、末尾非用户消息、超长都会被拒绝', async () => {
  const token = await login();
  const cases = [
    { messages: [] },
    { messages: [{ role: 'system', content: 'x' }] },
    { messages: [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }] },
    { messages: [{ role: 'user', content: 'a'.repeat(20001) }] },
    { messages: [{ role: 'user', content: '' }] },
    { messages: 'nope' },
  ];
  for (const body of cases) {
    const r = await req('/api/chat', { method: 'POST', token, body });
    assert.equal(r.status, 400, JSON.stringify(body).slice(0, 80));
  }
});

test('登录失败次数过多会被限流', async () => {
  const app = createApp(loadConfig({ APP_PASSWORD: PASSWORD, SESSION_SECRET: 'x'.repeat(16) }));
  const s = await new Promise((r) => {
    const srv = app.listen(0, '127.0.0.1', () => r(srv));
  });
  const u = `http://127.0.0.1:${s.address().port}/api/login`;
  const statuses = [];
  for (let i = 0; i < 11; i++) {
    const r = await fetch(u, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ password: 'bad-password' }),
    });
    statuses.push(r.status);
  }
  assert.equal(statuses[9], 401);
  assert.equal(statuses[10], 429);
  await new Promise((r) => s.close(r));
});

test('前端入口可访问（需先执行 npm run build）', async () => {
  const r = await req('/');
  assert.equal(r.status, 200);
  assert.equal(r.headers.get('cache-control'), 'no-cache');
  const html = await r.text();
  assert.match(html, /<div id="app"/);
  assert.equal(r.headers.get('x-frame-options'), null);
});
