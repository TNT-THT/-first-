import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { validateSettings } from '../server/settings-store.js';
import { startMockLLM } from './mock-llm.js';

const PASSWORD = 'correct-horse-battery';
let llm;
let server;
let base;
let cfg;
let apiKeyDir;

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
  apiKeyDir = await mkdtemp(path.join(tmpdir(), 'oj-ai-api-keys-'));
  cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    LLM_BASE_URL: llm.url,
    LLM_API_KEY: 'sk-test-123',
    LLM_MODEL: 'qwen-test',
    API_KEYS: 'sk-external-api-key-1,sk-external-api-key-2',
    API_ADMIN_PASSWORD: 'manager-secret-test-password',
    API_KEYS_FILE: path.join(apiKeyDir, 'keys.json'),
    API_SETTINGS_FILE: path.join(apiKeyDir, 'settings.json'),
    CORS_ORIGINS: '*',
    PUBLIC_BASE_URL: 'https://ai.example.test',
    API_MAX_TOKENS: '128',
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
  await rm(apiKeyDir, { recursive: true, force: true });
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
  assert.ok(!text.includes('sk-external-api-key-1'));
  assert.ok(!text.includes(PASSWORD));
  const data = JSON.parse(text);
  assert.equal(data.model, 'qwen-test');
  assert.ok(Array.isArray(data.allowedParentOrigins));
  assert.equal(data.apiKeyManagerEnabled, true);
  assert.equal(data.apiDefaultTemperature, 0.3);
  assert.equal(data.apiMaxTokens, 128);
  assert.equal(text.includes('manager-secret-test-password'), false);
});

test('/api/info 与 /api/docs 提供两种远程接入说明且不泄露密钥', async () => {
  const infoRes = await req('/api/info');
  assert.equal(infoRes.status, 200);
  const info = await infoRes.json();
  assert.equal(info.appName, 'LLM托管平台——天蓝色1.0');
  assert.equal(info.baseUrl, 'https://ai.example.test');
  assert.equal(info.endpoints.chatCompletions, 'https://ai.example.test/v1/chat/completions');
  assert.equal(info.auth.apiKeyConfigured, true);
  assert.ok(!JSON.stringify(info).includes('sk-external-api-key-1'));

  const docsRes = await req('/api/docs');
  assert.equal(docsRes.status, 200);
  assert.match(docsRes.headers.get('content-type'), /text\/html/);
  const html = await docsRes.text();
  assert.match(html, /自建服务器 API/);
  assert.match(html, /网页 API 调试台与管理员参数控制台/);
  assert.match(html, /网页链接接入/);
  assert.match(html, /https:\/\/ai\.example\.test\/v1\/chat\/completions/);
  assert.ok(!html.includes('sk-external-api-key-1'));
});

test('网站端 API Key 管理：独立管理员口令、创建只展示一次、持久化与撤销', async () => {
  const disabled = await req('/api/admin/api-keys');
  assert.equal(disabled.status, 401, '没有 API 管理令牌不能读取 Key 列表');

  const badLogin = await req('/api/admin/api-keys/login', {
    method: 'POST', body: { password: 'wrong-manager-password' },
  });
  assert.equal(badLogin.status, 401);

  const webToken = await login();
  const webTokenCannotManage = await req('/api/admin/api-keys', { token: webToken });
  assert.equal(webTokenCannotManage.status, 401, '普通网站会话不能管理 API Key');

  const adminLogin = await req('/api/admin/api-keys/login', {
    method: 'POST', body: { password: 'manager-secret-test-password' },
  });
  assert.equal(adminLogin.status, 200);
  assert.match(adminLogin.headers.get('cache-control'), /no-store/);
  const adminToken = (await adminLogin.json()).token;
  assert.match(adminToken, /^v1\./);

  const createdRes = await req('/api/admin/api-keys', {
    method: 'POST', token: adminToken, body: { label: 'Hydro backend' },
  });
  assert.equal(createdRes.status, 201);
  const created = await createdRes.json();
  assert.match(created.key, /^ojai_[A-Za-z0-9_-]{40,}$/);
  assert.equal(created.label, 'Hydro backend');
  assert.equal(typeof created.id, 'string');

  const listRes = await req('/api/admin/api-keys', { token: adminToken });
  assert.equal(listRes.status, 200);
  assert.match(listRes.headers.get('cache-control'), /no-store/);
  const listed = await listRes.json();
  assert.equal(listed.environmentKeys, 2);
  assert.equal(listed.keys.length, 1);
  assert.equal(JSON.stringify(listed).includes(created.key), false, '列表不应再次泄露明文 Key');

  const storePath = path.join(apiKeyDir, 'keys.json');
  const disk = await readFile(storePath, 'utf8');
  assert.equal(disk.includes(created.key), false, '磁盘仅存摘要，不保存明文 Key');
  assert.match(disk, /[a-f0-9]{64}/);
  assert.equal((await stat(storePath)).mode & 0o777, 0o600, 'Key 存储文件仅供进程所有者读取');
  assert.equal((await stat(apiKeyDir)).mode & 0o777, 0o700, 'Key 存储目录仅供进程所有者访问');

  const usable = await req('/v1/models', { token: created.key });
  assert.equal(usable.status, 200, '网站生成的 Key 能调用 OpenAI 兼容 API');

  // 模拟 Node 进程重启后从持久化文件恢复密钥。
  const restarted = createApp(cfg);
  const restartedServer = await new Promise((resolve) => {
    const srv = restarted.listen(0, '127.0.0.1', () => resolve(srv));
  });
  const afterRestart = await fetch(`http://127.0.0.1:${restartedServer.address().port}/v1/models`, {
    headers: { authorization: `Bearer ${created.key}` },
  });
  assert.equal(afterRestart.status, 200);
  await new Promise((resolve) => restartedServer.close(resolve));

  const revoked = await req(`/api/admin/api-keys/${encodeURIComponent(created.id)}`, {
    method: 'DELETE', token: adminToken,
  });
  assert.equal(revoked.status, 200);
  const afterRevoke = await req('/v1/models', { token: created.key });
  assert.equal(afterRevoke.status, 401, '撤销后 API Key 立即失效');
});

test('/v1/models：API Key 鉴权、OpenAI 格式模型列表与 CORS', async () => {
  const unauthorized = await req('/v1/models');
  assert.equal(unauthorized.status, 401);
  assert.equal(typeof (await unauthorized.json()).error.message, 'string');

  const r = await req('/v1/models', { token: 'sk-external-api-key-1' });
  assert.equal(r.status, 200);
  const data = await r.json();
  assert.equal(data.object, 'list');
  assert.deepEqual(data.data.map((m) => m.id), ['qwen-test']);

  const cors = await fetch(`${base}/v1/models`, {
    method: 'OPTIONS',
    headers: { origin: 'https://client.example.test', 'access-control-request-method': 'POST' },
  });
  assert.equal(cors.status, 204);
  assert.equal(cors.headers.get('access-control-allow-origin'), '*');
  assert.match(cors.headers.get('access-control-allow-headers'), /Authorization/i);
});

test('误把完整对话接口填为 Base URL 时的兼容路径，以及 /v1 未知路径的 JSON 404', async () => {
  // 客户端在误填的 Base URL 后拼接 /models 取模型列表
  const models = await req('/v1/chat/completions/models', { token: 'sk-external-api-key-1' });
  assert.equal(models.status, 200);
  const modelsData = await models.json();
  assert.deepEqual(modelsData.data.map((m) => m.id), ['qwen-test']);

  // 同样误填时拼接出的双重对话路径也能正常对话
  const chat = await req('/v1/chat/completions/chat/completions', {
    method: 'POST',
    token: 'sk-external-api-key-1',
    body: { model: 'qwen-test', messages: [{ role: 'user', content: '你好' }], stream: false },
  });
  assert.equal(chat.status, 200);
  const chatData = await chat.json();
  assert.equal(chatData.object, 'chat.completion');

  // 其他未知 /v1 路径返回 OpenAI 风格 JSON（而不是 HTML），并提示 Base URL 的正确填法
  const missing = await req('/v1/what/is/this', { token: 'sk-external-api-key-1' });
  assert.equal(missing.status, 404);
  assert.match(missing.headers.get('content-type'), /application\/json/);
  const missingData = await missing.json();
  assert.match(missingData.error.message, /Base URL 以 \/v1 结尾/);
});

test('网站登录令牌不能调用 /v1；API 管理员令牌可用于网页调试台', async () => {
  const webToken = await login();
  const webSessionDenied = await req('/v1/models', { token: webToken });
  assert.equal(webSessionDenied.status, 401, '普通网站会话不能作为 API 凭证');

  const adminLogin = await req('/api/admin/api-keys/login', {
    method: 'POST', body: { password: 'manager-secret-test-password' },
  });
  assert.equal(adminLogin.status, 200);
  const adminToken = (await adminLogin.json()).token;
  const models = await req('/v1/models', { token: adminToken });
  assert.equal(models.status, 200);
  assert.deepEqual((await models.json()).data.map((model) => model.id), ['qwen-test']);

  const completion = await req('/v1/chat/completions', {
    method: 'POST',
    token: adminToken,
    body: { model: 'qwen-test', messages: [{ role: 'user', content: '管理员网页调试台测试' }], stream: false },
  });
  assert.equal(completion.status, 200);
  assert.equal((await completion.json()).choices[0].message.content, '你好，世界$O(n)$');

  const cannotManage = await req('/api/admin/api-keys', { token: webToken });
  assert.equal(cannotManage.status, 401);
});

test('管理员参数控制台：服务端鉴权、写入不回显密钥、运行时更新与持久化恢复', async () => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-ai-settings-'));
  const settingsPath = path.join(dir, 'api-settings.json');
  const makeConfig = () => loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'settings-test-secret-settings-test',
    LLM_BASE_URL: llm.url,
    LLM_API_KEY: 'sk-settings-env',
    LLM_MODEL: 'qwen-test',
    API_KEYS: 'sk-settings-user-key',
    API_ADMIN_PASSWORD: 'manager-settings-test-password',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: settingsPath,
    API_MAX_TOKENS: '128',
    CORS_ORIGINS: '*',
  });
  const listen = (app) => new Promise((resolve) => {
    const srv = app.listen(0, '127.0.0.1', () => resolve(srv));
  });
  const settingsServer = await listen(createApp(makeConfig()));
  const settingsBase = `http://127.0.0.1:${settingsServer.address().port}`;
  const settingsReq = (route, { method = 'GET', token, body } = {}) => fetch(settingsBase + route, {
    method,
    headers: {
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...(body !== undefined ? { 'content-type': 'application/json' } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  try {
    assert.equal((await settingsReq('/api/admin/settings')).status, 401);
    const webLogin = await settingsReq('/api/login', {
      method: 'POST', body: { password: PASSWORD },
    });
    const webToken = (await webLogin.json()).token;
    assert.equal((await settingsReq('/api/admin/settings', { token: webToken })).status, 401);
    assert.equal((await settingsReq('/v1/models', { token: webToken })).status, 401);

    const adminLogin = await settingsReq('/api/admin/api-keys/login', {
      method: 'POST', body: { password: 'manager-settings-test-password' },
    });
    assert.equal(adminLogin.status, 200);
    const adminToken = (await adminLogin.json()).token;
    const initial = await settingsReq('/api/admin/settings', { token: adminToken });
    assert.equal(initial.status, 200);
    const initialData = await initial.json();
    assert.equal(initialData.llmApiKeyConfigured, true);
    assert.deepEqual(initialData.corsOrigins, ['*']);
    assert.ok(!JSON.stringify(initialData).includes('sk-settings-env'));
    assert.match(initial.headers.get('cache-control'), /no-store/);

    const invalidUrl = await settingsReq('/api/admin/settings', {
      method: 'PUT', token: adminToken, body: { llmBaseUrl: 'file:///etc/passwd' },
    });
    assert.equal(invalidUrl.status, 400, '只允许 HTTP(S) 上游地址');
    const secretParam = await settingsReq('/api/admin/settings', {
      method: 'PUT', token: adminToken, body: { defaultGenerationParams: { upstream_api_key: 'not-a-setting' } },
    });
    assert.equal(secretParam.status, 400, '额外参数不允许包含 API Key 字段');
    const badOrigin = await settingsReq('/api/admin/settings', {
      method: 'PUT', token: adminToken, body: { corsOrigins: ['https://example.com/some/path'] },
    });
    assert.equal(badOrigin.status, 400, '跨域白名单只能填写网站来源，不能带路径');

    const writtenKey = 'sk-settings-console-new-secret';
    const update = await settingsReq('/api/admin/settings', {
      method: 'PUT', token: adminToken,
      body: {
        llmBaseUrl: llm.url,
        llmModel: 'qwen-settings',
        llmApiKey: writtenKey,
        llmTemperature: 0.8,
        systemPrompt: '管理员参数控制台提示词',
        apiMaxTokens: 64,
        defaultGenerationParams: { top_p: 0.7, stop: ['<end>'], vendor_option: { quality: 'high' } },
        corsOrigins: ['https://pd.geneluo.fun/', 'https://*.client.example.test'],
      },
    });
    assert.equal(update.status, 200);
    const updated = await update.json();
    assert.equal(updated.llmModel, 'qwen-settings');
    assert.equal(updated.apiMaxTokens, 64);
    assert.deepEqual(updated.corsOrigins, ['https://pd.geneluo.fun', 'https://*.client.example.test'], '白名单应规范化并保存');
    assert.equal(updated.llmApiKeyConfigured, true);

    // 白名单保存后立即生效：名单内来源拿到跨域与本机网络许可，名单外来源拿不到
    const preflight = (origin, extra = {}) => fetch(`${settingsBase}/v1/models`, {
      method: 'OPTIONS',
      headers: { origin, 'access-control-request-method': 'GET', 'access-control-request-headers': 'authorization', ...extra },
    });
    const allowedPre = await preflight('https://pd.geneluo.fun', { 'access-control-request-private-network': 'true' });
    assert.equal(allowedPre.headers.get('access-control-allow-origin'), 'https://pd.geneluo.fun');
    assert.equal(allowedPre.headers.get('access-control-allow-private-network'), 'true', '白名单来源的本机网络预检应获得许可');
    const deniedPre = await preflight('https://evil.example.test', { 'access-control-request-private-network': 'true' });
    assert.equal(deniedPre.headers.get('access-control-allow-origin'), null);
    assert.equal(deniedPre.headers.get('access-control-allow-private-network'), null, '名单外来源不得获得本机网络许可');
    assert.ok(!JSON.stringify(updated).includes(writtenKey), 'API 管理响应不能回显上游 Key');

    const settingsFile = await readFile(settingsPath, 'utf8');
    assert.ok(settingsFile.includes(writtenKey), '管理员设置需要持久化到服务端文件');
    assert.equal((await stat(settingsPath)).mode & 0o777, 0o600);
    assert.equal((await stat(dir)).mode & 0o777, 0o700);

    const models = await settingsReq('/v1/models', { token: 'sk-settings-user-key' });
    assert.equal(models.status, 200);
    assert.deepEqual((await models.json()).data.map((model) => model.id), ['qwen-settings']);
    const before = llm.requests.length;
    const completion = await settingsReq('/v1/chat/completions', {
      method: 'POST', token: 'sk-settings-user-key',
      body: {
        model: 'qwen-settings',
        messages: [{ role: 'user', content: 'third-party client test' }],
        top_p: 0.9,
        presence_penalty: 0.1,
        stream: false,
      },
    });
    assert.equal(completion.status, 200);
    const upstream = llm.requests[before];
    assert.equal(upstream.headers.authorization, `Bearer ${writtenKey}`);
    assert.equal(upstream.body.temperature, 0.8);
    assert.equal(upstream.body.max_tokens, 64);
    assert.equal(upstream.body.top_p, 0.9, '调用方可覆盖默认生成参数');
    assert.deepEqual(upstream.body.stop, ['<end>']);
    assert.deepEqual(upstream.body.vendor_option, { quality: 'high' });
    assert.equal(upstream.body.presence_penalty, 0.1);

    const restartedServer = await listen(createApp(makeConfig()));
    const restartedBase = `http://127.0.0.1:${restartedServer.address().port}`;
    const persisted = await fetch(`${restartedBase}/v1/models`, {
      headers: { authorization: 'Bearer sk-settings-user-key' },
    });
    assert.equal(persisted.status, 200);
    assert.deepEqual((await persisted.json()).data.map((model) => model.id), ['qwen-settings']);
    const persistedPre = await fetch(`${restartedBase}/v1/models`, {
      method: 'OPTIONS',
      headers: { origin: 'https://pd.geneluo.fun', 'access-control-request-method': 'GET' },
    });
    assert.equal(persistedPre.headers.get('access-control-allow-origin'), 'https://pd.geneluo.fun', '白名单应在重启后继续生效');
    await new Promise((resolve) => restartedServer.close(resolve));

    const reset = await settingsReq('/api/admin/settings', { method: 'DELETE', token: adminToken });
    assert.equal(reset.status, 200);
    const resetData = await reset.json();
    assert.equal(resetData.llmModel, 'qwen-test');
    assert.equal(resetData.apiMaxTokens, 128);
    assert.deepEqual(resetData.corsOrigins, ['*'], '恢复 .env 配置时白名单也应回到启动值');
    assert.equal(resetData.llmApiKeyConfigured, true);
  } finally {
    await new Promise((resolve) => settingsServer.close(resolve));
    await rm(dir, { recursive: true, force: true });
  }
});

test('跨域许可只发给 /v1，白名单来源也不能跨域调用 /api 接口', async () => {
  const restricted = createApp(loadConfig({ APP_PASSWORD: PASSWORD, CORS_ORIGINS: 'https://pd.geneluo.fun' }));
  const s = await new Promise((resolve) => {
    const srv = restricted.listen(0, '127.0.0.1', () => resolve(srv));
  });
  const origin = 'https://pd.geneluo.fun';
  for (const p of ['/api/api-keys/apply', '/api/login', '/api/admin/settings', '/healthz']) {
    const pre = await fetch(`http://127.0.0.1:${s.address().port}${p}`, {
      method: 'OPTIONS',
      headers: { origin, 'access-control-request-method': 'POST' },
    });
    assert.equal(pre.headers.get('access-control-allow-origin'), null, `${p} 不应获得跨域许可`);
  }
  const v1 = await fetch(`http://127.0.0.1:${s.address().port}/v1/models`, {
    method: 'OPTIONS',
    headers: { origin, 'access-control-request-method': 'GET' },
  });
  assert.equal(v1.headers.get('access-control-allow-origin'), origin, '/v1 应获得跨域许可');
  await new Promise((resolve) => s.close(resolve));
});

test('跨域白名单校验：https://*.com 这类一级后缀通配过宽，应被拒绝', () => {
  assert.throws(() => validateSettings({ corsOrigins: ['https://*.com'] }), /至少需要写到二级域名/);
  assert.deepEqual(validateSettings({ corsOrigins: ['https://*.example.com'] }).corsOrigins, ['https://*.example.com']);
});

test('CORS_ORIGINS 支持显式来源与通配子域白名单', async () => {
  const restricted = createApp(loadConfig({ APP_PASSWORD: PASSWORD, CORS_ORIGINS: 'https://*.client.example.test' }));
  const s = await new Promise((resolve) => {
    const srv = restricted.listen(0, '127.0.0.1', () => resolve(srv));
  });
  const url = `http://127.0.0.1:${s.address().port}/v1/models`;
  const child = await fetch(url, {
    method: 'OPTIONS',
    headers: { origin: 'https://oj.client.example.test', 'access-control-request-method': 'POST' },
  });
  assert.equal(child.headers.get('access-control-allow-origin'), 'https://oj.client.example.test');
  const apex = await fetch(url, {
    method: 'OPTIONS',
    headers: { origin: 'https://client.example.test', 'access-control-request-method': 'POST' },
  });
  assert.equal(apex.headers.get('access-control-allow-origin'), null);
  await new Promise((resolve) => s.close(resolve));
});

test('/v1/chat/completions：非流式调用支持 Bearer / x-api-key，返回 OpenAI 格式', async () => {
  const before = llm.requests.length;
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-api-key': 'sk-external-api-key-2' },
    body: JSON.stringify({
      model: 'qwen-test',
      messages: [{ role: 'system', content: '请简短回答。' }, { role: 'user', content: '你好' }],
      temperature: 0.6,
      max_tokens: 32,
      top_p: 0.85,
      frequency_penalty: 0.25,
      stream: false,
    }),
  });
  assert.equal(r.status, 200);
  const data = await r.json();
  assert.equal(data.object, 'chat.completion');
  assert.equal(data.model, 'qwen-test');
  assert.match(data.id, /^chatcmpl-/);
  assert.equal(data.choices[0].message.role, 'assistant');
  assert.equal(data.choices[0].message.content, '你好，世界$O(n)$');
  assert.ok(data.usage.total_tokens > 0);
  const upstream = llm.requests[before];
  assert.equal(upstream.body.stream, true);
  assert.equal(upstream.body.temperature, 0.6);
  assert.equal(upstream.body.max_tokens, 32);
  assert.equal(upstream.body.top_p, 0.85);
  assert.equal(upstream.body.frequency_penalty, 0.25);
  assert.equal(upstream.body.messages[1].role, 'system');
  assert.equal(upstream.body.messages[2].content, '你好');
});

test('/v1/chat/completions：流式 SSE 补齐 OpenAI 元数据；模型与参数非法时拒绝', async () => {
  const before = llm.requests.length;
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer sk-external-api-key-1' },
    body: JSON.stringify({
      model: 'qwen-test',
      messages: [{ role: 'user', content: '你好' }],
      stream: true,
    }),
  });
  assert.equal(r.status, 200);
  assert.match(r.headers.get('content-type'), /text\/event-stream/);
  const raw = await r.text();
  const chunks = raw.split('\n').filter((line) => line.startsWith('data: ') && !line.includes('[DONE]'));
  assert.ok(chunks.length >= 1);
  const first = JSON.parse(chunks[0].slice(6));
  assert.match(first.id, /^chatcmpl-/);
  assert.equal(first.object, 'chat.completion.chunk');
  assert.equal(first.model, 'qwen-test');
  assert.ok(raw.includes('data: [DONE]'));
  assert.equal(llm.requests[before].body.max_tokens, 128, '未指定时应用服务端 API_MAX_TOKENS 上限');

  const invalidModel = await req('/v1/chat/completions', {
    method: 'POST', token: 'sk-external-api-key-1',
    body: { model: 'other-model', messages: [{ role: 'user', content: 'x' }] },
  });
  assert.equal(invalidModel.status, 404);
  assert.equal((await invalidModel.json()).error.type, 'model_not_found');
  const invalidTemperature = await req('/v1/chat/completions', {
    method: 'POST', token: 'sk-external-api-key-1',
    body: { model: 'qwen-test', temperature: 3, messages: [{ role: 'user', content: 'x' }] },
  });
  assert.equal(invalidTemperature.status, 400);
  const unsupportedTools = await req('/v1/chat/completions', {
    method: 'POST', token: 'sk-external-api-key-1',
    body: { model: 'qwen-test', tools: [], messages: [{ role: 'user', content: 'x' }] },
  });
  assert.equal(unsupportedTools.status, 400, '不支持的工具调用不能被悄悄转发');

  const malformed = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { authorization: 'Bearer sk-external-api-key-1', 'content-type': 'application/json' },
    body: '{',
  });
  assert.equal(malformed.status, 400);
  assert.equal(typeof (await malformed.json()).error.message, 'string');
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
