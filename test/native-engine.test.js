// 本机大模型引擎全栈测试：用假 Hugging Face + 假 llama-server（Node 脚本）
// 跑通“查量化 → 下载 → 加载 → 网页聊天 SSE → /v1 非流式 → 用量统计”整条链路，
// 并验证权限边界（未登录 401、未知模型 404、未下载 409）。不访问真实网络。
import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { NativeEngine } from '../server/native/engine.js';

const PASSWORD = 'correct-horse-battery';
const ADMIN = 'manager-secret-test-password';
const MODEL_ID = 'Qwen3-30B-A3B'; // bartowski 根目录分片布局

const STUB_LLAMA = `#!/usr/bin/env node
const http = require('node:http');
const args = process.argv.slice(2);
const port = Number(args[args.indexOf('--port') + 1]);
const apiKey = args[args.indexOf('--api-key') + 1];
http.createServer((req, res) => {
  if (req.headers.authorization !== 'Bearer ' + apiKey) {
    res.writeHead(401); res.end('unauthorized'); return;
  }
  if (req.url === '/health') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok' })); return;
  }
  if (req.url === '/v1/chat/completions' && req.method === 'POST') {
    let raw = '';
    req.on('data', (c) => { raw += c; });
    req.on('end', () => {
      const body = JSON.parse(raw || '{}');
      if (body.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const chunk = (delta, finish = null, usage) => JSON.stringify({
          id: 'chatcmpl-stub', object: 'chat.completion.chunk', created: 1, model: 'stub',
          choices: [{ index: 0, delta, finish_reason: finish }], ...(usage ? { usage } : {}),
        });
        res.write('data: ' + chunk({ role: 'assistant' }) + '\\n\\n');
        res.write('data: ' + chunk({ reasoning_content: '先思考' }) + '\\n\\n');
        res.write('data: ' + chunk({ content: '最终答案' }) + '\\n\\n');
        res.write('data: ' + chunk({}, 'stop') + '\\n\\n');
        res.write('data: ' + chunk({}, null, { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 }) + '\\n\\n');
        res.write('data: [DONE]\\n\\n');
        res.end(); return;
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({
        id: 'chatcmpl-stub', object: 'chat.completion', created: 1, model: 'stub',
        choices: [{ index: 0, message: { role: 'assistant', reasoning_content: '先思考', content: '最终答案' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
      }));
    });
    return;
  }
  res.writeHead(404); res.end('not found');
}).listen(port, '127.0.0.1');
`;

const TREE = [
  { type: 'file', path: 'Qwen3-30B-A3B-Q4_K_M-00001-of-00002.gguf', size: 11, lfs: { oid: 'a'.repeat(64), size: 11 } },
  { type: 'file', path: 'Qwen3-30B-A3B-Q4_K_M-00002-of-00002.gguf', size: 12, lfs: { oid: 'b'.repeat(64), size: 12 } },
  { type: 'file', path: 'Qwen3-30B-A3B-Q2_K.gguf', size: 7, lfs: { oid: 'c'.repeat(64), size: 7 } },
  { type: 'file', path: 'mmproj-F16.gguf', size: 99, lfs: { oid: 'd'.repeat(64), size: 99 } },
  { type: 'file', path: 'README.md', size: 5 },
];

function stubFetch(realFetch = fetch) {
  return async (url, opts) => {
    const u = String(url);
    if (u.includes('/api/models/') && u.includes('/tree/')) {
      return new Response(JSON.stringify(TREE), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/resolve/main/')) {
      const name = decodeURIComponent(u.split('/resolve/main/')[1]);
      const size = name.includes('-00002-of-') ? 12 : 11;
      return new Response('G'.repeat(size), { headers: { 'content-type': 'application/octet-stream' } });
    }
    if (u.startsWith('http://127.0.0.1')) return realFetch(url, opts);
    throw new Error(`测试中不允许访问：${u}`);
  };
}

async function setup(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-native-'));
  const stubBin = path.join(dir, 'llama-server');
  await writeFile(stubBin, STUB_LLAMA);
  await chmod(stubBin, 0o755);
  const fetchImpl = stubFetch();
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    fetchImpl,
    env: { LLAMA_SERVER_BIN: stubBin },
    logger: { warn() {}, log() {}, error() {} },
  });
  const cfg = loadConfig({
    APP_PASSWORD: PASSWORD,
    SESSION_SECRET: 'test-secret-test-secret',
    API_ADMIN_PASSWORD: ADMIN,
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
    USAGE_STATS_FILE: path.join(dir, 'usage.json'),
  });
  const app = createApp(cfg, { nativeEngine: engine });
  const server = await new Promise((resolve) => {
    const srv = app.listen(0, '127.0.0.1', () => resolve(srv));
  });
  t.after(async () => {
    await engine.unload().catch(() => {});
    await new Promise((r) => server.close(r));
    await rm(dir, { recursive: true, force: true });
  });
  return { base: `http://127.0.0.1:${server.address().port}`, engine };
}

async function login(base, password = PASSWORD) {
  const res = await fetch(`${base}/api/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  assert.equal(res.status, 200);
  return (await res.json()).token;
}

async function adminLogin(base) {
  const res = await fetch(`${base}/api/admin/api-keys/login`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password: ADMIN }),
  });
  assert.equal(res.status, 200);
  return (await res.json()).token;
}

const auth = (token) => ({ authorization: `Bearer ${token}`, 'content-type': 'application/json' });

test('状态与量化：目录公开可读，量化按仓库文件解析且带显存估算', { timeout: 30000 }, async (t) => {
  const { base } = await setup(t);
  const status = await (await fetch(`${base}/api/native/status`)).json();
  assert.equal(status.enabled, true);
  assert.equal(status.binaryReady, true);
  assert.equal(status.binarySource, 'env');
  assert.equal(status.models.length, 11);
  assert.equal(status.models.find((m) => m.id === MODEL_ID).downloaded.length, 0);

  const token = await login(base);
  const res = await fetch(`${base}/api/native/models/${MODEL_ID}/quants`, { headers: auth(token) });
  assert.equal(res.status, 200);
  const { quants } = await res.json();
  assert.deepEqual(quants.map((q) => q.quant), ['Q2_K', 'Q4_K_M']); // 按体积升序，mmproj 被排除
  assert.equal(quants.find((q) => q.quant === 'Q4_K_M').files, 2);
  assert.ok(quants.every((q) => q.ramGB > 0));

  const noAuth = await fetch(`${base}/api/native/models/${MODEL_ID}/quants`);
  assert.equal(noAuth.status, 401, '量化查询需要网站登录');
});

test('下载→加载→聊天→/v1→用量：整条链路与权限边界', { timeout: 60000 }, async (t) => {
  const { base } = await setup(t);
  const admin = await adminLogin(base);
  const session = await login(base);

  // 未下载时聊天应明确 409，而不是假装能跑
  const early = await fetch(`${base}/api/native/chat`, {
    method: 'POST',
    headers: auth(session),
    body: JSON.stringify({ model: MODEL_ID, messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(early.status, 409);
  assert.match((await early.json()).error, /尚未下载/);

  // 普通会话不能下载权重（需要 API 管理员）
  const forbidden = await fetch(`${base}/api/native/models/${MODEL_ID}/download`, {
    method: 'POST',
    headers: auth(session),
    body: JSON.stringify({ quant: 'Q4_K_M' }),
  });
  assert.equal(forbidden.status, 401);

  // 管理员下载 → 等完成
  const dl = await fetch(`${base}/api/native/models/${MODEL_ID}/download`, {
    method: 'POST',
    headers: auth(admin),
    body: JSON.stringify({ quant: 'Q4_K_M' }),
  });
  assert.equal(dl.status, 202);
  let downloaded = null;
  for (let i = 0; i < 50 && !downloaded; i += 1) {
    const s = await (await fetch(`${base}/api/native/status`)).json();
    downloaded = s.models.find((m) => m.id === MODEL_ID).downloaded.find((d) => d.quant === 'Q4_K_M');
    if (!downloaded) await new Promise((r) => setTimeout(r, 100));
  }
  assert.ok(downloaded, '下载应在限定时间内完成');
  assert.equal(downloaded.files, 2);

  // 网页聊天（流式）：思考被折叠为 <think>
  const chat = await fetch(`${base}/api/native/chat`, {
    method: 'POST',
    headers: auth(session),
    body: JSON.stringify({ model: MODEL_ID, messages: [{ role: 'user', content: '你好' }], stream: true }),
  });
  assert.equal(chat.status, 200);
  const text = await chat.text();
  assert.match(text, /data: \[DONE\]/);
  // 像真实客户端一样拼接各分片的 delta.content 再断言
  const joined = text
    .split('\n')
    .filter((l) => l.startsWith('data: ') && !l.includes('[DONE]'))
    .map((l) => JSON.parse(l.slice(6)))
    .map((d) => d.choices?.[0]?.delta?.content || '')
    .join('');
  assert.equal(joined, '<think>先思考</think>最终答案');

  // /v1 非流式（管理员令牌也可作 API 凭证）+ 模型列表包含本机大模型
  const models = await (await fetch(`${base}/v1/models`, { headers: auth(admin) })).json();
  assert.ok(models.data.some((m) => m.id === 'DeepSeek-R1-671B'));
  const v1 = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: auth(admin),
    body: JSON.stringify({ model: MODEL_ID, messages: [{ role: 'user', content: '你好' }] }),
  });
  assert.equal(v1.status, 200);
  const v1Body = await v1.json();
  assert.equal(v1Body.choices[0].message.content, '<think>先思考</think>最终答案');
  assert.equal(v1Body.usage.total_tokens, 7);

  // 用量统计应记录这次 /v1 调用（按模型聚合）
  const stats = await (await fetch(`${base}/api/admin/usage-stats`, { headers: auth(admin) })).json();
  assert.ok(
    stats.models?.[MODEL_ID] || JSON.stringify(stats).includes(MODEL_ID),
    '用量统计中应出现本机大模型',
  );

  // 卸载后恢复空闲
  const unloaded = await fetch(`${base}/api/native/unload`, { method: 'POST', headers: auth(admin) });
  assert.equal(unloaded.status, 200);
  const after = await (await fetch(`${base}/api/native/status`)).json();
  assert.equal(after.running.state, 'idle');

  // 未知模型 404
  const unknown = await fetch(`${base}/api/native/chat`, {
    method: 'POST',
    headers: auth(session),
    body: JSON.stringify({ model: 'Nope-999B', messages: [{ role: 'user', content: 'x' }] }),
  });
  assert.equal(unknown.status, 404);
});

test('直连下载：已下完的分片不再重复下载，.part 续传，多分片并行', { timeout: 30000 }, async (t) => {
  // 一档 3 个分片：第 1 片已下完转正、第 2 片 .part 有 4 字节、第 3 片没下过。
  // 以前重试会把第 1 片从头再拉一遍（用户眼里就是“失败后从头开始”），且是逐片排队。
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-native-resume-'));
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const SHARDS = [
    { path: 'Qwen3-30B-A3B-Q4_K_M-00001-of-00003.gguf', content: 'AAAAAAAAAA' },
    { path: 'Qwen3-30B-A3B-Q4_K_M-00002-of-00003.gguf', content: '0123456789' },
    { path: 'Qwen3-30B-A3B-Q4_K_M-00003-of-00003.gguf', content: 'zzzzzzzzzz' },
  ];
  const fetchCounts = new Map();
  const seenRanges = new Map();
  let active = 0;
  let maxActive = 0;
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify(SHARDS.map((s) => ({ type: 'file', path: s.path, size: 10, lfs: { size: 10 } }))), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/resolve/main/')) {
      const name = decodeURIComponent(u.split('/resolve/main/')[1]);
      const shard = SHARDS.find((s) => s.path === name);
      assert.ok(shard, `不应请求清单外文件：${name}`);
      fetchCounts.set(name, (fetchCounts.get(name) || 0) + 1);
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((r) => setTimeout(r, 30)); // 撑开时间窗，让并行可见
      active -= 1;
      const range = opts.headers?.range || opts.headers?.Range || '';
      if (range) {
        seenRanges.set(name, range);
        const from = Number(range.match(/bytes=(\d+)-/)?.[1] || 0);
        return new Response(shard.content.slice(from), {
          status: 206,
          headers: { 'content-type': 'application/octet-stream', 'content-length': String(10 - from) },
        });
      }
      return new Response(shard.content, { headers: { 'content-type': 'application/octet-stream', 'content-length': '10' } });
    }
    throw new Error(`测试中不允许访问：${u}`);
  };
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    fetchImpl,
    env: {},
    logger: { warn() {}, log() {}, error() {} },
  });
  // 预置磁盘状态：第 1 片完整、第 2 片残片
  const qdir = engine.modelDir(MODEL_ID, 'Q4_K_M');
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, SHARDS[0].path), SHARDS[0].content);
  await writeFile(path.join(qdir, `${SHARDS[1].path}.part`), SHARDS[1].content.slice(0, 4));

  const job = await engine.download(MODEL_ID, 'Q4_K_M');
  const deadline = Date.now() + 10000;
  while (job.state === 'downloading' && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 20));
  }
  assert.equal(job.state, 'done');
  // 第 1 片一字节都不许再请求；第 2 片必须按 4 字节续传而不是从头
  assert.equal(fetchCounts.get(SHARDS[0].path) || 0, 0);
  assert.equal(seenRanges.get(SHARDS[1].path), 'bytes=4-');
  assert.equal(fetchCounts.get(SHARDS[2].path), 1);
  // 3 个分片至少有 2 路同时在拉（并行而非排队）
  assert.ok(maxActive >= 2, `期望并行下载，实际最大并发 ${maxActive}`);
  // 三片内容完整、清单已登记
  for (const s of SHARDS) {
    assert.equal(await readFile(path.join(qdir, s.path), 'utf8'), s.content);
  }
  const downloaded = await engine.downloadedQuants(MODEL_ID);
  assert.deepEqual(downloaded.map((d) => d.quant), ['Q4_K_M']);
});
