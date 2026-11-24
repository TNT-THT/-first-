// 浏览器通道：浏览器取 Hugging Face 文件树/字节流，本地服务分组与落盘。
// 覆盖量化清单预览、prepare→chunk→finish 全链、断点续传、偏移错位保护、
// 收尾尺寸校验与恶意路径过滤。服务端一侧用假 HF，不访问真实网络。
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { NativeEngine } from '../server/native/engine.js';

const PASSWORD = 'correct-horse-battery';
const ADMIN = 'manager-secret-test-password';
const MODEL_ID = 'Qwen3-32B';
const REPO = 'bartowski/Qwen3-32B-GGUF';
const CONTENT = Buffer.from('0123456789'); // 10 字节假权重

const TREE = [
  { type: 'file', path: 'Qwen3-32B-Q4_K_M.gguf', size: 10, lfs: { size: 10 } },
  { type: 'file', path: 'README.md', size: 3 },
];

function stubFetch() {
  return async (url) => {
    const u = String(url);
    if (u === `https://huggingface.co/api/models/${REPO}/tree/main?recursive=true`) {
      return new Response(JSON.stringify(TREE), { headers: { 'content-type': 'application/json' } });
    }
    throw new Error(`测试中不允许访问：${u}`);
  };
}

async function setup(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-bd-'));
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    fetchImpl: stubFetch(),
    env: {},
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

async function login(base, url, password) {
  const res = await fetch(`${base}${url}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ password }) });
  assert.equal(res.status, 200);
  return (await res.json()).token;
}

const auth = (token) => (token ? { authorization: `Bearer ${token}`, 'content-type': 'application/json' } : { 'content-type': 'application/json' });

async function postChunk(base, admin, { offset, bytes }) {
  return fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/chunk`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${admin}`,
      'content-type': 'application/octet-stream',
      'x-bd-quant': 'Q4_K_M',
      'x-bd-file': encodeURIComponent('Qwen3-32B-Q4_K_M.gguf'),
      'x-bd-offset': String(offset),
    },
    body: bytes,
  });
}

test('浏览器通道：清单分组、全链下载、续传、错位保护、收尾校验', { timeout: 60000 }, async (t) => {
  const { base, engine } = await setup(t);
  const admin = await login(base, '/api/admin/api-keys/login', ADMIN);
  const session = null; // 入口登录已删除：普通接口匿名访问，auth(session) 仅保留 content-type

  // 量化清单：浏览器传文件树，本地分组；含逃逸路径的恶意文件被过滤
  const evilTree = [...TREE, { type: 'file', path: '../evil-Q8_0.gguf', size: 6, lfs: { size: 6 } }];
  const preview = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/quants-preview`, {
    method: 'POST', headers: auth(session), body: JSON.stringify({ files: evilTree }),
  });
  assert.equal(preview.status, 200);
  const previewData = await preview.json();
  assert.deepEqual(previewData.quants.map((q) => q.quant), ['Q4_K_M']);
  // 清单预览匿名可读（入口锁已删除后）
  const noAuth = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/quants-preview`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ files: TREE }),
  });
  assert.equal(noAuth.status, 200, '清单预览匿名可读（入口锁已删除）');
  // 服务端直连清单（兜底）与浏览器通道结果一致
  const direct = await (await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/quants`, { headers: auth(session) })).json();
  assert.deepEqual(direct.quants.map((q) => q.quant), previewData.quants.map((q) => q.quant));

  // 准备：返回下载地址与偏移；非管理员不能准备
  const prepRes = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/prepare`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M', files: TREE }),
  });
  assert.equal(prepRes.status, 200);
  const prep = await prepRes.json();
  assert.equal(prep.state, 'downloading');
  assert.equal(prep.files.length, 1);
  assert.equal(prep.files[0].offset, 0);
  assert.match(prep.files[0].url, /huggingface\.co\/bartowski\/Qwen3-32B-GGUF\/resolve\/main\//);

  // 偏移错位：409 且带真实落盘偏移，前端据此校准
  const bad = await postChunk(base, admin, { offset: 999, bytes: CONTENT.subarray(0, 4) });
  assert.equal(bad.status, 409);
  assert.equal((await bad.json()).received, 0);

  // 传前 4 字节后中断（标记失败），.part 保留
  const c1 = await postChunk(base, admin, { offset: 0, bytes: CONTENT.subarray(0, 4) });
  assert.equal(c1.status, 200);
  assert.equal((await c1.json()).received, 4);
  const fail = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/fail`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M', error: '模拟中断' }),
  });
  assert.equal(fail.status, 200);

  // 没传完就收尾：409 且提示可续传
  const earlyFinish = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/finish`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M' }),
  });
  assert.equal(earlyFinish.status, 409);
  assert.match((await earlyFinish.json()).error, /续传/);

  // 重新准备：偏移应为已落盘的 4 字节（断点续传）
  const prep2 = await (await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/prepare`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M', files: TREE }),
  })).json();
  assert.equal(prep2.files[0].offset, 4);

  // 续传剩余 6 字节并收尾
  const c2 = await postChunk(base, admin, { offset: 4, bytes: CONTENT.subarray(4) });
  assert.equal(c2.status, 200);
  assert.equal((await c2.json()).received, 10);
  const finish = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/finish`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M' }),
  });
  assert.equal(finish.status, 200);
  assert.equal((await finish.json()).state, 'done');

  // 落盘内容与清单识别
  const saved = await readFile(path.join(engine.modelDir(MODEL_ID, 'Q4_K_M'), 'Qwen3-32B-Q4_K_M.gguf'));
  assert.deepEqual([...saved], [...CONTENT]);
  const downloaded = await engine.downloadedQuants(MODEL_ID);
  assert.deepEqual(downloaded.map((d) => d.quant), ['Q4_K_M']);
  // 恶意文件没有落到任何地方
  await assert.rejects(() => stat(path.join(engine.modelDir(MODEL_ID), 'evil-Q8_0.gguf')));
});

test('浏览器通道：大仓库文件树不再被 256KB 上限拒绝，量化带类型与可直接使用标注', { timeout: 60000 }, async (t) => {
  const { base } = await setup(t);
  const session = null; // 入口登录已删除：普通接口匿名访问，auth(session) 仅保留 content-type
  // 2500 个文件的仓库（请求体 > 300KB）：历史上这里被 express 256KB 上限直接 413，
  // 用户只能回退到连不通的软件直连，表现为“量化列表/下载都用不了”。
  const bigTree = [];
  for (let i = 0; i < 2500; i += 1) {
    bigTree.push({ path: `some/very/long/directory/name/file-${i}-Q4_K_M.gguf`, size: 100 + i, sha256: 'a'.repeat(64) });
  }
  const body = JSON.stringify({ files: bigTree });
  assert.ok(body.length > 256 * 1024, `测试前提：请求体确已超过 256KB（${body.length}）`);
  const preview = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/quants-preview`, {
    method: 'POST', headers: auth(session), body,
  });
  assert.equal(preview.status, 200);
  const data = await preview.json();
  assert.equal(data.quants.length, 1);
  assert.equal(data.quants[0].quant, 'Q4_K_M');
  assert.match(data.quants[0].typeLabel, /GGUF/);
  assert.equal(data.quants[0].bitsPerWeight, 4);
  assert.equal(data.quants[0].directUse, true);
  assert.match(data.quants[0].useNote, /直接加载/);
});

test('浏览器通道：页面关掉后任务不再假装下载中，直连不返回假任务', { timeout: 60000 }, async (t) => {
  const { base, engine } = await setup(t);
  const admin = await login(base, '/api/admin/api-keys/login', ADMIN);
  const files = [{ path: 'Qwen3-32B-Q4_K_M.gguf', size: 10, sha256: '' }];

  // 准备并传 4 字节，然后“关页面”：客户端的 fail 通知发不出来
  const prep = await (await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/prepare`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M', files }),
  })).json();
  assert.equal(prep.state, 'downloading');
  const c1 = await postChunk(base, admin, { offset: 0, bytes: CONTENT.subarray(0, 4) });
  assert.equal(c1.status, 200);

  // 浏览器通道活跃时，直连不许抢同一份文件（更不许返回浏览器任务假装直连在下载）
  const conflict = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/download`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M' }),
  });
  assert.equal(conflict.status, 409);

  // 模拟 3 分钟无人续传（把最后收块时间拨到过去）
  const job = engine.downloads.get(`${MODEL_ID}@Q4_K_M`);
  assert.ok(job, '应有浏览器通道任务');
  job.lastChunkAt = Date.now() - 10 * 60 * 1000;

  // 状态接口：任务转成可重试的中断状态，且不把内部结构（entry/Map）吐给前端
  const st = await (await fetch(`${base}/api/native/status`)).json();
  const dl = st.models.find((m) => m.id === MODEL_ID).downloads.find((d) => d.quant === 'Q4_K_M');
  assert.equal(dl.state, 'error');
  assert.match(dl.error, /续传/);
  assert.equal(dl.via, 'browser');
  assert.equal('entry' in dl, false);
  assert.equal('receivedByFile' in dl, false);

  // 重新准备让任务回到下载中，再用英文裸错标失败：下发给界面前也要换成中文
  await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/prepare`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M', files }),
  });
  const failRaw = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/browser-download/fail`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M', error: 'fetch failed' }),
  });
  assert.equal(failRaw.status, 200);
  const stRaw = await (await fetch(`${base}/api/native/status`)).json();
  const dlRaw = stRaw.models.find((m) => m.id === MODEL_ID).downloads.find((d) => d.quant === 'Q4_K_M');
  assert.doesNotMatch(dlRaw.error, /^fetch failed$/);
  assert.match(dlRaw.error, /连接中断/);

  // 此时再走直连：必须是新的服务端任务（不能再返回那个假“下载中”）
  const direct = await fetch(`${base}/api/native/models/${encodeURIComponent(MODEL_ID)}/download`, {
    method: 'POST', headers: auth(admin), body: JSON.stringify({ quant: 'Q4_K_M' }),
  });
  assert.equal(direct.status, 202);
  // 测试桩里直连下载必失败，最终应落到 error 且通道不再标 browser
  const deadline = Date.now() + 10000;
  let last;
  while (Date.now() < deadline) {
    const s = await (await fetch(`${base}/api/native/status`)).json();
    last = s.models.find((m) => m.id === MODEL_ID).downloads.find((d) => d.quant === 'Q4_K_M');
    if (last && last.state === 'error' && last.via === 'server') break;
    await new Promise((r) => setTimeout(r, 120));
  }
  assert.equal(last.state, 'error');
  assert.equal(last.via, 'server');
});
