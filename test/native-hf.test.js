// HF 桥接全链路：搜索 → 一键使用（GGUF 下载 / Safetensors 转换 / ONNX 云端）→ 聊天。
// 假 Hub API + 假 llama-server + 假转换工具，不访问真实网络、不需要真 Python/torch。
import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { NativeEngine } from '../server/native/engine.js';

const PASSWORD = 'correct-horse-battery';
const ADMIN = 'manager-secret-test-password';

const STUB_LLAMA = `#!/usr/bin/env node
const http = require('node:http');
const args = process.argv.slice(2);
const port = Number(args[args.indexOf('--port') + 1]);
http.createServer((req, res) => {
  if (req.url === '/health') { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"status":"ok"}'); return; }
  if (req.url === '/v1/chat/completions' && req.method === 'POST') {
    let raw = ''; req.on('data', (c) => { raw += c; });
    req.on('end', () => {
      const body = JSON.parse(raw || '{}');
      if (body.stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        res.write('data: ' + JSON.stringify({ id: 'x', object: 'chat.completion.chunk', created: 1, model: 'stub', choices: [{ index: 0, delta: { content: '本地回答' }, finish_reason: null }] }) + '\\n\\n');
        res.write('data: [DONE]\\n\\n'); res.end(); return;
      }
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ id: 'x', object: 'chat.completion', created: 1, model: 'stub', choices: [{ index: 0, message: { role: 'assistant', content: '本地回答' }, finish_reason: 'stop' }], usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 } }));
    });
    return;
  }
  res.writeHead(404); res.end('nf');
}).listen(port, '127.0.0.1');
`;

const FAKE_PYTHON = `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
if (process.argv.includes('--version')) { console.log('Python 3.12.0'); process.exit(0); }
if (process.argv.includes('-c')) process.exit(0); // 依赖探测：假装 transformers/torch 已装
if (process.argv.includes('--outfile')) {
  const out = process.argv[process.argv.indexOf('--outfile') + 1];
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, 'FAKE-GGUF');
  process.exit(0);
}
if (process.argv.includes('--model-dir')) {
  // 扮演 runner.py：/health + /v1/chat/completions
  const port = Number(process.argv[process.argv.indexOf('--port') + 1]);
  http.createServer((req, res) => {
    if (req.url === '/health') { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"status":"ok"}'); return; }
    if (req.url === '/v1/chat/completions' && req.method === 'POST') {
      let raw = ''; req.on('data', (c) => { raw += c; });
      req.on('end', () => {
        const body = JSON.parse(raw || '{}');
        if (body.stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream' });
          res.write('data: ' + JSON.stringify({ id: 'p', object: 'chat.completion.chunk', created: 1, model: 'py', choices: [{ index: 0, delta: { content: 'Python本地回答' }, finish_reason: null }] }) + '\\n\\n');
          res.write('data: [DONE]\\n\\n'); res.end(); return;
        }
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ id: 'p', object: 'chat.completion', created: 1, model: 'py', choices: [{ index: 0, message: { role: 'assistant', content: 'Python本地回答' }, finish_reason: 'stop' }], usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 } }));
      });
      return;
    }
    res.writeHead(404); res.end('nf');
  }).listen(port, '127.0.0.1');
}
`;

const TREES = {
  'someone/Tiny-GGUF': [
    { type: 'file', path: 'Tiny-Q4_K_M.gguf', size: 5, lfs: { size: 5 } },
    { type: 'file', path: 'README.md', size: 3 },
  ],
  'someone/Tiny-ST': [
    { type: 'file', path: 'config.json', size: 10 },
    { type: 'file', path: 'model.safetensors', size: 100, lfs: { size: 100 } },
  ],
  'someone/Tiny-ONNX': [
    { type: 'file', path: 'model.onnx', size: 50, lfs: { size: 50 } },
    { type: 'file', path: 'config.json', size: 10 },
  ],
};

function stubFetch(realFetch = fetch) {
  return async (url, opts) => {
    const u = String(url);
    if (u === 'https://router.huggingface.co/v1/chat/completions') {
      return new Response(JSON.stringify({
        id: 'hf1', object: 'chat.completion', created: 1, model: 'someone/Tiny-ONNX',
        choices: [{ index: 0, message: { role: 'assistant', content: '云端回答' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 },
      }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.startsWith('https://huggingface.co/api/models?')) {
      return new Response(JSON.stringify([
        { id: 'someone/Tiny-GGUF', author: 'someone', downloads: 10, likes: 1, tags: ['gguf'] },
      ]), { headers: { 'content-type': 'application/json' } });
    }
    const infoMatch = u.match(/^https:\/\/huggingface\.co\/api\/models\/([^?]+)$/);
    if (infoMatch && !u.includes('/tree/')) {
      return new Response(JSON.stringify({ author: 'someone', downloads: 10, likes: 1, tags: [] }), { headers: { 'content-type': 'application/json' } });
    }
    const treeMatch = u.match(/\/api\/models\/(.+)\/tree\//);
    if (treeMatch) {
      const repo = decodeURIComponent(treeMatch[1]);
      return new Response(JSON.stringify(TREES[repo] || []), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/resolve/main/')) {
      const name = decodeURIComponent(u.split('/resolve/main/')[1].split('?')[0]);
      const repo = decodeURIComponent(u.split('/resolve/main/')[0].split('/').slice(-2).join('/'));
      const entry = (TREES[repo] || []).find((f) => f.path === name);
      const size = entry?.lfs?.size ?? entry?.size ?? 10;
      return new Response('X'.repeat(size), { headers: { 'content-type': 'application/octet-stream', 'content-length': String(size) } });
    }
    if (u.startsWith('http://127.0.0.1')) return realFetch(url, opts);
    throw new Error(`测试中不允许访问：${u}`);
  };
}

async function setup(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-hf-'));
  const stubBin = path.join(dir, 'llama-server');
  await writeFile(stubBin, STUB_LLAMA);
  await chmod(stubBin, 0o755);
  const fakePy = path.join(dir, 'python');
  await writeFile(fakePy, FAKE_PYTHON);
  await chmod(fakePy, 0o755);
  const fakeScript = path.join(dir, 'convert_hf_to_gguf.py');
  await writeFile(fakeScript, '# fake converter script\n');
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    fetchImpl: stubFetch(),
    env: { LLAMA_SERVER_BIN: stubBin, LLAMA_CONVERT_SCRIPT: fakeScript, PYTHON_BIN: fakePy },
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

const auth = (token) => ({ authorization: `Bearer ${token}`, 'content-type': 'application/json' });

async function waitFor(base, fn, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const s = await (await fetch(`${base}/api/native/status`)).json();
    if (fn(s)) return s;
    await new Promise((r) => setTimeout(r, 120));
  }
  throw new Error('等待状态超时');
}

test('HF 桥接：搜索、GGUF 一键下载并聊天、ONNX 云端、Safetensors 转换', { timeout: 90000 }, async (t) => {
  const { base } = await setup(t);
  const admin = await login(base, '/api/admin/api-keys/login', ADMIN);
  const session = await login(base, '/api/login', PASSWORD);

  // 搜索（桥接）
  const search = await (await fetch(`${base}/api/native/hf/search?q=tiny`, { headers: auth(session) })).json();
  assert.equal(search.results[0].repo, 'someone/Tiny-GGUF');
  const noAuth = await fetch(`${base}/api/native/hf/search?q=tiny`);
  assert.equal(noAuth.status, 401);

  // GGUF：一键使用 → 下载 → 聊天
  const use1 = await (await fetch(`${base}/api/native/hf/use`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ repo: 'someone/Tiny-GGUF' }) })).json();
  assert.equal(use1.action, 'download');
  assert.equal(use1.id, 'hf/someone/Tiny-GGUF');
  await waitFor(base, (s) => s.models.find((m) => m.id === use1.id)?.downloaded?.length);
  const chat1 = await (await fetch(`${base}/api/native/chat`, { method: 'POST', headers: auth(session), body: JSON.stringify({ model: use1.id, messages: [{ role: 'user', content: 'hi' }], stream: false }) })).json();
  assert.equal(chat1.choices[0].message.content, '本地回答');

  // /v1/models 应包含自定义模型
  const models = await (await fetch(`${base}/v1/models`, { headers: auth(admin) })).json();
  assert.ok(models.data.some((m) => m.id === 'hf/someone/Tiny-GGUF'));

  // ONNX（特殊格式）：一键使用默认走本地 Python 运行（下载原格式在本机跑）
  const use2 = await (await fetch(`${base}/api/native/hf/use`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ repo: 'someone/Tiny-ONNX' }) })).json();
  assert.equal(use2.action, 'python');
  assert.equal(use2.runnerReady, true);
  assert.ok(use2.options.some((o) => o.type === 'local-python' && o.local));
  assert.ok(use2.options.some((o) => o.type === 'cloud' && !o.local), '云端应只是可切换选项之一');
  const py = await fetch(`${base}/api/native/hf/python`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ id: use2.id }) });
  assert.equal(py.status, 202);
  await waitFor(base, (s) => s.models.find((m) => m.id === use2.id)?.downloaded?.some((d) => d.quant === 'SNAPSHOT'));
  const chatPy = await (await fetch(`${base}/api/native/chat`, { method: 'POST', headers: auth(session), body: JSON.stringify({ model: use2.id, messages: [{ role: 'user', content: 'hi' }], stream: false }) })).json();
  assert.equal(chatPy.choices[0].message.content, 'Python本地回答');

  // 云端是可切换选项：无 Token 先提示，配 Token 后切到云端并优先于本地
  const useCloud0 = await (await fetch(`${base}/api/native/hf/use`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ repo: 'someone/Tiny-ONNX', mode: 'cloud' }) })).json();
  assert.equal(useCloud0.action, 'needs-token');
  const tok = await fetch(`${base}/api/native/hf/token`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ token: 'hf_test_token' }) });
  assert.equal(tok.status, 200);
  const use3 = await (await fetch(`${base}/api/native/hf/use`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ repo: 'someone/Tiny-ONNX', mode: 'cloud' }) })).json();
  assert.equal(use3.action, 'remote');
  const chat2 = await (await fetch(`${base}/api/native/chat`, { method: 'POST', headers: auth(session), body: JSON.stringify({ model: use3.id, messages: [{ role: 'user', content: 'hi' }], stream: false }) })).json();
  assert.equal(chat2.choices[0].message.content, '云端回答');

  // 再切回本地：点本地运行后聊天重新走本机
  const backLocal = await fetch(`${base}/api/native/hf/python`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ id: use3.id }) });
  assert.equal(backLocal.status, 202);
  const chatBack = await (await fetch(`${base}/api/native/chat`, { method: 'POST', headers: auth(session), body: JSON.stringify({ model: use3.id, messages: [{ role: 'user', content: 'hi' }], stream: false }) })).json();
  assert.equal(chatBack.choices[0].message.content, 'Python本地回答');

  // Safetensors：一键使用 → 转换（假转换工具）→ 本地聊天
  const use4 = await (await fetch(`${base}/api/native/hf/use`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ repo: 'someone/Tiny-ST' }) })).json();
  assert.equal(use4.action, 'convert');
  assert.equal(use4.converterReady, true);
  const conv = await fetch(`${base}/api/native/hf/convert`, { method: 'POST', headers: auth(admin), body: JSON.stringify({ id: use4.id }) });
  assert.equal(conv.status, 202);
  const done = await waitFor(base, (s) => {
    const m = s.models.find((x) => x.id === use4.id);
    return m?.downloaded?.some((d) => d.quant === 'CONVERTED');
  });
  assert.ok(done);
  const chat3 = await (await fetch(`${base}/api/native/chat`, { method: 'POST', headers: auth(session), body: JSON.stringify({ model: use4.id, messages: [{ role: 'user', content: 'hi' }], stream: false }) })).json();
  assert.equal(chat3.choices[0].message.content, '本地回答');
});
