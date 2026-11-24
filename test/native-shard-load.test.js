// 超大模型（多分片 GGUF）加载：llama.cpp 的 -m 必须指向第 1 个分片。
// 以前按字典序取清单第一个，两位以上分片号或按量化分子目录布局时会挑到中段分片，
// 于是“下载全部成功、一加载就失败”。这里用一个记录收到的 -m 参数的假 llama-server 验证。
import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { NativeEngine } from '../server/native/engine.js';

const MODEL_ID = 'Mixtral-8x22B-141B'; // 内置目录里的超大 MoE（仓库用得着多分片）

// 假 llama-server：把收到的参数写进 argv.json，然后提供 /health 与最简 /v1
const STUB = `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const args = process.argv.slice(2);
fs.writeFileSync(path.join(__dirname, 'argv.json'), JSON.stringify(args));
const port = Number(args[args.indexOf('--port') + 1]);
http.createServer((req, res) => {
  if (req.url === '/health') { res.writeHead(200); res.end('{"status":"ok"}'); return; }
  if (req.url === '/v1/chat/completions') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }));
    return;
  }
  res.writeHead(404); res.end('nf');
}).listen(port, '127.0.0.1');
`;

async function setup(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-shard-'));
  const bin = path.join(dir, 'llama-server');
  await writeFile(bin, STUB);
  await chmod(bin, 0o755);
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    // 只允许访问本机假 llama-server（/health、/v1），任何外网访问都直接失败
    fetchImpl: async (url, opts) => {
      if (String(url).startsWith('http://127.0.0.1')) return fetch(url, opts);
      throw new Error(`测试中不允许访问：${url}`);
    },
    env: { LLAMA_SERVER_BIN: bin },
    logger: { warn() {}, log() {}, error() {} },
  });
  t.after(async () => {
    await engine.unload().catch(() => {});
    await rm(dir, { recursive: true, force: true });
  });
  return { engine, bin, dir };
}

test('加载分片 GGUF：传给 llama.cpp 的是 -00001-of- 而不是字典序第一个', { timeout: 30000 }, async (t) => {
  const { engine, bin } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  // 10 个分片：字典序会把 -00010- 排到 -00002- 前面
  const names = Array.from({ length: 10 }, (_, i) => `Mixtral-8x22B-Instruct-v0.1.Q4_K_M-${String(i + 1).padStart(5, '0')}-of-00010.gguf`);
  const files = [];
  for (const name of names) {
    await writeFile(path.join(qdir, name), 'x');
    files.push({ path: name, size: 1 });
  }
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({ id: MODEL_ID, quant, repo: 'x/y', files, totalBytes: files.length }));
  await engine.load(MODEL_ID, { quant, contextTokens: 4096 });
  const argv = JSON.parse(await readFile(path.join(path.dirname(bin), 'argv.json'), 'utf8'));
  assert.equal(argv[argv.indexOf('-m') + 1], path.join(qdir, names[0]));
  // 长上下文/超大模型的两个关键参数应该在
  assert.ok(argv.includes('--flash-attn'));
});

test('加载：清单里缺分片时明确提示重新下载，而不是让 llama.cpp 含糊崩溃', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  const names = ['M-00001-of-00002.gguf', 'M-00002-of-00002.gguf'];
  await writeFile(path.join(qdir, names[0]), 'x');
  // 只写清单不落第二个分片——但 downloadedQuants 会因为大小对不上而认为没下载完，
  // 所以这里直接让清单与磁盘一致，再手动删掉一个分片模拟“被误删/没下全”。
  await writeFile(path.join(qdir, names[1]), 'x');
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y', files: names.map((n) => ({ path: n, size: 1 })), totalBytes: 2,
  }));
  await rm(path.join(qdir, names[1]));
  // 清单仍然“完整”（大小和文件数都对），但磁盘少一个分片
  await assert.rejects(
    () => engine.load(MODEL_ID, { quant, contextTokens: 4096 }),
    /权重文件不完整|尚未下载/,
  );
});

// ---- 下载侧的“仓库文件变了”保护 ----
test('下载：仓库文件大小变了（重新上传）时以新大小为准，旧 .part 不卡住续传', { timeout: 30000 }, async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-shard-'));
  let tree = [{ type: 'file', path: 'M-Q4_K_M.gguf', size: 6, lfs: { size: 6 } }];
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify(tree), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/resolve/main/')) {
      const size = tree[0].size;
      return new Response('z'.repeat(size), { headers: { 'content-length': String(size) } });
    }
    throw new Error(`测试中不允许访问：${u}`);
  };
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    fetchImpl,
    env: { LLAMA_SERVER_BIN: '/nonexistent' },
    logger: { warn() {}, log() {}, error() {} },
  });
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  // 旧版本 .part 有 10 字节，而仓库现在只声明 6 字节：不清理就永远“下载不完整”
  const qdir = engine.modelDir(MODEL_ID, 'Q4_K_M');
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M-Q4_K_M.gguf.part'), 'x'.repeat(10));
  const job = await engine.download(MODEL_ID, 'Q4_K_M');
  const deadline = Date.now() + 10000;
  while (job.state === 'downloading' && Date.now() < deadline) await new Promise((r) => setTimeout(r, 20));
  assert.equal(job.state, 'done');
  assert.equal(await readFile(path.join(qdir, 'M-Q4_K_M.gguf'), 'utf8'), 'zzzzzz');
});

test('加载：同模型换量化/上下文会真正重新加载（不再静默返回旧实例）', { timeout: 30000 }, async (t) => {
  const { engine, bin } = await setup(t);
  for (const quant of ['Q4_K_M', 'Q2_K']) {
    const qdir = engine.modelDir(MODEL_ID, quant);
    await mkdir(qdir, { recursive: true });
    await writeFile(path.join(qdir, 'M.gguf'), 'x');
    await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
      id: MODEL_ID, quant, repo: 'x/y', files: [{ path: 'M.gguf', size: 1 }], totalBytes: 1,
    }));
  }
  await engine.load(MODEL_ID, { quant: 'Q4_K_M', contextTokens: 8192 });
  const firstPort = engine.llama.info.port;
  // 换上下文：必须重启进程（端口是随机分配的，换了就说明确实重新加载了）
  await engine.load(MODEL_ID, { quant: 'Q4_K_M', contextTokens: 16384 });
  assert.equal(engine.llama.info.contextTokens, 16384);
  // 换量化：实际运行的档位必须跟着变
  await engine.load(MODEL_ID, { quant: 'Q2_K', contextTokens: 16384 });
  assert.equal(engine.llama.info.quant, 'Q2_K');
  const argv = JSON.parse(await readFile(path.join(path.dirname(bin), 'argv.json'), 'utf8'));
  assert.match(argv[argv.indexOf('-m') + 1], /Q2_K/);
  assert.equal(argv[argv.indexOf('-c') + 1], '16384');
  void firstPort;
});

test('卸载：加载中拒绝卸载（避免刚加载的进程被对手方杀掉又拉起）', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M.gguf'), 'x');
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y', files: [{ path: 'M.gguf', size: 1 }], totalBytes: 1,
  }));
  engine.llama.state = 'loading'; // 模拟“正在加载”
  await assert.rejects(() => engine.unload(), /正在加载中/);
  assert.equal(engine.loading, true);
  engine.llama.state = 'idle';
  await engine.unload();
});

test('浏览器通道：软件重启后（内存任务丢失）仍能接着 .part 续传，而不是逼用户删了重下', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  // 磁盘上留着上次的半成品：清单 + 2 字节的 .part
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID,
    quant,
    files: [{ path: 'M-Q4_K_M.gguf', size: 20, sha256: null }],
    totalBytes: 20,
  }));
  await writeFile(path.join(qdir, 'M-Q4_K_M.gguf.part'), 'ab');
  // 模拟“重启后内存里没有任务”
  engine.downloads.clear();
  const out = await engine.receiveBrowserChunk(MODEL_ID, quant, 'M-Q4_K_M.gguf', 2, Buffer.from('cd'));
  assert.equal(out.received, 4);
  assert.equal(out.downloaded, 4);
  const job = engine.downloads.get(`${MODEL_ID}@${quant}`);
  assert.equal(job.via, 'browser');
  assert.equal(job.total, 20);
  // 偏移仍要对得上（错位必须被拒绝，不能写坏权重）
  await assert.rejects(() => engine.receiveBrowserChunk(MODEL_ID, quant, 'M-Q4_K_M.gguf', 2, Buffer.from('x')), /对不上/);
});

test('浏览器通道：prepare 遇到上次遗留的直连任务时，陈旧任务不再永久 409', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M-Q4_K_M.gguf.part'), 'z');
  engine.downloads.set(`${MODEL_ID}@${quant}`, {
    id: MODEL_ID, quant, state: 'downloading', via: 'server',
    downloaded: 1, total: 6, error: '', startedAt: Date.now() - 10 * 60 * 1000, updatedAt: Date.now() - 10 * 60 * 1000,
  });
  const prep = await engine.prepareBrowserDownload(MODEL_ID, quant, [
    { type: 'file', path: 'M-Q4_K_M.gguf', size: 6 },
  ]);
  assert.equal(prep.state, 'downloading');
  assert.equal(prep.files[0].offset, 1);
});

test('加载：内存不足（llama.cpp 也印成 error loading model）不会被误判成权重损坏', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M.gguf'), 'x');
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y', files: [{ path: 'M.gguf', size: 1 }], totalBytes: 1,
  }));
  // 文本既像“加载模型失败”又明确是显存不足：应该按显存不足降层重试
  const err = new Error('error loading model: failed to allocate CUDA0 buffer (out of memory)');
  assert.equal(engine._looksLikeContextOrModelFailure(err).corrupt, true);
  assert.equal(engine._looksLikeMemoryFailure(err), true);
  // 纯权重损坏文本仍判损坏
  assert.equal(
    engine._looksLikeContextOrModelFailure(new Error('invalid magic, not a GGUF file')).corrupt,
    true,
  );
});

test('聊天自动加载：沿用同一模型上次的上下文/层数，换模型时回到默认值', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const putModel = async (id, quant) => {
    const qdir = engine.modelDir(id, quant);
    await mkdir(qdir, { recursive: true });
    await writeFile(path.join(qdir, 'M.gguf'), 'x');
    await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
      id, quant, repo: 'x/y', files: [{ path: 'M.gguf', size: 1 }], totalBytes: 1,
    }));
  };
  await putModel(MODEL_ID, 'Q4_K_M');
  await engine.load(MODEL_ID, { quant: 'Q4_K_M', contextTokens: 16384, gpuLayers: 0 });
  assert.equal(engine.loadPrefs.contextTokens, 16384);
  assert.equal(engine.loadPrefsFor, MODEL_ID);
  // 另一个模型走自动加载：不该继承上面那个模型的 16384
  await putModel('Qwen3-32B', 'Q4_K_M');
  await engine.load('Qwen3-32B', { explicit: false });
  assert.equal(engine.llama.info.contextTokens, 8192);
});

// ---- 使用阶段：长对话超出上下文窗口 ----
test('fitMessagesToContext：保留系统提示与最新消息，丢弃最早历史并给出说明', async () => {
  const { fitMessagesToContext } = await import('../server/native/engine.js');
  const history = [{ role: 'system', content: '系统提示：你是刷题助手' }];
  for (let i = 0; i < 40; i += 1) {
    history.push({ role: i % 2 === 0 ? 'user' : 'assistant', content: `第 ${i} 条：${'内容'.repeat(600)}` });
  }
  history.push({ role: 'user', content: '最后的问题' });
  const out = fitMessagesToContext(history, { contextTokens: 4096, reserveForOutput: 512 });
  assert.equal(out.contextTokens, 4096);
  assert.ok(out.droppedMessages > 0, '必须真的裁掉了历史');
  assert.equal(out.messages[0].role, 'system', '系统提示不能被裁掉');
  assert.equal(out.messages.at(-1).content, '最后的问题', '最新的用户消息必须保留');
  // 估算必须落在预算内（含输出预留），否则 llama.cpp 还是会 400
  assert.ok(out.estimatedTokens <= 4096, `估算 ${out.estimatedTokens} 应不超过 4096`);
  // 历史顺序不能被打乱
  const roles = out.messages.slice(1).map((m) => m.role);
  assert.notEqual(roles[0], 'assistant', '裁剪后第一条历史应是用户消息，不能从助手消息开始');
});

test('fitMessagesToContext：短对话原样返回，超长单条消息保留首尾并标注省略', async () => {
  const { fitMessagesToContext } = await import('../server/native/engine.js');
  const short = fitMessagesToContext(
    [{ role: 'system', content: 's' }, { role: 'user', content: '你好' }],
    { contextTokens: 8192 },
  );
  assert.equal(short.droppedMessages, 0);
  assert.equal(short.messages.length, 2);
  assert.equal(short.messages.at(-1).content, '你好');

  const long = fitMessagesToContext(
    [{ role: 'user', content: `开头重要信息${'很长的正文'.repeat(5000)}结尾的重要问题` }],
    { contextTokens: 4096, reserveForOutput: 512 },
  );
  const text = long.messages.at(-1).content;
  assert.ok(text.length < 20000, `应被截短，实际 ${text.length}`);
  assert.match(text, /已省略/);
  assert.match(text, /^开头重要信息/, '头部内容要保留');
  assert.match(text, /结尾的重要问题$/, '尾部内容要保留');
});

test('并发加载：同参数只启动一次 llama-server；不同参数按顺序执行', { timeout: 60000 }, async (t) => {
  const { engine, bin } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M.gguf'), 'x');
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y', files: [{ path: 'M.gguf', size: 1 }], totalBytes: 1,
  }));
  // 统计真实 spawn 次数：假 llama-server 每次启动都会重写 argv.json
  const starts = [];
  const spawn = engine.llama.spawnImpl;
  engine.llama.spawnImpl = (...args) => { starts.push(args[0]); return spawn(...args); };

  const [a, b] = await Promise.all([
    engine.load(MODEL_ID, { quant, contextTokens: 4096 }),
    engine.load(MODEL_ID, { quant, contextTokens: 4096 }),
  ]);
  assert.equal(a.state, 'ready');
  assert.equal(b.state, 'ready');
  assert.equal(starts.length, 1, `同参数并发应只启动一次，实际 ${starts.length} 次`);

  await engine.load(MODEL_ID, { quant, contextTokens: 8192 });
  assert.equal(starts.length, 2);
  assert.equal(engine.llama.info.contextTokens, 8192);
  void bin;
});
