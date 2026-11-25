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
  // 长上下文/超大模型的关键参数应该在
  assert.ok(argv.includes('--flash-attn'), `实际参数：${argv.join(' ')}`);
  assert.ok(argv.includes('--cache-reuse'));
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
  // 清单仍然“完整”（大小和文件数都对），但磁盘少一个分片。
  // 提示要落在“没下全、可续传”上，而不是含糊的崩溃，也不是“尚未下载”。
  await assert.rejects(
    () => engine.load(MODEL_ID, { quant, contextTokens: 4096 }),
    /没下载完|权重文件不完整/,
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

test('流式转发：llama.cpp 在流里报错时，网页/API 能收到明确原因而不是空回答', { timeout: 30000 }, async () => {
  const { streamNativeToHttp } = await import('../server/native/routes.js');
  const sse = [
    'data: {"choices":[{"index":0,"delta":{"content":"前半段"},"finish_reason":null}]}',
    '',
    'data: {"error":{"message":"out of context"}}',
    '',
  ].join('\n');
  const engine = {
    async chatCompletion() {
      return { ok: true, body: (async function* gen() { yield Buffer.from(sse, 'utf8'); })() };
    },
  };
  const chunks = [];
  const headers = {};
  const res = {
    status() {},
    setHeader(k, v) { headers[k] = v; },
    flushHeaders() {},
    write(s) { chunks.push(s); },
    end() {},
  };
  const out = await streamNativeToHttp(engine, {
    model: 'Qwen3-32B', messages: [{ role: 'user', content: 'hi' }], res,
  });
  const text = chunks.join('');
  assert.equal(out.content, '前半段');
  assert.match(text, /llama\.cpp 生成失败：out of context/);
  assert.match(text, /data: \[DONE\]/);
  assert.equal(headers['Content-Type'], 'text/event-stream; charset=utf-8');
});

// ---- llama-server 崩溃：必须把真实死因带出来，并能自愈 ----
// 现场日志（Windows，0xC0000005）：
//   [native] llama-server 加载失败，聊天自动以安全配置重新加载：llama-server 已退出（代码 3221225477…）
//   [chat] unexpected: Error: llama-server 已被停止或替换。
// 成因：exit 回调先把 child 置空，waitHealthy 轮询看到 child 变了就抛“已被停止或替换”，
// 真正的崩溃死因被这句假错误盖掉，上层既看不清原因也无法判断该不该降级重试。

const CRASH_SCRIPT = `#!/usr/bin/env node\nprocess.exit(3221225477);\n`;

test('崩溃：waitHealthy 抛出的是带退出码的真实死因，而不是“已被停止或替换”', { timeout: 30000 }, async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-crash-'));
  const bin = path.join(dir, 'llama-server');
  await writeFile(bin, CRASH_SCRIPT);
  await chmod(bin, 0o755);
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const { LlamaProcess } = await import('../server/native/llama.js');
  const proc = new LlamaProcess({
    fetchImpl: async () => { throw new Error('还没监听'); },
    logger: { warn() {}, log() {}, error() {} },
  });
  const err = await proc.start({
    binary: bin, modelFile: '/x/M-00001-of-00002.gguf', modelId: 'm', quant: 'Q4_K_M', port: 18771,
  }).catch((e) => e);
  assert.match(err.message, /代码 5/, `应带退出码，实际：${err.message}`);
  assert.ok(!/已被停止或替换/.test(err.message), '真实死因不能被“已被停止或替换”覆盖');
  // 这个桩一句话都不输出就退出：必须按“缺 DLL/参数/程序不对”解释，
  // 而不是继续告诉用户“内存不足”——用户正是被这句话误导的。
  assert.match(err.message, /一句话都没输出|缺少 DLL/);
  assert.ok(!/常见原因：内存不足/.test(err.message), '静默退出不能沿用“内存不足”的结论');
  assert.equal(proc.state, 'error');
  assert.equal(proc.status().exitCode, 5);
  assert.equal(proc.status().silentExit, true);
  await proc.stop();
});

test('崩溃：聊天请求自动降级重载，仍然失败时给出可执行建议（502 而不是 unexpected）', { timeout: 60000 }, async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-crash2-'));
  const bin = path.join(dir, 'llama-server');
  await writeFile(bin, CRASH_SCRIPT);
  await chmod(bin, 0o755);
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
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
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M-00001-of-00002.gguf'), 'x');
  await writeFile(path.join(qdir, 'M-00002-of-00002.gguf'), 'x');
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y',
    files: [{ path: 'M-00001-of-00002.gguf', size: 1 }, { path: 'M-00002-of-00002.gguf', size: 1 }],
    totalBytes: 2,
  }));
  const err = await engine.chatCompletion({ model: MODEL_ID, messages: [{ role: 'user', content: 'hi' }] })
    .then(() => null, (e) => e);
  assert.ok(err, '必须抛错');
  assert.equal(err.status, 502, '应是可读的 502，而不是没有 status 的意外异常');
  assert.ok(!/已被停止或替换/.test(err.message));
  assert.match(err.message, /换更小的量化|内存|准备运行库/);
});

test('崩溃后自愈：state===\x27error\x27 时下一条聊天会重新加载，而不是对着尸体转发', { timeout: 60000 }, async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-crash3-'));
  const marker = path.join(dir, 'crashed-once.flag');
  // 第一次启动崩溃，之后正常服务
  const bin = path.join(dir, 'llama-server');
  await writeFile(bin, `#!/usr/bin/env node
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const marker = ${JSON.stringify(marker)};
if (!fs.existsSync(marker)) { fs.writeFileSync(marker, 'x'); process.exit(3221225477); }
const args = process.argv.slice(2);
const port = Number(args[args.indexOf('--port') + 1]);
http.createServer((req, res) => {
  if (req.url === '/health') { res.writeHead(200); res.end('{}'); return; }
  if (req.url === '/v1/chat/completions') {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ choices: [{ index: 0, message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }));
    return;
  }
  res.writeHead(404); res.end('nf');
}).listen(port, '127.0.0.1');
`);
  await chmod(bin, 0o755);
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
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
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M.gguf'), 'x');
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y', files: [{ path: 'M.gguf', size: 1 }], totalBytes: 1,
  }));
  // 管理员先显式设定 16K 上下文（这是用户设定，任何自动降级都不能改它）
  await engine.load(MODEL_ID, { quant, contextTokens: 16384, gpuLayers: 0 }).catch(() => {});
  engine.llama.state = 'error'; // 模拟运行中崩溃
  engine.llama.error = 'llama-server 已退出（代码 3221225477，即 0xC0000005）。';
  const res = await engine.chatCompletion({ model: MODEL_ID, messages: [{ role: 'user', content: 'hi' }] });
  const data = await res.json();
  assert.equal(data.choices[0].message.content, 'ok', '崩溃后应自愈并正常回答');
  assert.equal(engine.llama.state, 'ready');
  // 安全重载不能污染用户设定：上下文仍是 16384，不会被悄悄降成 4096
  assert.equal(engine.loadPrefs?.contextTokens, 16384, '自动降级重载不能覆盖管理员设定的上下文');
});

// ---- 加载前体检：0xC0000005 到底是“内存不够”还是“权重缺片/截断” ----
// 以前两种都只会得到一句“已退出（代码 3221225477）”，用户既不敢删也不敢重下。

const GSTR = (s) => {
  const b = Buffer.from(s, 'utf8');
  const h = Buffer.alloc(8);
  h.writeBigUInt64LE(BigInt(b.length));
  return Buffer.concat([h, b]);
};
const U32 = (v) => {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(v);
  return b;
};
/** 造一个真实结构的 GGUF 头（含架构/层数/上下文，够 readGgufMeta 解析）。 */
function ggufHead({ arch = 'qwen3moe', blockCount = 48, splitCount = 0 } = {}) {
  const parts = [];
  const kv = (k, t, p) => parts.push(GSTR(k), U32(t), p);
  kv('general.architecture', 8, GSTR(arch));
  kv(`${arch}.context_length`, 4, U32(262144));
  kv(`${arch}.block_count`, 4, U32(blockCount));
  if (splitCount) {
    kv('split.count', 4, U32(splitCount));
    kv('split.no', 4, U32(0));
  }
  const head = Buffer.alloc(24);
  head.write('GGUF', 0, 'ascii');
  head.writeUInt32LE(3, 4);
  head.writeBigUInt64LE(0n, 8);
  head.writeBigUInt64LE(BigInt(parts.length / 3), 16);
  // 补到 8KB，模拟“真实体积的分片”（体检对 <4KB 的文件只警告不拦）
  const body = Buffer.concat([head, ...parts]);
  return Buffer.concat([body, Buffer.alloc(Math.max(0, 8192 - body.length))]);
}

/** 准备好一个“已下载”的量化目录，返回 paths。 */
async function putQuant(engine, quant, files) {
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  for (const f of files) await writeFile(path.join(qdir, f.path), f.content);
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y',
    files: files.map((f) => ({ path: f.path, size: f.content.length })),
    totalBytes: files.reduce((n, f) => n + f.content.length, 0),
  }));
  return qdir;
}

test('加载体检：真实大小的入口分片 GGUF 头坏了 → 直接说“这个文件损坏”，不跑十几分钟阶梯', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  await putQuant(engine, quant, [
    { path: 'M-00001-of-00002.gguf', content: Buffer.alloc(9000, 7) }, // 不是 GGUF
    { path: 'M-00002-of-00002.gguf', content: Buffer.alloc(16, 1) },
  ]);
  const err = await engine.load(MODEL_ID, { quant, contextTokens: 4096 }).then(() => null, (e) => e);
  assert.ok(err, '必须抛错');
  assert.equal(err.corrupt, true);
  assert.match(err.message, /GGUF 文件头读不出来|损坏/);
  assert.match(err.message, /删除这一档量化后重新下载/);
  // 测试用的假 llama-server 一次都不该被启动：坏文件交给它只是白崩一次
  assert.equal(engine.llama.state, 'idle');
});

test('加载体检：分片字节数比清单少（下载中断/文件被截断）→ 明确提示重新下载续传', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  const qdir = engine.modelDir(MODEL_ID, quant);
  await mkdir(qdir, { recursive: true });
  await writeFile(path.join(qdir, 'M-00001-of-00002.gguf'), ggufHead());
  await writeFile(path.join(qdir, 'M-00002-of-00002.gguf'), Buffer.alloc(10, 1)); // 只有 10 字节
  await writeFile(path.join(qdir, 'manifest.json'), JSON.stringify({
    id: MODEL_ID, quant, repo: 'x/y',
    files: [
      { path: 'M-00001-of-00002.gguf', size: 8192 },
      { path: 'M-00002-of-00002.gguf', size: 5000 }, // 清单说 5000，磁盘只有 10
    ],
    totalBytes: 13192,
  }));
  const err = await engine.load(MODEL_ID, { quant, contextTokens: 4096 }).then(() => null, (e) => e);
  assert.ok(err);
  // 关键回归：不能报成“量化还没有下载”——用户明明已经把几十 GB 下到硬盘上了
  assert.ok(!/还没有下载/.test(err.message), `不该说“还没下载”：${err.message}`);
  assert.match(err.message, /没下载完|没下完/);
  assert.match(err.message, /续传/);
  assert.equal(engine.llama.state, 'idle');
});

test('加载体检：完好权重不会把用户筛掉，且内存需求被算出来并带进状态', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  await putQuant(engine, quant, [
    { path: 'M-00001-of-00002.gguf', content: ggufHead({ blockCount: 48, splitCount: 2 }) },
    { path: 'M-00002-of-00002.gguf', content: Buffer.alloc(4096, 1) },
  ]);
  await engine.load(MODEL_ID, { quant, contextTokens: 4096 });
  assert.equal(engine.llama.state, 'ready');
  const st = await engine.status();
  assert.ok(st.memory, '状态里要带内存体检结果，面板才能显示“要多少/有多少”');
  assert.ok(st.memory.neededGB > 0);
  assert.match(st.memory.summary, /GB/);
  // 实际生效参数要能看到（默认 fa=auto / cache-reuse 256 / mmap）
  assert.equal(st.running.flags.flashAttn, 'auto');
  assert.equal(st.running.flags.mmap, true);
  assert.equal(st.running.contextTokens, 4096);
});

test('加载体检：文件头声明的分片数比本地多 → 判为分片不全，而不是让 llama.cpp 崩', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  // 头里写共 5 片，实际只有 2 片
  await putQuant(engine, quant, [
    { path: 'M-00001-of-00005.gguf', content: ggufHead({ splitCount: 5 }) },
    { path: 'M-00002-of-00005.gguf', content: Buffer.alloc(4096, 1) },
  ]);
  const err = await engine.load(MODEL_ID, { quant, contextTokens: 4096 }).then(() => null, (e) => e);
  assert.ok(err);
  assert.match(err.message, /分片不全|共 5 个分片/);
  assert.equal(engine.llama.state, 'idle');
});

test('回退阶梯：崩溃后先只关 flash-attention（保住用户上下文），再逐级降层/关 mmap', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const list = engine._buildLoadAttempts({ requested: 40, ctx: 8192 });
  // 第一步必须是原配置，否则等于一上来就先阉割性能
  assert.deepEqual(
    { layers: list[0].layers, ctx: list[0].contextTokens, fa: list[0].flashAttn, reuse: list[0].cacheReuse, mmap: list[0].mmap },
    { layers: 40, ctx: 8192, fa: 'auto', reuse: 256, mmap: true },
  );
  // 第二步只关 flash-attention：层数与上下文都保住
  assert.equal(list[1].flashAttn, 'off');
  assert.equal(list[1].layers, 40);
  assert.equal(list[1].contextTokens, 8192);
  // 最后必须落到“纯 CPU + 不用 mmap”的最保险组合
  const last = list.at(-1);
  assert.equal(last.layers, 0);
  assert.equal(last.mmap, false);
  assert.equal(last.flashAttn, null);
  // 不能有重复配置（重复只会让用户白等）
  const keys = list.map((a) => `${a.layers}|${a.contextTokens}|${a.flashAttn}|${a.cacheReuse}|${a.mmap}|${a.batch}`);
  assert.equal(new Set(keys).size, keys.length);
  // 层数为 0 时不会退化成“只有一组”
  assert.ok(list.length >= 4, `阶梯太短：${list.length}`);
});

test('内存体检：权重远超本机内存时给出“换小量化”的判断（不拦截加载）', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  // 405B 的 Q4 量级：单看数字就应判 tight
  const huge = engine._memoryReport(230e9, 0, 8192);
  assert.equal(huge.tight, true);
  assert.match(huge.summary, /GB/);
  assert.match(huge.summary, /换更小的量化|不够/);
  // 小模型不该被误判
  const tiny = engine._memoryReport(1.5e9, 0, 4096);
  assert.equal(tiny.tight, false);
  assert.ok(tiny.neededGB < huge.neededGB);
});

test('回退阶梯：救命重载（safe）不从原配置重头来，避免把失败配置再跑一遍', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const normal = engine._buildLoadAttempts({ requested: 0, ctx: 4096 });
  const safe = engine._buildLoadAttempts({ requested: 0, ctx: 4096, safe: true });
  assert.ok(safe.length > 0 && safe.length < normal.length, 'safe 阶段必须更短');
  // 第一阶段刚试过的“原配置”不能再出现
  assert.ok(!safe.some((a) => a.flashAttn === 'auto' && a.batch === 0 && a.cacheReuse === 256));
  // 但最保守的那一档必须仍在（否则崩溃后没有任何翻身机会）
  assert.equal(safe.at(-1).mmap, false);
  assert.equal(safe.at(-1).layers, 0);
});

test('救命重载：上一次加载失败的组合不会被原样再跑一遍（记在实例上，成功即清空）', { timeout: 60000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  await putQuant(engine, quant, [{ path: 'M.gguf', content: ggufHead() }]);
  // 先换成“永远崩溃”的 llama-server：第一阶段所有组合都会失败
  const crashBin = path.join(path.dirname(engine.env.LLAMA_SERVER_BIN), 'crash-llama');
  await writeFile(crashBin, '#!/usr/bin/env node\nprocess.exit(3221225477);\n');
  await chmod(crashBin, 0o755);
  engine.env.LLAMA_SERVER_BIN = crashBin;
  const tried = [];
  const origStart = engine.llama.start.bind(engine.llama);
  engine.llama.start = (opts) => {
    tried.push(`${opts.gpuLayers}|${opts.contextTokens}|${opts.flashAttn}|${opts.cacheReuse}|${opts.mmap}|${opts.batch}`);
    return origStart(opts);
  };
  await engine.load(MODEL_ID, { quant, contextTokens: 4096 }).catch(() => {});
  const failed = engine._failedAttempts?.get(`${MODEL_ID}@${quant}`);
  assert.ok(failed && failed.size > 0, '失败的组合必须被记住，否则救命重载会整套重跑');
  const firstRound = tried.length;
  assert.ok(firstRound >= 4, `第一阶段应真的试过多组，实际 ${firstRound}`);
  // 全部组合都失败过了：救命重载不该再原样重跑，而是直接说清“都试过了”+
  // 下一步该干什么（用户不必再等一轮注定失败的启动）。
  const exhausted = await engine.load(MODEL_ID, { quant, contextTokens: 4096, gpuLayers: 0, explicit: false, safe: true })
    .then(() => null, (e) => e);
  assert.ok(exhausted, '全部失败后必须抛错');
  assert.match(exhausted.message, /都试过了/);
  assert.match(exhausted.message, /重新下载|准备运行库/);
  assert.equal(tried.length, firstRound, '全都失败过就不该再启动一次 —— 白等');

  // 只失败了一部分时：救命重载只跑“还没试过”的那几组，一个也不重复
  const keys = [...failed];
  failed.delete(keys[keys.length - 1]); // 假设最后一档还没试过
  // 快照：这次救命重载会重新把失败的组合写回记录，所以要在跑之前取
  const knownFailed = new Set(failed);
  await engine.load(MODEL_ID, { quant, contextTokens: 4096, gpuLayers: 0, explicit: false, safe: true }).catch(() => {});
  const secondRound = tried.slice(firstRound);
  assert.ok(secondRound.length > 0, '还有没试过的组合时必须去试');
  assert.equal(secondRound.filter((k) => knownFailed.has(k)).length, 0, `救命重载重复了已失败的组合：${secondRound.join('、')}`);
  // 换个能成功的 llama-server：加载成功后失败记录要清空（否则下次会少试好几档）
  engine.env.LLAMA_SERVER_BIN = path.join(path.dirname(crashBin), 'llama-server');
  await engine.load(MODEL_ID, { quant, contextTokens: 4096 }).catch(() => {});
  assert.equal(engine._failedAttempts?.get(`${MODEL_ID}@${quant}`), undefined, '加载成功后失败记录必须清空');
});

// ---- 用户报告：16.6 GB 总内存 vs 15.5 GB 需求，软件却说“总内存都不够” ----
// 两个真问题：① 内存公式根本没读层数，KV 缓存被低估；② “总内存够不够”与
// “当前可用够不够”混为一谈，结论自相矛盾；③ 更关键的是：用户那次 llama-server
// 一句日志都没有，根本不是内存问题，软件却一直往内存上引。

test('内存体检：层数参与 KV 计算，且区分“总内存不够”与“当前可用不够”', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  const weights = 13.5e9; // 35B-A3B 的 Q2_K 量级
  const withLayers = engine._memoryReport({ totalBytes: weights, blockCount: 48 }, 0, 8192);
  const noLayers = engine._memoryReport({ totalBytes: weights, blockCount: null }, 0, 8192);
  // 48 层 × 8192 上下文的 KV 必须真的被算进去（以前固定 0.5 GB）
  // 48 层 × 8192 上下文 × (K+V) × 2 字节 × 8 KV 头 × 128 维 ≈ 1.6 GB
  assert.ok(withLayers.kvGB > 1.4, `48 层 8192 上下文的 KV 应约 1.6 GB（以前恒为 0.5），实际 ${withLayers.kvGB}`);
  assert.ok(withLayers.kvGB < 2.2, `KV 估算不应离谱偏高，实际 ${withLayers.kvGB}`);
  assert.ok(withLayers.neededGB > noLayers.neededGB, '读到层数后需求必须更高（说明层数真的参与了计算）');
  // 上下文翻倍，KV 也要翻倍
  const big = engine._memoryReport({ totalBytes: weights, blockCount: 48 }, 0, 16384);
  assert.ok(Math.abs(big.kvGB - withLayers.kvGB * 2) < 0.5, '上下文翻倍时 KV 应近似翻倍');
  // 结论必须自洽：需求没超过总内存时，绝不能报“总内存都不够”
  assert.equal(withLayers.exceedsTotal, withLayers.neededGB > withLayers.totalGB);
  if (!withLayers.exceedsTotal) assert.ok(!/总内存都不够|比整机物理内存还多/.test(withLayers.summary));
  // 超过总内存时，建议必须是“换量化/换机器”；仅仅可用不够时，建议是“关掉别的程序”
  const huge = engine._memoryReport({ totalBytes: 300e9, blockCount: 80 }, 0, 8192);
  assert.equal(huge.exceedsTotal, true);
  assert.match(huge.summary, /换更小的量化|换机器/);
  assert.ok(withLayers.summary.includes('内存'), withLayers.summary);
});

test('诊断：llama-server 一句日志都没有就退出 → 判“运行库/程序问题”，不再说成内存不足', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  // 造一个静默崩溃：不输出任何东西就退出
  engine.llama.logTail = [];
  engine.llama.exitInfo = { code: 3221225477, silent: true };
  engine.llama.state = 'error';
  engine.llama.error = 'llama-server 启动后一句话都没输出就退出了';
  const silent = engine._diagnoseFailure(new Error(engine.llama.error));
  assert.equal(silent.kind, 'silent');
  assert.match(silent.advice, /准备运行库|OJ_LLAMA_VARIANT/);
  // 关键：静默退出不能被判成内存不足
  assert.notEqual(silent.kind, 'memory');
  assert.equal(engine._looksLikeMemoryFailure(new Error('llama-server 已退出（代码 1）')), false,
    '只有“已退出”这一句不足以判定内存不足');

  // 有真实分配失败日志时才算内存问题
  engine.llama.logTail = ['llama_model_load: error loading model: failed to allocate buffer (out of memory)'];
  engine.llama.exitInfo = { code: 3, silent: false };
  assert.equal(engine._diagnoseFailure(new Error('llama-server 已退出（代码 3）')).kind, 'memory');

  // 缺 DLL 的说法要指向运行库
  engine.llama.logTail = ['error while loading shared libraries: libcudart.so.12: cannot open shared object file'];
  const runtime = engine._diagnoseFailure(new Error('llama-server 已退出（代码 127）'));
  assert.equal(runtime.kind, 'runtime');
  assert.match(runtime.advice, /准备运行库|OJ_LLAMA_VARIANT/);
});

test('启动前自检：可执行文件 0 字节时直接判死，并给出重装指引（不必等 6 次崩溃）', { timeout: 30000 }, async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-pre-'));
  const bin = path.join(dir, 'llama-server');
  await writeFile(bin, '');
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const { LlamaProcess } = await import('../server/native/llama.js');
  const proc = new LlamaProcess({
    spawnImpl: () => { throw new Error('不该被启动'); },
    fetchImpl: async () => { throw new Error('不该联网'); },
    logger: { warn() {}, log() {}, error() {} },
  });
  const err = await proc.start({ binary: bin, modelFile: '/x/M.gguf', modelId: 'm', quant: 'Q4_K_M', port: 18779 })
    .then(() => null, (e) => e);
  assert.ok(err, '必须直接失败');
  assert.match(err.message, /0 字节|重新下载/);
});

test('启动前自检：路径不存在时给出明确提示，而不是让 spawn 抛系统错', { timeout: 30000 }, async t => {
  const { LlamaProcess } = await import('../server/native/llama.js');
  const proc = new LlamaProcess({
    spawnImpl: () => { throw new Error('不该被启动'); },
    fetchImpl: async () => { throw new Error('不该联网'); },
    logger: { warn() {}, log() {}, error() {} },
  });
  const err = await proc.start({ binary: '/nonexistent/llama-server', modelFile: '/x/M.gguf', modelId: 'm', quant: 'Q4_K_M', port: 18780 })
    .then(() => null, (e) => e);
  assert.ok(err);
  assert.match(err.message, /找不到 llama-server|准备运行库/);
});

test('启动前自检：探测到版本不支持某参数时自动省略（而不是让它把进程打崩）', { timeout: 30000 }, async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-caps-'));
  const bin = path.join(dir, 'llama-server');
  // 一个“老版本”：--help 里没有 --flash-attn / --cache-reuse
  await writeFile(bin, `#!/usr/bin/env node
const args = process.argv.slice(2);
if (args.includes('--help')) {
  console.log('usage: llama-server [options]\\n  -m, --model PATH\\n  --host HOST\\n  --port PORT\\n  -c, --ctx-size N\\n  -ngl N\\n  --jinja');
  process.exit(0);
}
require('node:fs').writeFileSync(require('node:path').join(__dirname, 'argv.json'), JSON.stringify(args));
process.exit(3221225477);
`);
  await chmod(bin, 0o755);
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const { LlamaProcess } = await import('../server/native/llama.js');
  const proc = new LlamaProcess({ fetchImpl: async () => { throw new Error('no'); }, logger: { warn() {}, log() {}, error() {} } });
  await proc.start({ binary: bin, modelFile: '/x/M.gguf', modelId: 'm', quant: 'Q4_K_M', port: 18781, flashAttn: 'auto', cacheReuse: 256 })
    .then(() => null, (e) => e);
  const argv = JSON.parse(await readFile(path.join(dir, 'argv.json'), 'utf8'));
  assert.ok(!argv.includes('--flash-attn'), `不支持的参数必须省略：${argv.join(' ')}`);
  assert.ok(!argv.includes('--cache-reuse'), `不支持的参数必须省略：${argv.join(' ')}`);
  assert.ok(argv.includes('--jinja'), '支持的参数必须保留');
  await proc.stop();
});

test('诊断不自我误导：外层不会把内层生成的解释文本当成新证据（“这不是内存不足”不能被判成内存不足）', { timeout: 30000 }, async (t) => {
  const { engine } = await setup(t);
  // 静默退出的原始证据
  engine.llama.logTail = [];
  engine.llama.exitInfo = { code: 3221225477, silent: true };
  const inner = Object.assign(
    new Error('判断：llama-server 启动即退出且没有任何日志。这通常不是内存不够（内存不足时会先打印加载进度）。'),
    { diag: 'silent' },
  );
  // 已经定过性的错误：必须沿用，不能被句子里“内存不足”四个字带偏
  const again = engine._diagnoseFailure(inner);
  assert.equal(again.kind, 'silent');
  // 我们自己的包装文本不参与“内存不足”判定
  assert.equal(engine._looksLikeMemoryFailure(inner), false);
  // 真正的 out of memory 日志仍然要判内存
  engine.llama.logTail = ['llama_model_load: error loading model: failed to allocate buffer, out of memory'];
  assert.equal(engine._looksLikeMemoryFailure(new Error('llama-server 已退出（代码 3）')), true);
});

test('最终报错不重复刷同一段内存说明，且结论与建议一致', { timeout: 60000 }, async (t) => {
  const { engine } = await setup(t);
  const quant = 'Q4_K_M';
  await putQuant(engine, quant, [{ path: 'M.gguf', content: ggufHead() }]);
  const dir = path.dirname(engine.env.LLAMA_SERVER_BIN);
  // 一句日志都没有就退出的 llama-server（用户现场）
  await writeFile(path.join(dir, 'silent-llama'), '#!/usr/bin/env node\nprocess.exit(3221225477);\n');
  await chmod(path.join(dir, 'silent-llama'), 0o755);
  engine.env.LLAMA_SERVER_BIN = path.join(dir, 'silent-llama');
  const err = await engine.chatCompletion({ model: MODEL_ID, messages: [{ role: 'user', content: 'hi' }] })
    .then(() => null, (e) => e);
  assert.ok(err);
  assert.equal(err.status, 502);
  assert.equal(err.diag, 'silent', `静默退出必须定性为 silent，实际 ${err.diag}`);
  // 同一句“内存情况”不能出现两次
  const memCount = (err.message.match(/内存情况：/g) || []).length;
  assert.ok(memCount <= 1, `内存说明重复了 ${memCount} 次：\n${err.message}`);
  // 静默退出时不该引导用户去换量化（那是内存类故障的建议）
  assert.ok(!/换更小的量化/.test(err.message), `静默退出不该建议换量化：\n${err.message}`);
  assert.match(err.message, /准备运行库|OJ_LLAMA_VARIANT/);
});
