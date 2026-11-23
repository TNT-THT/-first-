// 本机导入全适应测试：浏览器先把文件下到电脑、再导入软件这条路线。
// 覆盖单/分片/多量化 GGUF、原格式快照、混合仓库、zip 整包、无权重拒绝、
// 断点续传（含偏移错位 409 对齐）、重名递增，以及接口权限与前端驱动续传。
// 全程不访问真实网络（引擎 fetchImpl 一律拒绝联网）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { NativeEngine } from '../server/native/engine.js';
import { importLocalModel } from '../web/src/engines/native-client.js';

const PASSWORD = 'correct-horse-battery';
const ADMIN = 'manager-secret-test-password';

async function makeEngine(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-import-'));
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    fetchImpl: async () => { throw new Error('导入测试不允许联网'); },
    env: {},
    logger: { warn() {}, log() {}, error() {} },
  });
  t.after(() => rm(dir, { recursive: true, force: true }));
  return engine;
}

/** 按小片把一组文件喂进导入会话并收尾（模拟浏览器分片上传）。 */
async function importFiles(engine, importId, files, { name, sliceBytes = 4 } = {}) {
  await engine.beginImport(importId, files.map((f) => ({ path: f.path, size: f.data.length })));
  for (const f of files) {
    let off = 0;
    while (off < f.data.length) {
      const out = await engine.writeImportSlice(importId, f.path, off, f.data.subarray(off, off + sliceBytes));
      off = out.received;
    }
  }
  return engine.finishImport(importId, { name });
}

test('导入单个 GGUF：自动识别量化、登记为本地导入、可直接加载', async (t) => {
  const engine = await makeEngine(t);
  const data = Buffer.from('GGUF-fake-weights-0123456789');
  const out = await importFiles(engine, 'imp-test0001', [{ path: 'Qwen-Test-Q4_K_M.gguf', data }]);
  assert.deepEqual(out.quants, ['Q4_K_M']);
  assert.equal(out.snapshot, false);
  assert.match(out.id, /^local\//);
  const model = engine.resolveModel(out.id);
  assert.equal(model.kind, 'import');
  assert.equal(model.vendor, '本地导入');
  const downloaded = await engine.downloadedQuants(out.id);
  assert.equal(downloaded.length, 1);
  assert.equal(downloaded[0].quant, 'Q4_K_M');
  assert.equal(downloaded[0].directUse, true);
  const onDisk = await readFile(path.join(engine.modelDir(out.id, 'Q4_K_M'), 'Qwen-Test-Q4_K_M.gguf'));
  assert.ok(onDisk.equals(data));
});

test('导入分片 GGUF：同量化多分片归为一档', async (t) => {
  const engine = await makeEngine(t);
  const out = await importFiles(engine, 'imp-test0002', [
    { path: 'Test-Q8_0-00001-of-00002.gguf', data: Buffer.from('part-one-') },
    { path: 'Test-Q8_0-00002-of-00002.gguf', data: Buffer.from('part-two-') },
  ]);
  assert.deepEqual(out.quants, ['Q8_0']);
  const downloaded = await engine.downloadedQuants(out.id);
  assert.equal(downloaded.length, 1);
  assert.equal(downloaded[0].files, 2);
  assert.equal(downloaded[0].totalBytes, 18);
});

test('导入多量化混装：每一档都落地', async (t) => {
  const engine = await makeEngine(t);
  const out = await importFiles(engine, 'imp-test0003', [
    { path: 'M-Q4_K_M.gguf', data: Buffer.from('q4-weights') },
    { path: 'M-IQ2_XXS.gguf', data: Buffer.from('iq-weights') },
  ]);
  assert.deepEqual([...out.quants].sort(), ['IQ2_XXS', 'Q4_K_M']);
  const downloaded = await engine.downloadedQuants(out.id);
  assert.deepEqual(downloaded.map((d) => d.quant).sort(), ['IQ2_XXS', 'Q4_K_M']);
});

test('导入认不出量化标记的 GGUF：按文件名归档也不丢', async (t) => {
  const engine = await makeEngine(t);
  const out = await importFiles(engine, 'imp-test0004', [
    { path: 'some-odd-name.gguf', data: Buffer.from('odd-weights') },
  ]);
  assert.equal(out.quants.length, 1);
  const downloaded = await engine.downloadedQuants(out.id);
  assert.equal(downloaded.length, 1);
  assert.equal(downloaded[0].files, 1);
});

test('只有配置/说明文件、没有任何权重：明确报错且不登记', async (t) => {
  const engine = await makeEngine(t);
  const before = engine.custom.size;
  await assert.rejects(
    importFiles(engine, 'imp-test0005', [
      { path: 'config.json', data: Buffer.from('{"architectures":[]}') },
      { path: 'tokenizer.json', data: Buffer.from('{}') },
      { path: 'README.md', data: Buffer.from('# hi') },
    ]),
    /没有找到可识别的模型权重/,
  );
  assert.equal(engine.custom.size, before);
});

test('导入 Safetensors 快照：落 SNAPSHOT 档，本地短路不联网', async (t) => {
  const engine = await makeEngine(t);
  const weights = Buffer.from('safetensors-fake-weights');
  const out = await importFiles(engine, 'imp-test0006', [
    { path: 'config.json', data: Buffer.from('{"architectures":["X"]}'), },
    { path: 'model.safetensors', data: weights },
    { path: 'tokenizer.json', data: Buffer.from('{}') },
  ]);
  assert.deepEqual(out.quants, []);
  assert.equal(out.snapshot, true);
  assert.ok(out.formats.includes('safetensors'));
  const downloaded = await engine.downloadedQuants(out.id);
  assert.equal(downloaded.length, 1);
  assert.equal(downloaded[0].quant, 'SNAPSHOT');
  assert.equal(downloaded[0].directUse, false);
  // 转换/Python 走 _ensureSnapshot：本地已有快照必须直接命中（fetchImpl 联网即抛）
  const dir = await engine._ensureSnapshot(out.id);
  const onDisk = await readFile(path.join(dir, 'model.safetensors'));
  assert.ok(onDisk.equals(weights));
});

test('导入混合仓库：GGUF 与其它权重两条路线都在', async (t) => {
  const engine = await makeEngine(t);
  const out = await importFiles(engine, 'imp-test0007', [
    { path: 'M-Q4_K_M.gguf', data: Buffer.from('gguf-weights') },
    { path: 'model.safetensors', data: Buffer.from('st-weights') },
    { path: 'config.json', data: Buffer.from('{}') },
  ]);
  assert.deepEqual(out.quants, ['Q4_K_M']);
  assert.equal(out.snapshot, true);
  const downloaded = await engine.downloadedQuants(out.id);
  assert.deepEqual(downloaded.map((d) => d.quant).sort(), ['Q4_K_M', 'SNAPSHOT']);
});

/** 手工拼一个 stored（不压缩）zip，字段布局与 archive.js 读者一致。 */
function makeZip(entries) {
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const e of entries) {
    const name = Buffer.from(e.name, 'utf8');
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 8);
    local.writeUInt32LE(e.data.length, 18);
    local.writeUInt32LE(e.data.length, 22);
    local.writeUInt16LE(name.length, 26);
    locals.push(local, name, e.data);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0, 10);
    central.writeUInt32LE(e.data.length, 20);
    central.writeUInt32LE(e.data.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt32LE(offset, 42);
    centrals.push(central, name);
    offset += 30 + name.length + e.data.length;
  }
  const cd = Buffer.concat(centrals);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cd.length, 12);
  eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, cd, eocd]);
}

test('导入 zip 整包：软件内解压再识别，苹果杂物跳过', async (t) => {
  const engine = await makeEngine(t);
  const zip = makeZip([
    { name: 'repo/config.json', data: Buffer.from('{}') },
    { name: 'repo/Zipped-Q4_K_M.gguf', data: Buffer.from('zipped-weights') },
    { name: '__MACOSX/._Zipped-Q4_K_M.gguf', data: Buffer.from('junk') },
    { name: '.DS_Store', data: Buffer.from('junk') },
  ]);
  const out = await importFiles(engine, 'imp-test0008', [{ path: 'model.zip', data: zip }]);
  assert.deepEqual(out.quants, ['Q4_K_M']);
  const downloaded = await engine.downloadedQuants(out.id);
  assert.equal(downloaded.length, 1);
  const onDisk = await readFile(path.join(engine.modelDir(out.id, 'Q4_K_M'), 'repo/Zipped-Q4_K_M.gguf'));
  assert.equal(onDisk.toString(), 'zipped-weights');
});

test('导入断点续传：中断后按已收字节续传，偏移错位 409 对齐不写坏', async (t) => {
  const engine = await makeEngine(t);
  const data = Buffer.from('0123456789ABCDEF'); // 16 字节
  await engine.beginImport('imp-test0009', [{ path: 'Big-Q4_K_M.gguf', size: data.length }]);
  const first = await engine.writeImportSlice('imp-test0009', 'Big-Q4_K_M.gguf', 0, data.subarray(0, 6));
  assert.equal(first.received, 6);
  // 还没传完就 finish：明确报未完成（客户端据此重选文件续传）
  await assert.rejects(engine.finishImport('imp-test0009', {}), /还没传完整/);
  // 重新开始同一组文件：状态如实报告已收 6 字节
  const again = await engine.beginImport('imp-test0009', [{ path: 'Big-Q4_K_M.gguf', size: data.length }]);
  assert.deepEqual(again.files, [{ path: 'Big-Q4_K_M.gguf', size: 16, received: 6 }]);
  // 偏移错位：返回 409 且带真实偏移，服务端不落一个字节
  await assert.rejects(
    engine.writeImportSlice('imp-test0009', 'Big-Q4_K_M.gguf', 2, data.subarray(2, 8)),
    (err) => err.status === 409 && err.received === 6 && /写入位置对不上/.test(err.message),
  );
  const rest = await engine.writeImportSlice('imp-test0009', 'Big-Q4_K_M.gguf', 6, data.subarray(6));
  assert.equal(rest.received, 16);
  const out = await engine.finishImport('imp-test0009', {});
  const onDisk = await readFile(path.join(engine.modelDir(out.id, 'Q4_K_M'), 'Big-Q4_K_M.gguf'));
  assert.ok(onDisk.equals(data)); // 内容逐字节无损，没有被错位写坏
  // 取消导入清暂存：会话没了，再查状态应 404
  await engine.beginImport('imp-test0010', [{ path: 'X-Q4_K_M.gguf', size: 3 }]);
  await engine.cancelImport('imp-test0010');
  await assert.rejects(engine.importSessionStatus('imp-test0010'), /导入会话不存在/);
});

test('导入重名：同名再导入自动递增编号，不覆盖先前的', async (t) => {
  const engine = await makeEngine(t);
  const a = await importFiles(engine, 'imp-test0011', [{ path: 'Same-Q4_K_M.gguf', data: Buffer.from('aaa') }]);
  const b = await importFiles(engine, 'imp-test0012', [{ path: 'Same-Q4_K_M.gguf', data: Buffer.from('bbb') }]);
  assert.notEqual(a.id, b.id);
  assert.match(b.id, /-2$/);
});

test('导入接口权限与 HTTP 全程：未验证 401，管理员可走通且状态可见', { timeout: 30000 }, async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-import-http-'));
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    fetchImpl: async () => { throw new Error('导入测试不允许联网'); },
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
    await new Promise((r) => server.close(r));
    await rm(dir, { recursive: true, force: true });
  });
  const base = `http://127.0.0.1:${server.address().port}`;

  const noAuth = await fetch(`${base}/api/native/import/begin`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ importId: 'imp-http0001', files: [{ path: 'A-Q4_K_M.gguf', size: 4 }] }),
  });
  assert.equal(noAuth.status, 401);

  const adminRes = await fetch(`${base}/api/admin/api-keys/login`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password: ADMIN }),
  });
  assert.equal(adminRes.status, 200);
  const { token } = await adminRes.json();
  const auth = { authorization: `Bearer ${token}` };

  const begin = await fetch(`${base}/api/native/import/begin`, {
    method: 'POST', headers: { ...auth, 'content-type': 'application/json' },
    body: JSON.stringify({ importId: 'imp-http0001', files: [{ path: 'A-Q4_K_M.gguf', size: 4 }] }),
  });
  assert.equal(begin.status, 200);
  assert.deepEqual((await begin.json()).files, [{ path: 'A-Q4_K_M.gguf', size: 4, received: 0 }]);

  const slice = await fetch(`${base}/api/native/import/imp-http0001/file`, {
    method: 'POST',
    headers: { ...auth, 'content-type': 'application/octet-stream', 'x-import-path': encodeURIComponent('A-Q4_K_M.gguf'), 'x-import-offset': '0' },
    body: Buffer.from('GGUF'),
  });
  assert.equal(slice.status, 200);
  assert.equal((await slice.json()).received, 4);

  const finish = await fetch(`${base}/api/native/import/imp-http0001/finish`, {
    method: 'POST', headers: { ...auth, 'content-type': 'application/json' }, body: JSON.stringify({}),
  });
  assert.equal(finish.status, 201);
  const done = await finish.json();
  assert.match(done.id, /^local\//);

  const status = await fetch(`${base}/api/native/status`, { headers: auth });
  assert.equal(status.status, 200);
  const body = await status.json();
  const listed = (body.models || []).find((m) => m.id === done.id);
  assert.ok(listed, '导入的模型应出现在本机模型列表');
  assert.equal(listed.kind, 'import');
  assert.deepEqual(listed.downloaded.map((d) => d.quant), ['Q4_K_M']);
});

/** 前端驱动用的假导入服务器：严格按偏移校验，错位回 409 带真实位置。 */
function fakeImportServer() {
  const stored = new Map(); // path -> Buffer
  const stats = { acceptedBytes: 0, requests: 0 };
  const fetchImpl = async (url, opts = {}) => {
    stats.requests += 1;
    const respond = (status, obj) => ({ ok: status < 300, status, json: async () => obj });
    const u = String(url);
    if (u.endsWith('/api/native/import/begin')) {
      const body = JSON.parse(opts.body);
      return respond(200, {
        importId: body.importId,
        files: body.files.map((f) => ({ path: f.path, size: f.size, received: stored.get(f.path)?.length || 0 })),
      });
    }
    if (u.includes('/file')) {
      const rel = decodeURIComponent(opts.headers['x-import-path']);
      const offset = Number(opts.headers['x-import-offset']);
      const cur = stored.get(rel)?.length || 0;
      if (offset !== cur) return respond(409, { error: '写入位置对不上', received: cur });
      const chunk = Buffer.from(opts.body);
      stored.set(rel, Buffer.concat([stored.get(rel) || Buffer.alloc(0), chunk]));
      stats.acceptedBytes += chunk.length;
      return respond(200, { received: stored.get(rel).length });
    }
    if (u.endsWith('/finish')) {
      return respond(201, { id: 'local/fake', name: 'Fake', formats: ['gguf'], quants: ['Q4_K_M'], snapshot: false });
    }
    return respond(404, { error: 'not found' });
  };
  return { fetchImpl, stored, stats };
}

test('前端导入驱动：中断后重跑同组文件只补未传部分并完成', async () => {
  const server = fakeImportServer();
  const fileA = new Blob([Buffer.from('0123456789')]); // 10
  const fileB = new Blob([Buffer.from('abcdef')]); // 6
  const entries = [
    { file: fileA, path: 'A-Q4_K_M.gguf' },
    { file: fileB, path: 'B-Q4_K_M.gguf' },
  ];
  // 第一次：传到第 9 个字节时网络断掉
  let seen = 0;
  const flaky = async (url, opts) => {
    if (String(url).includes('/file')) {
      seen += opts.body.byteLength;
      if (seen > 9) throw new TypeError('连接中断');
    }
    return server.fetchImpl(url, opts);
  };
  await assert.rejects(
    importLocalModel({ entries, adminToken: 'tok', fetchImpl: flaky, sliceBytes: 4 }),
    /连接中断/,
  );
  assert.ok(server.stats.acceptedBytes >= 4 && server.stats.acceptedBytes < 16);
  // 第二次同一组文件：只补剩下的，不从头传
  const before = server.stats.acceptedBytes;
  const progress = [];
  const out = await importLocalModel({
    entries, adminToken: 'tok', fetchImpl: server.fetchImpl, sliceBytes: 4,
    onProgress: (sent, total) => progress.push([sent, total]),
  });
  assert.equal(out.id, 'local/fake');
  assert.equal(server.stats.acceptedBytes - before, 16 - before);
  assert.equal(server.stored.get('A-Q4_K_M.gguf').toString(), '0123456789');
  assert.equal(server.stored.get('B-Q4_K_M.gguf').toString(), 'abcdef');
  assert.deepEqual(progress.at(-1), [16, 16]);
});
