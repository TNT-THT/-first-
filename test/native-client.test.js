// 网页端量化列表请求：成功、服务端报错、长时间无响应（超时）都要有明确结果，
// 不能点了按钮后无声无息（用户感知为“没反应”）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { downloadModelViaBrowser, fetchModelInfoViaBrowser, fetchNativeQuants, fetchQuantsViaBrowser, parseRepoInput, searchHfViaBrowser, useRepoViaBrowser } from '../web/src/engines/native-client.js';

function withFetch(impl, fn) {
  const orig = globalThis.fetch;
  globalThis.fetch = impl;
  return Promise.resolve()
    .then(fn)
    .finally(() => { globalThis.fetch = orig; });
}

test('fetchNativeQuants：成功返回量化列表', async () => {
  await withFetch(
    async () => new Response(JSON.stringify({ quants: [{ quant: 'Q4_K_M' }] }), { headers: { 'content-type': 'application/json' } }),
    async () => {
      const quants = await fetchNativeQuants('DeepSeek-R1-671B', 'token-abc');
      assert.equal(quants.length, 1);
      assert.equal(quants[0].quant, 'Q4_K_M');
    },
  );
});

test('fetchNativeQuants：服务端错误原样带出中文原因', async () => {
  await withFetch(
    async () => new Response(JSON.stringify({ error: '读取模型仓库失败（HTTP 404）' }), { status: 404, headers: { 'content-type': 'application/json' } }),
    async () => {
      await assert.rejects(() => fetchNativeQuants('x/y', 't'), /仓库失败（HTTP 404）/);
    },
  );
});

test('fetchNativeQuants：服务端迟迟不响应时超时给说法', async () => {
  await withFetch(
    (url, opts) => new Promise((_, reject) => {
      const fail = () => {
        const err = new Error('aborted');
        err.name = 'TimeoutError';
        reject(err);
      };
      opts.signal.addEventListener('abort', fail);
      setTimeout(fail, 60); // 保底：保活事件循环并保证 reject
    }),
    async () => {
      await assert.rejects(() => fetchNativeQuants('x/y', 't', { timeoutMs: 30 }), /读取量化列表超时/);
    },
  );
});

test('fetchQuantsViaBrowser：浏览器取文件树后交本地分组', async () => {
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    calls.push(String(url));
    if (String(url).includes('huggingface.co/api/models/')) {
      return new Response(JSON.stringify([
        { type: 'file', path: 'Tiny-Q4_K_M.gguf', lfs: { size: 10 } },
        { type: 'file', path: 'README.md', size: 3 },
      ]), { headers: { 'content-type': 'application/json' } });
    }
    if (String(url).includes('/quants-preview')) {
      const body = JSON.parse(opts.body);
      assert.equal(body.files[0].size, 10); // lfs size 已归一化
      assert.equal(body.files[0].path, 'Tiny-Q4_K_M.gguf');
      return new Response(JSON.stringify({ quants: [{ quant: 'Q4_K_M' }] }), { headers: { 'content-type': 'application/json' } });
    }
    throw new Error('意外请求：' + url);
  };
  const quants = await fetchQuantsViaBrowser({ id: 'hf/someone/Tiny-GGUF', repo: 'someone/Tiny-GGUF', token: 't', fetchImpl });
  assert.deepEqual(quants, [{ quant: 'Q4_K_M' }]);
  assert.equal(calls.length, 2);
});

/** 一个本地接口桩：记录 chunk 字节流，最终拼出完整内容供断言。 */
function fakeLocalApi({ prepFiles, initialReceived = 0 } = {}) {
  // initialReceived 模拟 .part 里已有字节（断点续传场景），偏移校准都以它为准
  const state = { chunks: [], finished: false, failed: false, received: initialReceived };
  const apiFetch = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes('/browser-download/prepare')) {
      return new Response(JSON.stringify({ id: 'm', quant: 'Q4_K_M', state: 'downloading', total: 10, files: prepFiles }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/browser-download/chunk')) {
      const bytes = new Uint8Array(await new Response(opts.body).arrayBuffer());
      const offset = Number(opts.headers['x-bd-offset']);
      if (offset !== state.received) {
        return new Response(JSON.stringify({ error: '对不上', received: state.received }), { status: 409, headers: { 'content-type': 'application/json' } });
      }
      state.chunks.push(bytes);
      state.received += bytes.length;
      return new Response(JSON.stringify({ received: state.received, downloaded: state.received, total: 10 }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/browser-download/finish')) {
      state.finished = true;
      return new Response(JSON.stringify({ id: 'm', quant: 'Q4_K_M', state: 'done' }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/browser-download/fail')) {
      state.failed = true;
      return new Response(JSON.stringify({ ok: true }), { headers: { 'content-type': 'application/json' } });
    }
    throw new Error('意外本地请求：' + u);
  };
  return { state, apiFetch };
}

test('downloadModelViaBrowser：取树→准备→分块上传→收尾，字节完整有序', async () => {
  const fileUrl = 'https://huggingface.co/someone/Tiny-GGUF/resolve/main/Tiny-Q4_K_M.gguf';
  const { state, apiFetch } = fakeLocalApi({
    prepFiles: [{ path: 'Tiny-Q4_K_M.gguf', size: 10, url: fileUrl, offset: 0, done: false }],
  });
  const hfFetch = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify([{ type: 'file', path: 'Tiny-Q4_K_M.gguf', lfs: { size: 10 } }]), { headers: { 'content-type': 'application/json' } });
    }
    if (u === fileUrl) {
      const stream = new ReadableStream({
        start(c) {
          c.enqueue(new TextEncoder().encode('0123'));
          c.enqueue(new TextEncoder().encode('456789'));
          c.close();
        },
      });
      return new Response(stream, { status: 200 });
    }
    return apiFetch(url, opts);
  };
  const notes = [];
  const out = await downloadModelViaBrowser({
    id: 'hf/someone/Tiny-GGUF', repo: 'someone/Tiny-GGUF', quant: 'Q4_K_M',
    adminToken: 'adm', fetchImpl: hfFetch, chunkBytes: 4, onStatus: (t) => notes.push(t),
  });
  assert.equal(out.state, 'done');
  assert.equal(state.finished, true);
  assert.equal(state.failed, false);
  const joined = Buffer.concat(state.chunks.map((c) => Buffer.from(c))).toString();
  assert.equal(joined, '0123456789');
  // 进度反馈：第一个字节还没传就先有 0%，过程中持续更新，最后明确报完成
  assert.ok(notes.length >= 3, `进度应多次上报，实际 ${notes.length} 次`);
  assert.match(notes[1], /（0%）/);
  assert.match(notes.at(-1), /下载完成/);
});

test('downloadModelViaBrowser：断点被忽略（已有偏移却回 200）时报错并标记失败，交给上层回退', async () => {
  const fileUrl = 'https://huggingface.co/someone/Tiny-GGUF/resolve/main/Tiny-Q4_K_M.gguf';
  const { state, apiFetch } = fakeLocalApi({
    prepFiles: [{ path: 'Tiny-Q4_K_M.gguf', size: 10, url: fileUrl, offset: 4, done: false }],
    initialReceived: 4,
  });
  const hfFetch = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify([{ type: 'file', path: 'Tiny-Q4_K_M.gguf', lfs: { size: 10 } }]), { headers: { 'content-type': 'application/json' } });
    }
    if (u === fileUrl) {
      // 无视 Range，永远整文件 200
      return new Response('0123456789', { status: 200 });
    }
    return apiFetch(url, opts);
  };
  await assert.rejects(
    () => downloadModelViaBrowser({
      id: 'hf/someone/Tiny-GGUF', repo: 'someone/Tiny-GGUF', quant: 'Q4_K_M',
      adminToken: 'adm', fetchImpl: hfFetch, chunkBytes: 4,
    }),
    /Range 被忽略/,
  );
  assert.equal(state.failed, true);
  assert.equal(state.finished, false);
});

test('parseRepoInput：关键词不是仓库，owner/repo 与完整链接才是', () => {
  assert.equal(parseRepoInput('Qwen 3'), null);
  assert.equal(parseRepoInput('someone/Tiny-GGUF'), 'someone/Tiny-GGUF');
  assert.equal(parseRepoInput('https://huggingface.co/someone/Tiny-GGUF/tree/main'), 'someone/Tiny-GGUF');
});

test('searchHfViaBrowser：浏览器取搜索结果，本地服务只做格式判定', async () => {
  const seen = [];
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    seen.push(u);
    if (u.startsWith('https://huggingface.co/api/models?')) {
      assert.match(u, /search=Tiny/);
      return new Response(JSON.stringify([
        { id: 'acme/Tiny-GGUF', author: 'acme', downloads: 5, likes: 1, pipeline_tag: 'text-generation', library_name: '', tags: ['gguf'], gated: false },
        { id: 'acme/Secret', private: true },
      ]), { headers: { 'content-type': 'application/json' } });
    }
    if (u === '/api/native/hf/search-meta') {
      const body = JSON.parse(opts.body);
      assert.equal(body.results.length, 1); // 私有仓库已在浏览器侧过滤
      assert.equal(body.results[0].pipeline, 'text-generation'); // 字段已归一化
      return new Response(JSON.stringify({ results: [{ ...body.results[0], formats: ['gguf'] }] }), { headers: { 'content-type': 'application/json' } });
    }
    throw new Error('意外请求：' + u);
  };
  const out = await searchHfViaBrowser('Tiny', 'tok', { fetchImpl });
  assert.equal(out.results.length, 1);
  assert.deepEqual(out.results[0].formats, ['gguf']);
  assert.equal(seen.length, 2);
});

test('searchHfViaBrowser：粘贴仓库链接时直接取详情并给一条结果', async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u === 'https://huggingface.co/api/models/acme/Tiny-GGUF') {
      return new Response(JSON.stringify({ author: 'acme', downloads: 5, likes: 1, tags: ['gguf'] }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify([{ type: 'file', path: 'Tiny-Q4_K_M.gguf', lfs: { size: 10 } }]), { headers: { 'content-type': 'application/json' } });
    }
    if (u === '/api/native/hf/model-meta') {
      return new Response(JSON.stringify({ repo: 'acme/Tiny-GGUF', name: 'Tiny-GGUF', downloads: 5, likes: 1, formats: ['gguf'], route: { route: 'local' }, files: [], totalGB: 0 }), { headers: { 'content-type': 'application/json' } });
    }
    throw new Error('意外请求：' + u);
  };
  const out = await searchHfViaBrowser('https://huggingface.co/acme/Tiny-GGUF', 'tok', { fetchImpl });
  assert.equal(out.direct, 'acme/Tiny-GGUF');
  assert.equal(out.results.length, 1);
  assert.deepEqual(out.results[0].formats, ['gguf']);
});

test('fetchModelInfoViaBrowser：meta 与文件树分开取回，回传本地判定时文件清单已瘦身', async () => {
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === 'https://huggingface.co/api/models/acme/Tiny-GGUF') {
      return new Response(JSON.stringify({ author: 'acme', downloads: 5, likes: 1, tags: ['gguf'] }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify([
        { type: 'file', path: 'Tiny-Q4_K_M.gguf', lfs: { size: 10, oid: 'deadbeef' } },
        { type: 'file', path: 'config.json', size: 3 },
      ]), { headers: { 'content-type': 'application/json' } });
    }
    if (u === '/api/native/hf/model-meta') {
      const body = JSON.parse(opts.body);
      assert.equal(body.files.length, 2); // 格式判定需要全树，不能只留 gguf
      assert.equal(body.files[0].sha256, undefined); // 但哈希等冗余字段不回传
      assert.equal(body.files[0].size, 10);
      return new Response(JSON.stringify({ repo: 'acme/Tiny-GGUF', name: 'Tiny-GGUF', formats: ['gguf'], route: { route: 'local' }, quants: [{ quant: 'Q4_K_M', directUse: true }] }), { headers: { 'content-type': 'application/json' } });
    }
    throw new Error('意外请求：' + u);
  };
  const info = await fetchModelInfoViaBrowser('acme/Tiny-GGUF', 'tok', { fetchImpl });
  assert.equal(info.quants[0].directUse, true);
});

test('useRepoViaBrowser：判定走浏览器取数，GGUF 接着走浏览器下载传完字节', async () => {
  const fileUrl = 'https://huggingface.co/acme/Tiny-GGUF/resolve/main/Tiny-Q4_K_M.gguf';
  const { state, apiFetch } = fakeLocalApi({
    prepFiles: [{ path: 'Tiny-Q4_K_M.gguf', size: 10, url: fileUrl, offset: 0, done: false }],
  });
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === 'https://huggingface.co/api/models/acme/Tiny-GGUF') {
      return new Response(JSON.stringify({ author: 'acme', tags: ['gguf'] }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify([{ type: 'file', path: 'Tiny-Q4_K_M.gguf', lfs: { size: 10 } }]), { headers: { 'content-type': 'application/json' } });
    }
    if (u === fileUrl) {
      return new Response('0123456789', { status: 200 });
    }
    if (u === '/api/native/hf/use') {
      const body = JSON.parse(opts.body);
      assert.equal(body.viaBrowser, true);
      assert.equal(body.files, undefined); // info 嵌套传递
      assert.equal(body.info.files.length, 1);
      return new Response(JSON.stringify({ id: 'hf/acme/Tiny-GGUF', action: 'download', via: 'browser', quant: 'Q4_K_M' }), { status: 202, headers: { 'content-type': 'application/json' } });
    }
    return apiFetch(url, opts);
  };
  const { out, downloadState } = await useRepoViaBrowser({ repo: 'acme/Tiny-GGUF', mode: 'local', adminToken: 'adm', fetchImpl });
  assert.equal(out.action, 'download');
  assert.equal(downloadState, 'browser');
  assert.equal(state.finished, true);
  const joined = Buffer.concat(state.chunks.map((c) => Buffer.from(c))).toString();
  assert.equal(joined, '0123456789');
});

test('searchHfViaBrowser：浏览器连不上 HF 时抛中文错，交给上层回退直连', async () => {
  const fetchImpl = async () => { throw new TypeError('fetch failed'); };
  await assert.rejects(() => searchHfViaBrowser('Tiny', 'tok', { fetchImpl }), /无法连接 Hugging Face/);
});

test('downloadModelViaBrowser：多个分片并行上传，字节仍各自完整有序', async () => {
  const base = 'https://huggingface.co/acme/Tiny-GGUF/resolve/main/';
  const files = [
    { path: 'Tiny-Q4_K_M-00001-of-00002.gguf', size: 6, url: `${base}Tiny-Q4_K_M-00001-of-00002.gguf`, offset: 0, done: false },
    { path: 'Tiny-Q4_K_M-00002-of-00002.gguf', size: 4, url: `${base}Tiny-Q4_K_M-00002-of-00002.gguf`, offset: 0, done: false },
  ];
  const contents = { [files[0].path]: 'ABCDEF', [files[1].path]: 'wxyz' };
  // 按文件分开记账的本地接口桩（并行时各文件的偏移互不干扰）
  const received = new Map(files.map((f) => [f.path, []]));
  let finished = false;
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes('/tree/main')) {
      return new Response(JSON.stringify(files.map((f) => ({ type: 'file', path: f.path, lfs: { size: f.size } }))), { headers: { 'content-type': 'application/json' } });
    }
    if (u.startsWith(base)) {
      const name = decodeURIComponent(u.slice(base.length));
      return new Response(contents[name], { status: 200 });
    }
    if (u.includes('/browser-download/prepare')) {
      return new Response(JSON.stringify({ id: 'm', quant: 'Q4_K_M', state: 'downloading', total: 10, files }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/browser-download/chunk')) {
      const name = decodeURIComponent(opts.headers['x-bd-file']);
      const bytes = new Uint8Array(await new Response(opts.body).arrayBuffer());
      const offset = Number(opts.headers['x-bd-offset']);
      const soFar = received.get(name).reduce((n, c) => n + c.length, 0);
      if (offset !== soFar) {
        return new Response(JSON.stringify({ error: '对不上', received: soFar }), { status: 409, headers: { 'content-type': 'application/json' } });
      }
      received.get(name).push(bytes);
      return new Response(JSON.stringify({ received: soFar + bytes.length, downloaded: soFar + bytes.length, total: 10 }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/browser-download/finish')) {
      finished = true;
      return new Response(JSON.stringify({ id: 'm', quant: 'Q4_K_M', state: 'done' }), { headers: { 'content-type': 'application/json' } });
    }
    throw new Error('意外请求：' + u);
  };
  const out = await downloadModelViaBrowser({
    id: 'hf/acme/Tiny-GGUF', repo: 'acme/Tiny-GGUF', quant: 'Q4_K_M',
    adminToken: 'adm', fetchImpl, chunkBytes: 2,
  });
  assert.equal(out.state, 'done');
  assert.equal(finished, true);
  for (const f of files) {
    const joined = Buffer.concat(received.get(f.path).map((c) => Buffer.from(c))).toString();
    assert.equal(joined, contents[f.path]);
  }
});

test('useRepoViaBrowser：浏览器下载失败时回退直连且带回真实原因，不许只剩 fetch failed', async () => {
  let treeCalls = 0;
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u === 'https://huggingface.co/api/models/acme/Tiny-GGUF') {
      return new Response(JSON.stringify({ author: 'acme', tags: ['gguf'] }), { headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/tree/main')) {
      treeCalls += 1;
      // /use 前的第一次取树成功；真正下载时再取树断网——模拟浏览器通道中途失败
      if (treeCalls > 1) throw new TypeError('fetch failed');
      return new Response(JSON.stringify([{ type: 'file', path: 'Tiny-Q4_K_M.gguf', lfs: { size: 10 } }]), { headers: { 'content-type': 'application/json' } });
    }
    if (u === '/api/native/hf/use') {
      return new Response(JSON.stringify({ id: 'hf/acme/Tiny-GGUF', action: 'download', via: 'browser', quant: 'Q4_K_M' }), { status: 202, headers: { 'content-type': 'application/json' } });
    }
    if (u.includes('/models/hf%2Facme%2FTiny-GGUF/download')) {
      return new Response(JSON.stringify({ id: 'hf/acme/Tiny-GGUF', quant: 'Q4_K_M', state: 'downloading' }), { status: 202, headers: { 'content-type': 'application/json' } });
    }
    throw new Error('意外请求：' + u);
  };
  const { downloadState, downloadError } = await useRepoViaBrowser({ repo: 'acme/Tiny-GGUF', mode: 'local', adminToken: 'adm', fetchImpl });
  assert.equal(downloadState, 'fallback');
  assert.match(downloadError, /无法连接 Hugging Face/);
});
