// 网页端量化列表请求：成功、服务端报错、长时间无响应（超时）都要有明确结果，
// 不能点了按钮后无声无息（用户感知为“没反应”）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { downloadModelViaBrowser, fetchNativeQuants, fetchQuantsViaBrowser } from '../web/src/engines/native-client.js';

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
  const out = await downloadModelViaBrowser({
    id: 'hf/someone/Tiny-GGUF', repo: 'someone/Tiny-GGUF', quant: 'Q4_K_M',
    adminToken: 'adm', fetchImpl: hfFetch, chunkBytes: 4,
  });
  assert.equal(out.state, 'done');
  assert.equal(state.finished, true);
  assert.equal(state.failed, false);
  const joined = Buffer.concat(state.chunks.map((c) => Buffer.from(c))).toString();
  assert.equal(joined, '0123456789');
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
