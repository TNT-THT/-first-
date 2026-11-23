// 网页端量化列表请求：成功、服务端报错、长时间无响应（超时）都要有明确结果，
// 不能点了按钮后无声无息（用户感知为“没反应”）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchNativeQuants } from '../web/src/engines/native-client.js';

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
