// 构建脚本的下载重试：Release 不能因为构建机网络抖一下就没产物。
//
// v1.0.20 和 v1.0.22 两次 Release 都少了 `LLM.-.1.0.exe` 和 `LLM.-.1.0.zip`——
// 也就是用户真正要下载的那两个。原因在 standalone job：
//     const dl = await fetch(asset.browser_download_url, …)
// 裸 fetch，一次连接失败就抛，没有重试、没有超时。
//
// 证据：失败那次整个步骤只用了 23 秒（连 exe 注入都没跑完），
// 而单独重跑同样的脚本要 48 秒且完全成功 —— 说明失败发生在 fetch 本身，
// 一个字节都没收到，是链路抖动而非代码/参数问题。
import test from 'node:test';
import assert from 'node:assert/strict';
import { fetchWithRetry } from '../standalone/build-standalone.mjs';

test('前几次连接失败会自动重试，最终成功', async () => {
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    if (calls < 3) throw new Error('fetch failed');
    return { ok: true, status: 200 };
  };
  const res = await fetchWithRetry('https://example.test/x', { attempts: 4, fetchImpl, sleep: async () => {} });
  assert.equal(res.ok, true);
  assert.equal(calls, 3, '应重试两次后成功');
});

test('一直连不上时如实抛出最后一个错误，不会假装成功', async () => {
  let calls = 0;
  const fetchImpl = async () => { calls += 1; throw new Error('fetch failed'); };
  await assert.rejects(
    () => fetchWithRetry('https://example.test/x', { attempts: 3, fetchImpl, sleep: async () => {} }),
    /fetch failed/,
  );
  assert.equal(calls, 3, '应把配置的次数用完');
});

test('5xx / 429 值得重试', async () => {
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    if (calls === 1) return { ok: false, status: 503 };
    if (calls === 2) return { ok: false, status: 429 };
    return { ok: true, status: 200 };
  };
  const res = await fetchWithRetry('https://example.test/x', { attempts: 4, fetchImpl, sleep: async () => {} });
  assert.equal(res.status, 200);
  assert.equal(calls, 3);
});

test('4xx 不重试（重试也没用，立刻把真实状态码交出去）', async () => {
  let calls = 0;
  const fetchImpl = async () => { calls += 1; return { ok: false, status: 404 }; };
  const res = await fetchWithRetry('https://example.test/x', { attempts: 4, fetchImpl, sleep: async () => {} });
  assert.equal(res.status, 404);
  assert.equal(calls, 1, '404 不该反复请求');
});
