// 文本模型并行预下载：12 路并发、跳过已缓存分片、暂时性错误重试、永久错误记录后继续（内存版 Cache Storage 与 fetch）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  TEXT_CACHE_SCOPE,
  TEXT_PREFETCH_CONCURRENCY,
  isPrefetchableRecord,
  prefetchTextModelShards,
  textModelBaseUrl,
} from '../web/src/engines/text-prefetch.js';

function memoryCaches() {
  const store = new Map();
  return {
    async open(name) {
      if (!store.has(name)) store.set(name, new Map());
      const m = store.get(name);
      return {
        async keys() {
          return [...m.keys()].map((url) => ({ url }));
        },
        async match(url) {
          const key = typeof url === 'string' ? url : url.url;
          if (!m.has(key)) return undefined;
          const e = m.get(key);
          return new Response(e.bytes.slice(), { status: 200, headers: e.headers });
        },
        async put(url, res) {
          if (!res.ok) throw new TypeError('bad response');
          m.set(url, { bytes: new Uint8Array(await res.arrayBuffer()), headers: Object.fromEntries(res.headers) });
        },
        async delete(url) {
          const key = typeof url === 'string' ? url : url.url;
          return m.delete(key);
        },
      };
    },
  };
}

const BASE = 'https://huggingface.co/org/model/resolve/main/';
const instantSleep = async () => {};

function manifest(count) {
  return {
    records: Array.from({ length: count }, (_, i) => ({ dataPath: `params_shard_${i}.bin`, nbytes: 10 })),
  };
}

/** 模拟 HF：清单与分片。faults: { 分片名: 状态码 | 'network' }，每个故障只出现一次。 */
function server({ count = 30, faults = {}, delayMs = 3 } = {}) {
  const calls = [];
  let inflight = 0;
  let maxInflight = 0;
  const fail = new Map(Object.entries(faults));
  const fetchImpl = async (url, opts = {}) => {
    calls.push(url);
    if (opts.signal?.aborted) throw Object.assign(new Error('aborted'), { name: 'AbortError' });
    const name = url.slice(BASE.length);
    if (name === 'tensor-cache.json') return new Response(JSON.stringify(manifest(count)), { status: 200 });
    inflight += 1;
    maxInflight = Math.max(maxInflight, inflight);
    try {
      await new Promise((r) => setTimeout(r, delayMs));
      if (fail.has(name)) {
        const f = fail.get(name);
        fail.delete(name);
        if (f === 'network') throw new TypeError('network error');
        return new Response('fault', { status: f });
      }
      return new Response(new Uint8Array(10), { status: 200 });
    } finally {
      inflight -= 1;
    }
  };
  fetchImpl.calls = calls;
  Object.defineProperty(fetchImpl, 'maxInflight', { get: () => maxInflight });
  return fetchImpl;
}

test('默认并发为 12，缓存作用域为 webllm/model', () => {
  assert.equal(TEXT_PREFETCH_CONCURRENCY, 12);
  assert.equal(TEXT_CACHE_SCOPE, 'webllm/model');
});

test('分片全部下载到 webllm/model，键为完整地址（与 WebLLM 读取的键一致）', async () => {
  const storage = memoryCaches();
  const f = server({ count: 5 });
  const r = await prefetchTextModelShards({ modelUrl: BASE, storage, fetchImpl: f, sleep: instantSleep });
  assert.deepEqual(r.failed, []);
  const cache = await storage.open(TEXT_CACHE_SCOPE);
  const keys = (await cache.keys()).map((k) => k.url).sort();
  const expected = [BASE + 'tensor-cache.json', ...Array.from({ length: 5 }, (_, i) => `${BASE}params_shard_${i}.bin`)].sort();
  assert.deepEqual(keys, expected);
});

test('并发：同时最多 12 个分片请求，且不止 1 个', async () => {
  const storage = memoryCaches();
  const f = server({ count: 40, delayMs: 5 });
  await prefetchTextModelShards({ modelUrl: BASE, storage, fetchImpl: f, sleep: instantSleep });
  assert.ok(f.maxInflight >= 2, `最大并发为 ${f.maxInflight}`);
  assert.ok(f.maxInflight <= 12, `最大并发为 ${f.maxInflight}，超过 12`);
});

test('已缓存的分片不再下载（续传）', async () => {
  const storage = memoryCaches();
  await prefetchTextModelShards({ modelUrl: BASE, storage, fetchImpl: server({ count: 4 }), sleep: instantSleep });
  const second = server({ count: 4 });
  const r = await prefetchTextModelShards({ modelUrl: BASE, storage, fetchImpl: second, sleep: instantSleep });
  assert.equal(r.cached, 4);
  assert.deepEqual(
    second.calls.filter((u) => u.includes('params_shard_')),
    [],
    '第二次不应请求任何分片',
  );
});

test('暂时性错误（503、网络中断）自动重试，不报错', async () => {
  const storage = memoryCaches();
  const f = server({ count: 3, faults: { 'params_shard_1.bin': 503, 'params_shard_2.bin': 'network' } });
  const waits = [];
  const r = await prefetchTextModelShards({
    modelUrl: BASE,
    storage,
    fetchImpl: f,
    sleep: async (ms) => {
      waits.push(ms);
    },
  });
  assert.deepEqual(r.failed, []);
  assert.equal(waits.length, 2, '两个分片各重试一次');
  assert.equal(r.cached, 3);
});

test('永久错误（404）记录后继续下载其他分片，不抛出', async () => {
  const storage = memoryCaches();
  const f = server({ count: 3, faults: { 'params_shard_0.bin': 404 } });
  const r = await prefetchTextModelShards({ modelUrl: BASE, storage, fetchImpl: f, sleep: instantSleep });
  assert.deepEqual(r.failed, [BASE + 'params_shard_0.bin']);
  assert.equal(r.cached, 2, '3 个分片中 2 个成功');
});

test('取消：抛出 AbortError', async () => {
  const storage = memoryCaches();
  const ctl = new AbortController();
  ctl.abort();
  await assert.rejects(
    prefetchTextModelShards({ modelUrl: BASE, storage, fetchImpl: server({ count: 2 }), signal: ctl.signal, sleep: instantSleep }),
    (err) => err.name === 'AbortError',
  );
});

test('进度：报告分片数与字节比例，结束时为 1', async () => {
  const storage = memoryCaches();
  const reports = [];
  await prefetchTextModelShards({
    modelUrl: BASE,
    storage,
    fetchImpl: server({ count: 4 }),
    sleep: instantSleep,
    onProgress: (p) => reports.push(p),
  });
  assert.equal(reports[0].progress, 0);
  assert.equal(reports.at(-1).progress, 1);
  assert.match(reports.at(-1).text, /4\/4 个分片/);
});

test('模型地址：与 WebLLM 相同的规范化（补 "/"，补 resolve/main/）', () => {
  assert.equal(textModelBaseUrl('https://huggingface.co/org/model'), 'https://huggingface.co/org/model/resolve/main/');
  assert.equal(textModelBaseUrl('https://huggingface.co/org/model/'), 'https://huggingface.co/org/model/resolve/main/');
  assert.equal(textModelBaseUrl('https://huggingface.co/org/model/resolve/v2/'), 'https://huggingface.co/org/model/resolve/v2/');
  assert.equal(textModelBaseUrl(''), null);
});

test('只预下载 https 的真实地址；本地“新建 AI”的虚拟地址不下载', () => {
  assert.equal(isPrefetchableRecord({ model: 'https://huggingface.co/org/model' }), true);
  assert.equal(isPrefetchableRecord({ model: 'https://local.oj-ai.invalid/models/abc' }), false);
  assert.equal(isPrefetchableRecord({ model: 'http://example.com/m' }), false);
  assert.equal(isPrefetchableRecord({}), false);
});

/** 支持 Range 的分片服务器；第一次请求 params_shard_0.bin 的 Range 时，先送出 dropAt 个字节后断开。 */
function rangeServer({ dropAt }) {
  const calls = [];
  let dropped = false;
  const data = Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const fn = async (url, opts = {}) => {
    const name = url.slice(BASE.length);
    if (name === 'tensor-cache.json') return new Response(JSON.stringify(manifest(1)), { status: 200 });
    const range = opts.headers?.Range;
    calls.push({ name, range: range || null });
    let start = 0;
    let end = data.length - 1;
    if (range) {
      const m = /bytes=(\d+)-(\d+)/.exec(range);
      start = Number(m[1]);
      end = Number(m[2]);
    }
    const slice = data.slice(start, end + 1);
    const drop = !dropped && range && name === 'params_shard_0.bin';
    if (drop) dropped = true;
    let sent = false;
    const stream = new ReadableStream(
      {
        pull(c) {
          if (!drop) {
            c.enqueue(slice);
            c.close();
            return;
          }
          if (!sent) {
            sent = true;
            c.enqueue(slice.slice(0, dropAt));
            return;
          }
          c.error(new TypeError('network error'));
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(stream, { status: range ? 206 : 200 });
  };
  fn.calls = calls;
  return fn;
}

test('网络中断：分片已收到的字节保存，下次从断点用 Range 续传', async () => {
  const storage = memoryCaches();
  const f = rangeServer({ dropAt: 2 });
  const r = await prefetchTextModelShards({
    modelUrl: BASE,
    storage,
    fetchImpl: f,
    partBytes: 4,
    concurrency: 1,
    sleep: instantSleep,
  });
  assert.deepEqual(r.failed, []);
  const ranges = f.calls.filter((c) => c.name === 'params_shard_0.bin').map((c) => c.range);
  assert.deepEqual(ranges, ['bytes=0-3', 'bytes=2-3', 'bytes=4-7', 'bytes=8-9'], '中断处之后从第 2 个字节继续');
  const cache = await storage.open(TEXT_CACHE_SCOPE);
  const res = await cache.match(BASE + 'params_shard_0.bin');
  assert.deepEqual(new Uint8Array(await res.arrayBuffer()), Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]));
});

test('连接卡住（既无数据也不报错）：超过 stallMs 后中止，已收到的字节保存，自动重试续传', async () => {
  const storage = memoryCaches();
  const full = Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const calls = [];
  let stalled = false;
  const fetchImpl = async (url, opts = {}) => {
    const name = url.slice(BASE.length);
    const range = opts.headers?.Range || null;
    if (name === 'tensor-cache.json') return new Response(JSON.stringify(manifest(1)), { status: 200 });
    calls.push({ name, range });
    if (!range && !stalled) {
      stalled = true;
      const body = new ReadableStream(
        {
          pull(c) {
            if (!body.sent) {
              body.sent = true;
              c.enqueue(full.slice(0, 4));
              return undefined;
            }
            return new Promise((_, reject) => {
              opts.signal?.addEventListener('abort', () => {
                c.error(Object.assign(new Error('aborted'), { name: 'AbortError' }));
                reject(new Error('aborted'));
              });
            });
          },
        },
        { highWaterMark: 0 },
      );
      return new Response(body, { status: 200 });
    }
    const m = range ? /bytes=(\d+)-(\d+)/.exec(range) : null;
    return new Response(m ? full.slice(Number(m[1]), Number(m[2]) + 1) : full.slice(), { status: m ? 206 : 200 });
  };
  const r = await prefetchTextModelShards({
    modelUrl: BASE,
    storage,
    fetchImpl,
    concurrency: 1,
    sleep: instantSleep,
    stallMs: 50,
  });
  assert.deepEqual(r.failed, []);
  const ranges = calls.filter((c) => c.name === 'params_shard_0.bin').map((c) => c.range);
  assert.deepEqual(ranges, [null, 'bytes=4-9'], '卡住前收到的 4 字节不重下');
  const cache = await storage.open(TEXT_CACHE_SCOPE);
  const res = await cache.match(BASE + 'params_shard_0.bin');
  assert.deepEqual(new Uint8Array(await res.arrayBuffer()), full);
});

/** 3 个分片（各 16 字节），按 4 字节一块返回；记录每次请求；receivedLimit 之后用户取消。 */
function streamServer({ sizes = [16, 16, 16], chunk = 4 } = {}) {
  const data = sizes.map((n, i) => Uint8Array.from({ length: n }, (_, k) => (k + i * 37) % 251));
  const calls = [];
  const fetchImpl = async (url, opts = {}) => {
    const name = url.slice(BASE.length);
    if (name === 'tensor-cache.json') {
      return new Response(JSON.stringify({ records: sizes.map((_, i) => ({ dataPath: `params_shard_${i}.bin`, nbytes: sizes[i] })) }), { status: 200 });
    }
    const i = Number(/params_shard_(\d+)/.exec(name)[1]);
    const range = opts.headers?.Range || null;
    calls.push({ name, range, at: Date.now() });
    const m = range ? /bytes=(\d+)-(\d+)/.exec(range) : null;
    const start = m ? Number(m[1]) : 0;
    const end = m ? Number(m[2]) : sizes[i] - 1;
    let pos = start;
    const body = new ReadableStream(
      {
        async pull(c) {
          if (opts.signal?.aborted) {
            c.error(Object.assign(new Error('aborted'), { name: 'AbortError' }));
            return;
          }
          if (pos > end) {
            c.close();
            return;
          }
          const n = Math.min(chunk, end + 1 - pos);
          c.enqueue(data[i].slice(pos, pos + n));
          pos += n;
          await new Promise((r) => setTimeout(r, 2));
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(body, { status: m ? 206 : 200 });
  };
  fetchImpl.calls = calls;
  return fetchImpl;
}

test('取消后不再有后台下载：等待中的分片不会在取消后发起请求', async () => {
  const storage = memoryCaches();
  const f = streamServer();
  const ctl = new AbortController();
  await assert.rejects(
    prefetchTextModelShards({
      modelUrl: BASE,
      storage,
      fetchImpl: f,
      signal: ctl.signal,
      concurrency: 1,
      sleep: instantSleep,
      onProgress: ({ progress }) => {
        if (progress >= 0.2) ctl.abort();
      },
    }),
    (err) => err.name === 'AbortError',
  );
  const countAtReject = f.calls.length;
  await new Promise((r) => setTimeout(r, 60));
  assert.equal(f.calls.length, countAtReject, '取消返回之后不再发起任何请求');
});

test('取消后续传：只请求缺少的字节，首次进度等于已保存的比例', async () => {
  const storage = memoryCaches();
  const ctl = new AbortController();
  // 第一次：收到 20 字节（分片 0 全部 16 字节 + 分片 1 的 4 字节）后取消
  await assert.rejects(
    prefetchTextModelShards({
      modelUrl: BASE,
      storage,
      fetchImpl: streamServer(),
      signal: ctl.signal,
      concurrency: 1,
      sleep: instantSleep,
      onProgress: ({ progress }) => {
        if (progress >= 20 / 48) ctl.abort();
      },
    }),
    (err) => err.name === 'AbortError',
  );
  // 第二次：从已保存的位置继续
  const f = streamServer();
  const reported = [];
  await prefetchTextModelShards({
    modelUrl: BASE,
    storage,
    fetchImpl: f,
    sleep: instantSleep,
    onProgress: ({ progress }) => reported.push(progress),
  });
  const bytes = f.calls.reduce((n, c) => n + (c.range ? Number(/bytes=(\d+)-(\d+)/.exec(c.range)[2]) - Number(/bytes=(\d+)-/.exec(c.range)[1]) + 1 : 16), 0);
  assert.equal(bytes, 28, '网络上只传输缺少的 28 字节（48 - 已保存的 20）');
  assert.ok(Math.abs(reported[0] - 20 / 48) < 1e-9, `首次进度为已保存的比例（实际 ${reported[0]}）`);
  const cache = await storage.open(TEXT_CACHE_SCOPE);
  for (let i = 0; i < 3; i++) assert.ok(await cache.match(BASE + `params_shard_${i}.bin`), `分片 ${i} 已完整`);
});
