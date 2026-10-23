// 生图模型下载：分段并行、断点续传、自动重试（下载过程不报错）、永久错误在缓存校验时报告（内存版 Cache Storage 与 fetch）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sourceUrls } from '../web/src/image/catalog.js';
import { IMAGE_CACHE, deleteImageModelFiles, downloadImageModel, imageCacheStatus } from '../web/src/image/model-files.js';

function memoryCaches() {
  const store = new Map();
  return {
    _store: store,
    async open(name) {
      if (!store.has(name)) store.set(name, new Map());
      const m = store.get(name);
      return {
        async keys() {
          return [...m.keys()].map((url) => ({ url }));
        },
        async match(url) {
          if (!m.has(url)) return undefined;
          const e = m.get(url);
          return new Response(e.bytes.slice(), { status: 200, headers: e.headers });
        },
        async put(url, res) {
          if (!res.ok) throw new TypeError('bad response');
          m.set(url, { bytes: new Uint8Array(await res.arrayBuffer()), headers: Object.fromEntries(res.headers) });
        },
        async delete(url) {
          return m.delete(url);
        },
      };
    },
  };
}

const DATA = {
  'unet/weights.pb': Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), // 10 字节：partBytes=4 时分 3 段
  'v.json': Uint8Array.from([123, 125]),
};

function model(extra = {}) {
  return {
    id: 'resume-test',
    label: '续传测试',
    family: 'sd15',
    repo: 'org/repo',
    revision: 'main',
    status: 'available',
    files: {
      unetWeights: { path: 'unet/weights.pb', size: 10 },
      vocab: { path: 'v.json', size: 2 },
    },
    ...extra,
  };
}

const abortError = () => Object.assign(new Error('aborted'), { name: 'AbortError' });
const instantSleep = async () => {};

/**
 * 模拟服务器。options:
 * - faults: { 请求序号(从 0 计): 'network' | 'abort' | 状态码数字 }，只在该次请求出现一次
 * - ctl: 配合 'abort' 使用：调用 ctl.abort()（模拟用户取消）
 * - delayMs: 每次请求的延迟（用于统计并行数）
 * - 永久状态码（如 404）在指定序号出现；暂时性状态码（如 503）可重试
 */
function server(options = {}) {
  const calls = [];
  let count = 0;
  let inflight = 0;
  const fn = async (url, opts = {}) => {
    const name = Object.keys(DATA).find((p) => url.endsWith(`/${p}`));
    if (!name) return new Response('nope', { status: 404 });
    const data = DATA[name];
    const range = opts.headers?.Range;
    calls.push({ name, range: range || null });
    if (opts.signal?.aborted) throw abortError();
    const seq = count;
    count += 1;
    inflight += 1;
    fn.maxInflight = Math.max(fn.maxInflight, inflight);
    try {
      if (options.delayMs) await new Promise((r) => setTimeout(r, options.delayMs));
      const fault = options.faults?.[seq];
      if (fault === 'network') throw new TypeError('network error');
      if (fault === 'abort') {
        options.ctl?.abort();
        throw abortError();
      }
      if (typeof fault === 'number') return new Response('fault', { status: fault });
      if (range && !options.noRange) {
        const m = /bytes=(\d+)-(\d+)/.exec(range);
        return new Response(data.slice(Number(m[1]), Number(m[2]) + 1), { status: 206 });
      }
      return new Response(data.slice(), { status: 200 });
    } finally {
      inflight -= 1;
    }
  };
  fn.calls = calls;
  fn.maxInflight = 0;
  return fn;
}

async function cachedBytes(storage, url) {
  const cache = await storage.open(IMAGE_CACHE);
  const res = await cache.match(url);
  return res ? new Uint8Array(await res.arrayBuffer()) : null;
}

test('分段下载：大文件按段下载，结果与原文件一致，分段在完成后删除', async () => {
  const storage = memoryCaches();
  const f = server();
  const st = await downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 4, sleep: instantSleep });
  assert.equal(st.complete, true);
  assert.deepEqual(
    f.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range),
    ['bytes=0-3', 'bytes=4-7', 'bytes=8-9'],
  );
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
  const parts = await storage.open('oj-ai-image-parts-v1');
  assert.equal((await parts.keys()).length, 0);
});

test('暂时性网络错误：自动重试，下载不报错，进度中给出重试信息', async () => {
  const storage = memoryCaches();
  const f = server({ faults: { 1: 'network' } }); // 第 2 次请求（第 2 段）网络中断
  const retries = [];
  const st = await downloadImageModel(model(), {
    storage,
    fetchImpl: f,
    partBytes: 4,
    concurrency: 1,
    sleep: instantSleep,
    onProgress: (p) => p.retry && retries.push(p.retry),
  });
  assert.equal(st.complete, true, '中断后自动续上，最终完整');
  assert.equal(retries.length, 1);
  assert.match(retries[0].message, /network error/);
  assert.equal(retries[0].attempt, 1);
  assert.deepEqual(
    f.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range),
    ['bytes=0-3', 'bytes=4-7', 'bytes=4-7', 'bytes=8-9'],
    '只重试失败的那一段',
  );
});

test('暂时性 HTTP 错误（503）同样自动重试；整文件下载也一样', async () => {
  const storage = memoryCaches();
  const f = server({ faults: { 0: 503, 2: 429 } });
  const st = await downloadImageModel(model(), { storage, fetchImpl: f, sleep: instantSleep });
  assert.equal(st.complete, true);
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).vocab), DATA['v.json']);
});

test('退避间隔按 2 的幂递增，并受上限限制', async () => {
  const storage = memoryCaches();
  const waits = [];
  const ctl = new AbortController();
  const f = server({ faults: { 0: 'network', 1: 'network', 2: 'network', 3: 'network', 4: 'network' } });
  await assert.rejects(
    downloadImageModel(model(), {
      storage,
      fetchImpl: f,
      signal: ctl.signal,
      retryBaseMs: 1,
      retryMaxMs: 3,
      sleep: async (ms, signal) => {
        waits.push(ms);
        if (waits.length === 4) ctl.abort(); // 第 4 次等待后用户取消
        if (signal?.aborted) throw abortError();
      },
    }),
    (err) => err.name === 'AbortError',
  );
  assert.deepEqual(waits, [1, 2, 3, 3]);
});

test('永久错误（404）：不中断，其他文件照常下载；全部完成后在缓存校验时报告，修复后只补下缺失文件', async () => {
  const storage = memoryCaches();
  const f = server({ faults: { 0: 404 } }); // 第一个请求（unet 权重）返回 404
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: f, sleep: instantSleep }),
    (err) => {
      assert.match(err.message, /缓存校验未通过，缺少 1 个文件/);
      assert.match(err.message, /HTTP 404/);
      assert.match(err.message, /unet\/weights\.pb/);
      return true;
    },
  );
  const st = await imageCacheStatus(model(), { storage });
  assert.deepEqual(st.missing, ['unetWeights'], '其他文件已下载');
  assert.ok(st.cached >= 1);
  const retry = server();
  const done = await downloadImageModel(model(), { storage, fetchImpl: retry, sleep: instantSleep });
  assert.equal(done.complete, true);
  assert.equal(retry.calls.filter((c) => c.name === 'v.json').length, 0, '已完成的 vocab 不再请求');
});

test('并行时某段永久错误：其他进行中的段仍会完成并保存，续传只重下失败的段', async () => {
  const storage = memoryCaches();
  // 三段同时发出（并行 4）；第 2 次请求（第 2 段）返回 404，第 1、3 段正常完成
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: server({ faults: { 1: 404 }, delayMs: 5 }), partBytes: 4, concurrency: 4, sleep: instantSleep }),
    /缓存校验未通过/,
  );
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.equal(st.partsCached, 2, '进行中的两段应被保存');
  const retry = server();
  await downloadImageModel(model(), { storage, fetchImpl: retry, partBytes: 4, concurrency: 4, sleep: instantSleep });
  const ranges = retry.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.deepEqual(ranges, ['bytes=4-7'], '只重下失败的第 2 段');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

test('中断（用户取消）后已完成的分段保留，下次只请求缺少的段', async () => {
  const storage = memoryCaches();
  const ctl = new AbortController();
  // 第 0 次请求（第 1 段）成功，第 1 次请求（第 2 段）时用户取消
  await assert.rejects(
    downloadImageModel(model(), {
      storage,
      fetchImpl: server({ faults: { 1: 'abort' }, ctl }),
      signal: ctl.signal,
      partBytes: 4,
      concurrency: 1,
      sleep: instantSleep,
    }),
    (err) => err.name === 'AbortError',
  );
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.equal(st.complete, false);
  assert.equal(st.partsCached, 1);
  assert.equal(st.partialBytes, 4);
  assert.equal(st.doneBytes, 4);

  const retry = server();
  const done = await downloadImageModel(model(), { storage, fetchImpl: retry, partBytes: 4, concurrency: 1, sleep: instantSleep });
  assert.equal(done.complete, true);
  const weightsRanges = retry.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.deepEqual(weightsRanges, ['bytes=4-7', 'bytes=8-9'], '已完成的第 1 段不应重新下载');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

test('并行：同一文件的多个段同时下载', async () => {
  const storage = memoryCaches();
  const f = server({ delayMs: 5 });
  await downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 2, concurrency: 4, sleep: instantSleep });
  assert.ok(f.maxInflight >= 2, `最大并行数为 ${f.maxInflight}`);
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

test('服务器忽略 Range（返回 200）时，改为整文件下载，结果仍然正确', async () => {
  const storage = memoryCaches();
  const f = server({ noRange: true });
  const st = await downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 4, sleep: instantSleep });
  assert.equal(st.complete, true);
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
  const ranges = f.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.equal(ranges.at(-1), null, '最后一次请求是整文件（没有 Range 头）');
});

test('进度：按字节报告，开始时先报告已保存的字节，结束时等于总字节数', async () => {
  const storage = memoryCaches();
  const ctl = new AbortController();
  await assert.rejects(
    downloadImageModel(model(), {
      storage,
      fetchImpl: server({ faults: { 1: 'abort' }, ctl }),
      signal: ctl.signal,
      partBytes: 4,
      concurrency: 1,
      sleep: instantSleep,
    }),
    (err) => err.name === 'AbortError',
  );
  const events = [];
  await downloadImageModel(model(), {
    storage,
    fetchImpl: server(),
    partBytes: 4,
    concurrency: 1,
    sleep: instantSleep,
    onProgress: (p) => events.push({ bytes: p.bytes, totalBytes: p.totalBytes, key: p.key }),
  });
  assert.deepEqual(events[0], { bytes: 4, totalBytes: 12, key: null }, '开始时已有 4 字节');
  assert.ok(events.some((e) => e.key === 'unetWeights' && e.bytes === 8), '每段完成时都会报告');
  const last = events[events.length - 1];
  assert.equal(last.bytes, 12);
  for (let i = 1; i < events.length; i += 1) assert.ok(events[i].bytes >= events[i - 1].bytes, '字节数不应减少');
});

test('取消：取消后立即停止（包括重试等待中），并抛出 AbortError', async () => {
  const storage = memoryCaches();
  const ctl = new AbortController();
  const f = server({ faults: { 0: 'network', 1: 'network', 2: 'network' } });
  const sleep = async (ms, signal) => {
    ctl.abort();
    if (signal?.aborted) throw abortError();
  };
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: f, signal: ctl.signal, sleep }),
    (err) => err.name === 'AbortError',
  );
  assert.equal(f.calls.length, 1, '等待中被取消，没有继续重试');
});

test('删除：同时删除未完成下载的分段', async () => {
  const storage = memoryCaches();
  const ctl = new AbortController();
  await assert.rejects(
    downloadImageModel(model(), {
      storage,
      fetchImpl: server({ faults: { 1: 'abort' }, ctl }),
      signal: ctl.signal,
      partBytes: 4,
      concurrency: 1,
      sleep: instantSleep,
    }),
    (err) => err.name === 'AbortError',
  );
  await deleteImageModelFiles(model(), { storage });
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.equal(st.partsCached, 0);
  assert.equal(st.doneBytes, 0);
});

test('不可用的模型（如网页版 fp32）拒绝下载', async () => {
  const storage = memoryCaches();
  await assert.rejects(downloadImageModel(model({ status: 'unavailable' }), { storage, fetchImpl: server() }), /暂不可用/);
});
