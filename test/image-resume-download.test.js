// 生图模型下载：分段并行、断点续传、中断后保留进度、服务器不支持 Range 时回退（内存版 Cache Storage 与 fetch）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sourceUrls } from '../web/src/image/catalog.js';
import {
  IMAGE_CACHE,
  IMAGE_PART_CACHE,
  deleteImageModelFiles,
  downloadImageModel,
  imageCacheStatus,
} from '../web/src/image/model-files.js';

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

/**
 * 模拟服务器。options:
 * - failAt: 第几次请求（从 0 计）抛出网络错误（模拟中断）
 * - noRange: 忽略 Range，直接返回 200 整文件
 * - delayMs: 每次请求的延迟（用于统计并行数）
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
      if (options.failAt === seq) throw new TypeError('network error');
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
  const st = await downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 4 });
  assert.equal(st.complete, true);
  assert.deepEqual(f.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range), [
    'bytes=0-3',
    'bytes=4-7',
    'bytes=8-9',
  ]);
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
  assert.equal((await storage.open(IMAGE_PART_CACHE)).keys ? (await (await storage.open(IMAGE_PART_CACHE)).keys()).length : 0, 0);
});

test('中断续传：中断后已完成的分段保留，下次只请求缺少的段', async () => {
  const storage = memoryCaches();
  // 第 0 次请求成功（第 1 段），第 1 次请求（第 2 段）中断
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: server({ failAt: 1 }), partBytes: 4, concurrency: 1 }),
    /network error/,
  );
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.equal(st.complete, false);
  assert.equal(st.partsCached, 1);
  assert.equal(st.partialBytes, 4);
  assert.equal(st.doneBytes, 4);

  const retry = server();
  const done = await downloadImageModel(model(), { storage, fetchImpl: retry, partBytes: 4, concurrency: 1 });
  assert.equal(done.complete, true);
  const weightsRanges = retry.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.deepEqual(weightsRanges, ['bytes=4-7', 'bytes=8-9'], '已完成的第 1 段不应重新下载');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

test('并行：同一文件的多个段同时下载', async () => {
  const storage = memoryCaches();
  const f = server({ delayMs: 5 });
  await downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 2, concurrency: 4 });
  assert.ok(f.maxInflight >= 2, `最大并行数为 ${f.maxInflight}`);
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

test('服务器忽略 Range（返回 200）时，改为整文件下载，结果仍然正确', async () => {
  const storage = memoryCaches();
  const f = server({ noRange: true });
  const st = await downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 4 });
  assert.equal(st.complete, true);
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
  const ranges = f.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.equal(ranges.at(-1), null, '最后一次请求是整文件（没有 Range 头）');
});

test('进度：按字节报告，开始时先报告已保存的字节，结束时等于总字节数', async () => {
  const storage = memoryCaches();
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: server({ failAt: 1 }), partBytes: 4, concurrency: 1 }),
    /network error/,
  );
  const events = [];
  await downloadImageModel(model(), {
    storage,
    fetchImpl: server(),
    partBytes: 4,
    concurrency: 1,
    onProgress: (p) => events.push({ bytes: p.bytes, totalBytes: p.totalBytes, key: p.key }),
  });
  assert.deepEqual(events[0], { bytes: 4, totalBytes: 12, key: null }, '开始时已有 4 字节');
  assert.ok(events.some((e) => e.key === 'unetWeights' && e.bytes === 8), '每段完成时都会报告');
  const last = events[events.length - 1];
  assert.equal(last.bytes, 12);
  for (let i = 1; i < events.length; i += 1) assert.ok(events[i].bytes >= events[i - 1].bytes, '字节数不应减少');
});

test('中止：中止后已完成的分段保留，并抛出 AbortError', async () => {
  const storage = memoryCaches();
  const ctl = new AbortController();
  await assert.rejects(
    downloadImageModel(model(), {
      storage,
      fetchImpl: server(),
      partBytes: 4,
      concurrency: 1,
      signal: ctl.signal,
      onProgress: (p) => {
        if (p.key === null) return;
        if (p.bytes >= 4) ctl.abort();
      },
    }),
    (err) => err.name === 'AbortError',
  );
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.ok(st.partsCached >= 1);
  assert.equal(st.complete, false);
});

test('删除：同时删除未完成下载的分段', async () => {
  const storage = memoryCaches();
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: server({ failAt: 1 }), partBytes: 4, concurrency: 1 }),
    /network error/,
  );
  await deleteImageModelFiles(model(), { storage });
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.equal(st.partsCached, 0);
  assert.equal(st.doneBytes, 0);
});

test('不可用的模型（如网页版 fp32）拒绝下载', async () => {
  const storage = memoryCaches();
  await assert.rejects(
    downloadImageModel(model({ status: 'unavailable' }), { storage, fetchImpl: server() }),
    /暂不可用/,
  );
});

test('并行时某段失败：其他进行中的段仍会完成并保存，续传只重下失败的段', async () => {
  const storage = memoryCaches();
  // 三段同时发出（并行 4）；第 2 次请求（第 2 段）失败，第 1、3 段正常完成
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: server({ failAt: 1, delayMs: 5 }), partBytes: 4, concurrency: 4 }),
    /network error/,
  );
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.equal(st.partsCached, 2, '进行中的两段应被保存');
  const retry = server();
  await downloadImageModel(model(), { storage, fetchImpl: retry, partBytes: 4, concurrency: 4 });
  const ranges = retry.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.deepEqual(ranges, ['bytes=4-7'], '只重下失败的第 2 段');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});
