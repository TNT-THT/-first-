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
      const fault = options.faultFor ? options.faultFor(name, range) : options.faults?.[seq];
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
      concurrency: 1, // 单个文件的退避序列（并行时各文件各自退避）
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
  // 只让 unetWeights 的第 2 段（bytes=4-7）返回 404
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: server({ faultFor: (name, range) => (range === 'bytes=4-7' ? 404 : undefined), delayMs: 5 }), partBytes: 4, concurrency: 4, sleep: instantSleep }),
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
  // 两个文件各发出一次请求（并行），取消后没有再重试
  assert.equal(f.calls.length, 2, '取消后没有继续重试');
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

test('文件并行：多个文件与分段共用同一并发上限（同时最多 concurrency 个请求，且不止 1 个）', async () => {
  const storage = memoryCaches();
  const data = {
    'f1.bin': Uint8Array.from([1, 2]),
    'f2.bin': Uint8Array.from([3, 4]),
    'f3.bin': Uint8Array.from([5, 6]),
    'f4.bin': Uint8Array.from([7, 8]),
    'f5.bin': Uint8Array.from([9, 10]),
    'big.bin': Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
  };
  let inflight = 0;
  let maxInflight = 0;
  const fetchImpl = async (url, opts = {}) => {
    const name = Object.keys(data).find((p) => url.endsWith(`/${p}`));
    if (!name) return new Response('nope', { status: 404 });
    inflight += 1;
    maxInflight = Math.max(maxInflight, inflight);
    try {
      await new Promise((r) => setTimeout(r, 5));
      const range = opts.headers?.Range;
      if (range) {
        const m = /bytes=(\d+)-(\d+)/.exec(range);
        return new Response(data[name].slice(Number(m[1]), Number(m[2]) + 1), { status: 206 });
      }
      return new Response(data[name].slice(), { status: 200 });
    } finally {
      inflight -= 1;
    }
  };
  const files = Object.fromEntries(
    Object.keys(data).map((p) => [p.replace('.bin', ''), { path: p, size: data[p].length }]),
  );
  const m = { id: 'parallel-files', label: '并行文件', family: 'sd15', repo: 'org/repo', revision: 'main', status: 'available', files };
  const st = await downloadImageModel(m, { storage, fetchImpl, partBytes: 4, concurrency: 4, sleep: instantSleep });
  assert.equal(st.complete, true);
  assert.ok(maxInflight >= 2, `最大并行请求数为 ${maxInflight}，文件没有并行`);
  assert.ok(maxInflight <= 4, `最大并行请求数为 ${maxInflight}，超过上限 4`);
  assert.deepEqual(await cachedBytes(storage, sourceUrls(m).big), data['big.bin']);
});

/**
 * 模拟网络中断：第一次请求 unet/weights.pb 时，先送出 dropAt 个字节，然后断开。
 * mode: 'network'（网络错误，可重试）| 'abort'（模拟用户取消）| 'norange'（忽略 Range，整文件 200）
 */
function flakyServer({ dropAt, mode = 'network' }) {
  const calls = [];
  let dropped = false;
  const fn = async (url, opts = {}) => {
    const name = Object.keys(DATA).find((p) => url.endsWith(`/${p}`));
    if (!name) return new Response('nope', { status: 404 });
    const range = opts.headers?.Range;
    calls.push({ name, range: range || null });
    const data = DATA[name];
    let start = 0;
    let end = data.length - 1;
    const useRange = Boolean(range) && mode !== 'norange';
    if (useRange) {
      const m = /bytes=(\d+)-(\d+)/.exec(range);
      start = Number(m[1]);
      end = Number(m[2]);
    }
    const slice = data.slice(start, end + 1);
    // norange 模式下中断落在整文件请求上（Range 请求只会收到 200 并被放弃）
    const drop = !dropped && name === 'unet/weights.pb' && dropAt !== undefined && (mode !== 'norange' || !range);
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
          if (mode === 'abort') c.error(abortError());
          else c.error(new TypeError('network error'));
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(stream, { status: useRange ? 206 : 200 });
  };
  fn.calls = calls;
  return fn;
}

test('网络中断：已收到的字节保存，下次从断点用 Range 续传（不重下已收到的部分）', async () => {
  const storage = memoryCaches();
  const f = flakyServer({ dropAt: 2 });
  const st = await downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 4, concurrency: 1, sleep: instantSleep });
  assert.equal(st.complete, true);
  const ranges = f.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.deepEqual(ranges, ['bytes=0-3', 'bytes=2-3', 'bytes=4-7', 'bytes=8-9'], '中断处之后从第 2 个字节继续');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

test('网络中断后等待重试期间，状态已包含中断前收到的字节', async () => {
  const storage = memoryCaches();
  const f = flakyServer({ dropAt: 3 });
  let midPartial = null;
  await downloadImageModel(model(), {
    storage,
    fetchImpl: f,
    partBytes: 4,
    concurrency: 1,
    sleep: async () => {
      midPartial = await imageCacheStatus(model(), { storage, partBytes: 4 });
    },
  });
  assert.equal(midPartial.partialBytes, 3, '断开时已保存 3 个字节');
});

test('用户取消时，已收到的字节同样保存', async () => {
  const storage = memoryCaches();
  const f = flakyServer({ dropAt: 2, mode: 'abort' });
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: f, partBytes: 4, concurrency: 1, sleep: instantSleep }),
    (err) => err.name === 'AbortError',
  );
  const st = await imageCacheStatus(model(), { storage, partBytes: 4 });
  assert.equal(st.partialBytes, 2, '取消时已收到的 2 个字节被保存');
  const retry = flakyServer({});
  await downloadImageModel(model(), { storage, fetchImpl: retry, partBytes: 4, concurrency: 1, sleep: instantSleep });
  const ranges = retry.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.equal(ranges[0], 'bytes=2-3', '续传从第 2 个字节开始');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

test('服务器忽略 Range 时中断：已收到的字节保存，重试后结果正确', async () => {
  const storage = memoryCaches();
  const f = flakyServer({ dropAt: 6, mode: 'norange' });
  let midPartial = null;
  const st = await downloadImageModel(model(), {
    storage,
    fetchImpl: f,
    partBytes: 4,
    concurrency: 1,
    sleep: async () => {
      midPartial = await imageCacheStatus(model(), { storage, partBytes: 4 });
    },
  });
  assert.equal(st.complete, true);
  assert.equal(midPartial.partialBytes, 6, '第一段完整（4 字节）+ 第二段已收到 2 字节');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(model()).unetWeights), DATA['unet/weights.pb']);
});

const MB = 1024 * 1024;

function patternData(size) {
  const data = new Uint8Array(size);
  for (let i = 0; i < size; i += 1) data[i] = i % 251;
  return data;
}

test('写入频率：每写满 1 MB 即保存；下载未结束、未出错时，已写满的块也已经保存', async () => {
  const storage = memoryCaches();
  const SIZE = 3 * MB; // 文件 3 MB，网络只送到 2.5 MB 就停住
  const SENT = 2.5 * MB;
  const data = patternData(SIZE);
  const big = {
    id: 'freq-test',
    label: '写入频率测试',
    family: 'sd15',
    repo: 'org/repo',
    revision: 'main',
    status: 'available',
    files: { big: { path: 'big.bin', size: SIZE } },
  };
  const ctl = new AbortController();
  const fetchImpl = async () => {
    let pos = 0;
    const stream = new ReadableStream(
      {
        start(c) {
          ctl.signal.addEventListener('abort', () => c.error(abortError()));
        },
        pull(c) {
          if (pos < SENT) {
            const n = Math.min(256 * 1024, SENT - pos);
            c.enqueue(data.slice(pos, pos + n));
            pos += n;
            return undefined;
          }
          return new Promise(() => {}); // 之后一直没有数据（网络很慢）
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(stream, { status: 200 });
  };
  const p = downloadImageModel(big, { storage, fetchImpl, signal: ctl.signal, sleep: instantSleep });
  let st = null;
  for (let k = 0; k < 400; k += 1) {
    st = await imageCacheStatus(big, { storage });
    if (st.partialBytes >= 2 * MB) break;
    await new Promise((r) => setTimeout(r, 5));
  }
  assert.equal(st.partialBytes, 2 * MB, '已写满的两块（2 MB）已保存，未写满的 0.5 MB 尚未保存');
  ctl.abort();
  await assert.rejects(p, (err) => err.name === 'AbortError');
  const after = await imageCacheStatus(big, { storage });
  assert.equal(after.partialBytes, SENT, '取消时未写满的块也保存（共 2.5 MB）');
});

test('块级续传：中断时已写满的块与未写满的部分都保存，续传从断点开始', async () => {
  const storage = memoryCaches();
  const SIZE = 3 * MB;
  const DROP = 1.5 * MB;
  const data = patternData(SIZE);
  const big = {
    id: 'block-resume',
    label: '块级续传测试',
    family: 'sd15',
    repo: 'org/repo',
    revision: 'main',
    status: 'available',
    files: { big: { path: 'big.bin', size: SIZE } },
  };
  const calls = [];
  let dropped = false;
  const fetchImpl = async (url, opts = {}) => {
    const range = opts.headers?.Range;
    calls.push(range || null);
    const start = range ? Number(/bytes=(\d+)-/.exec(range)[1]) : 0;
    const drop = !dropped;
    if (drop) dropped = true;
    const end = drop ? DROP : SIZE;
    let sent = false;
    const stream = new ReadableStream(
      {
        pull(c) {
          if (!sent) {
            sent = true;
            c.enqueue(data.slice(start, end));
            if (!drop) c.close();
            return;
          }
          if (drop) c.error(new TypeError('network error'));
          else c.close();
        },
      },
      { highWaterMark: 0 },
    );
    return new Response(stream, { status: range ? 206 : 200 });
  };
  const st = await downloadImageModel(big, { storage, fetchImpl, sleep: instantSleep });
  assert.equal(st.complete, true);
  assert.deepEqual(calls, [null, `bytes=${DROP}-${SIZE - 1}`], '第一次整文件请求；续传从 1.5 MB 处开始');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(big).big), data);
});

/** 用于下面几个测试：直接给出每次请求的响应（不经过 server() 的故障表）。 */
function scripted(handler) {
  const calls = [];
  const fn = async (url, opts = {}) => {
    const name = Object.keys(DATA).find((p) => url.endsWith(`/${p}`));
    const range = opts.headers?.Range || null;
    calls.push({ name, range });
    if (!name) return new Response('nope', { status: 404 });
    return handler({ name, range, data: DATA[name], signal: opts.signal, calls });
  };
  fn.calls = calls;
  return fn;
}

/** 先送 sent 字节，之后一直不再有数据（也不报错），直到请求被中止。 */
function stallingBody(data, sent, signal) {
  let sentOnce = false;
  return new ReadableStream(
    {
      pull(c) {
        if (!sentOnce) {
          sentOnce = true;
          c.enqueue(data.slice(0, sent));
          return undefined;
        }
        return new Promise((_, reject) => {
          signal?.addEventListener('abort', () => {
            c.error(abortError());
            reject(abortError());
          });
        });
      },
    },
    { highWaterMark: 0 },
  );
}

test('连接卡住（既无数据也不报错）：超过 stallMs 后中止，已收到的字节保存，自动重试续传', async () => {
  const storage = memoryCaches();
  const m = model();
  const stallUrl = sourceUrls(m).unetWeights;
  let stalled = false;
  const fetchImpl = scripted(({ name, range, data, signal }) => {
    if (name === 'unet/weights.pb' && !range && !stalled) {
      stalled = true;
      return new Response(stallingBody(data, 4, signal), { status: 200 });
    }
    if (range) {
      const m2 = /bytes=(\d+)-(\d+)/.exec(range);
      return new Response(data.slice(Number(m2[1]), Number(m2[2]) + 1), { status: 206 });
    }
    return new Response(data.slice(), { status: 200 });
  });
  const retries = [];
  const st = await downloadImageModel(m, {
    storage,
    fetchImpl,
    stallMs: 50,
    sleep: instantSleep,
    onProgress: ({ retry }) => retry && retries.push(retry),
  });
  assert.equal(st.complete, true);
  assert.ok(retries.some((r) => r.message.includes('没有数据')), '卡住时给出重试提示');
  const weightCalls = fetchImpl.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.deepEqual(weightCalls, [null, 'bytes=4-9'], '卡住前收到的 4 字节不重下');
  assert.deepEqual(await cachedBytes(storage, stallUrl), DATA['unet/weights.pb']);
});

test('HF 链接出错（续传时返回 403）：已保存的字节保留，报错说明会保留；修复后只补下缺失的部分', async () => {
  const storage = memoryCaches();
  const m = model();
  let mode = 'drop';
  const fetchImpl = scripted(({ name, range, data }) => {
    if (name === 'unet/weights.pb' && !range && mode === 'drop') {
      let sent = false;
      const stream = new ReadableStream(
        {
          pull(c) {
            if (!sent) {
              sent = true;
              c.enqueue(data.slice(0, 4));
              return;
            }
            c.error(new TypeError('network error'));
          },
        },
        { highWaterMark: 0 },
      );
      return new Response(stream, { status: 200 });
    }
    if (name === 'unet/weights.pb' && range && mode === 'forbidden') return new Response('denied', { status: 403 });
    if (range) {
      const m2 = /bytes=(\d+)-(\d+)/.exec(range);
      return new Response(data.slice(Number(m2[1]), Number(m2[2]) + 1), { status: 206 });
    }
    return new Response(data.slice(), { status: 200 });
  });
  // 第一次断开后，续传时 HF 返回 403（链接出错）
  await assert.rejects(
    downloadImageModel(m, {
      storage,
      fetchImpl,
      sleep: instantSleep,
      onProgress: ({ retry }) => {
        if (retry) mode = 'forbidden';
      },
    }),
    /已保存的部分会保留/,
  );
  const kept = await imageCacheStatus(m, { storage });
  assert.equal(kept.partialBytes, 4, '403 之后，已保存的 4 字节仍然保留');

  // 链接修复后：只补下缺失的部分
  mode = 'ok';
  fetchImpl.calls.length = 0;
  const st = await downloadImageModel(m, { storage, fetchImpl, sleep: instantSleep });
  assert.equal(st.complete, true);
  const weightCalls = fetchImpl.calls.filter((c) => c.name === 'unet/weights.pb').map((c) => c.range);
  assert.deepEqual(weightCalls, ['bytes=4-9'], '只请求缺少的字节');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(m).unetWeights), DATA['unet/weights.pb']);
});

test('服务器忽略 Range、且已有保存的部分：提示“正在跳过已保存的部分”，进度不重复计数', async () => {
  const storage = memoryCaches();
  const m = model();
  let first = true;
  const fetchImpl = scripted(({ name, data }) => {
    if (name === 'unet/weights.pb' && first) {
      first = false;
      let sent = false;
      const stream = new ReadableStream(
        {
          pull(c) {
            if (!sent) {
              sent = true;
              c.enqueue(data.slice(0, 4));
              return;
            }
            c.error(new TypeError('network error'));
          },
        },
        { highWaterMark: 0 },
      );
      return new Response(stream, { status: 200 });
    }
    // 之后所有请求都返回整文件（忽略 Range）
    return new Response(data.slice(), { status: 200 });
  });
  const notes = [];
  let lastBytes = 0;
  let totalBytes = 0;
  await downloadImageModel(m, {
    storage,
    fetchImpl,
    sleep: instantSleep,
    onProgress: ({ note, bytes, totalBytes: tb, retry }) => {
      if (note) notes.push(note);
      if (!retry) lastBytes = bytes;
      totalBytes = tb;
    },
  });
  assert.ok(notes.some((n) => n.includes('跳过已保存的部分')), '跳过时给出提示');
  assert.equal(lastBytes, totalBytes, '字节数不重复计数');
  assert.deepEqual(await cachedBytes(storage, sourceUrls(m).unetWeights), DATA['unet/weights.pb']);
});

test('拼接阶段有进度提示（进度条停在已保存的字节数，并说明正在整理文件）', async () => {
  const storage = memoryCaches();
  const m = model();
  const notes = [];
  await downloadImageModel(m, {
    storage,
    fetchImpl: server(),
    sleep: instantSleep,
    onProgress: ({ note }) => note && notes.push(note),
  });
  assert.ok(notes.some((n) => n.includes('正在整理文件（unet/weights.pb）')), '拼接开始时提示');
  assert.ok(notes.some((n) => n.includes('100%')), '拼接进度达到 100%');
});
