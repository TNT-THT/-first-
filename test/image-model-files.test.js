// 生图模型的本机下载 / 删除 / 导出 / 导入测试（内存版 Cache Storage 与 fetch）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sourceUrls } from '../web/src/image/catalog.js';
import {
  IMAGE_CACHE,
  buildImageModelFile,
  deleteImageModelFiles,
  downloadImageModel,
  imageCacheStatus,
  importImageModelFile,
  readImageModelHeader,
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

const BYTES = {
  'te/model.onnx': Uint8Array.from([1, 2, 3, 4, 5]),
  'unet/model.onnx': Uint8Array.from([9, 8, 7]),
  'v.json': Uint8Array.from([123, 125]),
};

function model(extra = {}) {
  return {
    id: 'test-sd',
    label: '测试模型',
    family: 'sd15',
    repo: 'org/repo',
    revision: 'main',
    status: 'available',
    files: {
      textEncoder: { path: 'te/model.onnx', size: 5 },
      unet: { path: 'unet/model.onnx', size: 3 },
      vocab: { path: 'v.json', size: 2 },
    },
    ...extra,
  };
}

function fakeFetch({ missing = new Set() } = {}) {
  const calls = [];
  const fn = async (url) => {
    calls.push(url);
    const path = Object.keys(BYTES).find((p) => url.endsWith(`/${p}`));
    if (!path || missing.has(path)) return new Response('nope', { status: 404 });
    return new Response(BYTES[path].slice(), { status: 200 });
  };
  fn.calls = calls;
  return fn;
}

test('状态：未下载时列出全部缺失文件', async () => {
  const storage = memoryCaches();
  const st = await imageCacheStatus(model(), { storage });
  assert.equal(st.complete, false);
  assert.equal(st.cached, 0);
  assert.equal(st.total, 3);
  assert.equal(st.bytes, 10);
});

test('下载：全部写入缓存，第二次下载不会重复请求（续传）', async () => {
  const storage = memoryCaches();
  const f = fakeFetch();
  const progress = [];
  const st = await downloadImageModel(model(), { storage, fetchImpl: f, onProgress: (p) => progress.push(p.done) });
  assert.equal(st.complete, true);
  assert.equal(f.calls.length, 3);
  // 进度包括字节级事件，done（已完成文件数）应按 0,1,2,3 递增，中间不回退
  assert.deepEqual([...new Set(progress)], [0, 1, 2, 3], '开始时先报告 0，之后每个文件完成时递增');
  const again = fakeFetch();
  await downloadImageModel(model(), { storage, fetchImpl: again });
  assert.equal(again.calls.length, 0);
});

test('下载：某个文件永久失败（404）时不中断，其他文件照常下载；全部完成后在缓存校验时报错，修复后只补下缺失文件', async () => {
  const storage = memoryCaches();
  await assert.rejects(
    downloadImageModel(model(), { storage, fetchImpl: fakeFetch({ missing: new Set(['unet/model.onnx']) }) }),
    /缓存校验未通过，缺少 1 个文件：下载失败（HTTP 404）：unet\/model\.onnx/,
  );
  const st = await imageCacheStatus(model(), { storage });
  assert.equal(st.complete, false);
  // 下载阶段不因 404 中断：unet 之后的 vocab 也已下载
  assert.deepEqual(st.missing, ['unet']);
  const retry = fakeFetch();
  await downloadImageModel(model(), { storage, fetchImpl: retry });
  assert.deepEqual(retry.calls, [sourceUrls(model()).unet], '已下载的 textEncoder 与 vocab 不再请求');
});

test('下载：暂时性错误（网络中断、503）自动重试，不报错', async () => {
  const storage = memoryCaches();
  let n = 0;
  const flaky = async (url, opts) => {
    n += 1;
    if (n === 1) throw new TypeError('Failed to fetch');
    if (n === 3) return new Response('busy', { status: 503 });
    return fakeFetch()(url, opts);
  };
  const st = await downloadImageModel(model(), { storage, fetchImpl: flaky, sleep: async () => {} });
  assert.equal(st.complete, true);
});

test('删除：移除该模型的全部缓存文件，不影响其他模型', async () => {
  const storage = memoryCaches();
  await downloadImageModel(model(), { storage, fetchImpl: fakeFetch() });
  await storage.open(IMAGE_CACHE).then((c) => c.put('https://example.com/other.bin', new Response(new Uint8Array([1]))));
  assert.equal(await deleteImageModelFiles(model(), { storage }), 3);
  assert.equal((await imageCacheStatus(model(), { storage })).cached, 0);
  assert.equal(store(storage).has('https://example.com/other.bin'), true);
});

function store(storage) {
  return storage._store.get(IMAGE_CACHE);
}

test('导出：未完整下载时拒绝导出，并提示先下载', async () => {
  const storage = memoryCaches();
  await assert.rejects(buildImageModelFile(model(), { storage }), /请先点击“下载到本机”/);
});

test('导出与导入：往返后缓存内容完全一致，文件名与头部正确', async () => {
  const src = memoryCaches();
  await downloadImageModel(model(), { storage: src, fetchImpl: fakeFetch() });
  const { blob, filename, entryCount } = await buildImageModelFile(model(), { storage: src, now: new Date('2026-10-23T00:00:00Z') });
  assert.equal(filename, 'test-sd.ojimage');
  assert.equal(entryCount, 3);
  const header = await readImageModelHeader(blob);
  assert.equal(header.header.modelId, 'test-sd');
  assert.equal(header.header.format, 'oj-ai-image-model');

  const dst = memoryCaches();
  const res = await importImageModelFile(blob, { models: [model()], storage: dst });
  assert.equal(res.entryCount, 3);
  assert.equal((await imageCacheStatus(model(), { storage: dst })).complete, true);
  for (const url of Object.values(sourceUrls(model()))) {
    assert.deepEqual(store(dst).get(url).bytes, store(src).get(url).bytes);
  }
});

test('导入：拒绝站点配置中不存在的模型', async () => {
  const src = memoryCaches();
  await downloadImageModel(model(), { storage: src, fetchImpl: fakeFetch() });
  const { blob } = await buildImageModelFile(model(), { storage: src });
  await assert.rejects(importImageModelFile(blob, { models: [], storage: memoryCaches() }), /不在本站配置中/);
  await assert.rejects(
    importImageModelFile(blob, { models: [model({ status: 'planned' })], storage: memoryCaches() }),
    /暂不可用/,
  );
});

test('导入：地址或大小与站点配置不一致时拒绝（防止被替换为其他仓库的文件）', async () => {
  const src = memoryCaches();
  await downloadImageModel(model(), { storage: src, fetchImpl: fakeFetch() });
  const { blob } = await buildImageModelFile(model(), { storage: src });
  const otherRepo = model({ repo: 'evil/repo' });
  await assert.rejects(importImageModelFile(blob, { models: [otherRepo], storage: memoryCaches() }), /地址与站点配置不一致/);
  const otherSize = model({ files: { ...model().files, vocab: { path: 'v.json', size: 99 } } });
  await assert.rejects(importImageModelFile(blob, { models: [otherSize], storage: memoryCaches() }), /大小与站点配置不一致/);
});

test('导入：拒绝聊天模型的 .ojmodel 文件与非本应用文件，并报告损坏的文件', async () => {
  const chat = new Blob([new TextEncoder().encode('OJMD'), new Uint8Array([2, 0, 0, 0]), new TextEncoder().encode('{}{}')]);
  await assert.rejects(importImageModelFile(chat, { models: [model()], storage: memoryCaches() }), /头部|格式/);
  await assert.rejects(importImageModelFile(new Blob(['hello world, not ours']), { models: [model()], storage: memoryCaches() }), /不是本应用导出/);

  const src = memoryCaches();
  await downloadImageModel(model(), { storage: src, fetchImpl: fakeFetch() });
  const { blob } = await buildImageModelFile(model(), { storage: src });
  const truncated = blob.slice(0, blob.size - 4);
  await assert.rejects(importImageModelFile(truncated, { models: [model()], storage: memoryCaches() }), /数据不完整/);
});
