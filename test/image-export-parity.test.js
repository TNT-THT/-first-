// 生图模型与文本模型的功能对齐：分段导出（可流式写入）、参数随文件保存/导入、完整模型列表（下拉框 ✓）、
// 记住上次选择、参数清洗。内存版 Cache Storage 与 fetch，与 image-model-files.test.js 相同。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sourceUrls } from '../web/src/image/catalog.js';
import {
  IMAGE_CACHE,
  deleteImageModelFiles,
  downloadImageModel,
  importImageModelFile,
  listCompleteImageModelIds,
  prepareImageModelExport,
  readImageModelHeader,
} from '../web/src/image/model-files.js';
import { sanitizeImageParams, setImageParams, getImageSettings } from '../web/src/image/params.js';
import {
  IMAGE_MODEL_KEY,
  getSavedImageModel,
  resolveStartupImageModel,
  saveImageModel,
} from '../web/src/default-model.js';
import { imagePickerOptions, IMAGE_FILE_PICKER_TYPES } from '../web/src/save-location.js';

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
    id: 'parity-sd',
    label: '对齐测试模型',
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

function fakeFetch() {
  return async (url) => {
    const path = Object.keys(BYTES).find((p) => url.endsWith(`/${p}`));
    if (!path) return new Response('nope', { status: 404 });
    return new Response(BYTES[path].slice(), { status: 200 });
  };
}

async function fullyDownloaded() {
  const storage = memoryCaches();
  await downloadImageModel(model(), { storage, fetchImpl: fakeFetch() });
  return storage;
}

async function bytesOf(parts) {
  const out = [];
  for (const p of parts) {
    const u8 = p instanceof Blob ? new Uint8Array(await p.arrayBuffer()) : p;
    out.push(...u8);
  }
  return Uint8Array.from(out);
}

test('分段导出：返回可逐段写入的各部分，总大小等于各部分之和，且能被导入回读', async () => {
  const storage = await fullyDownloaded();
  const progress = [];
  const res = await prepareImageModelExport(model(), { storage, onProgress: (p) => progress.push(p.done) });
  assert.equal(res.filename, 'parity-sd.ojimage');
  assert.equal(res.entryCount, 3);
  assert.deepEqual(progress, [1, 2, 3], '每个文件打包完成时报告进度');
  // 各段拼接后的字节数与记录的大小一致（流式写入不会超出或少写）
  const bytes = await bytesOf(res.parts);
  assert.equal(bytes.byteLength, res.size);
  // 导出的文件可以被读取头部，并且导入时校验通过
  const file = new Blob(res.parts);
  const { header } = await readImageModelHeader(file);
  assert.equal(header.modelId, 'parity-sd');
  assert.equal(header.entries.length, 3);
});

test('分段导出：未完整下载时拒绝导出，并提示先下载', async () => {
  await assert.rejects(prepareImageModelExport(model(), { storage: memoryCaches() }), /请先点击“下载到本机”/);
});

test('参数：导出时随文件保存；导入时返回参数，没有参数的文件返回 null', async () => {
  const storage = await fullyDownloaded();
  const params = sanitizeImageParams({ steps: 12, width: 512, bogus: 1 });
  const withParams = await prepareImageModelExport(model(), { storage, params });
  const file = new Blob(withParams.parts);
  const target = memoryCaches();
  const imported = await importImageModelFile(file, { models: [model()], storage: target });
  assert.deepEqual(imported.params, { steps: 12, width: 512 });
  assert.equal(imported.entryCount, 3);

  const plain = await prepareImageModelExport(model(), { storage });
  const imported2 = await importImageModelFile(new Blob(plain.parts), { models: [model()], storage: memoryCaches() });
  assert.equal(imported2.params, null);
});

test('参数：导出时空参数不写入头部字段', async () => {
  const storage = await fullyDownloaded();
  const res = await prepareImageModelExport(model(), { storage, params: {} });
  const { header } = await readImageModelHeader(new Blob(res.parts));
  assert.equal(header.params, undefined);
});

test('清洗：只保留合法键，并限制在范围内；全部无效时返回 null', () => {
  assert.deepEqual(sanitizeImageParams({ steps: 500, guidance: 'x', seed: 42, extra: 1 }), { steps: 50, seed: 42 });
  assert.equal(sanitizeImageParams({ foo: 1 }), null);
  assert.equal(sanitizeImageParams(null), null);
});

test('设置参数：setImageParams 仍按原规则合并（与导入时写入的方式一致）', () => {
  const storage = new Map();
  const fakeStorage = { getItem: (k) => storage.get(k) ?? null, setItem: (k, v) => storage.set(k, v), removeItem: (k) => storage.delete(k) };
  setImageParams('m1', { steps: 8 }, { storage: fakeStorage });
  setImageParams('m1', { width: 384 }, { storage: fakeStorage });
  assert.deepEqual(getImageSettings('m1', { storage: fakeStorage }).params, { steps: 8, width: 384 });
});

test('下拉框 ✓：只标记全部文件都在本机缓存中的模型', async () => {
  const storage = await fullyDownloaded();
  const other = model({ id: 'other-sd', repo: 'org/other-repo' });
  const done = await listCompleteImageModelIds([model(), other], { storage });
  assert.deepEqual([...done], ['parity-sd']);
});

test('删除后不再显示 ✓', async () => {
  const storage = await fullyDownloaded();
  await deleteImageModelFiles(model(), { storage });
  assert.equal((await listCompleteImageModelIds([model()], { storage })).size, 0);
  assert.equal(storage._store.get(IMAGE_CACHE).size, 0);
});

test('记住上次选择：刷新后恢复；上次的模型不可选时回到第一个可选模型', () => {
  const data = new Map();
  const storage = { getItem: (k) => data.get(k) ?? null, setItem: (k, v) => data.set(k, v), removeItem: (k) => data.delete(k) };
  assert.equal(getSavedImageModel(storage), null);
  saveImageModel('sd15-nmkd-fp16', storage);
  assert.equal(getSavedImageModel(storage), 'sd15-nmkd-fp16');
  assert.ok(data.has(IMAGE_MODEL_KEY));

  const models = [
    { id: 'sd15-nmkd-fp32', selectable: false },
    { id: 'sd15-nmkd-fp16', selectable: true },
    { id: 'sdxs', selectable: true },
  ];
  assert.equal(resolveStartupImageModel({ models, saved: 'sd15-nmkd-fp16' }), 'sd15-nmkd-fp16');
  assert.equal(resolveStartupImageModel({ models, saved: 'sd15-nmkd-fp32' }), 'sd15-nmkd-fp16', '不可选的保存值被忽略');
  assert.equal(resolveStartupImageModel({ models, saved: 'gone' }), 'sd15-nmkd-fp16');
  assert.equal(resolveStartupImageModel({ models: [], saved: null }), null);

  saveImageModel(null, storage);
  assert.equal(getSavedImageModel(storage), null);
});

test('保存位置：生图模型使用 .ojimage 文件类型', () => {
  const opts = imagePickerOptions('x.ojimage');
  assert.equal(opts.suggestedName, 'x.ojimage');
  assert.deepEqual(IMAGE_FILE_PICKER_TYPES[0].accept, { 'application/octet-stream': ['.ojimage'] });
  // 下载在对齐后应能处理任意大小的导出文件：导出函数没有单文件大小上限（见 prepareImageModelExport）
  assert.ok(sourceUrls(model()).unet.endsWith('/unet/model.onnx'));
});
