import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildModelFile,
  collectModelEntries,
  importModelFile,
  listCachedModelIds,
  readModelFileHeader,
  validateHeader,
  cleanModelUrl,
  CACHE_SCOPES,
} from '../web/src/model-file.js';

// 内存版 Cache Storage，接口与浏览器 Cache API 的子集一致（open / keys / match / put / delete）
class MemCache {
  constructor() {
    this.map = new Map();
  }
  async match(req) {
    const url = typeof req === 'string' ? req : req.url;
    const v = this.map.get(url);
    return v ? new Response(v.slice(0), { status: 200 }) : undefined;
  }
  async put(req, res) {
    const url = typeof req === 'string' ? req : req.url;
    this.map.set(url, new Uint8Array(await res.arrayBuffer()));
  }
  async keys() {
    return [...this.map.keys()].map((url) => ({ url }));
  }
  async delete(req) {
    return this.map.delete(typeof req === 'string' ? req : req.url);
  }
}
class MemStorage {
  constructor() {
    this.scopes = new Map();
  }
  async open(name) {
    if (!this.scopes.has(name)) this.scopes.set(name, new MemCache());
    return this.scopes.get(name);
  }
}

const RECORD = {
  model_id: 'Demo-1.5B-q4f16_1-MLC',
  model: 'https://huggingface.co/mlc-ai/Demo-1.5B-q4f16_1-MLC',
  model_lib: 'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Demo.wasm',
};
const BASE = cleanModelUrl(RECORD.model); // .../Demo-1.5B-q4f16_1-MLC/resolve/main/

function bytes(n, seed) {
  const a = new Uint8Array(n);
  for (let i = 0; i < n; i++) a[i] = (i * 31 + seed) & 0xff;
  return a;
}

/** 填充一个完整模型到存储中，返回期望的 url -> 字节 映射。 */
async function populate(storage, record = RECORD, base = BASE) {
  const expected = new Map();
  const put = async (scope, url, data) => {
    await (await storage.open(scope)).put(url, new Response(data));
    expected.set(url, data);
  };
  const index = { records: [{ dataPath: 'params_shard_0.bin' }, { dataPath: 'params_shard_1.bin' }] };
  await put('webllm/model', new URL('tensor-cache.json', base).href, new TextEncoder().encode(JSON.stringify(index)));
  await put('webllm/model', new URL('params_shard_0.bin', base).href, bytes(70000, 1));
  await put('webllm/model', new URL('params_shard_1.bin', base).href, bytes(1234, 2));
  await put('webllm/model', new URL('tokenizer.json', base).href, bytes(999, 3));
  await put('webllm/config', new URL('mlc-chat-config.json', base).href, new TextEncoder().encode('{"ok":true}'));
  await put('webllm/wasm', record.model_lib, bytes(5000, 4));
  return expected;
}

async function exportToFile(storage, record = RECORD) {
  const { blob, filename } = await buildModelFile(record, { storage });
  const file = new File([blob], filename);
  return file;
}

test('cleanModelUrl 与 WebLLM 的规则一致', () => {
  assert.equal(cleanModelUrl('https://huggingface.co/a/b'), 'https://huggingface.co/a/b/resolve/main/');
  assert.equal(cleanModelUrl('https://huggingface.co/a/b/resolve/main/'), 'https://huggingface.co/a/b/resolve/main/');
});

test('导出：缺少关键文件时给出可读错误', async () => {
  const storage = new MemStorage();
  await populate(storage);
  await (await storage.open('webllm/config')).delete(new URL('mlc-chat-config.json', BASE).href);
  await assert.rejects(buildModelFile(RECORD, { storage }), /缺少：mlc-chat-config\.json/);
});

test('导出：权重分片缺失时拒绝导出', async () => {
  const storage = new MemStorage();
  await populate(storage);
  await (await storage.open('webllm/model')).delete(new URL('params_shard_1.bin', BASE).href);
  await assert.rejects(buildModelFile(RECORD, { storage }), /权重文件不完整/);
});

test('往返：导出后导入到全新存储，所有缓存键与字节完全一致（离线可用的前提）', async () => {
  const source = new MemStorage();
  const expected = await populate(source);
  const file = await exportToFile(source);

  const target = new MemStorage();
  const result = await importModelFile(file, RECORD, { storage: target });
  assert.equal(result.modelId, RECORD.model_id);
  assert.equal(result.entryCount, expected.size);

  for (const [url, data] of expected) {
    const scope = url === RECORD.model_lib ? 'webllm/wasm' : url.endsWith('mlc-chat-config.json') ? 'webllm/config' : 'webllm/model';
    const res = await (await target.open(scope)).match(url);
    assert.ok(res, `缺少 ${url}`);
    assert.deepEqual(new Uint8Array(await res.arrayBuffer()), data, url);
  }

  // 导入后的存储应能被识别为完整模型，并出现在"已缓存"列表中
  const entries = await collectModelEntries(RECORD, target);
  assert.equal(entries.length, expected.size);
  assert.ok((await listCachedModelIds([RECORD], target)).has(RECORD.model_id));
});

test('导出不删除源缓存（导出后原设备仍可使用）', async () => {
  const storage = new MemStorage();
  const expected = await populate(storage);
  await exportToFile(storage);
  for (const url of expected.keys()) {
    const scope = url === RECORD.model_lib ? 'webllm/wasm' : url.endsWith('mlc-chat-config.json') ? 'webllm/config' : 'webllm/model';
    assert.ok(await (await storage.open(scope)).match(url), url);
  }
});

test('头部：可读取且 data 偏移正确', async () => {
  const storage = new MemStorage();
  await populate(storage);
  const file = await exportToFile(storage);
  const { header, dataStart } = await readModelFileHeader(file);
  assert.equal(header.format, 'oj-ai-model');
  assert.equal(header.modelId, RECORD.model_id);
  assert.equal(header.entries.length, 6);
  const last = header.entries.at(-1);
  assert.equal(dataStart + last.offset + last.size, file.size);
});

test('拒绝：魔数错误、截断文件、非本应用文件', async () => {
  const storage = new MemStorage();
  await populate(storage);
  const file = await exportToFile(storage);

  const bad = new File([new Uint8Array([1, 2, 3, 4, 0, 0, 0, 0])], 'x.ojmodel');
  await assert.rejects(readModelFileHeader(bad), /不是本应用导出的模型文件/);

  const truncated = file.slice(0, Math.floor(file.size / 2));
  await assert.rejects(importModelFile(truncated, RECORD, { storage: new MemStorage() }), /数据不完整/);

  const tiny = new File([new Uint8Array(3)], 'tiny.ojmodel');
  await assert.rejects(readModelFileHeader(tiny), /不完整/);
});

test('拒绝：模型 id 不一致', async () => {
  const storage = new MemStorage();
  await populate(storage);
  const file = await exportToFile(storage);
  const other = { ...RECORD, model_id: 'Other-1B-q4f16_1-MLC' };
  await assert.rejects(importModelFile(file, other, { storage: new MemStorage() }), /与所选模型/);
});

test('拒绝：下载地址不一致（例如镜像设置不同）', async () => {
  const storage = new MemStorage();
  await populate(storage);
  const file = await exportToFile(storage);
  const mirrored = { ...RECORD, model: 'https://huggingface.co/someone/Demo-1.5B-q4f16_1-MLC' };
  await assert.rejects(importModelFile(file, mirrored, { storage: new MemStorage() }), /下载地址与当前配置不一致/);
});

test('校验：未知缓存类别与非法偏移会被拒绝', () => {
  const base = {
    format: 'oj-ai-model',
    version: 1,
    modelId: 'm',
    entries: [{ scope: 'webllm/model', url: 'https://huggingface.co/a/resolve/main/x', size: 1, offset: 0 }],
  };
  assert.doesNotThrow(() => validateHeader(base));
  assert.throws(() => validateHeader({ ...base, entries: [{ ...base.entries[0], scope: 'evil' }] }), /未知的缓存类别/);
  assert.throws(() => validateHeader({ ...base, entries: [{ ...base.entries[0], offset: -1 }] }), /偏移或大小无效/);
  assert.throws(() => validateHeader({ ...base, version: 99 }), /高于当前支持的版本/);
  assert.throws(() => validateHeader({ ...base, format: 'other' }), /格式不受支持/);
  assert.deepEqual(CACHE_SCOPES, ['webllm/model', 'webllm/config', 'webllm/wasm']);
});

test('已缓存检测：只有关键文件齐全的模型才被标记', async () => {
  const storage = new MemStorage();
  await populate(storage);
  const other = { ...RECORD, model_id: 'Empty-0.5B', model: 'https://huggingface.co/mlc-ai/Empty-0.5B' };
  const cached = await listCachedModelIds([RECORD, other], storage);
  assert.ok(cached.has(RECORD.model_id));
  assert.ok(!cached.has('Empty-0.5B'));
});
