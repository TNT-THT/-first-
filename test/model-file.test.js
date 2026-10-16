import { test } from 'node:test';
import { LOCAL_MODEL_PREFIX, isLocalRecord } from '../web/src/model-source.js';
import assert from 'node:assert/strict';
import {
  buildModelFile,
  collectModelEntries,
  importModelFile,
  listCachedModelIds,
  removeLocalModelFiles,
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

/** 把原始 url 映射到导入后的本地 url；wasm 保持不变。 */
function localUrlOf(url, localBase) {
  if (url === RECORD.model_lib) return url;
  return localBase + url.slice(BASE.length);
}

test('往返：导出后导入为“新建AI”，缓存写入本地虚拟地址，字节完全一致', async () => {
  const source = new MemStorage();
  const expected = await populate(source);
  const file = await exportToFile(source);

  const target = new MemStorage();
  const { record, entryCount } = await importModelFile(file, { storage: target, localId: 'local-test-1' });
  assert.equal(entryCount, expected.size);
  assert.equal(record.model_id, 'local-test-1');
  assert.ok(record.model.startsWith(LOCAL_MODEL_PREFIX));
  assert.equal(record.model_lib, RECORD.model_lib);
  assert.ok(isLocalRecord(record));

  const localBase = cleanModelUrl(record.model);
  for (const [url, data] of expected) {
    const scope = url === RECORD.model_lib ? 'webllm/wasm' : url.endsWith('mlc-chat-config.json') ? 'webllm/config' : 'webllm/model';
    const res = await (await target.open(scope)).match(localUrlOf(url, localBase));
    assert.ok(res, `缺少 ${localUrlOf(url, localBase)}`);
    assert.deepEqual(new Uint8Array(await res.arrayBuffer()), data, url);
  }

  // 导入后应能被识别为完整模型，并出现在"已缓存"列表中
  const entries = await collectModelEntries(record, target);
  assert.equal(entries.length, expected.size);
  assert.ok((await listCachedModelIds([record], target)).has('local-test-1'));
});

test('导入不覆盖官方缓存：目标存储中已有同名官方模型时，两者互不影响', async () => {
  const source = new MemStorage();
  await populate(source);
  const file = await exportToFile(source);

  const target = new MemStorage();
  const official = await populate(target);
  await importModelFile(file, { storage: target, localId: 'local-copy' });

  // 官方地址下的文件仍然存在且字节未变
  for (const url of official.keys()) {
    const scope = url === RECORD.model_lib ? 'webllm/wasm' : url.endsWith('mlc-chat-config.json') ? 'webllm/config' : 'webllm/model';
    const res = await (await target.open(scope)).match(url);
    assert.ok(res, url);
  }
  // 两次导入同一文件得到不同的本地 id，彼此独立
  const a = await importModelFile(file, { storage: target });
  const b = await importModelFile(file, { storage: target });
  assert.notEqual(a.record.model_id, b.record.model_id);
  assert.notEqual(a.record.model, b.record.model);
});

test('截断的文件导入失败时，不写入任何条目', async () => {
  const source = new MemStorage();
  await populate(source);
  const file = await exportToFile(source);
  const target = new MemStorage();
  // 截断文件：数据不完整，写入前即失败
  await assert.rejects(importModelFile(file.slice(0, Math.floor(file.size / 2)), { storage: target }), /数据不完整/);
  assert.equal((await target.open('webllm/model').then((c) => c.keys())).length, 0);
});

test('拒绝：文件头中的下载地址与来源不一致（文件被篡改）', async () => {
  const storage = new MemStorage();
  await populate(storage);
  const file = await exportToFile(storage);
  const { header, dataStart } = await readModelFileHeader(file);
  const tampered = { ...header, entries: header.entries.map((e, i) => (i === 0 ? { ...e, url: 'https://evil.example/x.bin' } : e)) };
  const json = new TextEncoder().encode(JSON.stringify(tampered));
  const prefix = new Uint8Array(8);
  prefix.set(new TextEncoder().encode('OJMD'), 0);
  new DataView(prefix.buffer).setUint32(4, json.byteLength, true);
  const bad = new File([prefix, json, await file.slice(dataStart).arrayBuffer()], 'bad.ojmodel');
  await assert.rejects(importModelFile(bad, { storage: new MemStorage() }), /与来源不一致/);
});

test('删除新建AI：只清除本地权重与配置，保留可能被官方共用的 wasm', async () => {
  const source = new MemStorage();
  await populate(source);
  const file = await exportToFile(source);
  const target = new MemStorage();
  const { record } = await importModelFile(file, { storage: target });
  const removed = await removeLocalModelFiles(record, target);
  assert.equal(removed, 5); // 权重 3 + tokenizer 1 + 配置 1，wasm 保留
  assert.ok(await (await target.open('webllm/wasm')).match(RECORD.model_lib));
  assert.ok(!(await listCachedModelIds([record], target)).has(record.model_id));
  await assert.rejects(removeLocalModelFiles(RECORD, target), /只能删除新建/);
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
  await assert.rejects(importModelFile(truncated, { storage: new MemStorage() }), /数据不完整/);

  const tiny = new File([new Uint8Array(3)], 'tiny.ojmodel');
  await assert.rejects(readModelFileHeader(tiny), /不完整/);
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
