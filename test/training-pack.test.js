// 跨语言检查：training/pack_ojmodel.py 打包的文件必须能被网页端的 importModelFile 导入。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { importModelFile, listCachedModelIds, readModelFileHeader } from '../web/src/model-file.js';

const here = dirname(fileURLToPath(import.meta.url));
const PACK = join(here, '..', 'training', 'pack_ojmodel.py');
const WASM_URL = 'https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Demo.wasm';

function python() {
  for (const cmd of ['python3', 'python']) {
    const r = spawnSync(cmd, ['--version']);
    if (r.status === 0) return cmd;
  }
  return null;
}
const PY = python();

class MemCache {
  constructor() {
    this.map = new Map();
  }
  async match(req) {
    const v = this.map.get(typeof req === 'string' ? req : req.url);
    return v ? new Response(v.slice(0), { status: 200 }) : undefined;
  }
  async put(req, res) {
    this.map.set(typeof req === 'string' ? req : req.url, new Uint8Array(await res.arrayBuffer()));
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

function makeFixture(root) {
  const dir = join(root, 'mlc');
  mkdirSync(dir);
  writeFileSync(join(dir, 'tensor-cache.json'), JSON.stringify({ records: [{ dataPath: 'params_shard_0.bin' }] }));
  writeFileSync(join(dir, 'params_shard_0.bin'), Buffer.from(Array.from({ length: 4096 }, (_, i) => (i * 7) & 0xff)));
  writeFileSync(join(dir, 'tokenizer.json'), '{"version":"1.0"}');
  writeFileSync(join(dir, 'mlc-chat-config.json'), '{"model_type":"qwen2"}');
  const wasm = join(root, 'model.wasm');
  writeFileSync(wasm, Buffer.from([0, 97, 115, 109, 1, 0, 0, 0]));
  return { dir, wasm };
}

test('训练运行时打包的 .ojmodel 可以被网页端导入（跨语言格式一致）', { skip: !PY && 'python3 不可用' }, async () => {
  const root = mkdtempSync(join(tmpdir(), 'ojpack-'));
  try {
    const { dir, wasm } = makeFixture(root);
    const out = join(root, 'my-tuned.ojmodel');
    const r = spawnSync(PY, [
      PACK, '--dir', dir, '--wasm', wasm, '--model-lib-url', WASM_URL,
      '--model-id', 'my-tuned', '--label', '我的微调版', '--out', out,
    ], { encoding: 'utf8' });
    assert.equal(r.status, 0, r.stderr);

    const file = new File([readFileSync(out)], 'my-tuned.ojmodel');
    const { header } = await readModelFileHeader(file);
    assert.equal(header.label, '我的微调版');
    assert.equal(header.source.model_lib, WASM_URL);

    const storage = new MemStorage();
    const { record, entryCount } = await importModelFile(file, { storage, localId: 'trained-1' });
    assert.equal(entryCount, header.entries.length);
    assert.ok(record.model.startsWith('https://local.oj-ai.invalid/models/'));
    assert.ok((await listCachedModelIds([record], storage)).has('trained-1'));
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('打包失败时返回非零退出码并给出中文原因', { skip: !PY && 'python3 不可用' }, () => {
  const root = mkdtempSync(join(tmpdir(), 'ojpack-'));
  try {
    const { dir, wasm } = makeFixture(root);
    const r = spawnSync(PY, [
      PACK, '--dir', dir, '--wasm', wasm, '--model-lib-url', WASM_URL,
      '--model-id', 'bad id', '--label', 'x', '--out', join(root, 'x.ojmodel'),
    ], { encoding: 'utf8' });
    assert.notEqual(r.status, 0);
    assert.match(r.stderr, /打包失败/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
