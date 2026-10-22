// 待转换的第三方模型：只登记、不进入官方列表，且与官方模型的 model_id 完全分开。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pendingModels, PENDING_STATUS } from '../web/src/pending-models.js';
import { buildModelList } from '../web/src/model-source.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const webConfig = JSON.parse(readFileSync(join(root, 'web', 'public', 'config.json'), 'utf8'));
const { prebuiltAppConfig } = await import(join(root, 'node_modules', '@mlc-ai', 'web-llm', 'lib', 'index.js'));

const webllm = webConfig.webllm;
const pending = pendingModels(webllm);
const officialIds = new Set([
  ...webllm.models.map((m) => m.id),
  ...webllm.extraRecords.map((r) => r.model_id),
  ...prebuiltAppConfig.model_list.map((r) => r.model_id),
]);

const THIRD_PARTY = [
  {
    id: 'Huihui-Qwen3.5-4B-abliterated-q4f16_1-MLC',
    repo: 'https://huggingface.co/kamekichi1231/Huihui-Qwen3.5-4B-abliterated-q4f16_1-MLC',
    lib: 'Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm',
    vram: 3867.82,
  },
  {
    id: 'Leyten-Qwen3.5-9B-compute-q4f16_1-MLC',
    repo: 'https://huggingface.co/Leyten/Qwen3.5-9B-compute-q4f16_1-MLC',
    lib: 'Qwen3.5-9B-q4f16_1_cs1k-webgpu.wasm',
    vram: 6433.01,
  },
  {
    id: 'Huihui-Qwen3.5-2B-abliterated-q4f16_1-MLC',
    repo: 'https://huggingface.co/RaiseRuntimeError/Huihui-Qwen3.5-2B-abliterated-q4f16_1-MLC',
    lib: 'Qwen3.5-2B-q4f16_1_cs1k-webgpu.wasm',
    vram: 2245.44,
  },
];

test('待转换列表已清空：两个第三方模型已改为正式登记（不再是待转换）', () => {
  assert.deepEqual(webllm.pendingModels, []);
  assert.equal(pending.length, 0);
});

test('两个第三方模型登记在下拉框与 extraRecords 中，指向指定的 MLC 仓库与官方 cs1k 运行库', () => {
  for (const t of THIRD_PARTY) {
    const item = webllm.models.find((m) => m.id === t.id);
    assert.ok(item, `下拉框缺少 ${t.id}`);
    assert.equal(item.reasoning, true, t.id);
    assert.match(item.label, /纯文本/, t.id);
    assert.match(item.label, /去审查/, t.id);
    const rec = webllm.extraRecords.find((r) => r.model_id === t.id);
    assert.ok(rec, `extraRecords 缺少 ${t.id}`);
    assert.equal(rec.model, t.repo, t.id);
    assert.ok(rec.model_lib.endsWith(t.lib), t.id);
    assert.equal(rec.vram_required_MB, t.vram, t.id);
    assert.deepEqual(rec.overrides, { context_window_size: 4096, max_history_size: 1 }, t.id);
  }
});

test('第三方模型的 model_id 与官方预编译记录不冲突，且互不重复', () => {
  const builtin = new Set(prebuiltAppConfig.model_list.map((r) => r.model_id));
  for (const t of THIRD_PARTY) assert.equal(builtin.has(t.id), false, `${t.id} 与官方记录冲突`);
  assert.equal(new Set(THIRD_PARTY.map((t) => t.id)).size, THIRD_PARTY.length);
});

test('第三方模型进入模型列表（可被加载），待转换列表中没有残留条目', () => {
  const list = buildModelList(prebuiltAppConfig.model_list, webllm.extraRecords, '');
  for (const t of THIRD_PARTY) assert.equal(list.some((r) => r.model_id === t.id), true, t.id);
});

test('pendingModels：忽略非 pending 状态、缺字段、与官方冲突和重复的条目', () => {
  const cfg = {
    models: [{ id: 'Official-A' }],
    extraRecords: [{ model_id: 'Official-B' }],
    pendingModels: [
      { id: 'X1', label: '甲', status: 'pending', note: '备注：a' },
      { id: 'X1', label: '甲重复', status: 'pending' },
      { id: 'Official-A', label: '冲突', status: 'pending' },
      { id: 'Official-B', label: '冲突2', status: 'pending' },
      { id: 'X2', label: '', status: 'pending' },
      { id: 'X3', label: '已启用', status: 'enabled' },
      { id: '', label: '无 id', status: 'pending' },
    ],
  };
  assert.deepEqual(pendingModels(cfg), [{ id: 'X1', label: '甲', note: '备注：a', source: '' }]);
  assert.deepEqual(pendingModels({}), []);
});
