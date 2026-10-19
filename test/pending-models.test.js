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

test('配置中登记了两个待转换模型，且都标记为 pending', () => {
  assert.equal(webllm.pendingModels.length, 2);
  assert.equal(pending.length, 2);
  for (const m of webllm.pendingModels) assert.equal(m.status, PENDING_STATUS, m.id);
});

test('两个待转换模型对应指定的 Hugging Face 仓库，并带有备注', () => {
  const byRepo = Object.fromEntries(pending.map((m) => [m.source, m]));
  assert.ok(byRepo['https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive']);
  assert.ok(byRepo['https://huggingface.co/DavidAU/Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-GGUF']);
  for (const m of pending) {
    assert.match(m.label, /待转换/, m.id);
    assert.match(m.label, /暂不可用/, m.id);
    assert.match(m.note, /^备注：/, m.id);
    assert.match(m.note, /GGUF/, m.id);
  }
});

test('待转换模型与官方模型的 model_id 完全分开', () => {
  for (const m of pending) {
    assert.equal(officialIds.has(m.id), false, `${m.id} 与官方模型冲突`);
  }
  assert.equal(new Set(pending.map((m) => m.id)).size, pending.length);
});

test('待转换模型不进入模型列表（不会被当作可加载的运行记录）', () => {
  const list = buildModelList(prebuiltAppConfig.model_list, webllm.extraRecords, '');
  for (const m of pending) {
    assert.equal(list.some((r) => r.model_id === m.id), false, m.id);
  }
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
