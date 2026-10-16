// 训练清单必须与网页下拉框和 web-llm 内置记录保持一致（运行库、覆盖项、原始地址）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildCatalog } from '../training/build_catalog.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const webConfig = JSON.parse(readFileSync(join(root, 'web', 'public', 'config.json'), 'utf8'));
const catalog = JSON.parse(readFileSync(join(root, 'training', 'catalog.json'), 'utf8'));
const { prebuiltAppConfig } = await import(join(root, 'node_modules', '@mlc-ai', 'web-llm', 'lib', 'index.js'));
const records = new Map(prebuiltAppConfig.model_list.map((r) => [r.model_id, r]));

test('catalog.json 与生成脚本的输出一致（修改网页模型后需重新生成）', async () => {
  const fresh = await buildCatalog();
  assert.deepEqual(catalog.models, fresh.models);
});

test('清单覆盖网页下拉框中的每个模型，且顺序一致', () => {
  assert.deepEqual(catalog.models.map((m) => m.id), webConfig.webllm.models.map((m) => m.id));
});

test('每个条目的运行库、覆盖项与原始地址与 web-llm 记录一致', () => {
  for (const m of catalog.models) {
    const rec = records.get(m.id);
    assert.ok(rec, `web-llm 中没有 ${m.id}`);
    assert.equal(m.model_lib_url, rec.model_lib, m.id);
    assert.deepEqual(m.overrides, rec.overrides || {}, m.id);
    assert.equal(m.source_model, rec.model, m.id);
    assert.equal(m.context_window_size, (rec.overrides || {}).context_window_size ?? null, m.id);
  }
});

test('清单中的事实字段都已填写，且未标记为已核对', () => {
  for (const m of catalog.models) {
    assert.equal(m.verified, false, m.id);
    assert.match(m.hf_repo, /^[^/]+\/[^/]+$/, m.id);
    assert.ok(m.conv_template, m.id);
  }
});
