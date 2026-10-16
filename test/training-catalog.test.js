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
  assert.deepEqual(
    catalog.models.filter((m) => m.in_dropdown).map((m) => m.id),
    webConfig.webllm.models.map((m) => m.id),
  );
});

test('仅训练目录的超大模型不在下拉框中，且没有官方运行库', () => {
  const dropdownIds = new Set(webConfig.webllm.models.map((m) => m.id));
  const only = catalog.models.filter((m) => !m.in_dropdown);
  assert.deepEqual(only.map((m) => m.id).sort(), ['Qwen2.5-32B-Instruct', 'Yi-1.5-34B-Chat']);
  for (const m of only) {
    assert.equal(dropdownIds.has(m.id), false, m.id);
    assert.equal(records.has(m.id), false, m.id);
    assert.equal(m.model_lib_url, null, m.id);
    assert.equal(m.size_class, '超大', m.id);
    assert.equal(m.inference_vram_source, 'estimate', m.id);
  }
});

test('下拉框标签都写明了推理显存需求，超大模型的显存不低于 30 GB', () => {
  for (const m of webConfig.webllm.models) {
    assert.match(m.label, /需约 [\d.]+ GB 显存/, m.id);
  }
  const big = catalog.models.filter((m) => m.size_class === '超大' && m.in_dropdown);
  assert.equal(big.length, 2);
  for (const m of big) assert.ok(m.inference_vram_gb >= 30, m.id);
});

test('每个条目的运行库、覆盖项与原始地址与 web-llm 记录一致', () => {
  for (const m of catalog.models.filter((x) => x.in_dropdown)) {
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
