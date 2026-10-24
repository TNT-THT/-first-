// 每个模型都有备注：文本模型（webllm）与生图模型都写明参数量级与显存占用（估计值，并注明未实测）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(fs.readFileSync(path.join(here, '..', 'web', 'public', 'config.json'), 'utf8'));

test('文本模型：每个都有备注，写明参数量级与显存估计', () => {
  assert.ok(cfg.webllm.models.length > 0);
  for (const m of cfg.webllm.models) {
    assert.ok(m.note, `${m.id} 缺少备注`);
    assert.match(m.note, /参数量级：约 [\d.]+B/, `${m.id} 缺少参数量级`);
    assert.match(m.note, /显存：标签中约 [\d.]+ GB（估计/, `${m.id} 缺少显存估计`);
  }
  for (const m of cfg.webllm.models.filter((x) => x.id.startsWith('Qwen3.5'))) {
    assert.match(m.note, /尚未在本项目中实测/, `${m.id} 的备注应注明未实测`);
  }
});

test('生图模型：每个都有备注（参数量级与显存），标签中也有显存估计', () => {
  assert.equal(cfg.imageModels.length, 13);
  for (const m of cfg.imageModels) {
    assert.match(m.note, /参数量级：/, `${m.id} 缺少参数量级`);
    assert.match(m.note, /显存/, `${m.id} 缺少显存`);
    assert.match(m.label, /显存估计|显存需求高/, `${m.id} 标签缺少显存`);
  }
});

test('生图备注：不再写入未核对的许可证条款编号', () => {
  for (const m of cfg.imageModels) {
    assert.ok(!m.note.includes('附件 A'), `${m.id} 仍含未核对的许可证条款`);
  }
});
