import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_MODEL_KEY,
  getSavedDefaultModel,
  saveDefaultModel,
  resolveStartupModel,
} from '../web/src/default-model.js';

function memoryStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
  };
}

test('保存的默认模型可以读回，清空后恢复为无保存', () => {
  const s = memoryStorage();
  assert.equal(getSavedDefaultModel(s), null);
  saveDefaultModel('Qwen3-8B-q4f16_1-MLC', s);
  assert.equal(s.getItem(DEFAULT_MODEL_KEY), 'Qwen3-8B-q4f16_1-MLC');
  assert.equal(getSavedDefaultModel(s), 'Qwen3-8B-q4f16_1-MLC');
  saveDefaultModel(null, s);
  assert.equal(getSavedDefaultModel(s), null);
});

test('存储不可用时不抛错', () => {
  const broken = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('denied'); },
    removeItem() { throw new Error('denied'); },
  };
  assert.equal(getSavedDefaultModel(broken), null);
  assert.doesNotThrow(() => saveDefaultModel('x', broken));
  assert.doesNotThrow(() => saveDefaultModel(null, broken));
});

test('启动模型：已保存且可选则用保存的，否则用配置的 defaultModel', () => {
  const availableIds = ['Qwen2.5-3B-Instruct-q4f16_1-MLC', 'Qwen3-8B-q4f16_1-MLC'];
  const defaultModel = 'Qwen2.5-3B-Instruct-q4f16_1-MLC';
  assert.equal(resolveStartupModel({ availableIds, defaultModel, saved: 'Qwen3-8B-q4f16_1-MLC' }), 'Qwen3-8B-q4f16_1-MLC');
  assert.equal(resolveStartupModel({ availableIds, defaultModel, saved: null }), defaultModel);
  // 已删除的“新建 AI”不在列表中：忽略保存值
  assert.equal(resolveStartupModel({ availableIds, defaultModel, saved: 'local-deleted' }), defaultModel);
});
