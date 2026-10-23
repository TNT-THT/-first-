import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_PRESET_ID,
  findPreset,
  getPresetId,
  layerParams,
  PRESET_KEY,
  PRESETS,
  presetOverrides,
  setPresetId,
} from '../web/src/presets.js';
import { WINDOW_LIMITS } from '../web/src/budget.js';

function memoryStorage(init = {}) {
  const data = { ...init };
  return {
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = String(v);
    },
    data,
  };
}

test('预设列表：id 唯一，默认预设在第一位且不覆盖任何数值', () => {
  const ids = PRESETS.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length);
  assert.equal(PRESETS[0].id, DEFAULT_PRESET_ID);
  assert.deepEqual(presetOverrides(DEFAULT_PRESET_ID), {});
});

test('预设取值都在引擎允许的范围内', () => {
  for (const p of PRESETS) {
    if (p.temperature !== null) assert.ok(p.temperature >= 0 && p.temperature <= 1.5, p.id);
    if (p.maxTokens !== null) assert.ok(p.maxTokens >= 64 && p.maxTokens <= 16384, p.id);
    if (p.contextWindow !== null) {
      assert.ok(p.contextWindow >= WINDOW_LIMITS.contextWindow.min, p.id);
      assert.ok(p.contextWindow <= WINDOW_LIMITS.contextWindow.max, p.id);
    }
    if (p.repetitionPenalty !== null) assert.ok(p.repetitionPenalty >= 1 && p.repetitionPenalty <= 2, p.id);
  }
});

test('长文章预设：窗口 20000、输出 16384、关闭思考', () => {
  const p = findPreset('long-form');
  assert.equal(p.contextWindow, 20000);
  assert.equal(p.maxTokens, 16384);
  assert.equal(p.thinking, false);
  assert.deepEqual(presetOverrides('long-form'), {
    temperature: 0.8,
    maxTokens: 16384,
    contextWindow: 20000,
    slidingWindow: 20000,
    repetitionPenalty: 1.15,
  });
});

test('未设置、无效或损坏的存储值都回到默认预设', () => {
  assert.equal(getPresetId(memoryStorage()), DEFAULT_PRESET_ID);
  assert.equal(getPresetId(memoryStorage({ [PRESET_KEY]: 'no-such' })), DEFAULT_PRESET_ID);
  const broken = {
    getItem() {
      throw new Error('denied');
    },
  };
  assert.equal(getPresetId(broken), DEFAULT_PRESET_ID);
});

test('设置预设：按浏览器保存；无效 id 不写入', () => {
  const storage = memoryStorage();
  assert.equal(setPresetId('code', storage), true);
  assert.equal(getPresetId(storage), 'code');
  assert.equal(setPresetId('bogus', storage), false);
  assert.equal(getPresetId(storage), 'code');
});

test('分层优先级：默认值 < 预设 < 管理员参数', () => {
  const defaults = { temperature: 0.3, maxTokens: 16384, contextWindow: 16384 };
  const preset = { temperature: 0.8, maxTokens: 16384, contextWindow: 20000 };
  // 只有预设：预设覆盖默认值
  assert.deepEqual(layerParams(defaults, preset, {}), { temperature: 0.8, maxTokens: 16384, contextWindow: 20000 });
  // 管理员参数优先于预设
  assert.deepEqual(layerParams(defaults, preset, { temperature: 0.1 }), {
    temperature: 0.1,
    maxTokens: 16384,
    contextWindow: 20000,
  });
  // 预设为空（默认预设）时沿用默认值
  assert.deepEqual(layerParams(defaults, {}, {}), defaults);
});

test('分层：管理员参数中的 null/undefined 不覆盖预设', () => {
  const merged = layerParams({ temperature: 0.3 }, { temperature: 0.8 }, { temperature: undefined });
  assert.equal(merged.temperature, 0.8);
  const merged2 = layerParams({ temperature: 0.3 }, { temperature: 0.8 }, { temperature: null });
  assert.equal(merged2.temperature, 0.8);
});

test('分层不修改入参', () => {
  const defaults = { temperature: 0.3 };
  const admin = { temperature: 0.2 };
  layerParams(defaults, {}, admin);
  assert.deepEqual(defaults, { temperature: 0.3 });
  assert.deepEqual(admin, { temperature: 0.2 });
});
