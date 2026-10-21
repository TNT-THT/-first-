// 滑动窗口开关的偏好（默认关闭，与思考开关相同保存在本浏览器）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { getSlidingPref, setSlidingPref, SLIDING_PREF_KEY } from '../web/src/sliding.js';

function memoryStorage(initial = {}) {
  const data = { ...initial };
  return {
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = String(v);
    },
  };
}

test('滑动窗口默认关闭', () => {
  assert.equal(getSlidingPref(memoryStorage()), false);
});

test('开关可保存并读回', () => {
  const s = memoryStorage();
  setSlidingPref(true, s);
  assert.equal(getSlidingPref(s), true);
  assert.equal(s.getItem(SLIDING_PREF_KEY), 'on');
  setSlidingPref(false, s);
  assert.equal(getSlidingPref(s), false);
  assert.equal(s.getItem(SLIDING_PREF_KEY), 'off');
});

test('存储不可用时不抛错，按关闭处理', () => {
  const broken = {
    getItem() {
      throw new Error('denied');
    },
    setItem() {
      throw new Error('denied');
    },
  };
  assert.equal(getSlidingPref(broken), false);
  assert.doesNotThrow(() => setSlidingPref(true, broken));
});
