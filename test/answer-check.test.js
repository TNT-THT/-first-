import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  CHECKS_KEY,
  MAX_CHECK_RULES_PER_MODEL,
  addCheckRule,
  checkRuleError,
  findViolations,
  listCheckRules,
  normalizeForCheck,
  removeCheckRule,
} from '../web/src/answer-check.js';

function memoryStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
  };
}

test('规范化：忽略大小写、全角半角与空白', () => {
  assert.equal(normalizeForCheck('O（ n ）'), normalizeForCheck('o(n)'));
  assert.equal(normalizeForCheck('Ｏ(Ｎ)'), 'o(n)');
});

test('输入检查：空值、过短、与正确说法相同都会被拒绝', () => {
  assert.match(checkRuleError('', 'x'), /错误的说法/);
  assert.match(checkRuleError('O(n)', ' '), /正确的说法/);
  assert.match(checkRuleError('n', 'O(log n)'), /太短/);
  assert.match(checkRuleError('O(N)', 'o(n)'), /相同/);
  assert.equal(checkRuleError('O(n)', 'O(log n)'), null);
});

test('新增、同一错误说法更新正确说法、删除，并按模型分开保存', () => {
  const storage = memoryStorage();
  const a = addCheckRule('A', { wrong: 'O(n)', right: 'O(log n)' }, { storage, now: 1 });
  assert.equal(a.updated, false);
  assert.equal(listCheckRules('A', { storage }).length, 1);
  // 规范化后相同的错误说法：更新，不重复新增
  const b = addCheckRule('A', { wrong: 'o（n）', right: 'O(log n) 对于有序数组' }, { storage, now: 2 });
  assert.equal(b.updated, true);
  assert.equal(listCheckRules('A', { storage }).length, 1);
  assert.equal(listCheckRules('A', { storage })[0].right, 'O(log n) 对于有序数组');
  // 另一个模型不受影响
  assert.equal(listCheckRules('B', { storage }).length, 0);
  removeCheckRule('A', a.rule.id, { storage });
  assert.equal(listCheckRules('A', { storage }).length, 0);
  assert.equal(JSON.parse(storage.getItem(CHECKS_KEY)).models.A, undefined);
});

test('每个模型最多保留的规则数', () => {
  const storage = memoryStorage();
  for (let i = 0; i < MAX_CHECK_RULES_PER_MODEL + 5; i += 1) {
    addCheckRule('A', { wrong: `错误${i}说法`, right: `正确${i}说法` }, { storage, now: i + 1 });
  }
  const rules = listCheckRules('A', { storage });
  assert.equal(rules.length, MAX_CHECK_RULES_PER_MODEL);
  assert.equal(rules.at(-1).wrong, `错误${MAX_CHECK_RULES_PER_MODEL + 4}说法`);
});

test('命中：回答中出现错误说法（忽略大小写与空白）', () => {
  const rules = [{ wrong: 'O(n)', right: 'O(log n)' }];
  assert.equal(findViolations('二分查找的最坏复杂度是 o (n)。', rules).length, 1);
  assert.equal(findViolations('二分查找的最坏复杂度是 O(log n)。', rules).length, 0);
});

test('不误判：正确说法中含有错误说法，或错误说法前有否定词', () => {
  const rules = [{ wrong: 'O(n)', right: 'O(log n)' }];
  // 只出现了正确说法 O(log n)，其中的 “n” 不应被当成错误
  assert.equal(findViolations('复杂度是 O(log n)', rules).length, 0);
  // 否定句：“不是 O(n)”表示否认错误说法
  assert.equal(findViolations('二分查找是 O(log n)，不是 O(n)', rules).length, 0);
  assert.equal(findViolations('二分查找并非 O(n)', rules).length, 0);
});

test('规则为空或回答为空时不命中', () => {
  assert.deepEqual(findViolations('', [{ wrong: 'O(n)', right: 'x' }]), []);
  assert.deepEqual(findViolations('O(n)', []), []);
});

test('存储不可用或数据损坏时不抛错', () => {
  const broken = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('denied'); },
    removeItem() { throw new Error('denied'); },
  };
  assert.deepEqual(listCheckRules('A', { storage: broken }), []);
  assert.doesNotThrow(() => addCheckRule('A', { wrong: 'O(n)', right: 'O(log n)' }, { storage: broken }));
  const bad = { getItem: () => '{not json', setItem() {}, removeItem() {} };
  assert.deepEqual(listCheckRules('A', { storage: bad }), []);
});
