// 重复输出的检测与裁剪
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { findTailRepeat, REPEAT_MIN_UNIT, trimRepetition } from '../web/src/repetition.js';

const UNIT40 = '这是一段会被反复输出的内容，它的长度足够长用来测试循环检测的逻辑，';

test('正常回答不被判断为重复', () => {
  const text = '二分查找的思路是：每次取中点，比较后缩小一半的搜索区间。时间复杂度为 O(log n)。';
  assert.equal(findTailRepeat(text), null);
  assert.equal(trimRepetition(text), text);
});

test('同一段文字连续重复 3 次以上会被检测到，并且只保留第一份', () => {
  const head = '先给出结论：';
  const text = head + UNIT40.repeat(4);
  const hit = findTailRepeat(text);
  assert.ok(hit, '应检测到重复');
  assert.equal(hit.period, UNIT40.length);
  assert.ok(hit.count >= 3);
  assert.equal(trimRepetition(text), head + UNIT40);
});

test('流式输出停在重复单元中间时也能检测到（不依赖对齐位置）', () => {
  const text = '开头' + UNIT40.repeat(3) + UNIT40.slice(0, 17);
  assert.ok(findTailRepeat(text));
  assert.equal(trimRepetition(text), '开头' + UNIT40);
});

test('短于最短单元的重复不视为循环', () => {
  const short = '好的。'.repeat(20); // 单元 3 个字符
  assert.equal(findTailRepeat(short), null);
  assert.ok(REPEAT_MIN_UNIT >= 30);
});

test('只有分隔符的重复（没有字母或汉字）不视为循环', () => {
  const sep = '=' .repeat(120);
  assert.equal(findTailRepeat(sep), null);
});

test('只重复两次不视为循环', () => {
  const text = '前文' + UNIT40.repeat(2);
  assert.equal(findTailRepeat(text), null);
});

test('参数行可设置 repetitionPenalty，范围为 1–2', async () => {
  const { parseFeedbackParams, sanitizeParams } = await import('../web/src/adaptation.js');
  assert.deepEqual(parseFeedbackParams('参数：repetitionPenalty=1.2'), { repetitionPenalty: 1.2 });
  assert.deepEqual(parseFeedbackParams('参数：repetitionPenalty = 5'), { repetitionPenalty: 2 });
  assert.deepEqual(sanitizeParams({ repetitionPenalty: 0.5 }), { repetitionPenalty: 1 });
});
