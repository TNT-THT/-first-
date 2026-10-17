// 上下文窗口（65,536 token）、token 预算、前缀稳定性与显存降级。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildContext, estimateTokens, historyTokenBudget, CONTEXT_SAFETY_TOKENS } from '../web/src/store.js';
import {
  applyContextWindow,
  contextWindowLadder,
  DEFAULT_CONTEXT_WINDOW,
  isMemoryError,
} from '../web/src/model-source.js';

test('token 估算：中文按约 1.5 token/字，英文按约 3 字符/token，空文本为 0', () => {
  assert.equal(estimateTokens(''), 0);
  assert.equal(estimateTokens(undefined), 0);
  assert.equal(estimateTokens('abcdef'), 2);
  assert.equal(estimateTokens('你好'), 3);
  assert.ok(estimateTokens('中'.repeat(100)) > estimateTokens('a'.repeat(100)));
});

test('历史预算：窗口减去输出预留（不超过窗口一半）、系统提示词与余量', () => {
  assert.equal(
    historyTokenBudget({ windowTokens: 65536, maxOutputTokens: 16384, reservedTokens: 300 }),
    65536 - 16384 - 300 - CONTEXT_SAFETY_TOKENS,
  );
  // 显存不足降级到 4096 时，输出预留被限制为窗口的一半，仍保留可用的历史空间
  const small = historyTokenBudget({ windowTokens: 4096, maxOutputTokens: 16384, reservedTokens: 100 });
  assert.equal(small, 4096 - 2048 - 100 - CONTEXT_SAFETY_TOKENS);
  assert.ok(small > 0);
  // 极端情况下仍有最小预算
  assert.equal(historyTokenBudget({ windowTokens: 512, maxOutputTokens: 16384, reservedTokens: 9999 }), 256);
});

test('按 token 预算构造上下文：不超过预算，保留最新一条，第一条为用户消息', () => {
  const msgs = [];
  for (let i = 0; i < 40; i++) {
    msgs.push({ role: 'user', content: `问题${i} ` + '背景'.repeat(300) });
    msgs.push({ role: 'assistant', content: `回答${i} ` + '内容'.repeat(300), model: 'A', modelLabel: 'A' });
  }
  msgs.push({ role: 'user', content: '最后的问题' });
  const budget = 6000;
  const ctx = buildContext(msgs, { maxMessages: 200, tokenBudget: budget, stableBlock: 8, currentModel: 'A' });
  assert.equal(ctx.at(-1).content, '最后的问题');
  assert.equal(ctx[0].role, 'user');
  const used = ctx.reduce((n, m) => n + estimateTokens(m.content) + 4, 0);
  assert.ok(used <= budget, `used ${used} > budget ${budget}`);
  assert.ok(ctx.length < msgs.length);
});

test('token 模式：多轮对话中历史前缀保持稳定（未超出预算时每轮都是上一轮的延续）', () => {
  const history = [];
  let prev = null;
  for (let turn = 0; turn < 20; turn++) {
    history.push({ role: 'user', content: `第${turn}个问题` });
    const ctx = buildContext(history, { maxMessages: 200, tokenBudget: 100000, stableBlock: 8, currentModel: 'A' });
    if (prev) assert.deepEqual(ctx.slice(0, prev.length), prev, `turn ${turn} 的前缀被改变`);
    prev = ctx;
    history.push({ role: 'assistant', content: `第${turn}个回答`, model: 'A', modelLabel: 'A' });
  }
  assert.equal(prev.length, 39);
});

test('token 模式：超出预算后起点按 8 条对齐，同一区间内前缀不变', () => {
  const history = [];
  const starts = [];
  for (let turn = 0; turn < 60; turn++) {
    history.push({ role: 'user', content: '问'.repeat(400) });
    const ctx = buildContext(history, { maxMessages: 200, tokenBudget: 5000, stableBlock: 8, currentModel: 'A' });
    starts.push(history.length - ctx.length);
    history.push({ role: 'assistant', content: '答'.repeat(400), model: 'A', modelLabel: 'A' });
  }
  // 起点只在 8 的倍数上取值（超出预算之后）
  for (const s of starts) assert.ok(s === 0 || s % 8 === 0, `起点 ${s} 不是 8 的倍数`);
  // 起点单调不减，并且相邻轮次之间最多前移一次
  for (let i = 1; i < starts.length; i++) assert.ok(starts[i] >= starts[i - 1]);
  const distinct = new Set(starts).size;
  // 60 轮共 120 条消息，起点每 8 条消息才会变化一次
  assert.ok(distinct <= Math.ceil((60 * 2) / 8) + 1, `起点变化过于频繁：${distinct} 种`);
});

test('字符模式（远程）：与原行为一致，不做对齐', () => {
  const msgs = [];
  for (let i = 0; i < 10; i++) msgs.push({ role: i % 2 ? 'assistant' : 'user', content: `消息${i}` });
  const a = buildContext(msgs, { maxMessages: 4 });
  assert.equal(a.length, 4);
  assert.equal(a[0].content, '消息6'); // 最近 4 条为 消息6 ~ 消息9，第一条为用户消息
});

test('窗口覆盖：为每条记录设置 context_window_size，不修改原记录，保留滑动窗口记录', () => {
  const records = [
    { model_id: 'a', overrides: { context_window_size: 4096 } },
    { model_id: 'b' },
    { model_id: 'c', overrides: { sliding_window_size: 2048 } },
  ];
  const out = applyContextWindow(records, 65536);
  assert.equal(out[0].overrides.context_window_size, 65536);
  assert.equal(out[1].overrides.context_window_size, 65536);
  assert.deepEqual(out[2], records[2]);
  assert.equal(records[0].overrides.context_window_size, 4096);
  assert.equal(records[1].overrides, undefined);
  assert.deepEqual(applyContextWindow(records, 0), records);
  assert.deepEqual(applyContextWindow(records, 'abc'), records);
});

test('显存降级序列：先请求的窗口，再依次更小的档位', () => {
  assert.equal(DEFAULT_CONTEXT_WINDOW, 65536);
  assert.deepEqual(contextWindowLadder(65536), [65536, 32768, 16384, 8192, 4096]);
  assert.deepEqual(contextWindowLadder(16384), [16384, 8192, 4096]);
  assert.deepEqual(contextWindowLadder(3000), [3000]);
  assert.deepEqual(contextWindowLadder(undefined), [65536, 32768, 16384, 8192, 4096]);
});

test('显存错误判断：只有显存相关的错误才会触发降级重试', () => {
  assert.equal(isMemoryError({ name: 'DeviceLostError', message: 'x' }), true);
  assert.equal(isMemoryError(new Error('GPU ran out of memory (OOM)')), true);
  assert.equal(isMemoryError(new Error("Failed to execute 'createBuffer' on 'GPUDevice'")), true);
  assert.equal(isMemoryError(new Error('This model requires WebGPU extension shader-f16')), false);
  assert.equal(isMemoryError(new Error('Failed to fetch')), false);
  assert.equal(isMemoryError(undefined), false);
});
