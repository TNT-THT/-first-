// 历史截断的丢弃粒度（chunk）：chunk=1 与原来的逐条丢弃一致；chunk>1 时起点固定，减少 KV 缓存失效
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildContext } from '../web/src/store.js';
import { estimatePromptTokens, fitHistory } from '../web/src/budget.js';

/** 原始实现（逐条从前往后丢弃），用于对照 chunk=1。 */
function buildContextReference(messages, { maxMessages, maxChars }) {
  const picked = [];
  let total = 0;
  for (let i = messages.length - 1; i >= 0 && picked.length < maxMessages; i--) {
    const m = messages[i];
    if (m.error || !m.content || m.fixed) continue;
    if (total + m.content.length > maxChars && picked.length > 0) break;
    total += m.content.length;
    picked.unshift({ role: m.role, content: m.content });
  }
  while (picked.length && picked[0].role !== 'user') picked.shift();
  return picked;
}

function rng(seed) {
  let x = seed;
  return () => {
    x = (x * 1103515245 + 12345) & 0x7fffffff;
    return x / 0x7fffffff;
  };
}

test('buildContext（chunk=1）与原逐条实现完全一致（随机输入）', () => {
  const rand = rng(7);
  for (let trial = 0; trial < 300; trial++) {
    const n = Math.floor(rand() * 40);
    const msgs = [];
    for (let i = 0; i < n; i++) {
      const len = rand() < 0.1 ? 0 : Math.floor(rand() * 900);
      msgs.push({
        role: i % 2 ? 'assistant' : 'user',
        content: '字'.repeat(len),
        ...(rand() < 0.05 ? { error: true } : {}),
        ...(rand() < 0.05 ? { fixed: true } : {}),
      });
    }
    const maxMessages = 1 + Math.floor(rand() * 16);
    const maxChars = Math.floor(rand() * 4000);
    assert.deepEqual(
      buildContext(msgs, { maxMessages, maxChars, chunk: 1 }),
      buildContextReference(msgs, { maxMessages, maxChars }),
      `trial ${trial}`,
    );
  }
});

test('buildContext（chunk=6）：超出后一次丢一段，之后几轮起点保持不变', () => {
  const msgs = [];
  const starts = [];
  for (let t = 1; t <= 40; t++) {
    msgs.push({ role: 'user', content: `问${t}` + '问'.repeat(20) });
    msgs.push({ role: 'assistant', content: `答${t}` + '答'.repeat(300) });
    const ctx = buildContext(msgs.slice(0, -1).concat([]), { maxMessages: 15, maxChars: 8000, chunk: 6 });
    // 当前提问包含在历史中（与 send() 一致）
    const withQ = buildContext(msgs, { maxMessages: 15, maxChars: 8000, chunk: 6 });
    starts.push(withQ[0].content.match(/问(\d+)/)?.[1] ?? null);
    assert.ok(ctx.length >= 1);
  }
  // 起点变化的次数应明显少于逐条丢弃（逐条：从第 8 轮起几乎每轮都变）
  let changes = 0;
  for (let i = 1; i < starts.length; i++) if (starts[i] !== starts[i - 1]) changes += 1;
  assert.ok(changes <= 14, `起点变化 ${changes} 次`);
  // 任何时候都不超过上限，且第一条是用户消息
  for (let t = 1; t <= 40; t++) {
    const m = [];
    for (let i = 1; i <= t; i++) {
      m.push({ role: 'user', content: `问${i}` });
      m.push({ role: 'assistant', content: `答${i}` });
    }
    const ctx = buildContext(m, { maxMessages: 15, maxChars: 100000, chunk: 6 });
    assert.ok(ctx.length <= 15);
    assert.equal(ctx[0].role, 'user');
  }
});

test('fitHistory（chunk=6）：保留的起点落在 1 + 6k 上，系统提示与最后一条始终保留', () => {
  const sys = { role: 'system', content: '系统' };
  const make = (n) => {
    const out = [sys];
    for (let i = 0; i < n; i++) out.push({ role: i % 2 ? 'assistant' : 'user', content: `第${i}条` + '字'.repeat(200) });
    return out;
  };
  for (let n = 3; n <= 40; n++) {
    const msgs = make(n);
    const out = fitHistory(msgs, 1500, { chunk: 6 });
    assert.equal(out[0], msgs[0]);
    assert.equal(out[out.length - 1], msgs[msgs.length - 1]);
    if (out.length < msgs.length) {
      const start = msgs.indexOf(out[1]);
      // 只有存在能放下的对齐起点时才要求对齐；都放不下时退回逐条丢弃（保证不超出窗口）
      const aligned = [];
      for (let k = 1; k <= n - 1; k += 6) aligned.push(k);
      const someAlignedFits = aligned.some((k) => estimatePromptTokens([msgs[0], ...msgs.slice(k)]) <= 1500);
      if (someAlignedFits) assert.equal((start - 1) % 6, 0, `n=${n} start=${start}`);
    }
    assert.ok(estimatePromptTokens(out) <= 1500 || out.length === 2);
  }
});

test('fitHistory：chunk 默认为 1，结果与逐条丢弃一致', () => {
  const sys = { role: 'system', content: '系统' };
  const msgs = [sys];
  for (let i = 0; i < 20; i++) msgs.push({ role: i % 2 ? 'assistant' : 'user', content: '字'.repeat(100 + i * 7) });
  const withDefault = fitHistory(msgs, 900);
  const withChunk1 = fitHistory(msgs, 900, { chunk: 1 });
  assert.deepEqual(withDefault, withChunk1);
});
