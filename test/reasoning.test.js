// 推理模型：思考过程拆分、思考模式开关、历史会话折叠窗口（纯逻辑）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { splitThinking, stripThinking } from '../web/src/think.js';
import {
  getThinkingPref,
  isReasoningModel,
  MAX_STORED_THINKING,
  setThinkingPref,
  THINKING_PREF_KEY,
  thinkingRequestOptions,
  truncateThinking,
} from '../web/src/reasoning.js';
import { expandVisibleStart, initialVisibleStart, RECENT_RENDER_COUNT, UNFOLD_STEP } from '../web/src/store.js';

function memoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
  };
}

test('思考拆分：完整的 <think> 段落进入思考过程，回答独立', () => {
  const p = splitThinking('<think>\n先想想\n</think>\n\n答案是 42');
  assert.equal(p.thinking, '先想想');
  assert.equal(p.answer, '答案是 42');
  assert.equal(p.thinkingOpen, false);
});

test('思考拆分：只有结束标签（开始标签在模板中）时，之前的内容是思考过程', () => {
  const p = splitThinking('思考内容……\n</think>\n最终答案');
  assert.equal(p.thinking, '思考内容……');
  assert.equal(p.answer, '最终答案');
  assert.equal(p.thinkingOpen, false);
});

test('思考拆分：流式输出中未闭合的思考仍在进行，回答为空', () => {
  const p = splitThinking('<think>正在思考……');
  assert.equal(p.thinking, '正在思考……');
  assert.equal(p.answer, '');
  assert.equal(p.thinkingOpen, true);
});

test('思考拆分：普通回答没有思考过程', () => {
  const p = splitThinking('普通回答');
  assert.deepEqual(p, { thinking: '', answer: '普通回答', thinkingOpen: false });
});

test('思考拆分：多段思考合并，回答与 stripThinking 完全一致', () => {
  const samples = [
    '<think>a</think>X<think>b</think>Y',
    '<think>思考 </think>回答开始，',
    '<think>正在思考……',
    '思考内容……\n</think>\n最终答案',
    '<think>\n先想想\n</think>\n\n答案是 42',
    '普通回答，没有标签',
    '',
    '<think>x</think>',
    '<THINK>大写</THINK>回答',
  ];
  for (const raw of samples) {
    assert.equal(splitThinking(raw).answer, stripThinking(raw), `回答不一致：${JSON.stringify(raw)}`);
  }
  assert.equal(splitThinking('<think>a</think>X<think>b</think>Y').thinking, 'a\n\nb');
});

test('思考开关：默认开启；偏好保存在本浏览器中', () => {
  const storage = memoryStorage();
  assert.equal(getThinkingPref(storage), true);
  setThinkingPref(false, storage);
  assert.equal(storage.getItem(THINKING_PREF_KEY), 'off');
  assert.equal(getThinkingPref(storage), false);
  setThinkingPref(true, storage);
  assert.equal(getThinkingPref(storage), true);
});

test('推理模型判断：读取配置中的 reasoning 字段；“新建AI”沿用来源模型', () => {
  const models = [
    { id: 'Qwen3-4B-q4f16_1-MLC', reasoning: true },
    { id: 'Qwen2.5-3B-Instruct-q4f16_1-MLC' },
  ];
  const localEntries = [
    { id: 'imp-1', originId: 'Qwen3-4B-q4f16_1-MLC', baseLabel: 'Qwen3 4B' },
    { id: 'imp-2', originId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', baseLabel: 'Qwen2.5 3B（非推理）' },
    { id: 'imp-3', originId: 'unknown-model', baseLabel: 'Qwen3 0.6B（推理·小）' },
  ];
  const lists = { models, localEntries };
  assert.equal(isReasoningModel('Qwen3-4B-q4f16_1-MLC', lists), true);
  assert.equal(isReasoningModel('Qwen2.5-3B-Instruct-q4f16_1-MLC', lists), false);
  assert.equal(isReasoningModel('imp-1', lists), true);
  assert.equal(isReasoningModel('imp-2', lists), false);
  assert.equal(isReasoningModel('imp-3', lists), true);
  assert.equal(isReasoningModel('no-such-model', lists), false);
  assert.equal(isReasoningModel('', lists), false);
});

test('请求参数：只有推理模型且关闭思考时才传 enable_thinking: false', () => {
  assert.deepEqual(thinkingRequestOptions({ supportsThinking: true, enabled: false }), {
    extra_body: { enable_thinking: false },
  });
  assert.deepEqual(thinkingRequestOptions({ supportsThinking: true, enabled: true }), {});
  assert.deepEqual(thinkingRequestOptions({ supportsThinking: false, enabled: false }), {});
});

test('思考过程保存：超过上限时截断并注明', () => {
  assert.equal(truncateThinking('短'), '短');
  const long = 'x'.repeat(MAX_STORED_THINKING + 10);
  const cut = truncateThinking(long);
  assert.ok(cut.startsWith('x'.repeat(MAX_STORED_THINKING)));
  assert.match(cut, /较长/);
  assert.ok(cut.length < long.length + 60);
});

test('历史折叠：只渲染最近的消息，较早的消息可分批展开', () => {
  assert.equal(initialVisibleStart(5), 0);
  assert.equal(initialVisibleStart(RECENT_RENDER_COUNT), 0);
  assert.equal(initialVisibleStart(100), 100 - RECENT_RENDER_COUNT);
  assert.equal(expandVisibleStart(80, UNFOLD_STEP), 60);
  assert.equal(expandVisibleStart(10, UNFOLD_STEP), 0);
  assert.equal(expandVisibleStart(80, Infinity), 0);
});
