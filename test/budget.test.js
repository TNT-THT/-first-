// 上下文预算：窗口解析、提示词估计、单次 max_tokens 计算（纯逻辑）
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ANSWER_MIN_TOKENS,
  FORCE_ANSWER_NOTE,
  fitHistory,
  forcedAnswerMessages,
  forcedPromptTokens,
  estimatePromptTokens,
  engineIsStale,
  estimateTokens,
  planGeneration,
  systemPromptFits,
  resolveWindows,
  thinkingMeter,
  WINDOW_LIMITS,
} from '../web/src/budget.js';
import { sanitizeParams, parseFeedbackParams } from '../web/src/adaptation.js';

const DEFAULTS = { contextWindow: 16384, thinkingWindow: 4096 };

test('默认窗口：上下文 16384，思考 4096；推理模型引擎窗口 = 两者之和', () => {
  const r = resolveWindows({ defaults: DEFAULTS, reserveThinking: true });
  assert.equal(r.contextWindow, 16384);
  assert.equal(r.thinkingWindow, 4096);
  assert.equal(r.engineContextSize, 20480);
});

test('非推理模型不占用思考窗口', () => {
  const r = resolveWindows({ defaults: DEFAULTS, reserveThinking: false });
  assert.equal(r.engineContextSize, 16384);
});

test('管理员参数覆盖默认值，并夹在允许范围内', () => {
  const r = resolveWindows({
    defaults: DEFAULTS,
    overrides: { contextWindow: 99999, thinkingWindow: 100 },
    reserveThinking: true,
  });
  assert.equal(r.contextWindow, WINDOW_LIMITS.contextWindow.max);
  assert.equal(r.thinkingWindow, WINDOW_LIMITS.thinkingWindow.min);
  const low = resolveWindows({ defaults: DEFAULTS, overrides: { contextWindow: 1 } });
  assert.equal(low.contextWindow, WINDOW_LIMITS.contextWindow.min);
});

test('最大值：上下文 20000，思考 8000', () => {
  const r = resolveWindows({
    defaults: DEFAULTS,
    overrides: { contextWindow: 20000, thinkingWindow: 8000 },
    reserveThinking: true,
  });
  assert.equal(r.engineContextSize, 28000);
});

test('估计 token：CJK 按约 1.5，其他字符按约 0.6，且偏保守', () => {
  assert.equal(estimateTokens(''), 0);
  assert.equal(estimateTokens('你好'), 3);
  assert.equal(estimateTokens('abcdefghij'), 6);
  assert.ok(estimateTokens('你'.repeat(100)) >= 100);
});

test('提示词估计包含每条消息的模板开销', () => {
  const one = estimatePromptTokens([{ role: 'user', content: 'hi' }]);
  const two = estimatePromptTokens([
    { role: 'user', content: 'hi' },
    { role: 'user', content: 'hi' },
  ]);
  assert.ok(two > one);
  assert.equal(estimatePromptTokens([]), 8);
});

test('max_tokens = 思考窗口 + 回答；提示词 + 思考 + 回答 不超过引擎窗口', () => {
  const plan = planGeneration({
    promptTokens: 2000,
    contextWindow: 16384,
    thinkingWindow: 4096,
    supportsThinking: true,
    thinkingOn: true,
    answerCap: 16384,
  });
  assert.equal(plan.overflow, false);
  assert.equal(plan.thinkingTokens, 4096);
  assert.equal(plan.answerTokens, 16384 - 2000);
  assert.equal(plan.maxTokens, 4096 + 14384);
  assert.ok(2000 + plan.maxTokens <= 16384 + 4096);
});

test('回答长度受最大输出长度限制', () => {
  const plan = planGeneration({
    promptTokens: 100,
    contextWindow: 16384,
    thinkingWindow: 4096,
    supportsThinking: true,
    thinkingOn: false,
    answerCap: 800,
  });
  assert.equal(plan.thinkingTokens, 0);
  assert.equal(plan.maxTokens, 800);
});

test('关闭思考或非推理模型时不预留思考空间', () => {
  const off = planGeneration({ promptTokens: 0, contextWindow: 16384, thinkingWindow: 4096, supportsThinking: true, thinkingOn: false, answerCap: 16384 });
  assert.equal(off.maxTokens, 16384);
  const plain = planGeneration({ promptTokens: 0, contextWindow: 16384, thinkingWindow: 4096, supportsThinking: false, thinkingOn: true, answerCap: 16384 });
  assert.equal(plain.maxTokens, 16384);
});

test('提示词占满窗口时报告溢出，而不是发送后被引擎拒绝', () => {
  const plan = planGeneration({
    promptTokens: 16384 - ANSWER_MIN_TOKENS + 1,
    contextWindow: 16384,
    thinkingWindow: 4096,
    supportsThinking: true,
    thinkingOn: true,
    answerCap: 16384,
  });
  assert.equal(plan.overflow, true);
  assert.equal(plan.maxTokens, 0);
});

test('参数清洗与解析接受 contextWindow / thinkingWindow', () => {
  assert.deepEqual(sanitizeParams({ contextWindow: 18000.6, thinkingWindow: 9000, bogus: 1 }), {
    contextWindow: 18001,
    thinkingWindow: 8000,
  });
  assert.deepEqual(parseFeedbackParams('参数：contextWindow=12000, thinkingWindow=2048'), {
    contextWindow: 12000,
    thinkingWindow: 2048,
  });
});

test('思考计量：只计 <think> 内的块，预算用完时返回 cut', () => {
  const m = thinkingMeter(3);
  assert.equal(m.step('<think>'), false);
  assert.equal(m.step('abc'), false);
  assert.equal(m.step('def'), true);
  assert.equal(m.cut, true);
  assert.equal(m.tokens, 3);
  assert.equal(m.thinking(), 'abcdef');
  assert.equal(m.step('more'), false, '已截断后不再计数');
});

test('思考计量：思考已结束或没有思考标记时不计数', () => {
  const closed = thinkingMeter(2);
  closed.step('<think>x</think>');
  closed.step('answer');
  closed.step('answer2');
  assert.equal(closed.cut, false);
  const plain = thinkingMeter(2);
  plain.step('直接回答');
  plain.step('继续');
  assert.equal(plain.cut, false);
});

test('没有预算时不计量', () => {
  const m = thinkingMeter(null);
  for (let i = 0; i < 10; i += 1) m.step('<think>x');
  assert.equal(m.cut, false);
});

test('历史超出时从最早的消息开始丢弃，保留系统提示与当前提问', () => {
  const msgs = [{ role: 'system', content: 's' }];
  for (let i = 0; i < 6; i += 1) msgs.push({ role: 'user', content: '字'.repeat(300) });
  const fitted = fitHistory(msgs, 700);
  assert.equal(fitted[0].role, 'system');
  assert.equal(fitted.at(-1), msgs.at(-1));
  assert.ok(fitted.length < msgs.length);
  assert.ok(estimatePromptTokens(fitted) <= 700 || fitted.length === 2);
  assert.equal(msgs.length, 7, '原数组不被修改');
});

test('强制输出：说明与已写出的思考附加到最后一条用户消息，且不修改原消息', () => {
  const msgs = [
    { role: 'system', content: 'sys' },
    { role: 'user', content: '问题' },
  ];
  const out = forcedAnswerMessages(msgs, '  思考内容  ');
  assert.equal(out.length, 2);
  assert.ok(out[1].content.startsWith('问题'));
  assert.ok(out[1].content.includes(FORCE_ANSWER_NOTE));
  assert.ok(out[1].content.includes('思考内容'));
  assert.equal(msgs[1].content, '问题');
});

test('强制输出的提示词估计包含已用完的思考预算', () => {
  const base = estimatePromptTokens([{ role: 'user', content: 'hi' }]);
  assert.ok(forcedPromptTokens([{ role: 'user', content: 'hi' }], 4096) >= base + 4096);
});

test('关闭思考时引擎窗口只用上下文窗口（推理模型也一样）', () => {
  // 推理模型开启思考：预留思考窗口
  assert.equal(resolveWindows({ defaults: DEFAULTS, reserveThinking: true }).engineContextSize, 20480);
  // 推理模型关闭思考：不预留，引擎窗口 = 16384
  assert.equal(resolveWindows({ defaults: DEFAULTS, reserveThinking: false }).engineContextSize, 16384);
});

test('管理员信息放不下时报错而非截断：systemPromptFits', () => {
  const small = { role: 'system', content: '短' };
  const big = { role: 'system', content: '长'.repeat(100000) };
  assert.equal(systemPromptFits(small, 16384), true);
  assert.equal(systemPromptFits(big, 16384), false);
  assert.equal(systemPromptFits(small, 16384, 16000), false);
});

test('engineIsStale：模型或窗口与已加载引擎不一致时才需要释放', () => {
  const same = { loadedId: 'm', loadedCtx: 16384, modelId: 'm', engineContextSize: 16384 };
  assert.equal(engineIsStale(same), false);
  assert.equal(engineIsStale({ ...same, engineContextSize: 4096 }), true);
  assert.equal(engineIsStale({ ...same, modelId: 'other' }), true);
  // 没有引擎时 loadedId 为 null，与任何模型都不一致（调用方另行判断是否有引擎）
  assert.equal(engineIsStale({ ...same, loadedId: null, loadedCtx: null }), true);
});
