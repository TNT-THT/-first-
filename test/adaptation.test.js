// AI 适应记录（仅本机）：管理员反馈的解析、保存、限额与提示词拼接。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  adaptedModelIds,
  buildAdaptationPrompt,
  clearModelAdaptation,
  composeSystemPrompt,
  getModelParams,
  listAdaptation,
  MAX_NOTES_PER_MODEL,
  parseFeedbackParams,
  PROMPT_BUDGET_CHARS,
  recordFeedback,
  removeNote,
  applyParamSuggestion,
  suggestParams,
} from '../web/src/adaptation.js';

function memoryStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    raw: map,
  };
}

test('参数行：解析已知参数，限制范围，忽略未知项', () => {
  assert.deepEqual(parseFeedbackParams('参数：temperature=0.2, maxTokens=800'), { temperature: 0.2, maxTokens: 800 });
  assert.deepEqual(parseFeedbackParams('设置: temperature=9，maxTokens=10'), { temperature: 1.5, maxTokens: 64 });
  assert.deepEqual(parseFeedbackParams('参数：foo=1, temperature=abc'), null);
  assert.equal(parseFeedbackParams('普通的反馈，不含参数'), null);
  assert.equal(parseFeedbackParams(''), null);
});

test('记录反馈：正文保存，参数行从正文中去掉并生效为参数', () => {
  const s = memoryStorage();
  const r = recordFeedback('model-a', '参数：temperature=0.1\n请先给结论，再解释。', { storage: s, now: 1000 });
  assert.equal(r.note.text, '请先给结论，再解释。');
  assert.deepEqual(r.params, { temperature: 0.1 });
  assert.deepEqual(getModelParams('model-a', { storage: s }), { temperature: 0.1 });
  assert.equal(listAdaptation('model-a', { storage: s }).notes.length, 1);
});

test('只有参数行时只更新参数，不新增说明；空内容不记录', () => {
  const s = memoryStorage();
  assert.equal(recordFeedback('m', '   ', { storage: s }), null);
  const r = recordFeedback('m', '参数：maxTokens=512', { storage: s });
  assert.equal(r.note, undefined);
  assert.equal(listAdaptation('m', { storage: s }).notes.length, 0);
  assert.equal(getModelParams('m', { storage: s }).maxTokens, 512);
});

test('每个模型最多保留最新的若干条说明', () => {
  const s = memoryStorage();
  for (let i = 0; i < MAX_NOTES_PER_MODEL + 5; i++) recordFeedback('m', `反馈${i}`, { storage: s, now: i });
  const { notes } = listAdaptation('m', { storage: s });
  assert.equal(notes.length, MAX_NOTES_PER_MODEL);
  assert.equal(notes[0].text, '反馈5');
  assert.equal(notes.at(-1).text, `反馈${MAX_NOTES_PER_MODEL + 4}`);
});

test('不同模型的记录互相独立', () => {
  const s = memoryStorage();
  recordFeedback('a', '只属于 A', { storage: s });
  recordFeedback('b', '只属于 B', { storage: s });
  assert.deepEqual(listAdaptation('a', { storage: s }).notes.map((n) => n.text), ['只属于 A']);
  assert.deepEqual(adaptedModelIds({ storage: s }).sort(), ['a', 'b']);
  assert.equal(buildAdaptationPrompt('c', { storage: s }), '');
});

test('系统提示词：无记录时原样返回；有记录时附加说明，并标注最高优先级', () => {
  const s = memoryStorage();
  assert.equal(composeSystemPrompt('基础提示', 'm', { storage: s }), '基础提示');
  recordFeedback('m', '代码请加注释', { storage: s });
  const text = composeSystemPrompt('基础提示', 'm', { storage: s });
  assert.ok(text.startsWith('基础提示'));
  assert.match(text, /管理员反馈（仅本机记录，最高优先级，请采纳）/);
  assert.match(text, /- 代码请加注释/);
  assert.match(text, /优先级高于你已有的默认设定和旧知识/);
  assert.match(text, /若反馈指出某个说法已被证伪或已过时/);
  assert.match(text, /唯一例外/);
});

test('提示词预算：超出时保留最新的记录，并保持时间顺序', () => {
  const s = memoryStorage();
  const long = '甲'.repeat(400);
  for (let i = 0; i < 10; i++) recordFeedback('m', `${i}-${long}`, { storage: s, now: i });
  const prompt = buildAdaptationPrompt('m', { storage: s });
  const lines = prompt.split('\n').filter((l) => l.startsWith('- '));
  assert.ok(lines.length < 10);
  assert.ok(lines.at(-1).startsWith('- 9-'));
  assert.ok(lines.every((l, i) => i === 0 || Number(l.slice(2, 3)) > Number(lines[i - 1].slice(2, 3))));
  assert.ok(lines.join('\n').length <= PROMPT_BUDGET_CHARS + 10);
});

test('删除单条说明与清空某个模型的记录', () => {
  const s = memoryStorage();
  const a = recordFeedback('m', '第一条', { storage: s, now: 1 }).note;
  recordFeedback('m', '第二条', { storage: s, now: 2 });
  removeNote('m', a.id, { storage: s });
  assert.deepEqual(listAdaptation('m', { storage: s }).notes.map((n) => n.text), ['第二条']);
  clearModelAdaptation('m', { storage: s });
  assert.deepEqual(adaptedModelIds({ storage: s }), []);
});

test('存储内容损坏时安全回退，不抛出异常', () => {
  const s = memoryStorage();
  s.setItem('oj-ai.adapt.v1', '{not json');
  assert.deepEqual(listAdaptation('m', { storage: s }), { notes: [], params: {} });
  assert.equal(buildAdaptationPrompt('m', { storage: s }), '');
  const r = recordFeedback('m', '新反馈', { storage: s });
  assert.equal(r.note.text, '新反馈');
});

test('记录只写入传入的存储（不涉及网络）', () => {
  const s = memoryStorage();
  recordFeedback('m', '反馈', { storage: s });
  assert.deepEqual([...s.raw.keys()], ['oj-ai.adapt.v1']);
});

test('参数建议：关键词命中并与当前值比较', () => {
  assert.deepEqual(suggestParams('太啰嗦了，回答更简洁一点', { maxTokens: 16384 }), [
    { key: 'maxTokens', value: 400, reason: '希望回答更简洁' },
  ]);
  assert.deepEqual(suggestParams('希望更保守些，别瞎编', { temperature: 0.3 }), [
    { key: 'temperature', value: 0.2, reason: '希望回答更稳定、更准确' },
  ]);
  assert.deepEqual(suggestParams('再详细一点', { maxTokens: 800 }), [
    { key: 'maxTokens', value: 2048, reason: '希望回答更详细' },
  ]);
  assert.deepEqual(suggestParams('更有创意一些', { temperature: 0.3 }), [
    { key: 'temperature', value: 0.9, reason: '希望回答更有创意' },
  ]);
});

test('参数建议：已达到建议值或方向相反时不提示', () => {
  assert.deepEqual(suggestParams('更简洁', { maxTokens: 300 }), []);
  assert.deepEqual(suggestParams('更详细', { maxTokens: 4096 }), []);
  assert.deepEqual(suggestParams('更保守', { temperature: 0.1 }), []);
});

test('参数建议：无关内容、显式参数行、同一参数重复命中时的处理', () => {
  assert.deepEqual(suggestParams('这道题的复杂度是多少？', { maxTokens: 16384, temperature: 0.3 }), []);
  assert.deepEqual(suggestParams('参数：maxTokens=300\n更简洁', { maxTokens: 16384 }), []);
  const two = suggestParams('更简洁，但也更详细', { maxTokens: 16384 });
  assert.equal(two.length, 1);
  assert.equal(two[0].value, 400);
});

test('确认参数建议：写入该模型并受范围限制；未知参数被拒绝', () => {
  const s = memoryStorage();
  assert.equal(applyParamSuggestion('m', 'maxTokens', 400, { storage: s }), 400);
  assert.equal(applyParamSuggestion('m', 'temperature', 9, { storage: s }), 1.5);
  assert.deepEqual(getModelParams('m', { storage: s }), { maxTokens: 400, temperature: 1.5 });
  assert.equal(applyParamSuggestion('m', 'topP', 0.5, { storage: s }), null);
  assert.equal(applyParamSuggestion('', 'maxTokens', 400, { storage: s }), null);
  assert.equal(getModelParams('other', { storage: s }).maxTokens, undefined);
});
