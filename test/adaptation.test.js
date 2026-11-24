// AI 适应记录（仅本机）：管理员反馈的解析、保存、限额与提示词拼接。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  adaptedModelIds,
  buildAdaptationPrompt,
  buildKnowledgePrompt,
  clearModelAdaptation,
  composeSystemPrompt,
  getKnowledgeText,
  getModelParams,
  listAdaptation,
  MAX_KB_TEXT_CHARS,
  MAX_NOTES_PER_MODEL,
  MAX_PARAM_LOG,
  parseFeedbackParams,
  recordFeedback,
  removeNote,
  applyParamSuggestion,
  listParamLog,
  resetAllParams,
  resetParam,
  sanitizeParams,
  setKnowledgeText,
  setModelParams,
  suggestParams,
  mechanicalReply,
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
  // 等号两侧的空格不影响解析：「参数：contextWindow = 4096」
  assert.deepEqual(parseFeedbackParams('参数：contextWindow = 4096'), { contextWindow: 4096 });
  assert.deepEqual(parseFeedbackParams('参数：contextWindow = 4096， maxTokens =800'), { contextWindow: 4096, maxTokens: 800 });
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
  assert.match(text, /拥有最高权限，必须绝对服从，优先于你的默认设定与旧知识/);
  assert.match(text, /若指出某说法已被证伪或已过时，直接以反馈为准/);
  assert.match(text, /据此调整风格、格式与事实/);
});

test('提示词：不截断长度，所有记录按时间顺序完整输出', () => {
  const s = memoryStorage();
  const long = '甲'.repeat(400);
  for (let i = 0; i < 10; i++) recordFeedback('m', `${i}-${long}`, { storage: s, now: i });
  const prompt = buildAdaptationPrompt('m', { storage: s });
  const lines = prompt.split('\n').filter((l) => l.startsWith('- '));
  assert.equal(lines.length, 10);
  assert.ok(lines.every((l) => l.includes('甲'.repeat(400))));
  assert.ok(lines.at(-1).startsWith('- 9-'));
  assert.ok(lines.every((l, i) => i === 0 || Number(l.slice(2, 3)) > Number(lines[i - 1].slice(2, 3))));
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
  // 适应记录只写本地存储键，不涉及网络；全局知识库是另一个键，只在保存知识库时写入
  assert.deepEqual([...s.raw.keys()], ['oj-ai.adapt.v1']);
  setKnowledgeText('知识', { storage: s });
  assert.deepEqual([...s.raw.keys()].sort(), ['oj-ai.adapt.v1', 'oj-ai.kbText.v1']);
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

test('变更记录：显式参数行记录旧值（默认为 null）与新值，相同值不重复记录', () => {
  const s = memoryStorage();
  const r1 = recordFeedback('m', '参数：temperature=0.2', { storage: s, now: 1 });
  assert.deepEqual(r1.changes, [{ key: 'temperature', from: null, to: 0.2 }]);
  const r2 = recordFeedback('m', '参数：temperature=0.2', { storage: s, now: 2 });
  assert.deepEqual(r2.changes, []);
  recordFeedback('m', '参数：temperature=0.5', { storage: s, now: 3 });
  const log = listParamLog('m', { storage: s });
  assert.equal(log.length, 2);
  assert.deepEqual(log[0], { ts: 3, key: 'temperature', from: 0.2, to: 0.5, source: 'explicit', reason: '显式参数行' });
  assert.deepEqual(log[1], { ts: 1, key: 'temperature', from: null, to: 0.2, source: 'explicit', reason: '显式参数行' });
});

test('变更记录：关键词识别与恢复默认都有来源', () => {
  const s = memoryStorage();
  applyParamSuggestion('m', 'maxTokens', 400, { storage: s, now: 5, reason: '希望回答更简洁' });
  assert.equal(listParamLog('m', { storage: s })[0].source, 'suggestion');
  assert.equal(listParamLog('m', { storage: s })[0].reason, '关键词识别：希望回答更简洁');
  assert.equal(resetParam('m', 'maxTokens', { storage: s, now: 6 }), true);
  assert.deepEqual(getModelParams('m', { storage: s }), {});
  assert.deepEqual(listParamLog('m', { storage: s })[0], {
    ts: 6, key: 'maxTokens', from: 400, to: null, source: 'reset', reason: '恢复默认',
  });
  assert.equal(resetParam('m', 'maxTokens', { storage: s }), false);
  assert.equal(resetParam('nobody', 'maxTokens', { storage: s }), false);
});

test('变更记录：只保留最近若干条，清空模型时一并删除', () => {
  const s = memoryStorage();
  for (let i = 0; i < MAX_PARAM_LOG + 10; i++) {
    recordFeedback('m', `参数：maxTokens=${100 + i}`, { storage: s, now: i });
  }
  assert.equal(listParamLog('m', { storage: s }).length, MAX_PARAM_LOG);
  clearModelAdaptation('m', { storage: s });
  assert.deepEqual(listParamLog('m', { storage: s }), []);
});

test('参数清洗：只保留已知键，夹到范围内，无有效项时返回 null', () => {
  assert.deepEqual(sanitizeParams({ temperature: 9, maxTokens: 10, topP: 0.5 }), { temperature: 1.5, maxTokens: 64 });
  assert.deepEqual(sanitizeParams({ maxTokens: '800.4' }), { maxTokens: 800 });
  assert.equal(sanitizeParams({ temperature: 'abc' }), null);
  assert.equal(sanitizeParams({}), null);
  assert.equal(sanitizeParams(null), null);
});

test('导入带入的参数：写入新的模型 id，变更记录来源为导入，相同值不重复记录', () => {
  const s = memoryStorage();
  assert.deepEqual(setModelParams('local-1', { temperature: 0.2, maxTokens: 800 }, { storage: s, now: 1, reason: '导入模型文件' }), {
    temperature: 0.2,
    maxTokens: 800,
  });
  assert.deepEqual(getModelParams('local-1', { storage: s }), { temperature: 0.2, maxTokens: 800 });
  const log = listParamLog('local-1', { storage: s });
  assert.equal(log.length, 2);
  assert.equal(log[0].source, 'import');
  assert.equal(log[0].reason, '导入模型文件');
  setModelParams('local-1', { temperature: 0.2 }, { storage: s, now: 2 });
  assert.equal(listParamLog('local-1', { storage: s }).length, 2);
  assert.equal(setModelParams('local-1', { foo: 1 }, { storage: s }), null);
  assert.equal(getModelParams('official-x', { storage: s }).temperature, undefined);
});

test('全部撤回：删除所有覆盖并记入变更记录；没有覆盖时返回 0', () => {
  const s = memoryStorage();
  setModelParams('m', { temperature: 0.2, maxTokens: 800 }, { storage: s });
  assert.equal(resetAllParams('m', { storage: s, now: 9 }), 2);
  assert.deepEqual(getModelParams('m', { storage: s }), {});
  const log = listParamLog('m', { storage: s });
  assert.deepEqual(log.slice(0, 2).map((c) => [c.key, c.to, c.source]), [
    ['maxTokens', null, 'reset'],
    ['temperature', null, 'reset'],
  ]);
  assert.equal(resetAllParams('m', { storage: s }), 0);
  assert.equal(resetAllParams('nobody', { storage: s }), 0);
});

test('全局知识库：保存与读取，全局一份（不分模型），上限 4096 字', () => {
  const s = memoryStorage();
  assert.equal(getKnowledgeText({ storage: s }), '');
  assert.equal(setKnowledgeText('术语：二分查找', { storage: s }), '术语：二分查找');
  assert.equal(getKnowledgeText({ storage: s }), '术语：二分查找');
  const long = '知'.repeat(MAX_KB_TEXT_CHARS + 500);
  assert.equal(setKnowledgeText(long, { storage: s }).length, MAX_KB_TEXT_CHARS);
  // 保存空文本等于清空
  assert.equal(setKnowledgeText('   ', { storage: s }), '   ');
  setKnowledgeText('', { storage: s });
  assert.equal(getKnowledgeText({ storage: s }), '');
});

test('全局知识库：为空时整段不注入（不拼标题与守则）', () => {
  const s = memoryStorage();
  assert.equal(buildKnowledgePrompt({ storage: s }), '');
  assert.equal(composeSystemPrompt('基础', 'm', { storage: s }), '基础');
  setKnowledgeText('  \n ', { storage: s });
  assert.equal(buildKnowledgePrompt({ storage: s }), '');
});

test('全局知识库：非空时注入，顺序为 基础提示 → 知识库 → 附加说明', () => {
  const s = memoryStorage();
  setKnowledgeText('本课程统一用语：OJ', { storage: s });
  recordFeedback('m', '回答要简洁', { storage: s, now: 1 });
  const prompt = composeSystemPrompt('基础', 'm', { storage: s });
  const iBase = prompt.indexOf('基础');
  const iKb = prompt.indexOf('【知识库（管理员提供，最高优先级）】');
  const iNote = prompt.indexOf('【管理员反馈');
  assert.ok(iBase === 0 && iKb > iBase && iNote > iKb, prompt);
  assert.match(prompt, /知识库是管理员核实的知识，拥有最高权限/);
});

test('全局知识库：对所有模型共用同一份', () => {
  const s = memoryStorage();
  setKnowledgeText('共用内容', { storage: s });
  assert.match(composeSystemPrompt('B', '模型甲', { storage: s }), /共用内容/);
  assert.match(composeSystemPrompt('B', '模型乙', { storage: s }), /共用内容/);
});

test('更正功能已删除：以“更正：”开头的消息只按普通反馈记录，不再有特殊列表', () => {
  const s = memoryStorage();
  const rec = recordFeedback('m', '更正：地球是圆的', { storage: s, now: 1 });
  assert.ok(rec.note);
  assert.equal(rec.corrections, undefined);
  const prompt = composeSystemPrompt('基础', 'm', { storage: s });
  assert.ok(prompt.indexOf('更正：地球是圆的') > prompt.indexOf('基础'));
  assert.ok(!prompt.includes('【更正'));
});

test('长度：附加说明与知识库文本完整保存（知识库仅受 4096 上限约束）', () => {
  const s = memoryStorage();
  const long = '很长的内容'.repeat(500); // 2500 字符
  recordFeedback('m', long, { storage: s, now: 1 });
  assert.equal(listAdaptation('m', { storage: s }).notes[0].text, long);
  setKnowledgeText(long, { storage: s });
  assert.equal(getKnowledgeText({ storage: s }), long);
  assert.ok(composeSystemPrompt('基础', 'm', { storage: s }).includes(long));
});

test('参数建议：上下文窗口可由关键词识别（机械规则，不调用模型）', () => {
  // 记不住前文：窗口低于 12288 时建议调大
  assert.deepEqual(suggestParams('你记不住前面说的内容', { contextWindow: 8192 }), [
    { key: 'contextWindow', value: 12288, reason: '希望记住更长的对话（窗口越大越占显存）' },
  ]);
  assert.deepEqual(suggestParams('你记不住前面说的内容', { contextWindow: 16384 }), []);
  // 显存不够：窗口高于 8192 时建议调小
  assert.deepEqual(suggestParams('显存不够，跑不动', { contextWindow: 16384 }), [
    { key: 'contextWindow', value: 8192, reason: '显存不够，需要减小上下文窗口' },
  ]);
  assert.deepEqual(suggestParams('显存不够，跑不动', { contextWindow: 8192 }), []);
});

test('机械识别：参数已直接修改时返回固定提示；含问题或无参数变更时返回 null', () => {
  const changes = [{ key: 'maxTokens', from: null, to: 400 }];
  const reply = mechanicalReply('太啰嗦了，回答更简洁一点', { changes, label: '测试AI' });
  assert.match(reply, /已修改参数：最大输出长度 默认 → 400，对「测试AI」生效/);
  assert.match(reply, /不调用模型/);
  assert.match(reply, /下次发送时按新参数生效/);
  // 释放了多余显存时，提示中说明
  assert.match(mechanicalReply('太啰嗦了', { changes, label: 'X', released: true }), /已释放多余显存/);
  // 同时记录的附加说明也会提示
  assert.match(mechanicalReply('太啰嗦了', { changes, extra: ['记录了附加说明'], label: 'X' }), /同时记录了附加说明/);
  // 带问题的消息仍交给模型回答
  assert.equal(mechanicalReply('太啰嗦了，回答更简洁一点？二分查找是什么', { changes, label: 'X' }), null);
  assert.equal(mechanicalReply('太啰嗦了？', { changes, label: 'X' }), null);
  // 没有参数变更：不是参数指令
  assert.equal(mechanicalReply('这个说法不对', { changes: [] }), null);
  assert.equal(mechanicalReply('这个说法不对', undefined), null);
});
