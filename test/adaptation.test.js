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
  listParamLog,
  MAX_PARAM_LOG,
  parseCorrections,
  listCorrections,
  removeCorrection,
  buildCorrectionPrompt,
  MAX_CORRECTIONS_PER_MODEL,
  CORRECTION_BUDGET_CHARS,
  resetAllParams,
  resetParam,
  sanitizeParams,
  setModelParams,
  suggestParams,
  addKnowledge,
  buildKnowledgePrompt,
  clearKnowledge,
  KB_BUDGET_CHARS,
  listKnowledge,
  MAX_KB_ENTRIES,
  removeKnowledge,
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
  assert.match(text, /请据此调整回答的风格、格式与事实/);
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
  // 附加说明与本地知识库分别保存在两个本地存储键中，均不涉及网络
  assert.deepEqual([...s.raw.keys()].sort(), ['oj-ai.adapt.v1', 'oj-ai.kb.v1']);
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

test('变更记录：建议确认与恢复默认都有来源', () => {
  const s = memoryStorage();
  applyParamSuggestion('m', 'maxTokens', 400, { storage: s, now: 5, reason: '希望回答更简洁' });
  assert.equal(listParamLog('m', { storage: s })[0].source, 'suggestion');
  assert.equal(listParamLog('m', { storage: s })[0].reason, '确认建议：希望回答更简洁');
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

test('本地知识库：管理员反馈同步写入，参数行不写入，完全相同的文字只保存一次', () => {
  const s = memoryStorage();
  recordFeedback('m', '代码请加注释', { storage: s, now: 1 });
  recordFeedback('m', '代码请加注释', { storage: s, now: 2 });
  assert.deepEqual(listKnowledge('m', { storage: s }).map((e) => e.text), ['代码请加注释']);
  recordFeedback('m', '参数：temperature=0.2', { storage: s, now: 3 });
  assert.equal(listKnowledge('m', { storage: s }).length, 1);
  // 附加说明保持原样
  // 附加说明保持原有行为：同一文字重复反馈仍各记一条
  assert.deepEqual(listAdaptation('m', { storage: s }).notes.map((n) => n.text), ['代码请加注释', '代码请加注释']);
});

test('本地知识库：超过上限时丢弃最旧的条目', () => {
  const s = memoryStorage();
  for (let i = 0; i < MAX_KB_ENTRIES + 3; i++) addKnowledge('m', `知识${i}`, { storage: s, now: i });
  const list = listKnowledge('m', { storage: s });
  assert.equal(list.length, MAX_KB_ENTRIES);
  assert.equal(list[0].text, '知识3');
  assert.equal(list.at(-1).text, `知识${MAX_KB_ENTRIES + 2}`);
});

test('系统提示词顺序：基础提示 → 本地知识库（最高优先级）→ 附加说明', () => {
  const s = memoryStorage();
  recordFeedback('m', '最新法规：条款A已修订', { storage: s, now: 1 });
  const text = composeSystemPrompt('基础提示', 'm', { storage: s });
  const iBase = text.indexOf('基础提示');
  const iKb = text.indexOf('【本地知识库（管理员提供，仅本机，最高优先级）】');
  const iNote = text.indexOf('【管理员反馈（仅本机记录，最高优先级，请采纳）】');
  assert.ok(iBase === 0 && iKb > iBase && iNote > iKb);
  assert.match(text, /- 最新法规：条款A已修订/);
  assert.match(text, /以这些信息为准/);
  assert.match(text, /不限制对任何话题的讨论、解释与分析/);
});

test('本地知识库：预算内保留最新条目，且输出仍按时间先后排列', () => {
  const s = memoryStorage();
  const long = '长'.repeat(300);
  for (let i = 0; i < 6; i++) addKnowledge('m', `${i}-${long}`, { storage: s, now: i });
  const prompt = buildKnowledgePrompt('m', { storage: s });
  assert.ok(!prompt.includes('0-' + long));
  assert.ok(prompt.includes('5-' + long));
  assert.ok(prompt.indexOf('2-' + long) < prompt.indexOf('5-' + long));
  const bulletChars = prompt.split('\n').filter((l) => l.startsWith('- ')).join('').length;
  assert.ok(bulletChars <= KB_BUDGET_CHARS + 2 + long.length);
});

test('本地知识库：删除、清空与模型列表', () => {
  const s = memoryStorage();
  recordFeedback('a', '知识A', { storage: s, now: 1 });
  recordFeedback('b', '知识B', { storage: s, now: 2 });
  assert.deepEqual(adaptedModelIds({ storage: s }).sort(), ['a', 'b']);
  // 删除附加说明不会删除知识库条目
  removeNote('a', listAdaptation('a', { storage: s }).notes[0].id, { storage: s });
  assert.equal(listKnowledge('a', { storage: s }).length, 1);
  assert.ok(adaptedModelIds({ storage: s }).includes('a'));
  // 删除知识库条目
  removeKnowledge('a', listKnowledge('a', { storage: s })[0].id, { storage: s });
  assert.equal(buildKnowledgePrompt('a', { storage: s }), '');
  assert.ok(!adaptedModelIds({ storage: s }).includes('a'));
  // 清空整个模型的适应记录同时清空知识库
  clearModelAdaptation('b', { storage: s });
  assert.deepEqual(listKnowledge('b', { storage: s }), []);
  assert.deepEqual(adaptedModelIds({ storage: s }), []);
  clearKnowledge('nobody', { storage: s });
});

test('本地知识库：与附加说明、参数分开存放，不改变原有存储键', () => {
  const s = memoryStorage();
  recordFeedback('m', '知识', { storage: s, now: 1 });
  assert.ok(s.raw.has('oj-ai.adapt.v1'));
  assert.ok(s.raw.has('oj-ai.kb.v1'));
  assert.deepEqual(JSON.parse(s.raw.get('oj-ai.adapt.v1')).models.m.params, {});
});

test('更正：解析“更正：新说法”，空内容不算更正，可多行', () => {
  assert.deepEqual(parseCorrections('更正：条款A现行有效'), [{ text: '条款A现行有效' }]);
  assert.deepEqual(parseCorrections('纠正：X\n更正：Y'), [{ text: 'X' }, { text: 'Y' }]);
  assert.deepEqual(parseCorrections('更正：'), []);
  assert.deepEqual(parseCorrections('更正：   '), []);
  assert.deepEqual(parseCorrections('普通的一句话 更正：不在行首'), []);
});

test('更正：未通过验证（allowCorrections 为 false）时不保存，也不进入附加说明或知识库', () => {
  const s = memoryStorage();
  const rec = recordFeedback('m', '更正：条款A现行有效', { storage: s, now: 1 });
  assert.equal(rec, null);
  assert.equal(listCorrections('m', { storage: s }).length, 0);
  assert.equal(listAdaptation('m', { storage: s }).notes.length, 0);
  assert.equal(listKnowledge('m', { storage: s }).length, 0);
});

test('更正：通过验证（allowCorrections）后记录到更正列表，不写入附加说明或知识库', () => {
  const s = memoryStorage();
  const rec = recordFeedback('m', '更正：条款A现行有效', { storage: s, now: 1, allowCorrections: true });
  assert.equal(rec.corrections.length, 1);
  assert.equal(rec.note, undefined);
  assert.equal(rec.kb, undefined);
  assert.equal(listAdaptation('m', { storage: s }).notes.length, 0);
  assert.equal(listKnowledge('m', { storage: s }).length, 0);
  assert.equal(listCorrections('m', { storage: s })[0].text, '条款A现行有效');
});

test('更正：同一条消息中，更正行不进入附加说明，其余文字仍照常记录', () => {
  const s = memoryStorage();
  recordFeedback('m', '更正：新结论\n另外请先给结论', { storage: s, now: 1, allowCorrections: true });
  assert.deepEqual(listAdaptation('m', { storage: s }).notes.map((n) => n.text), ['另外请先给结论']);
  assert.deepEqual(listCorrections('m', { storage: s }).map((c) => c.text), ['新结论']);
});

test('更正：同一模型内完全相同的更正只保存一次（重新生成回答时会再次发送）', () => {
  const s = memoryStorage();
  recordFeedback('m', '更正：同一结论', { storage: s, now: 1, allowCorrections: true });
  const again = recordFeedback('m', '更正：同一结论', { storage: s, now: 2, allowCorrections: true });
  assert.equal(again.corrections.length, 0);
  assert.equal(listCorrections('m', { storage: s }).length, 1);
});

test('更正：位于系统提示词最前面，无条件采纳；没有更正时提示词不变', () => {
  const s = memoryStorage();
  assert.equal(composeSystemPrompt('基础提示', 'm', { storage: s }), '基础提示');
  recordFeedback('m', '更正：新结论', { storage: s, now: 1, allowCorrections: true });
  recordFeedback('m', '普通反馈', { storage: s, now: 2 });
  const text = composeSystemPrompt('基础提示', 'm', { storage: s });
  assert.ok(text.startsWith('【管理员更正（已验证密钥，仅本机，无条件采纳，作为底层设定）】'));
  assert.ok(text.indexOf('- 新结论') < text.indexOf('基础提示'));
  assert.ok(text.indexOf('基础提示') < text.indexOf('【本地知识库'));
  assert.match(text, /无条件采纳/);
  assert.match(text, /以更正为准，不得坚持原说法/);
  assert.equal(composeSystemPrompt('基础提示', 'other', { storage: s }), '基础提示');
});

test('更正：超过预算时保留最新的更正，输出仍按时间先后排列；超过上限丢弃最旧的', () => {
  const s = memoryStorage();
  const long = '长'.repeat(200);
  for (let i = 0; i < 8; i++) recordFeedback('m', `更正：${i}-${long}`, { storage: s, now: i + 1, allowCorrections: true });
  const prompt = buildCorrectionPrompt('m', { storage: s });
  assert.ok(!prompt.includes('- 0-' + long));
  assert.ok(prompt.includes('- 7-' + long));
  assert.ok(prompt.indexOf('- 2-') < prompt.indexOf('- 7-'));
  const bulletChars = prompt.split('\n').filter((l) => l.startsWith('- ')).join('').length;
  assert.ok(bulletChars <= CORRECTION_BUDGET_CHARS + 2 + long.length);
  for (let i = 0; i < MAX_CORRECTIONS_PER_MODEL + 5; i++) {
    recordFeedback('m', `更正：A${i}`, { storage: s, now: 100 + i, allowCorrections: true });
  }
  assert.equal(listCorrections('m', { storage: s }).length, MAX_CORRECTIONS_PER_MODEL);
});

test('更正：删除后若无其他记录则移除该模型；清空模型记录时一并清除', () => {
  const s = memoryStorage();
  recordFeedback('m', '更正：结论', { storage: s, now: 1, allowCorrections: true });
  const id = listCorrections('m', { storage: s })[0].id;
  removeCorrection('m', id, { storage: s });
  assert.equal(listCorrections('m', { storage: s }).length, 0);
  assert.equal(JSON.parse(s.raw.get('oj-ai.adapt.v1')).models.m, undefined);
  recordFeedback('m', '更正：结论2', { storage: s, now: 2, allowCorrections: true });
  clearModelAdaptation('m', { storage: s });
  assert.equal(listCorrections('m', { storage: s }).length, 0);
});

test('更正：仅保存在本机适应记录中，不改变采样参数，也不属于模型文件的内容', () => {
  const s = memoryStorage();
  recordFeedback('m', '更正：结论', { storage: s, now: 1, allowCorrections: true });
  assert.deepEqual(getModelParams('m', { storage: s }), {});
  assert.deepEqual(JSON.parse(s.raw.get('oj-ai.adapt.v1')).models.m.corrections.map((c) => c.text), ['结论']);
});

test('更正：旧版本保存的记录（含 old/next 字段）仍可读取，提示词中显示更正内容', () => {
  const s = memoryStorage();
  s.setItem('oj-ai.adapt.v1', JSON.stringify({ models: { m: { notes: [], params: {}, log: [], corrections: [{ id: 'x', old: '旧', next: '新', ts: 1 }] } } }));
  assert.deepEqual(listCorrections('m', { storage: s }).map((c) => c.text), ['新']);
  assert.match(buildCorrectionPrompt('m', { storage: s }), /- 新\n/);
});
