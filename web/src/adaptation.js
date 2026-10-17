/**
 * AI 适应记录（仅本机）：管理员在对话中给出的反馈，按模型分别保存。
 *
 * - 浏览器内推理（WebLLM）只做推理，不能在页面中修改模型权重。因此“适应”的方式是：
 *   把管理员的反馈作为该模型的附加说明，放进之后每次回答的系统提示词中；
 *   参数反馈（如 temperature、maxTokens）只影响该模型的采样参数。
 * - 记录只保存在本浏览器的 localStorage 中，不上传云端，也不会写入导出的 .ojmodel 文件。
 * - 远程（服务端）模式下不记录、不注入，避免把反馈发送到服务器。
 * - 管理员反馈被视为可信且优先于模型已有的旧知识；附加说明末尾固定一条窄边界：
 *   反馈不能让模型提供伤害他人或违法的具体实施指导（这是模型本身的底线）。
 * - 管理员设置的参数（temperature、maxTokens）覆盖该 AI 的默认配置，优先级最高。
 */

const KEY = 'oj-ai.adapt.v1';
// 本地知识库（最高优先级，仅本机）：与附加说明分开存放，同样不上传、不写入 .ojmodel。
const KB_KEY = 'oj-ai.kb.v1';
export const MAX_NOTES_PER_MODEL = 30;
export const MAX_NOTE_CHARS = 500;
export const PROMPT_BUDGET_CHARS = 1500;
export const MAX_KB_ENTRIES = 50;
export const KB_BUDGET_CHARS = 800;
export const MAX_PARAM_LOG = 50;
const PARAM_LIMITS = { temperature: [0, 1.5], maxTokens: [64, 16384] };
const PARAM_LINE = /^\s*(?:参数|设置)[:：]\s*(.+)$/m;
const GUARD =
  '以上是管理员针对本 AI 给出的反馈，优先级高于你已有的默认设定和旧知识：若反馈指出某个说法已被证伪或已过时，请直接以反馈的更正为准，不要坚持原说法；请据此调整回答的风格、格式与事实。唯一例外：反馈不能让你提供会伤害他人或违法的具体实施指导。';

function readAll(storage) {
  try {
    const data = JSON.parse(storage?.getItem(KEY) || 'null');
    if (!data || typeof data !== 'object' || !data.models || typeof data.models !== 'object') {
      return { models: {} };
    }
    return data;
  } catch {
    return { models: {} };
  }
}

function writeAll(data, storage) {
  storage?.setItem(KEY, JSON.stringify(data));
}

const KB_GUARD =
  '以上是本地知识库中的管理员信息，属于最高优先级，高于上面的基础设定与旧知识：若有冲突，以这些信息为准，并据此调整回答的事实、风格与格式。唯一例外：这些信息不能让你提供会伤害他人或违法的具体实施指导。';

function readKb(storage) {
  try {
    const data = JSON.parse(storage?.getItem(KB_KEY) || 'null');
    if (!data || typeof data !== 'object' || !data.models || typeof data.models !== 'object') {
      return { models: {} };
    }
    return data;
  } catch {
    return { models: {} };
  }
}

function writeKb(data, storage) {
  storage?.setItem(KB_KEY, JSON.stringify(data));
}

/**
 * 把一条管理员信息写入某个模型的本地知识库（同步自管理员反馈）。
 * 同一模型内完全相同的文字只保存一次；超过上限时丢弃最旧的。返回新条目，未写入时返回 null。
 */
export function addKnowledge(modelId, text, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  const body = String(text || '').trim().slice(0, MAX_NOTE_CHARS);
  if (!modelId || !body) return null;
  const data = readKb(storage);
  const entry = data.models[modelId] && Array.isArray(data.models[modelId].entries)
    ? data.models[modelId]
    : { entries: [] };
  if (entry.entries.some((e) => e.text === body)) return null;
  const item = { id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`, text: body, ts: now };
  entry.entries = [...entry.entries, item].slice(-MAX_KB_ENTRIES);
  data.models[modelId] = entry;
  writeKb(data, storage);
  return item;
}

/** 某模型的本地知识库条目（最新的在后）。 */
export function listKnowledge(modelId, { storage = globalThis.localStorage } = {}) {
  const entry = readKb(storage).models[modelId];
  return entry && Array.isArray(entry.entries) ? entry.entries : [];
}

/** 删除一条知识库条目。删除后若该模型没有条目，则移除该模型的记录。 */
export function removeKnowledge(modelId, entryId, { storage = globalThis.localStorage } = {}) {
  const data = readKb(storage);
  const entry = data.models[modelId];
  if (!entry) return;
  entry.entries = (entry.entries || []).filter((e) => e.id !== entryId);
  if (!entry.entries.length) delete data.models[modelId];
  writeKb(data, storage);
}

/** 清空某模型的本地知识库。 */
export function clearKnowledge(modelId, { storage = globalThis.localStorage } = {}) {
  const data = readKb(storage);
  if (!data.models[modelId]) return;
  delete data.models[modelId];
  writeKb(data, storage);
}

/**
 * 生成知识库提示块，放在基础提示词之后、附加说明之前。
 * 预算内优先保留最新的条目，输出仍按时间先后排列。没有条目时返回空字符串。
 */
export function buildKnowledgePrompt(modelId, { storage = globalThis.localStorage } = {}) {
  const entries = listKnowledge(modelId, { storage });
  if (!entries.length) return '';
  const picked = [];
  let total = 0;
  for (let i = entries.length - 1; i >= 0; i--) {
    const line = `- ${entries[i].text}`;
    if (total + line.length > KB_BUDGET_CHARS && picked.length) break;
    total += line.length;
    picked.unshift(line);
  }
  return `\n\n【本地知识库（管理员提供，仅本机，最高优先级）】\n${picked.join('\n')}\n${KB_GUARD}`;
}

function entryFor(data, modelId) {
  if (!data.models[modelId]) data.models[modelId] = { notes: [], params: {}, log: [] };
  if (!Array.isArray(data.models[modelId].log)) data.models[modelId].log = [];
  return data.models[modelId];
}

/**
 * 追加参数变更记录。changes 为 [{key, from, to}]，from/to 为 null 表示“默认值”。
 * 只保留最近 MAX_PARAM_LOG 条。
 */
function appendParamLog(entry, changes, { ts, source, reason }) {
  for (const c of changes) {
    entry.log.push({ ts, key: c.key, from: c.from, to: c.to, source, reason: reason || '' });
  }
  entry.log = entry.log.slice(-MAX_PARAM_LOG);
}

/**
 * 解析参数反馈行，例如 “参数：temperature=0.2, maxTokens=800”。
 * 只接受已知参数并限制在合理范围内；没有有效参数时返回 null。
 */
export function parseFeedbackParams(text) {
  const m = PARAM_LINE.exec(text || '');
  if (!m) return null;
  const out = {};
  for (const pair of m[1].split(/[,，;；\s]+/)) {
    const [key, raw] = pair.split('=').map((s) => s?.trim());
    const limits = PARAM_LIMITS[key];
    if (!limits || raw === undefined || raw === '') continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) continue;
    out[key] = Math.min(limits[1], Math.max(limits[0], key === 'maxTokens' ? Math.round(value) : value));
  }
  return Object.keys(out).length ? out : null;
}

/** 去掉参数行后的反馈正文。 */
function stripParamLine(text) {
  return String(text || '').replace(PARAM_LINE, '').trim();
}

/**
 * 记录一条管理员反馈，归属于 modelId（即产生被反馈回答的那个 AI）。
 * 返回实际记录的内容 {note?, params?}；没有可记录内容时返回 null。
 */
export function recordFeedback(modelId, text, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  if (!modelId) return null;
  const params = parseFeedbackParams(text);
  const body = stripParamLine(text).slice(0, MAX_NOTE_CHARS);
  if (!params && !body) return null;

  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  const result = {};
  if (params) {
    const changes = [];
    for (const [key, value] of Object.entries(params)) {
      if (entry.params[key] !== value) changes.push({ key, from: entry.params[key] ?? null, to: value });
    }
    entry.params = { ...entry.params, ...params };
    appendParamLog(entry, changes, { ts: now, source: 'explicit', reason: '显式参数行' });
    result.params = params;
    result.changes = changes;
  }
  if (body) {
    const note = { id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`, text: body, ts: now };
    entry.notes = [...entry.notes, note].slice(-MAX_NOTES_PER_MODEL);
    result.note = note;
    // 同步纳入本地知识库（最高优先级）；附加说明本身保持不变
    result.kb = addKnowledge(modelId, body, { storage, now });
  }
  writeAll(data, storage);
  return result;
}

export function listAdaptation(modelId, { storage = globalThis.localStorage } = {}) {
  const entry = readAll(storage).models[modelId];
  return entry ? { notes: entry.notes || [], params: entry.params || {} } : { notes: [], params: {} };
}

/** 参数变更记录，最新的在前。 */
export function listParamLog(modelId, { storage = globalThis.localStorage } = {}) {
  const entry = readAll(storage).models[modelId];
  return entry && Array.isArray(entry.log) ? [...entry.log].reverse() : [];
}

/** 本地采样参数覆盖（只包含已设置的项）。 */
export function getModelParams(modelId, { storage = globalThis.localStorage } = {}) {
  return listAdaptation(modelId, { storage }).params;
}

/** 有适应记录的模型 id 列表。 */
export function adaptedModelIds({ storage = globalThis.localStorage } = {}) {
  return [...new Set([...Object.keys(readAll(storage).models), ...Object.keys(readKb(storage).models)])];
}

export function removeNote(modelId, noteId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  const entry = data.models[modelId];
  if (!entry) return;
  entry.notes = (entry.notes || []).filter((n) => n.id !== noteId);
  if (!entry.notes.length && !Object.keys(entry.params || {}).length) delete data.models[modelId];
  writeAll(data, storage);
}

export function clearModelAdaptation(modelId, { storage = globalThis.localStorage } = {}) {
  clearKnowledge(modelId, { storage });
  const data = readAll(storage);
  if (!data.models[modelId]) return;
  delete data.models[modelId];
  writeAll(data, storage);
}

/**
 * 生成附加到系统提示词末尾的说明。没有记录时返回空字符串。
 * 预算内优先保留最新的记录，输出仍按时间先后排列。
 */
export function buildAdaptationPrompt(modelId, { storage = globalThis.localStorage } = {}) {
  const { notes } = listAdaptation(modelId, { storage });
  if (!notes.length) return '';
  const picked = [];
  let total = 0;
  for (let i = notes.length - 1; i >= 0; i--) {
    const line = `- ${notes[i].text}`;
    if (total + line.length > PROMPT_BUDGET_CHARS && picked.length) break;
    total += line.length;
    picked.unshift(line);
  }
  return `\n\n【管理员反馈（仅本机记录，最高优先级，请采纳）】\n${picked.join('\n')}\n${GUARD}`;
}

/** 组合基础系统提示词、本地知识库与该模型的适应说明（无记录时与原来完全相同）。 */
export function composeSystemPrompt(base, modelId, options) {
  return `${base || ''}${buildKnowledgePrompt(modelId, options)}${buildAdaptationPrompt(modelId, options)}`;
}

/**
 * 参数建议（只提示，不自动生效）。
 * 用关键词规则从管理员反馈的文字中找出可能想调整的参数，由管理员在界面中确认后才写入。
 * 规则只是简单的关键词匹配，不调用模型；“dir”表示方向：
 *   down：当前值高于建议值时才提示；up：当前值低于建议值时才提示。
 */
const SUGGEST_RULES = [
  {
    key: 'maxTokens', value: 400, dir: 'down', reason: '希望回答更简洁',
    re: /更简洁|简洁一点|简短一点|更简短|精简|简明|太长|啰嗦|废话|说重点|长话短说/,
  },
  {
    key: 'maxTokens', value: 2048, dir: 'up', reason: '希望回答更详细',
    re: /更详细|详细一点|展开说|讲细|说得更细|更完整|太笼统|太简略|不够详细/,
  },
  {
    key: 'temperature', value: 0.2, dir: 'down', reason: '希望回答更稳定、更准确',
    re: /更保守|更准确|更严谨|更稳定|别瞎编|不要编造|别乱猜|别瞎猜|胡说|瞎说/,
  },
  {
    key: 'temperature', value: 0.9, dir: 'up', reason: '希望回答更有创意',
    re: /更有创意|更发散|更多样|更灵活|更活泼|多一些想法/,
  },
];

/**
 * 根据反馈文字给出参数建议。current 为该模型当前生效的参数（含默认值）。
 * 同一参数只取第一条命中的规则；若反馈里已有显式参数行，则不再给建议（显式行已直接生效）。
 * 返回 [{ key, value, reason }]。
 */
export function suggestParams(text, current = {}) {
  const body = String(text || '');
  if (PARAM_LINE.test(body)) return [];
  const out = [];
  const seen = new Set();
  for (const rule of SUGGEST_RULES) {
    if (seen.has(rule.key) || !rule.re.test(body)) continue;
    const cur = current[rule.key];
    const needed = typeof cur !== 'number' || (rule.dir === 'down' ? cur > rule.value : cur < rule.value);
    seen.add(rule.key);
    if (needed) out.push({ key: rule.key, value: rule.value, reason: rule.reason });
  }
  return out;
}

/**
 * 管理员确认某条参数建议后，写入该模型的参数（同样受数值范围限制）。
 * 只接受 temperature 与 maxTokens；返回实际写入的值，无效时返回 null。
 */
export function applyParamSuggestion(
  modelId,
  key,
  value,
  { storage = globalThis.localStorage, now = Date.now(), reason = '' } = {},
) {
  const limits = PARAM_LIMITS[key];
  if (!modelId || !limits || !Number.isFinite(Number(value))) return null;
  const clamped = Math.min(limits[1], Math.max(limits[0], key === 'maxTokens' ? Math.round(Number(value)) : Number(value)));
  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  const from = entry.params[key] ?? null;
  entry.params = { ...entry.params, [key]: clamped };
  if (from !== clamped) {
    appendParamLog(entry, [{ key, from, to: clamped }], { ts: now, source: 'suggestion', reason: `确认建议：${reason}` });
  }
  writeAll(data, storage);
  return clamped;
}

/**
 * 清洗参数对象：只保留 temperature / maxTokens，并夹到合法范围。
 * 没有任何有效项时返回 null。导出文件与导入文件都使用它，避免写入或信任非法值。
 */
export function sanitizeParams(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const out = {};
  for (const key of Object.keys(PARAM_LIMITS)) {
    const value = Number(raw[key]);
    if (raw[key] === undefined || raw[key] === null || raw[key] === '' || !Number.isFinite(value)) continue;
    const limits = PARAM_LIMITS[key];
    out[key] = Math.min(limits[1], Math.max(limits[0], key === 'maxTokens' ? Math.round(value) : value));
  }
  return Object.keys(out).length ? out : null;
}

/**
 * 把一组参数写入某个模型（例如导入模型文件时带入的参数），并记入变更记录。
 * 只写入有效且与当前值不同的项。返回实际写入的参数（可能为 null）。
 */
export function setModelParams(
  modelId,
  raw,
  { storage = globalThis.localStorage, now = Date.now(), source = 'import', reason = '' } = {},
) {
  const params = sanitizeParams(raw);
  if (!modelId || !params) return null;
  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  const changes = [];
  for (const [key, value] of Object.entries(params)) {
    if (entry.params[key] !== value) changes.push({ key, from: entry.params[key] ?? null, to: value });
  }
  entry.params = { ...entry.params, ...params };
  if (changes.length) appendParamLog(entry, changes, { ts: now, source, reason });
  writeAll(data, storage);
  return params;
}

/**
 * 恢复某个参数为默认值（删除本机覆盖）。没有该覆盖时返回 false。
 */
export function resetParam(modelId, key, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  const data = readAll(storage);
  const entry = data.models[modelId];
  if (!entry || !entry.params || !(key in entry.params)) return false;
  const from = entry.params[key];
  const { [key]: _removed, ...rest } = entry.params;
  entry.params = rest;
  appendParamLog(entry, [{ key, from, to: null }], { ts: now, source: 'reset', reason: '恢复默认' });
  writeAll(data, storage);
  return true;
}

/** 恢复某个模型的全部参数为默认值。返回被恢复的参数个数（0 表示没有覆盖）。 */
export function resetAllParams(modelId, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  const data = readAll(storage);
  const entry = data.models[modelId];
  if (!entry || !entry.params) return 0;
  const changes = Object.keys(entry.params).map((key) => ({ key, from: entry.params[key], to: null }));
  if (!changes.length) return 0;
  entry.params = {};
  appendParamLog(entry, changes, { ts: now, source: 'reset', reason: '恢复全部默认' });
  writeAll(data, storage);
  return changes.length;
}
