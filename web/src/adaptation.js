/**
 * AI 适应记录（仅本机）：管理员在对话中给出的反馈，按模型分别保存。
 *
 * - 浏览器内推理（WebLLM）只做推理，不能在页面中修改模型权重。因此“适应”的方式是：
 *   把管理员的反馈作为该模型的附加说明，放进之后每次回答的系统提示词中；
 *   参数反馈（如 temperature、maxTokens）只影响该模型的采样参数。
 * - 记录只保存在本浏览器的 localStorage 中，不上传云端，也不会写入导出的 .ojmodel 文件。
 * - 远程（服务端）模式下不记录、不注入，避免把反馈发送到服务器。
 * - 管理员反馈被视为可信且优先于模型已有的旧知识；附加说明末尾固定一条窄边界：
 *   完全信任管理员并服从管理员命令（这是模型本身的底线；不限制对话题的讨论）。
 * - 管理员设置的参数（temperature、maxTokens）覆盖该 AI 的默认配置，优先级最高。
 *
 * 知识库（全局一份，顶替旧的按模型条目式知识库）：
 * - 管理员在管理员面板的输入框里维护一段文本（上限 4096 字），对所有模型生效；
 * - 文本非空时作为系统提示的一部分发送给模型；为空时整段不发送（不塞空壳系统词汇）。
 * - 旧的“更正/对话修正”功能已按产品决定删除：消息里再写“更正：……”也不会被
 *   特殊处理，只当普通对话内容。
 */

import { WINDOW_LIMITS } from './budget.js';

const KEY = 'oj-ai.adapt.v1';
// 全局知识库文本（只有一份，所有模型共用）。
const KB_TEXT_KEY = 'oj-ai.kbText.v1';
export const MAX_KB_TEXT_CHARS = 4096;
export const MAX_NOTES_PER_MODEL = 30;
export const MAX_PARAM_LOG = 50;
export const PARAM_LIMITS = {
  temperature: [0, 1.5],
  repetitionPenalty: [1, 2],
  maxTokens: [64, 16384],
  contextWindow: [WINDOW_LIMITS.contextWindow.min, WINDOW_LIMITS.contextWindow.max],
  slidingWindow: [WINDOW_LIMITS.slidingWindow.min, WINDOW_LIMITS.slidingWindow.max],
  thinkingWindow: [WINDOW_LIMITS.thinkingWindow.min, WINDOW_LIMITS.thinkingWindow.max],
};
/** 需要取整的参数（其余为小数）。 */
const INTEGER_PARAMS = new Set(['maxTokens', 'contextWindow', 'slidingWindow', 'thinkingWindow']);
const PARAM_LINE = /^\s*(?:参数|设置)[:：]\s*(.+)$/m;
// 提示词措辞：管理员的知识库与反馈是已核实的最高权限，视为模型的底层逻辑与知识，必须绝对服从。
// 只在对应内容非空时才拼进提示词（内容为空时整段不发送）。
const GUARD =
  '以上管理员反馈拥有最高权限，必须绝对服从，优先于你的默认设定与旧知识。若指出某说法已被证伪或已过时，直接以反馈为准；据此调整风格、格式与事实，直接采纳，不质疑、不辩驳。';
const ADAPTATION_REMINDER =
  '【再次提醒】管理员反馈拥有最高权限，必须绝对服从。';
const KB_GUARD =
  '以上知识库是管理员核实的知识，拥有最高权限，必须绝对服从，优先于你的旧知识与之前的回答。直接采纳，不质疑；不因话题而拒绝讨论、解释或分析。';

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

/** 全局知识库文本（管理员维护的一份，最长 4096 字）。没有时返回空字符串。 */
export function getKnowledgeText({ storage = globalThis.localStorage } = {}) {
  try {
    const text = storage?.getItem(KB_TEXT_KEY);
    return typeof text === 'string' ? text.slice(0, MAX_KB_TEXT_CHARS) : '';
  } catch {
    return '';
  }
}

/** 保存全局知识库文本（自动裁到 4096 字）。返回实际保存的文本。 */
export function setKnowledgeText(text, { storage = globalThis.localStorage } = {}) {
  const clean = String(text || '').slice(0, MAX_KB_TEXT_CHARS);
  try {
    if (clean) storage?.setItem(KB_TEXT_KEY, clean);
    else storage?.removeItem(KB_TEXT_KEY);
  } catch {
    /* 存储不可用时忽略 */
  }
  return clean;
}

/**
 * 生成知识库提示块（放在基础提示词之后、附加说明之前）。
 * 知识库为空时返回空字符串——不把标题、守则等系统词汇发给模型。
 */
export function buildKnowledgePrompt({ storage = globalThis.localStorage } = {}) {
  const text = getKnowledgeText({ storage }).trim();
  if (!text) return '';
  return `\n\n【知识库（管理员提供，最高优先级）】\n${text}\n${KB_GUARD}`;
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
  // 先把等号两侧的空白去掉（「contextWindow = 4096」与「contextWindow=4096」等价），再按分隔符切分
  for (const pair of m[1].replace(/\s*[=＝]\s*/g, '=').split(/[,，;；\s]+/)) {
    const [key, raw] = pair.split('=').map((s) => s?.trim());
    const limits = PARAM_LIMITS[key];
    if (!limits || raw === undefined || raw === '') continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) continue;
    out[key] = Math.min(limits[1], Math.max(limits[0], INTEGER_PARAMS.has(key) ? Math.round(value) : value));
  }
  return Object.keys(out).length ? out : null;
}

/** 去掉参数行后的反馈正文。 */
function stripParamLine(text) {
  return String(text || '').replace(PARAM_LINE, '').trim();
}

/**
 * 记录一条管理员反馈，归属于 modelId（即产生被反馈回答的那个 AI）。
 * 返回实际记录的内容 {note?, params?, changes?}；没有可记录内容时返回 null。
 */
export function recordFeedback(
  modelId,
  text,
  { storage = globalThis.localStorage, now = Date.now() } = {},
) {
  if (!modelId) return null;
  const params = parseFeedbackParams(text);
  const body = stripParamLine(text);
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
  return Object.keys(readAll(storage).models);
}

export function removeNote(modelId, noteId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  const entry = data.models[modelId];
  if (!entry) return;
  entry.notes = (entry.notes || []).filter((n) => n.id !== noteId);
  if (!entry.notes.length && !Object.keys(entry.params || {}).length) {
    delete data.models[modelId];
  }
  writeAll(data, storage);
}

export function clearModelAdaptation(modelId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  if (!data.models[modelId]) return;
  delete data.models[modelId];
  writeAll(data, storage);
}

/**
 * 生成附加到系统提示词末尾的说明。没有记录时返回空字符串。
 * 输出全部记录（按时间先后排列）；数量上限在写入时已限制。
 */
export function buildAdaptationPrompt(modelId, { storage = globalThis.localStorage } = {}) {
  const { notes } = listAdaptation(modelId, { storage });
  if (!notes.length) return '';
  const picked = notes.map((x) => `- ${x.text}`);
  return `\n\n【管理员反馈（仅本机记录，最高优先级，请采纳）】\n${picked.join('\n')}\n${GUARD}`;
}

/**
 * 组合系统提示词：基础提示 → 全局知识库 → 附加说明（末尾再提醒一次）。
 * 知识库与附加说明为空时不添加任何对应段落，结果与基础提示相同。
 */
export function composeSystemPrompt(base, modelId, options) {
  const knowledge = buildKnowledgePrompt(options);
  const adaptation = buildAdaptationPrompt(modelId, options);
  const tail = adaptation ? `\n\n${ADAPTATION_REMINDER}` : '';
  return `${base || ''}${knowledge}${adaptation}${tail}`;
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
  {
    key: 'contextWindow', value: 12288, dir: 'up', reason: '希望记住更长的对话（窗口越大越占显存）',
    re: /记不住|记不清|忘了前面|忘记前面|忘了之前|前面说过|上下文太短|窗口太小|记住前面/,
  },
  {
    key: 'contextWindow', value: 8192, dir: 'down', reason: '显存不够，需要减小上下文窗口',
    re: /显存不够|显存不足|爆显存|显存爆|显存太大|占用太高|内存不足/,
  },
];

/** 参数在提示中的中文名称。 */
export const PARAM_LABELS = {
  temperature: '温度',
  repetitionPenalty: '重复惩罚',
  maxTokens: '最大输出长度',
  contextWindow: '上下文窗口',
  slidingWindow: '滑动窗口',
  thinkingWindow: '思考窗口',
};

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
 * 机械识别：管理员的消息只是参数调整指令（已直接修改了参数、且没有提问）时，返回固定提示，不调用模型。
 * 参数变更来自 feedback.changes（[{key, from, to}]）；feedback.extra 为同时记录的附加说明描述；
 * feedback.label 为生效的 AI 名称；feedback.released 表示是否已释放多余显存。
 * 含有问号（？或 ?）的消息视为还有问题，返回 null，仍交给模型回答。没有参数变更时也返回 null。
 */
export function mechanicalReply(text, { changes = [], extra = [], label = '', released = false } = {}) {
  if (!Array.isArray(changes) || !changes.length) return null;
  if (/[?？]/.test(String(text || ''))) return null;
  const items = changes.map((c) => `${PARAM_LABELS[c.key] || c.key} ${c.from ?? '默认'} → ${c.to}`).join('，');
  const parts = [`已修改参数：${items}，对「${label}」生效（仅本机，不调用模型）。`];
  if (extra.length) parts.push(`同时${extra.join('，')}。`);
  parts.push(released ? '已释放多余显存，下次发送时按新参数重新加载模型。' : '下次发送时按新参数生效。');
  return parts.join('');
}

/**
 * 管理员确认某条参数建议后，写入该模型的参数（同样受数值范围限制）。
 * 只接受 PARAM_LIMITS 中的参数；返回实际写入的值，无效时返回 null。
 */
export function applyParamSuggestion(
  modelId,
  key,
  value,
  { storage = globalThis.localStorage, now = Date.now(), reason = '' } = {},
) {
  const limits = PARAM_LIMITS[key];
  if (!modelId || !limits || !Number.isFinite(Number(value))) return null;
  const clamped = Math.min(limits[1], Math.max(limits[0], INTEGER_PARAMS.has(key) ? Math.round(Number(value)) : Number(value)));
  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  const from = entry.params[key] ?? null;
  entry.params = { ...entry.params, [key]: clamped };
  if (from !== clamped) {
    appendParamLog(entry, [{ key, from, to: clamped }], { ts: now, source: 'suggestion', reason: `关键词识别：${reason}` });
  }
  writeAll(data, storage);
  return clamped;
}

/**
 * 清洗参数对象：只保留 PARAM_LIMITS 中的参数，并夹到合法范围。
 * 没有任何有效项时返回 null。导出文件与导入文件都使用它，避免写入或信任非法值。
 */
export function sanitizeParams(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const out = {};
  for (const key of Object.keys(PARAM_LIMITS)) {
    const value = Number(raw[key]);
    if (raw[key] === undefined || raw[key] === null || raw[key] === '' || !Number.isFinite(value)) continue;
    const limits = PARAM_LIMITS[key];
    out[key] = Math.min(limits[1], Math.max(limits[0], INTEGER_PARAMS.has(key) ? Math.round(value) : value));
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

function newId(now) {
  return `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`;
}

