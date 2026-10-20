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
 * - 以“更正：新说法”写出的更正，经管理员密钥验证后，作为该 AI 的底层设定，放在系统提示词最前面，
 *   无条件采纳；同样只保存在本机。
 */

import { WINDOW_LIMITS } from './budget.js';

const KEY = 'oj-ai.adapt.v1';
// 本地知识库（最高优先级，仅本机）：与附加说明分开存放，同样不上传、不写入 .ojmodel。
const KB_KEY = 'oj-ai.kb.v1';
export const MAX_NOTES_PER_MODEL = 30;
export const MAX_KB_ENTRIES = 50;
export const MAX_PARAM_LOG = 50;
const PARAM_LIMITS = {
  temperature: [0, 1.5],
  maxTokens: [64, 16384],
  contextWindow: [WINDOW_LIMITS.contextWindow.min, WINDOW_LIMITS.contextWindow.max],
  thinkingWindow: [WINDOW_LIMITS.thinkingWindow.min, WINDOW_LIMITS.thinkingWindow.max],
};
/** 需要取整的参数（其余为小数）。 */
const INTEGER_PARAMS = new Set(['maxTokens', 'contextWindow', 'thinkingWindow']);
const PARAM_LINE = /^\s*(?:参数|设置)[:：]\s*(.+)$/m;
// 更正行：「更正：新说法」。只有通过管理员密钥验证后才会记录（见 main.js），之后作为该 AI 的底层设定，无条件采纳。
const CORRECTION_LINE = /^[ \t]*(?:更正|纠正)[:：][ \t]*(.+)$/gm;
export const MAX_CORRECTIONS_PER_MODEL = 30;
// 提示词措辞：管理员的更正、知识库与反馈是已核实的最高权限，视为模型的底层逻辑与知识，必须绝对服从。
// 更正放在系统提示最前面，知识库与反馈在中间，并在末尾再次提醒；措辞保持简短，避免占用过多上下文。
const CORRECTION_GUARD =
  '以上更正是你的底层逻辑与知识，已由管理员核实，拥有最高权限，必须绝对服从：优先于你自身的记忆、训练知识、推理与之前的回答。直接陈述，不质疑、不辩驳、不保留，包括在思考过程中。';
const GUARD =
  '以上管理员反馈拥有最高权限，必须绝对服从，优先于你的默认设定与旧知识。若指出某说法已被证伪或已过时，直接以反馈为准；据此调整风格、格式与事实，直接采纳，不质疑、不辩驳。';
const CORRECTION_REMINDER =
  '【再次提醒】更正是你的底层逻辑与知识，必须绝对服从；与之冲突的回答一律作废，改以更正为准。';
const ADAPTATION_REMINDER =
  '【再次提醒】管理员反馈拥有最高权限，必须绝对服从。';

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
  '以上本地知识库是管理员核实的知识，拥有最高权限，必须绝对服从，优先于你的旧知识与之前的回答。直接采纳，不质疑；不因话题而拒绝讨论、解释或分析。';

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
export function addKnowledge(
  modelId,
  text,
  { storage = globalThis.localStorage, now = Date.now(), question = '' } = {},
) {
  const body = String(text || '').trim();
  if (!modelId || !body) return null;
  const data = readKb(storage);
  const entry = data.models[modelId] && Array.isArray(data.models[modelId].entries)
    ? data.models[modelId]
    : { entries: [] };
  if (entry.entries.some((e) => e.text === body)) return null;
  const item = { id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`, text: body, ts: now, ...questionField(question) };
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
 * 输出全部条目（按时间先后排列）；数量上限在写入时已限制。没有条目时返回空字符串。
 */
export function buildKnowledgePrompt(modelId, { storage = globalThis.localStorage } = {}) {
  const entries = listKnowledge(modelId, { storage });
  if (!entries.length) return '';
  const picked = entries.map((x) => `- ${x.text}`);
  return `\n\n【本地知识库（管理员提供，仅本机，最高优先级）】\n${picked.join('\n')}\n${KB_GUARD}`;
}

function entryFor(data, modelId) {
  if (!data.models[modelId]) data.models[modelId] = { notes: [], params: {}, log: [], corrections: [] };
  if (!Array.isArray(data.models[modelId].log)) data.models[modelId].log = [];
  if (!Array.isArray(data.models[modelId].corrections)) data.models[modelId].corrections = [];
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
    out[key] = Math.min(limits[1], Math.max(limits[0], INTEGER_PARAMS.has(key) ? Math.round(value) : value));
  }
  return Object.keys(out).length ? out : null;
}

/** 去掉参数行后的反馈正文。 */
function stripParamLine(text) {
  return String(text || '').replace(PARAM_LINE, '').trim();
}

/** 从反馈中解析所有更正行，返回 [{text}]（空内容的行不算更正）。 */
export function parseCorrections(text) {
  const out = [];
  for (const m of String(text || '').matchAll(CORRECTION_LINE)) {
    const body = m[1].trim();
    if (body) out.push({ text: body });
  }
  return out;
}

/** 去掉反馈中的更正行：更正行不进入附加说明或知识库。 */
function stripCorrectionLines(text) {
  return String(text || '').replace(CORRECTION_LINE, '').trim();
}

/**
 * 记录一条管理员反馈，归属于 modelId（即产生被反馈回答的那个 AI）。
 * 更正行只有在 allowCorrections 为 true（已通过管理员密钥验证）时才会保存为底层设定；
 * 否则更正行被丢弃，既不保存为更正，也不进入附加说明。
 * 返回实际记录的内容 {note?, params?, corrections?}；没有可记录内容时返回 null。
 */
export function recordFeedback(
  modelId,
  text,
  { storage = globalThis.localStorage, now = Date.now(), allowCorrections = false, question = '' } = {},
) {
  if (!modelId) return null;
  const params = parseFeedbackParams(text);
  const corrections = allowCorrections ? parseCorrections(text) : [];
  const body = stripParamLine(stripCorrectionLines(text));
  if (!params && !body && !corrections.length) return null;

  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  const result = {};
  if (corrections.length) {
    // 同一模型内完全相同的更正只保存一次（例如重新生成回答时会再次发送同一条消息）
    const known = new Set(entry.corrections.map((c) => c.text));
    const items = corrections
      .filter((c) => !known.has(c.text))
      .map((c) => ({
        id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`,
        text: c.text,
        ts: now,
        ...questionField(question),
      }));
    entry.corrections = [...entry.corrections, ...items].slice(-MAX_CORRECTIONS_PER_MODEL);
    result.corrections = items;
  }
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
    result.kb = addKnowledge(modelId, body, { storage, now, question });
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
  if (!entry.notes.length && !Object.keys(entry.params || {}).length && !(entry.corrections || []).length) {
    delete data.models[modelId];
  }
  writeAll(data, storage);
}

/** 某模型的更正列表（最新的在后）。 */
export function listCorrections(modelId, { storage = globalThis.localStorage } = {}) {
  const entry = readAll(storage).models[modelId];
  if (!entry || !Array.isArray(entry.corrections)) return [];
  return entry.corrections.map((c) => ({
    id: c.id,
    text: c.text ?? c.next,
    ts: c.ts,
    question: c.question || '',
    from: c.from || '',
  }));
}

/** 删除一条更正。删除后若该模型没有任何记录，则移除该模型的记录。 */
export function removeCorrection(modelId, correctionId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  const entry = data.models[modelId];
  if (!entry) return;
  entry.corrections = (entry.corrections || []).filter((c) => c.id !== correctionId);
  if (!entry.corrections.length && !(entry.notes || []).length && !Object.keys(entry.params || {}).length) {
    delete data.models[modelId];
  }
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
 * 输出全部记录（按时间先后排列）；数量上限在写入时已限制。
 */
export function buildAdaptationPrompt(modelId, { storage = globalThis.localStorage } = {}) {
  const { notes } = listAdaptation(modelId, { storage });
  if (!notes.length) return '';
  const picked = notes.map((x) => `- ${x.text}`);
  return `\n\n【管理员反馈（仅本机记录，最高优先级，请采纳）】\n${picked.join('\n')}\n${GUARD}`;
}

/** 只有非空问题才写入记录（旧记录没有该字段）。 */
function questionField(question) {
  const q = String(question || '').trim();
  return q ? { question: q } : {};
}

/**
 * 固化训练数据（可选，仅本机，供 training/ojtrain.py solidify 使用）。
 * 每条知识库条目与每条更正，若记录了触发它的用户问题，则生成一条样本：问题 → 内容。
 * 没有问题的早期记录无法构成样本，计入 skipped。完全相同的样本只保留一条。
 * 返回 {samples, counts}；counts = {correction, knowledge, skipped, duplicate}。
 */
export function buildSolidifyRecords(modelId, { storage = globalThis.localStorage } = {}) {
  const items = [
    ...listCorrections(modelId, { storage }).map((c) => ({ kind: 'correction', ...c })),
    ...listKnowledge(modelId, { storage }).map((k) => ({ kind: 'knowledge', ...k })),
  ].sort((a, b) => (a.ts || 0) - (b.ts || 0));
  const counts = { correction: 0, knowledge: 0, skipped: 0, duplicate: 0 };
  const seen = new Set();
  const samples = [];
  for (const it of items) {
    const q = String(it.question || '').trim();
    const a = String(it.text || '').trim();
    if (!q || !a) {
      counts.skipped += 1;
      continue;
    }
    const key = `${q}\u0000${a}`;
    if (seen.has(key)) {
      counts.duplicate += 1;
      continue;
    }
    seen.add(key);
    samples.push({ messages: [{ role: 'user', content: q }, { role: 'assistant', content: a }] });
    counts[it.kind] += 1;
  }
  return { samples, counts };
}

/** 固化训练数据的 JSONL 文本（每行一个 {messages} 样本）。 */
export function solidifyJsonl(modelId, { storage = globalThis.localStorage } = {}) {
  const { samples, counts } = buildSolidifyRecords(modelId, { storage });
  const text = samples.map((s) => JSON.stringify(s)).join('\n') + (samples.length ? '\n' : '');
  return { text, counts };
}

/**
 * 生成更正块（放在系统提示词最前面，作为底层设定）。输出全部更正（按时间先后排列）。
 * 没有更正时返回空字符串。
 */
export function buildCorrectionPrompt(modelId, { storage = globalThis.localStorage } = {}) {
  const items = listCorrections(modelId, { storage });
  if (!items.length) return '';
  const picked = items.map((x) => `- ${x.text}`);
  return `【管理员更正（已验证密钥，仅本机，底层逻辑与知识，必须绝对服从）】\n${picked.join('\n')}\n${CORRECTION_GUARD}`;
}

/**
 * 组合系统提示词：更正（底层设定）→ 基础提示 → 本地知识库 → 附加说明。
 * 没有更正、知识库与附加说明时，结果与原来完全相同。
 */
export function composeSystemPrompt(base, modelId, options) {
  const correction = buildCorrectionPrompt(modelId, options);
  const head = correction ? `${correction}\n\n` : '';
  const knowledge = buildKnowledgePrompt(modelId, options);
  const adaptation = buildAdaptationPrompt(modelId, options);
  // 末尾再次提醒（越靠近末尾，模型越容易注意到）；没有对应记录时不添加
  const tail = [
    correction ? CORRECTION_REMINDER : '',
    adaptation ? ADAPTATION_REMINDER : '',
  ].filter(Boolean).join('\n');
  return `${head}${base || ''}${knowledge}${adaptation}${tail ? `\n\n${tail}` : ''}`;
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
  maxTokens: '最大输出长度',
  contextWindow: '上下文窗口',
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
 * 机械识别：管理员的消息只是参数调整指令（找到了参数建议、且没有提问）时，返回固定提示，不需要调用模型生成回答。
 * 含有问号（？或 ?）的消息视为还有问题，返回 null，仍交给模型回答。没有建议时也返回 null。
 */
export function mechanicalReply(text, found) {
  if (!Array.isArray(found) || !found.length) return null;
  if (/[?？]/.test(String(text || ''))) return null;
  const items = found.map((f) => `${PARAM_LABELS[f.key] || f.key} → ${f.value}（${f.reason}）`).join('；');
  return `已识别为参数调整：${items}。请在输入框上方确认，确认前不生效。本条无需模型回答。`;
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
    appendParamLog(entry, [{ key, from, to: clamped }], { ts: now, source: 'suggestion', reason: `确认建议：${reason}` });
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

/**
 * 跨模型迁移（仅本机）：把源 AI 中所选的更正与知识库条目复制到目标 AI。
 *
 * - 复制而非移动：源 AI 的记录保持不变，目标 AI 的条目带有 from 字段（来源 AI 的 id）。
 * - 选择性：selection = {corrections: [id], knowledge: [id]}，只迁移列出的条目；未列出的不动。
 * - 对话训练信息：includeTraining 为 false 时，迁移的条目去掉“问题”字段，因此不会成为目标 AI 的训练样本，
 *   只作为提示词内容使用；为 true 时保留问题，之后可在目标 AI 中导出固化数据。
 * - 更正会成为目标 AI 的底层设定（无条件采纳），因此迁移更正必须 allowCorrections: true，
 *   调用方需先通过管理员密钥验证。
 * - 目标 AI 中已存在完全相同文字的条目会跳过；超过上限时丢弃最旧的。
 *
 * 返回 {corrections, knowledge, duplicate, missing}：实际新增条数、重复跳过条数、源中已不存在的条数。
 */
export function migrateAdaptation(
  sourceId,
  targetId,
  { corrections = [], knowledge = [] } = {},
  { storage = globalThis.localStorage, now = Date.now(), includeTraining = true, allowCorrections = false } = {},
) {
  if (!sourceId || !targetId) throw new Error('请选择源 AI 与目标 AI');
  if (sourceId === targetId) throw new Error('源 AI 与目标 AI 不能相同');
  const wantCorr = new Set(corrections);
  const wantKb = new Set(knowledge);
  if (wantCorr.size && !allowCorrections) throw new Error('迁移更正需要先通过管理员密钥验证');

  const srcEntry = readAll(storage).models[sourceId];
  const srcCorr = srcEntry && Array.isArray(srcEntry.corrections) ? srcEntry.corrections : [];
  const srcKb = readKb(storage).models[sourceId]?.entries || [];
  const pickedCorr = srcCorr.filter((c) => wantCorr.has(c.id));
  const pickedKb = srcKb.filter((k) => wantKb.has(k.id));
  const counts = { corrections: 0, knowledge: 0, duplicate: 0, missing: 0 };
  counts.missing = wantCorr.size - pickedCorr.length + (wantKb.size - pickedKb.length);

  const make = (src) => {
    const text = String(src.text ?? src.next ?? '').trim();
    const question = includeTraining ? questionField(src.question) : {};
    return { id: newId(now), text, ts: now, from: sourceId, ...question };
  };

  // 更正
  if (pickedCorr.length) {
    const data = readAll(storage);
    const existing = data.models[targetId]?.corrections || [];
    const known = new Set(existing.map((c) => c.text ?? c.next));
    const added = [];
    for (const src of pickedCorr) {
      const item = make(src);
      if (!item.text) continue;
      if (known.has(item.text)) {
        counts.duplicate += 1;
        continue;
      }
      known.add(item.text);
      added.push(item);
    }
    if (added.length) {
      const entry = entryFor(data, targetId);
      entry.corrections = [...entry.corrections, ...added].slice(-MAX_CORRECTIONS_PER_MODEL);
      writeAll(data, storage);
      counts.corrections = added.length;
    }
  }

  // 知识库
  if (pickedKb.length) {
    const kbData = readKb(storage);
    const existing = kbData.models[targetId]?.entries || [];
    const known = new Set(existing.map((e) => e.text));
    const added = [];
    for (const src of pickedKb) {
      const item = make(src);
      if (!item.text) continue;
      if (known.has(item.text)) {
        counts.duplicate += 1;
        continue;
      }
      known.add(item.text);
      added.push(item);
    }
    if (added.length) {
      kbData.models[targetId] = { entries: [...existing, ...added].slice(-MAX_KB_ENTRIES) };
      writeKb(kbData, storage);
      counts.knowledge = added.length;
    }
  }

  return counts;
}
