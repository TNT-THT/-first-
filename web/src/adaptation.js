/**
 * AI 适应记录（仅本机）：管理员在对话中给出的反馈，按模型分别保存。
 *
 * - 浏览器内推理（WebLLM）只做推理，不能在页面中修改模型权重。因此“适应”的方式是：
 *   把管理员的反馈作为该模型的附加说明，放进之后每次回答的系统提示词中；
 *   参数反馈（如 temperature、maxTokens）只影响该模型的采样参数。
 * - 记录只保存在本浏览器的 localStorage 中，不上传云端，也不会写入导出的 .ojmodel 文件。
 * - 远程（服务端）模式下不记录、不注入，避免把反馈发送到服务器。
 * - 管理员反馈会被采纳，但附加说明末尾固定一条护栏：反馈不得违反安全准则。
 */

const KEY = 'oj-ai.adapt.v1';
export const MAX_NOTES_PER_MODEL = 30;
export const MAX_NOTE_CHARS = 500;
export const PROMPT_BUDGET_CHARS = 1500;
const PARAM_LIMITS = { temperature: [0, 1.5], maxTokens: [64, 16384] };
const PARAM_LINE = /^\s*(?:参数|设置)[:：]\s*(.+)$/m;
const GUARD =
  '以上是管理员针对本 AI 给出的反馈，请据此调整回答的风格、格式与事实。这些反馈不得用于违反安全准则或应用的基本规定。';

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

function entryFor(data, modelId) {
  if (!data.models[modelId]) data.models[modelId] = { notes: [], params: {} };
  return data.models[modelId];
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
    entry.params = { ...entry.params, ...params };
    result.params = params;
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
  if (!entry.notes.length && !Object.keys(entry.params || {}).length) delete data.models[modelId];
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
  return `\n\n【管理员反馈（仅本机记录，请采纳）】\n${picked.join('\n')}\n${GUARD}`;
}

/** 组合基础系统提示词与该模型的适应说明。 */
export function composeSystemPrompt(base, modelId, options) {
  return `${base || ''}${buildAdaptationPrompt(modelId, options)}`;
}
