/**
 * 生图参数与风格提示（纯逻辑，仅保存在本机）。
 *
 * - 管理员参数行：“参数：width=768, height=512, steps=25, guidance=6, seed=42, inputTokens=4096”。
 *   inputTokens 为描述输入上限（token，默认 4096，最高 16384），不影响生成结果，只决定何时拒绝过长的输入。
 *   只接受下表中的键，数值夹到范围内；width/height 取整到 8 的倍数。
 * - 管理员的风格提示（hints）：附加到提示词末尾。
 * - 全局知识库（管理员在聊天管理员面板的一个输入框里维护，对所有模型共用）：
 *   由调用方作为 lead 传进来，放在提示词最前面（优先保证被文本编码器读取）；
 *   知识库为空时不传任何前置内容（不塞空壳词汇）。
 *   注意：CLIP 每段读取 75 个内容 token，描述最多分 4 段（约 300 个 token）；超出部分不参与生图。
 *   管理员内容只占第一段，超出单段时拒绝生成（中文通常每字占多个 token）。
 *   旧的生图“更正/条目式知识库”已随全站对话修正功能一并删除，旧数据不再读取。
 * - 生效优先级：界面控件（默认值）< 管理员参数。
 * 存储键 oj-ai.image.v1，按图片模型分别保存（与聊天模型的适应记录 oj-ai.adapt.v1 分开）。
 */
import { normalizeSize, snapSize } from './sizes.js';
import { DEFAULT_INPUT_TOKENS, MAX_INPUT_TOKENS, MIN_INPUT_TOKENS } from './prompt-budget.js';

export const IMAGE_STORAGE_KEY = 'oj-ai.image.v1';
export const MAX_IMAGE_HINTS = 30;
export const MAX_IMAGE_NOTES = 30;

export const IMAGE_PARAM_LIMITS = {
  width: [256, 768],
  height: [256, 768],
  steps: [1, 50], // 下限由调度器决定（PNDM 至少 2 步，SDXS 等一步模型为 1 步），见 sd15.js
  guidance: [1, 20],
  seed: [0, 4294967295],
  // 描述输入上限（token）：仅管理员可通过参数行修改；默认 4096，硬上限 16384（见 prompt-budget.js）
  inputTokens: [MIN_INPUT_TOKENS, MAX_INPUT_TOKENS],
};
const INTEGER_KEYS = new Set(['width', 'height', 'steps', 'seed', 'inputTokens']);
const PARAM_LINE = /^\s*(?:参数|设置)[:：]\s*(.+)$/m;

/** 解析参数行；没有有效参数时返回 null。 */
export function parseImageParams(text) {
  const m = PARAM_LINE.exec(text || '');
  if (!m) return null;
  const out = {};
  for (const pair of m[1].split(/[,，;；\s]+/)) {
    const [key, raw] = pair.split('=').map((s) => s?.trim());
    const limits = IMAGE_PARAM_LIMITS[key];
    if (!limits || raw === undefined || raw === '') continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) continue;
    out[key] = clampParam(key, value);
  }
  return Object.keys(out).length ? out : null;
}

/** 去掉参数行后的正文（提示词）。 */
export function stripImageParamLine(text) {
  return String(text || '').replace(PARAM_LINE, '').trim();
}

/** 把一个值夹到允许范围并取整（尺寸取整到 8 的倍数）。 */
export function clampParam(key, value) {
  const [min, max] = IMAGE_PARAM_LIMITS[key];
  if (key === 'width' || key === 'height') return snapSize(Math.min(max, Math.max(min, value)));
  const v = Math.min(max, Math.max(min, value));
  return INTEGER_KEYS.has(key) ? Math.round(v) : v;
}

function readAll(storage) {
  try {
    const data = JSON.parse(storage?.getItem(IMAGE_STORAGE_KEY) || 'null');
    return data && typeof data === 'object' && data.models && typeof data.models === 'object'
      ? data
      : { models: {} };
  } catch {
    return { models: {} };
  }
}

function writeAll(data, storage) {
  storage?.setItem(IMAGE_STORAGE_KEY, JSON.stringify(data));
}

function entryFor(data, modelId) {
  const e = data.models[modelId] || { params: {}, hints: [] };
  e.params = e.params || {};
  e.hints = Array.isArray(e.hints) ? e.hints : [];
  data.models[modelId] = e;
  return e;
}

export function getImageSettings(modelId, { storage = globalThis.localStorage } = {}) {
  const e = readAll(storage).models[modelId];
  return {
    params: e?.params && typeof e.params === 'object' ? e.params : {},
    hints: Array.isArray(e?.hints) ? e.hints : [],
  };
}

/** 清洗管理员参数：只保留合法键并限制在范围内。没有有效参数时返回 null。 */
export function sanitizeImageParams(raw) {
  if (!raw || typeof raw !== 'object') return null;
  const clean = {};
  for (const key of Object.keys(IMAGE_PARAM_LIMITS)) {
    const v = Number(raw[key]);
    if (raw[key] === undefined || raw[key] === null || raw[key] === '' || !Number.isFinite(v)) continue;
    clean[key] = clampParam(key, v);
  }
  return Object.keys(clean).length ? clean : null;
}

/** 合并管理员参数（只保留合法键）。返回本次写入后的全部管理员参数。 */
export function setImageParams(modelId, raw, { storage = globalThis.localStorage } = {}) {
  if (!modelId) return null;
  const clean = sanitizeImageParams(raw);
  if (!clean) return null;
  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  entry.params = { ...entry.params, ...clean };
  writeAll(data, storage);
  return entry.params;
}

export function clearImageParams(modelId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  if (!data.models[modelId]) return false;
  data.models[modelId].params = {};
  writeAll(data, storage);
  return true;
}

/** 添加风格提示。空文本或与已有提示完全相同时返回 null；超过上限时丢弃最旧的。 */
export function addImageHint(modelId, text, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  const body = String(text || '').trim();
  if (!modelId || !body) return null;
  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  if (entry.hints.some((h) => h.text === body)) return null;
  const hint = { id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`, text: body, ts: now };
  entry.hints = [...entry.hints, hint].slice(-MAX_IMAGE_HINTS);
  writeAll(data, storage);
  return hint;
}

export function removeImageHint(modelId, hintId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  const entry = data.models[modelId];
  if (!entry) return false;
  const before = entry.hints.length;
  entry.hints = entry.hints.filter((h) => h.id !== hintId);
  writeAll(data, storage);
  return entry.hints.length !== before;
}

export function clearImageHints(modelId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  if (!data.models[modelId]) return false;
  data.models[modelId].hints = [];
  writeAll(data, storage);
  return true;
}

/** 有风格提示记录的生图 AI 的 id 列表（仅本机）。 */
export function imageAdminModelIds({ storage = globalThis.localStorage } = {}) {
  const models = readAll(storage).models;
  return Object.keys(models).filter((id) => Array.isArray(models[id]?.hints) && models[id].hints.length > 0);
}

/**
 * 提示词 = 前置内容（全局知识库）+ 用户描述 + 风格提示（逗号分隔）。
 * lead 为字符串数组；没有前置内容时结果与旧版本相同。
 */
export function buildImagePrompt(userPrompt, hints = [], lead = []) {
  const leading = lead.map((t) => String(t || '').trim()).filter(Boolean);
  return [...leading, String(userPrompt || '').trim(), ...hints.map((h) => h.text)].filter(Boolean).join(', ');
}

/** 生效的描述输入上限（token）：管理员参数优先，否则为默认值 4096；结果夹到 [1, 16384]。 */
export function effectiveInputLimit(params = {}) {
  const v = Number(params?.inputTokens);
  return Number.isFinite(v) ? clampParam('inputTokens', v) : DEFAULT_INPUT_TOKENS;
}

/** 全局知识库作为前置内容：非空时返回 [文本]，为空时返回 []（不注入）。 */
export function imageLeadTexts(knowledgeText) {
  const text = String(knowledgeText || '').trim();
  return text ? [text] : [];
}

/**
 * 生效的生图设置：界面控件 < 管理员参数。尺寸统一取整到 8 的倍数。
 * ui: {width,height,steps,guidance,seed}；admin: 管理员参数（可缺省键）。
 */
export function effectiveImageSettings(ui = {}, admin = {}) {
  const merged = { ...ui };
  for (const [k, v] of Object.entries(admin || {})) {
    if (v !== undefined && v !== null) merged[k] = v;
  }
  const size = normalizeSize(merged.width, merged.height);
  return {
    width: size.width,
    height: size.height,
    steps: Number.isFinite(merged.steps) ? merged.steps : 20,
    guidance: Number.isFinite(merged.guidance) ? merged.guidance : 7.5,
    seed: Number.isInteger(merged.seed) ? merged.seed : null,
  };
}
