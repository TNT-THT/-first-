/**
 * 生图参数与风格提示（纯逻辑，仅保存在本机）。
 *
 * - 管理员参数行：“参数：width=768, height=512, steps=25, guidance=6, seed=42”。
 *   只接受下表中的键，数值夹到范围内；width/height 取整到 8 的倍数。
 * - 管理员的风格提示（hints）：附加到提示词末尾。
 * - 管理员的更正（corrections，需管理员密钥验证）与本地知识库（knowledge）：
 *   作为该生图 AI 的底层设定，放在提示词最前面（优先保证被文本编码器读取）。
 *   注意：CLIP 只读取前 77 个 token（中文通常每字占多个 token），超出部分不会参与生图。
 * - 生效优先级：界面控件（默认值）< 管理员参数。
 * 存储键 oj-ai.image.v1，按图片模型分别保存（与聊天模型的适应记录 oj-ai.adapt.v1 分开）。
 */
import { normalizeSize, snapSize } from './sizes.js';

export const IMAGE_STORAGE_KEY = 'oj-ai.image.v1';
export const MAX_IMAGE_HINTS = 30;
export const MAX_IMAGE_NOTES = 30;

export const IMAGE_PARAM_LIMITS = {
  width: [256, 768],
  height: [256, 768],
  steps: [1, 50], // 下限由调度器决定（PNDM 至少 2 步，SDXS 等一步模型为 1 步），见 sd15.js
  guidance: [1, 20],
  seed: [0, 4294967295],
};
const INTEGER_KEYS = new Set(['width', 'height', 'steps', 'seed']);
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
  const e = data.models[modelId] || { params: {}, hints: [], corrections: [], knowledge: [] };
  e.params = e.params || {};
  e.hints = Array.isArray(e.hints) ? e.hints : [];
  e.corrections = Array.isArray(e.corrections) ? e.corrections : [];
  e.knowledge = Array.isArray(e.knowledge) ? e.knowledge : [];
  data.models[modelId] = e;
  return e;
}

export function getImageSettings(modelId, { storage = globalThis.localStorage } = {}) {
  const e = readAll(storage).models[modelId];
  return {
    params: e?.params && typeof e.params === 'object' ? e.params : {},
    hints: Array.isArray(e?.hints) ? e.hints : [],
    corrections: Array.isArray(e?.corrections) ? e.corrections : [],
    knowledge: Array.isArray(e?.knowledge) ? e.knowledge : [],
  };
}

/** 添加一条更正（底层设定）。调用方须已通过管理员密钥验证。空文本或重复时返回 null。 */
export function addImageCorrection(modelId, text, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  return addNote(modelId, 'corrections', text, { storage, now });
}

/** 添加一条本地知识库条目（仅本机）。空文本或重复时返回 null。 */
export function addImageKnowledge(modelId, text, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  return addNote(modelId, 'knowledge', text, { storage, now });
}

export function removeImageCorrection(modelId, id, { storage = globalThis.localStorage } = {}) {
  return removeNote(modelId, 'corrections', id, { storage });
}

export function removeImageKnowledge(modelId, id, { storage = globalThis.localStorage } = {}) {
  return removeNote(modelId, 'knowledge', id, { storage });
}

function addNote(modelId, field, text, { storage, now }) {
  const body = String(text || '').trim();
  if (!modelId || !body) return null;
  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  if (entry[field].some((n) => n.text === body)) return null;
  const note = { id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`, text: body, ts: now };
  entry[field] = [...entry[field], note].slice(-MAX_IMAGE_NOTES);
  writeAll(data, storage);
  return note;
}

function removeNote(modelId, field, id, { storage }) {
  const data = readAll(storage);
  const entry = data.models[modelId];
  if (!entry) return false;
  const before = entry[field].length;
  entry[field] = entry[field].filter((n) => n.id !== id);
  writeAll(data, storage);
  return entry[field].length !== before;
}

/** 合并管理员参数（只保留合法键）。返回本次写入后的全部管理员参数。 */
export function setImageParams(modelId, raw, { storage = globalThis.localStorage } = {}) {
  if (!modelId || !raw || typeof raw !== 'object') return null;
  const clean = {};
  for (const key of Object.keys(IMAGE_PARAM_LIMITS)) {
    const v = Number(raw[key]);
    if (raw[key] === undefined || raw[key] === null || raw[key] === '' || !Number.isFinite(v)) continue;
    clean[key] = clampParam(key, v);
  }
  if (!Object.keys(clean).length) return null;
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

/** 有管理员记录（更正、知识库或风格提示）的生图 AI 的 id 列表（仅本机）。 */
export function imageAdminModelIds({ storage = globalThis.localStorage } = {}) {
  const models = readAll(storage).models;
  return Object.keys(models).filter((id) => {
    const e = models[id];
    return ['corrections', 'knowledge', 'hints'].some((k) => Array.isArray(e?.[k]) && e[k].length > 0);
  });
}

/**
 * 把条目复制到某个生图 AI（仅本机；复制而非移动）。
 * items = {corrections, knowledge, hints}，每项为 [{text, from?}]。
 * - 更正会成为该生图 AI 的底层设定（放在提示词最前面），因此必须 allowCorrections（调用方已通过管理员密钥验证）。
 * - 完全相同文字的条目跳过（duplicate）；每类不超过上限，超出的不写入（overflow），不丢弃已有条目。
 * - 没有写入任何内容时不改动存储。
 * 返回 {corrections, knowledge, hints, duplicate, overflow}：各类实际新增条数。
 */
export function copyIntoImageModel(
  modelId,
  items = {},
  { storage = globalThis.localStorage, now = Date.now(), allowCorrections = false } = {},
) {
  if (!modelId) throw new Error('请选择目标生图 AI');
  if ((items.corrections || []).length && !allowCorrections) throw new Error('迁移更正需要先通过管理员密钥验证');
  const limits = { corrections: MAX_IMAGE_NOTES, knowledge: MAX_IMAGE_NOTES, hints: MAX_IMAGE_HINTS };
  const data = readAll(storage);
  const entry = entryFor(data, modelId);
  const counts = { corrections: 0, knowledge: 0, hints: 0, duplicate: 0, overflow: 0 };
  for (const field of ['corrections', 'knowledge', 'hints']) {
    const known = new Set(entry[field].map((n) => n.text));
    for (const src of items[field] || []) {
      const text = String(src?.text ?? '').trim();
      if (!text) continue;
      if (known.has(text)) {
        counts.duplicate += 1;
        continue;
      }
      if (entry[field].length >= limits[field]) {
        counts.overflow += 1;
        continue;
      }
      known.add(text);
      const note = { id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`, text, ts: now };
      if (src.from) note.from = String(src.from);
      entry[field] = [...entry[field], note];
      counts[field] += 1;
    }
  }
  if (counts.corrections || counts.knowledge || counts.hints) writeAll(data, storage);
  return counts;
}

/**
 * 提示词 = 前置内容（更正、知识库，按给定顺序）+ 用户描述 + 风格提示（逗号分隔）。
 * lead 为字符串数组；没有前置内容时结果与旧版本相同。
 */
export function buildImagePrompt(userPrompt, hints = [], lead = []) {
  const leading = lead.map((t) => String(t || '').trim()).filter(Boolean);
  return [...leading, String(userPrompt || '').trim(), ...hints.map((h) => h.text)].filter(Boolean).join(', ');
}

/** 管理员设定的前置内容：更正在前，其次是知识库。 */
export function imageLeadTexts(settings) {
  return [...(settings?.corrections || []), ...(settings?.knowledge || [])].map((n) => n.text);
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
