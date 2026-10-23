/**
 * 生图参数与风格提示（纯逻辑，仅保存在本机）。
 *
 * - 管理员参数行：“参数：width=768, height=512, steps=25, guidance=6, seed=42”。
 *   只接受下表中的键，数值夹到范围内；width/height 取整到 8 的倍数。
 * - 管理员的风格提示（hints）：附加到提示词末尾，与用户提示词一起交给文本编码器。
 *   CLIP 只读取前 77 个 token，超出部分不会参与生图。
 * - 生效优先级：界面控件（默认值）< 管理员参数。
 * 存储键 oj-ai.image.v1，按图片模型分别保存。
 */
import { normalizeSize, snapSize } from './sizes.js';

export const IMAGE_STORAGE_KEY = 'oj-ai.image.v1';
export const MAX_IMAGE_HINTS = 30;

export const IMAGE_PARAM_LIMITS = {
  width: [256, 768],
  height: [256, 768],
  steps: [2, 50],
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

/** 提示词 = 用户描述 + 风格提示（逗号分隔）。 */
export function buildImagePrompt(userPrompt, hints = []) {
  return [String(userPrompt || '').trim(), ...hints.map((h) => h.text)].filter(Boolean).join(', ');
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
