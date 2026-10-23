/**
 * 对话预设（按浏览器保存，所有用户都能使用）。
 *
 * 参数优先级（从低到高）：config.json 默认值 < 预设 < 管理员参数（参数行，仅本机）。
 * - 预设是一组“机械”参数：点击后立即生效，不调用模型。
 * - 预设里的 null 表示“不覆盖”，沿用 config.json 的默认值。
 * - 管理员仍可用参数行修改数值；“恢复默认”删除的是管理员参数，之后回到预设的值。
 * - 思考开关属于全局偏好：预设的 thinking 在点击时写入该偏好（之后仍可手动切换）。
 */
import { WINDOW_LIMITS } from './budget.js';

export const PRESET_KEY = 'oj-ai.preset.v1';
export const DEFAULT_PRESET_ID = 'default';

/** 预设可以覆盖的数值参数（与管理员参数相同的键名，便于统一显示）。 */
export const PRESET_NUMERIC_KEYS = ['temperature', 'maxTokens', 'contextWindow', 'slidingWindow', 'repetitionPenalty'];

export const PRESETS = [
  {
    id: DEFAULT_PRESET_ID,
    label: '默认',
    note: '使用 config.json 的默认值，思考模式开启。',
    thinking: true,
    temperature: null,
    maxTokens: null,
    contextWindow: null,
    slidingWindow: null,
    repetitionPenalty: null,
  },
  {
    id: 'long-form',
    label: '长文章（大体量文学文本）',
    note: '小说、散文、长篇创作：关闭思考，温度 0.8，重复惩罚 1.15，上下文 20000，输出上限 16384。窗口较大，占用更多显存。',
    thinking: false,
    temperature: 0.8,
    maxTokens: 16384,
    contextWindow: 20000,
    slidingWindow: 20000,
    repetitionPenalty: 1.15,
  },
  {
    id: 'code',
    label: '代码 / 解题',
    note: '编程、推理题：开启思考，温度 0.2，重复惩罚 1.05，输出上限 8192。',
    thinking: true,
    temperature: 0.2,
    maxTokens: 8192,
    contextWindow: 16384,
    slidingWindow: 16384,
    repetitionPenalty: 1.05,
  },
  {
    id: 'quick',
    label: '快速问答',
    note: '短问题，响应快、显存少：关闭思考，温度 0.5，输出上限 1024，上下文 4096。',
    thinking: false,
    temperature: 0.5,
    maxTokens: 1024,
    contextWindow: WINDOW_LIMITS.contextWindow.min,
    slidingWindow: WINDOW_LIMITS.slidingWindow.min,
    repetitionPenalty: 1.1,
  },
  {
    id: 'low-vram',
    label: '省显存',
    note: '显存紧张的电脑：关闭思考，温度 0.3，输出上限 4096，上下文 4096。',
    thinking: false,
    temperature: 0.3,
    maxTokens: 4096,
    contextWindow: WINDOW_LIMITS.contextWindow.min,
    slidingWindow: WINDOW_LIMITS.slidingWindow.min,
    repetitionPenalty: 1.1,
  },
];

export function findPreset(id) {
  return PRESETS.find((p) => p.id === id) || null;
}

/** 当前选择的预设（按浏览器保存）；未设置或无效时为默认。 */
export function getPresetId(storage = globalThis.localStorage) {
  try {
    const id = storage.getItem(PRESET_KEY);
    return findPreset(id) ? id : DEFAULT_PRESET_ID;
  } catch {
    return DEFAULT_PRESET_ID;
  }
}

export function setPresetId(id, storage = globalThis.localStorage) {
  if (!findPreset(id)) return false;
  storage.setItem(PRESET_KEY, id);
  return true;
}

/** 预设中的数值覆盖（只含非 null 的项）。 */
export function presetOverrides(id = getPresetId()) {
  const p = findPreset(id) || findPreset(DEFAULT_PRESET_ID);
  const out = {};
  for (const key of PRESET_NUMERIC_KEYS) {
    if (p[key] !== null && p[key] !== undefined) out[key] = p[key];
  }
  return out;
}

/**
 * 分层合并：默认值 < 预设 < 管理员参数。
 * admin 中的 undefined / null 不覆盖（它们表示“没有管理员设置”）。
 */
export function layerParams(defaults, preset = {}, admin = {}) {
  const out = { ...defaults };
  for (const layer of [preset, admin]) {
    for (const [key, value] of Object.entries(layer || {})) {
      if (value !== undefined && value !== null) out[key] = value;
    }
  }
  return out;
}
