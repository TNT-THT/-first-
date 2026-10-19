/**
 * 推理模型的“思考模式”开关与思考过程的保存规则（纯逻辑，便于测试）。
 *
 * - 开关只作用于推理模型（config.json 中 reasoning: true 的模型，以及基于它们导入/训练的“新建 AI”）。
 * - 默认开启思考（与之前的行为一致）；关闭后，WebLLM 会在回复前预填一段空的思考块，模型直接给出回答。
 * - 思考过程只在本机显示和保存，不会作为上下文发送给模型。
 */

export const THINKING_PREF_KEY = 'oj-ai.thinking.v1';
/** 历史记录中保存的思考过程最多字符数（超出部分截断，避免占满 localStorage）。 */
export const MAX_STORED_THINKING = 4000;

/** 读取思考模式偏好（默认开启）。 */
export function getThinkingPref(storage = globalThis.localStorage) {
  try {
    return storage.getItem(THINKING_PREF_KEY) !== 'off';
  } catch {
    return true;
  }
}

export function setThinkingPref(on, storage = globalThis.localStorage) {
  try {
    storage.setItem(THINKING_PREF_KEY, on ? 'on' : 'off');
  } catch {
    // 存储不可用时忽略：本次页面内仍按当前开关生效
  }
}

/**
 * 判断模型是否为推理模型。
 * @param {string} modelId
 * @param {{models?: Array<{id: string, reasoning?: boolean}>, localEntries?: Array<{id: string, originId?: string, baseLabel?: string}>}} lists
 *   models：config.json 中的下拉列表；localEntries：本机“新建 AI”登记表。
 */
export function isReasoningModel(modelId, { models = [], localEntries = [] } = {}) {
  if (!modelId) return false;
  const official = models.find((m) => m.id === modelId);
  if (official) return !!official.reasoning;
  const local = localEntries.find((e) => e.id === modelId);
  if (local) {
    const origin = local.originId && models.find((m) => m.id === local.originId);
    if (origin) return !!origin.reasoning;
    const base = local.baseLabel || '';
    return base.includes('推理') && !base.includes('非推理');
  }
  return false;
}

/**
 * 发送给 WebLLM 的思考相关参数。
 * 只有推理模型且关闭思考时才需要传 enable_thinking: false；其余情况沿用模型默认模板。
 */
export function thinkingRequestOptions({ supportsThinking, enabled }) {
  if (supportsThinking && enabled === false) return { extra_body: { enable_thinking: false } };
  return {};
}

/** 保存到历史记录前截断思考过程。 */
export function truncateThinking(text, max = MAX_STORED_THINKING) {
  const s = String(text || '');
  if (s.length <= max) return s;
  return `${s.slice(0, max)}\n…（思考过程较长，历史记录中只保存前 ${max} 字）`;
}
