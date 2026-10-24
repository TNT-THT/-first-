/**
 * 用户选择的“默认 AI 模型”（纯逻辑，便于测试）。
 *
 * - 在下拉框中切换模型时，选择会保存在本浏览器的 localStorage 中，下次打开页面时自动使用。
 * - 没有保存过选择时，使用 config.json 中的 defaultModel。
 * - 保存的模型不在下拉框中（例如已删除的“新建 AI”）时，忽略它并回到 defaultModel。
 */

export const DEFAULT_MODEL_KEY = 'oj-ai.defaultModel.v1';

/** 读取已保存的默认模型 id；没有则返回 null。 */
export function getSavedDefaultModel(storage = globalThis.localStorage) {
  try {
    return storage?.getItem(DEFAULT_MODEL_KEY) || null;
  } catch {
    return null;
  }
}

/** 保存默认模型 id（空值会清除保存的选择）。 */
export function saveDefaultModel(id, storage = globalThis.localStorage) {
  try {
    if (id) storage?.setItem(DEFAULT_MODEL_KEY, id);
    else storage?.removeItem(DEFAULT_MODEL_KEY);
  } catch {
    // 存储不可用（如隐私模式）时忽略，本次会话仍然有效
  }
}

/**
 * 决定启动时使用的模型：已保存且在可选列表中则用它，否则用 defaultModel。
 * @param {string[]} availableIds 下拉框中的模型 id
 */
export function resolveStartupModel({ availableIds, defaultModel, saved }) {
  if (saved && availableIds.includes(saved)) return saved;
  return defaultModel;
}

/** 生图模型的选择单独保存（与聊天模型互不影响）。 */
export const IMAGE_MODEL_KEY = 'oj-ai.imageModel.v1';

export function getSavedImageModel(storage = globalThis.localStorage) {
  try {
    return storage?.getItem(IMAGE_MODEL_KEY) || null;
  } catch {
    return null;
  }
}

export function saveImageModel(id, storage = globalThis.localStorage) {
  try {
    if (id) storage?.setItem(IMAGE_MODEL_KEY, id);
    else storage?.removeItem(IMAGE_MODEL_KEY);
  } catch {
    // 存储不可用时忽略
  }
}

/**
 * 生图模型的初始选择：上次保存且仍可选（available）则用它，否则用第一个可选的模型。
 * @param {Array<{id: string, selectable: boolean}>} models
 */
export function resolveStartupImageModel({ models, saved }) {
  const ok = (models || []).filter((m) => m.selectable);
  if (saved && ok.some((m) => m.id === saved)) return saved;
  return ok[0]?.id || null;
}
