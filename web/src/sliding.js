/**
 * “滑动窗口”开关的偏好（纯逻辑，便于测试）。
 *
 * - 与“思考模式”开关相同：在输入框旁切换，偏好保存在本浏览器。
 * - 默认关闭（沿用原来的上下文窗口与历史条数、字数限制）。
 * - 开启后，上下文窗口取滑动窗口大小（默认 4096，管理员可调），历史超出时最早的消息滚出窗口，
 *   系统提示（更正、知识库、反馈）始终保留在最前面。
 */

export const SLIDING_PREF_KEY = 'oj-ai.sliding.v1';

/** 读取滑动窗口偏好（默认关闭）。 */
export function getSlidingPref(storage = globalThis.localStorage) {
  try {
    return storage.getItem(SLIDING_PREF_KEY) === 'on';
  } catch {
    return false;
  }
}

export function setSlidingPref(on, storage = globalThis.localStorage) {
  try {
    storage.setItem(SLIDING_PREF_KEY, on ? 'on' : 'off');
  } catch {
    // 存储不可用时忽略：本次页面内仍按当前开关生效
  }
}
