/**
 * API 管理员验证的本机持久化。
 *
 * 需求：管理员口令只验证一次，之后（含关闭“API 接入”面板、刷新页面）都不用
 * 再验证；只有管理员主动点“锁定管理面板”或令牌到期才需要重新验证。
 * 因此登录成功后把 {token, expiresAt} 存进本机 localStorage，启动时恢复；
 * 服务端令牌为无状态签名令牌（见 server/auth.js），恢复后可直接使用。
 * 注意：令牌只保存在当前浏览器，任何能操作这台浏览器的人都能用管理员功能，
 * 这与“本机管理员面板”的定位一致；不用时请点“锁定管理面板”。
 */

export const API_ADMIN_SESSION_KEY = 'oj-ai.apiAdminSession.v1';

/** 保存管理员会话到存储（存储不可用时静默失败，仅本次页面有效）。 */
export function saveApiAdminSession(storage, session) {
  try {
    if (!session?.token) return;
    storage?.setItem(
      API_ADMIN_SESSION_KEY,
      JSON.stringify({ token: String(session.token), expiresAt: Number(session.expiresAt) || 0 }),
    );
  } catch {
    /* 存储被禁用：退化为页面内存令牌 */
  }
}

/**
 * 读取已保存的管理员会话；不存在或已过期返回 null（过期会顺手清除）。
 * @param {Storage-like} storage
 * @param {number} now 当前时间戳（毫秒）
 */
export function loadApiAdminSession(storage, now = Date.now()) {
  try {
    const raw = storage?.getItem(API_ADMIN_SESSION_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data?.token || (data.expiresAt && data.expiresAt <= now)) {
      clearApiAdminSession(storage);
      return null;
    }
    return { token: String(data.token), expiresAt: Number(data.expiresAt) || 0 };
  } catch {
    return null;
  }
}

/** 锁定：清除已保存的管理员会话。 */
export function clearApiAdminSession(storage) {
  try {
    storage?.removeItem(API_ADMIN_SESSION_KEY);
  } catch {
    /* 存储不可用时无事可做 */
  }
}
