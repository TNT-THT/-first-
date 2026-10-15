import { sha256TextHex } from './sha256.js';

/**
 * 纯静态站点的访问口令（软门禁）。
 * 口令的 SHA-256 哈希（加盐）保存在 config.json 中，明文不会出现在仓库里。
 * 注意：这只能挡住随意访问，不是服务端鉴权；浏览器内推理本身不产生 API 费用。
 */
const UNLOCK_KEY = 'oj-ai.unlock';
const UNLOCK_DAYS = 30;

export function hasAccessGate(access) {
  return !!access?.passwordSha256;
}

/**
 * 计算 UTF-8 文本的 SHA-256（十六进制）。
 * 安全上下文中优先用 crypto.subtle；在第三方页面的 sandbox iframe 等非安全上下文中
 * crypto.subtle 不存在，此时回退到纯 JavaScript 实现，结果完全相同。
 */
export async function sha256Hex(text) {
  const subtle = globalThis.crypto?.subtle;
  if (subtle && typeof subtle.digest === 'function') {
    try {
      const data = new TextEncoder().encode(text);
      const digest = await subtle.digest('SHA-256', data);
      return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
    } catch {
      /* 落到下面的纯 JavaScript 实现 */
    }
  }
  return sha256TextHex(text);
}

export async function verifyAccessPassword(access, password) {
  if (!hasAccessGate(access)) return true;
  if (typeof password !== 'string' || !password) return false;
  const hash = await sha256Hex(`${access.salt || ''}:${password}`);
  return hash === access.passwordSha256;
}

export function isUnlocked(access) {
  if (!hasAccessGate(access)) return true;
  try {
    const rec = JSON.parse(localStorage.getItem(UNLOCK_KEY) || 'null');
    return !!rec && typeof rec.exp === 'number' && rec.exp > Date.now();
  } catch {
    return false;
  }
}

export function setUnlocked() {
  try {
    localStorage.setItem(UNLOCK_KEY, JSON.stringify({ exp: Date.now() + UNLOCK_DAYS * 86400e3 }));
  } catch {
    /* 存储不可用时本次访问仍然有效 */
  }
}

export function clearUnlocked() {
  try {
    localStorage.removeItem(UNLOCK_KEY);
  } catch {
    /* ignore */
  }
}
