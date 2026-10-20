import { sha256TextHex } from './sha256.js';

/**
 * 纯静态站点的管理员口令（软门禁）。
 * 口令的 SHA-256 哈希（加盐）保存在 config.json 中，明文不会出现在仓库里。
 * 注意：这不是服务端鉴权；浏览器内推理本身不产生 API 费用。
 */
const ADMIN_KEY = 'oj-ai.admin';
const ADMIN_DAYS = 30;
// 旧版访客解锁记录不再使用
try {
  localStorage.removeItem('oj-ai.unlock');
} catch {
  /* ignore */
}

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

/**
 * 管理员权限（软门禁）。页面本身无需口令即可使用；管理员才能导出 / 导入模型。
 * 管理员密钥就是 config.json 中配置的口令（只校验哈希）。权限保存在本浏览器 30 天。
 * 注意：校验在浏览器中完成，无法防止有意绕过的人（哈希和盐是公开的）。
 */
export function isAdmin() {
  try {
    const rec = JSON.parse(localStorage.getItem(ADMIN_KEY) || 'null');
    return !!rec && typeof rec.exp === 'number' && rec.exp > Date.now();
  } catch {
    return false;
  }
}

export function grantAdmin() {
  try {
    localStorage.setItem(ADMIN_KEY, JSON.stringify({ exp: Date.now() + ADMIN_DAYS * 86400e3 }));
  } catch {
    /* 存储不可用时本次访问仍然有效 */
  }
}

export function revokeAdmin() {
  try {
    localStorage.removeItem(ADMIN_KEY);
  } catch {
    /* ignore */
  }
}

/** 管理员权限到期时间（用于界面显示）。 */
export function adminExpiry() {
  try {
    const rec = JSON.parse(localStorage.getItem(ADMIN_KEY) || 'null');
    return rec && typeof rec.exp === 'number' && rec.exp > Date.now() ? new Date(rec.exp) : null;
  } catch {
    return null;
  }
}
