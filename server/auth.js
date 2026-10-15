import { createHash, createHmac, timingSafeEqual } from 'node:crypto';

/**
 * 无状态令牌：v1.<过期时间戳ms>.<HMAC-SHA256 签名>
 * 使用 Authorization 请求头传递，而不是 Cookie。
 * 原因：本应用会被嵌入到其他网站的 iframe 中，第三方 Cookie 在很多浏览器里会被拦截。
 */
export function createToken(secret, ttlMs, now = Date.now()) {
  const exp = String(now + ttlMs);
  const sig = sign(secret, exp);
  return { token: `v1.${exp}.${sig}`, expiresAt: Number(exp) };
}

export function verifyToken(secret, token, now = Date.now()) {
  if (typeof token !== 'string') return false;
  const parts = token.split('.');
  if (parts.length !== 3 || parts[0] !== 'v1') return false;
  const [, exp, sig] = parts;
  if (!/^\d+$/.test(exp)) return false;
  if (Number(exp) <= now) return false;
  return safeEqual(sig, sign(secret, exp));
}

export function checkPassword(expected, input) {
  if (typeof input !== 'string' || !expected) return false;
  // 先哈希成定长摘要，避免泄露长度信息
  return safeEqual(sha256(expected), sha256(input));
}

function sign(secret, payload) {
  return createHmac('sha256', secret).update(`v1.${payload}`).digest('hex');
}

function sha256(s) {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

function safeEqual(a, b) {
  const ba = Buffer.from(String(a));
  const bb = Buffer.from(String(b));
  if (ba.length !== bb.length) return false;
  return timingSafeEqual(ba, bb);
}
