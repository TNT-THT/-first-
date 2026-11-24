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

/**
 * 账号令牌：u1.<账号id>.<过期时间戳ms>.<签名>，签名覆盖账号 id 与过期时间。
 * 与上面的匿名会话令牌分开命名空间，互不能冒充。
 */
export function createUserToken(secret, userId, ttlMs, now = Date.now()) {
  const exp = String(now + ttlMs);
  const sig = createHmac('sha256', secret).update(`u1.${userId}.${exp}`).digest('hex');
  return { token: `u1.${userId}.${exp}.${sig}`, expiresAt: Number(exp) };
}

/** 校验账号令牌，成功返回账号 id，失败返回 null。 */
export function verifyUserToken(secret, token, now = Date.now()) {
  if (typeof token !== 'string') return null;
  const parts = token.split('.');
  if (parts.length !== 4 || parts[0] !== 'u1') return null;
  const [, userId, exp, sig] = parts;
  if (!userId || !/^\d+$/.test(exp)) return null;
  if (Number(exp) <= now) return null;
  const expected = createHmac('sha256', secret).update(`u1.${userId}.${exp}`).digest('hex');
  return safeEqual(sig, expected) ? userId : null;
}

/**
 * 从请求中提取 Bearer 令牌：优先 Authorization: Bearer <token>，
 * 其次 x-api-key 头（方便不方便设置 Authorization 的调用方）。
 */
export function extractBearer(req) {
  const header = req.get?.('authorization') || req.headers?.authorization || '';
  const bearer = typeof header === 'string' ? header.match(/^Bearer\s+(.+)$/i) : null;
  if (bearer) return bearer[1].trim();
  const apiKey = req.get?.('x-api-key') || req.headers?.['x-api-key'] || '';
  return typeof apiKey === 'string' ? apiKey.trim() : '';
}

/**
 * 校验外部 API Key（模式一：自建服务器 API）。
 * expectedKeys 为服务端配置的 API_KEYS 列表（可多个，逗号分隔配置）。
 * 列表为空时一律返回 false，调用方应改用网页登录的会话令牌。
 * 比较先哈希成定长摘要，避免泄露长度信息；逐个比较时不短路泄露命中位置。
 */
export function checkApiKey(expectedKeys, input) {
  if (typeof input !== 'string' || !input) return false;
  const keys = Array.isArray(expectedKeys) ? expectedKeys : expectedKeys ? [expectedKeys] : [];
  if (keys.length === 0) return false;
  const digest = sha256(input);
  let ok = false;
  for (const key of keys) {
    if (typeof key !== 'string' || !key) continue;
    if (safeEqual(digest, sha256(key))) ok = true;
  }
  return ok;
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
