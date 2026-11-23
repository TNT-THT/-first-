/**
 * 本机大模型（llama.cpp 原生引擎）的网页端接口。
 *
 * 只有带服务端的部署（网站包、独立软件）才有 /api/native/*；纯静态网页版
 * 请求会 404，这里统一吞掉错误返回 null，界面据此隐藏大模型功能。
 */

/** 读取引擎状态与模型目录；不可用时返回 null。 */
export async function fetchNativeStatus() {
  try {
    const res = await fetch('/api/native/status', { cache: 'no-store' });
    if (!res.ok) return null;
    const data = await res.json();
    return data?.enabled ? data : null;
  } catch {
    return null;
  }
}

/** 管理员接口（下载/加载/卸载等）：带管理员令牌，失败抛 Error(含服务端消息)。 */
export async function nativeAdminRequest(path, { token, method = 'POST', body } = {}) {
  const headers = { 'content-type': 'application/json' };
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `请求失败（HTTP ${res.status}）`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/** HF 搜索（需网站登录令牌）：关键词或仓库链接都可以。 */
export async function hfSearch(query, token) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(`/api/native/hf/search?q=${encodeURIComponent(query || '')}`, { headers, cache: 'no-store' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `搜索失败（HTTP ${res.status}）`);
  return data;
}

/** HF 仓库详情（格式与运行路线）。 */
export async function hfModelInfo(repo, token) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(`/api/native/hf/model?repo=${encodeURIComponent(repo)}`, { headers, cache: 'no-store' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `读取仓库失败（HTTP ${res.status}）`);
  return data;
}

/** 某模型在仓库里可下载的量化列表（需网站登录令牌由 api.js 的 getToken 提供）。 */
export async function fetchNativeQuants(id, token) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(`/api/native/models/${encodeURIComponent(id)}/quants`, { headers, cache: 'no-store' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  return data.quants || [];
}
