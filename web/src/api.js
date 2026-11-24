const TOKEN_KEY = 'oj-ai.token';
// 第三方 iframe 中浏览器可能禁用 localStorage；保留页面生命周期内的内存令牌作为回退。
let memoryToken = '';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export function getToken() {
  try {
    const stored = localStorage.getItem(TOKEN_KEY) || '';
    if (stored) {
      memoryToken = stored;
      return stored;
    }
  } catch {
    /* 某些第三方 iframe 环境会禁用 localStorage，回退到页面内存令牌 */
  }
  return memoryToken;
}

export function setToken(token) {
  memoryToken = typeof token === 'string' ? token : '';
  try {
    if (memoryToken) localStorage.setItem(TOKEN_KEY, memoryToken);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* 某些嵌入环境禁用了存储；内存令牌只在当前页面生命周期内有效 */
  }
}

async function readError(res) {
  try {
    const data = await res.json();
    return data.error || `请求失败（HTTP ${res.status}）`;
  } catch {
    return `请求失败（HTTP ${res.status}）`;
  }
}

export async function fetchConfig() {
  const res = await fetch('/api/config', { cache: 'no-store' });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  return res.json();
}

/** 账号注册（开放自助注册）。成功后保存账号令牌并返回账号信息。 */
export async function accountRegister(name, password) {
  const res = await fetch('/api/account/register', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, password }),
  });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  const data = await res.json();
  setToken(data.token);
  return data;
}

/** 账号登录。成功后保存账号令牌并返回账号信息。 */
export async function accountLogin(name, password) {
  const res = await fetch('/api/account/login', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name, password }),
  });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  const data = await res.json();
  setToken(data.token);
  return data;
}

/** 校验当前账号令牌；有效返回 true 并续用，无效则清掉本地令牌。 */
export async function checkSession() {
  const token = getToken();
  if (!token) return false;
  const res = await fetch('/api/account/me', { headers: { authorization: `Bearer ${token}` } });
  if (!res.ok) {
    setToken('');
    return false;
  }
  return true;
}

/** 拉取本账号存在服务器端的数据（对话记录与参数设定）。未登录返回 null。 */
export async function fetchAccountData() {
  const token = getToken();
  if (!token) return null;
  const res = await fetch('/api/account/data', { headers: { authorization: `Bearer ${token}` } });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  return res.json();
}

/** 把本账号数据整体保存到服务器端（对话记录与参数设定按账号隔离）。 */
export async function putAccountData(payload) {
  const token = getToken();
  if (!token) return null;
  const res = await fetch('/api/account/data', {
    method: 'PUT',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` },
    body: JSON.stringify({ payload }),
  });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  return res.json();
}

/**
 * 以流式方式请求聊天接口，逐段产出模型生成的文本。
 * 服务端透传 OpenAI 兼容的 SSE 格式：每行 "data: {...}"，结束标记 "data: [DONE]"。
 */
export async function* streamChat({ messages, signal, model, thinking }) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      accept: 'text/event-stream',
      ...(getToken() ? { authorization: `Bearer ${getToken()}` } : {}),
    },
    // model 可选：本机桥接模式下网页用它指定本条请求使用的模型；不传时服务端用默认（第一个）模型
    // thinking：思考模式开关（false 时服务端会明确关掉模型思考；不传沿用模型默认）
    body: JSON.stringify({
      messages,
      ...(model ? { model } : {}),
      ...(thinking === undefined ? {} : { thinking: !!thinking }),
    }),
    signal,
  });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  if (!res.body) throw new ApiError(500, '浏览器不支持流式读取');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  // 解析一行 SSE；返回 {delta} | {done:true}
  const parseLine = (raw) => {
    const line = raw.trimEnd();
    if (!line.startsWith('data:')) return null;
    const data = line.slice(5).trim();
    if (!data) return null;
    if (data === '[DONE]') return { done: true };
    let json;
    try {
      json = JSON.parse(data);
    } catch {
      return null;
    }
    if (json.error) throw new ApiError(502, json.error.message || '模型返回错误');
    const delta = json.choices?.[0]?.delta?.content;
    return delta ? { delta } : null;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) !== -1) {
      const line = buf.slice(0, nl);
      buf = buf.slice(nl + 1);
      const item = parseLine(line);
      if (!item) continue;
      if (item.done) return;
      yield item.delta;
    }
  }
  buf += decoder.decode();
  if (buf) {
    const item = parseLine(buf);
    if (item && !item.done) yield item.delta;
  }
}
