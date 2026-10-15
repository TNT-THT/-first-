const TOKEN_KEY = 'oj-ai.token';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || '';
  } catch {
    return '';
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* 某些嵌入环境禁用了存储，此时只能保持当前会话内有效 */
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

export async function login(password) {
  const res = await fetch('/api/login', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ password }),
  });
  if (!res.ok) throw new ApiError(res.status, await readError(res));
  const data = await res.json();
  setToken(data.token);
  return data;
}

export async function checkSession() {
  const token = getToken();
  if (!token) return false;
  const res = await fetch('/api/session', { headers: { authorization: `Bearer ${token}` } });
  if (!res.ok) {
    setToken('');
    return false;
  }
  return true;
}

/**
 * 以流式方式请求聊天接口，逐段产出模型生成的文本。
 * 服务端透传 OpenAI 兼容的 SSE 格式：每行 "data: {...}"，结束标记 "data: [DONE]"。
 */
export async function* streamChat({ messages, signal }) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      accept: 'text/event-stream',
      authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify({ messages }),
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
