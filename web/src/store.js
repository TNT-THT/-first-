/**
 * 会话历史保存在浏览器 localStorage 中（只存在用户自己的浏览器里，服务端不保存聊天记录）。
 */
const KEY = 'oj-ai.sessions.v1';
export const MAX_SESSIONS = 50;
export const MAX_MESSAGES_PER_SESSION = 200;

function emptyStore() {
  return { activeId: null, sessions: [] };
}

export function loadStore() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return emptyStore();
    const data = JSON.parse(raw);
    if (!data || !Array.isArray(data.sessions)) return emptyStore();
    return data;
  } catch {
    return emptyStore();
  }
}

export function saveStore(store) {
  try {
    localStorage.setItem(KEY, JSON.stringify(store));
  } catch {
    /* 存储空间不足或被禁用时忽略，不影响当前对话 */
  }
}

export function newId() {
  if (globalThis.crypto?.randomUUID) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function createSession(store) {
  const now = Date.now();
  const session = { id: newId(), title: '新会话', createdAt: now, updatedAt: now, messages: [] };
  store.sessions.unshift(session);
  store.sessions = store.sessions.slice(0, MAX_SESSIONS);
  store.activeId = session.id;
  return session;
}

export function getSession(store, id = store.activeId) {
  return store.sessions.find((s) => s.id === id) || null;
}

export function deleteSession(store, id) {
  store.sessions = store.sessions.filter((s) => s.id !== id);
  if (store.activeId === id) store.activeId = store.sessions[0]?.id || null;
}

export function titleFrom(text) {
  const one = text.replace(/\s+/g, ' ').trim();
  return one.length > 24 ? `${one.slice(0, 24)}…` : one || '新会话';
}

/**
 * 构造发送给模型的上下文（多轮记忆）。
 * 从最新消息往前取，直到超过条数或字符预算；保证第一条一定是用户消息。
 */
export function buildContext(messages, { maxMessages = 20, maxChars = 40000 } = {}) {
  const picked = [];
  let total = 0;
  for (let i = messages.length - 1; i >= 0 && picked.length < maxMessages; i--) {
    const m = messages[i];
    if (m.error || !m.content) continue;
    if (total + m.content.length > maxChars && picked.length > 0) break;
    total += m.content.length;
    picked.unshift({ role: m.role, content: m.content });
  }
  while (picked.length && picked[0].role !== 'user') picked.shift();
  return picked;
}
