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
 * 其他 AI 给出的回复，在发给当前 AI 时加上来源标记，使其能分清“谁说过什么”，
 * 从而切换模型后仍能顺畅接续对话。当前模型自己的回复和旧版本没有标记的回复保持原样。
 */
export function attributeContent(m, currentModel = null) {
  if (m.role !== 'assistant' || !m.model || m.model === currentModel) return m.content;
  const name = m.modelLabel || m.model;
  return `【以下回复由其他AI「${name}」给出】\n${m.content}`;
}

/**
 * 构造发送给模型的上下文（多轮记忆）。
 * 从最新消息往前取，直到超过条数或字符预算；保证第一条一定是用户消息。
 * currentModel：本次回答所用的模型 id，用于判断历史回复是否来自其他 AI。
 */
/**
 * 构造发送给模型的上下文（多轮记忆）。
 * 规则：固定提示、错误与空消息不发送；最多 maxMessages 条、总字符不超过 maxChars；
 * 超出时从最早的消息开始丢弃，且保证第一条一定是用户消息。
 *
 * chunk（丢弃粒度）：起点只会落在 chunk 的倍数上，因此超出限制后一次丢掉一段，之后的几轮起点不变，
 * WebLLM 可以复用已经计算过的 KV 缓存（不必整段重新预填）。chunk 为 1 时与逐条丢弃完全相同。
 * currentModel：本次回答所用的模型 id，用于判断历史回复是否来自其他 AI。
 */
export function buildContext(
  messages,
  { maxMessages = 20, maxChars = 40000, currentModel = null, chunk = 1 } = {},
) {
  const eligible = messages.filter((m) => !(m.error || !m.content || m.fixed));
  const contents = eligible.map((m) => attributeContent(m, currentModel));
  const n = contents.length;
  if (!n) return [];
  // suffix[s] = 从 s 开始到末尾的字符数
  const suffix = new Array(n + 1).fill(0);
  for (let i = n - 1; i >= 0; i -= 1) suffix[i] = suffix[i + 1] + contents[i].length;
  const fits = (s) => n - s <= maxMessages && suffix[s] <= maxChars;
  const firstFit = (step) => {
    for (let s = 0; s < n; s += step) if (fits(s)) return s;
    return -1;
  };
  const step = Math.max(1, Math.floor(chunk));
  let start = firstFit(step);
  if (start < 0) start = firstFit(1); // 对齐的起点都放不下时，退回到逐条
  if (start < 0) start = n - 1; // 连最后一条都超出字数限制：仍保留它（与原行为一致）
  const picked = [];
  for (let i = start; i < n; i += 1) picked.push({ role: eligible[i].role, content: contents[i] });
  while (picked.length && picked[0].role !== 'user') picked.shift();
  return picked;
}

/** 打开历史会话时先渲染的最近消息条数（较早的消息折叠，点击再展开）。 */
export const RECENT_RENDER_COUNT = 20;
/** 每次展开较早消息的条数。 */
export const UNFOLD_STEP = 20;

/** 初始可见窗口的起点：只渲染最近 count 条。 */
export function initialVisibleStart(total, count = RECENT_RENDER_COUNT) {
  return Math.max(0, total - count);
}

/** 展开更早的消息后，可见窗口的新起点（不小于 0）。 */
export function expandVisibleStart(start, step = UNFOLD_STEP) {
  return Math.max(0, start - step);
}
