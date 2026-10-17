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

/** 每条消息的固定开销（角色标记等），用于 token 预算。 */
const MESSAGE_OVERHEAD_TOKENS = 4;
const CJK_RE = /[\u2e80-\u9fff\uac00-\ud7af\u3000-\u303f\uff00-\uffef]/g;

/**
 * 粗略估算文本的 token 数（不依赖具体分词器，偏保守，宁可多估）：
 * 中日韩字符按约 1.5 token 计，其余约 3 个字符计 1 token。
 * 估算只用于预算，不影响模型实际的分词结果。
 */
export function estimateTokens(text) {
  const s = String(text || '');
  const cjk = (s.match(CJK_RE) || []).length;
  return Math.ceil(cjk * 1.5) + Math.ceil((s.length - cjk) / 3);
}

/** 预留给分词误差与模板标记的余量。 */
export const CONTEXT_SAFETY_TOKENS = 512;

/**
 * 历史消息可用的 token 预算 = 窗口 − 输出预留 − 系统提示词 − 余量。
 * 输出预留取 maxOutputTokens，但不超过窗口的一半，保证小窗口（显存不足降级后）仍有可用空间。
 */
export function historyTokenBudget({ windowTokens, maxOutputTokens = 0, reservedTokens = 0 }) {
  const reserve = Math.min(Math.max(0, maxOutputTokens), Math.floor(windowTokens / 2));
  return Math.max(256, windowTokens - reserve - reservedTokens - CONTEXT_SAFETY_TOKENS);
}

/**
 * 构造发送给模型的上下文（多轮记忆）。
 * 从最新消息往前取，直到超过条数或预算；保证第一条一定是用户消息，且最新一条一定保留。
 * - 按字符预算（maxChars，默认）：与原来的行为完全一致，用于远程模式。
 * - 按 token 预算（tokenBudget）：用于本机模型。起点按 stableBlock 条对齐，
 *   使历史前缀在多轮对话中保持不变，WebLLM 即可复用已有的 KV 缓存，只预填新消息；
 *   只有超出预算时起点才会前移。
 * currentModel：本次回答所用的模型 id，用于判断历史回复是否来自其他 AI。
 */
export function buildContext(
  messages,
  { maxMessages = 20, maxChars = 40000, tokenBudget = null, stableBlock = 1, currentModel = null } = {},
) {
  const useTokens = Number.isFinite(tokenBudget) && tokenBudget > 0;
  const budget = useTokens ? tokenBudget : maxChars;
  const eligible = [];
  for (const m of messages) {
    if (m.error || !m.content) continue;
    const content = attributeContent(m, currentModel);
    const cost = useTokens ? estimateTokens(content) + MESSAGE_OVERHEAD_TOKENS : content.length;
    eligible.push({ role: m.role, content, cost });
  }

  // 1. 从最新往前取，直到超过条数或预算（至少保留最新一条）
  let start = eligible.length;
  let total = 0;
  for (let i = eligible.length - 1; i >= 0 && eligible.length - i <= maxMessages; i--) {
    if (total + eligible[i].cost > budget && start < eligible.length) break;
    total += eligible[i].cost;
    start = i;
  }

  // 2. token 模式：起点向后对齐到 stableBlock 的倍数，保证历史前缀稳定（不超过最新一条）
  if (useTokens && stableBlock > 1 && start > 0 && start < eligible.length) {
    start = Math.min(eligible.length - 1, Math.ceil(start / stableBlock) * stableBlock);
  }

  // 3. 第一条必须是用户消息
  while (start < eligible.length && eligible[start].role !== 'user') start++;
  return eligible.slice(start).map(({ role, content }) => ({ role, content }));
}
