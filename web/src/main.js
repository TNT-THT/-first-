import 'katex/dist/katex.min.css';
import './style.css';
import DOMPurify from 'dompurify';
import { renderMarkdown } from './markdown.js';
import { hasThinking, stripThinking } from './think.js';
import { isAllowedOrigin } from './origin.js';
import { ApiError, checkSession, fetchConfig, getToken, login, setToken } from './api.js';
import { createBackend } from './engines/backend.js';
import { checkWebGPU, getModelRecords, WEBGPU_MISSING } from './engines/webllm.js';
import {
  buildModelFile,
  importModelFile,
  listCachedModelIds,
  readModelFileHeader,
  removeLocalModelFiles,
} from './model-file.js';
import { adminExpiry, grantAdmin, hasAccessGate, isAdmin, revokeAdmin, verifyAccessPassword } from './access.js';
import { addLocalModel, loadLocalModels, localRecords, removeLocalModel } from './local-models.js';
import {
  adaptedModelIds,
  applyParamSuggestion,
  clearModelAdaptation,
  getModelParams,
  listAdaptation,
  listParamLog,
  recordFeedback,
  removeNote,
  resetParam,
  suggestParams,
} from './adaptation.js';
import {
  buildContext,
  createSession,
  deleteSession,
  getSession,
  loadStore,
  MAX_MESSAGES_PER_SESSION,
  saveStore,
  titleFrom,
} from './store.js';

// 构建模式：默认 'webllm'（纯静态，浏览器内推理）；VITE_ENGINE=remote 时使用自托管服务端
const MODE = import.meta.env.VITE_ENGINE === 'remote' ? 'remote' : 'webllm';

const $ = (sel) => document.querySelector(sel);
const els = {
  body: document.body,
  login: $('#login'),
  loginForm: $('#loginForm'),
  loginError: $('#loginError'),
  loginBtn: $('#loginBtn'),
  password: $('#password'),
  main: $('#main'),
  brandName: $('#brandName'),
  loginTitle: $('#loginTitle'),
  modelSelect: $('#modelSelect'),
  modelLabel: $('#modelLabel'),
  newChatBtn: $('#newChatBtn'),
  embedNewBtn: $('#embedNewBtn'),
  clearAllBtn: $('#clearAllBtn'),
  historyBtn: $('#historyBtn'),
  historyPanel: $('#historyPanel'),
  chatTitle: $('#chatTitle'),
  messages: $('#messages'),
  composer: $('#composer'),
  input: $('#input'),
  sendBtn: $('#sendBtn'),
  stopBtn: $('#stopBtn'),
  statusBar: $('#statusBar'),
  statusText: $('#statusText'),
  statusProgress: $('#statusProgress'),
  statusBarFill: $('#statusBarFill'),
  exportModelBtn: $('#exportModelBtn'),
  importModelBtn: $('#importModelBtn'),
  importModelInput: $('#importModelInput'),
  importTrainBtn: $('#importTrainBtn'),
  importTrainInput: $('#importTrainInput'),
  roleBtn: $('#roleBtn'),
  roleBadge: $('#roleBadge'),
  rolePanel: $('#rolePanel'),
  roleStatus: $('#roleStatus'),
  roleForm: $('#roleForm'),
  adminKey: $('#adminKey'),
  roleSubmit: $('#roleSubmit'),
  roleError: $('#roleError'),
  roleLogoutBtn: $('#roleLogoutBtn'),
  modelAdminHint: $('#modelAdminHint'),
  localModelList: $('#localModelList'),
  adaptHint: $('#adaptHint'),
  adaptSuggest: $('#adaptSuggest'),
  adaptList: $('#adaptList'),
};

// ---------- 运行参数 ----------
const params = new URLSearchParams(location.search);
const isEmbed = ['embed', 'hideSidebar'].some((k) => {
  const v = params.get(k);
  return v !== null && v !== '0' && v !== 'false';
});
const isEmbedded = window.parent !== window;

const state = {
  store: loadStore(),
  appConfig: { appName: 'OJ 刷题助手', allowedParentOrigins: [], access: {} },
  backend: null,
  streaming: null, // { controller, session, assistant }
  clearArmed: false,
};

// ---------- 工具函数 ----------
function scrollToBottom(force = false) {
  const m = els.messages;
  const nearBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 120;
  if (force || nearBottom) m.scrollTop = m.scrollHeight;
}

function persist() {
  saveStore(state.store);
}

function activeSession() {
  return getSession(state.store);
}

function ensureSession() {
  return activeSession() || createSession(state.store);
}

function sanitize(html) {
  return DOMPurify.sanitize(html, { ADD_ATTR: ['target'] });
}

function friendlyError(err, progressText = '') {
  if (err instanceof ApiError) return err.message;
  const m = err?.message || String(err);
  if (/context window|exceed/i.test(m)) {
    return '对话内容超过了模型的上下文长度。请新建会话，或精简问题后再发送。';
  }
  if (m.startsWith('当前浏览器')) return m;
  if (/Failed to fetch|NetworkError|network/i.test(m)) {
    const detail = [progressText && `失败时的进度：${progressText}`, `错误信息：${m}`].filter(Boolean).join('；');
    return (
      '模型文件下载失败：无法访问 Hugging Face / GitHub。请检查网络；如使用代理工具，请确认代理已开启且允许访问 huggingface.co，然后重试。\n\n' +
      '也可以在能正常访问的电脑上用“导出模型”保存文件，再到这里点“导入模型”。\n\n' +
      `（${detail}）`
    );
  }
  return `出错了：${m}`;
}

// ---------- 状态栏（行内提示，不使用弹窗） ----------
function showStatus(text, { progress = null, error = false } = {}) {
  els.statusBar.hidden = false;
  els.statusBar.classList.toggle('error', error);
  els.statusText.textContent = text;
  if (progress === null) {
    els.statusProgress.hidden = true;
  } else {
    els.statusProgress.hidden = false;
    els.statusBarFill.style.width = `${Math.round(progress * 100)}%`;
  }
}

function hideStatus() {
  els.statusBar.hidden = true;
  els.statusBar.classList.remove('error');
}

async function refreshIdleStatus() {
  if (state.backend.mode !== 'webllm') return hideStatus();
  if (!(await checkWebGPU())) {
    return showStatus(WEBGPU_MISSING, { error: true });
  }
  if (!state.backend.isReady()) {
    showStatus('首次发送时会下载并加载模型（体积较大，之后由浏览器缓存）。');
  } else {
    hideStatus();
  }
}

// ---------- 渲染：消息 ----------
function typingHtml(label) {
  return `<span class="typing" aria-label="正在生成"><i></i><i></i><i></i></span><span class="typing-text">${label}</span>`;
}

function paintAssistant(bubble, msg, { streaming = false, label = '正在思考…' } = {}) {
  bubble.classList.toggle('error', !!msg.error);
  bubble.classList.toggle('streaming', streaming);
  // 推理模型的思考过程不展示，只显示最终回答
  const visible = stripThinking(msg.content);
  if (!visible && streaming) {
    bubble.innerHTML = typingHtml(label);
    return;
  }
  let html = sanitize(renderMarkdown(visible));
  if (msg.stopped) html += '<p class="note">（已停止生成）</p>';
  bubble.innerHTML = html;
}

function createMessageEl(msg, { streaming = false } = {}) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${msg.role}`;
  if (msg.role === 'assistant' && msg.modelLabel) {
    // 标明回答的 AI，便于区分不同模型的回复
    const who = document.createElement('div');
    who.className = 'msg-model';
    who.textContent = msg.modelLabel;
    wrap.appendChild(who);
  }
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  if (msg.role === 'user') {
    bubble.textContent = msg.content;
  } else {
    paintAssistant(bubble, msg, { streaming });
  }
  wrap.appendChild(bubble);
  return { wrap, bubble };
}

/** 回复结束后，在该回复下方挂一个“重新生成”按钮。 */
function attachRegen(afterEl) {
  const bar = document.createElement('div');
  bar.className = 'msg-actions';
  bar.innerHTML = '<button type="button" class="btn ghost small" data-action="regen">重新生成</button>';
  afterEl.after(bar);
}

function renderMessages() {
  const session = activeSession();
  els.messages.replaceChildren();
  if (!session || session.messages.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.innerHTML =
      '<p class="empty-title">把题目交给我</p>' +
      '<p class="muted">可以粘贴题面、报错信息或你的代码，我会给出思路、复杂度分析和示例代码。</p>';
    els.messages.appendChild(empty);
  } else {
    for (const m of session.messages) els.messages.appendChild(createMessageEl(m).wrap);
  }
  scrollToBottom(true);
}

function renderSessionList() {
  const sessions = state.store.sessions;
  document.querySelectorAll('[data-session-list]').forEach((container) => {
    container.replaceChildren();
    if (sessions.length === 0) {
      const p = document.createElement('p');
      p.className = 'muted small pad';
      p.textContent = '暂无历史会话';
      container.appendChild(p);
      return;
    }
    for (const s of sessions) {
      const row = document.createElement('div');
      row.className = `session-item${s.id === state.store.activeId ? ' active' : ''}`;
      const pick = document.createElement('button');
      pick.type = 'button';
      pick.className = 'session-pick';
      pick.textContent = s.title;
      pick.title = s.title;
      pick.dataset.action = 'switch';
      pick.dataset.id = s.id;
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'session-del';
      del.textContent = '×';
      del.title = '删除会话';
      del.setAttribute('aria-label', '删除会话');
      del.dataset.action = 'delete';
      del.dataset.id = s.id;
      row.append(pick, del);
      container.appendChild(row);
    }
  });
  const session = activeSession();
  els.chatTitle.textContent = session ? session.title : '新会话';
}

function refreshAll() {
  renderSessionList();
  renderMessages();
}

// ---------- 发送 / 停止 ----------
function setBusy(busy) {
  els.sendBtn.hidden = busy;
  els.stopBtn.hidden = !busy;
  // 生成期间不能切换模型，否则回答的标注与实际生成的模型可能不一致
  els.modelSelect.disabled = busy;
}

/** 当前回答 AI 的显示名称（不含 ✓ 等缓存标记）。 */
function modelDisplayName(id) {
  if (MODE !== 'webllm') return state.appConfig.model || '服务端模型';
  const opt = [...els.modelSelect.options].find((o) => o.value === id);
  return opt?.dataset.label || state.backend.models.find((m) => m.id === id)?.label || id;
}

/**
 * 管理员（且为本机模式）在某个 AI 的回复之后发送的消息，视为对该 AI 的反馈：
 * 仅保存在本机，并在之后的回答中作为该 AI 的附加说明（参数行可调整该 AI 的采样参数）。
 * 反馈仍会照常发送给当前 AI，不影响对话。
 */
function recordAdminFeedback(session, text) {
  if (MODE !== 'webllm' || !isAdmin()) return;
  const last = [...session.messages].reverse().find((m) => m.role === 'assistant');
  if (!last || !last.model || last.error || !last.content) return;
  const rec = recordFeedback(last.model, text);
  if (!rec) return;
  const parts = [];
  if (rec.note) parts.push('已记录反馈');
  if (rec.params) {
    const changed = (rec.changes || []).map((c) => `${c.key} ${c.from ?? '默认'}→${c.to}`);
    parts.push('已更新参数 ' + (changed.length ? changed.join(', ') : '（未变化）'));
  }
  // 没有显式参数行时，按关键词给出“建议参数”，必须由管理员确认后才生效
  let suggested = 0;
  if (!rec.params) {
    const found = suggestParams(text, effectiveParams(last.model));
    suggested = addSuggestions(last.model, last.modelLabel || modelDisplayName(last.model), found);
    if (suggested) parts.push(`有 ${suggested} 条参数建议，请在输入框上方确认（确认前不生效）`);
  }
  if (parts.length) {
    showStatus(`${parts.join('，')}（仅本机，对「${modelDisplayName(last.model)}」生效，不上传）。`);
  }
  renderAdaptList();
}

/** 待管理员确认的参数建议（仅内存中保存；刷新页面或退出管理员后清空）。 */
let pendingSuggestions = [];

/** 某模型当前生效的参数：本机覆盖优先，否则使用配置默认值。 */
function effectiveParams(modelId) {
  const d = state.appConfig.webllm;
  return { temperature: d.temperature, maxTokens: d.maxTokens, ...getModelParams(modelId) };
}

function addSuggestions(modelId, modelLabel, found) {
  for (const f of found) {
    pendingSuggestions = pendingSuggestions.filter((x) => !(x.modelId === modelId && x.key === f.key));
    pendingSuggestions.push({ id: `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, modelId, modelLabel, ...f });
  }
  pendingSuggestions = pendingSuggestions.slice(-6);
  renderSuggestions();
  return found.length;
}

function dropPendingSuggestions(modelId) {
  pendingSuggestions = pendingSuggestions.filter((x) => x.modelId !== modelId);
  renderSuggestions();
}

/** 输入框上方的建议卡片：仅管理员可见。 */
function renderSuggestions() {
  const box = els.adaptSuggest;
  if (!box) return;
  box.replaceChildren();
  const items = isAdmin() && MODE === 'webllm' ? pendingSuggestions : [];
  box.hidden = items.length === 0;
  for (const s of items) {
    const row = document.createElement('div');
    row.className = 'adapt-suggest-row';
    const text = document.createElement('span');
    const name = s.key === 'maxTokens' ? 'maxTokens（最大输出长度）' : 'temperature（温度）';
    text.textContent = `检测到你${s.reason}，是否将「${s.modelLabel}」的 ${name} 设为 ${s.value}？`;
    const ok = document.createElement('button');
    ok.type = 'button';
    ok.className = 'btn primary small';
    ok.dataset.action = 'suggest-accept';
    ok.dataset.id = s.id;
    ok.textContent = '确认';
    const no = document.createElement('button');
    no.type = 'button';
    no.className = 'btn ghost small';
    no.dataset.action = 'suggest-dismiss';
    no.dataset.id = s.id;
    no.textContent = '忽略';
    row.append(text, ok, no);
    box.append(row);
  }
}

function acceptSuggestion(id) {
  if (!requireAdmin('确认参数建议')) return;
  const s = pendingSuggestions.find((x) => x.id === id);
  if (!s) return;
  const value = applyParamSuggestion(s.modelId, s.key, s.value, { reason: s.reason });
  pendingSuggestions = pendingSuggestions.filter((x) => x.id !== id);
  renderSuggestions();
  renderAdaptList();
  if (value === null) return showStatus('参数建议无效，未生效。', { error: true });
  showStatus(`已确认：「${s.modelLabel}」的 ${s.key} = ${value}（仅本机，从下一次回答起生效）。`);
}

function dismissSuggestion(id) {
  pendingSuggestions = pendingSuggestions.filter((x) => x.id !== id);
  renderSuggestions();
}

async function send(rawText) {
  const text = (rawText || '').trim();
  if (!text || state.streaming) return;

  const session = ensureSession();
  const isFirst = session.messages.length === 0;
  const userMsg = { role: 'user', content: text, ts: Date.now() };
  recordAdminFeedback(session, text);
  // 记录本次回答所用的 AI（名称快照保存，之后删除或改名模型也能正确标注）
  const modelId = state.backend.modelId;
  const modelLabel = modelDisplayName(modelId);
  const assistantMsg = { role: 'assistant', content: '', ts: Date.now(), model: modelId, modelLabel };
  session.messages.push(userMsg, assistantMsg);
  if (session.messages.length > MAX_MESSAGES_PER_SESSION) {
    session.messages.splice(0, session.messages.length - MAX_MESSAGES_PER_SESSION);
  }
  if (isFirst) session.title = titleFrom(text);
  session.updatedAt = Date.now();
  persist();

  // 上下文：不包含本次刚添加的空占位回复
  const context = buildContext(session.messages.slice(0, -1), { ...state.backend.limits, currentModel: modelId });

  const controller = new AbortController();
  state.streaming = { controller, session, assistant: assistantMsg };
  setBusy(true);

  // 只追加新节点，只更新最后一条助手消息的气泡，避免整页重绘
  els.messages.querySelector('.empty')?.remove();
  els.messages.appendChild(createMessageEl(userMsg).wrap);
  const { wrap: assistantWrap, bubble } = createMessageEl(assistantMsg, { streaming: true });
  els.messages.appendChild(assistantWrap);
  renderSessionList();
  scrollToBottom(true);

  let frame = 0;
  let progressText = '';
  const scheduleRender = () => {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      paintAssistant(bubble, assistantMsg, { streaming: true });
      scrollToBottom();
    });
  };

  try {
    if (!state.backend.isReady()) {
      paintAssistant(bubble, assistantMsg, { streaming: true, label: '正在加载模型…' });
      await state.backend.ensureReady(({ progress, text: t }) => {
        if (t) progressText = t;
        const pct = Math.round(progress * 100);
        showStatus(`正在加载模型 ${pct}%${t ? `：${t}` : ''}`, { progress });
      });
      refreshIdleStatus();
      refreshModelCacheStatus();
    }

    for await (const delta of state.backend.stream({ messages: context, signal: controller.signal })) {
      assistantMsg.content += delta;
      scheduleRender();
    }
    if (frame) cancelAnimationFrame(frame);
    if (controller.signal.aborted) assistantMsg.stopped = true;
    const raw = assistantMsg.content;
    assistantMsg.content = stripThinking(raw);
    if (!assistantMsg.content && !assistantMsg.stopped) {
      assistantMsg.content = hasThinking(raw)
        ? '（模型只输出了思考过程，没有给出答案。可以重试，或换一个模型。）'
        : '（模型没有返回内容，请重试）';
      assistantMsg.error = true;
    }
    paintAssistant(bubble, assistantMsg);
  } catch (err) {
    if (frame) cancelAnimationFrame(frame);
    if (err.name === 'AbortError' || controller.signal.aborted) {
      assistantMsg.stopped = true;
      paintAssistant(bubble, assistantMsg);
    } else if (err instanceof ApiError && err.status === 401 && state.backend.mode === 'remote') {
      // 服务端令牌失效：回到登录页。本次问题退回输入框，登录后可直接重新发送。
      setToken('');
      session.messages.splice(-2, 2);
      if (isFirst) session.title = '新会话';
      persist();
      els.input.value = text;
      autoGrow();
      showLogin('登录已过期，请重新输入密码');
      return;
    } else {
      assistantMsg.content = friendlyError(err, progressText);
      assistantMsg.error = true;
      paintAssistant(bubble, assistantMsg);
      if (state.backend.mode === 'webllm') refreshIdleStatus();
    }
  } finally {
    // 历史记录只保存最终回答，不保存思考过程
    assistantMsg.content = stripThinking(assistantMsg.content);
    state.streaming = null;
    setBusy(false);
    session.updatedAt = Date.now();
    persist();
    renderSessionList();
    // 仅当本条回复仍在会话中（未因登录过期被撤回）时才显示“重新生成”
    if (session.messages.includes(assistantMsg)) attachRegen(assistantWrap);
    scrollToBottom();
    els.input.focus({ preventScroll: true });
  }
}

function stop() {
  state.streaming?.controller.abort();
}

// ---------- 会话操作 ----------
function newChat() {
  if (state.streaming) stop();
  const current = activeSession();
  if (current && current.messages.length === 0) {
    state.store.activeId = current.id;
  } else {
    createSession(state.store);
  }
  persist();
  closeHistory();
  refreshAll();
  els.input.focus();
}

function switchSession(id) {
  if (state.streaming) stop();
  state.store.activeId = id;
  persist();
  closeHistory();
  refreshAll();
}

function removeSession(id) {
  if (state.streaming && state.streaming.session.id === id) stop();
  deleteSession(state.store, id);
  if (!state.store.activeId) createSession(state.store);
  persist();
  refreshAll();
}

function clearAll() {
  if (!state.clearArmed) {
    state.clearArmed = true;
    els.clearAllBtn.textContent = '再点一次确认';
    setTimeout(() => {
      state.clearArmed = false;
      els.clearAllBtn.textContent = '清空全部';
    }, 4000);
    return;
  }
  stop();
  state.store = { activeId: null, sessions: [] };
  createSession(state.store);
  persist();
  state.clearArmed = false;
  els.clearAllBtn.textContent = '清空全部';
  refreshAll();
}

function toggleHistory(force) {
  const open = typeof force === 'boolean' ? force : els.historyPanel.hidden;
  els.historyPanel.hidden = !open;
  els.historyBtn.setAttribute('aria-expanded', String(open));
}

function closeHistory() {
  toggleHistory(false);
}

/** 重新生成：撤回最后一轮（用户问题 + 回复），用同一问题重新提问。 */
async function regenerate() {
  if (state.streaming) return;
  const msgs = activeSession()?.messages || [];
  const last = msgs.at(-1);
  const prev = msgs.at(-2);
  if (!last || last.role !== 'assistant' || !prev || prev.role !== 'user') return;
  const text = prev.content;
  msgs.splice(-2, 2);
  persist();
  refreshAll();
  await send(text);
}

// ---------- 引用选中的回复文字 ----------
const QUOTE_MAX_CHARS = 4000;
let quoteEl = null;
let quoteText = '';
let flashTimer = 0;

function hideQuote() {
  quoteEl?.remove();
  quoteEl = null;
  quoteText = '';
}

function showQuote(text, rect) {
  if (!quoteEl) {
    quoteEl = document.createElement('button');
    quoteEl.type = 'button';
    quoteEl.className = 'quote-btn';
    quoteEl.textContent = '引用';
    quoteEl.title = '把选中的文字引用到输入框';
    // 按下时不清除文本选区，保证点击时能读取到选中内容
    quoteEl.addEventListener('mousedown', (e) => e.preventDefault());
    quoteEl.addEventListener('click', applyQuote);
    document.body.appendChild(quoteEl);
  }
  quoteText = text;
  const above = rect.top > 56;
  quoteEl.style.left = `${Math.min(window.innerWidth - 40, Math.max(40, rect.left + rect.width / 2))}px`;
  quoteEl.style.top = `${above ? rect.top - 8 : rect.bottom + 8}px`;
  quoteEl.style.transform = above ? 'translate(-50%, -100%)' : 'translateX(-50%)';
}

/** 只有选中“回复”中的文字时才显示引用按钮。 */
function updateQuote() {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || sel.rangeCount === 0) return hideQuote();
  const range = sel.getRangeAt(0);
  const node = range.commonAncestorContainer;
  const el = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
  if (!el?.closest('.msg.assistant .bubble')) return hideQuote();
  const text = sel.toString().trim();
  if (!text) return hideQuote();
  showQuote(text, range.getBoundingClientRect());
}

/** 把选中的内容以 Markdown 引用格式（> 开头）填入输入框，光标留在末尾，方便继续提问。 */
function applyQuote() {
  if (!quoteText) return hideQuote();
  const truncated = quoteText.length > QUOTE_MAX_CHARS;
  const body = truncated ? quoteText.slice(0, QUOTE_MAX_CHARS) : quoteText;
  const quoted = body
    .split('\n')
    .map((line) => (line.trim() ? `> ${line}` : '>'))
    .join('\n');
  const prev = els.input.value.replace(/\s+$/, '');
  els.input.value = `${prev ? `${prev}\n\n` : ''}${quoted}\n\n`;
  hideQuote();
  window.getSelection()?.removeAllRanges();
  autoGrow();
  els.input.focus();
  els.input.setSelectionRange(els.input.value.length, els.input.value.length);
  if (truncated) flashStatus(`引用内容较长，仅保留前 ${QUOTE_MAX_CHARS} 字。`);
}

/** 短暂提示后恢复为正常的空闲状态（不覆盖加载进度）。 */
function flashStatus(text) {
  if (state.streaming) return;
  showStatus(text);
  clearTimeout(flashTimer);
  flashTimer = setTimeout(refreshIdleStatus, 2500);
}

// ---------- 模型导出 / 导入 ----------
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}

function formatBytes(n) {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${Math.round(n / 1024 ** 2)} MB`;
  return `${Math.max(1, Math.round(n / 1024))} KB`;
}

function optionLabel(id) {
  const opt = [...els.modelSelect.options].find((o) => o.value === id);
  return opt?.dataset.label || id;
}

/** 基础名称（不含“导入/训练结果”后缀），用于导出文件头。 */
function baseLabelOf(id) {
  const local = loadLocalModels().find((e) => e.id === id);
  return local?.baseLabel || optionLabel(id);
}

/** 官方模型（含自定义 extraRecords）+ 本地“新建 AI”的模型记录，用于缓存检查与导出。 */
function getRecords() {
  const cfg = state.appConfig.webllm || {};
  return getModelRecords({
    mirrorOwner: cfg.mirrorOwner,
    extraRecords: [...(cfg.extraRecords || []), ...localRecords(loadLocalModels())],
  });
}

function makeOption(value, label) {
  const opt = document.createElement('option');
  opt.value = value;
  opt.textContent = label;
  opt.dataset.label = label;
  return opt;
}

/** 重建下拉框：官方模型在前，“新建AI”分组在后（仅本浏览器可用，不上传云端）。 */
function renderModelOptions(selectId = state.backend.modelId) {
  const children = state.backend.models.map((m) => makeOption(m.id, m.label));
  const locals = loadLocalModels();
  if (locals.length) {
    const group = document.createElement('optgroup');
    group.label = '新建AI';
    for (const e of locals) group.append(makeOption(e.id, e.label));
    children.push(group);
  }
  els.modelSelect.replaceChildren(...children);
  if ([...els.modelSelect.options].some((o) => o.value === selectId)) {
    els.modelSelect.value = selectId;
  }
}

/** 在下拉框中给已下载到本机的模型加上 ✓。 */
async function refreshModelCacheStatus() {
  if (MODE !== 'webllm' || !state.backend) return;
  try {
    const cached = await listCachedModelIds(await getRecords());
    for (const opt of els.modelSelect.options) {
      const label = opt.dataset.label || opt.value;
      opt.textContent = cached.has(opt.value) ? `✓ ${label}` : label;
    }
  } catch (err) {
    console.warn('无法检查模型缓存状态', err);
  }
}

/** 导出、导入、删除新建 AI 都需要管理员权限。 */
function requireAdmin(action) {
  if (isAdmin()) return true;
  showStatus(`${action}需要管理员权限。请点击右上角“更改权限”，输入管理员密钥。`, { error: true });
  return false;
}

async function exportCurrentModel() {
  if (!requireAdmin('导出模型')) return;
  if (state.streaming) return showStatus('请等待当前回复结束后再导出模型。');
  const modelId = state.backend.modelId;
  const label = optionLabel(modelId);
  els.exportModelBtn.disabled = true;
  try {
    const record = (await getRecords()).find((r) => r.model_id === modelId);
    if (!record) throw new Error(`找不到模型记录：${modelId}`);
    showStatus(`正在打包 ${label}…`, { progress: 0 });
    const { blob, filename, size, entryCount } = await buildModelFile(record, {
      label: baseLabelOf(modelId),
      onProgress: ({ done, total }) =>
        showStatus(`正在打包 ${label}（${done}/${total} 个文件）…`, { progress: done / total }),
    });
    downloadBlob(blob, filename);
    showStatus(`已导出 ${filename}（${formatBytes(size)}，${entryCount} 个文件）。导入后即可离线使用。`);
  } catch (err) {
    showStatus(`导出失败：${err.message}`, { error: true });
  } finally {
    els.exportModelBtn.disabled = !isAdmin();
  }
}

/**
 * 导入 .ojmodel 为“新建AI”。缓存写入本地虚拟地址，不会覆盖官方模型缓存；
 * 登记信息保存在 localStorage（oj-ai.localModels），不上传云端。
 * kind: 'import'（导入模型文件）| 'training'（导入训练结果）
 */
async function importModelFromFile(file, kind = 'import') {
  const action = kind === 'training' ? '导入训练结果' : '导入模型';
  if (!requireAdmin(action)) return;
  if (state.streaming) return showStatus(`请等待当前回复结束后再${action}。`);
  setImportButtons(false);
  try {
    const { header } = await readModelFileHeader(file);
    const baseLabel = header.label || state.backend.models.find((m) => m.id === header.modelId)?.label || header.modelId;
    const label = `${baseLabel}（${kind === 'training' ? '训练结果' : '导入'}）`;
    showStatus(`正在导入 ${label}（${formatBytes(file.size)}）…`, { progress: 0 });
    const { record } = await importModelFile(file, {
      onProgress: ({ done, total }) =>
        showStatus(`正在导入 ${label}（${done}/${total} 个文件）…`, { progress: done / total }),
    });
    addLocalModel({
      id: record.model_id,
      label,
      baseLabel,
      source: kind,
      originId: header.modelId,
      createdAt: new Date().toISOString(),
      record,
    });
    renderModelOptions(record.model_id);
    state.backend.setModel(record.model_id);
    await refreshModelCacheStatus();
    renderLocalList();
    showStatus(`导入完成：${label} 已加入“新建AI”，可离线使用。发送消息时将直接从本机加载。`);
  } catch (err) {
    showStatus(`${action}失败：${err.message}`, { error: true });
  } finally {
    setImportButtons(isAdmin());
    els.importModelInput.value = '';
    els.importTrainInput.value = '';
  }
}

function setImportButtons(enabled) {
  els.importModelBtn.disabled = !enabled;
  els.importTrainBtn.disabled = !enabled;
}

function deleteAdaptNote(modelId, noteId) {
  if (!requireAdmin('删除 AI 适应记录')) return;
  removeNote(modelId, noteId);
  renderAdaptList();
}

function clearAdaptation(modelId) {
  if (!requireAdmin('清空 AI 适应记录')) return;
  if (!window.confirm(`清空「${modelDisplayName(modelId)}」的全部适应记录（包括说明、参数与参数变更记录）？`)) return;
  clearModelAdaptation(modelId);
  dropPendingSuggestions(modelId);
  renderAdaptList();
}

async function deleteLocalModel(id) {
  if (!requireAdmin('删除新建AI')) return;
  const entry = loadLocalModels().find((e) => e.id === id);
  if (!entry) return;
  if (!window.confirm(`删除「${entry.label}」？将清除本机缓存的该模型文件，不影响官方模型。`)) return;
  try {
    const removed = await removeLocalModelFiles(entry.record);
    removeLocalModel(id);
    clearModelAdaptation(id);
    dropPendingSuggestions(id);
    if (state.backend.modelId === id) state.backend.setModel(state.appConfig.webllm.defaultModel);
    renderModelOptions(state.backend.modelId);
    renderLocalList();
    await refreshModelCacheStatus();
    showStatus(`已删除 ${entry.label}（清除 ${removed} 个缓存文件）。`);
  } catch (err) {
    showStatus(`删除失败：${err.message}`, { error: true });
  }
}

els.exportModelBtn.addEventListener('click', exportCurrentModel);
els.importModelBtn.addEventListener('click', () => {
  if (requireAdmin('导入模型')) els.importModelInput.click();
});
els.importTrainBtn.addEventListener('click', () => {
  if (requireAdmin('导入训练结果')) els.importTrainInput.click();
});
els.importModelInput.addEventListener('change', () => {
  const file = els.importModelInput.files?.[0];
  if (file) importModelFromFile(file, 'import');
});
els.importTrainInput.addEventListener('change', () => {
  const file = els.importTrainInput.files?.[0];
  if (file) importModelFromFile(file, 'training');
});

// 引用：在回复中选中文字后显示按钮
els.messages.addEventListener('mouseup', () => setTimeout(updateQuote, 0));
els.messages.addEventListener('touchend', () => setTimeout(updateQuote, 0), { passive: true });
els.messages.addEventListener('scroll', hideQuote, { passive: true });
document.addEventListener(
  'mousedown',
  (e) => {
    if (quoteEl && e.target !== quoteEl) hideQuote();
  },
  true
);

// ---------- 登录（仅远程模式；webllm 模式无需口令即可使用） ----------
function showLogin(message = '') {
  els.main.hidden = true;
  els.login.hidden = false;
  els.loginError.textContent = message;
  els.password.value = '';
  els.password.focus();
}

function showMain() {
  els.login.hidden = true;
  els.main.hidden = false;
  refreshAll();
  refreshIdleStatus();
  els.input.focus({ preventScroll: true });
}

els.loginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  els.loginError.textContent = '';
  els.loginBtn.disabled = true;
  try {
    await login(els.password.value);
    showMain();
  } catch (err) {
    els.loginError.textContent = err.message || '登录失败';
  } finally {
    els.loginBtn.disabled = false;
  }
});

// ---------- 权限（管理员） ----------
function toggleRolePanel(force) {
  const open = typeof force === 'boolean' ? force : els.rolePanel.hidden;
  if (open) {
    closeHistory();
    refreshRoleUI();
  }
  els.rolePanel.hidden = !open;
  els.roleBtn.setAttribute('aria-expanded', String(open));
  if (open && !els.roleForm.hidden) els.adminKey.focus();
}

function refreshRoleUI() {
  const admin = isAdmin();
  const exp = adminExpiry();
  const hasKey = hasAccessGate(state.appConfig.access);
  els.roleBadge.textContent = admin ? '管理员' : '访客';
  els.roleBadge.classList.toggle('is-admin', admin);
  els.roleForm.hidden = admin || !hasKey;
  els.roleLogoutBtn.hidden = !admin;
  if (admin) {
    els.roleStatus.textContent = `当前为管理员（有效至 ${exp.toLocaleDateString('zh-CN')}）。可导出和导入模型。`;
  } else if (hasKey) {
    els.roleStatus.textContent = '当前为访客。输入管理员密钥可升级，以导出和导入模型；聊天功能不受影响。';
  } else {
    els.roleStatus.textContent = '未配置管理员密钥，无法升级权限。';
  }
  els.exportModelBtn.disabled = !admin;
  setImportButtons(admin);
  els.modelAdminHint.textContent = admin
    ? 'AI 训练：请使用仓库中的 training/ 训练运行时（需要 GPU），训练结果用“导入训练结果”加入新建AI。'
    : '导出与导入模型仅限管理员。';
  renderLocalList();
  renderAdaptList();
  if (!admin) pendingSuggestions = [];
  renderSuggestions();
}

/** AI 适应记录：管理员可查看、删除（仅本机）。 */
function renderAdaptList() {
  const admin = isAdmin();
  const list = els.adaptList;
  if (!list) return;
  list.replaceChildren();
  if (MODE !== 'webllm' || !admin) {
    els.adaptHint.textContent = MODE !== 'webllm' ? '服务端模式下不记录 AI 适应反馈。' : 'AI 适应记录仅管理员可见。';
    return;
  }
  els.adaptHint.textContent =
    '管理员在 AI 回复之后发送的消息会记为该 AI 的反馈（仅本机，不上传）；' +
    '以“参数：temperature=0.2, maxTokens=800”开头的一行可调整该 AI 的参数。';
  const ids = adaptedModelIds();
  if (!ids.length) {
    const p = document.createElement('p');
    p.className = 'muted small';
    p.textContent = '暂无适应记录。';
    list.append(p);
    return;
  }
  for (const id of ids) {
    const { notes, params } = listAdaptation(id);
    const head = document.createElement('div');
    head.className = 'local-row';
    const name = document.createElement('div');
    name.className = 'local-name';
    name.textContent = `${modelDisplayName(id)}（${notes.length} 条${Object.keys(params).length ? '，有参数' : ''}）`;
    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'btn ghost small';
    clear.dataset.action = 'adapt-clear';
    clear.dataset.id = id;
    clear.textContent = '清空';
    head.append(name, clear);
    list.append(head);
    for (const n of notes) {
      const row = document.createElement('div');
      row.className = 'local-row';
      const t = document.createElement('div');
      t.className = 'muted small local-meta';
      t.textContent = n.text;
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn ghost small';
      del.dataset.action = 'adapt-delete';
      del.dataset.id = id;
      del.dataset.note = n.id;
      del.textContent = '删除';
      row.append(t, del);
      list.append(row);
    }
    renderParamDetails(list, id, params);
  }
}

const PARAM_SOURCE_LABEL = { explicit: '参数行', suggestion: '建议确认', reset: '恢复默认' };

function formatLogTime(ts) {
  return new Date(ts).toLocaleString('zh-CN', { hour12: false });
}

/** 当前生效值（区分默认与本机覆盖）、恢复默认按钮，以及最近的参数变更记录。 */
function renderParamDetails(list, modelId, overrides) {
  const eff = effectiveParams(modelId);
  const line = document.createElement('div');
  line.className = 'muted small local-meta';
  line.textContent =
    '当前生效：' +
    ['temperature', 'maxTokens']
      .map((k) => `${k} = ${eff[k]}（${k in overrides ? '本机覆盖' : '默认'}）`)
      .join('｜');
  list.append(line);

  for (const key of Object.keys(overrides)) {
    const row = document.createElement('div');
    row.className = 'local-row';
    const t = document.createElement('div');
    t.className = 'muted small local-meta';
    t.textContent = `${key} 已被覆盖为 ${overrides[key]}`;
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'btn ghost small';
    reset.dataset.action = 'adapt-reset';
    reset.dataset.id = modelId;
    reset.dataset.key = key;
    reset.textContent = '恢复默认';
    row.append(t, reset);
    list.append(row);
  }

  const log = listParamLog(modelId).slice(0, 10);
  if (log.length) {
    const title = document.createElement('div');
    title.className = 'muted small local-meta';
    title.textContent = `参数变更记录（最近 ${log.length} 条，最新在前）：`;
    list.append(title);
    for (const c of log) {
      const row = document.createElement('div');
      row.className = 'muted small local-meta';
      const src = PARAM_SOURCE_LABEL[c.source] || c.source;
      const why = c.source === 'suggestion' ? `（${c.reason.replace(/^确认建议：/, '')}）` : '';
      row.textContent =
        `${formatLogTime(c.ts)}  ${c.key}：${c.from ?? '默认'} → ${c.to ?? '默认'}  来源：${src}${why}`;
      list.append(row);
    }
  }
}

function resetAdaptParam(modelId, key) {
  if (!requireAdmin('恢复默认参数')) return;
  if (!resetParam(modelId, key)) return;
  renderAdaptList();
  showStatus(`已恢复「${modelDisplayName(modelId)}」的 ${key} 为默认值（仅本机，从下一次回答起生效）。`);
}

/** 新建AI 列表：显示导入地址（本地缓存标识），管理员可删除。 */
function renderLocalList() {
  const list = loadLocalModels();
  const admin = isAdmin();
  els.localModelList.replaceChildren();
  if (!list.length) {
    const p = document.createElement('p');
    p.className = 'muted small';
    p.textContent = '暂无“新建AI”。';
    els.localModelList.append(p);
    return;
  }
  for (const e of list) {
    const row = document.createElement('div');
    row.className = 'local-row';
    const name = document.createElement('div');
    name.className = 'local-name';
    name.textContent = e.label;
    const meta = document.createElement('div');
    meta.className = 'muted small local-meta';
    const addr = e.record?.model || '';
    meta.textContent = `${e.source === 'training' ? '训练结果' : '导入'} · 本机缓存 · ${addr}`;
    row.append(name, meta);
    if (admin) {
      const del = document.createElement('button');
      del.type = 'button';
      del.className = 'btn ghost small';
      del.dataset.action = 'local-delete';
      del.dataset.id = e.id;
      del.textContent = '删除';
      row.append(del);
    }
    els.localModelList.append(row);
  }
}

els.roleBtn.addEventListener('click', () => toggleRolePanel());
els.roleForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  els.roleError.textContent = '';
  if (!hasAccessGate(state.appConfig.access)) {
    els.roleError.textContent = '未配置管理员密钥';
    return;
  }
  els.roleSubmit.disabled = true;
  try {
    const ok = await verifyAccessPassword(state.appConfig.access, els.adminKey.value);
    if (!ok) throw new Error('管理员密钥错误');
    grantAdmin();
    els.adminKey.value = '';
    refreshRoleUI();
    showStatus('已升级为管理员，有效期 30 天。可导出和导入模型。');
  } catch (err) {
    els.roleError.textContent = err.message || '验证失败';
  } finally {
    els.roleSubmit.disabled = false;
  }
});
els.roleLogoutBtn.addEventListener('click', () => {
  revokeAdmin();
  refreshRoleUI();
  showStatus('已退出管理员。');
});

// ---------- 输入框 ----------
function autoGrow() {
  els.input.style.height = 'auto';
  els.input.style.height = `${Math.min(els.input.scrollHeight, 160)}px`;
}

els.input.addEventListener('input', autoGrow);
els.input.addEventListener('keydown', (e) => {
  // 中文输入法组合输入期间按 Enter 不应发送
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});
els.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = els.input.value;
  if (!text.trim()) return;
  els.input.value = '';
  autoGrow();
  send(text);
});
els.stopBtn.addEventListener('click', stop);

els.newChatBtn.addEventListener('click', newChat);
els.embedNewBtn.addEventListener('click', newChat);
els.clearAllBtn.addEventListener('click', clearAll);
els.historyBtn.addEventListener('click', () => {
  toggleRolePanel(false);
  toggleHistory();
});

els.modelSelect.addEventListener('change', () => {
  state.backend.setModel(els.modelSelect.value);
  const label = els.modelSelect.selectedOptions[0]?.textContent || els.modelSelect.value;
  showStatus(`已切换到 ${label}，发送消息时将自动加载。之后的回答会标注为该 AI，并沿用本会话的上下文。`);
});

document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-action]');
  if (btn && btn.dataset.action === 'switch') return switchSession(btn.dataset.id);
  if (btn && btn.dataset.action === 'delete') return removeSession(btn.dataset.id);
  if (btn && btn.dataset.action === 'copy') return copyCode(btn);
  if (btn && btn.dataset.action === 'regen') return regenerate();
  if (btn && btn.dataset.action === 'local-delete') return deleteLocalModel(btn.dataset.id);
  if (btn && btn.dataset.action === 'adapt-delete') return deleteAdaptNote(btn.dataset.id, btn.dataset.note);
  if (btn && btn.dataset.action === 'adapt-clear') return clearAdaptation(btn.dataset.id);
  if (btn && btn.dataset.action === 'adapt-reset') return resetAdaptParam(btn.dataset.id, btn.dataset.key);
  if (btn && btn.dataset.action === 'suggest-accept') return acceptSuggestion(btn.dataset.id);
  if (btn && btn.dataset.action === 'suggest-dismiss') return dismissSuggestion(btn.dataset.id);
  if (!e.target.closest('.chat-head')) {
    if (!els.historyPanel.hidden) closeHistory();
    if (!els.rolePanel.hidden) toggleRolePanel(false);
  }
});

async function copyCode(btn) {
  const code = btn.closest('.code-block')?.querySelector('code')?.textContent || '';
  let ok = false;
  try {
    await navigator.clipboard.writeText(code);
    ok = true;
  } catch {
    // iframe 未授权 clipboard-write 时回退到 execCommand
    const ta = document.createElement('textarea');
    ta.value = code;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      ok = document.execCommand('copy');
    } catch {
      ok = false;
    }
    ta.remove();
  }
  const old = btn.textContent;
  btn.textContent = ok ? '已复制' : '复制失败';
  setTimeout(() => (btn.textContent = old), 1500);
}

// ---------- 父页面通信（postMessage） ----------
/**
 * 父页面（如 Better Hydro 脚本）可以发送：
 *   { type: 'oj-ai:fill', text: '题目内容', mode: 'replace' | 'append', send: false }
 * 只有来自白名单 origin 且 source 为直接父窗口的消息才会被处理。
 * 默认只填入输入框，send: true 时才会直接发送。
 */
window.addEventListener('message', (e) => {
  if (e.source !== window.parent) return;
  if (!isAllowedOrigin(e.origin, state.appConfig.allowedParentOrigins || [])) return;
  const data = e.data;
  if (!data || typeof data !== 'object' || data.type !== 'oj-ai:fill') return;

  const text = String(data.text ?? '').slice(0, 20000);
  if (data.mode === 'append' && els.input.value.trim()) {
    els.input.value = `${els.input.value.replace(/\s+$/, '')}\n\n${text}`;
  } else {
    els.input.value = text;
  }
  autoGrow();
  els.input.focus();
  if (data.send === true) {
    els.composer.requestSubmit();
  }
  e.source.postMessage({ type: 'oj-ai:filled' }, e.origin);
});

function notifyParentReady() {
  if (!isEmbedded || !document.referrer) return;
  try {
    const parentOrigin = new URL(document.referrer).origin;
    if (isAllowedOrigin(parentOrigin, state.appConfig.allowedParentOrigins || [])) {
      window.parent.postMessage({ type: 'oj-ai:ready' }, parentOrigin);
    }
  } catch {
    /* referrer 不可用时忽略 */
  }
}

// ---------- 启动 ----------
async function loadAppConfig() {
  if (MODE === 'remote') {
    const c = await fetchConfig();
    return { appName: c.appName, allowedParentOrigins: c.allowedParentOrigins, access: {}, webllm: null, model: c.model };
  }
  const res = await fetch('config.json', { cache: 'no-store' });
  if (!res.ok) throw new Error(`无法读取 config.json（HTTP ${res.status}）`);
  return res.json();
}

async function init() {
  if (isEmbed) els.body.classList.add('embed');
  if (isEmbedded) els.body.classList.add('in-frame');

  try {
    state.appConfig = { ...state.appConfig, ...(await loadAppConfig()) };
  } catch (err) {
    console.error(err);
  }
  const appName = state.appConfig.appName || 'OJ 刷题助手';
  els.brandName.textContent = appName;
  els.loginTitle.textContent = appName;
  document.title = appName;

  state.backend = createBackend({ mode: MODE, appConfig: state.appConfig });
  if (MODE === 'webllm') {
    renderModelOptions(state.backend.modelId);
    els.modelSelect.hidden = false;
    els.modelSelect.title = '✓ 表示模型已下载到本机，可离线使用';
    els.modelLabel.hidden = true;
    setTimeout(refreshModelCacheStatus, 0);
    refreshRoleUI();
  } else {
    els.modelLabel.textContent = state.appConfig.model ? `模型：${state.appConfig.model}` : '';
    els.roleBtn.hidden = true;
  }

  if (state.store.sessions.length === 0) createSession(state.store);
  else if (!getSession(state.store)) state.store.activeId = state.store.sessions[0].id;
  persist();

  notifyParentReady();

  if (MODE === 'remote') {
    if (getToken() && (await checkSession())) showMain();
    else showLogin();
  } else {
    // webllm 模式：默认无需口令，直接进入聊天；管理员权限在“更改权限”中升级
    showMain();
  }
}

init();
