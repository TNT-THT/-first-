import 'katex/dist/katex.min.css';
import './style.css';
import DOMPurify from 'dompurify';
import { renderMarkdown } from './markdown.js';
import { hasThinking, stripThinking } from './think.js';
import { isAllowedOrigin } from './origin.js';
import { ApiError, checkSession, fetchConfig, getToken, login, setToken } from './api.js';
import { createBackend } from './engines/backend.js';
import { checkWebGPU, getModelRecords, WEBGPU_MISSING } from './engines/webllm.js';
import { buildModelFile, importModelFile, listCachedModelIds, readModelFileHeader } from './model-file.js';
import { hasAccessGate, isUnlocked, setUnlocked, verifyAccessPassword } from './access.js';
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
}

async function send(rawText) {
  const text = (rawText || '').trim();
  if (!text || state.streaming) return;

  const session = ensureSession();
  const isFirst = session.messages.length === 0;
  const userMsg = { role: 'user', content: text, ts: Date.now() };
  const assistantMsg = { role: 'assistant', content: '', ts: Date.now() };
  session.messages.push(userMsg, assistantMsg);
  if (session.messages.length > MAX_MESSAGES_PER_SESSION) {
    session.messages.splice(0, session.messages.length - MAX_MESSAGES_PER_SESSION);
  }
  if (isFirst) session.title = titleFrom(text);
  session.updatedAt = Date.now();
  persist();

  // 上下文：不包含本次刚添加的空占位回复
  const context = buildContext(session.messages.slice(0, -1), state.backend.limits);

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

function getRecords() {
  return getModelRecords(state.appConfig.webllm || {});
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

async function exportCurrentModel() {
  if (state.streaming) return showStatus('请等待当前回复结束后再导出模型。');
  const modelId = state.backend.modelId;
  const label = optionLabel(modelId);
  els.exportModelBtn.disabled = true;
  try {
    const record = (await getRecords()).find((r) => r.model_id === modelId);
    if (!record) throw new Error(`找不到模型记录：${modelId}`);
    showStatus(`正在打包 ${label}…`, { progress: 0 });
    const { blob, filename, size, entryCount } = await buildModelFile(record, {
      onProgress: ({ done, total }) =>
        showStatus(`正在打包 ${label}（${done}/${total} 个文件）…`, { progress: done / total }),
    });
    downloadBlob(blob, filename);
    showStatus(`已导出 ${filename}（${formatBytes(size)}，${entryCount} 个文件）。导入后即可离线使用。`);
  } catch (err) {
    showStatus(`导出失败：${err.message}`, { error: true });
  } finally {
    els.exportModelBtn.disabled = false;
  }
}

async function importModelFromFile(file) {
  if (state.streaming) return showStatus('请等待当前回复结束后再导入模型。');
  els.importModelBtn.disabled = true;
  try {
    const { header } = await readModelFileHeader(file);
    const record = (await getRecords()).find((r) => r.model_id === header.modelId);
    if (!record) throw new Error(`当前模型列表中没有 ${header.modelId}，请检查 config.json 中的模型配置。`);
    const label = optionLabel(record.model_id);
    showStatus(`正在导入 ${label}（${formatBytes(file.size)}）…`, { progress: 0 });
    await importModelFile(file, record, {
      onProgress: ({ done, total }) =>
        showStatus(`正在导入 ${label}（${done}/${total} 个文件）…`, { progress: done / total }),
    });
    await refreshModelCacheStatus();
    const inMenu = [...els.modelSelect.options].some((o) => o.value === record.model_id);
    if (inMenu) {
      state.backend.setModel(record.model_id);
      els.modelSelect.value = record.model_id;
      showStatus(`导入完成：${label} 已可离线使用。发送消息时将直接从本机加载。`);
    } else {
      showStatus(`导入完成，但 ${label} 不在下拉框中，无法选择使用。`, { error: true });
    }
  } catch (err) {
    showStatus(`导入失败：${err.message}`, { error: true });
  } finally {
    els.importModelBtn.disabled = false;
    els.importModelInput.value = '';
  }
}

els.exportModelBtn.addEventListener('click', exportCurrentModel);
els.importModelBtn.addEventListener('click', () => els.importModelInput.click());
els.importModelInput.addEventListener('change', () => {
  const file = els.importModelInput.files?.[0];
  if (file) importModelFromFile(file);
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

// ---------- 登录 ----------
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
    if (MODE === 'remote') {
      await login(els.password.value);
    } else {
      const ok = await verifyAccessPassword(state.appConfig.access, els.password.value);
      if (!ok) throw new Error('口令错误');
      setUnlocked();
    }
    showMain();
  } catch (err) {
    els.loginError.textContent = err.message || '登录失败';
  } finally {
    els.loginBtn.disabled = false;
  }
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
els.historyBtn.addEventListener('click', () => toggleHistory());

els.modelSelect.addEventListener('change', () => {
  state.backend.setModel(els.modelSelect.value);
  const label = els.modelSelect.selectedOptions[0]?.textContent || els.modelSelect.value;
  showStatus(`已切换到 ${label}，发送消息时将自动加载。`);
});

document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-action]');
  if (btn && btn.dataset.action === 'switch') return switchSession(btn.dataset.id);
  if (btn && btn.dataset.action === 'delete') return removeSession(btn.dataset.id);
  if (btn && btn.dataset.action === 'copy') return copyCode(btn);
  if (btn && btn.dataset.action === 'regen') return regenerate();
  if (!els.historyPanel.hidden && !e.target.closest('.chat-head')) closeHistory();
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
    els.modelSelect.replaceChildren(
      ...state.backend.models.map((m) => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.label;
        opt.dataset.label = m.label;
        return opt;
      })
    );
    els.modelSelect.value = state.backend.modelId;
    els.modelSelect.hidden = false;
    els.modelSelect.title = '✓ 表示模型已下载到本机，可离线使用';
    els.modelLabel.hidden = true;
    setTimeout(refreshModelCacheStatus, 0);
  } else {
    els.modelLabel.textContent = state.appConfig.model ? `模型：${state.appConfig.model}` : '';
  }

  if (state.store.sessions.length === 0) createSession(state.store);
  else if (!getSession(state.store)) state.store.activeId = state.store.sessions[0].id;
  persist();

  notifyParentReady();

  if (MODE === 'remote') {
    if (getToken() && (await checkSession())) showMain();
    else showLogin();
  } else if (isUnlocked(state.appConfig.access)) {
    showMain();
  } else {
    showLogin(hasAccessGate(state.appConfig.access) ? '' : '');
  }
}

init();
