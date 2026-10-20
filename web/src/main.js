import 'katex/dist/katex.min.css';
import './style.css';
import DOMPurify from 'dompurify';
import { renderMarkdown } from './markdown.js';
import { hasThinking, splitThinking } from './think.js';
import { truncateThinking } from './reasoning.js';
import { isAllowedOrigin } from './origin.js';
import { ApiError, checkSession, fetchConfig, getToken, login, setToken } from './api.js';
import { createBackend } from './engines/backend.js';
import { getSavedDefaultModel, resolveStartupModel, saveDefaultModel } from './default-model.js';
import { addCheckRule, checkRuleError, findViolations, listCheckRules } from './answer-check.js';
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
import { pendingModels } from './pending-models.js';
import { buildDataExport, DATA_SECTIONS, dataExportFilename } from './data-export.js';
import {
  adaptedModelIds,
  applyParamSuggestion,
  clearModelAdaptation,
  getModelParams,
  listAdaptation,
  listCorrections,
  listKnowledge,
  listParamLog,
  migrateAdaptation,
  parseCorrections,
  recordFeedback,
  removeCorrection,
  removeKnowledge,
  removeNote,
  resetAllParams,
  resetParam,
  sanitizeParams,
  setModelParams,
  solidifyJsonl,
  suggestParams,
  mechanicalReply,
} from './adaptation.js';
import {
  buildContext,
  createSession,
  deleteSession,
  expandVisibleStart,
  getSession,
  initialVisibleStart,
  loadStore,
  MAX_MESSAGES_PER_SESSION,
  saveStore,
  titleFrom,
  UNFOLD_STEP,
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
  thinkToggleLabel: $('#thinkToggleLabel'),
  thinkToggle: $('#thinkToggle'),
  statusBar: $('#statusBar'),
  statusText: $('#statusText'),
  statusProgress: $('#statusProgress'),
  statusBarFill: $('#statusBarFill'),
  exportModelBtn: $('#exportModelBtn'),
  downloadSiteBtn: $('#downloadSiteBtn'),
  exportDataBtn: $('#exportDataBtn'),
  dataSectionList: $('#dataSectionList'),
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
  correctionDialog: $('#correctionDialog'),
  correctionForm: $('#correctionForm'),
  correctionPreview: $('#correctionPreview'),
  correctionDesc: $('#correctionDesc'),
  correctionKey: $('#correctionKey'),
  markDialog: $('#markDialog'),
  markForm: $('#markForm'),
  markDesc: $('#markDesc'),
  markWrong: $('#markWrong'),
  markRight: $('#markRight'),
  markKey: $('#markKey'),
  markError: $('#markError'),
  markCancel: $('#markCancel'),
  correctionError: $('#correctionError'),
  correctionCancel: $('#correctionCancel'),
  correctionSubmit: $('#correctionSubmit'),
  roleLogoutBtn: $('#roleLogoutBtn'),
  modelAdminHint: $('#modelAdminHint'),
  localModelList: $('#localModelList'),
  adaptHint: $('#adaptHint'),
  adaptSuggest: $('#adaptSuggest'),
  paramOverrides: $('#paramOverrides'),
  exportWithParams: $('#exportWithParams'),
  exportParamsText: $('#exportParamsText'),
  exportParamsLabel: $('#exportParamsLabel'),
  adaptList: $('#adaptList'),
  migrateBox: $('#migrateBox'),
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
  foldSessionId: null, // 当前折叠状态对应的会话
  visibleStart: 0, // 历史会话中第一条已渲染消息的下标（之前的消息折叠）
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

/**
 * 思考过程（推理模型）：可折叠的小节，位于回答之上。
 * 思考进行中默认展开，回答开始后自动收起；用户手动点开或收起后不再自动变化。
 * 同一个节点在流式输出中反复复用，所以用户的展开状态不会被重绘打断。
 */
function paintThinking(bubble, thinking, { live = false } = {}) {
  let box = bubble.querySelector(':scope > details.thinking');
  if (!thinking) {
    box?.remove();
    return;
  }
  if (!box) {
    box = document.createElement('details');
    box.className = 'thinking';
    const summary = document.createElement('summary');
    const body = document.createElement('div');
    body.className = 'thinking-body';
    box.append(summary, body);
    box.addEventListener('click', (e) => {
      if (e.target.closest('summary')) box.dataset.manual = '1';
    });
    bubble.prepend(box);
  }
  box.querySelector('.thinking-body').textContent = thinking;
  box.querySelector('summary').textContent = live
    ? `思考中…（已写 ${thinking.length} 字）`
    : `思考过程（${thinking.length} 字，点击展开或收起）`;
  box.classList.toggle('live', live);
  if (!box.dataset.manual) box.open = live;
}

function paintAssistant(bubble, msg, { streaming = false, label = '正在思考…' } = {}) {
  bubble.classList.toggle('error', !!msg.error);
  bubble.classList.toggle('streaming', streaming);
  // 推理模型：思考过程单独展示（可折叠），回答正常渲染。思考过程不作为上下文发送给模型。
  const parts = splitThinking(msg.content);
  const thinking = parts.thinking || msg.thinking || '';
  const thinkingLive = streaming && parts.thinkingOpen;
  paintThinking(bubble, thinking, { live: thinkingLive });

  let answerEl = bubble.querySelector(':scope > .answer');
  if (!answerEl) {
    answerEl = document.createElement('div');
    answerEl.className = 'answer';
    bubble.appendChild(answerEl);
  }
  if (!parts.answer && streaming) {
    const hint = thinkingLive ? '正在思考…' : thinking ? '正在回答…' : label;
    answerEl.innerHTML = typingHtml(hint);
    return;
  }
  let html = sanitize(renderMarkdown(parts.answer));
  if (msg.stopped) html += '<p class="note">（已停止生成）</p>';
  answerEl.innerHTML = html;
  // 生成后的校验：回答完成后，若含有管理员标记的错误说法，在回答下方提示正确说法（不改写回答）
  bubble.querySelector(':scope > .check-warning')?.remove();
  if (!streaming && msg.role === 'assistant' && msg.model && !msg.error) {
    const hits = findViolations(parts.answer, listCheckRules(msg.model));
    if (hits.length) bubble.appendChild(buildCheckWarning(hits));
  }
}

/** 校验提示：列出命中的错误说法与对应的正确说法（用 textContent 写入，避免注入）。 */
function buildCheckWarning(hits) {
  const box = document.createElement('div');
  box.className = 'check-warning';
  const title = document.createElement('strong');
  title.textContent = '⚠ 回答中含有管理员标记为错误的说法（以下为正确说法，以更正为准）';
  box.appendChild(title);
  const list = document.createElement('ul');
  for (const { rule } of hits) {
    const li = document.createElement('li');
    li.textContent = `「${rule.wrong}」→ 正确说法：「${rule.right}」`;
    list.appendChild(li);
  }
  box.appendChild(list);
  return box;
}

/** 回复结束：拆分思考过程与回答。历史记录只保存回答，以及（截断后的）思考过程用于回看。 */
function finalizeAssistant(msg) {
  const parts = splitThinking(msg.content);
  msg.content = parts.answer;
  if (parts.thinking) msg.thinking = truncateThinking(parts.thinking);
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

/** 回复结束后，在该回复下方挂一个“重新生成”按钮；管理员还可以看到“标记错误”。 */
function attachRegen(afterEl, msg) {
  const bar = document.createElement('div');
  bar.className = 'msg-actions';
  bar.innerHTML = '<button type="button" class="btn ghost small" data-action="regen">重新生成</button>';
  if (MODE === 'webllm' && isAdmin() && msg?.model) {
    const mark = document.createElement('button');
    mark.type = 'button';
    mark.className = 'btn ghost small';
    mark.textContent = '标记错误';
    // 点击按钮前按下鼠标时记下选中的文字（点击后选区可能消失），只取这条回复中的选区
    let picked = '';
    mark.addEventListener('mousedown', () => {
      picked = selectedTextIn(afterEl.querySelector('.bubble'));
    });
    mark.addEventListener('click', () => openMarkDialog(msg, picked));
    bar.appendChild(mark);
  }
  afterEl.after(bar);
}

/** 当前选中的文字（仅当选区完全位于 el 内时返回）。 */
function selectedTextIn(el) {
  const sel = window.getSelection?.();
  if (!el || !sel || sel.isCollapsed) return '';
  const inside = el.contains(sel.anchorNode) && el.contains(sel.focusNode);
  return inside ? sel.toString().trim() : '';
}

/**
 * 管理员标记错误：填写错误说法与正确说法，验证管理员密钥后，
 * 保存为校验规则（以后回答再出现错误说法时提示），并把正确说法记为该 AI 的更正（仅本机）。
 */
function openMarkDialog(msg, presetWrong = '') {
  const dlg = els.markDialog;
  const session = activeSession();
  if (!dlg || !session) return;
  els.markWrong.value = presetWrong;
  els.markRight.value = '';
  els.markKey.value = '';
  els.markError.textContent = '';
  els.markDesc.textContent =
    `为「${modelDisplayName(msg.model)}」标记错误说法。之后该 AI 的回答若再出现错误说法，会在回答下方提示正确说法。` +
    '正确说法会同时作为该 AI 的更正（底层设定，仅本机）。输入管理员密钥验证后保存；取消则不记录。';
  els.markForm.onsubmit = async (e) => {
    e.preventDefault();
    els.markError.textContent = '';
    const wrong = els.markWrong.value;
    const right = els.markRight.value;
    // 先检查输入，再验证密钥，避免无效规则被保存
    const inputError = checkRuleError(wrong, right);
    if (inputError) {
      els.markError.textContent = inputError;
      return;
    }
    if (!hasAccessGate(state.appConfig.access)) {
      els.markError.textContent = '未配置管理员密钥，无法记录。';
      return;
    }
    const submitBtn = els.markForm.querySelector('[type="submit"]');
    if (submitBtn) submitBtn.disabled = true;
    try {
      const ok = await verifyAccessPassword(state.appConfig.access, els.markKey.value);
      if (!ok) {
        els.markError.textContent = '管理员密钥错误，未记录。';
        els.markKey.select();
        return;
      }
      const res = addCheckRule(msg.model, { wrong, right });
      if (res.error) {
        els.markError.textContent = res.error;
        return;
      }
      // 正确说法同时作为更正（已验证密钥，仅本机）；换行压成一行，符合更正格式
      recordFeedback(msg.model, `更正：${right.replace(/\s*\n\s*/g, ' ')}`, {
        allowCorrections: true,
        question: questionBefore(session, msg),
      });
      dlg.close();
      renderMessages();
      showStatus(`已记录：「${wrong.trim()}」→「${right.trim()}」（同时作为该 AI 的更正，仅本机）。`);
    } finally {
      if (submitBtn) submitBtn.disabled = false;
    }
  };
  els.markCancel.onclick = () => dlg.close();
  dlg.showModal();
  (presetWrong ? els.markRight : els.markWrong).focus();
}

/** 历史会话的折叠条：较早的消息未渲染，点击再展开（分批渲染，避免一次性加载很慢）。 */
function buildFoldBar() {
  const bar = document.createElement('div');
  bar.className = 'fold-bar';
  bar.innerHTML =
    '<span class="fold-text"></span>' +
    '<button type="button" class="btn ghost small" data-fold="more"></button>' +
    '<button type="button" class="btn ghost small" data-fold="all">全部展开</button>';
  bar.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-fold]');
    if (!btn) return;
    unfoldHistory(btn.dataset.fold === 'all' ? Infinity : UNFOLD_STEP);
  });
  return bar;
}

function updateFoldBar() {
  const bar = els.messages.querySelector('.fold-bar');
  if (!bar) return;
  const hidden = state.visibleStart;
  if (hidden <= 0) {
    bar.remove();
    return;
  }
  bar.querySelector('.fold-text').textContent = `较早的 ${hidden} 条消息已折叠，以加快加载。`;
  bar.querySelector('[data-fold="more"]').textContent = `展开更早的 ${Math.min(UNFOLD_STEP, hidden)} 条`;
  bar.querySelector('[data-fold="all"]').hidden = hidden <= UNFOLD_STEP;
}

/** 展开更早的消息：只插入新增的节点，并保持当前阅读位置不变。 */
function unfoldHistory(step) {
  const session = activeSession();
  const bar = els.messages.querySelector('.fold-bar');
  if (!session || !bar || state.visibleStart <= 0) return;
  const oldStart = state.visibleStart;
  const newStart = expandVisibleStart(oldStart, step);
  const heightBefore = els.messages.scrollHeight;
  const frag = document.createDocumentFragment();
  for (let i = newStart; i < oldStart; i++) frag.appendChild(createMessageEl(session.messages[i]).wrap);
  bar.after(frag);
  state.visibleStart = newStart;
  updateFoldBar();
  els.messages.scrollTop += els.messages.scrollHeight - heightBefore;
}

function renderMessages() {
  const session = activeSession();
  els.messages.replaceChildren();
  if (!session || session.messages.length === 0) {
    state.foldSessionId = null;
    state.visibleStart = 0;
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.innerHTML =
      '<p class="empty-title">把题目交给我</p>' +
      '<p class="muted">可以粘贴题面、报错信息或你的代码，我会给出思路、复杂度分析和示例代码。</p>';
    els.messages.appendChild(empty);
  } else {
    const total = session.messages.length;
    // 切换到另一个会话时，重新只渲染最近的消息；同一会话内保留用户已展开的范围
    if (state.foldSessionId !== session.id || state.visibleStart > total) {
      state.foldSessionId = session.id;
      state.visibleStart = initialVisibleStart(total);
    }
    if (state.visibleStart > 0) els.messages.appendChild(buildFoldBar());
    for (let i = state.visibleStart; i < total; i++) {
      els.messages.appendChild(createMessageEl(session.messages[i]).wrap);
    }
    updateFoldBar();
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
/** 可以接收管理员反馈的目标：会话中最后一条已完成的 AI 回复（且带有模型信息）。 */
function feedbackTarget(session) {
  const last = [...(session?.messages || [])].reverse().find((m) => m.role === 'assistant' && !m.fixed);
  if (!last || !last.model || last.error || !last.content) return null;
  return last;
}

/** 某条回复之前最近的用户消息（即它所回答的问题），用于固化训练数据。 */
function questionBefore(session, target) {
  const msgs = session?.messages || [];
  for (let j = msgs.indexOf(target) - 1; j >= 0; j -= 1) {
    if (msgs[j].role === 'user' && msgs[j].content) return msgs[j].content;
  }
  return '';
}

function recordAdminFeedback(session, text, { allowCorrections = false } = {}) {
  if (MODE !== 'webllm' || !isAdmin()) return { found: [] };
  const last = feedbackTarget(session);
  if (!last) return { found: [] };
  const question = questionBefore(session, last);
  const rec = recordFeedback(last.model, text, { allowCorrections, question });
  if (!rec) return { found: [] };
  const parts = [];
  if (rec.note) parts.push('已记录反馈');
  if (rec.corrections?.length) {
    parts.push(`已记录 ${rec.corrections.length} 条更正（已验证密钥，作为该 AI 的底层设定，无条件采纳）`);
  }
  if (rec.params) {
    const changed = (rec.changes || []).map((c) => `${c.key} ${c.from ?? '默认'}→${c.to}`);
    parts.push('已更新参数 ' + (changed.length ? changed.join(', ') : '（未变化）'));
  }
  // 没有显式参数行、也没有更正时，按关键词给出“建议参数”，必须由管理员确认后才生效
  let suggested = 0;
  let found = [];
  if (!rec.params && !rec.corrections?.length) {
    found = suggestParams(text, effectiveParams(last.model));
    suggested = addSuggestions(last.model, last.modelLabel || modelDisplayName(last.model), found);
    if (suggested) parts.push(`有 ${suggested} 条参数建议，请在输入框上方确认（确认前不生效）`);
  }
  if (parts.length) {
    showStatus(`${parts.join('，')}（仅本机，对「${modelDisplayName(last.model)}」生效，不上传）。`);
  }
  renderAdaptList();
  // 返回找到的参数建议，供发送流程做机械识别（找到建议且无提问时不调用模型）
  return { found, suggested };
}

/** 待管理员确认的参数建议（仅内存中保存；刷新页面或退出管理员后清空）。 */
let pendingSuggestions = [];

/** 某模型当前生效的参数：本机覆盖优先，否则使用配置默认值。 */
function effectiveParams(modelId) {
  const d = state.appConfig.webllm;
  return {
    temperature: d.temperature,
    maxTokens: d.maxTokens,
    contextWindow: d.contextWindowTokens,
    thinkingWindow: d.thinkingWindowTokens,
    ...getModelParams(modelId),
  };
}

/** 参数在界面中的名称（管理员可见的建议卡片、参数详情共用）。 */
const PARAM_NAMES = {
  temperature: 'temperature（温度）',
  maxTokens: 'maxTokens（最大输出长度）',
  contextWindow: 'contextWindow（上下文窗口）',
  thinkingWindow: 'thinkingWindow（思考窗口）',
};

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
    const name = PARAM_NAMES[s.key] || s.key;
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

/**
 * 管理员密钥验证（用于更正）。返回 Promise<boolean>：验证通过为 true；取消或密钥错误则不发送，文字保留在输入框中。
 */
function verifyCorrectionKey(text, desc = '') {
  return new Promise((resolve) => {
    if (!hasAccessGate(state.appConfig.access)) {
      showStatus('未配置管理员密钥，无法记录更正，消息未发送。', { error: true });
      resolve(false);
      return;
    }
    const dlg = els.correctionDialog;
    let settled = false;
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      els.correctionForm.removeEventListener('submit', onSubmit);
      els.correctionCancel.removeEventListener('click', onCancel);
      dlg.removeEventListener('close', onCancel);
      if (dlg.open) dlg.close();
      els.correctionKey.value = '';
      resolve(ok);
    };
    const onSubmit = async (e) => {
      e.preventDefault();
      els.correctionError.textContent = '';
      els.correctionSubmit.disabled = true;
      try {
        const ok = await verifyAccessPassword(state.appConfig.access, els.correctionKey.value);
        if (ok) finish(true);
        else {
          els.correctionError.textContent = '管理员密钥错误，更正未记录，消息未发送。';
          els.correctionKey.select();
        }
      } finally {
        els.correctionSubmit.disabled = false;
      }
    };
    const onCancel = () => finish(false);
    els.correctionForm.addEventListener('submit', onSubmit);
    els.correctionCancel.addEventListener('click', onCancel);
    dlg.addEventListener('close', onCancel);
    els.correctionPreview.textContent = text;
    els.correctionDesc.textContent = desc || '这条消息包含更正。输入管理员密钥验证后，才会作为该 AI 的底层设定记录（仅本机）；取消则不发送。';
    els.correctionError.textContent = '';
    els.correctionKey.value = '';
    dlg.showModal();
    els.correctionKey.focus();
  });
}

/** 固定提示（不调用模型）：作为助手消息显示，但不标注 AI、不作为反馈对象，也不发送给模型。 */
function showFixedReply(session, isFirst, userMsg, content) {
  session.messages.push(userMsg, { role: 'assistant', content, ts: Date.now(), fixed: true });
  if (session.messages.length > MAX_MESSAGES_PER_SESSION) {
    session.messages.splice(0, session.messages.length - MAX_MESSAGES_PER_SESSION);
  }
  if (isFirst) session.title = titleFrom(userMsg.content);
  session.updatedAt = Date.now();
  persist();
  refreshAll();
  scrollToBottom(true);
  els.input.focus({ preventScroll: true });
}

async function send(rawText) {
  const text = (rawText || '').trim();
  if (!text || state.streaming) return;

  // 更正需要管理员密钥验证（仅 WebLLM 本机模式、管理员、且存在可接收反馈的 AI 回复时）
  let allowCorrections = false;
  if (MODE === 'webllm' && isAdmin() && parseCorrections(text).length) {
    if (feedbackTarget(activeSession())) {
      const ok = await verifyCorrectionKey(text);
      if (!ok) {
        if (!els.input.value) {
          els.input.value = text;
          autoGrow();
        }
        return;
      }
      allowCorrections = true;
    } else {
      showStatus('更正需要在某个 AI 的回复之后发送，本条未记录为更正。');
    }
  }

  const session = ensureSession();
  const isFirst = session.messages.length === 0;
  const userMsg = { role: 'user', content: text, ts: Date.now() };
  const feedback = recordAdminFeedback(session, text, { allowCorrections });
  // 机械识别：管理员消息只是参数调整指令（找到参数建议、没有提问）时，给出固定提示，不调用模型
  const fixed = mechanicalReply(text, feedback?.found);
  if (fixed) {
    showFixedReply(session, isFirst, userMsg, fixed);
    return;
  }
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
      const loaded = await state.backend.ensureReady(({ progress, text: t }) => {
        if (t) progressText = t;
        const pct = Math.round(progress * 100);
        showStatus(`正在加载模型 ${pct}%${t ? `：${t}` : ''}`, { progress });
      });
      await refreshIdleStatus();
      refreshModelCacheStatus();
      // 输出自检的结果（异常、已切换到 fp32 等）
      if (loaded?.note) showStatus(loaded.note, { error: loaded.healthy === false });
    }

    for await (const delta of state.backend.stream({ messages: context, signal: controller.signal })) {
      assistantMsg.content += delta;
      scheduleRender();
    }
    if (frame) cancelAnimationFrame(frame);
    if (controller.signal.aborted) assistantMsg.stopped = true;
    const raw = assistantMsg.content;
    finalizeAssistant(assistantMsg);
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
      assistantMsg.thinking = '';
      assistantMsg.error = true;
      paintAssistant(bubble, assistantMsg);
      if (state.backend.mode === 'webllm') refreshIdleStatus();
    }
  } finally {
    // 历史记录只保存最终回答（思考过程另存，供回看，不发送给模型）
    finalizeAssistant(assistantMsg);
    state.streaming = null;
    setBusy(false);
    session.updatedAt = Date.now();
    persist();
    renderSessionList();
    // 仅当本条回复仍在会话中（未因登录过期被撤回）时才显示“重新生成”
    if (session.messages.includes(assistantMsg)) attachRegen(assistantWrap, assistantMsg);
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
  if (!last || last.role !== 'assistant' || last.fixed || !prev || prev.role !== 'user') return;
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
/** 思考模式开关：仅推理模型显示；偏好在本浏览器中保存（默认开启）。 */
function refreshThinkToggle() {
  const show = MODE === 'webllm' && !!state.backend?.supportsThinking();
  els.thinkToggleLabel.hidden = !show;
  els.thinkToggle.checked = state.backend?.thinkingEnabled() ?? true;
}

function renderModelOptions(selectId = state.backend.modelId) {
  const children = state.backend.models.map((m) => makeOption(m.id, m.label));
  // 待转换的第三方模型：禁用（尚不能运行），与官方模型分组显示，备注放在悬停提示中
  const pending = pendingModels(state.appConfig?.webllm);
  if (pending.length) {
    const group = document.createElement('optgroup');
    group.label = '待转换（暂不可用）';
    for (const p of pending) {
      const opt = makeOption(p.id, p.label);
      opt.disabled = true;
      opt.title = p.note || p.label;
      group.append(opt);
    }
    children.push(group);
  }
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
  refreshThinkToggle();
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

/** 下载网站包（静态版，构建时生成的 site.zip）。仅管理员。不包含任何 AI 模型。 */
async function downloadSitePackage() {
  if (!requireAdmin('下载网站包')) return;
  els.downloadSiteBtn.disabled = true;
  try {
    const res = await fetch('site.zip', { cache: 'no-store' });
    if (!res.ok) throw new Error(`找不到网站包（HTTP ${res.status}）。请先运行 npm run build 生成。`);
    const blob = await res.blob();
    downloadBlob(blob, 'oj-ai-site.zip');
    showStatus(`已下载网站包（${formatBytes(blob.size)}）。网站包不包含任何 AI 模型。`);
  } catch (err) {
    showStatus(`下载网站包失败：${err.message}`, { error: true });
  } finally {
    els.downloadSiteBtn.disabled = !isAdmin();
  }
}

/** 选择性导出本机数据（会话、管理员说明/参数/更正、知识库、新建AI清单）。仅管理员。 */
function exportLocalData() {
  if (!requireAdmin('导出本机数据')) return;
  const ids = [...els.dataSectionList.querySelectorAll('input.data-section:checked')].map((i) => i.value);
  if (!ids.length) return showStatus('请至少选择一项要导出的数据。', { error: true });
  const now = new Date();
  const data = buildDataExport(localStorage, ids, { now });
  const exported = Object.keys(data.sections);
  if (!exported.length) {
    return showStatus('所选数据在本机没有内容，未生成文件。', { error: true });
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  downloadBlob(blob, dataExportFilename(now));
  const missingNote = data.missing.length ? `（本机没有：${data.missing.join('、')}）` : '';
  showStatus(`已导出 ${exported.length} 项本机数据${missingNote}。文件只保存在本机，请妥善保管。`);
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
    // 勾选“同时保存自定义参数”时，把本机覆盖的 temperature / maxTokens 写入文件头部
    const overrides = sanitizeParams(listAdaptation(modelId).params);
    const withParams = !!overrides && els.exportWithParams.checked;
    showStatus(`正在打包 ${label}…`, { progress: 0 });
    const { blob, filename, size, entryCount } = await buildModelFile(record, {
      label: baseLabelOf(modelId),
      adaptation: withParams ? { params: overrides } : undefined,
      onProgress: ({ done, total }) =>
        showStatus(`正在打包 ${label}（${done}/${total} 个文件）…`, { progress: done / total }),
    });
    downloadBlob(blob, filename);
    const paramNote = withParams ? `，已保存自定义参数（${paramSummary(overrides)}）` : '';
    showStatus(`已导出 ${filename}（${formatBytes(size)}，${entryCount} 个文件${paramNote}）。导入后即可离线使用。`);
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
    const { record, adaptation } = await importModelFile(file, {
      onProgress: ({ done, total }) =>
        showStatus(`正在导入 ${label}（${done}/${total} 个文件）…`, { progress: done / total }),
    });
    // 文件中带有的自定义参数只作用于这个新建AI（以新的本地 id 保存）
    const carried = adaptation ? setModelParams(record.model_id, adaptation, { source: 'import', reason: '导入模型文件' }) : null;
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
    refreshThinkToggle();
    await refreshModelCacheStatus();
    renderLocalList();
    const carriedNote = carried ? `，并带入自定义参数（${paramSummary(carried)}）` : '';
    showStatus(`导入完成：${label} 已加入“新建AI”，可离线使用${carriedNote}。发送消息时将直接从本机加载。`);
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

/** 删除一条更正（仅本机）。 */
function deleteCorrection(modelId, correctionId) {
  if (!requireAdmin('删除更正')) return;
  removeCorrection(modelId, correctionId);
  renderAdaptList();
}

/** 删除本地知识库中的一条管理员信息（仅本机）。 */
function deleteKnowledgeEntry(modelId, entryId) {
  if (!requireAdmin('删除本地知识库条目')) return;
  removeKnowledge(modelId, entryId);
  renderAdaptList();
}

function exportSolidifyData(modelId) {
  if (!requireAdmin('导出固化数据')) return;
  const { text, counts } = solidifyJsonl(modelId);
  const total = counts.correction + counts.knowledge;
  const summary = `更正 ${counts.correction} 条，知识库 ${counts.knowledge} 条，跳过 ${counts.skipped} 条（无对应问题）`;
  if (!total) {
    showStatus(`没有可导出的固化样本：${summary}。`, { error: true });
    return;
  }
  const safe = String(modelId).replace(/[^\w.-]+/g, '_');
  downloadBlob(new Blob([text], { type: 'application/x-ndjson' }), `solidify-${safe}.jsonl`);
  showStatus(
    `已导出 ${total} 条固化样本（${summary}）。在本机运行：python3 training/ojtrain.py solidify <文件>（需要 GPU）。`,
  );
}

function clearAdaptation(modelId) {
  if (!requireAdmin('清空 AI 适应记录')) return;
  if (!window.confirm(`清空「${modelDisplayName(modelId)}」的全部适应记录（包括说明、本地知识库、参数与参数变更记录）？`)) return;
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
    if (getSavedDefaultModel() === id) saveDefaultModel(null);
    renderModelOptions(state.backend.modelId);
    renderLocalList();
    await refreshModelCacheStatus();
    showStatus(`已删除 ${entry.label}（清除 ${removed} 个缓存文件）。`);
  } catch (err) {
    showStatus(`删除失败：${err.message}`, { error: true });
  }
}

els.exportModelBtn.addEventListener('click', exportCurrentModel);
els.downloadSiteBtn.addEventListener('click', downloadSitePackage);
els.exportDataBtn.addEventListener('click', exportLocalData);
// 数据分区的复选框（默认全部勾选，管理员可取消）
els.dataSectionList.replaceChildren(
  ...DATA_SECTIONS.map((sec) => {
    const label = document.createElement('label');
    label.className = 'export-params';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = true;
    input.value = sec.id;
    input.className = 'data-section';
    label.append(input, ` ${sec.label}`);
    return label;
  }),
);
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
  els.downloadSiteBtn.disabled = !admin;
  els.exportDataBtn.disabled = !admin;
  // 网站包与本机数据导出只用于静态版（远程模式的数据在服务端）
  els.dataSectionList.closest('.data-export').hidden = MODE !== 'webllm';
  els.downloadSiteBtn.closest('.site-tools').hidden = MODE !== 'webllm';
  setImportButtons(admin);
  els.modelAdminHint.textContent = admin
    ? 'AI 训练：请使用仓库中的 training/ 训练运行时（需要 GPU），训练结果用“导入训练结果”加入新建AI。'
    : '导出与导入模型仅限管理员。';
  renderLocalList();
  renderAdaptList();
  if (!admin) pendingSuggestions = [];
  renderSuggestions();
  renderParamOverrides();
}

/** 跨模型迁移的选择状态（仅本页面会话有效，不保存）。 */
const migrateState = { source: '', target: '' };

function migrateEl(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** 下拉框中的全部 AI（官方、待转换、新建AI），都可以作为迁移目标。 */
function allModelOptions() {
  return [...els.modelSelect.options].map((o) => ({ id: o.value, label: o.dataset.label || o.value }));
}

function makeMigrateSelect(items, value, onChange) {
  const sel = document.createElement('select');
  sel.className = 'migrate-select';
  for (const it of items) {
    const opt = document.createElement('option');
    opt.value = it.id;
    opt.textContent = it.label;
    sel.append(opt);
  }
  sel.value = value;
  sel.addEventListener('change', () => onChange(sel.value));
  return sel;
}

function migrateItemRow({ kind, id, text, question, from }) {
  const row = migrateEl('label', 'local-row migrate-item');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = true;
  box.dataset.kind = kind;
  box.dataset.id = id;
  const label = migrateEl('div', 'muted small local-meta', `${kind === 'correction' ? '更正' : '知识库'}：${text}`);
  const tags = [question ? '含训练样本' : '', from ? `迁入自 ${modelDisplayName(from)}` : ''].filter(Boolean);
  if (tags.length) label.textContent += `（${tags.join('，')}）`;
  row.append(box, label);
  return row;
}

/**
 * 跨模型迁移（仅管理员、仅本机）：选择源 AI 与目标 AI，勾选要迁移的更正与知识库条目。
 * 复制而非移动；迁移更正需要管理员密钥验证（更正会成为目标 AI 的底层设定）。
 */
function renderMigrateBox() {
  const box = els.migrateBox;
  if (!box) return;
  box.replaceChildren();
  const show = MODE === 'webllm' && isAdmin();
  box.hidden = !show;
  if (!show) return;

  const sources = adaptedModelIds();
  const options = allModelOptions();
  if (!sources.includes(migrateState.source)) migrateState.source = sources[0] || '';
  if (!options.some((o) => o.id === migrateState.target && o.id !== migrateState.source)) {
    migrateState.target = options.find((o) => o.id !== migrateState.source)?.id || '';
  }

  box.append(
    migrateEl(
      'p',
      'muted small local-meta',
      '跨模型迁移（仅本机，不上传）：把源 AI 中勾选的更正与知识库条目复制到目标 AI。复制而非移动，源 AI 的记录保持不变。迁移更正需输入管理员密钥验证。',
    ),
  );
  if (!sources.length) {
    box.append(migrateEl('p', 'muted small', '暂无可迁移的适应记录。'));
    return;
  }

  const onSource = (v) => {
    migrateState.source = v;
    renderMigrateBox();
  };
  const onTarget = (v) => {
    migrateState.target = v;
    renderMigrateBox();
  };
  const srcRow = migrateEl('div', 'migrate-row');
  srcRow.append(
    migrateEl('span', 'muted small', '源 AI：'),
    makeMigrateSelect(
      sources.map((id) => ({ id, label: modelDisplayName(id) })),
      migrateState.source,
      onSource,
    ),
  );
  const tgtRow = migrateEl('div', 'migrate-row');
  tgtRow.append(
    migrateEl('span', 'muted small', '目标 AI：'),
    makeMigrateSelect(
      options.filter((o) => o.id !== migrateState.source),
      migrateState.target,
      onTarget,
    ),
  );
  box.append(srcRow, tgtRow);

  const list = migrateEl('div', 'local-list');
  const corr = listCorrections(migrateState.source);
  const kb = listKnowledge(migrateState.source);
  for (const c of corr) list.append(migrateItemRow({ kind: 'correction', id: c.id, text: c.text, question: c.question, from: c.from }));
  for (const k of kb) list.append(migrateItemRow({ kind: 'knowledge', id: k.id, text: k.text, question: k.question, from: k.from }));
  if (!corr.length && !kb.length) list.append(migrateEl('p', 'muted small', '该 AI 没有可迁移的更正或知识库条目。'));
  box.append(list);

  const training = document.createElement('label');
  training.className = 'migrate-training';
  const trainBox = document.createElement('input');
  trainBox.type = 'checkbox';
  trainBox.id = 'migrateTraining';
  trainBox.checked = true;
  training.append(trainBox, document.createTextNode(' 同时迁移对话训练样本（问题→回答，之后可在目标 AI 导出固化数据；取消则只作为提示词内容）'));
  box.append(training);

  const go = document.createElement('button');
  go.type = 'button';
  go.className = 'btn ghost small';
  go.dataset.action = 'migrate-run';
  go.textContent = '迁移所选到目标 AI';
  go.disabled = !migrateState.target;
  go.addEventListener('click', () => runMigration());
  box.append(go);
}

/** 执行迁移：勾选了更正时先做管理员密钥验证。 */
async function runMigration() {
  if (!requireAdmin('跨模型迁移')) return;
  const { source, target } = migrateState;
  if (!source || !target || source === target) {
    showStatus('请选择不同的源 AI 与目标 AI。', { error: true });
    return;
  }
  const checked = [...els.migrateBox.querySelectorAll('input[data-kind]:checked')];
  const corrections = checked.filter((i) => i.dataset.kind === 'correction').map((i) => i.dataset.id);
  const knowledge = checked.filter((i) => i.dataset.kind === 'knowledge').map((i) => i.dataset.id);
  if (!corrections.length && !knowledge.length) {
    showStatus('请先勾选要迁移的条目。', { error: true });
    return;
  }
  const includeTraining = !!els.migrateBox.querySelector('#migrateTraining')?.checked;
  const from = modelDisplayName(source);
  const to = modelDisplayName(target);
  if (corrections.length) {
    const texts = listCorrections(source).filter((c) => corrections.includes(c.id)).map((c) => `- ${c.text}`);
    const ok = await verifyCorrectionKey(
      `迁移 ${corrections.length} 条更正（${from} → ${to}）：\n${texts.join('\n')}`,
      `这些更正将成为「${to}」的底层设定（无条件采纳）。输入管理员密钥验证后执行；取消则不迁移。`,
    );
    if (!ok) return;
  }
  let result;
  try {
    result = migrateAdaptation(source, target, { corrections, knowledge }, {
      includeTraining,
      allowCorrections: corrections.length > 0,
    });
  } catch (err) {
    showStatus(`迁移失败：${err.message}`, { error: true });
    return;
  }
  const missing = result.missing ? `，源中已不存在 ${result.missing} 条` : '';
  showStatus(
    `已从「${from}」迁移到「${to}」：更正 ${result.corrections} 条，知识库 ${result.knowledge} 条，重复跳过 ${result.duplicate} 条${missing}。`,
  );
  renderAdaptList();
}

/** AI 适应记录：管理员可查看、删除（仅本机）。 */
function renderAdaptList() {
  const admin = isAdmin();
  renderExportParamsOption();
  renderMigrateBox();
  const list = els.adaptList;
  if (!list) return;
  list.replaceChildren();
  if (MODE !== 'webllm' || !admin) {
    els.adaptHint.textContent = MODE !== 'webllm' ? '服务端模式下不记录 AI 适应反馈。' : 'AI 适应记录仅管理员可见。';
    return;
  }
  els.adaptHint.textContent =
    '管理员在 AI 回复之后发送的消息会记为该 AI 的反馈（仅本机，不上传）；' +
    '以“参数：temperature=0.2, maxTokens=800”开头的一行可调整该 AI 的参数；' +
    '反馈同时写入该 AI 的本地知识库（最高优先级，仅本机，不上传，不随导出文件保存）；' +
    '以“更正：”开头的一行需输入管理员密钥验证后，才会作为该 AI 的底层设定，无条件采纳（仅本机）；' +
    '“导出固化数据”可把本 AI 的更正与知识库导出为训练文件（可选，需在本机用 training/ojtrain.py solidify 训练，需要 GPU；只包含记录了对应问题的条目）。';
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
    const kb = listKnowledge(id);
    const corrections = listCorrections(id);
    const head = document.createElement('div');
    head.className = 'local-row';
    const name = document.createElement('div');
    name.className = 'local-name';
    name.textContent = `${modelDisplayName(id)}（${notes.length} 条${corrections.length ? `，更正 ${corrections.length} 条` : ''}${kb.length ? `，知识库 ${kb.length} 条` : ''}${Object.keys(params).length ? '，有参数' : ''}）`;
    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'btn ghost small';
    clear.dataset.action = 'adapt-clear';
    clear.dataset.id = id;
    clear.textContent = '清空';
    const solidify = document.createElement('button');
    solidify.type = 'button';
    solidify.className = 'btn ghost small';
    solidify.dataset.action = 'adapt-solidify';
    solidify.dataset.id = id;
    solidify.textContent = '导出固化数据';
    // 多个按钮放进同一个容器，避免它们在网格的同一单元格中重叠
    const actions = document.createElement('div');
    actions.className = 'local-actions';
    actions.append(solidify, clear);
    head.append(name, actions);
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
    if (corrections.length) {
      const title = document.createElement('div');
      title.className = 'muted small local-meta';
      title.textContent = '更正（底层设定，无条件采纳，仅本机，经管理员密钥验证）：';
      list.append(title);
      for (const c of corrections) {
        const row = document.createElement('div');
        row.className = 'local-row';
        const t = document.createElement('div');
        t.className = 'muted small local-meta';
        t.textContent = `更正：${c.text}${c.from ? `（迁入自 ${modelDisplayName(c.from)}）` : ''}`;
        const del = document.createElement('button');
        del.type = 'button';
        del.className = 'btn ghost small';
        del.dataset.action = 'correction-delete';
        del.dataset.id = id;
        del.dataset.entry = c.id;
        del.textContent = '删除';
        row.append(t, del);
        list.append(row);
      }
    }
    if (kb.length) {
      const title = document.createElement('div');
      title.className = 'muted small local-meta';
      title.textContent = '本地知识库（最高优先级，仅本机）：';
      list.append(title);
      for (const k of kb) {
        const row = document.createElement('div');
        row.className = 'local-row';
        const t = document.createElement('div');
        t.className = 'muted small local-meta';
        t.textContent = `${k.text}${k.from ? `（迁入自 ${modelDisplayName(k.from)}）` : ''}`;
        const del = document.createElement('button');
        del.type = 'button';
        del.className = 'btn ghost small';
        del.dataset.action = 'kb-delete';
        del.dataset.id = id;
        del.dataset.entry = k.id;
        del.textContent = '删除';
        row.append(t, del);
        list.append(row);
      }
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
    ['temperature', 'maxTokens', 'contextWindow', 'thinkingWindow']
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

/**
 * 撤回参数覆盖：任何人都可以在“更改权限”面板中恢复某个参数为默认值（只影响本浏览器）。
 * 说明（notes）仍只能由管理员删除。
 */
function revertParam(modelId, key) {
  if (!requireAdmin('恢复默认参数')) return;
  if (!resetParam(modelId, key)) return;
  refreshParamViews();
  showStatus(`已恢复「${modelDisplayName(modelId)}」的 ${key} 为默认值（仅本机，从下一次回答起生效）。`);
}

function revertAllParams(modelId) {
  if (!requireAdmin('恢复默认参数')) return;
  const n = resetAllParams(modelId);
  if (!n) return;
  refreshParamViews();
  showStatus(`已恢复「${modelDisplayName(modelId)}」的全部 ${n} 项参数为默认值（仅本机，从下一次回答起生效）。`);
}

function refreshParamViews() {
  renderAdaptList();
  renderParamOverrides();
}

/** 参数摘要，例如 “temperature=0.2，maxTokens=800”。 */
function paramSummary(params) {
  return Object.entries(params).map(([k, v]) => `${k}=${v}`).join('，');
}

/**
 * “更改权限”面板中，所有用户都能看到当前模型的本机参数覆盖，并可撤回。
 * 参数对同一浏览器的所有用户生效，因此这里不限定管理员。
 */
function renderParamOverrides() {
  const box = els.paramOverrides;
  if (!box) return;
  box.replaceChildren();
  // 普通用户看不到任何参数改动（参数仍会生效，但界面不显示）
  box.hidden = true;
  if (MODE !== 'webllm' || !state.backend || !isAdmin()) return;
  box.hidden = false;
  const modelId = state.backend.modelId;
  const overrides = sanitizeParams(listAdaptation(modelId).params) || {};
  const head = document.createElement('p');
  head.className = 'muted small';
  head.textContent = `当前模型「${modelDisplayName(modelId)}」的自定义参数（对本浏览器所有用户生效）：`;
  box.append(head);
  const keys = Object.keys(overrides);
  if (!keys.length) {
    const none = document.createElement('p');
    none.className = 'muted small';
    none.textContent = '没有自定义参数，使用默认值。';
    box.append(none);
    return;
  }
  for (const key of keys) {
    const row = document.createElement('div');
    row.className = 'local-row';
    const t = document.createElement('div');
    t.className = 'small';
    t.textContent = `${key} = ${overrides[key]}（本机覆盖）`;
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn ghost small';
    btn.dataset.action = 'param-revert';
    btn.dataset.id = modelId;
    btn.dataset.key = key;
    btn.textContent = '恢复默认';
    row.append(t, btn);
    box.append(row);
  }
  if (keys.length > 1) {
    const all = document.createElement('button');
    all.type = 'button';
    all.className = 'btn ghost small';
    all.dataset.action = 'param-revert-all';
    all.dataset.id = modelId;
    all.textContent = '全部恢复默认';
    box.append(all);
  }
}

/** 导出选项：有自定义参数时可勾选保存；没有时禁用。 */
function renderExportParamsOption() {
  const admin = isAdmin() && MODE === 'webllm';
  els.exportParamsLabel.hidden = !admin;
  if (!admin) {
    els.exportWithParams.disabled = true;
    els.exportWithParams.checked = false;
    els.exportParamsText.textContent = '';
    return;
  }
  const overrides = sanitizeParams(listAdaptation(state.backend?.modelId).params);
  const has = !!overrides;
  els.exportWithParams.disabled = !has;
  els.exportWithParams.checked = has;
  els.exportParamsText.textContent = has
    ? `同时保存自定义参数（${paramSummary(overrides)}）`
    : '当前模型没有自定义参数可保存';
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

els.thinkToggle.addEventListener('change', () => {
  state.backend.setThinking(els.thinkToggle.checked);
  // 思考窗口在加载模型时确定：切换后下一次发送会按新窗口重新加载模型
  const eff = effectiveParams(state.backend.modelId);
  flashStatus(
    els.thinkToggle.checked
      ? `已开启思考模式：推理模型会先展示思考过程（可折叠），回答更充分但更慢。下次发送时将按 ${eff.contextWindow + eff.thinkingWindow} 窗口重新加载模型。`
      : `已关闭思考模式：推理模型直接给出回答，速度更快。下次发送时将按 ${eff.contextWindow} 窗口重新加载模型。`,
  );
});

els.modelSelect.addEventListener('change', () => {
  state.backend.setModel(els.modelSelect.value);
  // 切换模型即更改默认模型：保存在本浏览器，下次打开时使用
  saveDefaultModel(els.modelSelect.value);
  refreshThinkToggle();
  renderParamOverrides();
  renderExportParamsOption();
  const label = els.modelSelect.selectedOptions[0]?.textContent || els.modelSelect.value;
  // 超大模型：显存需求远超普通显卡（如 RTX 4060 的 8 GB），需要提前提醒
  if (label.includes('超大')) {
    showStatus(
      `${label} 为超大模型，需要约 30 GB 以上显存，8 GB 显卡（如 RTX 4060）无法加载或运行；` +
        '首次使用还需下载数十 GB 的文件。发送消息时可能加载失败。',
      { error: true },
    );
    return;
  }
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
  if (btn && btn.dataset.action === 'adapt-solidify') return exportSolidifyData(btn.dataset.id);
  if (btn && btn.dataset.action === 'kb-delete') return deleteKnowledgeEntry(btn.dataset.id, btn.dataset.entry);
  if (btn && btn.dataset.action === 'correction-delete') return deleteCorrection(btn.dataset.id, btn.dataset.entry);
  if (btn && btn.dataset.action === 'adapt-reset') return revertParam(btn.dataset.id, btn.dataset.key);
  if (btn && btn.dataset.action === 'param-revert') return revertParam(btn.dataset.id, btn.dataset.key);
  if (btn && btn.dataset.action === 'param-revert-all') return revertAllParams(btn.dataset.id);
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
    // 启动时使用用户上次保存的默认模型（若仍在下拉框中），否则使用配置中的 defaultModel
    const availableIds = [...new Set([...state.backend.models.map((m) => m.id), ...loadLocalModels().map((e) => e.id)])];
    state.backend.setModel(
      resolveStartupModel({
        availableIds,
        defaultModel: state.appConfig.webllm.defaultModel,
        saved: getSavedDefaultModel(),
      }),
    );
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
