import 'katex/dist/katex.min.css';
import './style.css';
import DOMPurify from 'dompurify';
import { renderMarkdown } from './markdown.js';
import { hasThinking, splitThinking } from './think.js';
import { setThinkingPref, truncateThinking } from './reasoning.js';
import { trimRepetition } from './repetition.js';
import { isAllowedOrigin } from './origin.js';
import { ApiError, checkSession, fetchConfig, getToken, login, setToken } from './api.js';
import { createBackend } from './engines/backend.js';
import { getSavedDefaultModel, resolveStartupModel, saveDefaultModel } from './default-model.js';
import { addCheckRule, checkRuleError, findViolations, listCheckRules } from './answer-check.js';
import { checkWebGPU, getModelRecords, releaseEngine, WEBGPU_MISSING } from './engines/webllm.js';
import { formatGB, imageModelsOf, isSelectable, sourceSizes, sourceUrls, totalBytes } from './image/catalog.js';
import { createImageClient } from './image/client.js';
import {
  addImageCorrection,
  addImageHint,
  addImageKnowledge,
  buildImagePrompt,
  copyIntoImageModel,
  effectiveInputLimit,
  imageAdminModelIds,
  clearImageParams,
  effectiveImageSettings,
  getImageSettings,
  imageLeadTexts,
  parseImageParams,
  removeImageCorrection,
  removeImageHint,
  removeImageKnowledge,
  setImageParams,
  stripImageParamLine,
} from './image/params.js';
import { randomSeed } from './image/rng.js';
import { joinLeadTexts, MAX_PROMPT_CHUNKS } from './image/prompt-budget.js';
import { addGalleryImage, clearGallery, deleteGalleryImage, listGallery } from './image/gallery-store.js';
import { buildGalleryArchive, readGalleryArchive } from './image/gallery-archive.js';
import {
  buildImageModelFile,
  deleteImageModelFiles,
  downloadImageModel,
  imageCacheStatus,
  importImageModelFile,
} from './image/model-files.js';
import { DEFAULT_SIZE, RATIOS, sizeForRatio } from './image/sizes.js';
import {
  buildModelFile,
  importModelFile,
  listCachedModelIds,
  modelFileName,
  readModelFileHeader,
  removeLocalModelFiles,
} from './model-file.js';
import {
  isPickerCancelled,
  savePickerOptions,
  supportsSaveLocationPicker,
} from './save-location.js';
import { adminExpiry, grantAdmin, hasAccessGate, isAdmin, revokeAdmin, verifyAccessPassword } from './access.js';
import { addLocalModel, loadLocalModels, localRecords, removeLocalModel } from './local-models.js';
import { pendingModels } from './pending-models.js';
import { DEFAULT_PRESET_ID, findPreset, getPresetId, layerParams, PRESETS, presetOverrides, setPresetId } from './presets.js';
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
  stripCorrectionLines,
  suggestParams,
  mechanicalReply,
} from './adaptation.js';
import { buildConversationMarkdown, conversationFilename } from './chat-export.js';
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
  trimSessionMessages,
  titleFrom,
  UNFOLD_STEP,
  undoableTurnIndex,
  undoLastTurn,
  newId,
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
  presetSelect: $('#presetSelect'),
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
  undoBtn: $('#undoBtn'),
  stopBtn: $('#stopBtn'),
  thinkToggleLabel: $('#thinkToggleLabel'),
  thinkToggle: $('#thinkToggle'),
  slidingToggleLabel: $('#slidingToggleLabel'),
  slidingToggle: $('#slidingToggle'),
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
  exportChatBtn: $('#exportChatBtn'),
  paramOverrides: $('#paramOverrides'),
  exportWithParams: $('#exportWithParams'),
  exportParamsText: $('#exportParamsText'),
  exportParamsLabel: $('#exportParamsLabel'),
  adaptList: $('#adaptList'),
  migrateBox: $('#migrateBox'),
  imageModeBtn: $('#imageModeBtn'),
  imagePanel: $('#imagePanel'),
  imageModelSelect: $('#imageModelSelect'),
  imageRatio: $('#imageRatio'),
  imageWidth: $('#imageWidth'),
  imageHeight: $('#imageHeight'),
  imageSteps: $('#imageSteps'),
  imageGuidance: $('#imageGuidance'),
  imageSeed: $('#imageSeed'),
  imageEffective: $('#imageEffective'),
  imageForm: $('#imageForm'),
  imagePrompt: $('#imagePrompt'),
  imageGenerateBtn: $('#imageGenerateBtn'),
  imageCancelBtn: $('#imageCancelBtn'),
  imageStatus: $('#imageStatus'),
  imageProgress: $('#imageProgress'),
  imageProgressFill: $('#imageProgressFill'),
  imageGallery: $('#imageGallery'),
  imageModelFiles: $('#imageModelFiles'),
  imageMigrateBox: $('#imageMigrateBox'),
  imageFilesStatus: $('#imageFilesStatus'),
  imageDownloadBtn: $('#imageDownloadBtn'),
  imageExportModelBtn: $('#imageExportModelBtn'),
  imageImportModelInput: $('#imageImportModelInput'),
  imageDeleteFilesBtn: $('#imageDeleteFilesBtn'),
  imageDownloadProgress: $('#imageDownloadProgress'),
  imageDownloadFill: $('#imageDownloadFill'),
  imageGalleryCount: $('#imageGalleryCount'),
  imageExportGalleryBtn: $('#imageExportGalleryBtn'),
  imageImportGalleryInput: $('#imageImportGalleryInput'),
  imageClearGalleryBtn: $('#imageClearGalleryBtn'),
  imageAdmin: $('#imageAdmin'),
  imageHintInput: $('#imageHintInput'),
  imageHintAdd: $('#imageHintAdd'),
  imageParamsClear: $('#imageParamsClear'),
  imageHintList: $('#imageHintList'),
  imageCorrectionInput: $('#imageCorrectionInput'),
  imageCorrectionAdd: $('#imageCorrectionAdd'),
  imageCorrectionList: $('#imageCorrectionList'),
  imageKnowledgeInput: $('#imageKnowledgeInput'),
  imageKnowledgeAdd: $('#imageKnowledgeAdd'),
  imageKnowledgeList: $('#imageKnowledgeList'),
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
  if (msg.looped) html += '<p class="note">（检测到重复输出，已自动停止；以上为第一份内容）</p>';
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
  // 标记错误针对即将回答的 AI（下拉框当前选中的模型），而不是这条回复的 AI
  const target = state.backend.modelId;
  els.markWrong.value = presetWrong;
  els.markRight.value = '';
  els.markKey.value = '';
  els.markError.textContent = '';
  els.markDesc.textContent =
    `为即将回答的「${modelDisplayName(target)}」标记错误说法。之后该 AI 的回答若再出现错误说法，会在回答下方提示正确说法。` +
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
      const res = addCheckRule(target, { wrong, right });
      if (res.error) {
        els.markError.textContent = res.error;
        return;
      }
      // 正确说法同时作为更正（已验证密钥，仅本机）；换行压成一行，符合更正格式
      recordFeedback(target, `更正：${right.replace(/\s*\n\s*/g, ' ')}`, {
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
  refreshUndoButton();
}

/** 撤回按钮：只有末尾是一轮完整（或失败）的对话，且没有在生成时才可用。 */
function refreshUndoButton() {
  const can = !state.streaming && undoableTurnIndex(activeSession()) >= 0;
  els.undoBtn.hidden = !can;
}

/**
 * 撤回上一轮：删除最后一问一答，问题放回输入框（不发送）。
 * 若输入框已有内容，则不撤回，避免覆盖用户正在编辑的文字。
 */
function undoTurn() {
  if (state.streaming) return;
  if (els.input.value.trim()) {
    flashStatus('输入框里还有未发送的内容，请先清空或发送，再撤回。');
    return;
  }
  const session = activeSession();
  const turn = undoLastTurn(session);
  if (!turn) {
    flashStatus('没有可以撤回的对话。');
    return;
  }
  session.updatedAt = Date.now();
  persist();
  refreshAll();
  els.input.value = turn.question;
  autoGrow();
  els.input.focus();
  flashStatus(
    turn.recorded
      ? '已撤回上一轮，问题已放回输入框。该轮的反馈、更正或参数修改已记录在本机，撤回不会回滚，请到适应面板中删除。'
      : '已撤回上一轮，问题已放回输入框（未发送）。',
  );
}

// ---------- 发送 / 停止 ----------
function setBusy(busy) {
  els.sendBtn.hidden = busy;
  els.stopBtn.hidden = !busy;
  refreshUndoButton();
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
 * 管理员（且为本机模式）在对话中的修改（参数行、参数关键词、附加说明、更正、标记错误）
 * 一律针对“即将回答的 AI”，即下拉框当前选中的模型（state.backend.modelId），而不是上一条回复的 AI。
 * 这些修改仅保存在本机，并作为该 AI 之后回答的附加说明或参数。
 */

/** 某条回复之前最近的用户消息（即它所回答的问题），用于固化训练数据。 */
function questionBefore(session, target) {
  const msgs = session?.messages || [];
  for (let j = msgs.indexOf(target) - 1; j >= 0; j -= 1) {
    if (msgs[j].role === 'user' && msgs[j].content) return msgs[j].content;
  }
  return '';
}

/** 某模型当前生效的参数：本机覆盖优先，否则使用配置默认值。 */
function effectiveParams(modelId) {
  const d = state.appConfig.webllm;
  return layerParams(
    {
      temperature: d.temperature,
      repetitionPenalty: d.repetitionPenalty,
      maxTokens: d.maxTokens,
      contextWindow: d.contextWindowTokens,
      slidingWindow: d.slidingWindowTokens,
      thinkingWindow: d.thinkingWindowTokens,
    },
    presetOverrides(),
    getModelParams(modelId),
  );
}

/** 当前生效的数值来源（用于参数详情）：管理员参数 > 预设 > 默认。 */
function paramSourceOf(key, modelId) {
  if (key in getModelParams(modelId)) return '本机覆盖';
  if (key in presetOverrides()) return `预设：${findPreset(getPresetId())?.label || ''}`;
  return '默认';
}

/** 参数变更的展示格式：key 原值→新值（原值为空时显示“默认”）。 */
function fmtChange(c) {
  return `${c.key} ${c.from ?? '默认'}→${c.to}`;
}

/**
 * 记录管理员在本条消息中的修改，并直接生效（不需要确认，不调用模型）。
 * 返回 {target, label, changes, extra, parts}：changes 为实际变更的参数 [{key, from, to}]，
 * extra 为同时记录的附加说明或更正描述。非管理员、非本机模式或没有 AI 时返回空结果。
 */
function recordAdminFeedback(session, text, { allowCorrections = false } = {}) {
  const none = { target: null, label: '', changes: [], extra: [], parts: [] };
  if (MODE !== 'webllm' || !isAdmin() || !state.backend) return none;
  const target = state.backend.modelId;
  if (!target) return none;
  const label = modelDisplayName(target);
  // 附加说明与更正关联的问题：会话中最近一条已完成回答所回答的问题
  const lastReply = [...(session?.messages || [])].reverse().find((m) => m.role === 'assistant' && !m.fixed && m.content && !m.error);
  const question = lastReply ? questionBefore(session, lastReply) : '';
  const rec = recordFeedback(target, text, { allowCorrections, question });
  const parts = [];
  const changes = [];
  const extra = [];
  if (rec?.note) {
    parts.push('已记录反馈');
    extra.push('记录了附加说明');
  }
  if (rec?.corrections?.length) {
    parts.push(`已记录 ${rec.corrections.length} 条更正（已验证密钥，作为该 AI 的底层设定，无条件采纳）`);
    extra.push(`记录了 ${rec.corrections.length} 条更正`);
  }
  if (rec?.params) {
    for (const c of rec.changes || []) changes.push({ key: c.key, from: c.from, to: c.to });
    parts.push('已更新参数 ' + (changes.length ? changes.map(fmtChange).join(', ') : '（未变化）'));
  }
  // 没有显式参数行、也没有更正时，按关键词直接修改参数（不需要确认，不调用模型）
  if (!rec?.params && !rec?.corrections?.length) {
    const current = effectiveParams(target);
    for (const f of suggestParams(text, current)) {
      const to = applyParamSuggestion(target, f.key, f.value, { reason: f.reason });
      if (to !== null) changes.push({ key: f.key, from: current[f.key], to });
    }
    if (changes.length) parts.push('已按关键词修改参数 ' + changes.map(fmtChange).join(', '));
  }
  if (parts.length) showStatus(`${parts.join('，')}（仅本机，对「${label}」生效，不上传）。`);
  renderAdaptList();
  renderParamOverrides();
  return { target, label, changes, extra, parts };
}

/**
 * 参数改变后释放多余显存：若已加载的引擎模型或窗口与当前设置不一致，则释放引擎（下次发送时按新设置加载）。
 * 返回是否真的释放了。只用于不调用模型的回复；需要回答的消息会在加载新引擎时替换旧引擎。
 */
async function releaseStaleEngine() {
  if (MODE !== 'webllm' || !state.backend?.releaseIfStale) return false;
  try {
    return await state.backend.releaseIfStale();
  } catch (err) {
    console.warn('释放显存失败', err);
    return false;
  }
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
  // 超过上限时按丢弃粒度整段删除最早的消息，保持之后几轮的前缀不变（减少重新预填）
  trimSessionMessages(session.messages, MAX_MESSAGES_PER_SESSION, state.appConfig.webllm.historyDropChunk ?? 1);
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

  // 更正需要管理员密钥验证（仅 WebLLM 本机模式、管理员）；更正针对即将回答的 AI
  let allowCorrections = false;
  if (MODE === 'webllm' && isAdmin() && parseCorrections(text).length) {
    const desc = `这条消息包含更正，将作为「${modelDisplayName(state.backend.modelId)}」（即将回答的 AI）的底层设定记录（仅本机）。输入管理员密钥验证后保存；取消则不发送。`;
    const ok = await verifyCorrectionKey(text, desc);
    if (!ok) {
      if (!els.input.value) {
        els.input.value = text;
        autoGrow();
      }
      return;
    }
    allowCorrections = true;
  }

  const session = ensureSession();
  const isFirst = session.messages.length === 0;
  const userMsg = { role: 'user', content: text, ts: Date.now() };
  const feedback = recordAdminFeedback(session, text, { allowCorrections });
  // 记录过反馈、更正或参数修改的轮次：撤回时需要提示（撤回不回滚这些记录）
  if (feedback.parts?.length) userMsg.recorded = true;
  // 机械识别：管理员消息只是参数调整指令（已直接修改参数、没有提问）时，不调用模型，给出固定提示
  if (mechanicalReply(text, feedback)) {
    const released = await releaseStaleEngine();
    showFixedReply(session, isFirst, userMsg, mechanicalReply(text, { ...feedback, released }));
    return;
  }
  // 记录本次回答所用的 AI（名称快照保存，之后删除或改名模型也能正确标注）
  const modelId = state.backend.modelId;
  const modelLabel = modelDisplayName(modelId);
  const assistantMsg = { role: 'assistant', content: '', ts: Date.now(), model: modelId, modelLabel };
  session.messages.push(userMsg, assistantMsg);
  // 超过上限时按丢弃粒度整段删除最早的消息，保持之后几轮的前缀不变（减少重新预填）
  trimSessionMessages(session.messages, MAX_MESSAGES_PER_SESSION, state.appConfig.webllm.historyDropChunk ?? 1);
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
      // 加载可以被停止按钮取消（signal），长时间没有进展会超时并提示重试
      const loaded = await state.backend.ensureReady(({ progress, text: t }) => {
        if (t) progressText = t;
        const pct = Math.round(progress * 100);
        showStatus(`正在加载模型 ${pct}%${t ? `：${t}` : ''}`, { progress });
      }, { signal: controller.signal });
      await refreshIdleStatus();
      refreshModelCacheStatus();
      // 输出自检的结果（异常、已切换到 fp32 等）
      if (loaded?.note) showStatus(loaded.note, { error: loaded.healthy === false });
    }

    let loopDetected = false;
    for await (const delta of state.backend.stream({
      messages: context,
      signal: controller.signal,
      onLoop: () => {
        loopDetected = true;
      },
    })) {
      assistantMsg.content += delta;
      scheduleRender();
    }
    if (frame) cancelAnimationFrame(frame);
    if (controller.signal.aborted) assistantMsg.stopped = true;
    if (loopDetected) {
      // 检测到重复输出：只保留第一份内容保存到历史（之后几轮不会被模型复制重复的部分）
      assistantMsg.content = trimRepetition(assistantMsg.content);
      assistantMsg.looped = true;
    }
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
    // 只清理本次发送自己的状态：若已被强制停止并开始了新的发送，不能清掉新一轮的状态
    if (state.streaming?.controller === controller) {
      state.streaming = null;
      setBusy(false);
    }
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
  const s = state.streaming;
  if (!s) return;
  if (!s.controller.signal.aborted) {
    // 第一次点击：请求停止，生成会在下一个 token 之前结束
    s.controller.abort();
    return;
  }
  // 已经请求过停止，但模型仍没有结束（例如显卡卡住）：由用户决定，立即释放模型并解除等待
  forceStop(s);
}

/** 强制停止：终止模型 worker，把当前回答标记为已停止，并恢复输入状态（不设时限，只由用户触发）。 */
function forceStop(s) {
  state.backend.forceRelease?.();
  s.assistant.stopped = true;
  finalizeAssistant(s.assistant);
  state.streaming = null;
  setBusy(false);
  persist();
  renderSessionList();
  renderMessages();
  showStatus('模型没有响应，已强制停止并释放模型。下次发送时会重新加载。', { error: true });
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
/** 导出当前对话为 Markdown 文件（所有用户可用，仅在本机生成，不上传）。 */
function exportCurrentChat() {
  const session = activeSession();
  const hasContent = !!session && session.messages.some((m) => m.content && !m.error);
  if (!hasContent) {
    showStatus('当前没有可导出的对话。', { error: true });
    return;
  }
  const md = buildConversationMarkdown(session);
  downloadBlob(new Blob([md], { type: 'text/markdown;charset=utf-8' }), conversationFilename(session.title));
  showStatus('已导出当前对话（Markdown 文件，仅保存在本机，不上传）。');
}
els.exportChatBtn.addEventListener('click', exportCurrentChat);

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

/** 滑动窗口开关：所有模型可用；偏好在本浏览器中保存（默认关闭）。 */
function refreshSlidingToggle() {
  const show = MODE === 'webllm';
  els.slidingToggleLabel.hidden = !show;
  els.slidingToggle.checked = state.backend?.slidingEnabled() ?? false;
}

/** 当前生效的上下文窗口（滑动窗口开启时为滑动窗口大小，否则为上下文窗口）。 */
function activeContextWindow(eff) {
  return state.backend?.slidingEnabled() ? eff.slidingWindow : eff.contextWindow;
}

/** 对话预设下拉框（按浏览器保存；所有用户可用）。 */
function renderPresetOptions() {
  if (!els.presetSelect) return;
  els.presetSelect.replaceChildren(...PRESETS.map((p) => makeOption(p.id, p.label)));
  els.presetSelect.value = getPresetId();
  els.presetSelect.title = findPreset(getPresetId())?.note || '';
}

/**
 * 选择预设：机械修改，立即生效，不调用模型。
 * 思考开关是全局偏好，因此预设的 thinking 在这里写入偏好；之后仍可手动切换。
 */
function applyPreset(id) {
  if (!setPresetId(id)) return;
  const preset = findPreset(id) || findPreset(DEFAULT_PRESET_ID);
  setThinkingPref(preset.thinking);
  renderPresetOptions();
  refreshThinkToggle();
  refreshParamViews();
  flashStatus(`已切换到预设「${preset.label}」：${preset.note} 下次发送时按新参数生效。`);
}

function renderModelOptions(selectId = state.backend.modelId) {
  // 官方模型的备注（参数量级、显存估计）放在悬停提示中，来自 config.json 的 webllm.models[].note
  const notes = new Map((state.appConfig?.webllm?.models || []).map((m) => [m.id, m.note || '']));
  const children = state.backend.models.map((m) => {
    const opt = makeOption(m.id, m.label);
    if (notes.get(m.id)) opt.title = notes.get(m.id);
    return opt;
  });
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
  refreshSlidingToggle();
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
    // 先让用户选择保存的盘符与文件夹：必须在点击后立即弹出（之后打包可能很久，浏览器会撤销授权）
    let handle = null;
    if (supportsSaveLocationPicker()) {
      try {
        handle = await window.showSaveFilePicker(savePickerOptions(modelFileName(record.model_id)));
      } catch (err) {
        if (isPickerCancelled(err)) return showStatus('已取消导出。');
        // 其他错误（例如嵌入到其他网站的 iframe 中被浏览器禁止）：回退到下载目录，不让导出失败
        handle = null;
      }
    }
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
    if (handle) {
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
    } else {
      // 不支持选择位置的浏览器：使用浏览器的下载目录（可在浏览器设置中选择“每次询问保存位置”）
      downloadBlob(blob, filename);
    }
    const savedName = handle?.name || filename;
    const where = handle ? '已保存到你选择的位置' : '已下载到浏览器的下载目录';
    const paramNote = withParams ? `，已保存自定义参数（${paramSummary(overrides)}）` : '';
    showStatus(`${where}：${savedName}（${formatBytes(size)}，${entryCount} 个文件${paramNote}）。导入后即可离线使用。`);
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
  refreshSlidingToggle();
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
  renderParamOverrides();
}

/** 跨模型迁移的选择状态（仅本页面会话有效，不保存）。聊天页与生图管理员面板各自一份。 */
const migrateState = { source: '', target: '' };
const imageMigrateState = { source: '', target: '' };

function migrateEl(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
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

/** 下拉框中的全部聊天 AI（官方、待转换、新建AI）。 */
function allModelOptions() {
  return [...els.modelSelect.options].map((o) => ({ id: o.value, label: o.dataset.label || o.value }));
}

/** 全部生图 AI 作为迁移目标（带前缀，便于与聊天 AI 区分）。 */
function imageTargetOptions() {
  return imageModels().map((m) => ({ id: m.id, label: `[生图] ${m.label}` }));
}

function isImageAdminId(id) {
  return !!imageModelById(id);
}

/** 迁移中显示的 AI 名称：生图 AI 用其名称，聊天 AI 用其显示名。 */
function adminDisplayName(id) {
  const im = imageModelById(id);
  return im ? `[生图] ${im.label}` : modelDisplayName(id);
}

/** 可作为来源的 AI：有更正、知识库或说明的聊天 AI，以及有更正、知识库或风格提示的生图 AI。 */
function migrationSourceIds() {
  return [...new Set([...adaptedModelIds(), ...imageAdminModelIds()])];
}

/**
 * 来源可选的目标：来自生图 AI 的只能迁入生图 AI（风格提示等只属于生图）；
 * 来自聊天 AI 的可迁入聊天 AI 或生图 AI（说明会作为生图 AI 的知识库条目）。
 */
function migrationTargetChoices(sourceId) {
  if (isImageAdminId(sourceId)) return imageTargetOptions().filter((o) => o.id !== sourceId);
  return [...allModelOptions(), ...imageTargetOptions()].filter((o) => o.id !== sourceId);
}

/** 来源 AI 中可迁移的条目，按类别分组。 */
function migratableItems(sourceId) {
  if (isImageAdminId(sourceId)) {
    const s = getImageSettings(sourceId);
    return {
      corrections: s.corrections.map((c) => ({ kind: 'correction', id: c.id, text: c.text })),
      knowledge: s.knowledge.map((k) => ({ kind: 'knowledge', id: k.id, text: k.text })),
      notes: [],
      hints: s.hints.map((h) => ({ kind: 'hint', id: h.id, text: h.text })),
    };
  }
  return {
    corrections: listCorrections(sourceId).map((c) => ({
      kind: 'correction',
      id: c.id,
      text: c.text,
      question: c.question,
      from: c.from,
    })),
    knowledge: listKnowledge(sourceId).map((k) => ({
      kind: 'knowledge',
      id: k.id,
      text: k.text,
      question: k.question,
      from: k.from,
    })),
    notes: listAdaptation(sourceId).notes.map((n) => ({ kind: 'note', id: n.id, text: n.text })),
    hints: [],
  };
}

function migrateItemRow({ kind, id, text, question, from }) {
  const row = migrateEl('label', 'local-row migrate-item');
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = true;
  box.dataset.kind = kind;
  box.dataset.id = id;
  const names = {
    correction: '更正',
    knowledge: '知识库',
    note: '说明（迁入生图 AI 后作为知识库条目）',
    hint: '风格提示',
  };
  const label = migrateEl('div', 'muted small local-meta', `${names[kind] || kind}：${text}`);
  const tags = [question ? '含训练样本' : '', from ? `迁入自 ${modelDisplayName(from)}` : ''].filter(Boolean);
  if (tags.length) label.textContent += `（${tags.join('，')}）`;
  row.append(box, label);
  return row;
}

/**
 * 迁移框（聊天页与生图管理员面板共用，仅管理员、仅本机）。
 * st：选择状态 {source, target}。fixedTarget：生图管理员面板中给出的目标生图 AI（目标固定，不显示目标下拉框）。
 * notify(text, {error})：显示结果的方式（聊天页用状态栏，生图页用生图状态）。onDone：迁移完成后的刷新。
 */
function renderMigrateInto(box, st, { fixedTarget = '', notify = showStatus, onDone = () => {} } = {}) {
  box.replaceChildren();
  const sources = migrationSourceIds().filter((id) => id !== fixedTarget);
  box.append(
    migrateEl(
      'p',
      'muted small local-meta',
      fixedTarget
        ? '从其他 AI 迁移到本生图 AI（仅本机，不上传）：把来源 AI 中勾选的更正、知识库、说明或风格提示复制过来，源 AI 保持不变。迁入的更正与知识库会放在本生图 AI 提示词的最前面。迁移更正需输入管理员密钥验证。'
        : '跨模型迁移（仅本机，不上传）：把源 AI 中勾选的条目复制到目标 AI，目标可以是聊天 AI，也可以是生图 AI。复制而非移动，源 AI 的记录保持不变。迁移更正需输入管理员密钥验证。',
    ),
  );
  if (!sources.length) {
    box.append(migrateEl('p', 'muted small', '暂无可迁移的记录。'));
    return;
  }
  if (!sources.includes(st.source)) st.source = sources[0];
  const choices = fixedTarget ? [] : migrationTargetChoices(st.source);
  if (!fixedTarget && !choices.some((o) => o.id === st.target)) st.target = choices[0]?.id || '';
  const target = fixedTarget || st.target;
  const targetIsImage = isImageAdminId(target);

  const rerender = () => renderMigrateInto(box, st, { fixedTarget, notify, onDone });
  const srcRow = migrateEl('div', 'migrate-row');
  srcRow.append(
    migrateEl('span', 'muted small', '源 AI：'),
    makeMigrateSelect(
      sources.map((id) => ({ id, label: adminDisplayName(id) })),
      st.source,
      (v) => {
        st.source = v;
        st.target = '';
        rerender();
      },
    ),
  );
  box.append(srcRow);
  if (!fixedTarget) {
    const tgtRow = migrateEl('div', 'migrate-row');
    tgtRow.append(
      migrateEl('span', 'muted small', '目标 AI：'),
      makeMigrateSelect(choices, st.target, (v) => {
        st.target = v;
        rerender();
      }),
    );
    box.append(tgtRow);
  }

  const items = migratableItems(st.source);
  const list = migrateEl('div', 'local-list');
  const sourceIsImage = isImageAdminId(st.source);
  const shown = [
    ...items.corrections,
    ...items.knowledge,
    // 说明只迁入生图 AI（作为知识库条目）；聊天 AI 之间的迁移仍只包含更正与知识库
    ...(targetIsImage && !sourceIsImage ? items.notes : []),
    // 风格提示只在生图 AI 之间迁移
    ...(targetIsImage && sourceIsImage ? items.hints : []),
  ];
  for (const it of shown) list.append(migrateItemRow({ ...it, from: it.from }));
  if (!shown.length) list.append(migrateEl('p', 'muted small', '该 AI 没有可迁移的条目。'));
  box.append(list);

  if (!targetIsImage) {
    const training = document.createElement('label');
    training.className = 'migrate-training';
    const trainBox = document.createElement('input');
    trainBox.type = 'checkbox';
    trainBox.id = 'migrateTraining';
    trainBox.checked = true;
    training.append(trainBox, document.createTextNode(' 同时迁移对话训练样本（问题→回答，之后可在目标 AI 导出固化数据；取消则只作为提示词内容）'));
    box.append(training);
  }

  const go = document.createElement('button');
  go.type = 'button';
  go.className = 'btn ghost small';
  go.dataset.action = 'migrate-run';
  go.textContent = '迁移所选到目标 AI';
  go.disabled = !target || !shown.length;
  go.addEventListener('click', () => runMigration(box, st, { fixedTarget, notify, onDone }));
  box.append(go);
}

/** 聊天页的迁移框。 */
function renderMigrateBox() {
  const box = els.migrateBox;
  if (!box) return;
  const show = MODE === 'webllm' && isAdmin();
  box.hidden = !show;
  box.replaceChildren();
  if (!show) return;
  renderMigrateInto(box, migrateState, { notify: showStatus, onDone: () => renderAdaptList() });
}

/** 生图管理员面板中的迁移框（目标为当前生图 AI）。 */
function renderImageMigrateBox() {
  const box = els.imageMigrateBox;
  if (!box) return;
  const model = imageModelById(els.imageModelSelect.value);
  const show = isAdmin() && !!model;
  box.hidden = !show;
  box.replaceChildren();
  if (!show) return;
  renderMigrateInto(box, imageMigrateState, {
    fixedTarget: model.id,
    notify: setImageStatus,
    onDone: () => {
      renderImageAdmin();
      renderImageEffective();
    },
  });
}

/** 执行迁移：勾选了更正时先做管理员密钥验证。 */
async function runMigration(box, st, { fixedTarget = '', notify = showStatus, onDone = () => {} } = {}) {
  if (!requireAdmin('跨模型迁移')) return;
  const source = st.source;
  const target = fixedTarget || st.target;
  if (!source || !target || source === target) {
    notify('请选择不同的源 AI 与目标 AI。', { error: true });
    return;
  }
  const checked = [...box.querySelectorAll('input[data-kind]:checked')];
  const pick = (kind) => checked.filter((i) => i.dataset.kind === kind).map((i) => i.dataset.id);
  const wanted = { correction: pick('correction'), knowledge: pick('knowledge'), note: pick('note'), hint: pick('hint') };
  const total = Object.values(wanted).reduce((n, ids) => n + ids.length, 0);
  if (!total) {
    notify('请先勾选要迁移的条目。', { error: true });
    return;
  }
  const items = migratableItems(source);
  const chosen = (kind) => items[kind === 'correction' ? 'corrections' : kind === 'knowledge' ? 'knowledge' : kind === 'note' ? 'notes' : 'hints'].filter((x) => wanted[kind].includes(x.id));
  const pickedCorr = chosen('correction');
  const from = adminDisplayName(source);
  const to = adminDisplayName(target);
  if (pickedCorr.length) {
    const texts = pickedCorr.map((c) => `- ${c.text}`);
    const ok = await verifyCorrectionKey(
      `迁移 ${pickedCorr.length} 条更正（${from} → ${to}）：\n${texts.join('\n')}`,
      `这些更正将成为「${to}」的底层设定（无条件采纳）。输入管理员密钥验证后执行；取消则不迁移。`,
    );
    if (!ok) return;
  }

  if (isImageAdminId(target)) {
    let result;
    try {
      result = copyIntoImageModel(
        target,
        {
          corrections: pickedCorr.map((c) => ({ text: c.text, from: source })),
          knowledge: [...chosen('knowledge'), ...chosen('note')].map((k) => ({ text: k.text, from: source })),
          hints: chosen('hint').map((h) => ({ text: h.text, from: source })),
        },
        { allowCorrections: pickedCorr.length > 0 },
      );
    } catch (err) {
      notify(`迁移失败：${err.message}`, { error: true });
      return;
    }
    const over = result.overflow ? `，超过上限未写入 ${result.overflow} 条` : '';
    notify(
      `已从「${from}」迁移到「${to}」：更正 ${result.corrections} 条，知识库 ${result.knowledge} 条，风格提示 ${result.hints} 条，重复跳过 ${result.duplicate} 条${over}。`,
    );
    onDone();
    return;
  }

  // 聊天 AI 之间的迁移（原有逻辑）
  const includeTraining = !!box.querySelector('#migrateTraining')?.checked;
  let result;
  try {
    result = migrateAdaptation(
      source,
      target,
      { corrections: wanted.correction, knowledge: wanted.knowledge },
      { includeTraining, allowCorrections: pickedCorr.length > 0 },
    );
  } catch (err) {
    notify(`迁移失败：${err.message}`, { error: true });
    return;
  }
  const missing = result.missing ? `，源中已不存在 ${result.missing} 条` : '';
  notify(
    `已从「${from}」迁移到「${to}」：更正 ${result.corrections} 条，知识库 ${result.knowledge} 条，重复跳过 ${result.duplicate} 条${missing}。`,
  );
  onDone();
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

const PARAM_SOURCE_LABEL = { explicit: '参数行', suggestion: '关键词识别', reset: '恢复默认' };

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
    ['temperature', 'repetitionPenalty', 'maxTokens', 'contextWindow', 'slidingWindow', 'thinkingWindow']
      .map((k) => `${k} = ${eff[k]}（${paramSourceOf(k, modelId)}）`)
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
      const why = c.source === 'suggestion' ? `（${c.reason.replace(/^关键词识别：/, '')}）` : '';
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
els.undoBtn.addEventListener('click', undoTurn);

els.newChatBtn.addEventListener('click', newChat);
els.embedNewBtn.addEventListener('click', newChat);
els.clearAllBtn.addEventListener('click', clearAll);
els.historyBtn.addEventListener('click', () => {
  toggleRolePanel(false);
  toggleHistory();
});

els.thinkToggle.addEventListener('change', async () => {
  state.backend.setThinking(els.thinkToggle.checked);
  const eff = effectiveParams(state.backend.modelId);
  const win = activeContextWindow(eff);
  // 引擎窗口随思考开关改变：释放多余显存（与参数、滑动窗口开关相同）；下次发送时按新窗口加载
  const released = await releaseStaleEngine();
  const tail = released ? '已释放多余显存，' : '';
  flashStatus(
    els.thinkToggle.checked
      ? `已开启思考模式：推理模型会先展示思考过程（可折叠），回答更充分但更慢。${tail}下次发送时将按 ${win + eff.thinkingWindow} 窗口重新加载模型（思考内容不计入窗口）。`
      : `已关闭思考模式：推理模型直接给出回答，速度更快。${tail}下次发送时将按 ${win} 窗口重新加载模型。`,
  );
});

els.slidingToggle.addEventListener('change', async () => {
  state.backend.setSliding(els.slidingToggle.checked);
  const eff = effectiveParams(state.backend.modelId);
  const win = activeContextWindow(eff);
  // 窗口改变后释放多余显存（与参数改变相同）；下次发送时按新窗口重新加载
  const released = await releaseStaleEngine();
  const tail = released ? '已释放多余显存，' : '';
  flashStatus(
    els.slidingToggle.checked
      ? `已开启滑动窗口：上下文固定为 ${win}（思考窗口另计），超出的最早历史会滚出窗口，管理员信息始终保留。${tail}下次发送时按新窗口加载模型。`
      : `已关闭滑动窗口：恢复为上下文窗口 ${eff.contextWindow}，并按 ${state.appConfig.webllm.maxMessages} 条、${state.appConfig.webllm.contextChars} 字限制历史。${tail}下次发送时按新窗口加载模型。`,
  );
});

els.presetSelect?.addEventListener('change', () => applyPreset(els.presetSelect.value));

els.modelSelect.addEventListener('change', () => {
  state.backend.setModel(els.modelSelect.value);
  // 切换模型即更改默认模型：保存在本浏览器，下次打开时使用
  saveDefaultModel(els.modelSelect.value);
  refreshThinkToggle();
  refreshSlidingToggle();
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
// ---------- 生图模式（仅本机 WebLLM 模式）----------
// 生图与聊天互斥：进入生图模式时释放聊天模型的显存；离开时终止生图 worker。
// 生图模型只在第一次生成时加载，启动页面时不下载、不加载。
const imageState = { active: false, busy: false, controller: null, client: null, gallery: [], downloadCtl: null, filesStatus: null, filesToken: 0 };

function imageModels() {
  return imageModelsOf(state.appConfig);
}

function imageModelById(id) {
  return imageModels().find((m) => m.id === id) || null;
}

function imageClient() {
  if (!imageState.client) {
    imageState.client = createImageClient({
      // 保持 new Worker(new URL(...)) 的字面写法，构建工具据此打包 worker
      createWorker: () => new Worker(new URL('./image/sd15-worker.js', import.meta.url), { type: 'module' }),
      scriptUrl: new URL('./image/sd15-worker.js', import.meta.url).href,
    });
  }
  return imageState.client;
}

function setImageStatus(text, { error = false } = {}) {
  els.imageStatus.textContent = text;
  els.imageStatus.classList.toggle('error', error);
}

function setImageProgress(fraction) {
  els.imageProgress.hidden = fraction === null;
  if (fraction !== null) els.imageProgressFill.style.width = `${Math.round(fraction * 100)}%`;
}

function setImageBusy(busy) {
  imageState.busy = busy;
  els.imageGenerateBtn.disabled = busy;
  els.imageCancelBtn.hidden = !busy;
  els.imageModelSelect.disabled = busy;
  if (!busy) setImageProgress(null);
}

function renderImageModelOptions() {
  const models = imageModels();
  els.imageModelSelect.replaceChildren(
    ...models.map((m) => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = isSelectable(m) ? m.label : `${m.label}（暂不可用）`;
      opt.disabled = !isSelectable(m);
      opt.title = m.note || '';
      return opt;
    }),
  );
  const first = models.find(isSelectable);
  if (first) els.imageModelSelect.value = first.id;
  fillImageControls(imageModelById(els.imageModelSelect.value));
}

/** 切换模型时，用该模型的默认值填充控件（只在切换模型时执行）。 */
function fillImageControls(model) {
  if (!model) return;
  const d = { ...DEFAULT_SIZE, steps: 20, guidance: 7.5, ...(model.defaults || {}) };
  els.imageWidth.value = d.width;
  els.imageHeight.value = d.height;
  els.imageSteps.value = d.steps;
  els.imageGuidance.value = d.guidance;
  els.imageSeed.value = '';
  syncRatioSelect();
  renderImageEffective();
  renderImageFiles();
}

function readImageControls() {
  const seedRaw = els.imageSeed.value.trim();
  return {
    width: Number(els.imageWidth.value),
    height: Number(els.imageHeight.value),
    steps: Number(els.imageSteps.value),
    guidance: Number(els.imageGuidance.value),
    seed: seedRaw === '' ? null : Number(seedRaw),
  };
}

/** 根据当前宽高选中匹配的比例；不匹配时为“自定义”。 */
function syncRatioSelect() {
  const w = Number(els.imageWidth.value);
  const h = Number(els.imageHeight.value);
  const match = RATIOS.find((r) => r.w * h === r.h * w);
  els.imageRatio.value = match ? match.id : 'custom';
}

/** 生效值：界面控件 < 管理员参数。 */
function currentImageSettings(model) {
  return effectiveImageSettings(readImageControls(), getImageSettings(model.id).params);
}

function renderImageEffective() {
  const model = imageModelById(els.imageModelSelect.value);
  if (!model) {
    els.imageEffective.textContent = '';
    return;
  }
  const { params, hints, corrections, knowledge } = getImageSettings(model.id);
  const eff = effectiveImageSettings(readImageControls(), params);
  const adminKeys = Object.keys(params);
  const parts = [
    `生效：${eff.width} × ${eff.height}，${eff.steps} 步，引导 ${eff.guidance}，种子 ${eff.seed ?? '随机'}`,
  ];
  if (adminKeys.length) parts.push(`管理员参数已覆盖：${adminKeys.map((k) => `${k}=${params[k]}`).join('，')}`);
  parts.push(`描述输入上限 ${effectiveInputLimit(params)} tokens（超过则拒绝生成；CLIP 每段读取 75 个，描述最多分 ${MAX_PROMPT_CHUNKS} 段，约 ${MAX_PROMPT_CHUNKS * 75} 个 token 参与生图）`);
  if (corrections.length) parts.push(`更正 ${corrections.length} 条（提示词最前面）`);
  if (knowledge.length) parts.push(`知识库 ${knowledge.length} 条（提示词最前面）`);
  if (hints.length) parts.push(`风格提示 ${hints.length} 条`);
  if (isSelectable(model)) parts.push(`模型约 ${formatGB(totalBytes(model))}（可点“下载到本机”保存，之后离线可用）`);
  els.imageEffective.textContent = parts.join('｜');
}

/** 管理员列表的一行：文字 + 删除按钮（删除需管理员权限）。 */
function imageNoteRow(text, onDelete, action) {
  const row = document.createElement('div');
  row.className = 'local-row';
  const label = document.createElement('div');
  label.className = 'muted small local-meta';
  label.textContent = text;
  const del = document.createElement('button');
  del.type = 'button';
  del.className = 'btn ghost small';
  del.textContent = '删除';
  del.addEventListener('click', () => {
    if (!requireAdmin(action)) return;
    onDelete();
    renderImageAdmin();
    renderImageEffective();
  });
  row.append(label, del);
  return row;
}

function renderImageAdmin() {
  const model = imageModelById(els.imageModelSelect.value);
  els.imageAdmin.hidden = !isAdmin() || !model;
  if (!model) return;
  const { hints, corrections, knowledge } = getImageSettings(model.id);
  els.imageHintList.replaceChildren(
    ...hints.map((h) => imageNoteRow(h.text, () => removeImageHint(model.id, h.id), '删除风格提示')),
  );
  els.imageCorrectionList.replaceChildren(
    ...corrections.map((c) => imageNoteRow(`更正：${c.text}`, () => removeImageCorrection(model.id, c.id), '删除更正')),
  );
  els.imageKnowledgeList.replaceChildren(
    ...knowledge.map((k) => imageNoteRow(`知识：${k.text}`, () => removeImageKnowledge(model.id, k.id), '删除知识库条目')),
  );
  renderImageMigrateBox();
}

/**
 * 为生图 AI 记录更正。更正会成为该 AI 的底层设定（放在提示词最前面），因此与聊天一样需要管理员密钥验证。
 * 返回新记录的条数；取消或密钥错误返回 null。
 */
async function recordImageCorrections(model, texts) {
  const ok = await verifyCorrectionKey(
    texts.join('\n'),
    `这些更正将作为「${model.label}」（生图 AI）的底层设定，放在每次生图提示词的最前面（仅本机）。输入管理员密钥验证后保存；取消则不保存。`,
  );
  if (!ok) return null;
  let added = 0;
  for (const t of texts) if (addImageCorrection(model.id, t)) added += 1;
  return added;
}

function imageErrorText(err) {
  const m = err?.message || String(err);
  if (/out of memory|Failed to allocate|OOM|device lost|GPUDevice|buffer.*size/i.test(m)) {
    return `显存不足或 GPU 资源丢失（${m}）。可以减小宽高、减少步数，或关闭占用显存的其他页面后重试。`;
  }
  // 运行库（网站包内的 ort/ 目录）加载失败：不是网络问题，而是文件缺失或服务器未提供该文件
  if (/dynamically imported module|ort-wasm|initWasm/i.test(m)) {
    return `生图运行库加载失败：${m}。请确认网站包完整（web/ort/ 目录存在）并刷新页面后重试。`;
  }
  if (/会话|session/i.test(m)) {
    return `生图模型无法在本机运行：${m}`;
  }
  if (/Failed to fetch|NetworkError|下载失败|network/i.test(m)) {
    return `模型文件下载失败：${m}。请检查网络是否能访问 huggingface.co，然后重试。`;
  }
  return `生图失败：${m}`;
}

async function rgbaToPngBlob(rgba, width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('无法生成图片文件'))), 'image/png');
  });
}

// ---------- 画廊（保存在本机 IndexedDB，刷新页面后保留）----------
function galleryItemFrom(record) {
  return { ...record, url: URL.createObjectURL(record.blob), unsaved: !!record.unsaved };
}

function revokeGalleryUrls(items) {
  for (const it of items) URL.revokeObjectURL(it.url);
}

async function loadGallery() {
  const rows = await listGallery();
  revokeGalleryUrls(imageState.gallery);
  imageState.gallery = rows.map(galleryItemFrom);
  renderImageGallery();
}

function renderImageGallery() {
  const n = imageState.gallery.length;
  els.imageGalleryCount.textContent = n ? `共 ${n} 张（保存在本机）` : '画廊为空';
  els.imageExportGalleryBtn.disabled = n === 0;
  els.imageClearGalleryBtn.hidden = n === 0;
  els.imageGallery.replaceChildren(...imageState.gallery.map(galleryCard));
}

function galleryCard(item) {
  const card = document.createElement('figure');
  card.className = 'image-card';
  const img = document.createElement('img');
  img.src = item.url;
  img.alt = item.prompt;
  const cap = document.createElement('figcaption');
  cap.className = 'muted small';
  cap.style.whiteSpace = 'pre-wrap';
  cap.textContent =
    `${item.width} × ${item.height}｜${item.steps} 步｜引导 ${item.guidance}｜种子 ${item.seed}` +
    (item.unsaved ? '｜未保存到画廊' : '') +
    `\n${item.prompt}`;
  const actions = document.createElement('div');
  actions.className = 'image-admin-row';
  const dl = document.createElement('a');
  dl.className = 'btn ghost small';
  dl.href = item.url;
  dl.download = item.filename;
  dl.textContent = '下载 PNG';
  const rm = document.createElement('button');
  rm.type = 'button';
  rm.className = 'btn ghost small';
  rm.textContent = '删除';
  rm.addEventListener('click', () => deleteGalleryItem(item));
  actions.append(dl, rm);
  card.append(img, cap, actions);
  return card;
}

async function deleteGalleryItem(item) {
  if (!window.confirm('删除这张图片？删除后无法恢复（可先导出画廊备份）。')) return;
  try {
    await deleteGalleryImage(item.id);
  } catch (err) {
    setImageStatus(`删除失败：${err.message}`, { error: true });
    return;
  }
  URL.revokeObjectURL(item.url);
  imageState.gallery = imageState.gallery.filter((x) => x !== item);
  renderImageGallery();
  setImageStatus('已删除这张图片。');
}

async function clearGalleryAll() {
  const n = imageState.gallery.length;
  if (!n) return;
  if (!window.confirm(`清空画廊中的 ${n} 张图片？此操作无法恢复（可先导出画廊备份）。`)) return;
  try {
    await clearGallery();
  } catch (err) {
    setImageStatus(`清空失败：${err.message}`, { error: true });
    return;
  }
  revokeGalleryUrls(imageState.gallery);
  imageState.gallery = [];
  renderImageGallery();
  setImageStatus('已清空画廊。');
}

/** 保存一张生成的图片。无法写入 IndexedDB 时仍显示在画廊中，并标记为未保存。 */
async function addImageResult(record) {
  const item = galleryItemFrom(record);
  try {
    await addGalleryImage(record);
  } catch (err) {
    item.unsaved = true;
    setImageStatus(`图片已生成，但无法保存到本机画廊（${err.message}）。请先点击“下载 PNG”保存。`, { error: true });
  }
  imageState.gallery = [item, ...imageState.gallery];
  renderImageGallery();
  return item;
}

/** 画廊导出：全部图片 + manifest.json（提示词、尺寸、步数、引导、种子、模型、时间）。 */
async function exportGallery() {
  if (!imageState.gallery.length) return;
  setImageStatus('正在打包画廊…');
  try {
    const items = [];
    for (const it of imageState.gallery) {
      items.push({ ...it, data: new Uint8Array(await it.blob.arrayBuffer()) });
    }
    const bytes = buildGalleryArchive(items);
    const stamp = new Date().toISOString().replace(/[-:]/g, '').slice(0, 15);
    downloadBlob(new Blob([bytes], { type: 'application/zip' }), `oj-ai-gallery-${stamp}.zip`);
    setImageStatus(`已导出 ${items.length} 张图片（zip，含 manifest.json）。`);
  } catch (err) {
    setImageStatus(`导出失败：${err.message}`, { error: true });
  }
}

const galleryKey = (it) => [it.model, it.seed, it.prompt, it.createdAt, it.width, it.height].join('|');

/** 画廊导入：跳过与已有图片完全相同（模型、种子、提示词、时间、尺寸）的条目。 */
async function importGalleryFile(file) {
  setImageStatus('正在导入画廊…');
  try {
    const { items, skipped } = await readGalleryArchive(new Uint8Array(await file.arrayBuffer()));
    const seen = new Set(imageState.gallery.map(galleryKey));
    let added = 0;
    let dup = 0;
    for (const it of items) {
      const key = galleryKey(it);
      if (seen.has(key)) {
        dup += 1;
        continue;
      }
      seen.add(key);
      await addGalleryImage({
        id: newId(),
        blob: new Blob([it.data], { type: 'image/png' }),
        filename: it.filename,
        prompt: it.prompt,
        width: it.width,
        height: it.height,
        steps: it.steps,
        guidance: it.guidance,
        seed: it.seed,
        model: it.model,
        createdAt: it.createdAt,
      });
      added += 1;
    }
    await loadGallery();
    const parts = [`已导入 ${added} 张`];
    if (dup) parts.push(`跳过 ${dup} 张重复`);
    if (skipped.length) parts.push(`${skipped.length} 个条目无效（${skipped[0].reason}）`);
    setImageStatus(`${parts.join('，')}。`, { error: added === 0 && skipped.length > 0 });
  } catch (err) {
    setImageStatus(`导入失败：${err.message}`, { error: true });
  }
}

// ---------- 模型文件（下载到本机 / 导出 / 导入 / 删除）----------
function selectedImageModel() {
  return imageModelById(els.imageModelSelect.value);
}

async function renderImageFiles() {
  const model = selectedImageModel();
  const show = !!model && isSelectable(model);
  els.imageModelFiles.hidden = !show;
  if (!show) return;
  const token = (imageState.filesToken += 1);
  let st;
  try {
    st = await imageCacheStatus(model);
  } catch (err) {
    if (token === imageState.filesToken) els.imageFilesStatus.textContent = `无法读取本机缓存：${err.message}`;
    return;
  }
  if (token !== imageState.filesToken) return;
  imageState.filesStatus = { ...st, modelId: model.id };
  const size = formatGB(totalBytes(model));
  const partial = st.cached > 0 || st.partialBytes > 0;
  if (st.complete) els.imageFilesStatus.textContent = `本机已下载（${size}），离线可以生成。`;
  else if (partial) els.imageFilesStatus.textContent = `本机部分下载（${formatGB(st.doneBytes)} / ${size}），点击“继续下载”将从断点继续，不会重下已保存的部分。`;
  else els.imageFilesStatus.textContent = `本机未下载（约 ${size}）。未下载时，生成会从 Hugging Face 读取。`;
  if (imageState.downloadCtl) els.imageDownloadBtn.textContent = '取消下载';
  else if (st.complete) els.imageDownloadBtn.textContent = '已下载到本机';
  else els.imageDownloadBtn.textContent = partial ? '继续下载' : '下载到本机';
  els.imageDownloadBtn.disabled = !imageState.downloadCtl && st.complete;
  els.imageExportModelBtn.disabled = !st.complete;
  els.imageDeleteFilesBtn.disabled = (st.cached === 0 && st.partsCached === 0) || !!imageState.downloadCtl;
}

async function downloadSelectedModel() {
  if (imageState.downloadCtl) {
    imageState.downloadCtl.abort();
    return;
  }
  const model = selectedImageModel();
  if (!model || !isSelectable(model)) return;
  const ctl = new AbortController();
  imageState.downloadCtl = ctl;
  els.imageDownloadProgress.hidden = false;
  els.imageDownloadFill.style.width = '0%';
  renderImageFiles();
  setImageStatus(`正在下载「${model.label}」（约 ${formatGB(totalBytes(model))}）…`);
  // 速度按本次下载开始后新增的字节计算（不计上次已保存的部分）
  const t0 = Date.now();
  let startBytes = null;
  try {
    await downloadImageModel(model, {
      signal: ctl.signal,
      onProgress: ({ done, total, bytes, totalBytes, retry, note }) => {
        if (startBytes === null) startBytes = bytes;
        els.imageDownloadFill.style.width = `${totalBytes ? Math.round((bytes / totalBytes) * 100) : 0}%`;
        if (retry) {
          // 暂时性错误：下载会自动重试，这里只提示，不作为错误显示
          setImageStatus(`网络不稳定，${Math.round(retry.waitMs / 1000)} 秒后自动重试（第 ${retry.attempt} 次）。已下载的部分会保留，无需操作。`);
          return;
        }
        if (note) {
          // 拼接、跳过已保存部分等阶段：进度条保持在已保存的字节数，并说明正在做什么
          setImageStatus(note);
          return;
        }
        const secs = (Date.now() - t0) / 1000;
        const mbps = secs > 0.5 ? (bytes - startBytes) / secs / 1e6 : 0;
        const speed = mbps > 0 ? `，约 ${mbps.toFixed(1)} MB/s` : '';
        setImageStatus(`正在下载：${formatGB(bytes)} / ${formatGB(totalBytes)}（第 ${Math.min(done + 1, total)}/${total} 个文件${speed}）…`);
      },
    });
    setImageStatus('模型已下载到本机，之后可离线生成。');
  } catch (err) {
    if (err.name === 'AbortError') setImageStatus('已取消下载。已保存的部分会保留，下次点击“继续下载”从断点继续。');
    else setImageStatus(`下载未完成：${err.message}`, { error: true });
  } finally {
    imageState.downloadCtl = null;
    els.imageDownloadProgress.hidden = true;
    renderImageFiles();
  }
}

async function exportSelectedModel() {
  const model = selectedImageModel();
  if (!model) return;
  els.imageExportModelBtn.disabled = true;
  setImageStatus('正在打包模型文件（文件较大，请耐心等待）…');
  try {
    const { blob, filename, size } = await buildImageModelFile(model);
    downloadBlob(blob, filename);
    setImageStatus(`已导出 ${filename}（${formatGB(size)}）。`);
  } catch (err) {
    setImageStatus(`导出失败：${err.message}`, { error: true });
  } finally {
    renderImageFiles();
  }
}

async function importModelFileFromPicker(file) {
  setImageStatus(`正在导入 ${file.name}（文件较大，请耐心等待）…`);
  try {
    const { model, entryCount } = await importImageModelFile(file, { models: imageModels() });
    if (els.imageModelSelect.value !== model.id) {
      els.imageModelSelect.value = model.id;
      fillImageControls(model);
    }
    setImageStatus(`已导入「${model.label}」（${entryCount} 个文件），已写入本机。`);
  } catch (err) {
    setImageStatus(`导入失败：${err.message}`, { error: true });
  } finally {
    els.imageImportModelInput.value = '';
    renderImageFiles();
  }
}

async function deleteSelectedModelFiles() {
  const model = selectedImageModel();
  if (!model) return;
  if (!window.confirm(`删除「${model.label}」在本机的文件（约 ${formatGB(totalBytes(model))}）？删除后生成时需要重新下载。`)) return;
  try {
    const n = await deleteImageModelFiles(model);
    setImageStatus(`已删除本机的 ${n} 个模型文件。`);
  } catch (err) {
    setImageStatus(`删除失败：${err.message}`, { error: true });
  } finally {
    renderImageFiles();
  }
}

async function enterImageMode() {
  if (imageState.active) return;
  if (state.streaming) {
    flashStatus('请先等待当前回答结束或点击“停止”，再进入生图模式。');
    return;
  }
  els.imageModeBtn.disabled = true;
  flashStatus('正在释放聊天模型的显存…');
  try {
    await releaseEngine();
  } finally {
    els.imageModeBtn.disabled = false;
  }
  imageState.active = true;
  els.body.classList.add('image-mode');
  els.imagePanel.hidden = false;
  els.imageModeBtn.textContent = '返回聊天';
  els.imageModeBtn.setAttribute('aria-pressed', 'true');
  renderImageModelOptions();
  renderImageAdmin();
  renderImageEffective();
  renderImageFiles();
  let galleryError = null;
  try {
    await loadGallery();
  } catch (err) {
    galleryError = err;
  }
  setImageStatus('已进入生图模式。聊天模型已释放；生图模型在第一次生成时加载。');
  if (galleryError) setImageStatus(`无法读取本机画廊：${galleryError.message}`, { error: true });
}

function exitImageMode() {
  if (!imageState.active) return;
  // 结束正在进行的生成，并终止生图 worker（释放显存）
  imageState.controller?.abort();
  imageState.client?.dispose();
  imageState.active = false;
  els.body.classList.remove('image-mode');
  els.imagePanel.hidden = true;
  els.imageModeBtn.textContent = '生图';
  els.imageModeBtn.setAttribute('aria-pressed', 'false');
  refreshThinkToggle();
  flashStatus('已返回聊天。生图模型已释放；发送消息时将重新加载聊天模型。');
}

async function runImageGenerate(event) {
  event.preventDefault();
  if (imageState.busy) return;
  const model = imageModelById(els.imageModelSelect.value);
  if (!model || !isSelectable(model)) {
    setImageStatus('请选择可用的模型。', { error: true });
    return;
  }
  // 下载进行中时不生成：生成会补下缺少的字节，与下载同时写同一份分段缓存会互相干扰
  if (imageState.downloadCtl) {
    setImageStatus('正在下载模型，请等待下载完成（或先取消下载）后再生成。', { error: true });
    return;
  }
  let text = els.imagePrompt.value;
  let recorded = null;
  // 管理员参数行：只记录，不作为提示词
  if (isAdmin()) {
    const params = parseImageParams(text);
    if (params) {
      recorded = setImageParams(model.id, params);
      text = stripImageParamLine(text);
      els.imagePrompt.value = text;
    }
  }
  // 管理员的“更正：”行：记录为该生图 AI 的更正（需密钥验证），并从提示词中去掉
  let correctionNote = '';
  if (isAdmin()) {
    const corr = parseCorrections(text);
    if (corr.length) {
      const added = await recordImageCorrections(model, corr.map((c) => c.text));
      if (added === null) {
        setImageStatus('未通过管理员密钥验证，更正未保存。提示词保持不变。', { error: true });
        return;
      }
      correctionNote = added ? `已记录 ${added} 条更正（仅本机，放在提示词最前面）` : '更正已存在，未重复保存';
      text = stripCorrectionLines(text);
      els.imagePrompt.value = text;
    }
  }
  if (!text.trim()) {
    if (recorded || correctionNote) {
      const parts = [];
      if (recorded) parts.push('已记录管理员生图参数');
      if (correctionNote) parts.push(correctionNote);
      setImageStatus(`${parts.join('，')}。`);
      renderImageAdmin();
      renderImageEffective();
    } else {
      setImageStatus('请先输入画面描述。', { error: true });
    }
    return;
  }
  if (!(await checkWebGPU())) {
    setImageStatus(WEBGPU_MISSING, { error: true });
    return;
  }
  const { hints, corrections, knowledge } = getImageSettings(model.id);
  const eff = currentImageSettings(model);
  if (eff.seed === null) eff.seed = randomSeed();
  // 管理员前置内容（更正、知识库）放在最前面；lead 单独传给 worker 做预算检查（超出则拒绝，不截断）
  const leadTexts = imageLeadTexts({ corrections, knowledge });
  const prompt = buildImagePrompt(text, hints, leadTexts);
  const controller = new AbortController();
  imageState.controller = controller;
  setImageBusy(true);
  const cached = !!imageState.filesStatus?.complete && imageState.filesStatus.modelId === model.id;
  setImageStatus(
    cached
      ? '正在从本机加载生图模型…'
      : '正在准备生图模型（未下载到本机，将从 Hugging Face 读取约 ' + formatGB(totalBytes(model)) + '）…',
  );
  try {
    setImageProgress(0);
    let top = 0; // 本次生成已显示的最高进度
    const res = await imageClient().generate(
      {
        prompt,
        // 描述（文本框内容）与输入上限：超出上限时在 worker 中拒绝生成
        input: text,
        maxInputTokens: effectiveInputLimit(getImageSettings(model.id).params),
        lead: joinLeadTexts(leadTexts),
        width: eff.width,
        height: eff.height,
        steps: eff.steps,
        guidance: eff.guidance,
        seed: eff.seed,
      },
      {
        urls: sourceUrls(model),
        sizes: sourceSizes(model),
        // 潜变量缩放由模型配置给出（缺省为 SD 1.x 的 0.18215）
        latentScale: model.latentScale,
        // 运行库目录以页面地址为准：开发服务器为 /ort/，网站包为 index.html 同级的 ort/
        wasmBase: new URL('ort/', document.baseURI).href,
        signal: controller.signal,
        onProgress: (p) => {
          // 进度条只前进不后退。读取与会话创建交错进行，worker 给出的 fraction 已是整体进度
          // （读取占 90%、会话占 10%，见 load-progress.js）
          const raise = (v) => {
            if (!Number.isFinite(v)) return;
            top = Math.max(top, v);
            setImageProgress(top);
          };
          if (p.stage === 'denoise' && p.total) {
            setImageStatus(`去噪中：第 ${p.step} / ${p.total} 步`);
            raise(p.step / p.total);
          } else if (p.stage === 'load' && p.total) {
            raise(p.fraction);
            setImageStatus(p.message);
          } else if (p.stage === 'session') {
            raise(p.fraction);
            setImageStatus(p.message);
          } else if (p.message) {
            setImageStatus(p.message);
          }
        },
      },
    );
    const blob = await rgbaToPngBlob(res.rgba, res.width, res.height);
    const item = await addImageResult({
      id: newId(),
      blob,
      filename: `${model.family || 'image'}-${eff.seed}-${res.width}x${res.height}.png`,
      prompt,
      width: res.width,
      height: res.height,
      steps: eff.steps,
      guidance: eff.guidance,
      seed: eff.seed,
      model: model.id,
      createdAt: Date.now(),
    });
    // 描述过长时末尾被截断（风格提示最先被截掉）；管理员内容不会被截断（超出时已在 worker 中拒绝）
    const cut = res.truncatedTokens
      ? `；提示词超出 CLIP 可用长度（最多 ${MAX_PROMPT_CHUNKS} 段，约 ${MAX_PROMPT_CHUNKS * 75} 个 token），末尾 ${res.truncatedTokens} 个 token 未参与生图，请缩短描述`
      : '';
    const chunkNote = res.chunkFallback
      ? `；长提示词分段编码失败（${res.chunkError || '模型不支持'}），已改用单段，仅前 75 个 token 参与生图`
      : res.promptChunks > 1
        ? `；提示词分为 ${res.promptChunks} 段编码`
        : '';
    if (!item.unsaved) setImageStatus(`已生成 ${res.width} × ${res.height}，种子 ${eff.seed}，已保存到画廊${chunkNote}${cut}。`);
  } catch (err) {
    if (err.name === 'AbortError') {
      setImageStatus('已取消。下次生成时会重新加载生图模型。');
    } else {
      setImageStatus(imageErrorText(err), { error: true });
    }
  } finally {
    imageState.controller = null;
    setImageBusy(false);
    renderImageAdmin();
  }
}

function initImageMode() {
  const usable = imageModels().some(isSelectable);
  els.imageModeBtn.hidden = !(MODE === 'webllm' && usable);
  if (!usable) return;
  els.imageModeBtn.addEventListener('click', () => (imageState.active ? exitImageMode() : enterImageMode()));
  els.imageForm.addEventListener('submit', runImageGenerate);
  els.imagePrompt.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      els.imageForm.requestSubmit();
    }
  });
  els.imageCancelBtn.addEventListener('click', () => imageState.controller?.abort());
  els.imageModelSelect.addEventListener('change', () => {
    fillImageControls(imageModelById(els.imageModelSelect.value));
    renderImageAdmin();
    renderImageFiles();
  });
  els.imageDownloadBtn.addEventListener('click', () => downloadSelectedModel());
  els.imageExportModelBtn.addEventListener('click', () => exportSelectedModel());
  els.imageImportModelInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) importModelFileFromPicker(file);
  });
  els.imageDeleteFilesBtn.addEventListener('click', () => deleteSelectedModelFiles());
  els.imageExportGalleryBtn.addEventListener('click', () => exportGallery());
  els.imageImportGalleryInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) importGalleryFile(file).finally(() => (e.target.value = ''));
  });
  els.imageClearGalleryBtn.addEventListener('click', () => clearGalleryAll());
  els.imageRatio.addEventListener('change', () => {
    if (els.imageRatio.value === 'custom') return;
    const long = Math.max(Number(els.imageWidth.value) || 0, Number(els.imageHeight.value) || 0);
    const size = sizeForRatio(els.imageRatio.value, long);
    els.imageWidth.value = size.width;
    els.imageHeight.value = size.height;
    renderImageEffective();
  });
  for (const el of [els.imageWidth, els.imageHeight]) {
    el.addEventListener('change', () => {
      syncRatioSelect();
      renderImageEffective();
    });
  }
  for (const el of [els.imageSteps, els.imageGuidance, els.imageSeed]) {
    el.addEventListener('input', renderImageEffective);
  }
  const addHint = () => {
    const model = imageModelById(els.imageModelSelect.value);
    if (!model || !requireAdmin('添加风格提示')) return;
    const hint = addImageHint(model.id, els.imageHintInput.value);
    if (!hint) {
      setImageStatus('风格提示为空，或与已有的提示相同。', { error: true });
      return;
    }
    els.imageHintInput.value = '';
    renderImageAdmin();
    renderImageEffective();
    setImageStatus('已添加风格提示（仅本机，之后每次生图都会附加）。');
  };
  els.imageHintAdd.addEventListener('click', addHint);
  els.imageCorrectionAdd.addEventListener('click', async () => {
    const model = imageModelById(els.imageModelSelect.value);
    if (!model || !requireAdmin('添加更正')) return;
    const text = els.imageCorrectionInput.value.trim();
    if (!text) {
      setImageStatus('请先输入更正内容。', { error: true });
      return;
    }
    const added = await recordImageCorrections(model, [text]);
    if (added === null) {
      setImageStatus('未通过管理员密钥验证，更正未保存。', { error: true });
      return;
    }
    if (added === 0) {
      setImageStatus('该更正已存在，未重复保存。', { error: true });
      return;
    }
    els.imageCorrectionInput.value = '';
    renderImageAdmin();
    renderImageEffective();
    setImageStatus('已添加更正（仅本机，放在该生图 AI 提示词的最前面）。');
  });
  els.imageKnowledgeAdd.addEventListener('click', () => {
    const model = imageModelById(els.imageModelSelect.value);
    if (!model || !requireAdmin('添加知识库条目')) return;
    if (!addImageKnowledge(model.id, els.imageKnowledgeInput.value)) {
      setImageStatus('知识库条目为空，或与已有条目相同。', { error: true });
      return;
    }
    els.imageKnowledgeInput.value = '';
    renderImageAdmin();
    renderImageEffective();
    setImageStatus('已添加知识库条目（仅本机，放在该生图 AI 提示词的最前面）。');
  });
  els.imageHintInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addHint();
    }
  });
  els.imageParamsClear.addEventListener('click', () => {
    const model = imageModelById(els.imageModelSelect.value);
    if (!model || !requireAdmin('清除管理员生图参数')) return;
    clearImageParams(model.id);
    renderImageEffective();
    setImageStatus('已清除管理员生图参数（界面控件的值重新生效）。');
  });
}

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
  // 生图按钮与事件需要在配置加载之后绑定
  initImageMode();
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
    renderPresetOptions();
    els.presetSelect.hidden = false;
    els.modelSelect.hidden = false;
    els.modelSelect.title = '✓ 表示模型已下载到本机，可离线使用';
    els.modelLabel.hidden = true;
    setTimeout(refreshModelCacheStatus, 0);
    refreshRoleUI();
  } else {
    els.modelLabel.textContent = state.appConfig.model ? `模型：${state.appConfig.model}` : '';
    els.roleBtn.hidden = true;
    els.presetSelect.hidden = true; // 预设只作用于本机模型
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
