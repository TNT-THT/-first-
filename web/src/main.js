import 'katex/dist/katex.min.css';
import './style.css';
import DOMPurify from 'dompurify';
import { renderMarkdown } from './markdown.js';
import { hasThinking, splitThinking } from './think.js';
import { setThinkingPref, truncateThinking } from './reasoning.js';
import { trimRepetition } from './repetition.js';
import { isAllowedOrigin } from './origin.js';
import { ApiError, accountLogin, accountRegister, checkSession, fetchAccountData, fetchConfig, getToken, putAccountData, setToken } from './api.js';
import { createBackend } from './engines/backend.js';
import {
  getSavedDefaultModel,
  getSavedImageModel,
  resolveStartupImageModel,
  resolveStartupModel,
  saveDefaultModel,
  saveImageModel,
} from './default-model.js';
import { addCheckRule, checkRuleError, findViolations, listCheckRules } from './answer-check.js';
import { checkWebGPU, getModelRecords, releaseEngine, WEBGPU_MISSING } from './engines/webllm.js';
import { formatGB, imageModelsOf, isSelectable, sourceSizes, sourceUrls, totalBytes } from './image/catalog.js';
import { createImageClient } from './image/client.js';
import { STANDALONE_ASSETS, TRAINING_PACKAGE_ASSET, isReleaseAsset, releaseAssetUrl, releasePageUrl } from './release-downloads.js';
import { estimatePromptTokens, estimateTokens } from './budget.js';
import {
  USAGE_STORAGE_KEY,
  clearUsageStats,
  formatTokens,
  loadUsageStats,
  recordUsage,
  saveUsageStats,
  summarizeUsage,
} from './usage-stats.js';
import {
  addImageHint,
  buildImagePrompt,
  effectiveInputLimit,
  clearImageParams,
  effectiveImageSettings,
  getImageSettings,
  imageLeadTexts,
  parseImageParams,
  removeImageHint,
  sanitizeImageParams,
  setImageParams,
  stripImageParamLine,
} from './image/params.js';
import { randomSeed } from './image/rng.js';
import { joinLeadTexts, MAX_PROMPT_CHUNKS } from './image/prompt-budget.js';
import { addGalleryImage, clearGallery, deleteGalleryImage, listGallery } from './image/gallery-store.js';
import { buildGalleryArchive, readGalleryArchive } from './image/gallery-archive.js';
import {
  deleteImageModelFiles,
  downloadImageModel,
  imageCacheStatus,
  imageModelFileName,
  importImageModelFile,
  listCompleteImageModelIds,
  prepareImageModelExport,
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
  imagePickerOptions,
  isPickerCancelled,
  savePickerOptions,
  supportsSaveLocationPicker,
} from './save-location.js';
import { adminExpiry, grantAdmin, hasAccessGate, isAdmin, revokeAdmin, verifyAccessPassword } from './access.js';
import { addLocalModel, loadLocalModels, localRecords, removeLocalModel } from './local-models.js';
import { pendingModels } from './pending-models.js';
import { clearApiAdminSession, loadApiAdminSession, saveApiAdminSession } from './api-admin-auth.js';
import { downloadModelViaBrowser, fetchModelInfoViaBrowser, fetchNativeQuants, fetchNativeStatus, fetchQuantsViaBrowser, hfModelInfo, hfSearch, importLocalModel, nativeAdminRequest, searchHfViaBrowser, useRepoViaBrowser } from './engines/native-client.js';
import { DEFAULT_PRESET_ID, findPreset, getPresetId, layerParams, PRESETS, presetOverrides, setPresetId } from './presets.js';
import { buildDataExport, DATA_SECTIONS, dataExportFilename } from './data-export.js';
import {
  adaptedModelIds,
  applyParamSuggestion,
  clearModelAdaptation,
  getKnowledgeText,
  getModelParams,
  listAdaptation,
  listParamLog,
  MAX_KB_TEXT_CHARS,
  PARAM_LABELS,
  PARAM_LIMITS,
  recordFeedback,
  removeNote,
  resetAllParams,
  resetParam,
  sanitizeParams,
  setKnowledgeText,
  setModelParams,
  suggestParams,
  mechanicalReply,
} from './adaptation.js';
import { buildConversationMarkdown, conversationFilename } from './chat-export.js';
import {
  cleanReaderTemplate,
  loadProviderSettings,
  loadSearchPrefs,
  normalizeWebSearch,
  planAdvanced,
  planKeyless,
  PROVIDER_LABELS,
  resolveSearchPrefs,
  saveProviderSettings,
  saveSearchPrefs,
  SEARCH_MODE_LABELS,
  SEARCH_MODES,
  searchAvailable,
} from './search/config.js';
import { buildSearchQuery, searchInputForTurn } from './search/query.js';
import {
  auditCitations,
  compactAttempts,
  compactSources,
  describeAttempts,
  hostOf,
  prepareSearchEvidence,
  sanitizeUrl,
  searchSummary,
  withSearchContext,
} from './search/context.js';
import { createSearchCache, runWebSearch } from './search/index.js';
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
  accountBtn: $('#accountBtn'),
  accountDialog: $('#accountDialog'),
  accountAnon: $('#accountAnon'),
  accountLogged: $('#accountLogged'),
  accountName: $('#accountName'),
  accountPassword: $('#accountPassword'),
  accountError: $('#accountError'),
  accountLoginBtn: $('#accountLoginBtn'),
  accountRegisterBtn: $('#accountRegisterBtn'),
  accountClose: $('#accountClose'),
  accountClose2: $('#accountClose2'),
  accountCurrentName: $('#accountCurrentName'),
  accountSyncStatus: $('#accountSyncStatus'),
  accountSyncNowBtn: $('#accountSyncNowBtn'),
  accountLogoutBtn: $('#accountLogoutBtn'),
  kbTextInput: $('#kbTextInput'),
  kbSaveBtn: $('#kbSaveBtn'),
  kbCharCount: $('#kbCharCount'),
  kbStatus: $('#kbStatus'),
  apiAccessBtn: $('#apiAccessBtn'),
  adminPanelBtn: $('#adminPanelBtn'),
  modelDownloadBtn: $('#modelDownloadBtn'),
  modelDownloadDialog: $('#modelDownloadDialog'),
  modelDownloadClose: $('#modelDownloadClose'),
  modelDownloadUnavailable: $('#modelDownloadUnavailable'),
  apiAccessDialog: $('#apiAccessDialog'),
  apiAccessClose: $('#apiAccessClose'),
  apiAdminLoginSection: $('#apiAdminLoginSection'),
  apiPlaygroundSection: $('#apiPlaygroundSection'),
  apiAdminSettingsSection: $('#apiAdminSettingsSection'),
  apiKeyManagerSection: $('#apiKeyManagerSection'),
  apiAccessBase: $('#apiAccessBase'),
  apiAccessEndpoint: $('#apiAccessEndpoint'),
  apiAccessModel: $('#apiAccessModel'),
  apiSelfApplySection: $('#apiSelfApplySection'),
  apiBridgeStatus: $('#apiBridgeStatus'),
  apiSelfApplyBtn: $('#apiSelfApplyBtn'),
  apiSelfKeyPanel: $('#apiSelfKeyPanel'),
  apiSelfKeyValue: $('#apiSelfKeyValue'),
  apiSelfKeyCopy: $('#apiSelfKeyCopy'),
  apiSelfApplyError: $('#apiSelfApplyError'),
  apiCurlExample: $('#apiCurlExample'),
  apiCurlCopy: $('#apiCurlCopy'),
  apiPlaygroundForm: $('#apiPlaygroundForm'),
  apiPlaygroundModel: $('#apiPlaygroundModel'),
  apiPlaygroundSystem: $('#apiPlaygroundSystem'),
  apiPlaygroundPrompt: $('#apiPlaygroundPrompt'),
  apiPlaygroundTemperature: $('#apiPlaygroundTemperature'),
  apiPlaygroundMaxTokens: $('#apiPlaygroundMaxTokens'),
  apiPlaygroundStream: $('#apiPlaygroundStream'),
  apiPlaygroundRequest: $('#apiPlaygroundRequest'),
  apiPlaygroundRequestCopy: $('#apiPlaygroundRequestCopy'),
  apiPlaygroundSend: $('#apiPlaygroundSend'),
  apiPlaygroundStop: $('#apiPlaygroundStop'),
  apiPlaygroundClear: $('#apiPlaygroundClear'),
  apiPlaygroundAnswerCopy: $('#apiPlaygroundAnswerCopy'),
  apiPlaygroundStatus: $('#apiPlaygroundStatus'),
  apiPlaygroundAnswer: $('#apiPlaygroundAnswer'),
  apiPlaygroundRaw: $('#apiPlaygroundRaw'),
  sharePromptInput: $('#sharePromptInput'),
  shareAutoSend: $('#shareAutoSend'),
  shareLinkGenerate: $('#shareLinkGenerate'),
  shareLinkCopy: $('#shareLinkCopy'),
  shareLinkOutput: $('#shareLinkOutput'),
  shareIframeOutput: $('#shareIframeOutput'),
  shareIframeCopy: $('#shareIframeCopy'),
  apiKeyManagerHint: $('#apiKeyManagerHint'),
  apiKeyAdminLoginForm: $('#apiKeyAdminLoginForm'),
  apiAdminSettingsForm: $('#apiAdminSettingsForm'),
  apiAdminBaseUrl: $('#apiAdminBaseUrl'),
  apiAdminModel: $('#apiAdminModel'),
  apiAdminApiKey: $('#apiAdminApiKey'),
  apiAdminClearApiKey: $('#apiAdminClearApiKey'),
  apiAdminApiKeyStatus: $('#apiAdminApiKeyStatus'),
  apiAdminSystemPrompt: $('#apiAdminSystemPrompt'),
  apiAdminTemperature: $('#apiAdminTemperature'),
  apiAdminMaxTokens: $('#apiAdminMaxTokens'),
  apiAdminGenerationParams: $('#apiAdminGenerationParams'),
  apiAdminCorsOrigins: $('#apiAdminCorsOrigins'),
  apiAdminSettingsSave: $('#apiAdminSettingsSave'),
  apiAdminSettingsReset: $('#apiAdminSettingsReset'),
  apiAdminSettingsStatus: $('#apiAdminSettingsStatus'),
  apiAdminSettingsError: $('#apiAdminSettingsError'),
  apiKeyAdminPassword: $('#apiKeyAdminPassword'),
  apiKeyAdminLoginBtn: $('#apiKeyAdminLoginBtn'),
  apiKeyAdminError: $('#apiKeyAdminError'),
  apiKeyManagerPanel: $('#apiKeyManagerPanel'),
  apiKeyCreateForm: $('#apiKeyCreateForm'),
  apiKeyLabelInput: $('#apiKeyLabelInput'),
  apiKeyCreateBtn: $('#apiKeyCreateBtn'),
  apiKeyNewPanel: $('#apiKeyNewPanel'),
  apiKeyNewValue: $('#apiKeyNewValue'),
  apiKeyNewCopy: $('#apiKeyNewCopy'),
  apiKeyManagerError: $('#apiKeyManagerError'),
  apiKeyList: $('#apiKeyList'),
  apiKeyAdminLogout: $('#apiKeyAdminLogout'),
  apiNativeSection: $('#apiNativeSection'),
  apiNativeBinaryStatus: $('#apiNativeBinaryStatus'),
  apiNativeAdminGate: $('#apiNativeAdminGate'),
  apiNativeGotoAdmin: $('#apiNativeGotoAdmin'),
  apiNativeBinaryBtn: $('#apiNativeBinaryBtn'),
  apiNativeCtx: $('#apiNativeCtx'),
  apiNativeGpuLayers: $('#apiNativeGpuLayers'),
  apiNativeUnloadBtn: $('#apiNativeUnloadBtn'),
  apiNativeHint: $('#apiNativeHint'),
  apiNativeLogWrap: $('#apiNativeLogWrap'),
  apiNativeLogTail: $('#apiNativeLogTail'),
  apiNativeList: $('#apiNativeList'),
  apiNativeError: $('#apiNativeError'),
  apiHfSearchInput: $('#apiHfSearchInput'),
  apiHfFilter: $('#apiHfFilter'),
  apiHfSearchBtn: $('#apiHfSearchBtn'),
  importNameInput: $('#importNameInput'),
  importFilesBtn: $('#importFilesBtn'),
  importFolderBtn: $('#importFolderBtn'),
  importFilesInput: $('#importFilesInput'),
  importFolderInput: $('#importFolderInput'),
  importProgressWrap: $('#importProgressWrap'),
  importProgressFill: $('#importProgressFill'),
  importProgressPct: $('#importProgressPct'),
  importProgressLabel: $('#importProgressLabel'),
  importStatus: $('#importStatus'),
  apiHfTokenInput: $('#apiHfTokenInput'),
  apiHfTokenBtn: $('#apiHfTokenBtn'),
  apiHfTokenStatus: $('#apiHfTokenStatus'),
  apiHfConvertToolBtn: $('#apiHfConvertToolBtn'),
  apiHfConvertToolStatus: $('#apiHfConvertToolStatus'),
  apiHfRunnerBtn: $('#apiHfRunnerBtn'),
  apiHfRunnerStatus: $('#apiHfRunnerStatus'),
  apiHfSearchStatus: $('#apiHfSearchStatus'),
  apiHfResults: $('#apiHfResults'),
  main: $('#main'),
  brandName: $('#brandName'),
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
  toolsBtn: $('#toolsBtn'),
  toolsBadge: $('#toolsBadge'),
  toolsPanel: $('#toolsPanel'),
  toolList: $('#toolList'),
  toolCard: $('#toolCard'),
  toolsClose: $('#toolsClose'),
  modelSelectCompact: $('#modelSelectCompact'),
  statusBar: $('#statusBar'),
  statusText: $('#statusText'),
  statusProgress: $('#statusProgress'),
  statusBarFill: $('#statusBarFill'),
  exportModelBtn: $('#exportModelBtn'),
  sitePackageBtns: [...document.querySelectorAll('[data-site-package]')],
  sitePackageHint: $('#sitePackageHint'),
  trainingPackageBtn: $('#trainingPackageBtn'),
  releasePageLink: $('#releasePageLink'),
  usageBtn: $('#usageBtn'),
  usageDialog: $('#usageDialog'),
  usageClose: $('#usageClose'),
  usageWebCards: $('#usageWebCards'),
  usageWebDays: $('#usageWebDays'),
  usageWebModels: $('#usageWebModels'),
  usageWebRecent: $('#usageWebRecent'),
  usageWebClear: $('#usageWebClear'),
  usageApiSection: $('#usageApiSection'),
  usageApiHint: $('#usageApiHint'),
  usageApiBody: $('#usageApiBody'),
  usageApiCards: $('#usageApiCards'),
  usageApiDays: $('#usageApiDays'),
  usageApiModels: $('#usageApiModels'),
  usageApiRecent: $('#usageApiRecent'),
  usageApiClear: $('#usageApiClear'),
  exportDataBtn: $('#exportDataBtn'),
  dataSectionList: $('#dataSectionList'),
  importModelBtn: $('#importModelBtn'),
  importModelInput: $('#importModelInput'),
  importTrainBtn: $('#importTrainBtn'),
  importTrainInput: $('#importTrainInput'),
  roleBtn: $('#roleBtn'),
  localApiInfo: $('#localApiInfo'),
  localApiEndpoint: $('#localApiEndpoint'),
  localApiCopy: $('#localApiCopy'),
  roleBadge: $('#roleBadge'),
  rolePanel: $('#rolePanel'),
  roleStatus: $('#roleStatus'),
  roleForm: $('#roleForm'),
  adminKey: $('#adminKey'),
  roleSubmit: $('#roleSubmit'),
  roleError: $('#roleError'),
  markDialog: $('#markDialog'),
  markForm: $('#markForm'),
  markDesc: $('#markDesc'),
  markWrong: $('#markWrong'),
  markRight: $('#markRight'),
  markKey: $('#markKey'),
  markError: $('#markError'),
  markCancel: $('#markCancel'),
  markSubmit: $('#markSubmit'),
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
  imageFilesStatus: $('#imageFilesStatus'),
  imageDownloadBtn: $('#imageDownloadBtn'),
  imageExportModelBtn: $('#imageExportModelBtn'),
  imageImportModelBtn: $('#imageImportModelBtn'),
  imageImportModelInput: $('#imageImportModelInput'),
  imageExportParamsLabel: $('#imageExportParamsLabel'),
  imageExportWithParams: $('#imageExportWithParams'),
  imageExportParamsText: $('#imageExportParamsText'),
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
};

// ---------- 运行参数 ----------
const params = new URLSearchParams(location.search);
const isEmbed = ['embed', 'hideSidebar'].some((k) => {
  const v = params.get(k);
  return v !== null && v !== '0' && v !== 'false';
});
const isEmbedded = window.parent !== window;
// 链接接入：?q=题目 可预填，显式添加 &send=1 才会自动发送。
// 值只作为普通文本写入输入框，且仅首次显示主界面时处理。
const linkPrompt = (params.get('q') || params.get('prompt') || params.get('text') || params.get('question') || '').slice(0, 20000);
let linkPromptHandled = false;
const linkModel = params.get('model') || '';

const state = {
  store: loadStore(),
  appConfig: { appName: 'OJ 刷题助手', allowedParentOrigins: [], access: {} },
  backend: null,
  apiKeyAdminToken: '', // 网站端 API Key 管理短期令牌，只保存在当前页面内存
  streaming: null, // { controller, session, assistant }
  clearArmed: false,
  foldSessionId: null, // 当前折叠状态对应的会话
  visibleStart: 0, // 历史会话中第一条已渲染消息的下标（之前的消息折叠）
  // 联网资料检索：站点配置 + 本机偏好 + 管理员来源设置；缓存仅保存在当前页面内存。
  search: { config: normalizeWebSearch(null), prefs: { enabled: false, mode: 'quick' }, settings: {}, cache: createSearchCache() },
  toolCardId: null, // 工具面板中当前展开的卡片 id
};

// ---------- 工具函数 ----------
function scrollToBottom(force = false) {
  const m = els.messages;
  const nearBottom = m.scrollHeight - m.scrollTop - m.clientHeight < 120;
  if (force || nearBottom) m.scrollTop = m.scrollHeight;
}

function persist() {
  saveStore(state.store);
  scheduleAccountSync();
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
    // 搜索发生在模型加载前；此时也要立刻展示检索状态，不能等模型产出首个 token。
    paintSearchCard(bubble, msg);
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
  paintSearchCard(bubble, msg);
}

/**
 * 联网资料检索的来源卡片（回答下方，可折叠）。
 * 搜索中显示进度；失败或没有结果时明确写出“本条回答未使用搜索结果”，避免用户误以为搜过了。
 * 标题、摘要、地址一律以纯文本写入，链接只允许 http(s)。
 */
function paintSearchCard(bubble, msg) {
  const info = msg.search;
  let box = bubble.querySelector(':scope > .search-card');
  if (!info) {
    box?.remove();
    return;
  }
  if (!box) {
    box = document.createElement('details');
    box.className = 'search-card';
    const summary = document.createElement('summary');
    const body = document.createElement('div');
    body.className = 'search-body';
    box.append(summary, body);
    const warning = bubble.querySelector(':scope > .check-warning');
    if (warning) bubble.insertBefore(box, warning);
    else bubble.appendChild(box);
  }
  const results = Array.isArray(info.results) ? info.results : [];
  const done = info.status === 'done' && results.length > 0;
  const attempts = describeAttempts(info.attempts, { providerLabels: PROVIDER_LABELS });
  const citations = auditCitations(msg.content, results.length, { sources: results });
  const key = [
    info.status,
    info.mode,
    info.query,
    results.length,
    info.reason || '',
    info.cached ? 'cached' : '',
    info.partial ? 'partial' : '',
    citations.references.join(','),
    citations.invalid.join(','),
    citations.weak?.join(',') || '',
    info.plan?.en || '',
    attempts.map((a) => `${a.id}:${a.status}:${a.count}:${a.message}`).join(','),
  ].join('|');
  if (box.dataset.key === key) return;
  box.dataset.key = key;
  box.classList.toggle('is-error', info.status === 'error');
  const summary = box.querySelector('summary');
  const body = box.querySelector('.search-body');
  body.replaceChildren();

  if (info.status === 'searching') {
    summary.textContent = `🌐 正在检索公开资料「${info.query || '…'}」…`;
    body.append(node('p', { class: 'muted small' }, '只会外发安全提取后的查询词；检测到疑似凭据时会跳过检索。'));
    box.open = true;
    return;
  }
  if (info.status === 'skipped') {
    // 寒暄 / 纯代码 / 空白：不发起请求，也不应显示“检索失败”。
    summary.textContent = `🌐 ${searchSummary(info, { providerLabels: PROVIDER_LABELS })}`;
    body.append(node('p', { class: 'muted small' }, '本条消息没有需要联网核实的内容，因此没有发起任何网络请求。'));
    return;
  }
  if (done) {
    summary.textContent = `🌐 ${searchSummary(info, { providerLabels: PROVIDER_LABELS })}（点击展开）`;
    const list = node('ol', { class: 'search-sources' });
    for (const r of results) {
      const url = sanitizeUrl(r.url);
      if (!url) continue;
      const host = hostOf(url);
      const title = String(r.title || '').trim() || host || url;
      const li = node('li', {});
      li.append(node('a', { href: url, target: '_blank', rel: 'noopener noreferrer', text: title }));
      if (host && host !== title) li.append(node('span', { class: 'search-host' }, host));
      if (r.publishedAt) li.append(node('span', { class: 'search-host' }, `来源时间：${String(r.publishedAt).slice(0, 10)}`));
      if (r.evidenceKind) li.append(node('span', { class: 'search-host' }, `资料类型：${r.evidenceKind}`));
      if (String(r.snippet || '').trim()) li.append(node('span', { class: 'search-snippet' }, r.snippet));
      // 透明展示模型实际收到的正文/答案摘录，而非只展示更短的搜索摘要。
      if (String(r.evidence || '').trim() && r.evidence !== r.snippet) {
        li.append(
          node(
            'details',
            { class: 'search-evidence-box' },
            node('summary', { class: 'muted small' }, `${r.evidenceKind || '模型使用的资料摘录'}（点开查看）`),
            node('p', { class: 'search-snippet' }, r.evidence),
          ),
        );
      }
      list.append(li);
    }
    body.append(list);
    if (attempts.length >= 2) {
      const statusList = node('ul', { class: 'search-attempts' });
      for (const a of attempts) {
        const label =
          a.status === 'ok'
            ? `返回 ${a.count} 条资料`
            : a.status === 'empty'
              ? '没有相关结果'
              : a.status === 'timeout'
                ? `超时${a.message ? `：${a.message}` : ''}`
                : a.status === 'skipped'
                  ? `已跳过${a.message ? `：${a.message}` : ''}`
                  : `请求失败${a.message ? `：${a.message}` : ''}`;
        statusList.append(node('li', { class: `search-attempt-${a.status}` }, `${a.label}：${label}`));
      }
      body.append(
        node(
          'details',
          { class: 'search-attempts-box' },
          node('summary', { class: 'muted small' }, `各来源情况（${attempts.length} 个来源，点开查看）`),
          statusList,
        ),
      );
    }
    const citationNote = citations.references.length
      ? `回答标注了资料 [${citations.references.join(']、[')}]。编号仅说明模型声称引用该资料，仍请打开来源核对。`
      : '回答没有标注有效的 [编号] 引用；下列是检索候选，不能自动视为已被回答证实。';
    body.append(node('p', { class: 'search-note' }, citationNote));
    if (citations.invalid.length) {
      body.append(node('p', { class: 'search-note' }, `回答出现不存在的资料编号：[${citations.invalid.join(']、[')}]。请勿把它当作有效引用。`));
    }
    if (citations.weak?.length) {
      body.append(
        node(
          'p',
          { class: 'search-note' },
          `引用 [${citations.weak.join(']、[')}] 与对应资料摘录几乎没有共同词，关联较弱，请打开来源核实。`,
        ),
      );
    }
    if (info.plan?.en) {
      const source =
        info.plan.enSource === 'glossary'
          ? '术语表翻译（保留问题意图）'
          : info.plan.enSource === 'fragment'
            ? '问题中的英文技术片段'
            : info.plan.enSource === 'langlinks'
              ? '维基条目名（langlinks）'
              : '自动生成';
      body.append(node('p', { class: 'search-note' }, `英文源实际查询：「${info.plan.en}」（来源：${source}）`));
    }
    body.append(
      node(
        'p',
        { class: 'search-note' },
        info.mode === 'smart'
          ? `深入检索：优先补取维基导言、Stack Overflow 回答或 HN 讨论；其他来源只有在配置正文读取服务时才会补正文。${info.partial ? ' 本次到达总超时，已保留已拿到的资料。' : ''}`
          : `快速检索：补取前几条维基导言 / Stack Overflow 答案，其余使用来源摘要；它不是完整网页抓取，也不保证覆盖整个互联网。${info.partial ? ' 本次到达总超时，来源可能不完整。' : ''}`,
      ),
    );
    return;
  }
  summary.textContent =
    info.status === 'stopped'
      ? '🌐 已取消联网资料检索，本条回答未使用外部资料'
      : info.status === 'error'
        ? `🌐 联网资料检索未使用${info.reason ? `（${info.reason}）` : ''}`
        : '🌐 没有找到相关资料，本条回答未使用外部资料';
  const failed = attempts.filter((a) => a.status !== 'ok');
  if (info.status === 'error' && failed.length) {
    const ul = node('ul', { class: 'search-failures' });
    for (const a of failed) {
      const status = a.status === 'empty' ? '没有相关结果' : a.status === 'timeout' ? '超时' : a.status === 'skipped' ? a.message || '已跳过' : a.message || '请求失败';
      ul.append(node('li', {}, `${a.label}：${status}`));
    }
    body.append(ul);
  }
  body.append(
    node(
      'p',
      { class: 'muted small' },
      info.reason?.includes('凭据')
        ? '消息仍会照常交给模型回答，但为保护隐私没有把任何查询发送到外部资料源。'
        : '默认来源是浏览器直连的公开 API（维基百科、Stack Overflow、Hacker News），可能受网络、跨域或速率限制影响。需要更广泛网页覆盖时，请由部署者配置受信任的 SearXNG 或搜索代理。',
    ),
  );
}

/** 校验提示：列出命中的错误说法与对应的正确说法（用 textContent 写入，避免注入）。 */
function buildCheckWarning(hits) {
  const box = document.createElement('div');
  box.className = 'check-warning';
  const title = document.createElement('strong');
  title.textContent = '⚠ 回答中含有管理员标记为错误的说法（以下为正确说法，请以此为准）';
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
 * 保存为校验规则：以后回答再出现错误说法时会提示，并展示正确说法（仅本机）。
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
    '保存后，本机今后回答中再出现错误说法时会给出提示。输入管理员密钥验证后保存；取消则不记录。';
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
      dlg.close();
      renderMessages();
      showStatus(`已保存校验规则：「${wrong.trim()}」出现时请核对，应为「${right.trim()}」（仅本机）。`);
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
      ? '已撤回上一轮，问题已放回输入框。该轮的反馈或参数修改已记录在本机，撤回不会回滚，请到适应面板中删除。'
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
  if (els.modelSelectCompact) els.modelSelectCompact.disabled = busy;
  // 生成期间也不能改思考 / 滑动窗口（会触发引擎重新加载）
  refreshTools();
}

/** 当前回答 AI 的显示名称（不含 ✓ 等缓存标记）。 */
function modelDisplayName(id) {
  if (MODE !== 'webllm') return state.appConfig.model || '服务端模型';
  for (const select of [els.modelSelect, els.modelSelectCompact]) {
    const opt = select && [...select.options].find((o) => o.value === id);
    if (opt) return opt.dataset.label || opt.textContent || id;
  }
  return state.backend.models.find((m) => m.id === id)?.label || id;
}

/**
 * 管理员（且为本机模式）在对话中的修改（参数行、参数关键词、附加说明、标记错误）
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
  const d = state.appConfig.webllm || {};
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
 * extra 为同时记录的附加说明描述。非管理员、非本机模式或没有 AI 时返回空结果。
 * （旧“更正/对话修正”已删除：消息内容只按反馈/参数处理。）
 */
function recordAdminFeedback(session, text) {
  const none = { target: null, label: '', changes: [], extra: [], parts: [] };
  if (MODE !== 'webllm' || !isAdmin() || !state.backend) return none;
  const target = state.backend.modelId;
  if (!target) return none;
  const label = modelDisplayName(target);
  const rec = recordFeedback(target, text);
  const parts = [];
  const changes = [];
  const extra = [];
  if (rec?.note) {
    parts.push('已记录反馈');
    extra.push('记录了附加说明');
  }
  if (rec?.params) {
    for (const c of rec.changes || []) changes.push({ key: c.key, from: c.from, to: c.to });
    parts.push('已更新参数 ' + (changes.length ? changes.map(fmtChange).join(', ') : '（未变化）'));
  }
  // 没有显式参数行时，按关键词直接修改参数（不需要确认，不调用模型）
  if (!rec?.params) {
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
 * 预设或“恢复默认”改变了引擎窗口时，同样释放多余显存（与管理员参数、思考/滑动窗口开关相同）。
 * 生成进行中不释放：释放会打断当前回答，新参数在下次发送时生效。
 */
async function releaseAfterSettingChange() {
  if (state.streaming) return false;
  return releaseStaleEngine();
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

  const session = ensureSession();
  const isFirst = session.messages.length === 0;
  const userMsg = { role: 'user', content: text, ts: Date.now() };
  const feedback = recordAdminFeedback(session, text);
  // 记录过反馈或参数修改的轮次：撤回时需要提示（撤回不回滚这些记录）
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

  // 用量统计：本次回答所用模型（远程模式下 backend.modelId 为空，按网页聊天选择/服务端模型计）。
  // 只记聚合计数（模型、估算 Token、状态），不记对话内容；统计失败绝不影响聊天。
  const usageModelId = modelId || (MODE === 'remote' ? localChatModelId || apiBridgeModels[0]?.id || state.appConfig.model || '' : '');
  let sentMessages = context;
  let usageNoted = false;
  const noteUsageOnce = (status) => {
    if (usageNoted) return;
    usageNoted = true;
    try {
      recordUsage(getWebUsageStats(), {
        model: usageModelId || 'unknown',
        source: 'web',
        promptTokens: estimatePromptTokens(sentMessages),
        completionTokens: estimateTokens(assistantMsg.content),
        status,
      });
      saveUsageStats(getWebUsageStats(), localStorage, USAGE_STORAGE_KEY);
    } catch { /* 统计失败绝不影响聊天 */ }
  };

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

  // 本次提问的联网资料检索资料块（只在本次请求中使用；历史里只保存来源清单）
  let searchBlock = '';
  // 模型加载与网络检索并行：检索最多 12 秒，串行会让首字延迟直接叠加。
  // 加载失败/取消不在这里抛出，等检索结束后统一处理，避免未处理的 rejection。
  let preload = null;
  let searchPhaseDone = false;
  if (shouldSearch() && !state.backend.isReady()) {
    let latest = null;
    const promise = state.backend.ensureReady(
      ({ progress, text: t }) => {
        latest = { progress, text: t };
        if (searchPhaseDone) {
          const pct = Math.round(progress * 100);
          showStatus(`正在加载模型 ${pct}%${t ? `：${t}` : ''}`, { progress });
        }
      },
      { signal: controller.signal },
    );
    promise.catch(() => {});
    preload = { promise, latest: () => latest };
  }
  try {
    // ---------- 联网资料检索 ----------
    // 先搜索再加载模型：搜索只需要网络，失败也不影响正常回答（会明确标注“未使用搜索结果”）
    if (shouldSearch()) {
      const previousUserText = session.messages
        .slice(0, -2)
        .filter((message) => message?.role === 'user')
        .at(-1)?.content || '';
      const searchInput = searchInputForTurn(text, previousUserText);
      const query = buildSearchQuery(searchInput);
      assistantMsg.search = { status: 'searching', query, mode: state.search.prefs.mode };
      paintAssistant(bubble, assistantMsg, { streaming: true, label: '正在联网资料检索…' });
      showStatus(`正在联网资料检索${query ? `：${query}` : '…'}`);
      try {
        const res = await runWebSearch(text, {
          queryText: searchInput,
          config: state.search.config,
          prefs: state.search.prefs,
          // 软件版（服务端模式）启用服务端网页搜索：搜索引擎 + 正文抓取，与浏览器资料源并行
          settings: { ...state.search.settings, serverSearch: MODE === 'remote' },
          cache: state.search.cache,
          signal: controller.signal,
          // 只有打到本站 /api/search 时才带登录令牌，其他公开资料源保持匿名直连
          fetchImpl: (url, opts = {}) => {
            if (String(url).startsWith('/api/search')) {
              const token = getToken();
              return fetch(url, {
                ...opts,
                headers: { ...(opts.headers || {}), ...(token ? { authorization: `Bearer ${token}` } : {}) },
              });
            }
            return fetch(url, opts);
          },
        });
        if (res.ok) {
          // 资料块和历史卡必须使用同一组、同一长度的摘录；不能把被上下文预算裁掉的候选
          // 伪装成模型已经看过的 [编号] 来源。
          const evidence = prepareSearchEvidence(res.results, {
            query: res.query,
            mode: res.mode,
            now: res.now || new Date(),
            maxChars: state.search.config.maxContextChars[res.mode] || 1600,
          });
          if (!evidence.results.length) {
            assistantMsg.search = {
              status: 'error',
              query: res.query,
              mode: res.mode,
              reason: '检索到的资料无法放入当前资料长度预算，本条回答未使用外部资料',
              attempts: compactAttempts(res.attempts),
            };
          } else {
            const evidenceProviders = [...new Set(evidence.results.map((item) => item.provider).filter(Boolean))];
            assistantMsg.search = {
              status: 'done',
              query: res.query,
              mode: res.mode,
              provider: evidenceProviders[0] || res.provider,
              providers: evidenceProviders,
              attempts: compactAttempts(res.attempts),
              plan: res.plan
                ? { en: res.plan.en || '', enSource: res.plan.enSource || '', keyword: res.plan.keyword || '' }
                : null,
              results: compactSources(evidence.results, { limit: state.search.config.maxResults }),
              at: Date.now(),
              retrievedAt: res.now || new Date().toISOString(),
              cached: !!res.cached,
              partial: !!res.partial,
            };
            searchBlock = evidence.block;
            showStatus(`已完成联网资料检索（${evidence.results.length} 条资料${res.cached ? '，使用页面缓存' : ''}），正在生成回答…`);
          }
        } else if (res.skipped) {
          // 寒暄 / 纯代码 / 空白：没有发起请求，显示「已跳过」而不是「失败」。
          assistantMsg.search = {
            status: 'skipped',
            query: res.query || '',
            mode: res.mode,
            reason: res.reason || '',
          };
        } else {
          assistantMsg.search = {
            status: 'error',
            query: res.query || query,
            mode: res.mode,
            reason: res.reason,
            // 记下每个来源的状态与失败原因（最多 6 条），用于说明“为什么没搜到”
            attempts: compactAttempts(res.attempts),
          };
        }
      } catch (err) {
        if (err?.name === 'AbortError' || controller.signal.aborted) {
          assistantMsg.stopped = true;
          assistantMsg.search = {
            status: 'stopped',
            query: assistantMsg.search?.query || query,
            mode: state.search.prefs.mode,
          };
        } else {
          assistantMsg.search = {
            status: 'error',
            query: assistantMsg.search?.query || query,
            mode: state.search.prefs.mode,
            reason: err?.message || String(err),
          };
        }
      }
      paintAssistant(bubble, assistantMsg, { streaming: true, label: '正在加载模型…' });
      scrollToBottom();
      persist();
    }
    // 在搜索阶段被停止：不加载模型、不生成
    if (controller.signal.aborted) {
      assistantMsg.stopped = true;
      finalizeAssistant(assistantMsg);
      paintAssistant(bubble, assistantMsg);
      return;
    }

    searchPhaseDone = true;
    let preloadResult = null;
    if (preload) {
      // 与检索并行启动的加载：这里只等它结束（进度已由回调上报）。
      preloadResult = await preload.promise;
    }
    if (!state.backend.isReady()) {
      paintAssistant(bubble, assistantMsg, { streaming: true, label: '正在加载模型…' });
      const latest = preload?.latest?.();
      if (latest) {
        const pct = Math.round((latest.progress || 0) * 100);
        showStatus(`正在加载模型 ${pct}%${latest.text ? `：${latest.text}` : ''}`, { progress: latest.progress || 0 });
      }
      // 加载可以被停止按钮取消（signal），长时间没有进展会超时并提示重试
      const loaded =
        preloadResult ??
        (await state.backend.ensureReady(({ progress, text: t }) => {
          if (t) progressText = t;
          const pct = Math.round(progress * 100);
          showStatus(`正在加载模型 ${pct}%${t ? `：${t}` : ''}`, { progress });
        }, { signal: controller.signal }));
      await refreshIdleStatus();
      refreshModelCacheStatus();
      // 输出自检的结果（异常、已切换到 fp32 等）
      if (loaded?.note) showStatus(loaded.note, { error: loaded.healthy === false });
    } else if (preloadResult?.note) {
      await refreshIdleStatus();
      refreshModelCacheStatus();
      showStatus(preloadResult.note, { error: preloadResult.healthy === false });
    }

    let loopDetected = false;
    sentMessages = searchBlock ? withSearchContext(context, searchBlock) : context;
    for await (const delta of state.backend.stream({
      messages: sentMessages,
      signal: controller.signal,
      // 本机版网页：带上聊天模型选择栏选中的模型（仅本条请求生效，API 的模型选择互不影响）
      // 本机版：把网页聊天选用的模型带上（桥接网页模型或本机大模型都靠它路由）
      model: MODE === 'remote' ? localChatModelId || undefined : undefined,
      onLoop: () => {
        loopDetected = true;
      },
    })) {
      assistantMsg.content += delta;
      scheduleRender();
    }
    if (frame) cancelAnimationFrame(frame);
    noteUsageOnce(controller.signal.aborted ? 'stopped' : 'ok');
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
      noteUsageOnce('stopped');
      paintAssistant(bubble, assistantMsg);
    } else if (err instanceof ApiError && err.status === 401 && state.backend.mode === 'remote') {
      // 账号令牌失效：清掉令牌并退回访客状态（普通使用不再需要登录）。
      // 本次问题退回输入框，重新登录账号后可直接再发送。
      setToken('');
      accountState.name = null;
      renderAccountUI();
      session.messages.splice(-2, 2);
      if (isFirst) session.title = '新会话';
      persist();
      els.input.value = text;
      autoGrow();
      showStatus('账号登录已过期，已切换回访客状态（不影响正常使用）；需要同步记录时请重新登录账号。', { error: true });
      return;
    } else {
      noteUsageOnce('error');
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

// ---------- 工具面板（思考模式 / 滑动窗口 / 联网资料检索） ----------
/*
 * 功能变多后，把这些开关统一收进输入框右侧的“工具”按钮：
 * 点开后是一列带开关的工具行，点某一行会在下方展开对应的说明卡片（参考 Qwen 网页端的工具选择）。
 * 每张卡片只负责展示与控制自己那一项，新增工具时在 toolDefs() 里加一条即可。
 */

/** 用 DOM API 建节点：搜索结果的标题、摘要一律按纯文本写入，避免被当成 HTML。 */
function node(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'text') el.textContent = value;
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2).toLowerCase(), value);
    else el.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

/** 当前的工具清单。available 为 false 时行仍然显示（附原因），避免用户以为功能消失了。 */
function toolDefs() {
  // 思考开关两种模式都可用：网页版作用于本机 WebLLM 模型，服务端模式透传给
  // 服务端下发 enable_thinking。此前服务端模式被写死为不可用，开关置灰且
  // 显示关闭、模型却照常思考，已修复为全程贯通。
  const thinkingSupported = !!state.backend?.supportsThinking();
  const cfg = state.search.config;
  const searchOk = searchAvailable(cfg, state.search.settings);
  return [
    {
      id: 'thinking',
      icon: '🧠',
      name: '思考模式',
      desc: thinkingSupported ? '先思考再回答，更慢但更充分' : '仅推理模型可用（当前模型不支持）',
      available: thinkingSupported,
      enabled: () => !!state.backend?.thinkingEnabled(),
      toggle: (on) => setThinkingTool(on),
      card: renderThinkingCard,
    },
    {
      id: 'sliding',
      icon: '▤',
      name: '滑动窗口',
      desc: '历史超出窗口时，最早的消息滚出',
      available: MODE === 'webllm',
      enabled: () => !!state.backend?.slidingEnabled(),
      toggle: (on) => setSlidingTool(on),
      card: renderSlidingCard,
    },
    {
      id: 'search',
      icon: '🌐',
      name: '联网资料检索',
      desc: searchOk
        ? `${SEARCH_MODE_LABELS[state.search.prefs.mode] || '联网资料检索'}：把公开资料作为参考`
        : cfg.enabled
          ? '未配置可用的资料源'
          : '本站已关闭联网资料检索',
      available: searchOk,
      enabled: () => !!state.search.prefs.enabled && searchOk,
      toggle: (on) => setSearchTool(on),
      card: renderSearchCard,
    },
  ];
}

function toolRow(def) {
  const on = def.available && def.enabled();
  const disabled = !def.available || !!state.streaming;
  const input = node('input', {
    type: 'checkbox',
    'data-tool-toggle': def.id,
    'aria-label': `${def.name}开关`,
  });
  input.checked = on;
  input.disabled = disabled;
  return node(
    'div',
    {
      class: `tool-item${def.available ? '' : ' is-unavailable'}${state.toolCardId === def.id ? ' is-open' : ''}`,
    },
    node(
      'button',
      {
        type: 'button',
        class: 'tool-main',
        'data-tool-open': def.id,
        'aria-expanded': state.toolCardId === def.id ? 'true' : 'false',
        'aria-controls': 'toolCard',
      },
      node('span', { class: 'tool-icon', 'aria-hidden': 'true' }, def.icon),
      node(
        'span',
        { class: 'tool-text' },
        node('span', { class: 'tool-name' }, def.name),
        node('span', { class: 'tool-desc' }, def.desc),
      ),
      node('span', { class: 'tool-chevron', 'aria-hidden': 'true' }, '›'),
    ),
    node(
      'label',
      { class: 'tool-switch', title: def.available ? `开启或关闭${def.name}` : def.desc },
      input,
      node('span', { class: 'switch-track' }, node('span', { class: 'switch-dot' })),
    ),
  );
}

/** 重建工具行、角标与已展开的卡片。状态变化（换模型、改参数、开始生成）后调用。 */
function refreshTools() {
  if (!els.toolList) return;
  const defs = toolDefs();
  const focused = document.activeElement;
  const focusKey = focused?.dataset?.toolToggle || focused?.dataset?.toolOpen || '';
  const focusInside = !!(focused && els.toolList.contains(focused));
  els.toolList.replaceChildren(...defs.map(toolRow));
  const activeDefs = defs.filter((d) => d.available && d.enabled());
  els.toolsBadge.hidden = activeDefs.length === 0;
  els.toolsBadge.textContent = String(activeDefs.length);
  els.toolsBtn.classList.toggle('has-active', activeDefs.length > 0);
  els.toolsBtn.title = activeDefs.length
    ? `工具（已开启：${activeDefs.map((d) => d.name).join('、')}）`
    : '工具：思考模式、滑动窗口、联网资料检索';
  const current = defs.find((d) => d.id === state.toolCardId);
  if (!current) closeToolCard();
  else renderToolCard(current);
  if (focusInside && focusKey) {
    const again =
      els.toolList.querySelector(`[data-tool-toggle="${focusKey}"]`) ||
      els.toolList.querySelector(`[data-tool-open="${focusKey}"]`);
    again?.focus({ preventScroll: true });
  }
}

/** 卡片外壳：标题 + 当前状态 + 内容。 */
function toolCardShell({ title, state: stateText, body }) {
  return node(
    'div',
    { class: 'tool-card-inner' },
    node(
      'div',
      { class: 'tool-card-head' },
      node('span', { class: 'tool-card-title' }, title),
      stateText ? node('span', { class: 'tool-card-state' }, stateText) : null,
    ),
    body,
  );
}

function renderToolCard(def) {
  els.toolCard.hidden = false;
  els.toolCard.replaceChildren(def.card(def));
}

function closeToolCard() {
  state.toolCardId = null;
  if (els.toolCard) {
    els.toolCard.hidden = true;
    els.toolCard.replaceChildren();
  }
}

/** 展开 / 收起某张卡片（同一时间只展开一张，再次点击同一行则收起）。 */
function openToolCard(id) {
  state.toolCardId = state.toolCardId === id ? null : id;
  refreshTools();
}

function toggleToolsPanel(force) {
  if (!els.toolsPanel) return;
  const open = typeof force === 'boolean' ? force : els.toolsPanel.hidden;
  els.toolsPanel.hidden = !open;
  els.toolsBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) {
    refreshTools();
    els.toolList.querySelector('[data-tool-open]')?.focus({ preventScroll: true });
  } else {
    closeToolCard();
  }
}

function renderThinkingCard() {
  if (MODE !== 'webllm') {
    // 服务端模式：开关经服务端下发 enable_thinking 给模型（本机大模型立刻生效）
    const on = !!state.backend?.thinkingEnabled();
    return toolCardShell({
      title: '思考模式',
      state: on ? '已开启' : '已关闭',
      body: node(
        'div',
        { class: 'tool-card-body' },
        node('p', {}, '开启后，推理模型（Qwen3、DeepSeek-R1 等）会先写出思考过程再回答；关闭后直接回答、速度更快。开关由服务端下发给模型，下一条消息立即生效，不需要重新加载模型。'),
        node('p', { class: 'tool-note' }, '少数模型对关闭思考的支持取决于其对话模板；若关闭后仍有思考过程，是该模型模板不接受此开关。'),
      ),
    });
  }
  const supported = !!state.backend?.supportsThinking();
  const on = supported && !!state.backend?.thinkingEnabled();
  const eff = effectiveParams(state.backend.modelId);
  const win = activeContextWindow(eff);
  const body = node(
    'div',
    { class: 'tool-card-body' },
    node(
      'p',
      {},
      '开启后，推理模型（Qwen3、DeepSeek-R1 等）会先写出思考过程再回答；思考过程可折叠查看，不计入上下文窗口，也不会作为历史发送给模型。',
    ),
    node(
      'p',
      { class: 'muted small' },
      supported
        ? `当前模型：${modelDisplayName(state.backend.modelId)}。引擎窗口：${
            on ? `${win} + ${eff.thinkingWindow}（思考窗口）= ${win + eff.thinkingWindow}` : String(win)
          }。切换后会释放多余显存，下次发送时按新窗口重新加载模型。`
        : '当前模型不是推理模型，开关不生效。切换到 Qwen3 / DeepSeek-R1 等推理模型后再试。',
    ),
    node('p', { class: 'tool-note' }, '关闭思考后回答更快，但复杂题目更容易出错；思考窗口用完时系统会强制输出答案。'),
  );
  return toolCardShell({
    title: '思考模式',
    state: supported ? (on ? '已开启' : '已关闭') : '不可用',
    body,
  });
}

function renderSlidingCard() {
  if (MODE !== 'webllm') {
    return toolCardShell({
      title: '滑动窗口',
      state: '不可用',
      body: node(
        'div',
        { class: 'tool-card-body' },
        node('p', { class: 'muted small' }, '服务端模式的上下文由服务器管理，本站不提供滑动窗口开关。'),
      ),
    });
  }
  const eff = effectiveParams(state.backend.modelId);
  const on = !!state.backend?.slidingEnabled();
  const win = activeContextWindow(eff);
  const cfg = state.appConfig.webllm || {};
  const body = node(
    'div',
    { class: 'tool-card-body' },
    node(
      'p',
      {},
      '开启后上下文窗口固定为滑动窗口大小，历史超出时最早的消息滚出窗口；全局知识库与管理员反馈始终保留在最前面。',
    ),
    node(
      'p',
      { class: 'muted small' },
      on
        ? `当前：滑动窗口 ${eff.slidingWindow} token（思考窗口另计）。历史不再按条数和字数截断。`
        : `当前：上下文窗口 ${eff.contextWindow} token；历史按 ${cfg.maxMessages ?? 15} 条、${cfg.contextChars ?? 8000} 字截断。开启后按 ${eff.slidingWindow} 计算。`,
    ),
    node(
      'p',
      { class: 'tool-note' },
      '窗口越大占用的显存越多。切换后会释放多余显存，下次发送时按新窗口加载；管理员可在参数中调整（滑动窗口 2048–20000，上下文窗口 4096–20000）。',
    ),
  );
  return toolCardShell({ title: '滑动窗口', state: on ? '已开启' : '已关闭', body });
}

/** 搜索方式的两张可选卡片 + 搜索源说明 + 隐私说明（管理员还能就地配置搜索源）。 */
function renderSearchCard() {
  const cfg = state.search.config;
  const prefs = state.search.prefs;
  const usable = searchAvailable(cfg, state.search.settings);
  const body = node('div', { class: 'tool-card-body' });

  if (!cfg.enabled) {
    body.append(
      node(
        'p',
        { class: 'muted small' },
        '部署者已在 config.json 中关闭联网资料检索（webSearch.enabled = false）。本机管理员也无法开启。',
      ),
    );
    return toolCardShell({ title: '联网资料检索', state: '已关闭', body });
  }

  body.append(node('p', {}, '搜索方式：'));
  const grid = node('div', { class: 'search-mode-grid' });
  for (const mode of SEARCH_MODES) {
    const active = prefs.mode === mode;
    grid.append(
      node(
        'button',
        {
          type: 'button',
          class: `search-mode-card${active ? ' is-active' : ''}`,
          'data-search-mode': mode,
          'aria-pressed': active ? 'true' : 'false',
        },
        node('span', { class: 'search-mode-name' }, SEARCH_MODE_LABELS[mode] || mode),
        node(
          'span',
          { class: 'search-mode-desc' },
          mode === 'quick'
            ? '只用公开资料源摘要：更快，但不是完整网页抓取'
            : `补取前 ${cfg.smart.maxPages} 条的维基导言、Stack 回答或 HN 讨论：更完整，受总超时约束`,
        ),
      ),
    );
  }
  body.append(grid);

  const advanced = planAdvanced(cfg, state.search.settings);
  const keyless = planKeyless(cfg);
  const names = (list) => list.map((p) => PROVIDER_LABELS[p.id] || p.id).join('、');
  body.append(
    node(
      'p',
      { class: 'muted small' },
      usable
        ? advanced.length
          ? `资料源：${names(advanced)} 与 ${names(keyless) || '无默认资料源'} 会并行交叉取样；单一自定义源不会独占答案。最多 ${cfg.maxResults} 条资料，单请求 ${Math.round(cfg.timeoutMs / 1000)} 秒、整次 ${Math.round(cfg.totalTimeoutMs / 1000)} 秒。`
          : `资料源：${names(keyless) || '无'}。维基/Stack/HN 是浏览器直连公开 API，不是完整互联网索引。最多 ${cfg.maxResults} 条资料，单请求 ${Math.round(cfg.timeoutMs / 1000)} 秒、整次 ${Math.round(cfg.totalTimeoutMs / 1000)} 秒。`
        : '没有可用的资料源：请在 config.json 的 webSearch.providers 中启用维基百科等公开资料源，或由管理员在下方填写受信任的 SearXNG / 自定义接口。',
    ),
  );
  body.append(
    node(
      'p',
      { class: 'tool-note' },
      '浏览器直连资料源会看到查询词与网络地址；不会发送对话历史或模型设置。若消息含疑似密钥、授权头、私钥或环境变量，系统会完全跳过联网检索。默认资料源覆盖维基百科、Stack Overflow 与 Hacker News；它们不是完整网页搜索。页面缓存最多保留数分钟且不单独写入 localStorage；为便于回看，本轮查询、资料链接、摘录与来源状态会随本机聊天历史保存，删除会话或清空本机数据即可移除。',
    ),
  );
  if (prefs.enabled && usable) {
    body.append(node('p', { class: 'muted small' }, `示例查询：「${buildSearchQuery(els.input?.value || '') || '（发送时从你的问题中提取）'}」`));
  }
  if (isAdmin()) {
    const details = node(
      'details',
      { class: 'search-advanced' },
      node('summary', { class: 'tool-subtitle' }, '进阶（可选）：自建搜索源'),
    );
    details.append(renderSearchSettings());
    body.append(details);
  }
  return toolCardShell({
    title: '联网资料检索',
    state: usable ? (prefs.enabled ? `已开启 · ${SEARCH_MODE_LABELS[prefs.mode]}` : '已关闭') : '不可用',
    body,
  });
}

/** 管理员：搜索源设置（仅本机 localStorage，不上传；与 config.json 的默认值叠加）。 */
function renderSearchSettings() {
  const settings = state.search.settings;
  const field = (labelText, key, placeholder, type = 'url') =>
    node(
      'label',
      { class: 'search-setting' },
      node('span', { class: 'small muted' }, labelText),
      node('input', {
        type,
        class: 'search-setting-input',
        'data-search-setting': key,
        placeholder,
        value: settings[key] || '',
        spellcheck: 'false',
        autocomplete: 'off',
      }),
    );
  const result = node('p', { class: 'muted small search-test-result', id: 'searchTestResult' });
  const box = node(
    'div',
    { class: 'search-settings' },
    node(
      'p',
      { class: 'muted small' },
      '默认来源只是公开资料 API，不是完整网页搜索。想检索更广网页可填写受信任服务；设置仅保存在本机。HTTPS 必填，本机 localhost/127.0.0.1 可用 HTTP。',
    ),
    field('SearXNG 地址（需开启 format=json 与跨域）', 'searxngBase', 'https://searx.example.com'),
    field('自定义接口（返回 {results:[{title,url,snippet}]}）', 'customEndpoint', 'https://example.com/search'),
    field('正文读取模板（可选，深入检索用；保留末尾 /，或使用 {url} 占位）', 'readerTemplate', 'https://r.jina.ai/', 'text'),
    node(
      'div',
      { class: 'search-setting-actions' },
      node('button', { type: 'button', class: 'btn ghost small', 'data-search-action': 'save' }, '保存'),
      node('button', { type: 'button', class: 'btn ghost small', 'data-search-action': 'test' }, '测试搜索'),
    ),
    node('p', { class: 'tool-note' }, '进阶来源会和默认资料源并行交叉取样；正文读取服务会收到待读取页面的 URL，请只填写你信任的 HTTPS 服务。'),
    result,
  );
  return box;
}

/** 从卡片里的输入框读取搜索源设置。 */
function readSearchSettingInputs() {
  const out = { ...state.search.settings };
  for (const input of els.toolCard.querySelectorAll('[data-search-setting]')) {
    out[input.dataset.searchSetting] = input.value.trim();
  }
  return out;
}

async function saveSearchSettingsFromCard({ test = false } = {}) {
  const raw = readSearchSettingInputs();
  const invalidReader = !!raw.readerTemplate && !cleanReaderTemplate(raw.readerTemplate);
  saveProviderSettings(raw);
  // 读取规范化后的值：正文模板必须保留 / 或 {url} 拼接边界。
  state.search.settings = loadProviderSettings();
  state.search.cache.clear();
  // 卡片会按新设置重建（搜索源顺序可能变化），因此先刷新再取结果节点
  refreshTools();
  const target = els.toolCard.querySelector('.search-test-result');
  if (!test) {
    const message = invalidReader
      ? '已保存可用设置；正文读取模板无效（仅允许 HTTPS 或本机 HTTP，且需以 /、?、&、=、: 结尾或包含 {url}），已忽略。'
      : '已保存（仅本机，不上传；已清除本页检索缓存）。';
    if (target) target.textContent = message;
    flashStatus(message);
    return;
  }
  if (target) target.textContent = '正在测试…';
  try {
    const res = await runWebSearch('二分查找 时间复杂度', {
      config: state.search.config,
      prefs: { ...state.search.prefs, enabled: true },
      settings: state.search.settings,
    });
    if (target) {
      target.textContent = res.ok
        ? `测试成功：${res.results.length} 个结果（${PROVIDER_LABELS[res.provider] || res.provider}）。第一条：${res.results[0].title}`
        : `测试失败：${res.reason || '没有结果'}`;
    }
  } catch (err) {
    if (target) target.textContent = `测试失败：${err?.message || err}`;
  }
}

/** 思考开关：改变引擎窗口，需要释放多余显存（生成中则等下次发送）。 */
async function setThinkingTool(on) {
  if (!state.backend) return;
  state.backend.setThinking(on);
  refreshTools();
  if (MODE !== 'webllm') {
    flashStatus(on
      ? '已开启思考模式：推理模型会先展示思考过程（可折叠），回答更充分但更慢。下一条消息立即生效。'
      : '已关闭思考模式：推理模型直接给出回答，速度更快。下一条消息立即生效。');
    return;
  }
  const eff = effectiveParams(state.backend.modelId);
  const win = activeContextWindow(eff);
  const released = state.streaming ? false : await releaseStaleEngine();
  const tail = released ? '已释放多余显存，' : '';
  flashStatus(
    on
      ? `已开启思考模式：推理模型会先展示思考过程（可折叠），回答更充分但更慢。${tail}下次发送时将按 ${win + eff.thinkingWindow} 窗口重新加载模型（思考内容不计入窗口）。`
      : `已关闭思考模式：推理模型直接给出回答，速度更快。${tail}下次发送时将按 ${win} 窗口重新加载模型。`,
  );
}

/** 滑动窗口开关：同样可能改变引擎窗口。 */
async function setSlidingTool(on) {
  if (!state.backend) return;
  state.backend.setSliding(on);
  refreshTools();
  const eff = effectiveParams(state.backend.modelId);
  const win = activeContextWindow(eff);
  const released = state.streaming ? false : await releaseStaleEngine();
  const tail = released ? '已释放多余显存，' : '';
  flashStatus(
    on
      ? `已开启滑动窗口：上下文固定为 ${win}（思考窗口另计），超出的最早历史会滚出窗口，管理员信息始终保留。${tail}下次发送时按新窗口加载模型。`
      : `已关闭滑动窗口：恢复为上下文窗口 ${eff.contextWindow}，并按 ${state.appConfig.webllm.maxMessages} 条、${state.appConfig.webllm.contextChars} 字限制历史。${tail}下次发送时按新窗口加载模型。`,
  );
}

/** 本次发送是否要走联网资料检索（开关已开，且至少有一个可用搜索源）。 */
function shouldSearch() {
  return !!state.search.prefs.enabled && searchAvailable(state.search.config, state.search.settings);
}

/** 联网资料检索开关：只改本机偏好，不影响引擎窗口，因此不需要重新加载模型。 */
function setSearchTool(on) {
  if (on && !searchAvailable(state.search.config, state.search.settings)) {
    showStatus('无法开启联网资料检索：没有可用的资料源。管理员可在工具面板中填写 SearXNG 或自定义接口。', { error: true });
    refreshTools();
    return;
  }
  state.search.prefs = { ...state.search.prefs, enabled: on, mode: state.search.prefs.mode };
  saveSearchPrefs(state.search.prefs);
  refreshTools();
  flashStatus(
    on
      ? `已开启${SEARCH_MODE_LABELS[state.search.prefs.mode]}：发送时会先检索公开资料，再把可审计摘录作为参考；疑似凭据会自动跳过联网。`
      : '已关闭联网资料检索：之后的回答不再访问网络。',
  );
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
async function applyPreset(id) {
  if (!setPresetId(id)) return;
  const preset = findPreset(id) || findPreset(DEFAULT_PRESET_ID);
  setThinkingPref(preset.thinking);
  renderPresetOptions();
  refreshTools();
  refreshParamViews();
  // 预设可能改变引擎窗口（上下文、滑动窗口、思考）：释放多余显存，下次发送时按新窗口加载
  const released = await releaseAfterSettingChange();
  const tail = released ? '已释放多余显存，' : '';
  flashStatus(`已切换到预设「${preset.label}」：${preset.note} ${tail}下次发送时按新参数生效。`);
}

/** 窄屏下拉框只显示模型名（去掉“（非推理·小·纯文本…）”这类说明），完整标签保留在 data-label 与悬停提示中。 */
function shortModelLabel(label) {
  const text = String(label || '');
  return text.split('（')[0].trim() || text;
}

/**
 * 生成模型下拉框的子节点。
 * compact 为 true 时（标题栏里的小下拉框）显示短名称，避免在 400px 宽时被截断。
 */
function buildModelOptions({ compact = false } = {}) {
  const rename = (opt, label) => {
    opt.dataset.short = shortModelLabel(label);
    opt.textContent = compact ? opt.dataset.short : label;
    return opt;
  };
  // 官方模型的备注（参数量级、显存估计）放在悬停提示中，来自 config.json 的 webllm.models[].note
  const notes = new Map((state.appConfig?.webllm?.models || []).map((m) => [m.id, m.note || '']));
  const children = state.backend.models.map((m) => {
    const opt = rename(makeOption(m.id, m.label), m.label);
    if (notes.get(m.id)) opt.title = notes.get(m.id);
    return opt;
  });
  // 待转换模型：禁用（尚不能运行），与正式模型分组显示，备注放在悬停提示中
  const pending = pendingModels(state.appConfig?.webllm);
  if (pending.length) {
    const group = document.createElement('optgroup');
    group.label = '待转换（暂不可用）';
    for (const p of pending) {
      const opt = rename(makeOption(p.id, p.label), p.label);
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
    for (const e of locals) group.append(rename(makeOption(e.id, e.label), e.label));
    children.push(group);
  }
  return children;
}

/** 重建下拉框：官方模型在前，“新建AI”分组在后（仅本浏览器可用，不上传云端）。 */
function renderModelOptions(selectId = state.backend.modelId) {
  // 侧边栏的大下拉框 + 窄屏 / 嵌入时标题栏里的小下拉框，两者内容与选中项始终同步
  for (const select of [els.modelSelect, els.modelSelectCompact]) {
    if (!select) continue;
    select.replaceChildren(...buildModelOptions({ compact: select === els.modelSelectCompact }));
    if ([...select.options].some((o) => o.value === selectId)) select.value = selectId;
  }
  refreshTools();
}

/** 在下拉框中给已下载到本机的模型加上 ✓。 */
async function refreshModelCacheStatus() {
  if (MODE !== 'webllm' || !state.backend) return;
  try {
    const cached = await listCachedModelIds(await getRecords());
    for (const select of [els.modelSelect, els.modelSelectCompact]) {
      const compact = select === els.modelSelectCompact;
      for (const opt of select?.options || []) {
        const base = (compact ? opt.dataset.short : opt.dataset.label) || opt.dataset.label || opt.value;
        opt.textContent = cached.has(opt.value) ? `✓ ${base}` : base;
      }
    }
  } catch (err) {
    console.warn('无法检查模型缓存状态', err);
  }
}

/** 是否为本机版（网站包）网页：服务端与网页在同一台电脑上，网页还负责用显卡提供模型。 */
function bridgePanelMode() {
  return MODE === 'remote' && !!state.appConfig.localBrowserBridge;
}

/**
 * 本机管理员面板的操作权限：静态版沿用管理员密钥（软门禁）；
 * 本机版网页只有登录后才能打开主界面，且面板管理的就是这台电脑浏览器里的模型与参数，
 * 因此登录后即可使用（API 的参数/Key 仍由“API 管理”的独立管理口令保护）。
 */
function panelAdmin() {
  return MODE === 'webllm' ? isAdmin() : bridgePanelMode();
}

/** 导出、导入、删除新建 AI 都需要管理员权限。 */
function requireAdmin(action) {
  if (panelAdmin()) return true;
  showStatus(`${action}需要管理员权限。请点击右上角“管理员面板”，输入管理员密钥。`, { error: true });
  return false;
}

/**
 * 下载网站包（构建时按平台生成的 site-<平台>.zip，每个包只含对应平台的内置运行时）。
 * 仅管理员。不包含任何 AI 模型。
 */
async function downloadSitePackage(file, btn) {
  const isTraining = file === TRAINING_PACKAGE_ASSET;
  const kind = isTraining ? '一站式训练包' : STANDALONE_ASSETS.includes(file) ? '独立软件' : '网站包';
  if (!requireAdmin(`下载${kind}`)) return;
  const label = btn.textContent;
  btn.disabled = true;
  btn.textContent = '正在开始下载…';
  showStatus(`正在开始下载${kind} ${file}，请留意浏览器右上角（下载按钮处）的进度。`);
  try {
    // 网站包与训练包发布在 GitHub Release 中（config.json 的 release 配置）；其余文件仍按相对路径从本站下载。
    const fromRelease = isReleaseAsset(file);
    const url = fromRelease ? releaseAssetUrl(state.appConfig?.release, file) : file;
    // 先用 HEAD 轻量确认文件存在（只取响应头，不下载包体）；HEAD 本身失败时不阻断，交给浏览器下载去处理。
    // Release 直达链接会跨域重定向，浏览器通常不允许 fetch 读取响应，这种失败同样不阻断。
    let missingStatus = 0;
    try {
      const head = await fetch(url, { method: 'HEAD', cache: 'no-store' });
      if (!head.ok) missingStatus = head.status;
    } catch {
      /* 静态服务器不支持 HEAD / 跨域重定向等情况：继续走浏览器原生下载 */
    }
    if (missingStatus) {
      throw new Error(
        fromRelease
          ? `找不到${kind} ${file}（HTTP ${missingStatus}）。可能尚未发布 Release，请到“查看全部下载”确认。`
          : `找不到${kind} ${file}（HTTP ${missingStatus}）。请先运行 npm run build 生成。`,
      );
    }
    // 用浏览器原生下载（<a download>）：浏览器下载栏会立刻显示真实进度。
    // 不要再先用 fetch 把整个包读进内存再落盘——那段时间页面毫无反应，最后却“突然”下载完成。
    // 跨域的 Release 链接 download 属性会被浏览器忽略，改用新标签页承载跳转，文件名由服务端响应决定（与资产名一致）。
    const a = document.createElement('a');
    a.href = url;
    a.download = file;
    if (fromRelease) {
      a.target = '_blank';
      a.rel = 'noopener';
    }
    document.body.appendChild(a);
    a.click();
    a.remove();
    btn.textContent = '已开始下载 ✓';
    showStatus(
      isTraining
        ? `已开始下载一站式训练包 ${file}：请看浏览器右上角的下载进度；下载完成后解压，按里面 training/README.md 的说明安装依赖并运行（需要 GPU）。`
        : `已开始下载网站包 ${file}：请看浏览器右上角的下载进度；下载完成后解压，双击里面的启动文件即可。网站包不包含任何 AI 模型。`,
    );
  } catch (err) {
    showStatus(`下载${kind}失败：${err.message}`, { error: true });
  } finally {
    btn.disabled = !isAdmin();
    setTimeout(() => {
      btn.textContent = label;
    }, 4000);
  }
}

/** 按当前电脑的系统给四个网站包下载按钮标一个推荐项（Mac 芯片无法可靠自动区分时默认推荐 Apple 芯片版）。 */
function renderSitePackageHint() {
  if (!els.sitePackageBtns?.length) return;
  const ua = navigator.userAgent || '';
  let file = '';
  let label = '';
  if (/Windows/i.test(ua)) {
    file = 'site-win-x64.zip';
    label = 'Windows 版';
  } else if (/Macintosh|Mac OS X/i.test(ua)) {
    file = 'site-darwin-arm64.zip';
    label = 'Mac（Apple 芯片）版';
  } else if (/Linux/i.test(ua)) {
    file = 'site-linux-x64.zip';
    label = 'Linux 版';
  }
  for (const btn of els.sitePackageBtns) {
    btn.classList.toggle('is-recommended', btn.dataset.sitePackage === file);
  }
  if (els.sitePackageHint) {
    els.sitePackageHint.textContent = file
      ? `按你正在用的这台电脑，推荐下载「${label}」。下错版本也没关系，回来重新下对应版本即可。`
      : '';
  }
}

// ---------- 用量统计面板 ----------
// 网页用量保存在本浏览器（聚合计数 + 定长近期日志，见 usage-stats.js）；
// API 用量由本机服务聚合，管理员登录 API 管理后经 /api/admin/usage-stats 读取。

let webUsageStats = null;
function getWebUsageStats() {
  if (!webUsageStats) webUsageStats = loadUsageStats(localStorage, USAGE_STORAGE_KEY);
  return webUsageStats;
}

function makeEl(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** 模型 ID → 展示名：网页下拉标签 > 本机“新建AI” > 桥接模型名 > 原 ID。 */
function usageModelLabel(id) {
  if (!id || id === 'unknown') return '未知模型';
  const known = state.appConfig.webllm?.models?.find((m) => m.id === id);
  if (known?.label) return known.label;
  try {
    const local = loadLocalModels().find((m) => m.id === id);
    if (local?.label) return local.label;
  } catch { /* 本机清单读取失败时回退到 ID */ }
  const bridged = apiBridgeModels.find((m) => m.id === id);
  if (bridged?.label || bridged?.name) return bridged.label || bridged.name;
  const native = nativeModels().find((m) => m.id === id);
  if (native?.name) return `${native.name}（本机大模型）`;
  return id;
}

function renderUsageCards(container, summary) {
  container.replaceChildren();
  const cards = [
    { label: '总调用', value: formatTokens(summary.totalCalls), sub: summary.totalErrors ? `其中失败 ${formatTokens(summary.totalErrors)} 次` : '暂无失败' },
    { label: '提示 Token', value: formatTokens(summary.promptTokens), sub: '发送给模型的输入' },
    { label: '生成 Token', value: formatTokens(summary.completionTokens), sub: '模型输出' },
    { label: '总 Token', value: formatTokens(summary.totalTokens), sub: '提示 + 生成', accent: true },
  ];
  for (const c of cards) {
    const box = makeEl('div', `usage-stat${c.accent ? ' is-accent' : ''}`);
    box.append(makeEl('div', 'usage-stat-value', c.value), makeEl('div', 'usage-stat-label', `${c.label} · ${c.sub}`));
    container.appendChild(box);
  }
}

function renderUsageDays(container, summary) {
  container.replaceChildren();
  const max = Math.max(1, ...summary.lastDays.map((d) => d.totalTokens));
  for (const d of summary.lastDays) {
    const wrap = makeEl('div', `usage-day${d.totalTokens ? '' : ' is-empty'}`);
    const bar = makeEl('div', 'usage-day-bar');
    bar.style.height = `${d.totalTokens ? Math.max(10, Math.round((d.totalTokens / max) * 100)) : 4}%`;
    bar.title = `${d.day}：调用 ${d.calls} 次，总 Token ${formatTokens(d.totalTokens)}`;
    wrap.append(bar, makeEl('div', 'usage-day-label', d.day.slice(5)));
    container.appendChild(wrap);
  }
  container.setAttribute('aria-label', `最近 7 天用量：${summary.lastDays.map((d) => `${d.day.slice(5)} ${formatTokens(d.totalTokens)}`).join('，')}`);
}

function renderUsageModels(container, summary) {
  container.replaceChildren();
  if (!summary.models.length) {
    container.appendChild(makeEl('div', 'usage-empty', '暂无记录。先在网页里和模型聊一句，这里就会出现统计。'));
    return;
  }
  const max = Math.max(1, ...summary.models.map((m) => m.totalTokens));
  for (const m of summary.models) {
    const row = makeEl('div', 'usage-model-row');
    const name = makeEl('div', 'usage-model-name', usageModelLabel(m.id));
    name.title = m.id;
    const total = makeEl('div', 'usage-model-total', `${formatTokens(m.totalTokens)} Token`);
    const meta = makeEl(
      'div',
      'usage-model-meta',
      `调用 ${formatTokens(m.calls)} 次${m.errors ? `（失败 ${formatTokens(m.errors)}）` : ''} · 提示 ${formatTokens(m.promptTokens)} · 生成 ${formatTokens(m.completionTokens)} · 今日 ${formatTokens(m.today.totalTokens)}`,
    );
    const bar = makeEl('div', 'usage-model-bar');
    const fill = document.createElement('i');
    fill.style.width = `${Math.max(2, Math.round((m.totalTokens / max) * 100))}%`;
    bar.appendChild(fill);
    row.append(name, total, meta, bar);
    container.appendChild(row);
  }
}

function renderUsageRecent(container, summary) {
  container.replaceChildren();
  if (!summary.recent.length) {
    container.appendChild(makeEl('div', 'usage-empty', '暂无使用日志。'));
    return;
  }
  const STATUS = { ok: ['成功', 'is-ok'], error: ['失败', 'is-error'], stopped: ['已停止', ''] };
  for (const e of summary.recent.slice(0, 15)) {
    const row = makeEl('div', 'usage-recent-row');
    const time = makeEl('span', 'usage-recent-time', new Date(e.t).toLocaleString('zh-CN', { hour12: false }));
    const model = makeEl('span', 'usage-recent-model', usageModelLabel(e.model));
    model.title = e.model;
    const [label, cls] = STATUS[e.status] || STATUS.ok;
    const chip = makeEl('span', `usage-chip ${cls}`.trim(), label);
    const tokens = makeEl(
      'span',
      'usage-recent-tokens',
      e.status === 'error' ? '未计 Token' : `提示 ${formatTokens(e.promptTokens)} → 生成 ${formatTokens(e.completionTokens)}`,
    );
    row.append(time, model, chip, tokens);
    container.appendChild(row);
  }
}

function renderUsageSummary(targets, summary) {
  renderUsageCards(targets.cards, summary);
  renderUsageDays(targets.days, summary);
  renderUsageModels(targets.models, summary);
  renderUsageRecent(targets.recent, summary);
}

const webUsageTargets = () => ({ cards: els.usageWebCards, days: els.usageWebDays, models: els.usageWebModels, recent: els.usageWebRecent });
const apiUsageTargets = () => ({ cards: els.usageApiCards, days: els.usageApiDays, models: els.usageApiModels, recent: els.usageApiRecent });

function renderWebUsage() {
  renderUsageSummary(webUsageTargets(), summarizeUsage(getWebUsageStats()));
}

async function loadApiUsage() {
  const token = state.apiKeyAdminToken;
  els.usageApiHint.textContent = '正在加载 API 用量…';
  els.usageApiBody.hidden = true;
  els.usageApiClear.hidden = true;
  try {
    const res = await fetch('/api/admin/usage-stats', {
      headers: { authorization: `Bearer ${token}` },
      cache: 'no-store',
    });
    if (res.status === 401) {
      // 令牌已失效（过期或被服务端拒绝）：锁定并提示重新验证，避免反复失败却不知道原因
      lockApiKeyManager();
      els.usageApiHint.textContent = '管理员验证已过期，请在“API 接入”中重新验证一次（之后长期有效）。';
      return;
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (state.apiKeyAdminToken !== token) return; // 加载期间管理员已退出/切换，丢弃旧结果
    els.usageApiHint.textContent = '由本机服务统计的全部 API 调用（含其他软件用 Key 发起的请求），口径与接口返回的 usage 一致（按文本长度估算）。';
    renderUsageSummary(apiUsageTargets(), data);
    els.usageApiBody.hidden = false;
    els.usageApiClear.hidden = false;
  } catch (err) {
    els.usageApiHint.textContent = `API 用量加载失败：${err.message}。请确认本机服务在运行、管理员登录未过期后重开本面板。`;
  }
}

function openUsagePanel() {
  renderWebUsage();
  if (els.usageApiSection) {
    els.usageApiSection.hidden = false;
    if (MODE !== 'remote') {
      els.usageApiHint.textContent = '当前是静态版（没有本机 API 服务），暂无 API 调用统计。在本机网站包启动的网页中，用管理员口令登录后可在这里查看。';
      els.usageApiBody.hidden = true;
      els.usageApiClear.hidden = true;
    } else if (state.apiKeyAdminToken) {
      loadApiUsage();
    } else {
      els.usageApiHint.textContent = state.appConfig.apiKeyManagerEnabled
        ? 'API 用量仅管理员可见：请先在“API 接入”中验证一次管理员口令（验证后长期有效，关闭面板、刷新页面都不用再验证；只有点“锁定管理面板”或 30 天到期才需重新验证），然后回到本面板查看。'
        : '服务端未配置 API_ADMIN_PASSWORD，API 用量不可见（管理员功能未启用）。';
      els.usageApiBody.hidden = true;
      els.usageApiClear.hidden = true;
    }
  }
  if (typeof els.usageDialog.showModal === 'function') els.usageDialog.showModal();
  else els.usageDialog.setAttribute('open', '');
}

function closeUsagePanel() {
  if (typeof els.usageDialog.close === 'function') els.usageDialog.close();
  else els.usageDialog.removeAttribute('open');
}

/** 选择性导出本机数据（会话、管理员说明/参数、全局知识库、新建AI清单）。仅管理员。 */
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
    const carriedNote = carried ? `，并带入自定义参数（${paramSummary(carried)}）` : '';
    if (bridgePanelMode()) {
      // 本机版：登记已通过 local-models 的变更事件同步给桥接，这里再刷新网页模型栏与 API 模型列表
      renderBridgeModelOptions();
      renderLocalList();
      refreshBridgeStatus();
      setTimeout(() => refreshBridgeStatus(), 800);
      showStatus(`导入完成：${label} 已加入“新建AI”。网页顶部模型栏可以选用它；API 也可以用它生成（请求中填 model="${record.model_id}"，或先 GET /v1/models 查看）${carriedNote}。`);
      return;
    }
    renderModelOptions(record.model_id);
    state.backend.setModel(record.model_id);
    refreshTools();
    await refreshModelCacheStatus();
    renderLocalList();
    showStatus(`导入完成：${label} 已加入“新建AI”，可离线使用${carriedNote}。发送消息时将直接从本机加载。`);
  } catch (err) {
    showStatus(`${action}失败：${err.message}`, { error: true });
  } finally {
    setImportButtons(panelAdmin());
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
    if (bridgePanelMode()) {
      if (localChatModelId === id) {
        localChatModelId = apiBridgeModels.find((m) => m.id !== id)?.id || '';
        saveLocalChatModel(localChatModelId);
      }
      renderBridgeModelOptions();
      renderLocalList();
      refreshBridgeStatus();
      showStatus(`已删除 ${entry.label}（清除 ${removed} 个缓存文件）。`);
      return;
    }
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
for (const btn of els.sitePackageBtns) {
  btn.addEventListener('click', () => downloadSitePackage(btn.dataset.sitePackage, btn));
}
els.trainingPackageBtn?.addEventListener('click', () => downloadSitePackage(TRAINING_PACKAGE_ASSET, els.trainingPackageBtn));
els.exportDataBtn.addEventListener('click', exportLocalData);
els.usageBtn?.addEventListener('click', openUsagePanel);
els.usageClose?.addEventListener('click', closeUsagePanel);
els.usageWebClear?.addEventListener('click', () => {
  if (!confirm('确定要清空本浏览器保存的网页用量统计吗？此操作不可恢复。')) return;
  webUsageStats = clearUsageStats();
  saveUsageStats(webUsageStats, localStorage, USAGE_STORAGE_KEY);
  renderWebUsage();
});
els.usageApiClear?.addEventListener('click', async () => {
  if (!state.apiKeyAdminToken) return;
  if (!confirm('确定要清空服务端保存的 API 用量统计吗？此操作不可恢复。')) return;
  try {
    const res = await fetch('/api/admin/usage-stats', {
      method: 'DELETE',
      headers: { authorization: `Bearer ${state.apiKeyAdminToken}` },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderUsageSummary(apiUsageTargets(), await res.json());
    els.usageApiHint.textContent = 'API 用量统计已清空。';
  } catch (err) {
    els.usageApiHint.textContent = `清空 API 统计失败：${err.message}`;
  }
});
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

function showMain() {
  els.main.hidden = false;
  refreshAll();
  refreshIdleStatus();
  applyLinkPrompt();
  els.input.focus({ preventScroll: true });
}

// ---------- 账号（仅服务端模式）：不登录也能正常使用；登录只为了把自己的
// 对话记录与参数设定按账号存到服务器，换设备登录同一账号即可恢复 ----------
// 注意：不再有网站入口登录页/入口口令，打开即用；只有管理员操作需要口令。
const accountState = { name: null };
const ACCOUNT_NAME_KEY = 'oj-ai.accountName.v1';
const ACCOUNT_SYNC_KEY = 'oj-ai.accountSync.v1';
/** 随账号同步的本机设定键（对话记录另走 sessions 整体同步）。 */
const ACCOUNT_SETTINGS_KEYS = [
  'oj-ai.thinking.v1',
  'oj-ai.sliding.v1',
  'oj-ai.preset.v1',
  'oj-ai.defaultModel.v1',
  'oj-ai.search.v1',
  'oj-ai.search.settings.v1',
  'oj-ai.localChatModel.v1',
  'oj-ai.adapt.v1',
  'oj-ai.kbText.v1',
];
let accountSyncTimer = null;
let applyingAccountData = false;

function accountLoggedIn() {
  return MODE === 'remote' && !!accountState.name && !!getToken();
}

function collectAccountPayload() {
  const settings = {};
  for (const key of ACCOUNT_SETTINGS_KEYS) {
    try {
      const v = localStorage.getItem(key);
      if (v !== null) settings[key] = v;
    } catch { /* 存储不可用 */ }
  }
  return { v: 1, sessions: state.store, settings };
}

/** 把服务器上的账号数据落到本机（设定写回 localStorage，对话整体替换）。 */
function applyAccountPayload(payload) {
  if (!payload || typeof payload !== 'object') return false;
  applyingAccountData = true;
  try {
    if (payload.settings && typeof payload.settings === 'object') {
      for (const key of ACCOUNT_SETTINGS_KEYS) {
        try {
          if (typeof payload.settings[key] === 'string') localStorage.setItem(key, payload.settings[key]);
          else localStorage.removeItem(key);
        } catch { /* 存储不可用 */ }
      }
    }
    if (payload.sessions && Array.isArray(payload.sessions.sessions)) {
      state.store = payload.sessions;
      if (!getSession(state.store)) state.store.activeId = state.store.sessions[0]?.id || null;
    }
    saveStore(state.store);
    return true;
  } finally {
    applyingAccountData = false;
  }
}

function markAccountSynced(updatedAt) {
  try {
    localStorage.setItem(ACCOUNT_SYNC_KEY, JSON.stringify({ updatedAt: updatedAt || Date.now() }));
  } catch { /* 存储不可用 */ }
}

function setAccountSyncText(text) {
  if (els.accountSyncStatus) els.accountSyncStatus.textContent = text;
}

/** 登录状态下把本机数据推到服务器（防抖；失败只提示不打断使用）。 */
function scheduleAccountSync() {
  if (!accountLoggedIn() || applyingAccountData) return;
  if (els.accountBtn) setAccountSyncText('有更改待同步…');
  clearTimeout(accountSyncTimer);
  accountSyncTimer = setTimeout(() => { void syncAccountNow(); }, 1200);
}

async function syncAccountNow() {
  if (!accountLoggedIn()) return false;
  try {
    const out = await putAccountData(collectAccountPayload());
    markAccountSynced(out?.updatedAt);
    setAccountSyncText(`已同步到服务器（${new Date().toLocaleTimeString()}）`);
    return true;
  } catch (err) {
    setAccountSyncText(`同步失败：${err.message || '网络错误'}（本机记录不受影响，稍后会再试）`);
    return false;
  }
}

/** 登录/注册成功后：服务器上有数据就拉下来恢复，没有就把本机数据推上去。 */
async function afterAccountAuth(data, { isRegister = false } = {}) {
  accountState.name = data?.account?.name || els.accountName.value.trim();
  try {
    localStorage.setItem(ACCOUNT_NAME_KEY, accountState.name);
  } catch { /* 存储不可用 */ }
  const remote = await fetchAccountData().catch(() => null);
  if (remote?.payload) {
    applyAccountPayload(remote.payload);
    markAccountSynced(remote.updatedAt);
    setAccountSyncText('已从服务器恢复你的对话与设定。');
  } else {
    await putAccountData(collectAccountPayload())
      .then((out) => markAccountSynced(out?.updatedAt))
      .catch(() => {});
    setAccountSyncText(isRegister ? '账号已创建，本机对话与设定已存到服务器。' : '已登录，本机对话与设定已存到服务器。');
  }
  renderAccountUI();
  refreshAll();
  renderPresetOptions();
  refreshTools();
}

function renderAccountUI() {
  if (!els.accountBtn) return;
  const logged = accountLoggedIn();
  els.accountBtn.textContent = logged ? `账号：${accountState.name}` : '账号';
  if (els.accountAnon) els.accountAnon.hidden = logged;
  if (els.accountLogged) els.accountLogged.hidden = !logged;
  if (logged && els.accountCurrentName) els.accountCurrentName.textContent = accountState.name;
}

/** 启动时恢复账号：有令牌则校验并按需拉取服务器数据（服务器更新时以服务器为准）。 */
async function initAccount() {
  if (MODE !== 'remote' || !els.accountBtn) return;
  els.accountBtn.hidden = false;
  if (!getToken()) {
    renderAccountUI();
    return;
  }
  const me = await checkSession().catch(() => null);
  if (!me) {
    setToken('');
    renderAccountUI();
    return;
  }
  accountState.name = me;
  renderAccountUI();
  try {
    const remote = await fetchAccountData();
    let localMark = 0;
    try {
      localMark = JSON.parse(localStorage.getItem(ACCOUNT_SYNC_KEY) || '{}').updatedAt || 0;
    } catch { /* 忽略 */ }
    if (remote?.payload && remote.updatedAt > localMark) {
      applyAccountPayload(remote.payload);
      markAccountSynced(remote.updatedAt);
      refreshAll();
      renderPresetOptions();
      refreshTools();
    } else if (!remote?.payload) {
      await syncAccountNow();
    }
  } catch {
    /* 同步失败不阻断使用，本机数据照常 */
  }
}

function applyLinkPrompt() {
  if (linkPromptHandled || !linkPrompt) return;
  linkPromptHandled = true;
  // 不覆盖用户已经输入的内容，也不对既有输入自动发送。
  if (els.input.value.trim()) return;
  els.input.value = linkPrompt;
  autoGrow();
  if (params.get('send') === '1') {
    // 等主界面完成显示后再走与手动发送相同的表单逻辑。
    setTimeout(() => {
      if (!els.main.hidden && els.input.value.trim() === linkPrompt.trim()) {
        els.composer.requestSubmit();
      }
    }, 0);
  }
}

els.accountBtn?.addEventListener('click', () => {
  renderAccountUI();
  if (els.accountError) els.accountError.textContent = '';
  els.accountDialog?.showModal();
});
els.accountClose?.addEventListener('click', () => els.accountDialog?.close());
els.accountClose2?.addEventListener('click', () => els.accountDialog?.close());

async function accountAuth(kind) {
  const name = els.accountName.value.trim();
  const password = els.accountPassword.value;
  els.accountError.textContent = '';
  els.accountLoginBtn.disabled = true;
  els.accountRegisterBtn.disabled = true;
  try {
    const data = kind === 'register' ? await accountRegister(name, password) : await accountLogin(name, password);
    els.accountPassword.value = '';
    await afterAccountAuth(data, { isRegister: kind === 'register' });
    showStatus(kind === 'register' ? `账号「${accountState.name}」注册成功并已登录。` : `已登录账号「${accountState.name}」。`);
  } catch (err) {
    els.accountError.textContent = err.message || (kind === 'register' ? '注册失败' : '登录失败');
  } finally {
    els.accountLoginBtn.disabled = false;
    els.accountRegisterBtn.disabled = false;
  }
}
els.accountLoginBtn?.addEventListener('click', () => void accountAuth('login'));
els.accountRegisterBtn?.addEventListener('click', () => void accountAuth('register'));
els.accountSyncNowBtn?.addEventListener('click', () => { void syncAccountNow(); });
els.accountLogoutBtn?.addEventListener('click', () => {
  // 退出登录：先把最新数据推上去，再清令牌；本机保留一份访客副本，不删对话
  void syncAccountNow().finally(() => {
    setToken('');
    accountState.name = null;
    try {
      localStorage.removeItem(ACCOUNT_NAME_KEY);
    } catch { /* 存储不可用 */ }
    renderAccountUI();
    els.accountDialog?.close();
    showStatus('已退出账号登录。当前浏览器里保留着刚才的对话（访客状态）；登录其他账号会恢复那个账号的数据。');
  });
});
window.addEventListener('beforeunload', () => {
  if (accountLoggedIn()) void syncAccountNow();
});
// 周期同步只对服务端模式有意义（静态版 MODE!=='remote' 时 accountLoggedIn 恒为 false）
if (MODE === 'remote') {
  setInterval(() => {
    if (accountLoggedIn() && !document.hidden) void syncAccountNow();
  }, 30000);
}

// ---------- 网站端远程接入：API Key 管理与网页链接生成 ----------
let apiKeyEnvironmentCount = 0;
let apiKeyManagerEpoch = 0;
let apiKeyNewId = '';
let apiPlaygroundController = null;
let apiPlaygroundRunId = 0;
let apiBridgeModels = [];

/** 本机大模型（llama.cpp 原生引擎）状态：只有带服务端的部署才有，静态版恒为 null。 */
let nativeStatus = null;
function nativeModels() {
  return Array.isArray(nativeStatus?.models) ? nativeStatus.models : [];
}
function isNativeModelId(id) {
  return nativeModels().some((m) => m.id === id);
}
/** 聊天可选的全部模型（桥接网页模型 + 本机大模型）。 */
function chatModelIds() {
  return [...apiBridgeModels.map((m) => m.id), ...nativeModels().map((m) => m.id)];
}

function apiDisplayModel() {
  // 本机大模型（llama.cpp）不依赖浏览器桥接：即使没开 LOCAL_BROWSER_BRIDGE，
  // 只要下载了本机大模型，调试台/示例也应该用它，而不是显示“服务端尚未配置模型”。
  const hasNative = nativeModels().length > 0;
  if (state.appConfig.localBrowserBridge || hasNative) {
    // 调试台/示例使用下拉框中选中的模型（每次请求可不同）；未选或已失效时用第一个
    const selected = els.apiPlaygroundModel?.value || '';
    return chatModelIds().includes(selected)
      ? selected
      : apiBridgeModels[0]?.id || nativeModels()[0]?.id || state.appConfig.model || '';
  }
  return state.appConfig.model || '';
}

/** 填充调试台的模型下拉框：桥接模式列出网页可用的全部模型；服务端模式只有服务器配置的那一个。 */
function renderPlaygroundModelOptions() {
  const select = els.apiPlaygroundModel;
  if (!select) return;
  const prev = select.value;
  const options = state.appConfig.localBrowserBridge || nativeModels().length
    ? [
        ...apiBridgeModels,
        ...nativeModels().map((m) => ({ id: m.id, label: `${m.name}（本机大模型·${m.paramsB}B）` })),
      ]
    : state.appConfig.model
      ? [{ id: state.appConfig.model, label: state.appConfig.model }]
      : [];
  select.replaceChildren(
    ...options.map((m) => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.label || m.id;
      return opt;
    }),
  );
  if (options.some((m) => m.id === prev)) select.value = prev;
  select.disabled = options.length <= 1;
  select.title = options.length > 1 ? '每次请求都可以选不同的模型' : '当前只有一个可用模型';
}

function accessBaseUrl() {
  return state.appConfig.publicBaseUrl || window.location.origin;
}

function renderApiAccessInfo() {
  const base = accessBaseUrl();
  const apiBase = new URL(state.appConfig.apiBaseUrl || '/v1', `${base}/`).href.replace(/\/$/, '');
  if (els.apiAccessBase) els.apiAccessBase.textContent = apiBase;
  els.apiAccessEndpoint.textContent = `${apiBase}/chat/completions`;
  if (state.appConfig.localBrowserBridge || nativeModels().length) {
    els.apiAccessModel.textContent = apiDisplayModel()
      ? `${apiDisplayModel()}（共 ${apiBridgeModels.length} 个网页模型${nativeModels().length ? ` + ${nativeModels().length} 个本机大模型` : ''}；API 以每次请求里的 model 为准，完整列表请求 GET /v1/models；下方调试台可切换这里显示的模型）`
      : '等待本机网页连接模型…';
  } else {
    els.apiAccessModel.textContent = state.appConfig.model || '服务端尚未配置模型';
  }
  renderPlaygroundModelOptions();
  if (els.localApiInfo) {
    // 本机版：把 API 地址也显示在“管理员面板”里（不用打开 API 接入弹窗就能看到并复制）
    els.localApiInfo.hidden = !bridgePanelMode();
    if (bridgePanelMode()) els.localApiEndpoint.textContent = apiBase;
  }
  updateApiCurlExample();
  const enabled = !!state.appConfig.apiKeyManagerEnabled;
  if (!enabled) {
    els.apiKeyManagerHint.textContent = state.appConfig.localBrowserBridge
      ? '本机管理员面板未启用：请在服务器 .env 设置 API_ADMIN_PASSWORD（至少 16 位）并重启；网站登录口令不能代替管理口令。'
      : state.appConfig.apiKeyAuth
        ? '外部 API Key 已可调用；网站内的调试台、参数控制台和 Key 管理需要站点所有者在 .env 设置 API_ADMIN_PASSWORD（至少 16 位）并重启。'
        : '管理员控制台未启用。请在服务器 .env 设置 API_ADMIN_PASSWORD（至少 16 位）并重启；普通网站口令不能代替它。';
  } else if (state.appConfig.localBrowserBridge) {
    els.apiKeyManagerHint.textContent = '本机管理员面板已启用。管理口令是首次启动时黑色窗口显示的“管理口令”，也保存在解压文件夹的 .local-api.json 中（字段 API_ADMIN_PASSWORD；找不到时请先在文件管理器中显示隐藏文件）。它与网站登录口令不同，请勿分享该文件。验证后可打开 API 调试台、参数控制台和 API Key 管理。';
  } else {
    els.apiKeyManagerHint.textContent = '此验证会解锁网页 API 调试台、服务端参数控制台和全站 API Key 管理；普通网站访问口令不能代替 API_ADMIN_PASSWORD。';
  }
  const adminVerified = enabled && !!state.apiKeyAdminToken;
  if (MODE === 'remote') {
    els.roleBadge.textContent = adminVerified ? '管理员' : '访客';
    els.roleBadge.classList.toggle('is-admin', adminVerified);
  }
  els.apiAdminLoginSection.hidden = adminVerified;
  els.apiKeyAdminLoginForm.hidden = !enabled;
  els.apiPlaygroundSection.hidden = !adminVerified;
  els.apiAdminSettingsSection.hidden = !adminVerified;
  els.apiKeyManagerSection.hidden = !adminVerified;
  els.apiKeyManagerPanel.hidden = !adminVerified;
}

let bridgeStatusRetries = 0;

async function refreshBridgeStatus() {
  if (!state.appConfig.localBrowserBridge || !els.apiSelfApplySection) return;
  els.apiSelfApplySection.hidden = false;
  try {
    const res = await fetch('/api/local-model-bridge/status', { cache: 'no-store' });
    const data = await res.json().catch(() => ({}));
    apiBridgeModels = Array.isArray(data.models) ? data.models.filter((m) => m && typeof m.id === 'string') : [];
    if (data.connected) {
      bridgeStatusRetries = 0;
      let text = `本机模型已连接：可用模型 ${apiBridgeModels.length} 个（保持本网页打开即可供 API 调用）。`;
      const protocol = Number(data.clientProtocolVersion || 0);
      const latestProtocol = Number(data.latestProtocolVersion || 2);
      if (protocol < latestProtocol) {
        text += ' 注意：当前提供 API 的模型网页未上报诊断信息，可能是旧标签页或旧版网页；生成失败时只能显示通用原因。请关闭其他服务网页标签，只保留本页并按 Ctrl+F5 刷新。';
      }
      if (data.lastError?.message) {
        text += ` 最近一次 API 生成失败：${data.lastError.message}`;
      }
      els.apiBridgeStatus.textContent = text;
    } else {
      els.apiBridgeStatus.textContent = '本机模型尚未连接：请保持本网页打开，API 会自动重连。';
      // 面板还开着时自动重试几次，免得用户刚打开面板、桥接还没连上就以为用不了。
      if (els.apiAccessDialog.open && bridgeStatusRetries < 5) {
        bridgeStatusRetries += 1;
        setTimeout(() => {
          if (els.apiAccessDialog.open) refreshBridgeStatus();
        }, 2000);
      }
    }
    renderApiAccessInfo();
    renderBridgeModelOptions();
    updateApiPlaygroundRequest();
  } catch {
    els.apiBridgeStatus.textContent = '本机模型状态读取失败。';
  }
}

// ---------- 本机大模型（llama.cpp 原生引擎） ----------

let nativeRefreshTimer = null;
// 下载速度采样：<模型@量化> 上次状态刷新时的已下载字节与时刻，用差值估算速度/剩余时间
const dlSpeedSamples = new Map();
// 浏览器通道最近一次失败原因（按模型 id）：行内小字会被状态刷新重建冲掉，
// 把原因存这里随行渲染，用户（和排障）才看得到浏览器通道到底死在哪一步，
// 而不是永远只剩回退直连的那句报错。
const browserDlErrors = new Map();

/** 拉取本机大模型状态并刷新下拉框与管理面板；静态版没有该接口，自动保持隐藏。 */
async function refreshNativeStatus() {
  if (MODE !== 'remote') return;
  const data = await fetchNativeStatus();
  const wasEnabled = !!nativeStatus;
  nativeStatus = data;
  if (els.apiNativeSection) els.apiNativeSection.hidden = !data;
  // 独立模型下载面板里给一句明确说法，不能打开后一片空白
  if (els.modelDownloadUnavailable) els.modelDownloadUnavailable.hidden = !!data;
  if (data) {
    renderNativePanel();
    renderBridgeModelOptions();
    renderPlaygroundModelOptions();
    // 下载/加载/转换进行中时加快刷新，方便看进度
    const busy = nativeModels().some((m) => (m.downloads || []).some((d) => d.state === 'downloading'))
      || nativeModels().some((m) => m.conversion?.state === 'running')
      || nativeModels().some((m) => m.pythonJob?.state === 'running')
      || nativeStatus.hf?.converterJob?.state === 'running'
      || nativeStatus.hf?.runnerJob?.state === 'running'
      || nativeStatus.running?.state === 'loading';
    if (busy) {
      clearTimeout(nativeRefreshTimer);
      nativeRefreshTimer = setTimeout(refreshNativeStatus, 2000);
    }
  } else if (wasEnabled) {
    renderBridgeModelOptions();
  }
}

function buildNativeModelRow(m, admin) {
  const row = document.createElement('div');
  row.className = 'native-model-row';
  const running = nativeStatus?.running || {};
  const isRunning = running.modelId === m.id && running.state === 'ready';
  const isLoading = running.modelId === m.id && running.state === 'loading';

  const head = document.createElement('div');
  head.className = 'native-model-head';
  const title = document.createElement('strong');
  title.textContent = m.name;
  const meta = document.createElement('span');
  meta.className = 'muted small';
  meta.textContent = `${m.vendor} · ${m.paramsB ? `${m.paramsB}B${m.activeB ? `（激活约 ${m.activeB}B）` : ''} · ` : ''}${m.reasoning ? '推理 · ' : ''}${m.kind === 'hf' ? `HF：${m.repo} · ` : ''}${nativeStateText(m)}`;
  head.append(title, meta);
  row.append(head);

  if (m.note) {
    const note = document.createElement('p');
    note.className = 'muted small';
    note.textContent = m.note;
    row.append(note);
  }

  const dl = (m.downloads || []).find((d) => d.state === 'downloading');
  if (dl) {
    const pct = dl.total ? Math.floor((dl.downloaded / dl.total) * 100) : 0;
    // 大号进度条：填充 + 条内百分比数字，必须一眼能看见下了多少
    const bar = document.createElement('div');
    bar.className = 'native-progress';
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuenow', String(pct));
    const fill = document.createElement('i');
    fill.style.width = `${pct}%`;
    const pctText = document.createElement('b');
    pctText.className = 'native-progress-pct';
    pctText.textContent = `${pct}%`;
    bar.append(fill, pctText);
    // 速度/剩余时间：用相邻两次状态采样估算（面板每 2 秒刷新一次，天然是采样点）
    let speedText = '';
    const sampleKey = `${m.id}@${dl.quant}`;
    const nowMs = Date.now();
    const prev = dlSpeedSamples.get(sampleKey);
    // 只用“新鲜”的采样：超过 15 秒没有增长就说明下载卡住或已切换到浏览器通道，
    // 此时继续显示上一段的“还剩 3 分钟”是在骗用户
    if (prev && nowMs - prev.at >= 400 && nowMs - prev.at < 15000 && dl.downloaded >= prev.downloaded) {
      const speed = (dl.downloaded - prev.downloaded) / ((nowMs - prev.at) / 1000);
      if (speed > 0) {
        const etaSec = dl.total > dl.downloaded ? Math.round((dl.total - dl.downloaded) / speed) : 0;
        const eta = etaSec >= 90 ? `约 ${Math.round(etaSec / 60)} 分钟` : `约 ${etaSec} 秒`;
        speedText = ` · 速度约 ${(speed / 1e6).toFixed(1)} MB/s · 还剩${eta}`;
      }
    }
    dlSpeedSamples.set(sampleKey, { downloaded: dl.downloaded, at: nowMs });
    const label = document.createElement('p');
    label.className = 'native-progress-label';
    label.textContent = `正在下载 ${dl.quant}：${(dl.downloaded / 1e9).toFixed(2)} / ${(dl.total / 1e9).toFixed(2)} GB（${pct}%）${speedText}，断网后重新点下载可续传。`;
    row.append(bar, label);
  }
  const dlError = (m.downloads || []).find((d) => d.state === 'error');
  if (dlError) {
    const err = document.createElement('p');
    err.className = 'error small';
    err.textContent = `下载 ${dlError.quant} 失败：${dlError.error || '未知错误'}（重新点下载可续传）`;
    row.append(err);
  }
  // 浏览器通道的失败原因常驻这一行（不被状态刷新冲掉）：这才是离用户网络最近的一手原因
  const browserErr = browserDlErrors.get(m.id);
  if (browserErr) {
    const bErr = document.createElement('p');
    bErr.className = 'error small';
    bErr.textContent = `浏览器通道上次失败：${browserErr} 已下载的部分都保留，重新点下载会接着下；把这句原因发给客服/开发者最能定位问题。`;
    row.append(bErr);
  }
  if (m.conversion) {
    const conv = document.createElement('p');
    conv.className = m.conversion.state === 'error' ? 'error small' : 'muted small';
    conv.textContent = m.conversion.state === 'running'
      ? `正在转换：${m.conversion.stage || '处理中'}…`
      : m.conversion.state === 'error'
        ? `转换失败：${m.conversion.error || '未知错误'}`
        : '转换完成，已生成 GGUF，可直接加载。';
    row.append(conv);
  }
  if (m.remote && !(m.downloaded || []).length) {
    const remote = document.createElement('p');
    remote.className = 'muted small';
    remote.textContent = '已启用云端使用（Hugging Face 云端推理）：在聊天页模型选择里选它即可，不占用本机内存。';
    row.append(remote);
  }
  if (m.pythonJob) {
    const pj = document.createElement('p');
    pj.className = m.pythonJob.state === 'error' ? 'error small' : 'muted small';
    pj.textContent = m.pythonJob.state === 'running'
      ? `本地运行（Python）：${m.pythonJob.stage || '启动中'}…`
      : m.pythonJob.state === 'error'
        ? `本地运行（Python）失败：${m.pythonJob.error || '未知错误'}`
        : '本地运行（Python）中。';
    row.append(pj);
  }

  const controls = document.createElement('div');
  controls.className = 'api-access-actions';

  const quantSelect = document.createElement('select');
  quantSelect.setAttribute('aria-label', `${m.name} 量化`);
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = (m.downloaded || []).length ? '选择量化（已下载的排在前面）' : '先获取量化列表';
  quantSelect.append(placeholder);
  for (const d of m.downloaded || []) {
    const opt = document.createElement('option');
    opt.value = d.quant;
    opt.textContent = `${d.quant}（${d.typeLabel || 'GGUF'} · 已下载 · ${(d.totalBytes / 1e9).toFixed(1)} GB · ${d.directUse === false ? '需转换后使用' : '可直接使用'}）`;
    quantSelect.append(opt);
  }
  controls.append(quantSelect);

  // 选中一档时在行里单独说明“这是什么、能不能直接用”，不用回下拉里找小字
  let rowQuants = [];
  quantSelect.addEventListener('change', () => {
    const q = rowQuants.find((x) => x.quant === quantSelect.value)
      || (m.downloaded || []).find((x) => x.quant === quantSelect.value);
    if (!q) return;
    const bits = q.bitsPerWeight ? `，每权重约 ${q.bitsPerWeight} 位` : '';
    const size = q.totalBytes ? ` · ${(q.totalBytes / 1e9).toFixed(1)} GB` : (q.sizeGB ? ` · ${q.sizeGB} GB` : '');
    const use = q.directUse === false ? '需要转换后才能用' : '下载后可直接使用';
    // 每层显存估算：权重大小 ÷ 总层数（层数来自 GGUF 文件头，服务端随量化信息下发）
    const layers = Number(q.layers) || 0;
    const bytes = Number(q.totalBytes) || (q.sizeGB ? q.sizeGB * 1e9 : 0);
    let gpuNote = '';
    if (layers > 0 && bytes > 0) {
      const perLayerMB = Math.round((bytes / layers / 1e6) * 10) / 10;
      const want = Math.max(0, Number(els.apiNativeGpuLayers?.value) || 0);
      gpuNote = `共 ${layers} 层，每层约占 ${perLayerMB} MB 显存${want ? `；当前填的卸载 ${want} 层大约需要 ${(perLayerMB * want / 1024).toFixed(2)} GB 显存（另需少量计算缓冲）` : ''}。显存不够时会自动减少层数，不会直接失败。`;
    }
    setQuantStatus(`已选 ${q.quant}：${q.typeLabel || 'GGUF 量化'}${bits}${size} · ${use}。${q.qualityNote || ''}${gpuNote}确认就点“下载所选量化”。`);
  });

  const mkBtn = (text, onClick, { disabled = false } = {}) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'btn ghost small';
    btn.textContent = text;
    btn.disabled = disabled;
    btn.addEventListener('click', onClick);
    return btn;
  };

  // 行内状态：点按钮的反馈就显示在这一行，不让用户去面板别处找红字
  const quantStatus = document.createElement('p');
  quantStatus.className = 'muted small';
  quantStatus.hidden = true;
  const setQuantStatus = (text, isError = false) => {
    quantStatus.textContent = text;
    quantStatus.className = isError ? 'error small' : 'muted small';
    quantStatus.hidden = !text;
  };
  // 导入模型的档位都在本地清单里、已在下拉中，不需要联网取列表
  if (m.kind !== 'import') controls.append(mkBtn('获取量化列表', async (ev) => {
    const btn = ev.currentTarget;
    btn.disabled = true;
    btn.textContent = '正在读取…';
    placeholder.textContent = '正在读取量化列表…';
    setQuantStatus('正在读取量化列表（优先经浏览器，和网页版同一条网络路）…');
    try {
      let quants = null;
      let viaBrowser = false;
      try {
        quants = await fetchQuantsViaBrowser({ id: m.id, repo: m.repo, token: getToken() });
        viaBrowser = true;
      } catch {
        quants = await fetchNativeQuants(m.id, getToken());
      }
      rowQuants = quants || [];
      placeholder.textContent = '选择量化（已下载的排在前面）';
      quantSelect.replaceChildren(placeholder, ...(m.downloaded || []).map((d) => {
        const opt = document.createElement('option');
        opt.value = d.quant;
        opt.textContent = `${d.quant}（${d.typeLabel || 'GGUF'} · 已下载 · ${(d.totalBytes / 1e9).toFixed(1)} GB · ${d.directUse === false ? '需转换后使用' : '可直接使用'}）`;
        return opt;
      }), ...quants.filter((q) => !(m.downloaded || []).some((d) => d.quant === q.quant)).map((q) => {
        const opt = document.createElement('option');
        opt.value = q.quant;
        opt.textContent = `${q.quant}（${q.typeLabel || 'GGUF 量化'}）· ${q.sizeGB} GB · 内存约 ${q.ramGB} GB · ${q.directUse === false ? '需要转换后才能用' : '下载后可直接使用'}`;
        opt.title = `${q.qualityNote || ''}${q.useNote || ''}`;
        return opt;
      }));
      setQuantStatus(`已读取 ${quants.length} 个量化（${viaBrowser ? '经浏览器通道读取' : '经软件直连读取'}）：每档都标了类型、大小和是否可直接使用，在左边下拉里选一档，再点“下载所选量化”。`);
      if (els.apiNativeError) els.apiNativeError.textContent = '';
    } catch (err) {
      placeholder.textContent = (m.downloaded || []).length ? '选择量化（已下载的排在前面）' : '先获取量化列表';
      setQuantStatus(err.message || '读取量化列表失败，请检查网络后重试。', true);
      if (els.apiNativeError) els.apiNativeError.textContent = err.message || '读取量化列表失败';
    } finally {
      btn.disabled = false;
      btn.textContent = '获取量化列表';
    }
  }));

  // 本机导入的模型权重已经在本地，没有任何“下载”动作，别给会注定失败的按钮
  if (m.kind !== 'import') {
    // 浏览器通道活跃下载中：按钮改成“继续下载（断点续传）”而不是一片灰。
    // 用户换页面/刷新后再次点击就走续传，本来就是设计允许的。
    const browserActive = !!dl && dl.via === 'browser' && dl.state === 'downloading';
    controls.append(mkBtn(
      isLoading ? '加载中…'
        : isRunning ? '运行中'
          : browserActive ? '继续下载（断点续传）'
            : dl ? '下载中…（点此查看进度）'
              : '下载所选量化',
      () => {
        void nativeAction(async () => {
          // 与普通网页版同一条网络路：浏览器负责从 Hugging Face 取数（自动走
          // 浏览器/系统代理），软件只落盘。浏览器通道不可用时自动回退软件直连。
          browserDlErrors.delete(m.id);
          try {
            await downloadModelViaBrowser({
              id: m.id,
              repo: m.repo,
              quant: quantSelect.value || undefined,
              adminToken: state.apiKeyAdminToken,
              onStatus: (text) => setQuantStatus(text),
            });
          } catch (browserErr) {
            const reason = browserErr?.message || '未知原因';
            // 原因常驻行内（browserDlErrors），状态刷新也不会冲掉；行内小字同步提示
            browserDlErrors.set(m.id, reason);
            // 权重文件本身拉不到（404/403/401：仓库改名、删档、门控仓库没填 Token）时，
            // 再走软件直连同样注定失败，只会让用户看到第二句更含糊的报错。
            const fatal = /HTTP (401|403|404)|需要授权|仓库不存在/.test(reason);
            if (fatal) {
              setQuantStatus(`浏览器通道下载失败（${reason}）。这个问题换用软件直连也不会成功：请确认仓库仍有该文件；门控/私有仓库请先保存 Hugging Face Token 再试。`, true);
              return;
            }
            setQuantStatus(`浏览器通道下载失败（${reason}），已自动改用软件直连下载…`, true);
            await nativeAdminRequest(`/api/native/models/${encodeURIComponent(m.id)}/download`, {
              token: state.apiKeyAdminToken,
              body: { quant: quantSelect.value || undefined },
            });
          }
        });
      },
      // 浏览器通道的“下载中”是可接管的：再点一次就按断点续传（换页面/刷新后尤其
      // 需要能点）；只有软件直连真在后台下载时才置灰，避免两条通道写同一文件。
      { disabled: !admin || isRunning || isLoading || (!!dl && dl.via !== 'browser') },
    ));
  }

  const hasLocalSnapshot = (m.downloaded || []).some((d) => d.quant === 'SNAPSHOT');
  if ((m.kind === 'hf' && !(m.downloaded || []).length
    && (m.formats || []).some((f) => f === 'safetensors' || f === 'pytorch'))
    || (m.kind === 'import' && hasLocalSnapshot)) {
    controls.append(mkBtn(m.conversion?.state === 'running' ? '转换中…' : '转换并使用', () => {
      void nativeAction(async () => {
        try {
          await nativeAdminRequest('/api/native/hf/convert', { token: state.apiKeyAdminToken, body: { id: m.id } });
        } catch (err) {
          if (err.status === 503) {
            await nativeAdminRequest('/api/native/hf/prepare-converter', { token: state.apiKeyAdminToken });
            if (els.apiNativeError) els.apiNativeError.textContent = '正在准备转换工具（只需一次），准备好后会自动开始转换；请稍等片刻再点一次“转换并使用”。';
            return;
          }
          throw err;
        }
      });
    }, { disabled: !admin || m.conversion?.state === 'running' }));
  }

  if ((m.kind === 'hf' && (m.formats || []).some((f) => ['safetensors', 'pytorch', 'onnx', 'mlx', 'coreml'].includes(f)))
    || (m.kind === 'import' && hasLocalSnapshot)) {
    controls.append(mkBtn(m.pythonJob?.state === 'running' ? '本地运行启动中…' : '本地运行（Python）', () => {
      void nativeAction(async () => {
        try {
          await nativeAdminRequest('/api/native/hf/python', { token: state.apiKeyAdminToken, body: { id: m.id } });
        } catch (err) {
          if (err.status === 503) {
            await nativeAdminRequest('/api/native/hf/prepare-runner', { token: state.apiKeyAdminToken });
            if (els.apiNativeError) els.apiNativeError.textContent = '正在准备本地运行时（只需一次），准备好后再点一次“本地运行（Python）”。';
            return;
          }
          throw err;
        }
      });
    }, { disabled: !admin || m.pythonJob?.state === 'running' }));
  }

  if (m.kind === 'hf' && !m.remote) {
    controls.append(mkBtn('云端使用', () => {
      void nativeAction(async () => {
        const out = await nativeAdminRequest('/api/native/hf/use', { token: state.apiKeyAdminToken, body: { repo: m.repo, mode: 'cloud' } });
        if (out.action === 'needs-token' && els.apiNativeError) {
          els.apiNativeError.textContent = '云端使用需要先在上方保存 HF Token；不想用云端可继续用本地下载/运行。';
          els.apiHfTokenInput?.focus();
        }
      });
    }, { disabled: !admin }));
  }

  controls.append(mkBtn('加载', () => {
    void nativeAction(async () => {
      await nativeAdminRequest('/api/native/load', {
        token: state.apiKeyAdminToken,
        body: {
          id: m.id,
          quant: quantSelect.value || undefined,
          contextTokens: Number(els.apiNativeCtx?.value) || 8192,
          gpuLayers: Math.max(0, Number(els.apiNativeGpuLayers?.value) || 0),
        },
      });
    });
  }, { disabled: !admin || isRunning || isLoading || !(m.downloaded || []).length }));

  // 只删选中的那一档：几十 GB 的量化常常下一档就够用，删全部会把另一档也一起清掉
  const pickedQuant = () => quantSelect.value
    || (m.downloaded || [])[0]?.quant
    || (m.downloaded || []).find((d) => d.quant !== 'SNAPSHOT')?.quant;
  controls.append(mkBtn('删除所选量化', () => {
    const q = pickedQuant();
    if (!q) return;
    const info = (m.downloaded || []).find((d) => d.quant === q);
    const size = info?.totalBytes ? `（约 ${(info.totalBytes / 1e9).toFixed(1)} GB）` : '';
    if (!window.confirm(`只删除 ${m.name} 的 ${q} 权重${size}？其它量化保留，之后需要时可重新下载。`)) return;
    void nativeAction(() => nativeAdminRequest(`/api/native/models/${encodeURIComponent(m.id)}`, {
      token: state.apiKeyAdminToken,
      method: 'DELETE',
      body: { quant: q },
    }));
  }, { disabled: !admin || isRunning || !(m.downloaded || []).length || (m.downloaded || []).length <= 1 }));

  controls.append(mkBtn('删除权重', () => {
    if (!window.confirm(`删除 ${m.name} 已下载的全部权重？之后需要重新下载。`)) return;
    void nativeAction(() => nativeAdminRequest(`/api/native/models/${encodeURIComponent(m.id)}`, {
      token: state.apiKeyAdminToken,
      method: 'DELETE',
      body: {},
    }));
  }, { disabled: !admin || isRunning || !(m.downloaded || []).length }));

  row.append(controls, quantStatus);
  return row;
}

/**
 * 在长动作（下载/转换准备等）执行期间每 2 秒刷新一次本机状态。
 *
 * 为什么需要它：refreshNativeStatus 的自动轮询要先“被观察到忙碌”才自我续期，
 * 而浏览器通道下载恰恰是在点击里同步跑的——动作没结束就没人刷状态，
 * 靠服务端状态渲染的那条进度条（百分比条 + 已下/总量）整个下载过程都不会出现，
 * 用户只剩一行攒够几 MB 才跳一次的文字，看着像“没有下载进度”。
 * 动作期间主动轮询后，进度条从下载第一秒就接管显示；文字行作为补充。
 */
async function withNativePolling(fn) {
  const timer = setInterval(() => { void refreshNativeStatus(); }, 2000);
  try {
    return await fn();
  } finally {
    clearInterval(timer);
  }
}

/** 执行管理员动作：失败显示原因（401 时按过期处理并锁定），之后刷新状态。 */
async function nativeAction(fn) {
  if (els.apiNativeError) els.apiNativeError.textContent = '';
  try {
    await withNativePolling(fn);
  } catch (err) {
    if (err.status === 401 && state.apiKeyAdminToken) {
      lockApiKeyManager();
      if (els.apiNativeError) els.apiNativeError.textContent = '管理员验证已过期，请重新验证后再操作大模型。';
    } else if (els.apiNativeError) {
      els.apiNativeError.textContent = err.message || '操作失败';
    }
  }
  await refreshNativeStatus();
}

function renderNativePanel() {
  if (!els.apiNativeList || !nativeStatus) return;
  const admin = !!state.apiKeyAdminToken;
  // 未验证时在下载面板顶部给明确入口，别让用户对着置灰的按钮猜
  if (els.apiNativeAdminGate) els.apiNativeAdminGate.hidden = admin;
  const running = nativeStatus.running || {};
  if (els.apiNativeBinaryStatus) {
    els.apiNativeBinaryStatus.textContent = nativeStatus.binaryReady
      ? `llama.cpp 运行库已就绪（${{ env: '环境变量指定', bundled: '随包附带', path: '系统已安装', downloaded: '已下载到本机' }[nativeStatus.binarySource] || '可用'}）。`
      : nativeStatus.binaryJob?.state === 'downloading'
        ? '正在下载 llama.cpp 运行库…'
        : 'llama.cpp 运行库尚未准备：点下方按钮从官方 GitHub 下载（约几十 MB），或自行安装 llama.cpp 后设置环境变量 LLAMA_SERVER_BIN。';
  }
  if (els.apiNativeBinaryBtn) {
    els.apiNativeBinaryBtn.disabled = !admin || nativeStatus.binaryReady;
    els.apiNativeBinaryBtn.textContent = nativeStatus.binaryReady ? '运行库已就绪' : '准备 llama.cpp 运行库';
  }
  if (els.apiNativeUnloadBtn) {
    els.apiNativeUnloadBtn.disabled = !admin || !running.modelId;
    els.apiNativeUnloadBtn.textContent = running.modelId
      ? `卸载当前模型（${running.modelId}${running.state === 'loading' ? '·加载中' : ''}）`
      : '卸载当前模型';
  }
  if (els.apiNativeHint) {
    const base = admin
      ? '下载/加载/删除会占用本机磁盘与内存，超大模型请先确认内存（内存+显存合计）足够。同一时间只运行一个大模型，加载新模型会自动卸载当前模型。'
      : '查看状态不需要验证；下载、加载、删除权重需要先在下方“管理员控制面板”验证口令（验证一次即可，之后不用重复验证，点“锁定管理面板”才退出）。';
    const gpuBits = [];
    if (running.state === 'ready' && running.gpuLayers !== null && running.gpuLayers !== undefined) {
      gpuBits.push(`当前模型实际卸载到 GPU 的层数：${running.gpuLayers} 层`);
    }
    // 最近一次生成因上下文不够而省略了历史：明确告诉用户“模型不是失忆，是窗口不够”
    if (nativeStatus.contextTrim) {
      const t = nativeStatus.contextTrim;
      gpuBits.push(`上一次对话太长，已省略最早的 ${t.droppedMessages} 条历史（约 ${t.estimatedTokens} tokens / 窗口 ${t.contextTokens}）以留出回答空间；需要模型记住更多内容请调大上下文或新建会话`);
    }
    if (nativeStatus.loadNote) {
      const n = nativeStatus.loadNote;
      gpuBits.push(typeof n === 'string'
        ? n
        : n.reason && Number(n.used) === Number(n.requested)
          // 只是上下文被模型上限截断（层数没动）：别写成“自动回退”，会被误读成显存不足
          ? `${n.reason}（实际上下文 ${n.contextTokens || n.used}）`
          : `加载时已自动回退（GPU 层数 请求 ${n.requested} → 实际 ${n.used}${n.contextTokens ? `，上下文 ${n.contextTokens}` : ''}）：${n.reason || '疑似显存/内存不足'}`);
    }
    els.apiNativeHint.textContent = gpuBits.length ? `${base} ${gpuBits.join('；')}。` : base;
  }
  // llama-server 自己的输出（日志尾部）：加载崩溃/失败时真正的死因写在这里，而不是聊天报错里
  if (els.apiNativeLogWrap && els.apiNativeLogTail) {
    const lines = Array.isArray(running.logTail) ? running.logTail : [];
    const show = !!running.error || lines.length > 0;
    els.apiNativeLogWrap.hidden = !show;
    if (show) {
      if (running.state === 'error') els.apiNativeLogWrap.open = true;
      els.apiNativeLogTail.textContent =
        (running.error ? `错误：${running.error}\n` : '') + (lines.length ? lines.join('\n') : '（暂无日志输出）');
    }
  }
  if (els.apiHfTokenStatus) {
    els.apiHfTokenStatus.textContent = nativeStatus.hf?.hasToken
      ? 'Token 已保存（云端模型可用；清空输入后保存可清除）'
      : '未保存 Token（仅云端模型需要，本机模型不用）';
  }
  if (els.apiHfConvertToolStatus) {
    const job = nativeStatus.hf?.converterJob;
    els.apiHfConvertToolStatus.textContent = job?.state === 'running'
      ? '正在准备转换工具（下载 llama.cpp 与 Python 依赖，可能需要几分钟）…'
      : job?.state === 'error'
        ? `准备失败：${job.error || '未知错误'}`
        : nativeStatus.hf?.converter?.ready
          ? '转换工具已就绪（Safetensors 等格式可一键转换）'
          : '转换工具未准备（只有转换 Transformers 格式时才需要）';
  }
  if (els.apiHfConvertToolBtn) els.apiHfConvertToolBtn.disabled = !admin || nativeStatus.hf?.converter?.ready;
  if (els.apiHfRunnerStatus) {
    const rjob = nativeStatus.hf?.runnerJob;
    els.apiHfRunnerStatus.textContent = rjob?.state === 'running'
      ? '正在准备本地运行时（下载 Python 依赖，可能需要几分钟）…'
      : rjob?.state === 'error'
        ? `准备失败：${rjob.error || '未知错误'}`
        : nativeStatus.hf?.runner?.ready
          ? '本地运行时已就绪（ONNX/AWQ 等特殊格式可本地运行）'
          : '本地运行时未准备（只有特殊格式本地运行时才需要）';
  }
  if (els.apiHfRunnerBtn) els.apiHfRunnerBtn.disabled = !admin || nativeStatus.hf?.runner?.ready;
  els.apiNativeList.replaceChildren(...nativeModels().map((m) => buildNativeModelRow(m, admin)));
}

/**
 * 本机版网页聊天自己选用的模型（与 API 调用方各自选的模型互不影响：
 * API 请求带什么模型就用什么模型，不会改掉这里的选择；这里切换也只影响网页接下来的发送）。
 * 选择记在本浏览器中，下次打开仍沿用。
 */
const LOCAL_CHAT_MODEL_KEY = 'oj-ai.localChatModel.v1';
let localChatModelId = '';

function loadLocalChatModel() {
  try {
    return localStorage.getItem(LOCAL_CHAT_MODEL_KEY) || '';
  } catch {
    return '';
  }
}

function saveLocalChatModel(id) {
  try {
    // 空值表示清除保存的选择（例如当前模型已被删除），必须删掉旧值，
    // 否则已删除的模型 id 会一直留在浏览器里，下次打开又被当成保存的选择。
    if (id) localStorage.setItem(LOCAL_CHAT_MODEL_KEY, id);
    else localStorage.removeItem(LOCAL_CHAT_MODEL_KEY);
  } catch {
    /* 存储不可用时只在本次页面内生效 */
  }
}

// ---------- Hugging Face 桥接（搜索 → 拉取 → 点击使用） ----------

let hfSearchResults = [];

function hfRouteGuess(formats = []) {
  if (formats.includes('gguf')) return { key: 'local', label: '可直接运行（GGUF）' };
  if (formats.includes('safetensors') || formats.includes('pytorch')) return { key: 'convert', label: '可转换后运行' };
  if (formats.length) return { key: 'cloud', label: '云端可用' };
  return { key: 'cloud', label: '点一键使用自动判断' };
}

async function runHfSearch() {
  if (!els.apiHfResults) return;
  const q = els.apiHfSearchInput?.value.trim() || '';
  if (els.apiHfSearchStatus) els.apiHfSearchStatus.textContent = '正在搜索 Hugging Face（经浏览器，和网页版同一条网络路）…';
  try {
    let data;
    let viaBrowser = true;
    try {
      data = await searchHfViaBrowser(q, getToken());
    } catch {
      viaBrowser = false;
      data = await hfSearch(q, getToken());
    }
    hfSearchResults = data.results || [];
    if (els.apiHfSearchStatus) {
      els.apiHfSearchStatus.textContent = hfSearchResults.length
        ? `找到 ${hfSearchResults.length} 个模型（${viaBrowser ? '经浏览器通道' : '经软件直连'}搜索），点“一键使用”即可（系统自动按格式处理）。`
        : '没有找到相关模型，换个关键词试试（如 Qwen、Llama、Mistral）。';
    }
    renderHfResults();
  } catch (err) {
    if (els.apiHfSearchStatus) els.apiHfSearchStatus.textContent = `搜索失败：${err.message}`;
  }
}

function renderHfResults() {
  if (!els.apiHfResults) return;
  const filter = els.apiHfFilter?.value || 'all';
  const admin = !!state.apiKeyAdminToken;
  const rows = hfSearchResults
    .filter((r) => filter === 'all' || hfRouteGuess(r.formats || []).key === filter)
    .map((r) => {
      const row = document.createElement('div');
      row.className = 'native-model-row';
      const head = document.createElement('div');
      head.className = 'native-model-head';
      const title = document.createElement('strong');
      title.textContent = r.repo;
      const meta = document.createElement('span');
      meta.className = 'muted small';
      const guess = hfRouteGuess(r.formats || []);
      meta.textContent = `↓${(r.downloads || 0).toLocaleString()} · ♥${r.likes || 0} · ${guess.label}${(r.formats || []).length ? ` · 格式：${r.formats.join('/')}` : ''}`;
      head.append(title, meta);
      row.append(head);
      const detail = document.createElement('p');
      detail.className = 'muted small';
      detail.hidden = true;
      row.append(detail);
      const controls = document.createElement('div');
      controls.className = 'api-access-actions';
      const mkBtn = (text, onClick, disabled = false) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn ghost small';
        btn.textContent = text;
        btn.disabled = disabled;
        btn.addEventListener('click', onClick);
        return btn;
      };
      controls.append(mkBtn('一键使用', () => void hfUseRepo(r.repo, 'auto'), !admin));
      controls.append(mkBtn('下载到本地', () => void hfUseRepo(r.repo, 'local'), !admin));
      controls.append(mkBtn('云端使用', () => void hfUseRepo(r.repo, 'cloud'), !admin));
      controls.append(mkBtn('查看详情', async () => {
        detail.hidden = false;
        detail.textContent = '正在读取仓库文件与格式（经浏览器读取）…';
        try {
          let info;
          try {
            info = await fetchModelInfoViaBrowser(r.repo, getToken());
          } catch {
            info = await hfModelInfo(r.repo, getToken());
          }
          const quantText = (info.quants || []).length
            ? ` 可选量化：${info.quants.map((x) => `${x.quant}（${x.typeLabel}，${x.sizeGB} GB，${x.directUse ? '下载后可直接使用' : '需要转换'}）`).join('；')}。`
            : '';
          detail.textContent = `${info.route.label}：${info.route.detail} 格式：${info.formats.join('/') || '未知'}${info.variants?.length ? `（${info.variants.join('/')}）` : ''}；仓库共 ${info.totalGB} GB。${quantText}文件示例：${info.files.slice(0, 5).map((f) => f.path).join('、')}`;
        } catch (err) {
          detail.textContent = `读取失败：${err.message}`;
        }
      }));
      if (!admin) {
        const tip = document.createElement('span');
        tip.className = 'muted small';
        tip.textContent = '下载/使用需先在下方验证管理员口令（只验证一次）';
        controls.append(tip);
      }
      row.append(controls);
      return row;
    });
  els.apiHfResults.replaceChildren(...rows);
}

/** 使用 HF 模型：mode = auto（一键）| local（下载到本地）| cloud（云端，不下载）。 */
async function hfUseRepo(repo, mode = 'auto') {
  if (els.apiHfSearchStatus) els.apiHfSearchStatus.textContent = `正在处理 ${repo}（优先经浏览器，和网页版同一条网络路）…`;
  // 下载可能持续很久：期间保持状态轮询，下方模型列表里会长出实时进度条
  await withNativePolling(() => hfUseRepoInner(repo, mode));
}

async function hfUseRepoInner(repo, mode = 'auto') {
  try {
    let out;
    let downloadState = 'none';
    let downloadError = '';
    try {
      const via = await useRepoViaBrowser({
        repo,
        mode,
        adminToken: state.apiKeyAdminToken,
        onStatus: (text) => { if (els.apiHfSearchStatus) els.apiHfSearchStatus.textContent = text; },
      });
      out = via.out;
      downloadState = via.downloadState;
      downloadError = via.downloadError || '';
    } catch {
      // 浏览器通道不可用才回退软件直连（直连不通时错误会如实显示在下方）
      out = await nativeAdminRequest('/api/native/hf/use', { token: state.apiKeyAdminToken, body: { repo, mode } });
    }
    if (out.action === 'download') {
      if (downloadState === 'browser') {
        els.apiHfSearchStatus.textContent = `${repo}（${out.quant || '推荐量化'}）已通过浏览器通道下载完成，下方模型列表里点“加载”即可使用；之后在聊天页模型选择里也能选它。`;
      } else if (downloadState === 'fallback') {
        els.apiHfSearchStatus.textContent = `浏览器通道不可用（${downloadError || '未知原因'}），已改用软件直连下载 ${repo}（${out.quant || '推荐量化'}），进度在下方模型列表里看。`;
      } else {
        els.apiHfSearchStatus.textContent = `已开始下载 ${repo}（${out.quant || '推荐量化'}），进度在下方模型列表里看；下载完成并加载后，就能在聊天页模型选择里用它。`;
      }
    } else if (out.action === 'convert') {
      if (out.converterReady) {
        await nativeAdminRequest('/api/native/hf/convert', { token: state.apiKeyAdminToken, body: { id: out.id } });
        els.apiHfSearchStatus.textContent = `已开始把 ${repo} 转成 GGUF，转换进度在下方模型列表里看，完成后即可加载使用。`;
      } else {
        els.apiHfSearchStatus.textContent = `${repo} 需要先把权重转成 GGUF：请点上方“一键准备格式转换工具”（只需一次），准备好后再点一次。`;
      }
    } else if (out.action === 'python') {
      if (out.runnerReady) {
        await nativeAdminRequest('/api/native/hf/python', { token: state.apiKeyAdminToken, body: { id: out.id } });
        els.apiHfSearchStatus.textContent = `已开始下载 ${repo} 并在本机用 Python 运行时加载，进度在下方模型列表里看；加载完成后在聊天页模型选择里选它即可（CPU 上会慢一些）。`;
      } else {
        await nativeAdminRequest('/api/native/hf/prepare-runner', { token: state.apiKeyAdminToken });
        els.apiHfSearchStatus.textContent = `正在准备本地运行时（只需一次，约数百 MB），准备好后再点一次“下载到本地”即可。`;
      }
    } else if (out.action === 'remote') {
      els.apiHfSearchStatus.textContent = `${repo} 已启用云端使用：在聊天页模型选择里选它就能对话（走 Hugging Face 云端推理，不占用本机）。之后想下载到本地，点“下载到本地”即可切换。`;
    } else if (out.action === 'needs-token') {
      els.apiHfSearchStatus.textContent = `云端使用需要 HF Token：先在上方填入并保存，再点一次“云端使用”。不想用云端就点“下载到本地”。`;
      els.apiHfTokenInput?.focus();
    } else if (out.action === 'browser') {
      els.apiHfSearchStatus.textContent = `${repo} 是浏览器 MLC 格式：在网页版模型路线里使用（聊天页的浏览器模型）；软件版也可以切换为云端使用。`;
    } else {
      els.apiHfSearchStatus.textContent = `${repo}：${out.route?.detail || '暂未识别的格式，无法使用。'}`;
    }
  } catch (err) {
    if (err.status === 401 && state.apiKeyAdminToken) lockApiKeyManager();
    if (els.apiHfSearchStatus) els.apiHfSearchStatus.textContent = `使用失败：${err.message}`;
  }
  await refreshNativeStatus();
}

/** 本机大模型在下拉框/列表中显示的状态：运行中 > 下载中 > 已下载 > 未下载。 */
function nativeStateText(m) {
  if (nativeStatus?.running?.modelId === m.id && nativeStatus.running.state === 'ready') return '运行中';
  if (nativeStatus?.running?.modelId === m.id && nativeStatus.running.state === 'loading') return '加载中';
  const dl = (m.downloads || []).find((d) => d.state === 'downloading');
  if (dl) return `下载中 ${dl.total ? Math.floor((dl.downloaded / dl.total) * 100) : 0}%`;
  if ((m.downloaded || []).length) return '已下载';
  if (m.conversion?.state === 'running') return '转换中';
  if (m.remote) return '云端可用';
  return '未下载';
}

/** 把桥接可用模型填入聊天页的模型选择栏（侧边栏 + 窄屏各一个，始终同步）；本机大模型单独分组。 */
function renderBridgeModelOptions() {
  // 只要求“本机版网页（remote）”：未开启网页模型桥接（LOCAL_BROWSER_BRIDGE≠1）但启用了
  // 本机大模型的部署，以前这里直接 return，聊天页的模型下拉框整个不出现——下载好的
  // GGUF 大模型在界面上根本选不到，只能通过 /v1 API 调用。现在只要有本机大模型就渲染。
  if (MODE !== 'remote' || (!state.appConfig.localBrowserBridge && !nativeModels().length)) return;
  const ids = chatModelIds();
  if (!ids.length) {
    els.modelSelect.hidden = true;
    els.modelSelectCompact.hidden = true;
    els.modelLabel.hidden = false;
    els.modelLabel.textContent = state.appConfig.localBrowserBridge
      ? '模型：等待本机网页连接…'
      : '模型：尚未下载本机大模型（在“模型下载”里下载后可选）';
    return;
  }
  if (!ids.includes(localChatModelId)) {
    const saved = loadLocalChatModel();
    localChatModelId = ids.includes(saved) ? saved : ids[0];
    // 保存的选择已失效（如模型被删除）时把纠正后的选择写回，避免旧值一直残留
    if (saved !== localChatModelId) saveLocalChatModel(localChatModelId);
  }
  for (const select of [els.modelSelect, els.modelSelectCompact]) {
    if (!select) continue;
    const compact = select === els.modelSelectCompact;
    const children = apiBridgeModels.map((m) => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.dataset.label = m.label || m.id;
      opt.textContent = compact ? shortModelLabel(m.label || m.id) : m.label || m.id;
      return opt;
    });
    const natives = nativeModels();
    if (natives.length) {
      const group = document.createElement('optgroup');
      group.label = '本机大模型（llama.cpp 引擎）';
      for (const m of natives) {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.dataset.label = `${m.name}（${m.paramsB ? `${m.paramsB}B·` : ''}${m.kind === 'hf' ? 'Hugging Face' : '本机大模型'}·${nativeStateText(m)}）`;
        opt.textContent = compact ? shortModelLabel(opt.dataset.label) : opt.dataset.label;
        opt.title = m.note || '';
        group.append(opt);
      }
      children.push(group);
    }
    select.replaceChildren(...children);
    select.value = localChatModelId;
    select.hidden = false;
    select.title = '选择网页聊天使用的模型（与 API 调用各自选的模型互不影响）';
  }
  els.modelLabel.hidden = true;
}

/** 网页聊天切换模型：只记下选择，下一次发送时按这个模型请求；不影响 API 的模型选择。 */
function onBridgeModelChange(id) {
  const bridged = apiBridgeModels.find((m) => m.id === id);
  const native = nativeModels().find((m) => m.id === id);
  if (!bridged && !native) return;
  localChatModelId = id;
  saveLocalChatModel(id);
  if (els.modelSelect) els.modelSelect.value = id;
  if (els.modelSelectCompact) els.modelSelectCompact.value = id;
  if (native) {
    const stateText = nativeStateText(native);
    showStatus(
      stateText === '未下载'
        ? `网页聊天已切换到 ${native.name}（本机大模型），但权重尚未下载：请打开“API 接入 → 本机大模型”下载并加载后再发送。`
        : `网页聊天已切换到 ${native.name}（本机大模型·${stateText}）。下次发送时使用这个模型；API 调用选用的模型不受影响。`,
    );
    if (stateText !== '未下载' && !state.appConfig.localBrowserBridge && state.apiKeyAdminToken) {
      // 没有网页模型桥接时，本机大模型只能由服务端原生引擎跑。权重下载好但还没加载时
      // 聊天会等第一次请求才自动加载（几分钟无输出），这里提前在后台加载，用户发送前就能看到“加载中”。
      void nativeAdminRequest('/api/native/load', {
        token: state.apiKeyAdminToken,
        body: {
          id: native.id,
          quant: (native.downloaded || []).find((d) => d.quant !== 'SNAPSHOT')?.quant,
          contextTokens: Number(els.apiNativeCtx?.value) || 8192,
          gpuLayers: Math.max(0, Number(els.apiNativeGpuLayers?.value) || 0),
        },
      }).catch((err) => {
        console.warn('本机大模型预加载失败（发送时会再试）：', err?.message || err);
      });
    }
    return;
  }
  const label = bridged?.label || id;
  showStatus(`网页聊天已切换到 ${label}。下次发送时使用这个模型；API 调用选用的模型不受影响。`);
}

function apiPlaygroundPayload() {
  const messages = [];
  const system = els.apiPlaygroundSystem.value.trim();
  const prompt = els.apiPlaygroundPrompt.value.trim();
  if (system) messages.push({ role: 'system', content: system });
  if (prompt) messages.push({ role: 'user', content: prompt });
  const payload = {
    model: apiDisplayModel() || '<MODEL>',
    messages,
    stream: els.apiPlaygroundStream.checked,
  };
  const temperature = els.apiPlaygroundTemperature.value.trim();
  const maxTokens = els.apiPlaygroundMaxTokens.value.trim();
  if (temperature) payload.temperature = Number(temperature);
  if (maxTokens) payload.max_tokens = Number(maxTokens);
  return payload;
}

function updateApiPlaygroundRequest() {
  if (els.apiPlaygroundRequest) els.apiPlaygroundRequest.textContent = JSON.stringify(apiPlaygroundPayload(), null, 2);
}

function clearApiPlayground() {
  apiPlaygroundRunId += 1;
  apiPlaygroundController?.abort();
  apiPlaygroundController = null;
  els.apiPlaygroundSystem.value = '';
  els.apiPlaygroundPrompt.value = '';
  els.apiPlaygroundTemperature.value = String(state.appConfig.apiDefaultTemperature ?? 0.3);
  els.apiPlaygroundMaxTokens.value = '';
  els.apiPlaygroundStream.checked = true;
  els.apiPlaygroundSend.disabled = false;
  els.apiPlaygroundStop.hidden = true;
  els.apiPlaygroundStatus.textContent = '';
  els.apiPlaygroundAnswer.textContent = '';
  els.apiPlaygroundAnswerCopy.disabled = true;
  els.apiPlaygroundRaw.textContent = '';
  updateApiPlaygroundRequest();
}

async function readApiPlaygroundStream(response, runId) {
  if (!response.body) throw new Error('浏览器不支持流式读取');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let reachedDone = false;

  const parseLine = (raw) => {
    const line = raw.replace(/\r$/, '').trimEnd();
    if (!line.startsWith('data:')) return;
    const data = line.slice(5).trim();
    if (!data) return;
    if (data === '[DONE]') {
      reachedDone = true;
      return;
    }
    let chunk;
    try {
      chunk = JSON.parse(data);
    } catch {
      throw new Error('API 返回了无法解析的 SSE 数据');
    }
    if (chunk.error) throw new Error(chunk.error.message || '模型返回错误');
    if (runId !== apiPlaygroundRunId) return;
    els.apiPlaygroundRaw.textContent += `data: ${data}\n\n`;
    els.apiPlaygroundRaw.scrollTop = els.apiPlaygroundRaw.scrollHeight;
    const delta = chunk.choices?.[0]?.delta?.content;
    if (typeof delta === 'string' && delta) {
      els.apiPlaygroundAnswer.textContent += delta;
      els.apiPlaygroundAnswer.scrollTop = els.apiPlaygroundAnswer.scrollHeight;
      els.apiPlaygroundAnswerCopy.disabled = false;
    }
  };

  try {
    while (!reachedDone) {
      const { value, done } = await reader.read();
      if (runId !== apiPlaygroundRunId) return;
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        parseLine(line);
        if (reachedDone || runId !== apiPlaygroundRunId) break;
      }
    }
    buffer += decoder.decode();
    if (buffer && !reachedDone && runId === apiPlaygroundRunId) parseLine(buffer);
    if (!reachedDone && runId === apiPlaygroundRunId) throw new Error('流式响应提前结束，未收到结束标记。');
    if (reachedDone && runId === apiPlaygroundRunId) els.apiPlaygroundRaw.textContent += 'data: [DONE]\n\n';
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // 已取消的流可能已经释放 reader。
    }
  }
}

async function submitApiPlayground(event) {
  event.preventDefault();
  const prompt = els.apiPlaygroundPrompt.value.trim();
  if (!prompt) {
    els.apiPlaygroundStatus.textContent = '请输入用户消息。';
    els.apiPlaygroundPrompt.focus();
    return;
  }
  if (!apiDisplayModel()) {
    els.apiPlaygroundStatus.textContent = state.appConfig.localBrowserBridge
      ? '本机模型网页尚未连接或尚未上报模型，请保持本网页打开后重试。'
      : nativeModels().length === 0
        ? '还没有可用的本机大模型：先到“模型下载”里下载一个 GGUF 模型，或联系站点所有者配置模型。'
        : '服务器尚未配置模型，请联系站点所有者。';
    return;
  }
  const token = state.apiKeyAdminToken;
  if (!token) {
    els.apiPlaygroundStatus.textContent = '需要先验证 API_ADMIN_PASSWORD 才能使用网页调试台。';
    return;
  }

  const payload = apiPlaygroundPayload();
  updateApiPlaygroundRequest();
  const runId = ++apiPlaygroundRunId;
  const controller = new AbortController();
  apiPlaygroundController = controller;
  els.apiPlaygroundSend.disabled = true;
  els.apiPlaygroundStop.hidden = false;
  els.apiPlaygroundAnswer.textContent = '';
  els.apiPlaygroundAnswerCopy.disabled = true;
  els.apiPlaygroundRaw.textContent = '';
  els.apiPlaygroundStatus.textContent = payload.stream ? '正在连接 API，准备接收流式回答…' : '正在等待完整 JSON 响应…';

  try {
    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${token}`,
        accept: payload.stream ? 'text/event-stream' : 'application/json',
      },
      body: JSON.stringify(payload),
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      if (response.status === 401) {
        lockApiKeyManager();
        els.apiKeyAdminError.textContent = '管理员验证已过期，请重新验证 API_ADMIN_PASSWORD。';
        return;
      }
      throw new Error(data.error?.message || data.error || `请求失败（HTTP ${response.status}）`);
    }
    if (payload.stream) {
      await readApiPlaygroundStream(response, runId);
    } else {
      const data = await response.json();
      if (runId !== apiPlaygroundRunId) return;
      els.apiPlaygroundAnswer.textContent = data.choices?.[0]?.message?.content || '';
      els.apiPlaygroundAnswerCopy.disabled = !els.apiPlaygroundAnswer.textContent;
      els.apiPlaygroundRaw.textContent = JSON.stringify(data, null, 2);
    }
    if (runId === apiPlaygroundRunId) {
      els.apiPlaygroundStatus.textContent = '请求完成。管理员令牌仅通过 Authorization 请求头发送，未放入可复制的请求 JSON。';
    }
  } catch (error) {
    if (runId !== apiPlaygroundRunId) return;
    els.apiPlaygroundStatus.textContent = error.name === 'AbortError'
      ? '已停止本次请求。'
      : `请求失败：${error.message || '未知错误'}`;
  } finally {
    if (runId === apiPlaygroundRunId) {
      apiPlaygroundController = null;
      els.apiPlaygroundSend.disabled = false;
      els.apiPlaygroundStop.hidden = true;
    }
  }
}

function updateApiCurlExample() {
  if (!els.apiCurlExample) return;
  const key = els.apiKeyNewValue.value || els.apiSelfKeyValue?.value || '<API_KEY>';
  const body = {
    model: apiDisplayModel() || '<MODEL>',
    messages: [{ role: 'user', content: '你好，请用一句话介绍你自己。' }],
    stream: false,
  };
  els.apiCurlExample.textContent =
    `curl -sS "${els.apiAccessEndpoint.textContent}" \\\n` +
    `  -H "Authorization: Bearer ${key}" \\\n` +
    `  -H 'Content-Type: application/json' \\\n` +
    `  -d @- <<'JSON'\n${JSON.stringify(body, null, 2)}\nJSON`;
}

function makeShareUrl({ embed = false } = {}) {
  const url = new URL(`${accessBaseUrl()}/`);
  if (embed) url.searchParams.set('embed', '1');
  const prompt = els.sharePromptInput.value.trim();
  if (prompt) {
    url.searchParams.set('q', prompt);
    if (els.shareAutoSend.checked) url.searchParams.set('send', '1');
  }
  return url;
}

function updateIframeSnippet() {
  const url = makeShareUrl({ embed: true }).href.replaceAll('&', '&amp;').replaceAll('"', '&quot;');
  const title = String(state.appConfig.appName || 'AI 助手')
    .replaceAll('&', '&amp;')
    .replaceAll('"', '&quot;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
  els.shareIframeOutput.value = `<iframe src="${url}" width="420" height="640" style="border:0" title="${title}"></iframe>`;
}

function updateShareOutputs() {
  if (els.shareLinkOutput.value) els.shareLinkOutput.value = makeShareUrl().href;
  updateIframeSnippet();
}

function openApiAccess({ focusAdmin = false } = {}) {
  if (MODE !== 'remote') return;
  renderApiAccessInfo();
  refreshBridgeStatus();
  els.apiPlaygroundTemperature.value = String(state.appConfig.apiDefaultTemperature ?? 0.3);
  els.apiPlaygroundMaxTokens.max = String(state.appConfig.apiMaxTokens ?? 8192);
  updateApiPlaygroundRequest();
  els.shareLinkOutput.value = '';
  updateIframeSnippet();
  els.apiKeyAdminError.textContent = '';
  els.apiKeyManagerError.textContent = '';
  els.apiAccessDialog.showModal();
  if (state.apiKeyAdminToken) {
    loadManagedApiKeys();
    loadAdminSettings();
  }
  if (focusAdmin) {
    const target = state.apiKeyAdminToken ? els.apiAdminSettingsSection : els.apiAdminLoginSection;
    requestAnimationFrame(() => {
      target?.scrollIntoView({ block: 'start' });
      if (!state.apiKeyAdminToken && state.appConfig.apiKeyManagerEnabled) {
        els.apiKeyAdminPassword.focus({ preventScroll: true });
      }
    });
  }
}

function lockApiKeyManager() {
  // 只有管理员主动“锁定管理面板”、或服务端返回 401（令牌过期/失效）时才调用：
  // 清除内存与本机保存的令牌，之后必须重新验证。关闭面板、刷新页面都不锁定。
  apiKeyManagerEpoch += 1;
  clearApiPlayground();
  apiPlaygroundController = null;
  state.apiKeyAdminToken = '';
  clearApiAdminSession(localStorage);
  els.apiKeyAdminPassword.value = '';
  els.apiAdminApiKey.value = '';
  els.apiAdminClearApiKey.checked = false;
  apiKeyNewId = '';
  els.apiKeyNewValue.value = '';
  els.apiKeyNewPanel.hidden = true;
  if (els.apiSelfKeyValue) els.apiSelfKeyValue.value = '';
  if (els.apiSelfKeyPanel) els.apiSelfKeyPanel.hidden = true;
  if (els.apiSelfApplyError) els.apiSelfApplyError.textContent = '';
  updateApiCurlExample();
  els.apiKeyList.replaceChildren();
  els.apiKeyManagerError.textContent = '';
  renderApiAccessInfo();
}

async function apiKeyManagerRequest(path, { method = 'GET', body } = {}) {
  const headers = {};
  if (state.apiKeyAdminToken) headers.authorization = `Bearer ${state.apiKeyAdminToken}`;
  if (body !== undefined) headers['content-type'] = 'application/json';
  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    if (res.status === 401 && state.apiKeyAdminToken) {
      lockApiKeyManager();
      els.apiKeyAdminError.textContent = '管理员验证已过期，请重新验证 API_ADMIN_PASSWORD。';
    }
    throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  }
  return data;
}

async function loadManagedApiKeys() {
  const epoch = apiKeyManagerEpoch;
  try {
    const data = await apiKeyManagerRequest('/api/admin/api-keys');
    if (epoch !== apiKeyManagerEpoch || !els.apiAccessDialog.open || !state.apiKeyAdminToken) return;
    apiKeyEnvironmentCount = data.environmentKeys || 0;
    els.apiKeyManagerError.textContent = '';
    renderManagedApiKeys(data.keys || []);
  } catch (err) {
    if (epoch === apiKeyManagerEpoch && els.apiAccessDialog.open) {
      els.apiKeyManagerError.textContent = err.message || '无法读取 API Key 列表';
    }
  }
}

function fillAdminSettings(data) {
  els.apiAdminBaseUrl.value = data.llmBaseUrl || '';
  els.apiAdminModel.value = data.llmModel || '';
  els.apiAdminSystemPrompt.value = data.systemPrompt || '';
  els.apiAdminTemperature.value = String(data.llmTemperature ?? 0.3);
  els.apiAdminMaxTokens.value = String(data.apiMaxTokens ?? 8192);
  els.apiAdminGenerationParams.value = JSON.stringify(data.defaultGenerationParams || {}, null, 2);
  if (els.apiAdminCorsOrigins) els.apiAdminCorsOrigins.value = (data.corsOrigins || []).join('\n');
  els.apiAdminApiKey.value = '';
  els.apiAdminClearApiKey.checked = false;
  els.apiAdminApiKeyStatus.textContent = data.llmApiKeyConfigured
    ? '上游 API Key 已配置。为安全起见不会回显；输入新值可轮换，勾选上方选项可清除。'
    : '当前未配置上游 API Key（本地模型通常无需设置）。';
  state.appConfig.model = data.llmModel || null;
  state.appConfig.apiDefaultTemperature = data.llmTemperature;
  state.appConfig.apiMaxTokens = data.apiMaxTokens;
  els.apiPlaygroundTemperature.value = String(data.llmTemperature ?? 0.3);
  els.apiPlaygroundMaxTokens.max = String(data.apiMaxTokens ?? 8192);
  if (Number(els.apiPlaygroundMaxTokens.value) > data.apiMaxTokens) els.apiPlaygroundMaxTokens.value = '';
  updateApiPlaygroundRequest();
  renderApiAccessInfo();
}

async function loadAdminSettings() {
  const epoch = apiKeyManagerEpoch;
  try {
    const data = await apiKeyManagerRequest('/api/admin/settings');
    if (epoch !== apiKeyManagerEpoch || !els.apiAccessDialog.open || !state.apiKeyAdminToken) return;
    fillAdminSettings(data);
    els.apiAdminSettingsError.textContent = '';
    els.apiAdminSettingsStatus.textContent = '已加载当前服务端运行参数。';
  } catch (err) {
    if (epoch === apiKeyManagerEpoch && els.apiAccessDialog.open) {
      els.apiAdminSettingsError.textContent = err.message || '无法读取服务端参数';
    }
  }
}

function renderManagedApiKeys(keys) {
  els.apiKeyList.replaceChildren();
  const envNote = document.createElement('p');
  envNote.className = 'muted small';
  envNote.textContent = `服务器 .env 静态密钥：${apiKeyEnvironmentCount} 个（不在网页中展示或撤销）`;
  els.apiKeyList.append(envNote);
  if (!keys.length) {
    const empty = document.createElement('p');
    empty.className = 'muted small';
    empty.textContent = '还没有通过网站创建的 API Key。';
    els.apiKeyList.append(empty);
    return;
  }
  for (const key of keys) {
    const row = document.createElement('div');
    row.className = 'api-key-row';
    const main = document.createElement('div');
    main.className = 'api-key-row-main';
    const label = document.createElement('div');
    label.className = 'api-key-row-label';
    label.textContent = key.label;
    const meta = document.createElement('div');
    meta.className = 'api-key-row-meta';
    const date = new Date(key.createdAt);
    meta.textContent = `ojai_…${key.hint} · ${Number.isNaN(date.getTime()) ? key.createdAt : date.toLocaleString()}`;
    const revoke = document.createElement('button');
    revoke.type = 'button';
    revoke.className = 'btn ghost small';
    revoke.textContent = '撤销';
    revoke.title = '立即撤销此 Key；正在使用它的应用将无法继续调用';
    revoke.dataset.apiKeyRevoke = key.id;
    main.append(label, meta);
    row.append(main, revoke);
    els.apiKeyList.append(row);
  }
}

async function copyPlainText(value, button) {
  const text = String(value || '');
  if (!text) {
    showStatus('没有可复制的内容。', { error: true });
    return false;
  }
  let ok = false;
  try {
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
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
  const previous = button?.textContent;
  if (button) {
    button.textContent = ok ? '已复制' : '复制失败';
    setTimeout(() => (button.textContent = previous), 1500);
  }
  if (!ok) showStatus('复制失败，请手动选中并复制。', { error: true });
  return ok;
}

els.apiAccessBtn.addEventListener('click', () => openApiAccess());
els.adminPanelBtn.addEventListener('click', () => openApiAccess({ focusAdmin: true }));
els.apiAccessClose.addEventListener('click', () => els.apiAccessDialog.close());
els.modelDownloadBtn?.addEventListener('click', () => {
  // 独立“模型下载”面板：打开先刷一次状态，让进度条与任务状态立刻是最新的
  if (els.modelDownloadDialog && typeof els.modelDownloadDialog.showModal === 'function' && !els.modelDownloadDialog.open) {
    els.modelDownloadDialog.showModal();
  }
  void refreshNativeStatus();
});
els.modelDownloadClose?.addEventListener('click', () => els.modelDownloadDialog?.close());
els.apiNativeGotoAdmin?.addEventListener('click', () => {
  els.modelDownloadDialog?.close();
  openApiAccess({ focusAdmin: true });
});
els.apiSelfApplyBtn?.addEventListener('click', async () => {
  els.apiSelfApplyError.textContent = '';
  els.apiSelfApplyBtn.disabled = true;
  try {
    const res = await fetch('/api/api-keys/apply', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ label: '网页自助申请' }),
      cache: 'no-store',
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `申请失败（HTTP ${res.status}）`);
    els.apiSelfKeyValue.value = data.key || '';
    els.apiSelfKeyPanel.hidden = false;
    els.apiSelfKeyValue.select();
    updateApiCurlExample();
  } catch (err) {
    els.apiSelfApplyError.textContent = err.message || '申请失败';
  } finally {
    els.apiSelfApplyBtn.disabled = false;
  }
});
els.apiSelfKeyCopy?.addEventListener('click', () => copyPlainText(els.apiSelfKeyValue.value, els.apiSelfKeyCopy));
els.apiAccessDialog.addEventListener('close', () => {
  // 管理员验证只验证一次：关闭面板不清令牌（令牌保存在本机浏览器中），
  // 只有点“锁定管理面板”或令牌过期才需要重新验证。这里只清临时内容。
  clearApiPlayground();
  els.sharePromptInput.value = '';
  els.shareAutoSend.checked = false;
  els.shareLinkOutput.value = '';
  els.shareIframeOutput.value = '';
});
els.sharePromptInput.addEventListener('input', updateShareOutputs);
els.shareAutoSend.addEventListener('change', updateShareOutputs);
els.shareLinkGenerate.addEventListener('click', () => {
  els.shareLinkOutput.value = makeShareUrl().href;
  updateIframeSnippet();
});
els.shareLinkCopy.addEventListener('click', () => copyPlainText(els.shareLinkOutput.value, els.shareLinkCopy));
els.shareIframeCopy.addEventListener('click', () => copyPlainText(els.shareIframeOutput.value, els.shareIframeCopy));
els.apiPlaygroundForm.addEventListener('submit', submitApiPlayground);
for (const input of [els.apiPlaygroundSystem, els.apiPlaygroundPrompt, els.apiPlaygroundTemperature, els.apiPlaygroundMaxTokens]) {
  input.addEventListener('input', updateApiPlaygroundRequest);
}
els.apiPlaygroundStream.addEventListener('change', updateApiPlaygroundRequest);
els.apiPlaygroundModel?.addEventListener('change', () => {
  renderApiAccessInfo();
  updateApiPlaygroundRequest();
});
els.apiPlaygroundRequestCopy.addEventListener('click', () => copyPlainText(els.apiPlaygroundRequest.textContent, els.apiPlaygroundRequestCopy));
els.apiPlaygroundAnswerCopy.addEventListener('click', () => copyPlainText(els.apiPlaygroundAnswer.textContent, els.apiPlaygroundAnswerCopy));
els.apiPlaygroundStop.addEventListener('click', () => {
  if (apiPlaygroundController) {
    els.apiPlaygroundStatus.textContent = '正在停止请求…';
    apiPlaygroundController.abort();
  }
});
els.apiPlaygroundClear.addEventListener('click', clearApiPlayground);
els.apiKeyAdminLoginForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const epoch = apiKeyManagerEpoch;
  els.apiKeyAdminError.textContent = '';
  els.apiKeyAdminLoginBtn.disabled = true;
  try {
    const res = await fetch('/api/admin/api-keys/login', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ password: els.apiKeyAdminPassword.value }),
      cache: 'no-store',
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `验证失败（HTTP ${res.status}）`);
    if (epoch !== apiKeyManagerEpoch || !els.apiAccessDialog.open) return;
    state.apiKeyAdminToken = data.token;
    // 验证一次即可：把令牌保存在本机浏览器，之后打开面板/刷新页面都不用再验证，
    // 只有点“锁定管理面板”或令牌到期（30 天）才需要重新验证。
    saveApiAdminSession(localStorage, { token: data.token, expiresAt: data.expiresAt });
    els.apiKeyAdminPassword.value = '';
    renderApiAccessInfo();
    requestAnimationFrame(() => els.apiAdminSettingsSection.scrollIntoView({ block: 'start' }));
    await Promise.all([loadManagedApiKeys(), loadAdminSettings()]);
  } catch (err) {
    if (epoch === apiKeyManagerEpoch && els.apiAccessDialog.open) {
      els.apiKeyAdminError.textContent = err.message || '管理员验证失败';
    }
  } finally {
    els.apiKeyAdminLoginBtn.disabled = false;
  }
});
els.apiAdminSettingsForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const epoch = apiKeyManagerEpoch;
  els.apiAdminSettingsError.textContent = '';
  els.apiAdminSettingsStatus.textContent = '';
  els.apiAdminSettingsSave.disabled = true;
  try {
    let generationParams;
    try {
      generationParams = JSON.parse(els.apiAdminGenerationParams.value || '{}');
    } catch {
      throw new Error('额外生成参数不是合法 JSON。');
    }
    if (!generationParams || typeof generationParams !== 'object' || Array.isArray(generationParams)) {
      throw new Error('额外生成参数必须是 JSON 对象。');
    }
    const apiKey = els.apiAdminApiKey.value.trim();
    if (apiKey && els.apiAdminClearApiKey.checked) {
      throw new Error('请只选择设置新 API Key 或清除旧 Key，不能同时选择。');
    }
    const patch = {
      llmBaseUrl: els.apiAdminBaseUrl.value.trim(),
      llmModel: els.apiAdminModel.value.trim(),
      systemPrompt: els.apiAdminSystemPrompt.value,
      llmTemperature: Number(els.apiAdminTemperature.value),
      apiMaxTokens: Number(els.apiAdminMaxTokens.value),
      defaultGenerationParams: generationParams,
      corsOrigins: (els.apiAdminCorsOrigins?.value || '')
        .split(/[\n,]+/)
        .map((s) => s.trim())
        .filter(Boolean),
    };
    if (apiKey) patch.llmApiKey = apiKey;
    else if (els.apiAdminClearApiKey.checked) patch.llmApiKey = '';

    const data = await apiKeyManagerRequest('/api/admin/settings', { method: 'PUT', body: patch });
    if (epoch !== apiKeyManagerEpoch || !els.apiAccessDialog.open || !state.apiKeyAdminToken) return;
    fillAdminSettings(data);
    els.apiAdminSettingsStatus.textContent = '已保存并立即应用到所有 API 调用。';
  } catch (err) {
    if (epoch === apiKeyManagerEpoch && els.apiAccessDialog.open) {
      els.apiAdminSettingsError.textContent = err.message || '保存服务端参数失败';
    }
  } finally {
    els.apiAdminSettingsSave.disabled = false;
  }
});
els.apiAdminSettingsReset.addEventListener('click', async () => {
  if (!window.confirm('恢复服务器启动时读取的 .env 参数？这会清除管理员控制台保存的所有覆盖值。')) return;
  const epoch = apiKeyManagerEpoch;
  els.apiAdminSettingsError.textContent = '';
  els.apiAdminSettingsStatus.textContent = '';
  els.apiAdminSettingsReset.disabled = true;
  try {
    const data = await apiKeyManagerRequest('/api/admin/settings', { method: 'DELETE' });
    if (epoch !== apiKeyManagerEpoch || !els.apiAccessDialog.open || !state.apiKeyAdminToken) return;
    fillAdminSettings(data);
    els.apiAdminSettingsStatus.textContent = '已恢复服务器启动时读取的 .env 配置。';
  } catch (err) {
    if (epoch === apiKeyManagerEpoch && els.apiAccessDialog.open) {
      els.apiAdminSettingsError.textContent = err.message || '恢复 .env 参数失败';
    }
  } finally {
    els.apiAdminSettingsReset.disabled = false;
  }
});
els.apiKeyCreateForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const epoch = apiKeyManagerEpoch;
  const adminToken = state.apiKeyAdminToken;
  els.apiKeyManagerError.textContent = '';
  els.apiKeyCreateBtn.disabled = true;
  try {
    const data = await apiKeyManagerRequest('/api/admin/api-keys', {
      method: 'POST',
      body: { label: els.apiKeyLabelInput.value },
    });
    if (epoch !== apiKeyManagerEpoch || !els.apiAccessDialog.open) {
      // 若面板关闭期间创建请求已成功，尽力立即撤销未能交付给用户的 Key。
      if (adminToken && data.id) {
        fetch(`/api/admin/api-keys/${encodeURIComponent(data.id)}`, {
          method: 'DELETE',
          headers: { authorization: `Bearer ${adminToken}` },
          cache: 'no-store',
        }).catch(() => {});
      }
      return;
    }
    els.apiKeyLabelInput.value = '';
    apiKeyNewId = data.id;
    els.apiKeyNewValue.value = data.key;
    els.apiKeyNewPanel.hidden = false;
    updateApiCurlExample();
    await loadManagedApiKeys();
  } catch (err) {
    if (epoch === apiKeyManagerEpoch && els.apiAccessDialog.open) {
      els.apiKeyManagerError.textContent = err.message || '创建 API Key 失败';
    }
  } finally {
    els.apiKeyCreateBtn.disabled = false;
  }
});
els.apiKeyNewCopy.addEventListener('click', () => copyPlainText(els.apiKeyNewValue.value, els.apiKeyNewCopy));
els.apiCurlCopy.addEventListener('click', () => copyPlainText(els.apiCurlExample.textContent, els.apiCurlCopy));
els.localApiCopy?.addEventListener('click', () => copyPlainText(els.localApiEndpoint.textContent, els.localApiCopy));
els.apiKeyAdminLogout.addEventListener('click', lockApiKeyManager);
els.apiNativeBinaryBtn?.addEventListener('click', () => {
  void nativeAction(async () => {
    await nativeAdminRequest('/api/native/binary', { token: state.apiKeyAdminToken });
  });
});
els.apiNativeUnloadBtn?.addEventListener('click', () => {
  void nativeAction(() => nativeAdminRequest('/api/native/unload', { token: state.apiKeyAdminToken }));
});
els.apiHfSearchBtn?.addEventListener('click', () => void runHfSearch());
els.importFilesBtn?.addEventListener('click', () => els.importFilesInput?.click());
els.importFolderBtn?.addEventListener('click', () => els.importFolderInput?.click());
els.importFilesInput?.addEventListener('change', () => void runImport(els.importFilesInput, false));
els.importFolderInput?.addEventListener('change', () => void runImport(els.importFolderInput, true));

/** 本机导入：把浏览器下载好的文件传进软件，服务器自动识别格式并登记。 */
async function runImport(input, fromFolder) {
  const files = [...(input?.files || [])];
  if (input) input.value = '';
  if (!files.length) return;
  if (!state.apiKeyAdminToken) {
    if (els.importStatus) els.importStatus.textContent = '导入需要先验证管理员口令：点面板顶部的“去验证管理员口令”，验证后回来再选文件。';
    return;
  }
  // 文件夹导入用相对路径（去掉第一层目录名，保留内部结构）；多选文件用文件名
  const entries = files.map((file) => {
    let rel = fromFolder ? (file.webkitRelativePath || file.name) : file.name;
    if (fromFolder && rel.includes('/')) rel = rel.split('/').slice(1).join('/');
    return { file, path: rel || file.name };
  }).filter((e) => e.file.size >= 0 && !e.path.endsWith('.DS_Store'));
  const setProgress = (sent, total) => {
    const pct = total ? Math.floor((sent / total) * 100) : 0;
    if (els.importProgressWrap) els.importProgressWrap.hidden = false;
    if (els.importProgressFill) els.importProgressFill.style.width = `${pct}%`;
    if (els.importProgressPct) els.importProgressPct.textContent = `${pct}%`;
    if (els.importProgressLabel) {
      els.importProgressLabel.textContent = `正在导入：${(sent / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB（${pct}%），全程只走本机，关掉页面下次选同一组文件可续传。`;
    }
  };
  if (els.importStatus) els.importStatus.textContent = '正在导入（把文件传进软件，随后自动识别格式）…';
  try {
    const out = await importLocalModel({
      entries,
      name: els.importNameInput?.value.trim() || undefined,
      adminToken: state.apiKeyAdminToken,
      onProgress: setProgress,
    });
    if (els.importProgressFill) els.importProgressFill.style.width = '100%';
    if (els.importProgressPct) els.importProgressPct.textContent = '100%';
    const routeText = (out.quants || []).length
      ? `已识别为 GGUF（${out.quants.join('、')}），在下方列表里点“加载”即可使用`
      : out.snapshot
        ? '已按原格式快照登记，在下方列表里点“转换并使用”或“本地运行（Python）”即可用'
        : '已登记';
    if (els.importStatus) els.importStatus.textContent = `导入完成：${out.name}。${routeText}。`;
  } catch (err) {
    if (err.status === 401 && state.apiKeyAdminToken) lockApiKeyManager();
    if (els.importStatus) els.importStatus.textContent = `导入失败：${err.message} 已传进软件的部分保留，重新选同一组文件会接着传。`;
  }
  await refreshNativeStatus();
}
els.apiHfSearchInput?.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    void runHfSearch();
  }
});
els.apiHfFilter?.addEventListener('change', () => renderHfResults());
els.apiHfTokenBtn?.addEventListener('click', () => {
  void nativeAction(async () => {
    const out = await nativeAdminRequest('/api/native/hf/token', {
      token: state.apiKeyAdminToken,
      body: { token: els.apiHfTokenInput.value },
    });
    els.apiHfTokenInput.value = '';
    if (els.apiHfTokenStatus) els.apiHfTokenStatus.textContent = out.hasToken ? 'Token 已保存（云端模型可用）' : 'Token 已清除';
  });
});
els.apiHfConvertToolBtn?.addEventListener('click', () => {
  void nativeAction(async () => {
    await nativeAdminRequest('/api/native/hf/prepare-converter', { token: state.apiKeyAdminToken });
    if (els.apiHfConvertToolStatus) els.apiHfConvertToolStatus.textContent = '正在准备转换工具（下载依赖中，可能需要几分钟）…';
  });
});
els.apiHfRunnerBtn?.addEventListener('click', () => {
  void nativeAction(async () => {
    await nativeAdminRequest('/api/native/hf/prepare-runner', { token: state.apiKeyAdminToken });
    if (els.apiHfRunnerStatus) els.apiHfRunnerStatus.textContent = '正在准备本地运行时（下载依赖中，可能需要几分钟）…';
  });
});
els.apiKeyList.addEventListener('click', async (e) => {
  const button = e.target.closest('[data-api-key-revoke]');
  if (!button) return;
  const revokeId = button.dataset.apiKeyRevoke;
  button.disabled = true;
  els.apiKeyManagerError.textContent = '';
  try {
    await apiKeyManagerRequest(`/api/admin/api-keys/${encodeURIComponent(revokeId)}`, { method: 'DELETE' });
    if (revokeId === apiKeyNewId) {
      apiKeyNewId = '';
      els.apiKeyNewValue.value = '';
      els.apiKeyNewPanel.hidden = true;
      updateApiCurlExample();
    }
    await loadManagedApiKeys();
    showStatus('API Key 已撤销。');
  } catch (err) {
    els.apiKeyManagerError.textContent = err.message || '撤销 API Key 失败';
  } finally {
    button.disabled = false;
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
  const bridgeMode = bridgePanelMode();
  const admin = panelAdmin();
  const exp = adminExpiry();
  const hasKey = hasAccessGate(state.appConfig.access);
  if (MODE === 'webllm') {
    els.roleBadge.textContent = admin ? '管理员' : '访客';
    els.roleBadge.classList.toggle('is-admin', admin);
  }
  els.roleForm.hidden = bridgeMode || admin || !hasKey;
  els.roleLogoutBtn.hidden = !admin || bridgeMode;
  if (bridgeMode) {
    els.roleStatus.textContent = '本机版已登录：可在这里导入“新建AI”、直接改参数（仅保存在本机浏览器）。导入的 AI 会自动加入网页模型栏与 API 的可用模型。';
  } else if (admin) {
    els.roleStatus.textContent = `当前为管理员（有效至 ${exp.toLocaleDateString('zh-CN')}）。可导出和导入模型。`;
  } else if (hasKey) {
    els.roleStatus.textContent = '当前为访客。输入管理员密钥可升级，以导出和导入模型；聊天功能不受影响。';
  } else {
    els.roleStatus.textContent = '未配置管理员密钥，无法升级权限。';
  }
  // 本机版的模型由桥接引擎持有，暂不支持导出为 .ojmodel 文件；导入与删除可用
  els.exportModelBtn.disabled = !admin || bridgeMode;
  for (const btn of els.sitePackageBtns) btn.disabled = !admin;
  if (els.trainingPackageBtn) els.trainingPackageBtn.disabled = !admin;
  els.exportDataBtn.disabled = !admin;
  // 网站包与本机数据导出只用于静态版（远程模式的数据在服务端）
  els.dataSectionList.closest('.data-export').hidden = MODE !== 'webllm';
  if (els.sitePackageBtns[0]) els.sitePackageBtns[0].closest('.site-tools').hidden = MODE !== 'webllm';
  renderSitePackageHint();
  setImportButtons(admin);
  els.modelAdminHint.textContent = bridgeMode
    ? '导入的“新建AI”会加入网页顶部模型栏，也可通过 API 调用（在请求中填它的模型 ID，或先 GET /v1/models 查看）。'
    : admin
      ? 'AI 训练：请使用仓库中的 training/ 训练运行时（需要 GPU），训练结果用“导入训练结果”加入新建AI。'
      : '导出与导入模型仅限管理员。';
  renderLocalList();
  renderAdaptList();
  renderParamOverrides();
  // 全局知识库：管理员可编辑并保存（非管理员只读回显）
  if (els.kbTextInput && document.activeElement !== els.kbTextInput) {
    els.kbTextInput.value = getKnowledgeText();
  }
  if (els.kbTextInput) els.kbTextInput.disabled = !admin;
  if (els.kbSaveBtn) els.kbSaveBtn.disabled = !admin;
  updateKbCharCount();
}

function updateKbCharCount() {
  if (els.kbCharCount && els.kbTextInput) {
    els.kbCharCount.textContent = `${els.kbTextInput.value.length} / ${MAX_KB_TEXT_CHARS}`;
  }
}
els.kbTextInput?.addEventListener('input', updateKbCharCount);
els.kbSaveBtn?.addEventListener('click', () => {
  if (!panelAdmin()) {
    if (els.kbStatus) els.kbStatus.textContent = '请先通过管理员验证，再保存知识库。';
    return;
  }
  const saved = setKnowledgeText(els.kbTextInput?.value || '');
  if (els.kbTextInput) { els.kbTextInput.value = saved; updateKbCharCount(); }
  if (els.kbStatus) {
    els.kbStatus.textContent = saved.trim()
      ? `已保存（${saved.length} 字），对所有模型生效。`
      : '已清空：之后不会再向 AI 发送知识库内容。';
  }
  scheduleAccountSync();
  showStatus('全局知识库已保存。');
});

/** AI 适应记录：管理员可查看、删除（仅本机）。全局知识库不在此列（上方输入框单独维护）。 */
function renderAdaptList() {
  const admin = isAdmin();
  renderExportParamsOption();
  const list = els.adaptList;
  if (!list) return;
  list.replaceChildren();
  if (MODE !== 'webllm' || !admin) {
    els.adaptHint.textContent = bridgePanelMode()
      ? '本机版请在上方“直接修改参数”中调整参数；通过对话记录的 AI 适应反馈不在本机版网页中提供。'
      : MODE !== 'webllm'
        ? '服务端模式下不记录 AI 适应反馈（知识库仍为全局一份，在本机浏览器保存）。'
        : 'AI 适应记录仅管理员可见。';
    return;
  }
  els.adaptHint.textContent =
    '管理员在 AI 回复之后发送的消息会记为该 AI 的附属说明（仅本机，不上传）；' +
    '以“参数：temperature=0.2, maxTokens=800”开头的一行可调整该 AI 的参数。' +
    '需要让所有 AI 都知道的内容，请写到上方“全局知识库”输入框（顶替原按模型知识库）。';
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
    name.textContent = `${modelDisplayName(id)}（${notes.length} 条说明${Object.keys(params).length ? '，有参数覆盖' : ''}）`;
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

const PARAM_SOURCE_LABEL = { explicit: '参数行', suggestion: '关键词识别', reset: '恢复默认', panel: '管理员面板' };

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
 * 撤回参数覆盖：任何人都可以在“管理员面板”中恢复某个参数为默认值（只影响本浏览器）。
 * 说明（notes）仍只能由管理员删除。
 */
async function revertParam(modelId, key) {
  if (!requireAdmin('恢复默认参数')) return;
  if (!resetParam(modelId, key)) return;
  refreshParamViews();
  const released = await releaseAfterSettingChange();
  const tail = released ? '已释放多余显存。' : '';
  showStatus(`已恢复「${modelDisplayName(modelId)}」的 ${key} 为默认值（仅本机，从下一次回答起生效）。${tail}`);
}

async function revertAllParams(modelId) {
  if (!requireAdmin('恢复默认参数')) return;
  const n = resetAllParams(modelId);
  if (!n) return;
  refreshParamViews();
  const released = await releaseAfterSettingChange();
  const tail = released ? '已释放多余显存。' : '';
  showStatus(`已恢复「${modelDisplayName(modelId)}」的全部 ${n} 项参数为默认值（仅本机，从下一次回答起生效）。${tail}`);
}

function refreshParamViews() {
  renderAdaptList();
  renderParamOverrides();
}

/** 参数摘要，例如 “temperature=0.2，maxTokens=800”。 */
function paramSummary(params) {
  return Object.entries(params).map(([k, v]) => `${k}=${v}`).join('，');
}

/** 可在管理员面板直接修改的参数（顺序即表单中的显示顺序）。 */
const PARAM_FORM_KEYS = ['temperature', 'repetitionPenalty', 'maxTokens', 'contextWindow', 'slidingWindow', 'thinkingWindow'];

/**
 * “管理员面板”中的参数区（仅管理员可见）：直接填数字即可修改当前模型的参数，
 * 不用再去对话里发“参数：temperature=…”指令。保存后立即生效（仅本机）。
 * 某项留空并保存 = 该项恢复默认/预设（去掉本机覆盖）；已有覆盖会在输入框中预填。
 */
function renderParamOverrides() {
  const box = els.paramOverrides;
  if (!box) return;
  box.replaceChildren();
  // 普通用户看不到任何参数改动（参数仍会生效，但界面不显示）
  box.hidden = true;
  if (!state.backend || !panelAdmin()) return;
  const bridgeMode = bridgePanelMode();
  if (MODE !== 'webllm' && !bridgeMode) return;
  box.hidden = false;
  // 静态版针对当前引擎模型；本机版针对网页模型栏选中的模型（桥接生成时同样会应用这些参数）
  const modelId = bridgeMode ? localChatModelId : state.backend.modelId;
  if (!modelId) {
    const wait = document.createElement('p');
    wait.className = 'muted small';
    wait.textContent = '等待本机模型连接后，可在这里直接修改参数。';
    box.append(wait);
    return;
  }
  const overrides = sanitizeParams(listAdaptation(modelId).params) || {};
  const eff = effectiveParams(modelId);
  const head = document.createElement('p');
  head.className = 'muted small';
  head.textContent = `直接修改当前模型「${modelDisplayName(modelId)}」的参数（仅本机，对本浏览器所有用户生效）：`;
  box.append(head);

  const form = document.createElement('form');
  form.className = 'param-form';
  for (const key of PARAM_FORM_KEYS) {
    const [min, max] = PARAM_LIMITS[key];
    const row = document.createElement('div');
    row.className = 'param-row';
    const label = document.createElement('label');
    label.className = 'small param-name';
    label.htmlFor = `paramInput-${key}`;
    label.textContent = `${PARAM_LABELS[key] || key}（${key}）`;
    const input = document.createElement('input');
    input.id = `paramInput-${key}`;
    input.type = 'number';
    input.min = String(min);
    input.max = String(max);
    input.step = key === 'temperature' || key === 'repetitionPenalty' ? '0.05' : '1';
    input.dataset.paramKey = key;
    if (key in overrides) input.value = String(overrides[key]);
    input.placeholder = `当前 ${eff[key]}`;
    input.title = `允许范围 ${min} – ${max}`;
    const hint = document.createElement('span');
    hint.className = 'muted small param-hint';
    hint.textContent = `范围 ${min}–${max}；当前生效 ${eff[key]}（${paramSourceOf(key, modelId)}）`;
    row.append(label, input, hint);
    form.append(row);
  }
  const actions = document.createElement('div');
  actions.className = 'param-actions';
  const save = document.createElement('button');
  save.type = 'submit';
  save.className = 'btn primary small';
  save.textContent = '保存参数';
  actions.append(save);
  if (Object.keys(overrides).length) {
    const all = document.createElement('button');
    all.type = 'button';
    all.className = 'btn ghost small';
    all.dataset.action = 'param-revert-all';
    all.dataset.id = modelId;
    all.textContent = '全部恢复默认';
    actions.append(all);
  }
  form.append(actions);
  const note = document.createElement('p');
  note.className = 'muted small';
  note.textContent = '填好后点“保存参数”立即生效，不用再去对话里发参数指令。某项留空并保存，表示该项恢复默认（去掉本机覆盖）。窗口类参数变化时会释放显存，下次发送时按新参数加载模型。';
  form.append(note);
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    void savePanelParams(modelId, form);
  });
  box.append(form);
}

/**
 * 保存管理员面板中直接填写的参数：写入的项成为本机覆盖，清空的项恢复默认。
 * 数值范围由 setModelParams / resetParam 按 PARAM_LIMITS 夹紧与记录，来源标记为 panel。
 */
async function savePanelParams(modelId, form) {
  if (!requireAdmin('修改参数')) return;
  const overrides = sanitizeParams(listAdaptation(modelId).params) || {};
  const eff = effectiveParams(modelId);
  const toWrite = {};
  const resetKeys = [];
  const invalid = [];
  for (const input of form.querySelectorAll('[data-param-key]')) {
    const key = input.dataset.paramKey;
    const raw = input.value.trim();
    if (!raw) {
      if (key in overrides) resetKeys.push(key);
      continue;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      invalid.push(PARAM_LABELS[key] || key);
      continue;
    }
    // 与当前覆盖（或无覆盖时的生效值）相同则不重复写入，避免产生无意义的变更记录
    if (key in overrides ? value !== overrides[key] : value !== eff[key]) toWrite[key] = value;
  }
  if (invalid.length) {
    showStatus(`参数填写有误：${invalid.join('、')} 不是有效数字，未保存任何修改。`, { error: true });
    return;
  }
  for (const key of resetKeys) resetParam(modelId, key);
  const written = Object.keys(toWrite).length
    ? setModelParams(modelId, toWrite, { source: 'panel', reason: '管理员面板直接输入' })
    : null;
  refreshParamViews();
  renderExportParamsOption();
  if (!written && !resetKeys.length) {
    showStatus('参数没有变化，未保存。');
    return;
  }
  const released = await releaseAfterSettingChange();
  const parts = [];
  if (written) parts.push(`已设置 ${paramSummary(written)}`);
  if (resetKeys.length) parts.push(`已恢复默认：${resetKeys.map((k) => PARAM_LABELS[k] || k).join('、')}`);
  const tail = released ? '已释放多余显存，下次发送时按新参数加载模型。' : '从下一次回答起生效。';
  showStatus(`已保存「${modelDisplayName(modelId)}」的参数（仅本机）：${parts.join('；')}。${tail}`);
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
  const admin = panelAdmin();
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
  // Shift+Enter 发送；Enter 换行（默认行为）。中文输入法组合输入期间按 Enter 不应发送
  if (e.key === 'Enter' && e.shiftKey && !e.isComposing) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});
els.composer.addEventListener('submit', (e) => {
  e.preventDefault();
  // 工具面板里的搜索源输入框按 Enter 只应结束输入，不应发送消息
  if (els.toolsPanel.contains(document.activeElement) && document.activeElement.matches('input, textarea')) return;
  const text = els.input.value;
  if (!text.trim()) return;
  els.input.value = '';
  autoGrow();
  toggleToolsPanel(false);
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

// ---------- 工具面板（思考模式 / 滑动窗口 / 联网资料检索）的事件 ----------
els.toolsBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  toggleToolsPanel();
});
els.toolsClose.addEventListener('click', () => {
  toggleToolsPanel(false);
  els.toolsBtn.focus({ preventScroll: true });
});
els.toolList.addEventListener('click', (e) => {
  const row = e.target.closest('[data-tool-open]');
  if (!row) return;
  e.preventDefault();
  openToolCard(row.dataset.toolOpen);
});
els.toolList.addEventListener('change', (e) => {
  const input = e.target.closest('[data-tool-toggle]');
  if (!input) return;
  const def = toolDefs().find((d) => d.id === input.dataset.toolToggle);
  if (!def || !def.available || state.streaming) {
    refreshTools();
    return;
  }
  def.toggle(input.checked);
});
els.toolCard.addEventListener('click', (e) => {
  const modeBtn = e.target.closest('[data-search-mode]');
  if (modeBtn) {
    state.search.prefs = { ...state.search.prefs, mode: modeBtn.dataset.searchMode };
    saveSearchPrefs(state.search.prefs);
    refreshTools();
    flashStatus(
      `已切换为${SEARCH_MODE_LABELS[state.search.prefs.mode]}（仅本机保存）：${
        state.search.prefs.mode === 'smart' ? '会补取维基导言、Stack 回答或 HN 讨论；已配置正文读取服务时才会读取其他网页，资料更完整但更慢。' : '只用资料源返回的摘要，更快。'
      }`,
    );
    return;
  }
  const action = e.target.closest('[data-search-action]');
  if (action) saveSearchSettingsFromCard({ test: action.dataset.searchAction === 'test' });
});
// 面板内输入框按 Enter 不提交聊天表单
els.toolsPanel.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && e.target.matches('input')) e.preventDefault();
});
// 点击面板外部或按 Esc 收起
document.addEventListener('click', (e) => {
  if (els.toolsPanel.hidden) return;
  // 面板内的点击会重绘工具行，事件冒泡到这里时 e.target 可能已经被移除，
  // 因此优先用 composedPath()（在派发时计算，包含已脱离文档的祖先）判断点击来源。
  const path = typeof e.composedPath === 'function' ? e.composedPath() : [];
  if (path.includes(els.toolsPanel) || path.includes(els.toolsBtn)) return;
  if (els.toolsPanel.contains(e.target) || els.toolsBtn.contains(e.target)) return;
  toggleToolsPanel(false);
});
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape' || els.toolsPanel.hidden) return;
  toggleToolsPanel(false);
  els.toolsBtn.focus({ preventScroll: true });
});

els.presetSelect?.addEventListener('change', () => applyPreset(els.presetSelect.value));

/** 切换模型：两个下拉框保持同步，并刷新工具面板（思考开关是否可用取决于模型）。 */
function onModelChange(id) {
  if (!id || state.streaming) return;
  if (MODE === 'remote') {
    // 本机版：只是记下网页聊天自己的选择（发送时逐条带上），不影响 API 各自选用的模型
    onBridgeModelChange(id);
    return;
  }
  state.backend.setModel(id);
  // 切换模型即更改默认模型：保存在本浏览器，下次打开时使用
  saveDefaultModel(id);
  els.modelSelect.value = id;
  if (els.modelSelectCompact) els.modelSelectCompact.value = id;
  refreshTools();
  renderParamOverrides();
  renderExportParamsOption();
  const select = els.modelSelectCompact && els.modelSelectCompact.value === id ? els.modelSelectCompact : els.modelSelect;
  const label = select.selectedOptions[0]?.textContent || id;
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
}

els.modelSelect.addEventListener('change', () => onModelChange(els.modelSelect.value));
els.modelSelectCompact?.addEventListener('change', () => onModelChange(els.modelSelectCompact.value));

document.addEventListener('click', (e) => {
  const btn = e.target.closest('[data-action]');
  if (btn && btn.dataset.action === 'switch') return switchSession(btn.dataset.id);
  if (btn && btn.dataset.action === 'delete') return removeSession(btn.dataset.id);
  if (btn && btn.dataset.action === 'copy') return copyCode(btn);
  if (btn && btn.dataset.action === 'regen') return regenerate();
  if (btn && btn.dataset.action === 'local-delete') return deleteLocalModel(btn.dataset.id);
  if (btn && btn.dataset.action === 'adapt-delete') return deleteAdaptNote(btn.dataset.id, btn.dataset.note);
  if (btn && btn.dataset.action === 'adapt-clear') return clearAdaptation(btn.dataset.id);
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
const imageState = {
  active: false,
  busy: false,
  controller: null,
  client: null,
  gallery: [],
  downloadCtl: null,
  filesStatus: null,
  filesToken: 0,
  preload: null, // { modelId, token, promise }：进入生图模式后在后台从本机加载模型
};

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
      opt.textContent = imageOptionLabel(m, false);
      opt.disabled = !isSelectable(m);
      opt.title = m.note || '';
      return opt;
    }),
  );
  // 与聊天模型一样记住上次的选择（刷新页面后仍是同一个模型）
  const startId = resolveStartupImageModel({
    models: models.map((m) => ({ id: m.id, selectable: isSelectable(m) })),
    saved: getSavedImageModel(),
  });
  if (startId) els.imageModelSelect.value = startId;
  fillImageControls(imageModelById(els.imageModelSelect.value));
  refreshImageCacheMarks();
}

/** 下拉框文字：不可用模型加“（暂不可用）”，已完整下载的加 ✓（与聊天模型一致）。 */
function imageOptionLabel(m, downloaded) {
  // 目录生成的说明里可能已含“暂不可用”，避免重复
  const base = isSelectable(m) || /暂不可用/.test(m.label) ? m.label : `${m.label}（暂不可用）`;
  return downloaded ? `✓ ${base}` : base;
}

/** 在下拉框中给已完整下载到本机的生图模型加上 ✓（与聊天模型一致）。 */
async function refreshImageCacheMarks() {
  try {
    const done = await listCompleteImageModelIds(imageModels());
    for (const opt of els.imageModelSelect.options) {
      const m = imageModelById(opt.value);
      if (!m) continue;
      opt.textContent = imageOptionLabel(m, done.has(m.id));
    }
  } catch (err) {
    console.warn('无法检查生图模型缓存状态', err);
  }
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
  const { params, hints } = getImageSettings(model.id);
  const eff = effectiveImageSettings(readImageControls(), params);
  const adminKeys = Object.keys(params);
  const parts = [
    `生效：${eff.width} × ${eff.height}，${eff.steps} 步，引导 ${eff.guidance}，种子 ${eff.seed ?? '随机'}`,
  ];
  if (adminKeys.length) parts.push(`管理员参数已覆盖：${adminKeys.map((k) => `${k}=${params[k]}`).join('，')}`);
  parts.push(`描述输入上限 ${effectiveInputLimit(params)} tokens（超过则拒绝生成；CLIP 每段读取 75 个，描述最多分 ${MAX_PROMPT_CHUNKS} 段，约 ${MAX_PROMPT_CHUNKS * 75} 个 token 参与生图）`);
  if (getKnowledgeText().trim()) parts.push('全局知识库已启用（提示词最前面）');
  if (hints.length) parts.push(`风格提示 ${hints.length} 条`);
  if (isSelectable(model)) parts.push(`模型约 ${formatGB(totalBytes(model))}（可点“下载到本机”保存，之后离线可用）`);
  els.imageEffective.textContent = parts.join('｜');
  renderImageExportParams(model);
}

/** 导出时“同时保存自定义参数”：只在管理员且该模型有参数时显示（与文本模型一致）。 */
function renderImageExportParams(model) {
  const params = sanitizeImageParams(getImageSettings(model.id).params);
  const show = isAdmin() && !!params;
  els.imageExportParamsLabel.hidden = !show;
  if (show) els.imageExportParamsText.textContent = `同时保存自定义参数（${paramSummary(params)}）`;
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
  const { hints } = getImageSettings(model.id);
  els.imageHintList.replaceChildren(
    ...hints.map((h) => imageNoteRow(h.text, () => removeImageHint(model.id, h.id), '删除风格提示')),
  );
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

/** 生图模式中权限变化后刷新：导出/导入/删除按钮、管理员面板、导出参数复选框。 */
function refreshImageRoleState() {
  if (!imageState.active) return;
  renderImageFiles();
  renderImageAdmin();
  renderImageEffective();
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
  // 导出与导入仅限管理员（与文本模型一致）；导出还需要文件完整下载到本机
  els.imageExportModelBtn.disabled = !st.complete || !isAdmin();
  els.imageImportModelBtn.disabled = !isAdmin() || !!imageState.downloadCtl || imageState.busy;
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
    refreshImageCacheMarks();
    preloadImageModel();
  } catch (err) {
    if (err.name === 'AbortError') setImageStatus('已取消下载。已保存的部分会保留，下次点击“继续下载”从断点继续。');
    else setImageStatus(`下载未完成：${err.message}`, { error: true });
  } finally {
    imageState.downloadCtl = null;
    els.imageDownloadProgress.hidden = true;
    renderImageFiles();
    refreshImageCacheMarks();
  }
}

async function exportSelectedModel() {
  if (!requireAdmin('导出生图模型')) return;
  const model = selectedImageModel();
  if (!model) return;
  if (imageState.busy || imageState.downloadCtl) return setImageStatus('请等待当前生成或下载结束后再导出。', { error: true });
  const withParams = !els.imageExportParamsLabel.hidden && els.imageExportWithParams.checked;
  const params = withParams ? sanitizeImageParams(getImageSettings(model.id).params) : null;
  const filename = imageModelFileName(model.id);
  // 先选择保存位置：必须在点击后立即弹出（之后打包可能很久，浏览器会撤销授权）。
  // 注意：这里不能有任何 await 在picker之前。
  let handle = null;
  if (supportsSaveLocationPicker()) {
    try {
      handle = await window.showSaveFilePicker(imagePickerOptions(filename));
    } catch (err) {
      if (isPickerCancelled(err)) return setImageStatus('已取消导出。');
      // 其他错误（例如嵌入到其他网站的 iframe 中被浏览器禁止）：回退到下载目录
      handle = null;
    }
  }
  els.imageExportModelBtn.disabled = true;
  setImageStatus(`正在打包「${model.label}」…`);
  try {
    const { parts, size, entryCount, filename: name } = await prepareImageModelExport(model, {
      params: params || undefined,
      onProgress: ({ done, total }) => setImageStatus(`正在打包「${model.label}」（${done}/${total} 个文件）…`),
    });
    if (handle) {
      // 逐段写入你选择的文件（不把整个模型合成为一个内存对象）
      const writable = await handle.createWritable();
      try {
        for (const part of parts) await writable.write(part);
        await writable.close();
      } catch (err) {
        await writable.abort?.(err).catch(() => {});
        throw err;
      }
    } else {
      downloadBlob(new Blob(parts, { type: 'application/octet-stream' }), name);
    }
    const where = handle ? `已保存到你选择的位置：${handle.name || name}` : '已下载到浏览器的下载目录';
    const paramNote = params ? `，已保存自定义参数（${paramSummary(params)}）` : '';
    setImageStatus(`${where}（${formatGB(size)}，${entryCount} 个文件${paramNote}）。导入后即可离线使用。`);
  } catch (err) {
    setImageStatus(`导出失败：${err.message}`, { error: true });
  } finally {
    renderImageFiles();
  }
}

async function importModelFileFromPicker(file) {
  if (!requireAdmin('导入生图模型')) {
    els.imageImportModelInput.value = '';
    return;
  }
  if (imageState.busy || imageState.downloadCtl) {
    els.imageImportModelInput.value = '';
    return setImageStatus('请等待当前生成或下载结束后再导入。', { error: true });
  }
  els.imageImportModelBtn.disabled = true;
  els.imageDownloadProgress.hidden = false;
  els.imageDownloadFill.style.width = '0%';
  setImageStatus(`正在导入 ${file.name}（${formatGB(file.size)}）…`);
  try {
    const { model, params } = await importImageModelFile(file, {
      models: imageModels(),
      onProgress: ({ done, total }) => {
        els.imageDownloadFill.style.width = `${Math.round((done / total) * 100)}%`;
        setImageStatus(`正在导入 ${file.name}（${done}/${total} 个文件）…`);
      },
    });
    // 文件中带有的自定义参数：写入该模型的本机参数
    const carried = params ? setImageParams(model.id, params) : null;
    // 导入后直接选中该模型（与文本模型“导入后即可使用”一致）
    els.imageModelSelect.value = model.id;
    saveImageModel(model.id);
    fillImageControls(model);
    renderImageAdmin();
    renderImageFiles();
    refreshImageCacheMarks();
    const carriedNote = carried ? `，并带入自定义参数（${paramSummary(carried)}）` : '';
    setImageStatus(`导入完成：「${model.label}」已写入本机，可离线使用${carriedNote}。`);
    stopImagePreload(); // 若之前预加载的是其他模型，先释放
    preloadImageModel();
  } catch (err) {
    setImageStatus(`导入失败：${err.message}`, { error: true });
  } finally {
    els.imageImportModelInput.value = '';
    els.imageDownloadProgress.hidden = true;
    els.imageDownloadFill.style.width = '0%';
    els.imageImportModelBtn.disabled = !isAdmin();
    renderImageFiles();
  }
}

async function deleteSelectedModelFiles() {
  const model = selectedImageModel();
  if (!model) return;
  if (!window.confirm(`删除「${model.label}」在本机的文件（约 ${formatGB(totalBytes(model))}）？删除后生成时需要重新下载。`)) return;
  try {
    // 若该模型正在预加载（或已加载在 worker 中），先释放，避免删除后仍占用内存
    stopImagePreload();
    const n = await deleteImageModelFiles(model);
    setImageStatus(`已删除本机的 ${n} 个模型文件。`);
  } catch (err) {
    setImageStatus(`删除失败：${err.message}`, { error: true });
  } finally {
    renderImageFiles();
    refreshImageCacheMarks();
  }
}

/**
 * 进入生图模式后，在后台把已完整下载的模型加载到 worker（与生成共用同一次加载）。
 * 页面刷新后 worker 会消失，所以刷新后的第一次进入仍需从本机读取一次，但不再等到第一次生成时才开始。
 */
async function preloadImageModel() {
  const model = selectedImageModel();
  if (!imageState.active || !model || !isSelectable(model) || imageState.downloadCtl || imageState.busy) return;
  if (imageState.preload?.modelId === model.id) return imageState.preload.promise;
  const st = await imageCacheStatus(model).catch(() => null);
  // 只预加载完整的模型；未完整时生成会按原有方式读取（不在这里重复读取）
  if (!st?.complete) return;
  if (!imageState.active || selectedImageModel()?.id !== model.id || imageState.busy) return;
  if (imageState.preload?.modelId === model.id) return imageState.preload.promise;
  const token = {};
  const current = () => imageState.preload?.token === token && !imageState.busy;
  const promise = imageClient()
    .load({
      urls: sourceUrls(model),
      sizes: sourceSizes(model),
      wasmBase: new URL('ort/', document.baseURI).href,
      onProgress: (msg) => {
        if (current()) setImageStatus(`正在从本机加载「${model.label}」…${msg.message || ''}`);
      },
    })
    .then(() => {
      if (current()) setImageStatus(`「${model.label}」已加载，可以直接生成。`);
    })
    .catch((err) => {
      if (err?.name === 'AbortError') return; // 切换模型或离开生图模式时主动结束
      if (current()) setImageStatus(`加载「${model.label}」失败：${imageErrorText(err)}`, { error: true });
    })
    .finally(() => {
      if (imageState.preload?.token === token) imageState.preload = null;
    });
  imageState.preload = { modelId: model.id, token, promise };
  return promise;
}

/** 结束预加载并释放 worker（生成进行中不打断）。 */
function stopImagePreload() {
  if (imageState.busy) return;
  imageState.preload = null;
  if (imageState.client?.running) imageState.client.dispose();
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
  preloadImageModel();
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
  imageState.preload = null;
  imageState.active = false;
  els.body.classList.remove('image-mode');
  els.imagePanel.hidden = true;
  els.imageModeBtn.textContent = '生图';
  els.imageModeBtn.setAttribute('aria-pressed', 'false');
  refreshTools();
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
  if (!text.trim()) {
    if (recorded) {
      setImageStatus('已记录管理员生图参数。');
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
  const { hints } = getImageSettings(model.id);
  const eff = currentImageSettings(model);
  if (eff.seed === null) eff.seed = randomSeed();
  // 全局知识库（非空时）放在最前面；lead 单独传给 worker 做预算检查（超出则拒绝，不截断）
  const leadTexts = imageLeadTexts(getKnowledgeText());
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
    // Shift+Enter 生成；Enter 换行（默认行为）
    if (e.key === 'Enter' && e.shiftKey && !e.isComposing) {
      e.preventDefault();
      els.imageForm.requestSubmit();
    }
  });
  els.imageCancelBtn.addEventListener('click', () => imageState.controller?.abort());
  els.imageModelSelect.addEventListener('change', () => {
    const model = imageModelById(els.imageModelSelect.value);
    saveImageModel(model?.id || null);
    stopImagePreload();
    fillImageControls(model);
    renderImageAdmin();
    renderImageFiles();
    preloadImageModel();
  });
  // 权限变化（登录、退出）时 refreshRoleUI 会更新角色标签；生图区据此同步导出/导入/参数按钮（不改动文本模型的代码）
  new MutationObserver(() => refreshImageRoleState()).observe(els.roleBadge, {
    childList: true,
    characterData: true,
    subtree: true,
  });
  // 管理员登录后，进入图片文件区时刷新导出/导入按钮的可用状态
  els.imageModelFiles.addEventListener('pointerenter', () => renderImageFiles());
  els.imageImportModelBtn.addEventListener('click', () => {
    if (requireAdmin('导入生图模型')) els.imageImportModelInput.click();
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
    return {
      appName: c.appName,
      allowedParentOrigins: c.allowedParentOrigins,
      access: {},
      webllm: null,
      model: c.model,
      apiBaseUrl: c.apiBaseUrl || '/v1',
      apiKeyAuth: !!c.apiKeyAuth,
      apiKeyManagerEnabled: !!c.apiKeyManagerEnabled,
      accounts: !!c.accounts,
      localBrowserBridge: !!c.localBrowserBridge,
      apiDefaultTemperature: Number.isFinite(Number(c.apiDefaultTemperature)) ? Number(c.apiDefaultTemperature) : 0.3,
      apiMaxTokens: Number.isInteger(Number(c.apiMaxTokens)) && Number(c.apiMaxTokens) > 0 ? Number(c.apiMaxTokens) : 8192,
      publicBaseUrl: c.publicBaseUrl || null,
    };
  }
  const res = await fetch('config.json', { cache: 'no-store' });
  if (!res.ok) throw new Error(`无法读取 config.json（HTTP ${res.status}）`);
  return res.json();
}

async function init() {
  if (MODE === 'remote' && els.accessDocsLink) els.accessDocsLink.hidden = false;
  if (MODE === 'remote' && els.apiAccessBtn) els.apiAccessBtn.hidden = false;
  if (MODE === 'remote' && els.adminPanelBtn) els.adminPanelBtn.hidden = false;
  if (MODE === 'remote' && els.modelDownloadBtn) els.modelDownloadBtn.hidden = false;
  if (isEmbed) els.body.classList.add('embed');
  if (isEmbedded) els.body.classList.add('in-frame');

  try {
    state.appConfig = { ...state.appConfig, ...(await loadAppConfig()) };
  } catch (err) {
    console.error(err);
  }
  const appName = state.appConfig.appName || 'OJ 刷题助手';
  if (els.releasePageLink) els.releasePageLink.href = releasePageUrl(state.appConfig?.release);
  els.brandName.textContent = appName;
  document.title = appName;

  // 联网资料检索：站点配置 + 本机偏好 + 管理员搜索源设置
  state.search.config = normalizeWebSearch(state.appConfig.webSearch);
  state.search.settings = loadProviderSettings();
  state.search.prefs = resolveSearchPrefs(state.search.config, loadSearchPrefs());

  state.backend = createBackend({ mode: MODE, appConfig: state.appConfig });
  // 恢复本机保存的 API 管理员验证：验证过一次后（含关掉面板、刷新页面）不用重复验证，
  // 只有点“锁定管理面板”或令牌到期（30 天）才需要重新验证。
  if (MODE === 'remote') {
    const savedAdmin = loadApiAdminSession(localStorage);
    if (savedAdmin) state.apiKeyAdminToken = savedAdmin.token;
  }
  // 本机浏览器模型桥接：把网页里的 WebGPU 引擎暴露给同机 API（保持网页打开即可）
  if (MODE === 'remote' && state.appConfig.localBrowserBridge) {
    try {
      const res = await fetch('config.json', { cache: 'no-store' });
      const siteConfig = res.ok ? await res.json() : null;
      if (siteConfig?.webllm) {
        // 保存网页模型配置：管理员面板的参数表单需要用它计算各参数的默认值与生效值
        state.appConfig.webllm = siteConfig.webllm;
        const { startLocalModelBridge } = await import('./local-model-bridge.js');
        startLocalModelBridge({ webllmCfg: siteConfig.webllm });
      }
    } catch (err) {
      console.warn('本机模型桥接启动失败：', err);
    }
    // 模型选择栏：拉取桥接可用模型并定期刷新（新导入模型后自动出现，不用刷新页面）
    localChatModelId = loadLocalChatModel();
    refreshBridgeStatus();
    setInterval(() => {
      if (!document.hidden) refreshBridgeStatus();
    }, 10000);
  }
  // 本机大模型（llama.cpp）：静态版没有该接口，会自动隐藏；网站包/独立软件中可用。
  // 以前这一步挂在“网页模型桥接”分支里：部署没开 LOCAL_BROWSER_BRIDGE 时，
  // 连已下载的 GGUF 模型都不会被加载出来，聊天页里根本选不到。
  if (MODE === 'remote') {
    refreshNativeStatus();
    // 下载/加载是后台任务（关掉面板也继续跑）：定期刷新，进度与完成状态才不会停在旧值
    setInterval(() => {
      if (!document.hidden) void refreshNativeStatus();
    }, 10000);
  }
  // 生图按钮与事件需要在配置加载之后绑定
  initImageMode();
  refreshTools();
  if (MODE === 'webllm') {
    // 启动时使用用户上次保存的默认模型（若仍在下拉框中），否则使用配置中的 defaultModel
    const availableIds = [...new Set([...state.backend.models.map((m) => m.id), ...loadLocalModels().map((e) => e.id)])];
    const requestedModel = availableIds.includes(linkModel) ? linkModel : getSavedDefaultModel();
    state.backend.setModel(
      resolveStartupModel({
        availableIds,
        defaultModel: state.appConfig.webllm.defaultModel,
        saved: requestedModel,
      }),
    );
    renderModelOptions(state.backend.modelId);
    renderPresetOptions();
    els.presetSelect.hidden = false;
    els.modelSelect.hidden = false;
    els.modelSelect.title = '✓ 表示模型已下载到本机，可离线使用';
    // 窄屏 / iframe 嵌入时侧边栏隐藏，标题栏里的小下拉框提供同样的切换能力
    els.modelSelectCompact.hidden = false;
    els.modelSelectCompact.title = '选择模型（窄屏下与侧边栏的下拉框同步）';
    els.modelLabel.hidden = true;
    setTimeout(refreshModelCacheStatus, 0);
    refreshRoleUI();
  } else {
    els.modelLabel.textContent = state.appConfig.model ? `模型：${state.appConfig.model}` : '';
    // 本机版保留“管理员面板”（导入新建AI、直接改参数等本机功能）；纯服务端模式没有这些本机数据，隐藏
    els.roleBtn.hidden = !state.appConfig.localBrowserBridge;
    els.presetSelect.hidden = true; // 预设只作用于本机模型
  }

  if (state.store.sessions.length === 0) createSession(state.store);
  else if (!getSession(state.store)) state.store.activeId = state.store.sessions[0].id;
  persist();

  notifyParentReady();

  // 两种模式都直接进入主界面：正常使用不上锁（无入口登录）。
  // 远程模式下再恢复账号登录状态（可选，不影响使用）。
  showMain();
  if (MODE === 'remote') await initAccount();
}

init();
