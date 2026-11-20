/**
 * 本机模型桥接客户端：把运行在本网页里的 WebLLM（WebGPU）引擎
 * 通过 WebSocket 暴露给同机的 Node API 服务（/v1/chat/completions）。
 *
 * 协议（JSON 文本帧）：
 *   → { type: 'ready', protocolVersion, tabId, models: [{ id, label }] }
 *                                                  连接后上报协议版本与可用模型（内置 + 本机导入）
 *   ← { type: 'request', id, payload }             payload: { model, messages, temperature, maxTokens, hardMaxTokens, params, systemPrompt }
 *   → { type: 'delta', id, content }               流式片段
 *   → { type: 'complete', id, finishReason }
 *   → { type: 'error', id, message, error }        error 为结构化诊断（阶段/错误名/原始原因）
 *   ← { type: 'cancel', id }                       中断生成
 *
 * 请求按顺序执行（引擎同一时间只处理一个生成）；断线后自动重连并重新上报模型列表。
 */
import { createBackend } from './engines/backend.js';
import { loadLocalModels } from './local-models.js';
import { collectModelEntries } from './model-file.js';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;

/** 桥接协议版本：服务端据此识别旧版网页，避免旧标签页静默顶替新版网页提供模型。 */
export const BRIDGE_PROTOCOL_VERSION = 2;

const BRIDGE_PHASE_LABELS = {
  'validate-model': '校验模型',
  'check-local-model': '检查本地导入模型缓存',
  'load-model': '加载模型',
  'prepare-generation': '准备生成参数',
  generate: '生成回答',
  unknown: '未知阶段',
};

/**
 * 把引擎抛出的各种错误（Error、DOMException、字符串、WebLLM 的普通对象）
 * 整理成一句能跨 WebSocket 传回服务端的说明。此前这里只取 err.message，
 * 拿不到时就只剩一句“本机模型生成失败”，真实原因（如 WebGPU 丢失、显存不足、
 * 模型下载失败）全被吞掉，服务端日志和调用方都无从排查。
 */
function safeRead(value, key) {
  try {
    return value?.[key];
  } catch {
    return undefined;
  }
}

function safeText(value, max = 700) {
  try {
    if (typeof value === 'string') return value.replace(/\s+/g, ' ').trim().slice(0, max);
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    if (typeof value === 'bigint') return `${value}n`;
    if (typeof value === 'symbol') return value.toString();
    return '';
  } catch {
    return '';
  }
}

function safeObjectTag(value) {
  try {
    return Object.prototype.toString.call(value).slice(8, -1) || 'Object';
  } catch {
    return '无法读取的错误对象';
  }
}

function describeErrorValue(err, depth = 0) {
  if (typeof err === 'string') return err.replace(/\s+/g, ' ').trim();
  if (err === null) return 'null';
  if (err === undefined) return '';
  if (typeof err !== 'object' && typeof err !== 'function') return safeText(err);

  const name = safeText(safeRead(err, 'name'), 120);
  const message = safeText(safeRead(err, 'message'), 700);
  let detail = [name && name !== 'Error' ? name : '', message].filter(Boolean).join(': ');

  if (!detail && depth < 3) {
    for (const key of ['error', 'cause', 'reason']) {
      const nested = safeRead(err, key);
      if (nested && nested !== err) {
        detail = describeErrorValue(nested, depth + 1);
        if (detail) break;
      }
    }
  }

  if (!detail) {
    const stackFirst = safeText(safeRead(err, 'stack'), 300).split('\n')[0] || '';
    if (stackFirst && stackFirst !== 'Error') detail = stackFirst;
  }

  if (!detail) {
    try {
      const parts = [];
      for (const key of Reflect.ownKeys(err).slice(0, 24)) {
        if (['name', 'message', 'stack', 'cause', 'error'].includes(String(key))) continue;
        const value = safeRead(err, key);
        const text = safeText(value, 160);
        if (text) parts.push(`${String(key)}=${text}`);
        if (parts.length >= 6) break;
      }
      if (parts.length) detail = parts.join(', ');
    } catch {
      // 代理对象连枚举都可能抛错：后面统一给可读兜底
    }
  }

  if (!detail || detail === '[object Object]') {
    detail = `浏览器引擎抛出了没有文字说明的${safeObjectTag(err)}，请结合下方控制台原始错误排查`;
  }

  if (depth === 0) {
    const cause = safeRead(err, 'cause');
    if (cause && cause !== err) {
      const causeText = describeErrorValue(cause, depth + 1);
      if (causeText && !detail.includes(causeText)) detail = `${detail}；原因：${causeText}`;
    }
  }
  return detail.replace(/\s+/g, ' ').trim().slice(0, 700);
}

/**
 * 生成可跨 WebSocket 回传的结构化错误报告。这个函数本身绝不能抛错：
 * 某些浏览器/WebLLM 错误对象带会抛错的 getter，旧实现若在这里二次抛错，
 * 网页就发不出 error 帧，服务端只能看到超时或通用兜底。
 */
export function bridgeErrorReport(err, context = {}) {
  const model = safeText(context.model, 256) || '未指定';
  const phase = BRIDGE_PHASE_LABELS[context.phase] ? context.phase : 'unknown';
  const phaseLabel = BRIDGE_PHASE_LABELS[phase];
  let detail = '';
  let name = '';
  let code = '';
  let stack = '';
  try {
    detail = describeErrorValue(err) || '浏览器引擎没有给出文字说明';
    name = safeText(safeRead(err, 'name'), 120);
    code = safeText(safeRead(err, 'code') ?? safeRead(err, 'status') ?? safeRead(err, 'statusCode'), 120);
    stack = safeText(safeRead(err, 'stack'), 1200);
  } catch {
    detail = '浏览器引擎抛出了无法读取的错误对象';
  }

  const extras = [];
  if (context.modelDiagnostics?.kind === 'local-import') {
    extras.push(`本地导入模型${context.modelDiagnostics.originId ? `（源模型 ${context.modelDiagnostics.originId}）` : ''}`);
  }
  if (context.readyNote) extras.push(`加载自检：${safeText(context.readyNote, 240)}`);
  if (Number.isFinite(context.deltaCount) && context.deltaCount > 0) {
    extras.push(`失败前已收到 ${context.deltaCount} 个输出片段`);
  }
  if (Number.isFinite(context.elapsedMs) && context.elapsedMs >= 0) {
    extras.push(`已运行 ${Math.round(context.elapsedMs)}ms`);
  }

  const phasePart = context.phase ? `，阶段 ${phaseLabel}` : '';
  const message = `本机模型生成失败（模型 ${model}${phasePart}）：${detail}${extras.length ? `；${extras.join('；')}` : ''}`
    .replace(/\s+/g, ' ')
    .slice(0, 950);
  return {
    message,
    model,
    phase,
    phaseLabel,
    name,
    code,
    detail,
    stack,
    deltaCount: Number.isFinite(context.deltaCount) ? context.deltaCount : 0,
    elapsedMs: Number.isFinite(context.elapsedMs) ? Math.round(context.elapsedMs) : null,
    readyNote: safeText(context.readyNote, 300),
    modelDiagnostics: context.modelDiagnostics || null,
    protocolVersion: BRIDGE_PROTOCOL_VERSION,
  };
}

export function bridgeErrorMessage(err, model, context = {}) {
  return bridgeErrorReport(err, { ...context, model }).message;
}

/** 汇总某个模型在桥接侧可见的来源信息，加载失败时随错误回传，避免只剩模型 ID。 */
export function bridgeModelDiagnostics(webllmCfg, modelId) {
  const local = loadLocalModels().find((entry) => entry.id === modelId);
  if (local) {
    const lib = String(local.record?.model_lib || '');
    return {
      kind: 'local-import',
      originId: String(local.originId || ''),
      modelLib: lib.split('/').pop() || '',
      vramRequiredMB: Number(local.record?.vram_required_MB) || null,
      contextWindowSize: Number(local.record?.overrides?.context_window_size) || null,
    };
  }
  const extra = (webllmCfg?.extraRecords || []).find((record) => record?.model_id === modelId);
  if (extra) {
    let modelHost = '';
    try {
      modelHost = new URL(extra.model).host;
    } catch {
      modelHost = '';
    }
    return {
      kind: 'custom-record',
      modelHost,
      modelLib: String(extra.model_lib || '').split('/').pop() || '',
      vramRequiredMB: Number(extra.vram_required_MB) || null,
      contextWindowSize: Number(extra.overrides?.context_window_size) || null,
    };
  }
  return { kind: (webllmCfg?.models || []).some((m) => m.id === modelId) ? 'builtin' : 'unknown' };
}

/**
 * 把服务端传来的 OpenAI 生成参数（params）映射为本机引擎可用的选项。
 * 服务端已提前拒绝不支持的参数；这里再做一次数值校验，非法值直接抛错回传原因，
 * 不静默丢弃——调用方必须能知道参数到底有没有生效。
 */
export function mapBridgeParams(params) {
  if (!params || typeof params !== 'object') return {};
  const out = {};
  const bad = (key, want) => new Error(`生成参数 ${key} 不合法：需要${want}`);
  if (params.top_p !== undefined && params.top_p !== null) {
    const v = Number(params.top_p);
    if (!Number.isFinite(v) || v <= 0 || v > 1) throw bad('top_p', ' 0 到 1 之间的数字');
    out.topP = v;
  }
  if (params.stop !== undefined && params.stop !== null) {
    const list = (Array.isArray(params.stop) ? params.stop : [params.stop])
      .map((s) => String(s))
      .filter(Boolean)
      .slice(0, 4);
    if (!list.length) throw bad('stop', '非空字符串或字符串数组');
    out.stop = list;
  }
  for (const [key, field] of [['frequency_penalty', 'frequencyPenalty'], ['presence_penalty', 'presencePenalty']]) {
    if (params[key] !== undefined && params[key] !== null) {
      const v = Number(params[key]);
      if (!Number.isFinite(v) || v < -2 || v > 2) throw bad(key, ' -2 到 2 之间的数字');
      out[field] = v;
    }
  }
  if (params.repetition_penalty !== undefined && params.repetition_penalty !== null) {
    const v = Number(params.repetition_penalty);
    if (!Number.isFinite(v) || v <= 0) throw bad('repetition_penalty', '大于 0 的数字');
    out.repetitionPenalty = v;
  }
  if (params.seed !== undefined && params.seed !== null) {
    const v = Number(params.seed);
    if (!Number.isInteger(v)) throw bad('seed', '整数');
    out.seed = v;
  }
  if (params.response_format !== undefined && params.response_format !== null) {
    const type = params.response_format?.type;
    if (type === 'json_object') out.responseFormat = { type: 'json_object' };
    else if (type === 'text') out.responseFormat = { type: 'text' };
    else throw new Error('本机模型仅支持 response_format 的 text 与 json_object 类型');
  }
  return out;
}

export function bridgeModels(webllmCfg) {
  const builtIn = (webllmCfg?.models || []).map((m) => ({ id: m.id, label: m.label || m.id }));
  const local = loadLocalModels().map((e) => ({ id: e.id, label: e.label || e.id }));
  const seen = new Set();
  const models = [];
  for (const m of [...builtIn, ...local]) {
    if (!m.id || seen.has(m.id)) continue;
    seen.add(m.id);
    models.push(m);
  }
  return models;
}

export function startLocalModelBridge({ webllmCfg, onStatus, backendFactory = createBackend } = {}) {
  const backend = backendFactory({ mode: 'webllm', appConfig: { webllm: webllmCfg } });
  let models = bridgeModels(webllmCfg);
  let knownIds = new Set(models.map((m) => m.id));
  const tabId = (() => {
    try {
      return globalThis.crypto?.randomUUID?.() || `tab-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
    } catch {
      return `tab-${Date.now().toString(36)}`;
    }
  })();

  let ws = null;
  let closed = false;
  let superseded = false;
  let reconnectTimer = null;
  let reconnectDelay = RECONNECT_BASE_MS;
  // 引擎同一时间只处理一个生成：请求排队执行
  let queue = Promise.resolve();
  const controllers = new Map();
  // 已被服务端取消、但还在排队没轮到的请求：轮到时直接跳过，不要白占显卡
  const cancelledIds = new Set();
  // 已确认缓存完整的本地导入模型（id -> 登记指纹）。全量枚举缓存不便宜，
  // 同一条登记只查一次；登记变化（导入/删除）时清空重查。
  const verifiedLocalModels = new Map();

  const send = (obj) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(obj));
      } catch {
        // 忽略：断线后由重连逻辑处理
      }
    }
  };

  const setStatus = (connected) => onStatus?.({ connected, models, protocolVersion: BRIDGE_PROTOCOL_VERSION, tabId });

  const readyMessage = () => ({
    type: 'ready',
    protocolVersion: BRIDGE_PROTOCOL_VERSION,
    clientVersion: `bridge-v${BRIDGE_PROTOCOL_VERSION}`,
    tabId,
    pageOrigin: (() => {
      try {
        return globalThis.location?.origin || '';
      } catch {
        return '';
      }
    })(),
    models,
  });

  let lastModelsSignature = '';
  const refreshModels = (announce = true) => {
    models = bridgeModels(webllmCfg);
    knownIds = new Set(models.map((m) => m.id));
    // 模型列表没变时不重复上报，避免每次请求前的刷新都给服务端发一遍 ready
    const signature = models.map((m) => m.id).join('\n');
    if (announce && signature !== lastModelsSignature) send(readyMessage());
    lastModelsSignature = signature;
    setStatus(!!ws && ws.readyState === WebSocket.OPEN);
  };

  async function handleRequest(id, payload) {
    if (cancelledIds.delete(id)) return; // 排队期间已被取消，直接跳过
    const controller = new AbortController();
    controllers.set(id, controller);
    const startedAt = Date.now();
    let model = '';
    let phase = 'validate-model';
    let readyInfo = null;
    let deltaCount = 0;
    let hasVisibleOutput = false;
    let modelDiagnostics = null;
    try {
      refreshModels(false);
      model = payload?.model || models[0]?.id;
      if (!model || !knownIds.has(model)) {
        send({ type: 'error', id, message: `模型不存在或本机网页未提供该模型：${model || '(未指定)'}` });
        return;
      }
      modelDiagnostics = bridgeModelDiagnostics(webllmCfg, model);
      backend.setModel(model);
      const localEntry = loadLocalModels().find((entry) => entry.id === model);
      if (localEntry) {
        const fingerprint = `${localEntry.id}|${localEntry.record?.model || ''}|${localEntry.record?.model_lib || ''}`;
        // 引擎已经装好这个模型时不用再查；同一登记在本次页面会话内也只全量查一次，
        // 避免每次 API 请求都枚举一遍 Cache Storage（模型多时明显拖慢首字）。
        if (verifiedLocalModels.get(model) !== fingerprint && backend.isReady?.() !== true) {
          // 本地导入模型先查缓存完整性：缺分片/配置时直接报清楚，不必等 GPU 加载到一半才失败
          phase = 'check-local-model';
          await collectModelEntries(localEntry.record);
          verifiedLocalModels.set(model, fingerprint);
        }
      }
      phase = 'load-model';
      readyInfo = await backend.ensureReady(() => {}, { signal: controller.signal });
      if (controller.signal.aborted) return;
      const messages = Array.isArray(payload.messages) ? payload.messages : [];
      phase = 'prepare-generation';
      const generation = mapBridgeParams(payload.params);
      phase = 'generate';
      for await (const delta of backend.stream({
        messages,
        signal: controller.signal,
        temperature: payload.temperature,
        maxTokens: payload.maxTokens,
        hardMaxTokens: payload.hardMaxTokens,
        generation,
        systemPrompt: payload.systemPrompt,
      })) {
        if (controller.signal.aborted) break;
        deltaCount += 1;
        if (String(delta || '').trim()) hasVisibleOutput = true;
        send({ type: 'delta', id, content: delta });
      }
      if (!controller.signal.aborted) {
        if (!hasVisibleOutput) {
          // GPU 短暂运行后“正常结束但零输出”不是成功：多半是模型立即 EOS、自检异常或模板不兼容
          const emptyError = new Error(
            '模型生成结束但没有输出任何内容。常见原因是模型立即结束、导入模型与运行库不兼容，或显卡/驱动让引擎提前退出；请先在网页聊天里用同一模型发一句“你好”验证。',
          );
          emptyError.name = 'EmptyModelOutputError';
          throw emptyError;
        }
        send({ type: 'complete', id, finishReason: 'stop' });
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        // 原样保留到浏览器控制台（F12 可见完整对象/堆栈），同时把结构化报告传回服务端与调用方
        console.error('[local-model-bridge] 本机模型生成失败：', err);
        const report = bridgeErrorReport(err, {
          model,
          phase,
          readyNote: readyInfo?.note || '',
          deltaCount,
          elapsedMs: Date.now() - startedAt,
          modelDiagnostics,
        });
        send({ type: 'error', id, message: report.message, error: report });
      }
    } finally {
      controllers.delete(id);
    }
  }

  function connect() {
    if (closed) return;
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${proto}//${location.host}/api/local-model-bridge`);
    ws.onopen = () => {
      reconnectDelay = RECONNECT_BASE_MS;
      send(readyMessage());
      setStatus(true);
    };
    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === 'request' && msg.id) {
        queue = queue.then(() => handleRequest(msg.id, msg.payload)).catch(() => {});
      } else if (msg.type === 'cancel' && msg.id) {
        // 正在生成的立即中断；还在排队的记下来，轮到时跳过
        if (controllers.has(msg.id)) controllers.get(msg.id).abort();
        else {
          if (cancelledIds.size > 256) cancelledIds.clear();
          cancelledIds.add(msg.id);
        }
      }
    };
    const scheduleReconnect = () => {
      setStatus(false);
      if (closed || superseded) return;
      reconnectTimer = setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
    };
    ws.onclose = (event) => {
      // 1012 = 已被另一个标签页接管：不要自动重连抢回去，否则新旧标签页会无限互相顶替
      if (event?.code === 1012) {
        superseded = true;
        setStatus(false);
        console.info('[local-model-bridge] 本标签页已被另一个服务网页接管，停止作为 API 模型提供方。');
        return;
      }
      scheduleReconnect();
    };
    ws.onerror = () => {
      try {
        ws.close();
      } catch {}
    };
  }

  const onLocalModelsChanged = () => {
    verifiedLocalModels.clear();
    refreshModels();
  };
  const onStorage = (event) => {
    if (event.key === null || event.key === 'oj-ai.localModels') {
      verifiedLocalModels.clear();
      refreshModels();
    }
  };
  globalThis.addEventListener?.('oj-ai:local-models-changed', onLocalModelsChanged);
  globalThis.addEventListener?.('storage', onStorage);

  connect();
  setStatus(false);

  return {
    stop() {
      closed = true;
      globalThis.removeEventListener?.('oj-ai:local-models-changed', onLocalModelsChanged);
      globalThis.removeEventListener?.('storage', onStorage);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      try {
        ws?.close();
      } catch {}
    },
  };
}
