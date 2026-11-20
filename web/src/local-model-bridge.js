/**
 * 本机模型桥接客户端：把运行在本网页里的 WebLLM（WebGPU）引擎
 * 通过 WebSocket 暴露给同机的 Node API 服务（/v1/chat/completions）。
 *
 * 协议（JSON 文本帧）：
 *   → { type: 'ready', models: [{ id, label }] }   连接后上报可用模型（内置 + 本机导入）
 *   ← { type: 'request', id, payload }             payload: { model, messages, temperature, maxTokens, hardMaxTokens, params, systemPrompt }
 *   → { type: 'delta', id, content }               流式片段
 *   → { type: 'complete', id, finishReason }
 *   → { type: 'error', id, message }
 *   ← { type: 'cancel', id }                       中断生成
 *
 * 请求按顺序执行（引擎同一时间只处理一个生成）；断线后自动重连并重新上报模型列表。
 */
import { createBackend } from './engines/backend.js';
import { loadLocalModels } from './local-models.js';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 10000;

/**
 * 把引擎抛出的各种错误（Error、DOMException、字符串、WebLLM 的普通对象）
 * 整理成一句能跨 WebSocket 传回服务端的说明。此前这里只取 err.message，
 * 拿不到时就只剩一句“本机模型生成失败”，真实原因（如 WebGPU 丢失、显存不足、
 * 模型下载失败）全被吞掉，服务端日志和调用方都无从排查。
 */
export function bridgeErrorMessage(err, model) {
  const name = err?.name ? String(err.name) : '';
  const message = err?.message ? String(err.message) : '';
  let detail = [name && name !== 'Error' ? name : '', message].filter(Boolean).join(': ');
  if (!detail && err?.stack) detail = String(err.stack).split('\n')[0] || '';
  if (!detail && typeof err === 'string') detail = err;
  if (!detail) {
    try {
      const json = JSON.stringify(err);
      if (json && json !== '{}' && json !== 'null') detail = json;
    } catch {
      // 循环引用等无法序列化的对象：落到 String(err)
    }
  }
  if (!detail || detail === '{}') detail = String(err ?? '未知错误');
  detail = detail.replace(/\s+/g, ' ').slice(0, 700);
  return `本机模型生成失败（模型 ${model || '未指定'}）：${detail}`;
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

export function startLocalModelBridge({ webllmCfg, onStatus } = {}) {
  const backend = createBackend({ mode: 'webllm', appConfig: { webllm: webllmCfg } });
  let models = bridgeModels(webllmCfg);
  let knownIds = new Set(models.map((m) => m.id));

  let ws = null;
  let closed = false;
  let reconnectTimer = null;
  let reconnectDelay = RECONNECT_BASE_MS;
  // 引擎同一时间只处理一个生成：请求排队执行
  let queue = Promise.resolve();
  const controllers = new Map();
  // 已被服务端取消、但还在排队没轮到的请求：轮到时直接跳过，不要白占显卡
  const cancelledIds = new Set();

  const send = (obj) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(obj));
      } catch {
        // 忽略：断线后由重连逻辑处理
      }
    }
  };

  const setStatus = (connected) => onStatus?.({ connected, models });

  let lastModelsSignature = '';
  const refreshModels = (announce = true) => {
    models = bridgeModels(webllmCfg);
    knownIds = new Set(models.map((m) => m.id));
    // 模型列表没变时不重复上报，避免每次请求前的刷新都给服务端发一遍 ready
    const signature = models.map((m) => m.id).join('\n');
    if (announce && signature !== lastModelsSignature) send({ type: 'ready', models });
    lastModelsSignature = signature;
    setStatus(!!ws && ws.readyState === WebSocket.OPEN);
  };

  async function handleRequest(id, payload) {
    if (cancelledIds.delete(id)) return; // 排队期间已被取消，直接跳过
    const controller = new AbortController();
    controllers.set(id, controller);
    let model = '';
    try {
      refreshModels(false);
      model = payload?.model || models[0]?.id;
      if (!model || !knownIds.has(model)) {
        send({ type: 'error', id, message: `模型不存在或本机网页未提供该模型：${model || '(未指定)'}` });
        return;
      }
      backend.setModel(model);
      await backend.ensureReady(() => {}, { signal: controller.signal });
      if (controller.signal.aborted) return;
      const messages = Array.isArray(payload.messages) ? payload.messages : [];
      const generation = mapBridgeParams(payload.params);
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
        send({ type: 'delta', id, content: delta });
      }
      if (!controller.signal.aborted) {
        send({ type: 'complete', id, finishReason: 'stop' });
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        // 原样保留到浏览器控制台（F12 可见完整对象/堆栈），同时把可读原因传回服务端与调用方
        console.error('[local-model-bridge] 本机模型生成失败：', err);
        send({ type: 'error', id, message: bridgeErrorMessage(err, model) });
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
      send({ type: 'ready', models });
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
      if (closed) return;
      reconnectTimer = setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
    };
    ws.onclose = scheduleReconnect;
    ws.onerror = () => {
      try {
        ws.close();
      } catch {}
    };
  }

  const onLocalModelsChanged = () => refreshModels();
  const onStorage = (event) => {
    if (event.key === null || event.key === 'oj-ai.localModels') refreshModels();
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
