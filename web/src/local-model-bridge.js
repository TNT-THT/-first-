/**
 * 本机模型桥接客户端：把运行在本网页里的 WebLLM（WebGPU）引擎
 * 通过 WebSocket 暴露给同机的 Node API 服务（/v1/chat/completions）。
 *
 * 协议（JSON 文本帧）：
 *   → { type: 'ready', models: [{ id, label }] }   连接后上报可用模型（内置 + 本机导入）
 *   ← { type: 'request', id, payload }             payload: { model, messages, temperature, maxTokens }
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

  const refreshModels = (announce = true) => {
    models = bridgeModels(webllmCfg);
    knownIds = new Set(models.map((m) => m.id));
    if (announce) send({ type: 'ready', models });
    setStatus(!!ws && ws.readyState === WebSocket.OPEN);
  };

  async function handleRequest(id, payload) {
    const controller = new AbortController();
    controllers.set(id, controller);
    try {
      refreshModels(false);
      const model = payload?.model || models[0]?.id;
      if (!model || !knownIds.has(model)) {
        send({ type: 'error', id, message: `模型不存在或本机网页未提供该模型：${model || '(未指定)'}` });
        return;
      }
      backend.setModel(model);
      await backend.ensureReady(() => {}, { signal: controller.signal });
      if (controller.signal.aborted) return;
      const messages = Array.isArray(payload.messages) ? payload.messages : [];
      for await (const delta of backend.stream({
        messages,
        signal: controller.signal,
        temperature: payload.temperature,
        maxTokens: payload.maxTokens,
      })) {
        if (controller.signal.aborted) break;
        send({ type: 'delta', id, content: delta });
      }
      if (!controller.signal.aborted) {
        send({ type: 'complete', id, finishReason: 'stop' });
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        send({ type: 'error', id, message: err?.message || '本机模型生成失败' });
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
        controllers.get(msg.id)?.abort();
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
