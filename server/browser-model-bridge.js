import { randomUUID } from 'node:crypto';
import { WebSocketServer, WebSocket } from 'ws';

const MAX_QUEUE = 8;
const REQUEST_TIMEOUT_MS = 10 * 60 * 1000;
/** 当前网站包的桥接协议版本；上报版本低于它的网页属于旧版页面（生成失败只回通用文案）。 */
export const LATEST_BRIDGE_PROTOCOL_VERSION = 2;

/** 本机 WebLLM 引擎真正支持的 OpenAI 生成参数；其余参数在服务端提前拒绝并说明，不静默丢弃。 */
export const BRIDGE_SUPPORTED_GENERATION_PARAMS = [
  'top_p',
  'stop',
  'frequency_penalty',
  'presence_penalty',
  'repetition_penalty',
  'seed',
  'response_format',
];

/** Routes OpenAI-compatible requests to the WebGPU engine running in the local browser tab. */
function cleanBridgeText(value, max = 1000) {
  return String(value ?? '').replace(/\s+/g, ' ').trim().slice(0, max);
}

function sanitizeModels(models) {
  return Array.isArray(models)
    ? models.filter((m) => m && typeof m.id === 'string').slice(0, 256).map((m) => ({
        id: m.id.slice(0, 256),
        label: String(m.label || m.id).slice(0, 256),
      }))
    : [];
}

function sanitizeBridgeErrorReport(input, fallbackModel = '') {
  if (!input || typeof input !== 'object') return null;
  const report = {
    message: cleanBridgeText(input.message, 1000),
    model: cleanBridgeText(input.model, 256) || fallbackModel,
    phase: cleanBridgeText(input.phase, 80),
    phaseLabel: cleanBridgeText(input.phaseLabel, 80),
    name: cleanBridgeText(input.name, 120),
    code: cleanBridgeText(input.code, 120),
    detail: cleanBridgeText(input.detail, 700),
    stack: cleanBridgeText(input.stack, 1200),
    deltaCount: Number.isFinite(input.deltaCount) ? input.deltaCount : 0,
    elapsedMs: Number.isFinite(input.elapsedMs) ? input.elapsedMs : null,
    readyNote: cleanBridgeText(input.readyNote, 300),
    protocolVersion: Number.isFinite(input.protocolVersion) ? input.protocolVersion : 0,
  };
  return report.message || report.detail ? report : null;
}

export class BrowserModelBridge {
  constructor() {
    this.client = null;
    this.clients = new Map();
    this.clientSequence = 0;
    this.models = [];
    this.pending = new Map();
    this.queue = Promise.resolve();
    this.waiting = 0;
    this.lastError = null;
    this.wss = new WebSocketServer({ noServer: true, maxPayload: 1024 * 1024 });
  }

  attach(server) {
    server.on('upgrade', (req, socket, head) => {
      let pathname;
      try {
        pathname = new URL(req.url, `http://${req.headers.host || 'localhost'}`).pathname;
      } catch {
        socket.destroy();
        return;
      }
      if (pathname !== '/api/local-model-bridge') return;
      // 只校验 Origin 的主机部分与请求 Host 一致（协议 http/https 取决于部署方式，
      // 经 HTTPS 反向代理时 Origin 是 https，写死 http 会永远对不上）。
      let originHost = '';
      try {
        originHost = new URL(req.headers.origin || '').host;
      } catch {
        /* 无效 Origin 按拒绝处理 */
      }
      if (!originHost || originHost !== req.headers.host) {
        socket.write('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');
        socket.destroy();
        return;
      }
      this.wss.handleUpgrade(req, socket, head, (ws) => this.wss.emit('connection', ws, req));
    });
    this.wss.on('connection', (ws) => {
      // 先登记为候选连接，不立刻顶替当前提供方；等它发来 ready（带协议版本）后再决定。
      // 旧版标签页不会上报协议版本，不能抢走新版标签页的提供方位置；
      // 被顶替的连接也不关闭（保留为备用），否则旧页会自动重连、与新页无限互相顶替。
      const info = {
        order: ++this.clientSequence,
        ready: false,
        readyCount: 0,
        protocolVersion: 0,
        clientVersion: '',
        tabId: '',
        pageOrigin: '',
        models: [],
      };
      this.clients.set(ws, info);
      ws.on('message', (raw) => this.#message(ws, raw));
      ws.on('close', () => {
        this.clients.delete(ws);
        const wasActive = this.client === ws;
        if (wasActive) {
          this.client = null;
          this.models = [];
        }
        // 该连接名下若还有未完成请求，立即失败，不要挂到超时
        for (const job of this.pending.values()) {
          if (job.client === ws) job.fail('本机网页与 API 桥接已断开，请保持网页打开。');
        }
        if (wasActive) this.#promoteBestStandby();
      });
      ws.on('error', () => {});
    });
    return this;
  }

  #promote(ws, { failPrevious = true } = {}) {
    const info = this.clients.get(ws);
    if (!info?.ready || ws.readyState !== WebSocket.OPEN) return false;
    const previous = this.client;
    if (previous && previous !== ws && failPrevious) {
      for (const job of this.pending.values()) {
        if (job.client === previous) job.fail('本机网页连接已被新的网页接管，请重试。');
      }
    }
    this.client = ws;
    this.models = [...info.models];
    return true;
  }

  #promoteBestStandby() {
    const candidates = [...this.clients.entries()]
      .filter(([ws, info]) => ws.readyState === WebSocket.OPEN && info.ready)
      .sort((a, b) => (b[1].protocolVersion - a[1].protocolVersion) || (b[1].order - a[1].order));
    if (candidates.length) this.#promote(candidates[0][0], { failPrevious: false });
  }

  /**
   * 若某个备用连接的协议比当前提供方更新，立即改由它提供。
   * 正常情况下新版页面的 ready 会当场接管；这里是每次请求前的兜底，
   * 防止任何时序下请求仍被发给旧版页面（它的失败只有一句通用文案）。
   */
  #promoteNewerStandby() {
    const activeInfo = this.client ? this.clients.get(this.client) : null;
    if (!activeInfo) return;
    const newer = [...this.clients.entries()]
      .filter(([ws, info]) => ws !== this.client && ws.readyState === WebSocket.OPEN && info.ready && info.protocolVersion > activeInfo.protocolVersion)
      .sort((a, b) => b[1].order - a[1].order);
    if (newer.length) this.#promote(newer[0][0]);
  }

  /**
   * 新版页面在场时，把旧版备用连接停放（1012 关闭）：旧版页面继续挂着只会
   * 在新版页面关闭后顶上来，并以一句通用文案吞掉真实失败原因。旧版页面会
   * 按它自己的退避自动重连；只要新版还在，它每次都会被停放，不再当提供方。
   * 同协议的备用连接保留，用于新版页面关闭时立即顶上。
   */
  #parkOutdatedStandbys() {
    const activeInfo = this.client ? this.clients.get(this.client) : null;
    if (!activeInfo || activeInfo.protocolVersion < LATEST_BRIDGE_PROTOCOL_VERSION) return;
    for (const [ws, info] of this.clients) {
      if (ws === this.client || !info.ready) continue;
      if (info.protocolVersion < activeInfo.protocolVersion && ws.readyState === WebSocket.OPEN) {
        try { ws.close(1012, 'superseded by newer bridge protocol'); } catch { /* 已断开则忽略 */ }
      }
    }
  }

  status() {
    const info = this.client ? this.clients.get(this.client) : null;
    return {
      connected: !!this.client && this.client.readyState === WebSocket.OPEN,
      models: [...this.models],
      queued: this.waiting,
      clientCount: [...this.clients.keys()].filter((ws) => ws.readyState === WebSocket.OPEN).length,
      clientProtocolVersion: info?.protocolVersion || 0,
      latestProtocolVersion: LATEST_BRIDGE_PROTOCOL_VERSION,
      clientVersion: info?.clientVersion || '',
      activeTabId: info?.tabId || '',
      activePageOrigin: info?.pageOrigin || '',
      lastError: this.lastError ? { ...this.lastError } : null,
    };
  }

  async run(payload, { signal, onDelta } = {}) {
    this.#promoteNewerStandby();
    if (!this.client || this.client.readyState !== WebSocket.OPEN) {
      const error = new Error('本机模型网页尚未连接。请在这台电脑打开服务网页，并保持网页打开。');
      error.status = 503;
      throw error;
    }
    if (this.waiting >= MAX_QUEUE) {
      const error = new Error('本机模型请求队列已满，请稍后重试。');
      error.status = 429;
      throw error;
    }
    this.waiting += 1;
    const operation = this.queue.then(() => this.#runOne(payload, { signal, onDelta }));
    this.queue = operation.catch(() => {});
    try {
      return await operation;
    } finally {
      this.waiting -= 1;
    }
  }

  #runOne(payload, { signal, onDelta }) {
    const client = this.client;
    if (!client || client.readyState !== WebSocket.OPEN) {
      const error = new Error('本机模型网页尚未连接。请重新打开服务网页。');
      error.status = 503;
      return Promise.reject(error);
    }
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        // 超时先通知浏览器取消生成，避免它继续占着显卡把这条请求跑完、拖住后续排队请求
        try {
          client.send(JSON.stringify({ type: 'cancel', id }));
        } catch {}
        finish(Object.assign(new Error('本机模型生成超时，请检查模型状态后重试。'), { status: 504 }));
      }, REQUEST_TIMEOUT_MS);
      const onAbort = () => {
        try { client.send(JSON.stringify({ type: 'cancel', id })); } catch {}
        finish(Object.assign(new Error('请求已取消。'), { name: 'AbortError', status: 499 }));
      };
      const cleanup = () => {
        clearTimeout(timer);
        signal?.removeEventListener('abort', onAbort);
        this.pending.delete(id);
      };
      const finish = (error, value) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (error) reject(error);
        else resolve(value);
      };
      if (signal?.aborted) return onAbort();
      signal?.addEventListener('abort', onAbort, { once: true });
      this.pending.set(id, {
        client,
        model: payload?.model || '',
        delta: (content) => onDelta?.(content),
        complete: (value) => finish(null, value),
        fail: (message, report = null) => {
          const error = Object.assign(new Error(message), { status: 503 });
          if (report) error.bridgeReport = report;
          finish(error);
        },
      });
      try {
        client.send(JSON.stringify({ type: 'request', id, payload }));
      } catch (error) {
        finish(error);
      }
    });
  }

  #message(ws, raw) {
    const info = this.clients.get(ws);
    if (!info) return;
    let message;
    try { message = JSON.parse(String(raw)); } catch { return; }
    // 畸形帧（如字面量 null）直接忽略，绝不能让一条坏消息打崩整个服务进程
    if (!message || typeof message !== 'object') return;
    if (message.type === 'ready') {
      info.ready = true;
      info.readyCount += 1;
      info.protocolVersion = Number.isFinite(message.protocolVersion) ? message.protocolVersion : 0;
      info.clientVersion = cleanBridgeText(message.clientVersion, 80);
      info.tabId = cleanBridgeText(message.tabId, 120);
      info.pageOrigin = cleanBridgeText(message.pageOrigin, 256);
      info.models = sanitizeModels(message.models);
      if (ws === this.client) {
        this.models = [...info.models];
      } else if (!this.client) {
        this.#promote(ws, { failPrevious: false });
      } else {
        const activeInfo = this.clients.get(this.client);
        const firstReady = info.readyCount === 1;
        // 新版优先；同版本只在候选连接第一次上报时按“新打开的标签页优先”接管，
        // 后续仅刷新模型列表的 ready 不能反复抢占，避免多标签页互相顶替。
        if (info.protocolVersion > (activeInfo?.protocolVersion || 0)) this.#promote(ws);
        else if (firstReady && info.protocolVersion === (activeInfo?.protocolVersion || 0)) this.#promote(ws);
      }
      this.#parkOutdatedStandbys();
      return;
    }
    if (ws !== this.client) return;
    const job = this.pending.get(message.id);
    if (!job) return;
    if (message.type === 'delta' && typeof message.content === 'string') {
      try { job.delta(message.content.slice(0, 65536)); } catch {}
    } else if (message.type === 'complete') {
      job.complete({ finishReason: String(message.finishReason || 'stop').slice(0, 40) });
    } else if (message.type === 'error') {
      const activeInfo = this.clients.get(ws);
      const report = sanitizeBridgeErrorReport(message.error, job.model);
      const rawText = cleanBridgeText(message.message, 1000);
      let text = report?.message || rawText;
      if (!text || text === '本机模型生成失败') {
        text = (activeInfo?.protocolVersion || 0) >= 2
          ? `本机模型生成失败（模型 ${job.model || '未指定'}）：新版模型网页没有提供结构化诊断，原始回传为“${rawText || '空'}”。请刷新服务网页后重试；若仍复现，请把 API 面板中的最近错误反馈给维护者。`
          : `本机模型生成失败（模型 ${job.model || '未指定'}）：当前提供 API 的是未上报诊断信息的旧版模型网页，它只回传了通用失败文案。请关闭所有旧的服务网页标签页，只保留从最新版网站包打开的服务网页，并按 Ctrl+F5 刷新后再试。`;
      }
      const lastError = {
        at: new Date().toISOString(),
        requestId: String(message.id || '').slice(0, 80),
        model: job.model || report?.model || '',
        phase: report?.phase || '',
        phaseLabel: report?.phaseLabel || '',
        name: report?.name || '',
        code: report?.code || '',
        message: text,
        detail: report?.detail || '',
        readyNote: report?.readyNote || '',
        deltaCount: report?.deltaCount || 0,
        elapsedMs: report?.elapsedMs ?? null,
        modelDiagnostics: message.error?.modelDiagnostics && typeof message.error.modelDiagnostics === 'object'
          ? message.error.modelDiagnostics
          : null,
        clientProtocolVersion: activeInfo?.protocolVersion || 0,
        clientVersion: activeInfo?.clientVersion || '',
        tabId: activeInfo?.tabId || '',
        pageOrigin: activeInfo?.pageOrigin || '',
      };
      this.lastError = lastError;
      job.fail(text, { ...report, message: text, model: lastError.model, protocolVersion: lastError.clientProtocolVersion });
    }
  }
}
