import { randomUUID } from 'node:crypto';
import { WebSocketServer, WebSocket } from 'ws';

const MAX_QUEUE = 8;
const REQUEST_TIMEOUT_MS = 10 * 60 * 1000;

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
export class BrowserModelBridge {
  constructor() {
    this.client = null;
    this.models = [];
    this.pending = new Map();
    this.queue = Promise.resolve();
    this.waiting = 0;
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
      const replaced = this.client;
      if (replaced && replaced !== ws && replaced.readyState === WebSocket.OPEN) {
        // 新连接接管前，先让旧连接名下未完成的请求立即失败，
        // 否则它们会无人应答、一直挂到 10 分钟超时。
        for (const job of this.pending.values()) {
          if (job.client === replaced) job.fail('本机网页连接已被新的网页接管，请重试。');
        }
        replaced.close(1012, 'replaced');
      }
      this.client = ws;
      this.models = [];
      ws.on('message', (raw) => this.#message(ws, raw));
      ws.on('close', () => {
        if (this.client === ws) {
          this.client = null;
          this.models = [];
        }
        // 该连接名下若还有未完成请求（无论它是否仍是当前连接），立即失败，不要挂到超时
        for (const job of this.pending.values()) {
          if (job.client === ws) job.fail('本机网页与 API 桥接已断开，请保持网页打开。');
        }
      });
      ws.on('error', () => {});
    });
    return this;
  }

  status() {
    return {
      connected: !!this.client && this.client.readyState === WebSocket.OPEN,
      models: [...this.models],
      queued: this.waiting,
    };
  }

  async run(payload, { signal, onDelta } = {}) {
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
        fail: (message) => finish(Object.assign(new Error(message), { status: 503 })),
      });
      try {
        client.send(JSON.stringify({ type: 'request', id, payload }));
      } catch (error) {
        finish(error);
      }
    });
  }

  #message(ws, raw) {
    if (ws !== this.client) return;
    let message;
    try { message = JSON.parse(String(raw)); } catch { return; }
    // 畸形帧（如字面量 null）直接忽略，绝不能让一条坏消息打崩整个服务进程
    if (!message || typeof message !== 'object') return;
    if (message.type === 'ready') {
      this.models = Array.isArray(message.models)
        ? message.models.filter((m) => m && typeof m.id === 'string').slice(0, 256).map((m) => ({
            id: m.id.slice(0, 256),
            label: String(m.label || m.id).slice(0, 256),
          }))
        : [];
      return;
    }
    const job = this.pending.get(message.id);
    if (!job) return;
    if (message.type === 'delta' && typeof message.content === 'string') {
      try { job.delta(message.content.slice(0, 65536)); } catch {}
    } else if (message.type === 'complete') {
      job.complete({ finishReason: String(message.finishReason || 'stop').slice(0, 40) });
    } else if (message.type === 'error') {
      const raw = String(message.message || '').trim();
      // 旧版网页只会回一句通用文案：补上模型与排查指引，避免真实原因被兜底吞掉
      job.fail(
        raw && raw !== '本机模型生成失败'
          ? raw.slice(0, 1000)
          : `本机模型生成失败（模型 ${job.model || '未指定'}）：网页未提供具体原因，请按 F12 查看浏览器控制台中的 [local-model-bridge] 日志，并确认使用的是最新版网站包。`,
      );
    }
  }
}
