import { randomUUID } from 'node:crypto';
import { WebSocketServer, WebSocket } from 'ws';

const MAX_QUEUE = 8;
const REQUEST_TIMEOUT_MS = 10 * 60 * 1000;

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
      const origin = req.headers.origin;
      const expected = `http://${req.headers.host}`;
      if (!origin || origin !== expected) {
        socket.write('HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n');
        socket.destroy();
        return;
      }
      this.wss.handleUpgrade(req, socket, head, (ws) => this.wss.emit('connection', ws, req));
    });
    this.wss.on('connection', (ws) => {
      if (this.client && this.client.readyState === WebSocket.OPEN) this.client.close(1012, 'replaced');
      this.client = ws;
      this.models = [];
      ws.on('message', (raw) => this.#message(ws, raw));
      ws.on('close', () => {
        if (this.client === ws) {
          this.client = null;
          this.models = [];
          for (const job of this.pending.values()) job.fail(new Error('本机网页与 API 桥接已断开，请保持网页打开。'));
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
      const timer = setTimeout(
        () => finish(Object.assign(new Error('本机模型生成超时，请检查模型状态后重试。'), { status: 504 })),
        REQUEST_TIMEOUT_MS,
      );
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
      job.fail(String(message.message || '本机模型生成失败').slice(0, 1000));
    }
  }
}
