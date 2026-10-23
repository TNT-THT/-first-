/**
 * 生图客户端（主线程）：管理生图 worker 的生命周期。
 * - 第一次生成时才创建 worker 并加载模型（启动页面时不下载、不加载）；
 * - 生成结束后 worker 保持运行，便于连续生成；离开生图模式或取消时终止 worker，释放显存；
 * - 取消是“终止 worker”：不尝试中断 GPU 上正在进行的计算。
 */

export function abortError() {
  const err = new Error('已取消生图');
  err.name = 'AbortError';
  return err;
}

/**
 * @param {object} p
 * @param {() => Worker} p.createWorker 创建 worker 的函数（测试中可以注入假的 worker）
 */
export function createImageClient({ createWorker }) {
  let worker = null;
  let nextId = 1;
  const pending = new Map();

  function onMessage(msg) {
    const job = pending.get(msg.id);
    if (!job) return;
    if (msg.type === 'progress') {
      job.onProgress?.(msg);
      return;
    }
    pending.delete(msg.id);
    if (msg.type === 'error') job.reject(new Error(msg.message));
    else job.resolve(msg);
  }

  function ensureWorker() {
    if (!worker) {
      worker = createWorker();
      worker.onmessage = (e) => onMessage(e.data);
      worker.onerror = (e) => {
        const message = e?.message || '生图 worker 出错';
        failAll(new Error(message));
        dispose();
      };
    }
    return worker;
  }

  function failAll(err) {
    for (const job of pending.values()) job.reject(err);
    pending.clear();
  }

  return {
    /**
     * 生成一张图。request: {prompt, negative?, width, height, steps, guidance, seed}；
     * urls: 模型文件地址（见 catalog.sourceUrls）。返回 {width, height, rgba}。
     */
    generate(request, { urls, onProgress, signal } = {}) {
      if (signal?.aborted) return Promise.reject(abortError());
      const id = nextId;
      nextId += 1;
      return new Promise((resolve, reject) => {
        const onAbort = () => {
          // 终止 worker（即释放其中的 GPU 资源），并拒绝本次请求
          dispose();
        };
        signal?.addEventListener('abort', onAbort, { once: true });
        pending.set(id, {
          onProgress,
          resolve: (msg) => {
            signal?.removeEventListener('abort', onAbort);
            resolve(msg);
          },
          reject: (err) => {
            signal?.removeEventListener('abort', onAbort);
            reject(signal?.aborted ? abortError() : err);
          },
        });
        try {
          ensureWorker().postMessage({ id, type: 'generate', urls, request });
        } catch (err) {
          pending.delete(id);
          reject(err);
        }
      });
    },
    /** 终止 worker（释放显存）。正在进行的请求以 AbortError 结束。 */
    dispose,
    /** worker 是否在运行（用于界面提示）。 */
    get running() {
      return worker !== null;
    },
  };

  function dispose() {
    const w = worker;
    worker = null;
    try {
      w?.terminate();
    } catch {
      // 忽略：worker 可能已经结束
    }
    failAll(abortError());
  }
}
