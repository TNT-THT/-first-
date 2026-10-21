/**
 * 推理引擎生命周期的纯逻辑（便于测试）：
 *   - 识别 GPU 资源被重置/丢失的错误（例如 mapAsync 报 “Instance reference no longer exists”）
 *   - 串行执行任务：加载模型等操作排队，前一个结束后才开始下一个
 */

export const GPU_RESET_MESSAGE = 'GPU 资源已重置，请重新发送。';

// 只匹配 GPU 相关的错误信息。注意：普通的用户中止（AbortError）不能匹配，否则会被误判。
const GPU_RESET_PATTERN =
  /mapAsync|device (was )?lost|context lost|Instance reference|GPUBuffer|GPUDevice|GPU process/i;

/** 是否为 GPU 资源被重置/丢失导致的错误。 */
export function isGpuResetError(err) {
  const text = `${err?.message ?? ''} ${String(err ?? '')}`;
  return GPU_RESET_PATTERN.test(text);
}

/** 包装成统一的错误：name 不是 AbortError，界面不会把它当作“用户停止”。 */
export function gpuResetError() {
  const e = new Error(GPU_RESET_MESSAGE);
  e.name = 'GpuResetError';
  return e;
}

/**
 * 串行队列：run(task) 返回 task 的结果；前一个任务（无论成功或失败）结束后才会执行下一个。
 */
export function createSerialQueue() {
  let tail = Promise.resolve();
  return (task) => {
    const run = tail.then(() => task());
    tail = run.catch(() => {});
    return run;
  };
}

/* ---------- 超时与取消（纯逻辑，便于测试） ---------- */

/** 模型加载超过这段时间没有任何进展（没有新的进度回调），视为卡住：取消加载并释放引擎。 */
export const LOAD_STALL_MS = 180_000;
/** 生成：从开始到第一段输出的最长等待（长提示词的预填可能较慢）。 */
export const STREAM_FIRST_TOKEN_MS = 180_000;
/** 生成：两段输出之间的最长等待。超过则视为卡住，释放引擎。 */
export const STREAM_IDLE_MS = 90_000;

/** 超时错误（name 为 TimeoutError，界面据此提示“请重试”，而不是当作用户停止）。 */
export class TimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = 'TimeoutError';
  }
}

export function isTimeoutError(err) {
  return err?.name === 'TimeoutError';
}

/** 与 fetch 的中止错误保持一致，界面会把它当作“用户停止”。 */
export function abortError() {
  const e = new Error('已停止');
  e.name = 'AbortError';
  return e;
}

/**
 * 在 promise 与超时之间竞速。超时后 reject TimeoutError；promise 本身不会被取消，
 * 调用方需要自行释放对应的资源（例如终止 worker）。
 */
export function withTimeout(promise, ms, message) {
  let timer = null;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new TimeoutError(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/**
 * 停滞监视：超过 stallMs 没有调用 touch() 则调用 onStall（只触发一次）。
 * 返回 {touch, stop}；完成后必须调用 stop()。
 */
export function createStallGuard({ stallMs, checkMs = 1000, message, onStall }) {
  let last = Date.now();
  let done = false;
  const timer = setInterval(() => {
    if (done || Date.now() - last <= stallMs) return;
    done = true;
    clearInterval(timer);
    onStall(new TimeoutError(message));
  }, checkMs);
  return {
    touch() {
      last = Date.now();
    },
    stop() {
      done = true;
      clearInterval(timer);
    },
  };
}

/**
 * 逐块读取异步可迭代对象：第一块最多等待 firstMs，之后每块最多等待 idleMs。
 * 超时时先调用 onTimeout（用于中断与释放引擎），再抛出 TimeoutError（消息分别为 firstMessage / idleMessage）。
 */
export async function* iterateWithTimeout(iterable, { firstMs, idleMs, firstMessage, idleMessage, onTimeout }) {
  const it = iterable[Symbol.asyncIterator]();
  let first = true;
  for (;;) {
    let r;
    try {
      r = await withTimeout(it.next(), first ? firstMs : idleMs, first ? firstMessage : idleMessage);
    } catch (err) {
      if (isTimeoutError(err)) onTimeout?.(err);
      throw err;
    }
    first = false;
    if (r.done) return;
    yield r.value;
  }
}
