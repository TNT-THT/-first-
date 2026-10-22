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

/* ---------- 取消（只由用户主动触发，不设时限） ---------- */

/** 与 fetch 的中止错误保持一致，界面会把它当作“用户停止”。 */
export function abortError() {
  const e = new Error('已停止');
  e.name = 'AbortError';
  return e;
}

/**
 * 进行中的生成的门闩。生成开始时 begin() 返回一个令牌，结束时 end(令牌)。
 * pending：当前未结束的生成（用于加载或释放前等待）。
 * releaseAll()：引擎被丢弃（worker 已终止）时调用，使等待者立即继续；
 *   被终止的生成永远不会自行结束，如果不在这里结束，之后的加载与释放都会永远等待。
 */
export function createInflightGate() {
  let current = null;
  return {
    begin() {
      let resolve = () => {};
      const promise = new Promise((r) => {
        resolve = r;
      });
      current = { promise, resolve };
      return current;
    },
    get pending() {
      return current ? current.promise : null;
    },
    end(token) {
      token?.resolve();
      if (current === token) current = null;
    },
    releaseAll() {
      if (current) current.resolve();
      current = null;
    },
  };
}
