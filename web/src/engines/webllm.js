/**
 * 浏览器内推理（WebLLM / WebGPU）。模型权重从 Hugging Face 下载一次后缓存在浏览器中，
 * 之后的对话完全在本机完成，不需要服务器，也不需要 API Key。
 */
export const WEBGPU_MISSING =
  '当前浏览器不支持 WebGPU，无法运行本地模型。请使用最新版 Chrome 或 Edge（桌面端），并确认显卡驱动正常。';

let engine = null;
let loadedId = null;

let gpuCheck = null;

/**
 * 真正探测 WebGPU 是否可用：存在 navigator.gpu 还不够，
 * 驱动被禁用或显卡不支持时 requestAdapter() 会返回 null。结果缓存，只探测一次。
 */
export function checkWebGPU() {
  if (!gpuCheck) {
    gpuCheck = (async () => {
      if (typeof navigator === 'undefined' || !navigator.gpu) return false;
      try {
        return !!(await navigator.gpu.requestAdapter());
      } catch {
        return false;
      }
    })();
  }
  return gpuCheck;
}

export function getLoadedModelId() {
  return loadedId;
}

export async function loadWebLLMEngine(modelId, onProgress) {
  if (!(await checkWebGPU())) {
    throw new Error(WEBGPU_MISSING);
  }
  const progress = (r) => onProgress?.({ progress: r.progress ?? 0, text: r.text || '' });
  // 动态导入：避免把较大的推理库打进首屏
  const { CreateWebWorkerMLCEngine } = await import('@mlc-ai/web-llm');

  loadedId = null;
  if (!engine) {
    const worker = new Worker(new URL('../llm-worker.js', import.meta.url), { type: 'module' });
    engine = await CreateWebWorkerMLCEngine(worker, modelId, { initProgressCallback: progress });
  } else {
    engine.setInitProgressCallback(progress);
    await engine.reload(modelId);
  }
  loadedId = modelId;
}

export async function* streamWebLLM({ messages, signal, temperature, maxTokens }) {
  if (!engine || !loadedId) throw new Error('模型尚未加载');
  const onAbort = () => {
    engine.interruptGenerate().catch(() => {});
  };
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    const stream = await engine.chat.completions.create({
      messages,
      stream: true,
      temperature,
      max_tokens: maxTokens,
    });
    for await (const chunk of stream) {
      if (signal?.aborted) break;
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) yield delta;
    }
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
}
