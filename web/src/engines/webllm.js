/**
 * 浏览器内推理（WebLLM / WebGPU）。模型权重从 Hugging Face 下载一次后缓存在浏览器中，
 * 之后的对话完全在本机完成，不需要服务器，也不需要 API Key。
 */
import {
  applyContextWindow,
  buildModelList,
  contextWindowLadder,
  DEFAULT_CONTEXT_WINDOW,
  isMemoryError,
} from '../model-source.js';

export const WEBGPU_MISSING =
  '当前浏览器不支持 WebGPU，无法运行本地模型。请使用最新版 Chrome 或 Edge（桌面端），并确认显卡驱动正常。';

let engine = null;
let worker = null;
let loadedId = null;
let loadedWindow = null;

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

/**
 * 某个模型实际生效的上下文窗口（token）。只对当前已加载的模型有效；其他模型返回 null。
 * 显存不足降级时，这里返回降级后的窗口。
 */
export function getLoadedContextWindow(modelId) {
  return loadedId && loadedId === modelId ? loadedWindow : null;
}

/**
 * 当前生效的完整模型记录（内置列表 + 自定义记录 + 镜像设置），用于导出 / 导入 / 缓存状态检查。
 * 动态导入推理库，避免首屏变重。
 */
export async function getModelRecords({ mirrorOwner, extraRecords } = {}) {
  const { prebuiltAppConfig } = await import('@mlc-ai/web-llm');
  return buildModelList(prebuiltAppConfig.model_list, extraRecords, mirrorOwner);
}

/** 创建或复用引擎并加载模型（appConfig 中已包含本次要使用的窗口）。 */
async function loadWithConfig(modelId, appConfig, progress) {
  // 动态导入：避免把较大的推理库打进首屏
  const { CreateWebWorkerMLCEngine } = await import('@mlc-ai/web-llm');
  if (!engine) {
    worker = new Worker(new URL('../llm-worker.js', import.meta.url), { type: 'module' });
    engine = await CreateWebWorkerMLCEngine(worker, modelId, { initProgressCallback: progress, appConfig });
  } else {
    engine.setInitProgressCallback(progress);
    engine.setAppConfig(appConfig);
    await engine.reload(modelId);
  }
}

/** 加载失败后彻底释放引擎与 Worker，下一次加载从干净的状态开始。 */
async function resetEngine() {
  const e = engine;
  const w = worker;
  engine = null;
  worker = null;
  try {
    await e?.unload?.();
  } catch {
    /* 忽略：引擎可能已经损坏 */
  }
  try {
    w?.terminate();
  } catch {
    /* ignore */
  }
}

/**
 * 加载模型，并把上下文窗口设为 contextWindowTokens（默认 65,536）。
 * 若因显存不足失败，依次改用更小的窗口重试（65,536 → 32,768 → … → 4,096）。
 * 返回 {contextWindowTokens, requestedTokens, reduced}，reduced 为 true 表示窗口已被降低。
 */
export async function loadWebLLMEngine(
  modelId,
  onProgress,
  { mirrorOwner, extraRecords, contextWindowTokens = DEFAULT_CONTEXT_WINDOW } = {},
) {
  if (!(await checkWebGPU())) {
    throw new Error(WEBGPU_MISSING);
  }
  const progress = (r) => onProgress?.({ progress: r.progress ?? 0, text: r.text || '' });
  const { prebuiltAppConfig } = await import('@mlc-ai/web-llm');
  // 合并自定义记录；配置了 mirrorOwner 时权重从自己的 Hugging Face 仓库下载
  const baseList = buildModelList(prebuiltAppConfig.model_list, extraRecords, mirrorOwner);
  const ladder = contextWindowLadder(contextWindowTokens);

  loadedId = null;
  loadedWindow = null;
  let lastError = null;
  for (let i = 0; i < ladder.length; i++) {
    const windowTokens = ladder[i];
    const appConfig = { ...prebuiltAppConfig, model_list: applyContextWindow(baseList, windowTokens) };
    try {
      await loadWithConfig(modelId, appConfig, progress);
      loadedId = modelId;
      loadedWindow = windowTokens;
      return { contextWindowTokens: windowTokens, requestedTokens: ladder[0], reduced: windowTokens < ladder[0] };
    } catch (err) {
      lastError = err;
      await resetEngine();
      if (!isMemoryError(err) || i === ladder.length - 1) throw err;
    }
  }
  throw lastError;
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
