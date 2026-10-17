/**
 * 浏览器内推理（WebLLM / WebGPU）。模型权重从 Hugging Face 下载一次后缓存在浏览器中，
 * 之后的对话完全在本机完成，不需要服务器，也不需要 API Key。
 */
import { buildModelList } from '../model-source.js';
import { decideVariant, diagnoseOutput, fp32VariantOf, HEALTH_MAX_TOKENS, HEALTH_PROMPT } from './health.js';

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

/**
 * 当前生效的完整模型记录（内置列表 + 自定义记录 + 镜像设置），用于导出 / 导入 / 缓存状态检查。
 * 动态导入推理库，避免首屏变重。
 */
export async function getModelRecords({ mirrorOwner, extraRecords } = {}) {
  const { prebuiltAppConfig } = await import('@mlc-ai/web-llm');
  return buildModelList(prebuiltAppConfig.model_list, extraRecords, mirrorOwner);
}

export async function loadWebLLMEngine(modelId, onProgress, { mirrorOwner, extraRecords } = {}) {
  if (!(await checkWebGPU())) {
    throw new Error(WEBGPU_MISSING);
  }
  const progress = (r) => onProgress?.({ progress: r.progress ?? 0, text: r.text || '' });
  // 动态导入：避免把较大的推理库打进首屏
  const { CreateWebWorkerMLCEngine, prebuiltAppConfig } = await import('@mlc-ai/web-llm');
  // 合并自定义记录；配置了 mirrorOwner 时权重从自己的 Hugging Face 仓库下载
  const appConfig = {
    ...prebuiltAppConfig,
    model_list: buildModelList(prebuiltAppConfig.model_list, extraRecords, mirrorOwner),
  };

  loadedId = null;
  if (!engine) {
    const worker = new Worker(new URL('../llm-worker.js', import.meta.url), { type: 'module' });
    engine = await CreateWebWorkerMLCEngine(worker, modelId, { initProgressCallback: progress, appConfig });
  } else {
    engine.setInitProgressCallback(progress);
    engine.setAppConfig(appConfig);
    await engine.reload(modelId);
  }
  loadedId = modelId;
}

/** 用固定提示词做一次确定性短输出，返回 {sample, reason}；reason 为 null 表示正常。 */
export async function runHealthCheck() {
  const res = await engine.chat.completions.create({
    messages: [{ role: 'user', content: HEALTH_PROMPT }],
    stream: false,
    temperature: 0,
    max_tokens: HEALTH_MAX_TOKENS,
  });
  const sample = res.choices?.[0]?.message?.content ?? '';
  return { sample, reason: diagnoseOutput(sample) };
}

/**
 * 加载模型并做输出自检。异常且存在 fp32 版本时，自动切换到 fp32，
 * 但对外仍以用户选中的 modelId 作为“当前模型”（对话归属、参数、适应记录都不变）。
 * 返回 {variant: 'default'|'fp16'|'fp32', healthy, note}；note 用于界面提示。
 */
export async function loadVerifiedWebLLMEngine(modelId, onProgress, opts = {}) {
  await loadWebLLMEngine(modelId, onProgress, opts);
  let check;
  try {
    check = await runHealthCheck();
  } catch {
    // 自检本身出错：不切换，只标记为未完成
    return { variant: 'default', healthy: null, note: '输出自检未能完成' };
  }
  const variantOf = (id) => (id.endsWith('q4f16_1-MLC') ? 'fp16' : id.endsWith('q4f32_1-MLC') ? 'fp32' : 'default');
  if (!check.reason) return { variant: variantOf(modelId), healthy: true, note: '' };

  const fp32 = fp32VariantOf(modelId);
  const records = fp32 ? await getModelRecords(opts) : [];
  const fp32Available = !!fp32 && records.some((r) => r.model_id === fp32);
  const decision = decideVariant({ modelId, healthy: false, fp32Available });
  if (decision.action !== 'fallback') {
    return {
      variant: variantOf(modelId),
      healthy: false,
      note: `输出自检异常（${check.reason}），且没有可用的 fp32 版本，回答可能乱码`,
    };
  }

  await loadWebLLMEngine(decision.to, onProgress, opts);
  loadedId = modelId; // 对外仍是用户选中的模型
  let second;
  try {
    second = await runHealthCheck();
  } catch {
    second = { reason: null };
  }
  return second.reason
    ? { variant: 'fp32', healthy: false, note: `已切换到 fp32，但自检仍异常（${second.reason}）` }
    : { variant: 'fp32', healthy: true, note: '检测到 fp16 输出异常，已自动切换到 fp32 版本' };
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
