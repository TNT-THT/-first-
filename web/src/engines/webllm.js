/**
 * 浏览器内推理（WebLLM / WebGPU）。模型权重从 Hugging Face 下载一次后缓存在浏览器中，
 * 之后的对话完全在本机完成，不需要服务器，也不需要 API Key。
 */
import { buildModelList } from '../model-source.js';
import { decideVariant, diagnoseOutput, fp32VariantOf, HEALTH_MAX_TOKENS, HEALTH_PROMPT } from './health.js';
import { createSerialQueue, gpuResetError, isGpuResetError } from './lifecycle.js';
import { thinkingRequestOptions } from '../reasoning.js';

export const WEBGPU_MISSING =
  '当前浏览器不支持 WebGPU，无法运行本地模型。请使用最新版 Chrome 或 Edge（桌面端），并确认显卡驱动正常。';

let engine = null;
let worker = null;
let loadedId = null;
// 当前进行中的生成（流式输出结束时才 resolve）。加载模型前必须等待它结束，否则 GPU 资源可能被提前销毁。
let inflight = null;
// 所有加载请求排队执行，避免两次 reload 交错。
const enqueueLoad = createSerialQueue();

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

/** 丢弃当前引擎与 worker；下次加载会创建新的 worker。 */
function resetEngine() {
  const w = worker;
  worker = null;
  engine = null;
  loadedId = null;
  try {
    w?.terminate();
  } catch {
    // 忽略：worker 可能已经结束
  }
}

/** 加载模型（排队执行）。 */
export function loadWebLLMEngine(modelId, onProgress, opts = {}) {
  return enqueueLoad(() => doLoadWebLLMEngine(modelId, onProgress, opts));
}

async function doLoadWebLLMEngine(modelId, onProgress, { mirrorOwner, extraRecords } = {}) {
  // 先等待进行中的生成结束（如有），再重新加载，避免 unload 时销毁仍在读回的 GPU 缓冲区
  if (inflight) {
    engine?.interruptGenerate().catch(() => {});
    await inflight;
  }
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
  try {
    if (!engine) {
      worker = new Worker(new URL('../llm-worker.js', import.meta.url), { type: 'module' });
      engine = await CreateWebWorkerMLCEngine(worker, modelId, { initProgressCallback: progress, appConfig });
    } else {
      engine.setInitProgressCallback(progress);
      engine.setAppConfig(appConfig);
      await engine.reload(modelId);
    }
  } catch (err) {
    if (isGpuResetError(err)) {
      resetEngine();
      throw gpuResetError();
    }
    throw err;
  }
  loadedId = modelId;
}

/** 用固定提示词做一次确定性短输出，返回 {sample, reason}；reason 为 null 表示正常。 */
export async function runHealthCheck() {
  try {
    const res = await engine.chat.completions.create({
      messages: [{ role: 'user', content: HEALTH_PROMPT }],
      stream: false,
      temperature: 0,
      max_tokens: HEALTH_MAX_TOKENS,
    });
    const sample = res.choices?.[0]?.message?.content ?? '';
    return { sample, reason: diagnoseOutput(sample) };
  } catch (err) {
    if (isGpuResetError(err)) {
      resetEngine();
      throw gpuResetError();
    }
    throw err;
  }
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
  } catch (err) {
    if (isGpuResetError(err)) throw err; // GPU 资源已重置：交给上层提示重新发送
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
  } catch (err) {
    if (isGpuResetError(err)) throw err;
    second = { reason: null };
  }
  return second.reason
    ? { variant: 'fp32', healthy: false, note: `已切换到 fp32，但自检仍异常（${second.reason}）` }
    : { variant: 'fp32', healthy: true, note: '检测到 fp16 输出异常，已自动切换到 fp32 版本' };
}

/**
 * 流式生成。supportsThinking / thinkingEnabled 决定是否关闭推理模型的思考（见 reasoning.js）。
 */
export async function* streamWebLLM({ messages, signal, temperature, maxTokens, supportsThinking = false, thinkingEnabled = true }) {
  if (!engine || !loadedId) throw new Error('模型尚未加载');
  if (signal?.aborted) return;
  const current = engine;
  let markDone;
  const done = new Promise((resolve) => {
    markDone = resolve;
  });
  inflight = done;
  const onAbort = () => {
    current.interruptGenerate().catch(() => {});
  };
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    const stream = await current.chat.completions.create({
      messages,
      stream: true,
      temperature,
      max_tokens: maxTokens,
      ...thinkingRequestOptions({ supportsThinking, enabled: thinkingEnabled }),
    });
    // 用户停止后仍把剩余的流读完（不再输出）：引擎会在下一个 token 前停止，
    // 读完才表示本次生成的 GPU 工作已经结束，之后才允许重新加载模型。
    for await (const chunk of stream) {
      if (signal?.aborted) continue;
      const delta = chunk.choices?.[0]?.delta?.content;
      if (delta) yield delta;
    }
  } catch (err) {
    if (isGpuResetError(err)) {
      resetEngine();
      throw gpuResetError();
    }
    throw err;
  } finally {
    signal?.removeEventListener('abort', onAbort);
    markDone();
    if (inflight === done) inflight = null;
  }
}
