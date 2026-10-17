import { streamChat } from '../api.js';
import { checkWebGPU, getLoadedModelId, loadVerifiedWebLLMEngine, streamWebLLM, WEBGPU_MISSING } from './webllm.js';
import { loadLocalModels, localRecords } from '../local-models.js';
import { composeSystemPrompt, getModelParams } from '../adaptation.js';

/**
 * 统一的推理后端接口：
 *   mode            'webllm'（默认，纯静态，浏览器内推理）| 'remote'（自托管服务端 + OpenAI 兼容 API）
 *   limits          上下文预算（发送给模型的历史消息条数 / 字符数）
 *   isReady()       模型是否已就绪
 *   ensureReady(cb) 必要时加载模型，cb 接收 {progress, text}
 *   stream({messages, signal}) 异步生成器，逐段产出文本
 */
export function createBackend({ mode, appConfig }) {
  return mode === 'remote' ? createRemoteBackend() : createWebLLMBackend(appConfig.webllm);
}

function createRemoteBackend() {
  return {
    mode: 'remote',
    limits: { maxMessages: 20, maxChars: 40000 },
    models: [],
    get modelId() {
      return '';
    },
    isReady: () => true,
    setModel() {},
    async ensureReady() {},
    async *stream({ messages, signal }) {
      yield* streamChat({ messages, signal });
    },
  };
}

function createWebLLMBackend(cfg) {
  let modelId = cfg.defaultModel;

  return {
    mode: 'webllm',
    limits: { maxMessages: cfg.maxMessages, maxChars: cfg.contextChars },
    models: cfg.models,
    get modelId() {
      return modelId;
    },
    isReady: () => getLoadedModelId() === modelId,
    setModel(id) {
      modelId = id;
    },
    async ensureReady(onProgress) {
      if (getLoadedModelId() === modelId) return;
      if (!(await checkWebGPU())) throw new Error(WEBGPU_MISSING);
      // 自定义记录 + 本地“新建 AI”（仅存在于浏览器缓存中）
      const extraRecords = [...(cfg.extraRecords || []), ...localRecords(loadLocalModels())];
      // 加载后做输出自检；fp16 异常时自动切换到 fp32（见 health.js）
      return loadVerifiedWebLLMEngine(modelId, onProgress, { mirrorOwner: cfg.mirrorOwner, extraRecords });
    },
    async *stream({ messages, signal }) {
      // 管理员反馈（仅本机）附加到该模型的系统提示词，并可覆盖该模型的采样参数
      const system = { role: 'system', content: composeSystemPrompt(cfg.systemPrompt, modelId) };
      const params = getModelParams(modelId);
      yield* streamWebLLM({
        messages: [system, ...messages],
        signal,
        temperature: params.temperature ?? cfg.temperature,
        maxTokens: params.maxTokens ?? cfg.maxTokens,
      });
    },
  };
}
