import { streamChat } from '../api.js';
import { checkWebGPU, getLoadedModelId, loadWebLLMEngine, streamWebLLM, WEBGPU_MISSING } from './webllm.js';

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
  const systemMessage = { role: 'system', content: cfg.systemPrompt };

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
      await loadWebLLMEngine(modelId, onProgress);
    },
    async *stream({ messages, signal }) {
      yield* streamWebLLM({
        messages: [systemMessage, ...messages],
        signal,
        temperature: cfg.temperature,
        maxTokens: cfg.maxTokens,
      });
    },
  };
}
