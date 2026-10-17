import { streamChat } from '../api.js';
import {
  checkWebGPU,
  getLoadedContextWindow,
  getLoadedModelId,
  loadWebLLMEngine,
  streamWebLLM,
  WEBGPU_MISSING,
} from './webllm.js';
import { loadLocalModels, localRecords } from '../local-models.js';
import { composeSystemPrompt, getModelParams } from '../adaptation.js';
import { DEFAULT_CONTEXT_WINDOW } from '../model-source.js';
import { estimateTokens, historyTokenBudget } from '../store.js';

/** 历史消息按 8 条对齐起点，使多轮对话的前缀保持稳定（便于复用 KV 缓存）。 */
const STABLE_BLOCK = 8;

/**
 * 统一的推理后端接口：
 *   mode            'webllm'（默认，纯静态，浏览器内推理）| 'remote'（自托管服务端 + OpenAI 兼容 API）
 *   limits          上下文预算（历史消息条数；远程模式按字符，本机模型按 token 预算）
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
    contextWindowTokens: null,
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
  const requestedWindow = Number(cfg.contextWindowTokens) > 0 ? Number(cfg.contextWindowTokens) : DEFAULT_CONTEXT_WINDOW;
  // 发送与预算计算使用同一份系统提示词，保证预算与实际输入一致
  const systemText = () => composeSystemPrompt(cfg.systemPrompt, modelId);

  return {
    mode: 'webllm',
    models: cfg.models,
    get modelId() {
      return modelId;
    },
    /** 当前模型实际生效的窗口（已加载且降级时为降级后的值，否则为配置的请求值）。 */
    get contextWindowTokens() {
      return getLoadedContextWindow(modelId) ?? requestedWindow;
    },
    /**
     * 历史消息预算：窗口减去输出预留（maxTokens，不超过窗口一半）与系统提示词，
     * 使一次请求的输入与输出都在窗口之内。
     */
    get limits() {
      const params = getModelParams(modelId);
      const maxOutputTokens = params.maxTokens ?? cfg.maxTokens;
      const reservedTokens = estimateTokens(systemText());
      return {
        maxMessages: cfg.maxMessages,
        tokenBudget: historyTokenBudget({
          windowTokens: this.contextWindowTokens,
          maxOutputTokens,
          reservedTokens,
        }),
        stableBlock: STABLE_BLOCK,
      };
    },
    isReady: () => getLoadedModelId() === modelId,
    setModel(id) {
      modelId = id;
    },
    /** 返回加载结果 {contextWindowTokens, requestedTokens, reduced}；已就绪时返回 null。 */
    async ensureReady(onProgress) {
      if (getLoadedModelId() === modelId) return null;
      if (!(await checkWebGPU())) throw new Error(WEBGPU_MISSING);
      // 自定义记录 + 本地“新建 AI”（仅存在于浏览器缓存中）
      const extraRecords = [...(cfg.extraRecords || []), ...localRecords(loadLocalModels())];
      return loadWebLLMEngine(modelId, onProgress, {
        mirrorOwner: cfg.mirrorOwner,
        extraRecords,
        contextWindowTokens: requestedWindow,
      });
    },
    async *stream({ messages, signal }) {
      // 管理员反馈（仅本机）附加到该模型的系统提示词，并可覆盖该模型的采样参数
      const system = { role: 'system', content: systemText() };
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
