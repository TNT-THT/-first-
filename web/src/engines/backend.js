import { streamChat } from '../api.js';
import {
  checkWebGPU,
  forceResetEngine,
  getLoadedContextWindow,
  getLoadedModelId,
  loadVerifiedWebLLMEngine,
  releaseEngine,
  streamWebLLM,
  WEBGPU_MISSING,
} from './webllm.js';
import { loadLocalModels, localRecords } from '../local-models.js';
import { composeSystemPrompt, getModelParams } from '../adaptation.js';
import { presetOverrides } from '../presets.js';
import { SEARCH_EVIDENCE_SYSTEM_GUARD } from '../search/context.js';
import { getThinkingPref, isReasoningModel, setThinkingPref } from '../reasoning.js';
import { getSlidingPref, setSlidingPref } from '../sliding.js';
import {
  FORCE_NOTE_RESERVE,
  fitHistory,
  forcedAnswerMessages,
  systemPromptFits,
  forcedPromptTokens,
  ANSWER_MIN_TOKENS,
  estimatePromptTokens,
  engineIsStale,
  planGeneration,
  resolveWindows,
} from '../budget.js';

/**
 * 统一的推理后端接口：
 *   mode            'webllm'（默认，纯静态，浏览器内推理）| 'remote'（自托管服务端 + OpenAI 兼容 API）
 *   limits          上下文预算（发送给模型的历史消息条数 / 字符数）；开启滑动窗口时不按条数与字数截断，只按窗口截断
 *   isReady()       模型是否已就绪
 *   ensureReady(cb) 必要时加载模型，cb 接收 {progress, text}
 *   stream({messages, signal}) 异步生成器，逐段产出文本
 *   supportsThinking() 当前模型是否为推理模型（可切换思考模式）
 *   thinkingEnabled() / setThinking(on) 思考模式开关（仅推理模型生效，默认开启）
 *   slidingEnabled() / setSliding(on) 滑动窗口开关（所有模型，默认关闭）
 */
export function createBackend({ mode, appConfig }) {
  return mode === 'remote' ? createRemoteBackend() : createWebLLMBackend(appConfig.webllm);
}

function createRemoteBackend() {
  return {
    mode: 'remote',
    limits: { maxMessages: 20, maxChars: 40000 },
    models: [],
    slidingEnabled: () => false,
    setSliding() {},
    forceRelease() {},
    get modelId() {
      return '';
    },
    isReady: () => true,
    setModel() {},
    supportsThinking: () => false,
    thinkingEnabled: () => true,
    setThinking() {},
    async ensureReady() {},
    async releaseIfStale() {
      return false;
    },
    async *stream({ messages, signal }) {
      yield* streamChat({ messages, signal });
    },
  };
}

/**
 * 某个模型生效的数值覆盖：预设（按浏览器）< 管理员参数（本机，优先）。
 * 没有覆盖的项沿用 config.json 的默认值。
 */
export function layeredOverrides(modelId) {
  return { ...presetOverrides(), ...getModelParams(modelId) };
}

/**
 * 某个模型当前的窗口：预设与管理员参数优先，否则使用配置默认值。
 * 引擎窗口 = 上下文窗口；只有推理模型且开启思考时才加上思考窗口。
 * 引擎窗口在加载时确定，所以切换思考开关后，下一次发送会按新窗口重新加载。
 */
function windowsFor(cfg, modelId, thinkingOn, sliding = getSlidingPref()) {
  const supportsThinking = isReasoningModel(modelId, { models: cfg.models, localEntries: loadLocalModels() });
  const thinkingActive = supportsThinking && !!thinkingOn;
  return {
    supportsThinking,
    thinkingActive,
    ...resolveWindows({
      defaults: {
        contextWindow: cfg.contextWindowTokens,
        slidingWindow: cfg.slidingWindowTokens,
        thinkingWindow: cfg.thinkingWindowTokens,
      },
      overrides: layeredOverrides(modelId),
      reserveThinking: thinkingActive,
      sliding,
    }),
  };
}

function createWebLLMBackend(cfg) {
  let modelId = cfg.defaultModel;
  // 模型与窗口都与已加载引擎一致才算就绪
  const loaded = () =>
    getLoadedModelId() === modelId &&
    getLoadedContextWindow() === windowsFor(cfg, modelId, getThinkingPref()).engineContextSize;

  return {
    mode: 'webllm',
    // 滑动窗口开启时历史不按条数与字数截断，由 stream() 中的 fitHistory 按窗口截断（最早的消息滚出窗口）
    // 丢弃粒度（historyDropChunk）：超出后一次丢一段，之后几轮的起点不变，减少 KV 缓存失效（重新预填）
    get limits() {
      const chunk = cfg.historyDropChunk ?? 1;
      return getSlidingPref()
        ? { maxMessages: Infinity, maxChars: Infinity, chunk }
        : { maxMessages: cfg.maxMessages, maxChars: cfg.contextChars, chunk };
    },
    models: cfg.models,
    get modelId() {
      return modelId;
    },
    isReady: loaded,
    setModel(id) {
      modelId = id;
    },
    supportsThinking: () =>
      isReasoningModel(modelId, { models: cfg.models, localEntries: loadLocalModels() }),
    thinkingEnabled: () => getThinkingPref(),
    setThinking(on) {
      setThinkingPref(on);
    },
    slidingEnabled: () => getSlidingPref(),
    setSliding(on) {
      setSlidingPref(on);
    },
    /** 用户已停止但模型仍无响应时的强制释放（立即终止 worker，不等待）。 */
    forceRelease() {
      forceResetEngine();
    },
    /**
     * 参数改变后释放多余显存：若已加载的引擎模型或窗口与当前设置不一致，则释放引擎。
     * 返回是否真的释放了（没有加载任何引擎时返回 false）。
     */
    async releaseIfStale() {
      const id = getLoadedModelId();
      if (!id) return false;
      const { engineContextSize } = windowsFor(cfg, modelId, getThinkingPref());
      if (!engineIsStale({ loadedId: id, loadedCtx: getLoadedContextWindow(), modelId, engineContextSize })) {
        return false;
      }
      return releaseEngine();
    },
    async ensureReady(onProgress, { signal } = {}) {
      if (loaded()) return;
      if (!(await checkWebGPU())) throw new Error(WEBGPU_MISSING);
      // 自定义记录 + 本地“新建 AI”（仅存在于浏览器缓存中）
      const extraRecords = [...(cfg.extraRecords || []), ...localRecords(loadLocalModels())];
      // 加载后做输出自检；fp16 异常时自动切换到 fp32（见 health.js）
      // 窗口与已加载引擎不一致时（例如管理员调整了窗口）会重新加载
      const { engineContextSize } = windowsFor(cfg, modelId, getThinkingPref());
      return loadVerifiedWebLLMEngine(modelId, onProgress, {
        mirrorOwner: cfg.mirrorOwner,
        extraRecords,
        contextWindowSize: engineContextSize,
        signal,
      });
    },
    async *stream({ messages, signal, onLoop }) {
      // 管理员反馈（仅本机）附加到该模型的系统提示词，并可覆盖该模型的采样参数
      const system = {
        role: 'system',
        content: `${composeSystemPrompt(cfg.systemPrompt, modelId)}\n\n${SEARCH_EVIDENCE_SYSTEM_GUARD}`,
      };
      const params = layeredOverrides(modelId);
      const repetitionPenalty = params.repetitionPenalty ?? cfg.repetitionPenalty ?? null;
      const thinkingOn = getThinkingPref();
      const w = windowsFor(cfg, modelId, thinkingOn);
      const temperature = params.temperature ?? cfg.temperature;
      const answerCap = params.maxTokens ?? cfg.maxTokens;
      const overflowError = () =>
        new Error('Prompt tokens exceed context window size：对话内容已占满上下文窗口，请新建对话或减少历史');
      // 历史超出提示词空间时从最早的消息开始丢弃；若启用了思考预算，为强制输出说明预留空间
      const reserve = w.thinkingActive ? FORCE_NOTE_RESERVE : 0;
      // 管理员信息（更正、知识库、反馈）从不截断：系统提示放不下时直接报错
      if (!systemPromptFits(system, w.contextWindow, reserve)) {
        throw new Error(
          `管理员信息（更正、知识库、反馈）与系统提示已超出${w.sliding ? '滑动' : '上下文'}窗口，为避免截断，本次未发送。请精简这些内容，或在参数中调大${w.sliding ? '滑动窗口（参数 slidingWindow，最大 20000）' : '上下文窗口（参数 contextWindow，最大 20000）'}。`,
        );
      }
      const all = fitHistory([system, ...messages], w.contextWindow - ANSWER_MIN_TOKENS - reserve, {
        chunk: cfg.historyDropChunk ?? 1,
      });
      // 提示词估计 + 思考预算 + 回答长度，保证不超过引擎窗口
      const plan = planGeneration({
        promptTokens: estimatePromptTokens(all),
        contextWindow: w.contextWindow,
        thinkingWindow: w.thinkingWindow,
        supportsThinking: w.supportsThinking,
        thinkingOn,
        answerCap,
      });
      if (plan.overflow) throw overflowError();
      const first = yield* streamWebLLM({
        messages: all,
        signal,
        temperature,
        maxTokens: plan.maxTokens,
        supportsThinking: w.supportsThinking,
        thinkingEnabled: thinkingOn,
        thinkingBudget: plan.thinkingTokens || null,
        repetitionPenalty,
        onLoop,
      });
      // 思考预算用完仍未回答：直接发起回答请求（关闭思考，并附上已写出的部分思考）
      if (first?.forced && !signal?.aborted) {
        const forced = forcedAnswerMessages(all, first.thinking);
        const direct = planGeneration({
          promptTokens: forcedPromptTokens(all, first.thinkingTokens),
          contextWindow: w.engineContextSize,
          thinkingWindow: 0,
          supportsThinking: false,
          thinkingOn: false,
          answerCap,
        });
        if (direct.overflow) throw overflowError();
        yield* streamWebLLM({
          messages: forced,
          signal,
          temperature,
          maxTokens: direct.maxTokens,
          supportsThinking: w.supportsThinking,
          thinkingEnabled: false,
          repetitionPenalty,
          onLoop,
        });
      }
    },
  };
}
