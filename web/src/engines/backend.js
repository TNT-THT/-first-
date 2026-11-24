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
import { SEARCH_EVIDENCE_SYSTEM_GUARD, hasSearchEvidence } from '../search/context.js';
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
    // 服务端（本机大模型/上游）支持思考开关：偏好与网页版共用同一存储，
    // 发送时透传给服务端，由服务端下发 enable_thinking 给模型。
    // 此前这里写死“不支持+恒开”，界面开关因此置灰且无效、模型却照常思考。
    supportsThinking: () => true,
    thinkingEnabled: () => getThinkingPref(),
    setThinking(on) {
      setThinkingPref(on);
    },
    async ensureReady() {},
    async releaseIfStale() {
      return false;
    },
    async *stream({ messages, signal, model }) {
      yield* streamChat({ messages, signal, model, thinking: getThinkingPref() });
    },
  };
}

/**
 * 把对话中夹杂的 system 消息折叠进开头唯一的 system，返回 { systemContent, rest }。
 *
 * OpenAI 调用方（第三方软件）通常自带 system 提示词，甚至不止一条或不在第一条；
 * 而本引擎前面还会再加本站的 system。WebLLM 的对话模板要求 system 只能是
 * 第一条且只能有一条，否则直接抛 SystemMessageOrderError、生成根本起不来
 * （Phi 等模板尤其严格）。合并顺序：本站（含管理员信息）在前，调用方的随后，
 * 内容已包含的片段不重复添加；rest 为去掉 system 后的其余消息，顺序不变。
 */
export function foldSystemMessages(systemContent, messages) {
  let merged = String(systemContent || '');
  const rest = [];
  for (const m of Array.isArray(messages) ? messages : []) {
    if (m?.role === 'system') {
      const text = String(m.content || '').trim();
      if (text && !merged.includes(text)) merged = merged ? `${merged}\n\n${text}` : text;
    } else if (m && typeof m === 'object') {
      rest.push(m);
    }
  }
  return { systemContent: merged, rest };
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
    async *stream({
      messages,
      signal,
      onLoop,
      temperature: temperatureOverride,
      maxTokens: maxTokensOverride,
      // 本机 API 桥接专用：服务端给出的输出硬上限（不覆盖本机参数，只封顶）
      hardMaxTokens = null,
      // 本机 API 桥接专用：已校验的生成参数（topP/stop/惩罚项/seed/responseFormat 等）
      generation = {},
      // 本机 API 桥接专用：服务端系统提示词，附加到本机系统提示词之后
      systemPrompt: extraSystemPrompt = '',
    }) {
      // 管理员反馈（仅本机）附加到该模型的系统提示词，并可覆盖该模型的采样参数。
      // 外部资料的安全守则只在本条真的带了资料时才追加：为空不导入给 AI。
      let systemContent = composeSystemPrompt(cfg.systemPrompt, modelId);
      if (hasSearchEvidence(messages)) systemContent += `\n\n${SEARCH_EVIDENCE_SYSTEM_GUARD}`;
      const extraSystem = String(extraSystemPrompt || '').trim();
      if (extraSystem && !systemContent.includes(extraSystem)) {
        systemContent = `${systemContent}\n\n${extraSystem}`;
      }
      // 调用方自带的 system 折叠进开头这一条，避免第二条 system 触发模板的 SystemMessageOrderError
      const folded = foldSystemMessages(systemContent, messages);
      const system = { role: 'system', content: folded.systemContent };
      const params = layeredOverrides(modelId);
      const repetitionPenalty = generation.repetitionPenalty ?? params.repetitionPenalty ?? cfg.repetitionPenalty ?? null;
      const thinkingOn = getThinkingPref();
      const w = windowsFor(cfg, modelId, thinkingOn);
      // 调用方显式给出的采样参数优先（本机 API 桥接透传），其次管理员参数，最后配置默认值
      const temperature = temperatureOverride ?? params.temperature ?? cfg.temperature;
      const localCap = maxTokensOverride ?? params.maxTokens ?? cfg.maxTokens;
      const answerCap = hardMaxTokens != null ? Math.min(localCap ?? hardMaxTokens, hardMaxTokens) : localCap;
      const overflowError = () =>
        new Error('Prompt tokens exceed context window size：对话内容已占满上下文窗口，请新建对话或减少历史');
      // 历史超出提示词空间时从最早的消息开始丢弃；若启用了思考预算，为强制输出说明预留空间
      const reserve = w.thinkingActive ? FORCE_NOTE_RESERVE : 0;
      // 管理员信息（全局知识库、反馈）从不截断：系统提示放不下时直接报错
      if (!systemPromptFits(system, w.contextWindow, reserve)) {
        throw new Error(
          `管理员信息（知识库、反馈）与系统提示已超出${w.sliding ? '滑动' : '上下文'}窗口，为避免截断，本次未发送。请精简这些内容，或在参数中调大${w.sliding ? '滑动窗口（参数 slidingWindow，最大 20000）' : '上下文窗口（参数 contextWindow，最大 20000）'}。`,
        );
      }
      const all = fitHistory([system, ...folded.rest], w.contextWindow - ANSWER_MIN_TOKENS - reserve, {
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
        topP: generation.topP ?? null,
        stop: generation.stop ?? null,
        frequencyPenalty: generation.frequencyPenalty ?? null,
        presencePenalty: generation.presencePenalty ?? null,
        seed: generation.seed ?? null,
        responseFormat: generation.responseFormat ?? null,
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
          topP: generation.topP ?? null,
          stop: generation.stop ?? null,
          frequencyPenalty: generation.frequencyPenalty ?? null,
          presencePenalty: generation.presencePenalty ?? null,
          seed: generation.seed ?? null,
          responseFormat: generation.responseFormat ?? null,
        });
      }
    },
  };
}
