/**
 * 上下文预算（纯逻辑，便于测试）。
 *
 * - 上下文窗口 W：提示词（系统提示 + 历史）与回答共用。默认 16384，最大 20000（管理员可在参数中调整）。
 * - 思考窗口 T：只有推理模型开启思考时才占用，与 W 分开计算。默认 4096，最大 8000（管理员可调）。
 * - 引擎的 context_window_size = W + T（推理模型）或 W（其他模型），加载模型时设置。
 * - 发送的 max_tokens = （思考开启 ? T : 0）+ 回答长度；回答长度 = min(最大输出长度, W − 提示词估计 tokens)。
 *   因此 提示词 + 思考 + 回答 ≤ W + T。
 * - T 是思考的预算，但不是硬性截断：模型思考超过 T 时，剩余给回答的空间会变少，总量仍不超过引擎窗口。
 */

export const WINDOW_LIMITS = {
  contextWindow: { min: 4096, max: 20000 },
  thinkingWindow: { min: 512, max: 8000 },
};

/** 回答至少保留的 tokens；提示词太长、连这点空间都没有时报错，而不是发送后被引擎拒绝。 */
export const ANSWER_MIN_TOKENS = 512;
/** 每条消息的模板开销（角色标记等），估计值。 */
const PER_MESSAGE_OVERHEAD = 8;

function clamp(value, [min, max]) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return null;
  return Math.min(max, Math.max(min, n));
}

/**
 * 保守估计文本的 token 数（宁可多估，避免提示词超出引擎窗口）：
 * CJK 字符按约 1.5 个 token 计，其他字符按约 0.6 个 token 计。实际分词通常更少。
 */
export function estimateTokens(text) {
  let cjk = 0;
  let other = 0;
  for (const ch of String(text ?? '')) {
    if (ch.codePointAt(0) >= 0x2e80) cjk += 1;
    else other += 1;
  }
  return Math.ceil(cjk * 1.5 + other * 0.6);
}

/** 整段提示词（消息列表）的估计 token 数。 */
export function estimatePromptTokens(messages) {
  let total = PER_MESSAGE_OVERHEAD;
  for (const m of messages) total += estimateTokens(m.content) + PER_MESSAGE_OVERHEAD;
  return total;
}

/**
 * 解析某个模型的窗口大小（管理员参数优先，否则使用配置默认值；都夹在允许范围内）。
 * reserveThinking：本次是否预留思考窗口（推理模型且开启思考）。关闭思考时引擎只用上下文窗口。
 * @returns {{contextWindow: number, thinkingWindow: number, engineContextSize: number}}
 */
export function resolveWindows({ defaults = {}, overrides = {}, reserveThinking = false } = {}) {
  const contextWindow =
    clamp(overrides.contextWindow ?? defaults.contextWindow, [
      WINDOW_LIMITS.contextWindow.min,
      WINDOW_LIMITS.contextWindow.max,
    ]) ?? WINDOW_LIMITS.contextWindow.min;
  const thinkingWindow =
    clamp(overrides.thinkingWindow ?? defaults.thinkingWindow, [
      WINDOW_LIMITS.thinkingWindow.min,
      WINDOW_LIMITS.thinkingWindow.max,
    ]) ?? WINDOW_LIMITS.thinkingWindow.min;
  return {
    contextWindow,
    thinkingWindow,
    engineContextSize: contextWindow + (reserveThinking ? thinkingWindow : 0),
  };
}

/**
 * 计算本次生成的 max_tokens。
 * @returns {{maxTokens: number, answerTokens: number, thinkingTokens: number, overflow: boolean}}
 *   overflow 为 true 时表示提示词已占满上下文窗口，应提示用户缩短或新建会话。
 */
export function planGeneration({
  promptTokens,
  contextWindow,
  thinkingWindow,
  supportsThinking = false,
  thinkingOn = true,
  answerCap,
}) {
  const room = contextWindow - promptTokens;
  if (room < ANSWER_MIN_TOKENS) {
    return { maxTokens: 0, answerTokens: 0, thinkingTokens: 0, overflow: true };
  }
  const answerTokens = Math.max(1, Math.min(Number(answerCap) || room, room));
  const thinkingTokens = supportsThinking && thinkingOn ? thinkingWindow : 0;
  return { maxTokens: thinkingTokens + answerTokens, answerTokens, thinkingTokens, overflow: false };
}

/* ---------- 思考预算用完后的直接输出 ---------- */

const THINK_OPEN = '<think>';
const THINK_CLOSE = '</think>';
/** 思考预算用完时，补在输出中的思考结束标记（前端据此把思考块收起，之后即为回答）。 */
export const THINK_CLOSE_TEXT = '\n</think>\n\n';
/** 强制输出时附加到最后一条用户消息后的说明。 */
export const FORCE_ANSWER_NOTE =
  '（系统提示：本次思考的预算已经用完。下面是你已经写出的部分思考，仅供参考。请不要继续思考，也不要重复这段内容，直接给出最终回答。）';
/** 为强制输出说明预留的提示词 tokens（估计值）。 */
export const FORCE_NOTE_RESERVE = estimateTokens(FORCE_ANSWER_NOTE) + 2 * PER_MESSAGE_OVERHEAD + 64;

/**
 * 思考计量：按生成步骤（每个增量块约为一个 token）计数，只统计仍在 <think> 内的部分。
 * - step(delta) 返回 true 表示预算刚好用完且思考仍未结束，调用方应中断生成并补上 THINK_CLOSE_TEXT。
 * - budget 为空或 0 时不计量（非推理模型、关闭思考、或没有思考预算）。
 * 注意：计数是近似值（以输出块为单位），并非精确的 token 数。
 */
export function thinkingMeter(budget) {
  let raw = '';
  let tokens = 0;
  let cut = false;
  return {
    get cut() {
      return cut;
    },
    get tokens() {
      return tokens;
    },
    /** 已输出的思考原文（不含 <think> 标记）。 */
    thinking() {
      const i = raw.indexOf(THINK_OPEN);
      return i < 0 ? '' : raw.slice(i + THINK_OPEN.length);
    },
    step(delta) {
      if (!budget || cut) return false;
      raw += delta || '';
      if (raw.includes(THINK_OPEN) && !raw.includes(THINK_CLOSE)) {
        tokens += 1;
        if (tokens >= budget) {
          cut = true;
          return true;
        }
      }
      return false;
    },
  };
}

/**
 * 历史超过预算时，从最早的消息开始丢弃（保留系统提示与最后一条当前提问）。
 * 返回新数组，不修改原数组。
 */
/**
 * 系统提示（含更正、知识库与反馈）本身能否放进上下文窗口。
 * 管理员信息不会被截断：放不下时调用方应报错，而不是静默删减。
 */
export function systemPromptFits(systemMessage, contextWindow, reserve = 0) {
  return estimatePromptTokens([systemMessage]) <= contextWindow - ANSWER_MIN_TOKENS - reserve;
}

export function fitHistory(messages, limitTokens) {
  const out = [...messages];
  while (out.length > 2 && estimatePromptTokens(out) > limitTokens) out.splice(1, 1);
  return out;
}

/**
 * 强制输出时的请求消息：把已写出的部分思考附加到最后一条用户消息后（只影响这次请求，不改历史）。
 */
export function forcedAnswerMessages(messages, thinking) {
  const out = messages.map((m) => ({ ...m }));
  const body = String(thinking || '').trim();
  const add = `\n\n${FORCE_ANSWER_NOTE}\n<已写出的思考>\n${body}\n</已写出的思考>`;
  let idx = -1;
  for (let i = out.length - 1; i >= 0; i -= 1) {
    if (out[i].role === 'user') {
      idx = i;
      break;
    }
  }
  if (idx >= 0) out[idx] = { ...out[idx], content: `${out[idx].content}${add}` };
  else out.push({ role: 'user', content: add.trim() });
  return out;
}

/** 强制输出请求的提示词 tokens：原提示词 + 已用完的思考预算（按实际计数）+ 说明文字。 */
export function forcedPromptTokens(messages, thinkingTokens) {
  return estimatePromptTokens(messages) + thinkingTokens + FORCE_NOTE_RESERVE;
}

/**
 * 已加载的引擎是否与当前设置不一致：模型不同，或上下文窗口不同（管理员调整了参数）。
 * 不一致时引擎占用的显存（KV 缓存按窗口分配）已经多余，应当释放；下次发送时按新设置重新加载。
 */
export function engineIsStale({ loadedId, loadedCtx, modelId, engineContextSize }) {
  return loadedId !== modelId || loadedCtx !== engineContextSize;
}
