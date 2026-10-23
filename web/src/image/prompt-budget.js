/**
 * 生图提示词的 CLIP 预算（纯逻辑，不依赖浏览器）。
 *
 * - CLIP 文本编码器每段读取 77 个 token，其中首尾各占 1 个（BOS、EOS），中间 75 个可用。
 * - 描述（用户输入）最多分为 MAX_PROMPT_CHUNKS 段分别编码（见 long-prompt.js），因此可用 token 为 段数 × 75。
 *   超出部分不参与生图，会报告截断的 token 数。
 * - 管理员的更正与知识库放在提示词最前面，并且必须生效：它们自身超过单段（75 个 token）时拒绝生成，
 *   而不是丢掉其中的一部分。它们只占第一段，因此即使分段失败回退到单段，也能完整保留。
 *
 * 注意：CLIP 不会“服从”文字指令。这里保证的是管理员内容位于最前、完整放入、不被截断。
 */

export const CLIP_CONTEXT_TOKENS = 77;
/** 每段去掉 BOS 与 EOS 之后的可用 token 数。 */
export const CLIP_CONTENT_TOKENS = CLIP_CONTEXT_TOKENS - 2;
/** 描述最多分段数（每段 75 个 token，共约 300 个 token）。超出部分截断并报告。 */
export const MAX_PROMPT_CHUNKS = 4;

/**
 * 描述（文本框）的输入上限（token）。默认 4096；管理员可用参数行调整，最高 16384（硬上限）。
 * 注意：输入上限与分段是两回事。超过 MAX_PROMPT_CHUNKS 段的部分仍会被截断，只是不再因输入上限拒绝。
 */
export const DEFAULT_INPUT_TOKENS = 4096;
export const MAX_INPUT_TOKENS = 16384;
export const MIN_INPUT_TOKENS = 1;

/** 不截断地统计文本占用的内容 token 数（不含 BOS 与 EOS）。 */
export function countContentTokens(tokenizer, text) {
  const body = String(text || '').trim();
  if (!body) return 0;
  return Math.max(0, tokenizer.rawTokenize(body).length - 2);
}

/**
 * 检查预算。
 * - input：用户在文本框中的描述；超过 maxInputTokens（默认 DEFAULT_INPUT_TOKENS）时抛出错误，拒绝生成。
 * - lead：管理员前置内容（更正与知识库，已用 ', ' 连接，与 buildImagePrompt 的前缀一致）；超过单段 75 个 token 时抛出错误。
 * - prompt：完整提示词；maxChunks 为允许的段数（默认 1，即不分段）。返回 truncatedTokens（末尾被截断的 token 数）。
 * 返回 {inputTokens, maxInputTokens, leadTokens, promptTokens, capacity, truncatedTokens}。
 */
export function checkPromptBudget(
  tokenizer,
  { input = null, maxInputTokens = DEFAULT_INPUT_TOKENS, lead = '', prompt = '', maxChunks = 1 } = {},
) {
  const max = Number.isInteger(tokenizer.maxLength) ? tokenizer.maxLength : CLIP_CONTEXT_TOKENS;
  const perChunk = max - 2;
  const chunks = Number.isInteger(maxChunks) && maxChunks >= 1 ? maxChunks : 1;
  const capacity = chunks * perChunk;
  // 输入上限先于 CLIP 检查：描述本身超过上限时，直接告知管理员可调的上限
  const limit = Number.isInteger(maxInputTokens) ? maxInputTokens : DEFAULT_INPUT_TOKENS;
  const inputTokens = input === null ? 0 : countContentTokens(tokenizer, input);
  if (input !== null && inputTokens > limit) {
    throw new Error(
      `描述共 ${inputTokens} 个 token，超过输入上限 ${limit} 个（管理员可调，最高 ${MAX_INPUT_TOKENS}），已拒绝生成。请缩短描述。`,
    );
  }
  const leadTokens = countContentTokens(tokenizer, lead);
  if (leadTokens > perChunk) {
    throw new Error(
      `管理员的更正与知识库共 ${leadTokens} 个 token，超过单段 CLIP 的 ${perChunk} 个可用 token，无法完整放入提示词，已拒绝生成。` +
        '请在管理员面板中删除或精简部分更正、知识库条目（中文每个字通常占多个 token）。',
    );
  }
  const promptTokens = countContentTokens(tokenizer, prompt);
  return {
    inputTokens,
    maxInputTokens: limit,
    leadTokens,
    promptTokens,
    capacity,
    truncatedTokens: Math.max(0, promptTokens - capacity),
  };
}

/**
 * 把前置内容拼成一个字符串（与 params.js 的 buildImagePrompt 使用相同的规则：去空白、去空项、以 ', ' 连接）。
 */
export function joinLeadTexts(lead = []) {
  return lead.map((t) => String(t || '').trim()).filter(Boolean).join(', ');
}
