/**
 * 生图提示词的 CLIP 预算（纯逻辑，不依赖浏览器）。
 *
 * - CLIP 文本编码器只读取 77 个 token，其中首尾各占 1 个（BOS、EOS），中间约 75 个可用。超出部分会被静默截断。
 * - 管理员的更正与知识库放在提示词最前面，并且必须生效：它们自身超出预算时拒绝生成（与聊天 AI 的
 *   “更正与知识库不会被截断、超出窗口则报错”一致），而不是丢掉其中的一部分。
 * - 用户的画面描述超出预算时仍然生成，但要告知截断了多少 token（截断从末尾开始，即风格提示先被截掉）。
 *
 * 注意：CLIP 不会“服从”文字指令。这里保证的是：管理员内容位于最前、完整放入、不被截断。
 */

export const CLIP_CONTEXT_TOKENS = 77;
/** 去掉 BOS 与 EOS 之后的可用 token 数。 */
export const CLIP_CONTENT_TOKENS = CLIP_CONTEXT_TOKENS - 2;

/**
 * 描述（文本框）的输入上限（token）。默认 4096；管理员可用参数行调整，最高 16384（硬上限）。
 * 注意：这只限制输入框接受的长度。SD 1.5 的 CLIP 只读取约 75 个 token，超出部分不会参与生图。
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
 * - lead：管理员前置内容（更正与知识库，已用 ', ' 连接，与 buildImagePrompt 的前缀一致）；超出 CLIP 预算时抛出错误。
 * - prompt：完整提示词；返回 truncatedTokens（末尾被截断的 token 数，0 表示完整放入）。
 * 返回 {inputTokens, maxInputTokens, leadTokens, promptTokens, capacity, truncatedTokens}。
 */
export function checkPromptBudget(
  tokenizer,
  { input = null, maxInputTokens = DEFAULT_INPUT_TOKENS, lead = '', prompt = '' } = {},
) {
  const max = Number.isInteger(tokenizer.maxLength) ? tokenizer.maxLength : CLIP_CONTEXT_TOKENS;
  const capacity = max - 2;
  // 输入上限先于 CLIP 检查：描述本身超过上限时，直接告知管理员可调的上限
  const limit = Number.isInteger(maxInputTokens) ? maxInputTokens : DEFAULT_INPUT_TOKENS;
  const inputTokens = input === null ? 0 : countContentTokens(tokenizer, input);
  if (input !== null && inputTokens > limit) {
    throw new Error(
      `描述共 ${inputTokens} 个 token，超过输入上限 ${limit} 个（管理员可调，最高 ${MAX_INPUT_TOKENS}），已拒绝生成。请缩短描述。`,
    );
  }
  const leadTokens = countContentTokens(tokenizer, lead);
  if (leadTokens > capacity) {
    throw new Error(
      `管理员的更正与知识库共 ${leadTokens} 个 token，超过 CLIP 的 ${capacity} 个可用 token，无法完整放入提示词，已拒绝生成。` +
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
