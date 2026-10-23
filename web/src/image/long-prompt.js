/**
 * 长提示词分段（纯逻辑）。
 *
 * CLIP 每段只能读取 77 个 token。超过单段的提示词拆成多段，每段独立编码，
 * 再把各段的文本嵌入在 token 维度上拼接，供 UNet 交叉注意力读取。
 * 每段布局为 [BOS, 最多 75 个内容 token, EOS, PAD...]，与 tokenizer.encode 的单段结果一致；
 * 因此不超过 75 个内容 token 时只有一段，与原来的行为逐位相同。
 */
import { CLIP_CONTENT_TOKENS, CLIP_CONTEXT_TOKENS, MAX_PROMPT_CHUNKS } from './prompt-budget.js';

/**
 * 把文本切成至多 maxChunks 段。超出 maxChunks × 75 的内容被截断（末尾），并报告 truncatedTokens。
 * 返回 {chunks: Int32Array[], contentTokens, truncatedTokens}；chunks 至少有一段。
 */
export function splitPromptIntoChunks(tokenizer, text, { maxChunks = 1 } = {}) {
  const chunkLimit = Number.isInteger(maxChunks) && maxChunks >= 1 ? Math.min(maxChunks, MAX_PROMPT_CHUNKS) : 1;
  const ids = tokenizer.rawTokenize(String(text || '').trim());
  const content = ids.slice(1, -1);
  const keep = Math.min(content.length, chunkLimit * CLIP_CONTENT_TOKENS);
  const kept = content.slice(0, keep);
  const count = Math.max(1, Math.ceil(kept.length / CLIP_CONTENT_TOKENS));
  const chunks = [];
  for (let c = 0; c < count; c++) {
    const piece = kept.slice(c * CLIP_CONTENT_TOKENS, (c + 1) * CLIP_CONTENT_TOKENS);
    const out = new Int32Array(CLIP_CONTEXT_TOKENS).fill(tokenizer.pad);
    out.set([tokenizer.bos, ...piece, tokenizer.eos]);
    chunks.push(out);
  }
  return { chunks, contentTokens: content.length, truncatedTokens: content.length - keep };
}
