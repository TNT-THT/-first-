/**
 * 去掉推理模型（如 Qwen3、DeepSeek-R1）输出的思考过程，只保留最终回答。
 *
 * 处理三种常见形式：
 * 1. 完整的 <think>…</think> 段落；
 * 2. 只有结束标签 </think>（开始标签已在提示模板中给出）；
 * 3. 流式输出中尚未闭合的 <think>…（只隐藏，等待结束标签到达）。
 */
export function stripThinking(text) {
  if (!text) return '';
  let out = String(text).replace(/<think>[\s\S]*?<\/think>/gi, '');
  const closeIdx = out.toLowerCase().lastIndexOf('</think>');
  if (closeIdx !== -1) out = out.slice(closeIdx + '</think>'.length);
  const openIdx = out.toLowerCase().indexOf('<think>');
  if (openIdx !== -1) out = out.slice(0, openIdx);
  return out.replace(/^\s+/, '');
}

/** 是否包含思考内容（用于区分“只输出了思考、没有答案”的情况）。 */
export function hasThinking(text) {
  return /<\/?think>/i.test(text || '');
}
