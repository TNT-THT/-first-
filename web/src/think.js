/**
 * 推理模型（如 Qwen3、DeepSeek-R1）的输出包含思考过程（<think>…</think>）和最终回答。
 *
 * 处理三种常见形式：
 * 1. 完整的 <think>…</think> 段落；
 * 2. 只有结束标签 </think>（开始标签已在提示模板中给出）；
 * 3. 流式输出中尚未闭合的 <think>…（思考仍在进行中）。
 */

const OPEN = '<think>';
const CLOSE = '</think>';

function tidy(s) {
  return String(s).replace(/^\s+|\s+$/g, '');
}

/**
 * 把模型输出拆分为思考过程与最终回答。
 * @returns {{thinking: string, answer: string, thinkingOpen: boolean}}
 *   thinkingOpen：思考尚未结束（只有开始标签，还没有结束标签）。
 * answer 与 stripThinking 的结果一致。
 */
export function splitThinking(raw) {
  const text = String(raw ?? '');
  const pieces = [];
  // 1. 先取出完整的 <think>…</think> 段落（可能有多段）
  let rest = text.replace(/<think>([\s\S]*?)<\/think>/gi, (_m, inner) => {
    pieces.push(inner);
    return '';
  });
  // 2. 只有结束标签：结束标签之前的内容是思考过程
  const closeIdx = rest.toLowerCase().lastIndexOf(CLOSE);
  if (closeIdx !== -1) {
    pieces.push(rest.slice(0, closeIdx));
    rest = rest.slice(closeIdx + CLOSE.length);
  }
  // 3. 尚未闭合的开始标签：之后的内容仍在思考中
  let thinkingOpen = false;
  const openIdx = rest.toLowerCase().indexOf(OPEN);
  if (openIdx !== -1) {
    pieces.push(rest.slice(openIdx + OPEN.length));
    rest = rest.slice(0, openIdx);
    thinkingOpen = true;
  }
  const thinking = pieces.map(tidy).filter(Boolean).join('\n\n');
  return { thinking, answer: rest.replace(/^\s+/, ''), thinkingOpen };
}

/** 去掉思考过程，只保留最终回答。 */
export function stripThinking(text) {
  if (!text) return '';
  return splitThinking(text).answer;
}

/** 是否包含思考内容（用于区分“只输出了思考、没有答案”的情况）。 */
export function hasThinking(text) {
  return /<\/?think>/i.test(text || '');
}
