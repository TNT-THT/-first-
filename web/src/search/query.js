/**
 * 从用户消息中提取搜索关键词（纯逻辑，便于测试）。
 *
 * 刷题场景常见的消息是“题面 + 输入输出格式 + 一大段代码”。整段丢给搜索引擎几乎搜不到东西，
 * 因此这里先把代码块、行内代码、链接和明显的代码行去掉，再取第一句作为查询。
 * 宁可查询短一点：词语太多会互相牵制，检索引擎的召回反而更差。
 */

const FENCED_CODE = /```[\s\S]*?(?:```|$)/g;
const INLINE_CODE = /`[^`\n]*`/g;
const URL_RE = /https?:\/\/[^\s]+/g;

/** 看着像代码的行（声明、控制流、以分号或花括号结尾的语句、缩进 4 格以上的行）。 */
export function isCodeLine(line) {
  const raw = String(line || '');
  const t = raw.trim();
  if (!t) return false;
  if (
    /^(#include|#define|using\s+namespace|import\s|from\s+\S+\s+import|def\s|class\s|public\s|private\s|protected\s|static\s|const\s|let\s|var\s|function\s|package\s|template\s*<|struct\s|int\s+main|void\s+main|for\s*\(|while\s*\(|if\s*\(|return\b|print\s*\(|console\.log|System\.out)/.test(
      t,
    )
  ) {
    return true;
  }
  // 以 ; 或 { } 结尾，并且含有赋值、比较或调用等代码符号
  if (/[;{}]\s*$/.test(t) && /[=()<>[\]]/.test(t)) return true;
  // 明显缩进的行（题面里的编号列表一般只有 1–3 个空格）
  if (/^ {4,}\S/.test(raw)) return true;
  return false;
}

/** 去掉代码与链接，只留下自然语言。 */
export function stripForQuery(text) {
  let s = String(text || '');
  s = s.replace(FENCED_CODE, ' ');
  s = s.replace(INLINE_CODE, ' ');
  s = s.replace(URL_RE, ' ');
  s = s
    .split('\n')
    .filter((line) => !isCodeLine(line))
    .join('\n');
  return s.replace(/[ \t]+/g, ' ').replace(/\n{2,}/g, '\n').trim();
}

export function hasCjk(text) {
  return /[\u3400-\u9fff\uf900-\ufaff]/.test(String(text || ''));
}

/** 取第一个“完整的意思”：按中文/英文句末标点切分，太短就继续拼下一句。 */
export function pickSentence(text, maxChars = 120) {
  const s = String(text || '').trim();
  if (!s) return '';
  const parts = s.split(/(?<=[。！？!?;；])/);
  let picked = '';
  for (const part of parts) {
    picked += part;
    if (picked.trim().length >= 4) break;
    if (picked.length > maxChars) break;
  }
  const q = picked.trim();
  if (q.length <= maxChars) return q;
  return q.slice(0, maxChars);
}

/**
 * 由用户消息生成搜索查询（最多 maxChars 个字符）。
 * 清洗后没有自然语言（例如整条都是代码）时返回空串：此时不搜索，直接让模型回答。
 */
export function buildSearchQuery(text, { maxChars = 120 } = {}) {
  const cleaned = stripForQuery(text);
  if (!cleaned) return '';
  const firstLine = cleaned.split('\n').find((line) => line.trim().length >= 2) || cleaned;
  const sentence = pickSentence(firstLine, maxChars) || firstLine;
  return sentence.replace(/\s+/g, ' ').trim().slice(0, maxChars);
}

/**
 * 维基百科的优先语言：有中文时先查中文站，否则先查英文站（算法术语的英文条目通常更全）。
 * 返回去重后的语言列表，供“第一个站点没有结果时再试第二个”使用。
 */
export function wikipediaLangs(query, { lang = 'zh', fallbackLang = 'en' } = {}) {
  const first = hasCjk(query) ? lang : 'en';
  const second = first === lang ? fallbackLang : lang;
  return [...new Set([first, second].filter(Boolean))];
}
