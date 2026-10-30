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

/** 后续追问通常省略主题；遇到明显的“再找找/不对”等短追问，沿用最近一个用户问题作为搜索词。 */
export function searchInputForTurn(text, previousUserText = '') {
  const current = String(text || '').trim();
  const previous = String(previousUserText || '').trim();
  const shortFollowUp = current.length <= 24 && /^(?:你)?(?:再|继续|重新)?(?:找找|查查|搜索|搜一下|查一下|核实|看看|不对|不是|还有吗|呢|？|\?)|你在找找/.test(current.replace(/[\s，。！？!?]/g, ''));
  return shortFollowUp && previous ? previous : current;
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

/**
 * 常见专有名词的中英对照。维基百科、Stack Overflow 等英文来源对英文条目更全，
 * 中文问题只查中文站经常“查无此条”（例如“2026 年诺贝尔物理学奖”在中文站可能还没有条目），
 * 因此维基百科的英文站回退会先把这些词换成英文，再发**同一个**请求（不增加请求次数）。
 * 只收录确定、无歧义的写法；没有命中时不会生成英文查询。
 */
export const TERM_GLOSSARY = {
  诺贝尔物理学奖: 'Nobel Prize in Physics',
  诺贝尔化学奖: 'Nobel Prize in Chemistry',
  诺贝尔生理学或医学奖: 'Nobel Prize in Physiology or Medicine',
  诺贝尔医学奖: 'Nobel Prize in Physiology or Medicine',
  诺贝尔文学奖: 'Nobel Prize in Literature',
  诺贝尔和平奖: 'Nobel Peace Prize',
  诺贝尔经济学奖: 'Nobel Memorial Prize in Economic Sciences',
  图灵奖: 'Turing Award',
  菲尔兹奖: 'Fields Medal',
  阿贝尔奖: 'Abel Prize',
  普利策奖: 'Pulitzer Prize',
  奥斯卡: 'Academy Awards',
  世界杯: 'FIFA World Cup',
  奥运会: 'Olympic Games',
  人工智能: 'artificial intelligence',
  机器学习: 'machine learning',
  深度学习: 'deep learning',
  大语言模型: 'large language model',
  量子计算: 'quantum computing',
  区块链: 'blockchain',
};

/**
 * 中文里对检索没有帮助的词。只收多字词：单字（“有”“的”“在”）往往是词的一部分，
 * 删掉会破坏原意（例如“有向图” → “向图”），因此一律保留。
 */
const ZH_STOPWORDS = [
  '请问', '想问一下', '想问', '想知道', '帮我查一下', '帮我查', '帮我', '告诉我', '介绍一下', '解释一下',
  '的获得者是谁', '的获得者', '获得者', '的得主', '得主是谁', '得主', '是谁', '是什么', '有哪些', '是哪个',
  '为什么', '怎么样', '怎么办', '如何获取', '多少', '是否', '能不能', '有没有', '一下',
];

/** 英文里可以从查询中去掉的词：疑问词与冠词（保留 in / of / for 等介词，英文检索需要它们）。 */
const EN_STOPWORDS = ['what', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how', 'which', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'the', 'a', 'an'];

function stripStopwords(text, list) {
  let s = ` ${String(text || '')} `;
  for (const w of [...list].sort((a, b) => b.length - a.length)) {
    const re = /^[\x00-\x7f]+$/.test(w) ? new RegExp(`(^|\\s)${w}(?=\\s|$)`, 'gi') : new RegExp(w, 'g');
    s = s.replace(re, ' ');
  }
  return s.replace(/[\s\u3000]+/g, ' ').trim();
}

function tidy(text) {
  return String(text || '')
    .replace(/[，。！？、；：""''（）()【】\[\]…—?!,.;:"'()]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** 中文查询的关键词式写法：只去掉明显的疑问句式，保留专有名词与单字。 */
export function zhKeywords(text) {
  const q = tidy(stripStopwords(tidy(text), ZH_STOPWORDS));
  // 只清理句末残留的单字虚词（“是多少”→“多少”之后剩下的“是”），句中的单字一律保留
  return q.length > 2 ? q.replace(/[是的有在了呢吗吧啊]+$/, '').trim() || q : q;
}

/** 英文查询的关键词式写法：去掉疑问词与冠词，保留介词。 */
export function enKeywords(text) {
  return tidy(stripStopwords(tidy(text), EN_STOPWORDS));
}

/** 把中文里的常见专有名词换成英文；没有命中或翻不干净时返回空串。 */
export function translateQuery(text) {
  let s = String(text || '');
  let hit = false;
  for (const [zh, en] of Object.entries(TERM_GLOSSARY).sort((a, b) => b[0].length - a[0].length)) {
    if (s.includes(zh)) {
      s = s.split(zh).join(` ${en} `);
      hit = true;
    }
  }
  if (!hit) return '';
  s = s.replace(/[年月份]/g, ' ');
  s = tidy(stripStopwords(tidy(s), ZH_STOPWORDS));
  s = tidy(stripStopwords(s, EN_STOPWORDS));
  // 还残留大量汉字说明没有真正翻译过去，作为英文查询没有意义
  if (!/[a-z]{2,}/i.test(s)) return '';
  const cjk = (s.match(/[\u3400-\u9fff]/g) || []).length;
  if (cjk > s.length * 0.3) return '';
  return s.slice(0, 120);
}

/**
 * 真正发给搜索引擎的查询（只生成一个，**不增加请求次数**）：
 * 中文问题去掉“的获得者是谁”这类疑问句式；英文问题去掉 what/who/the 这类词。
 * 关键词式查询比整句疑问句更容易命中条目（维基百科尤其明显）。
 */
export function searchKeywords(text) {
  const cleaned = pickSentence(stripForQuery(text), 120) || String(text || '');
  if (!cleaned) return '';
  const narrowed = hasCjk(cleaned) ? zhKeywords(cleaned) : enKeywords(cleaned);
  return (narrowed.length >= 2 ? narrowed : cleaned).replace(/\s+/g, ' ').trim().slice(0, 120);
}

/**
 * 某个语言的站点该用的查询：中文问题在英文站回退时先翻成英文（同一个请求，只是换了更可能命中的词）。
 */
export function queryForLang(query, lang, { defaultLang = 'zh' } = {}) {
  if (!query) return query;
  if (lang && lang !== defaultLang && hasCjk(query)) return translateQuery(query) || query;
  return query;
}

/**
 * 通用相关性门槛：避免把搜索引擎的错配结果（不论主题）当作回答证据。
 * 数字/英文按词匹配，中文按相邻二字词匹配；若查询含明确年份，结果也必须提到该年份。
 */
export function filterRelevantResults(query, results) {
  const original = String(query || '').toLowerCase();
  const translated = translateQuery(query).toLowerCase();
  const tokenize = (value) => {
    const s = String(value || '').toLowerCase();
    const tokens = new Set(s.match(/[a-z]{3,}|\d{2,}/g) || []);
    const cjk = s.match(/[\u3400-\u9fff\uf900-\ufaff]+/g) || [];
    for (const run of cjk) {
      if (run.length === 1) tokens.add(run);
      for (let i = 0; i + 1 < run.length; i += 1) tokens.add(run.slice(i, i + 2));
    }
    return tokens;
  };
  const queryVariants = [original, translated].filter(Boolean).map((q) => ({
    year: q.match(/(?:19|20)\d{2}/)?.[0],
    terms: tokenize(q),
  }));
  return (Array.isArray(results) ? results : []).filter((r) => {
    const content = `${r?.title || ''} ${r?.snippet || ''} ${r?.extract || ''}`;
    const candidates = tokenize(content);
    return queryVariants.some(({ year, terms }) => {
      if (!terms.size || (year && !candidates.has(year))) return false;
      // 未命中词典翻译的中文查询可能拿到英文站结果；此时无法跨语种逐词比对，
      // 只做年份约束（如有），避免误删确实相关的英文页面。
      if (hasCjk(original) && !hasCjk(r?.title) && /[a-z]{3,}/i.test(r?.title || '') && !translated) return true;
      let matches = 0;
      for (const term of terms) if (candidates.has(term)) matches += 1;
      return matches >= (terms.size >= 5 ? 2 : 1) && matches / terms.size >= 0.18;
    });
  });
}

