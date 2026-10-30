/**
 * 从用户消息中提取可安全外发的搜索词（纯逻辑，便于测试）。
 *
 * 刷题场景常见“题面 + 代码 + 日志”。搜索只应拿问题主题，绝不能把
 * 代码、命令行或疑似凭据当作查询词送给第三方。检测到疑似凭据时宁可
 * 整次跳过联网资料检索，也不尝试“聪明地”猜测哪一段可以泄露。
 */

const FENCED_CODE = /```[\s\S]*?(?:```|$)/g;
const INLINE_CODE = /`[^`\n]*`/g;
const URL_RE = /https?:\/\/[^\s]+/g;

const SECRET_PATTERNS = [
  { kind: '私钥', re: /-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----/i },
  { kind: '访问令牌', re: /\b(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|glpat-[A-Za-z0-9_-]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AIza[A-Za-z0-9_-]{12,}|sk-(?:proj-)?[A-Za-z0-9_-]{8,})\b/i },
  {
    kind: '凭据赋值',
    // 值再短也不应外发；宁可把“password=x”之类的代码示例当作敏感内容而跳过检索。
    re: /\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|secret|password|passwd|private[_-]?key|client[_-]?secret|session[_-]?token|cookie)\b\s*[:=]\s*["']?[A-Za-z0-9_.~+\/=:-]{1,}/i,
  },
  { kind: '授权头', re: /\b(?:authorization|proxy-authorization)\s*:\s*(?:bearer|basic|token)\s+\S+/i },
  { kind: 'Bearer 凭据', re: /\bbearer\s+[A-Za-z0-9._~+/=-]{8,}/i },
];

/** 返回疑似敏感内容类型；只返回类型，绝不把原始值回显到 UI 或日志。 */
export function findSensitiveContent(text) {
  const value = String(text || '');
  for (const item of SECRET_PATTERNS) {
    if (item.re.test(value)) return item.kind;
  }
  return '';
}

/** 看着像代码/命令/配置的行。宁可不搜，也不把粘贴的凭据或命令外发。 */
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
  // shell 命令和常见配置赋值以前会漏掉，尤其容易把 Authorization / API key 发出去。
  if (/^(?:[$#]\s*)?(?:curl|wget|http|httpie|ssh|scp|docker|kubectl|npm|npx|pnpm|yarn|pip(?:3)?|python(?:3)?|node|git|export|set)\b/i.test(t)) {
    return true;
  }
  if (/^(?:export\s+)?[A-Z][A-Z0-9_]{2,}\s*=/.test(t)) return true;
  if (/^[{[]\s*(?:["'][\w.-]+["']\s*:)?/.test(t)) return true;
  // 以 ; 或 { } 结尾，并且含有赋值、比较或调用等代码符号
  if (/[;{}]\s*$/.test(t) && /[=()<>[\]]/.test(t)) return true;
  // 明显缩进的行（题面里的编号列表一般只有 1–3 个空格）
  if (/^ {4,}\S/.test(raw)) return true;
  return false;
}

/** 去掉代码与链接，只留下自然语言。这里不负责敏感信息检测，见 analyzeSearchInput。 */
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

const QUERY_BOILERPLATE = /^(?:下面(?:是|附上)?(?:代码|题目|日志|报错)|(?:我的|有个)?(?:代码|题目|日志|报错)(?:如下|在下面)?|请看(?:下面|代码|题目))[：:。]?$/;
const QUESTION_HINT = /[？?]|(?:为什么|为何|怎么|如何|怎么办|是什么|是谁|哪里|多少|能否|有没有|是否|what|who|when|where|why|how|error|exception|fail|issue)/i;

/**
 * 从清洗后的多行文字中选最像真实问题的一行，避免“代码如下：”抢走查询位。
 * 不生成查询变体，也不把整段题面/对话发给搜索服务。
 */
function pickQueryLine(cleaned) {
  const lines = String(cleaned || '')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length >= 2 && !QUERY_BOILERPLATE.test(line));
  if (!lines.length) return '';
  return lines.find((line) => QUESTION_HINT.test(line)) || lines[0];
}

/**
 * 分析一次待检索消息。检测到疑似凭据时返回 blocked，而不是把敏感文本
 * 放进错误原因；调用方可继续正常向模型回答，但不会联网。
 */
export function analyzeSearchInput(text, { maxChars = 120 } = {}) {
  const sensitiveKind = findSensitiveContent(text);
  if (sensitiveKind) {
    return {
      query: '',
      blocked: true,
      reason: `检测到可能包含${sensitiveKind}，为避免外发已跳过联网资料检索`,
    };
  }
  const cleaned = stripForQuery(text);
  if (!cleaned) return { query: '', blocked: false, reason: '' };
  const line = pickQueryLine(cleaned) || cleaned;
  const sentence = pickSentence(line, maxChars) || line;
  return { query: sentence.replace(/\s+/g, ' ').trim().slice(0, maxChars), blocked: false, reason: '' };
}

/** 由用户消息生成安全的搜索查询（最多 maxChars 个字符）。 */
export function buildSearchQuery(text, options = {}) {
  return analyzeSearchInput(text, options).query;
}

/** 后续追问通常省略主题；遇到明显的“再找找/不对”等短追问，沿用最近一个用户问题作为搜索词。 */
export function searchInputForTurn(text, previousUserText = '') {
  const current = String(text || '').trim();
  const previous = String(previousUserText || '').trim();
  const shortFollowUp = current.length <= 24 && /^(?:你)?(?:再|继续|重新)?(?:找找|查查|搜索|搜一下|查一下|核实|看看|不对|不是|还有吗|呢|？|\?)|你在找找/.test(current.replace(/[\s，。！？!?]/g, ''));
  return shortFollowUp && previous ? previous : current;
}

/**
 * 维基百科的优先语言：有中文时先中文站，否则先英文站。
 * 英文回退仍会发生；若没有可靠翻译，相关性过滤会宁可丢弃无法核实的英文结果。
 */
export function wikipediaLangs(query, { lang = 'zh', fallbackLang = 'en' } = {}) {
  const first = hasCjk(query) ? lang : 'en';
  const second = first === lang ? fallbackLang : lang;
  return [...new Set([first, second].filter(Boolean))];
}

/** 常见专有名词与刷题术语的中英对照；没有命中时不硬翻。 */
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
  二分查找: 'binary search',
  二分搜索: 'binary search',
  时间复杂度: 'time complexity',
  最短路径: 'shortest path',
  动态规划: 'dynamic programming',
  深度优先搜索: 'depth first search',
  广度优先搜索: 'breadth first search',
  并查集: 'disjoint set union',
};

/** 中文里对检索没有帮助的词。 */
const ZH_STOPWORDS = [
  '请问', '想问一下', '想问', '想知道', '帮我查一下', '帮我查', '帮我', '告诉我', '介绍一下', '解释一下',
  '的获得者是谁', '的获得者', '获得者', '的得主', '得主是谁', '得主', '是谁', '是什么', '有哪些', '是哪个',
  '为什么', '怎么样', '怎么办', '如何获取', '多少', '是否', '能不能', '有没有', '一下',
];

/** 英文里可以从查询中去掉的词：疑问词与冠词。 */
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
  if (!/[a-z]{2,}/i.test(s)) return '';
  const cjk = (s.match(/[\u3400-\u9fff]/g) || []).length;
  if (cjk > s.length * 0.3) return '';
  return s.slice(0, 120);
}

/** 真正发给搜索引擎的关键词式查询。 */
export function searchKeywords(text) {
  const analyzed = analyzeSearchInput(text);
  if (analyzed.blocked || !analyzed.query) return '';
  const narrowed = hasCjk(analyzed.query) ? zhKeywords(analyzed.query) : enKeywords(analyzed.query);
  return (narrowed.length >= 2 ? narrowed : analyzed.query).replace(/\s+/g, ' ').trim().slice(0, 120);
}

/** 某个语言站点的查询：中文问题在英文站回退时优先用确定的词典翻译。 */
export function queryForLang(query, lang, { defaultLang = 'zh' } = {}) {
  if (!query) return query;
  if (lang && lang !== defaultLang && hasCjk(query)) return translateQuery(query) || query;
  return query;
}

function tokenize(value) {
  const s = String(value || '').toLowerCase();
  const tokens = new Set(s.match(/[a-z]{3,}|\d{2,}/g) || []);
  const cjk = s.match(/[\u3400-\u9fff\uf900-\ufaff]+/g) || [];
  for (const run of cjk) {
    if (run.length === 1) tokens.add(run);
    for (let i = 0; i + 1 < run.length; i += 1) tokens.add(run.slice(i, i + 2));
  }
  return tokens;
}

/**
 * 保守的相关性门槛：无可靠的跨语言翻译时，不把“英文标题”当作相关性的替代品。
 * 查询有年份时，候选资料必须出现同一年；这只是最低门槛，不把它伪装成时效验证。
 */
export function filterRelevantResults(query, results) {
  const safeQuery = buildSearchQuery(query) || String(query || '');
  const keywordQuery = hasCjk(safeQuery) ? zhKeywords(safeQuery) : enKeywords(safeQuery);
  const translated = translateQuery(keywordQuery).toLowerCase();
  const variants = [keywordQuery.toLowerCase(), translated]
    .filter(Boolean)
    .map((q) => ({ year: q.match(/(?:19|20)\d{2}/)?.[0], terms: tokenize(q) }));
  return (Array.isArray(results) ? results : []).filter((r) => {
    const content = `${r?.title || ''} ${r?.snippet || ''} ${r?.extract || ''}`;
    const candidates = tokenize(content);
    return variants.some(({ year, terms }) => {
      if (!terms.size || (year && !candidates.has(year))) return false;
      let matches = 0;
      for (const term of terms) if (candidates.has(term)) matches += 1;
      // 词典翻译后的中文查询不能因一个泛词（例如 search）就接纳任意英文页面。
      // 三个及以上术语至少命中两个；短查询仍允许一个精确术语命中。
      const required = terms.size >= 3 ? 2 : 1;
      return matches >= required && matches / terms.size >= 0.2;
    });
  });
}
