/**
 * 从用户消息中提取可安全外发的搜索词（纯逻辑，便于测试）。
 *
 * 刷题场景常见“题面 + 代码 + 日志”。搜索只应拿问题主题，绝不能把
 * 代码、命令行或疑似凭据当作查询词送给第三方。检测到疑似凭据时宁可
 * 整次跳过联网资料检索，也不尝试“聪明地猜测”哪一段可以泄露。
 *
 * 这一层是「检索质量」的第一道关：查询词是否忠实表达用户的问题，
 * 直接决定后面所有来源的命中率。因此这里的取舍是：
 * - 含中文的行按自然语言处理（真实命令几乎不会整行夹中文）；
 * - 识别 shell 命令时要求「工具名 + 参数/子命令」，而不是见到工具名就判死；
 * - 相关性判断从「硬删」改为「加权打分 + 阈值」，短技术词与英文词形变化都要能命中。
 */

const FENCED_CODE = /```[\s\S]*?(?:```|$)/g;
const INLINE_CODE = /`[^`\n]*`/g;
const URL_RE = /https?:\/\/[^\s]+/g;
const CJK_RE = /[\u3400-\u9fff\uf900-\ufaff]/;

const SECRET_PATTERNS = [
  { kind: '私钥', re: /-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----/i },
  { kind: '访问令牌', re: /\b(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|glpat-[A-Za-z0-9_-]{8,}|xox[baprs]-[A-Za-z0-9-]{8,}|AIza[A-Za-z0-9_-]{12,}|sk-(?:proj-|live-)?[A-Za-z0-9_-]{8,})\b/i },
  {
    kind: '凭据赋值',
    // 值再短也不应外发；宁可把“password=x”之类的代码示例当作敏感内容而跳过检索。
    re: /\b(?:api[_-]?key|access[_-]?token|auth(?:orization)?|secret|password|passwd|private[_-]?key|client[_-]?secret|session[_-]?token|cookie)\b\s*[:=]\s*["']?[A-Za-z0-9_.~+\/=:-]{1,}/i,
  },
  {
    kind: '凭据字面量',
    // JSON/对象/调用实参形态："password": "..."、"token", "ghp_..."（setItem 等第二参数）。
    re: /["'](?:password|passwd|pwd|pass|secret|api[_-]?key|apikey|access[_-]?token|auth[_-]?token|token|session[_-]?token|client[_-]?secret|private[_-]?key|cookie)["']\s*[,:]\s*["'][^"']{1,}["']/i,
  },
  {
    kind: '凭据存储调用',
    // localStorage/sessionStorage 对凭据类键名的读写，例如 setItem("password", "abc123")。
    re: /\b(?:local|session)Storage\s*\.\s*(?:set|get|remove)Item\s*\(\s*["'`][^"'`]*(?:passw|pwd|secret|token|api[_-]?key|apikey|auth|cookie|private[_-]?key)[^"'`]*["'`]/i,
  },
  { kind: '授权头', re: /\b(?:authorization|proxy-authorization)\s*:\s*(?:bearer|basic|token)\s+\S+/i },
  { kind: 'Bearer 凭据', re: /\bbearer\s+[A-Za-z0-9._~+/=-]{8,}/i },
  { kind: 'JWT', re: /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}/ },
  { kind: '云服务密钥', re: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/ },
  // 带密码的数据库/消息队列连接串：postgres://user:pass@host 之类。http(s) URL 由 stripForQuery 单独移除。
  { kind: '连接串凭据', re: /\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|amqp|mssql|clickhouse):\/\/[^\s:@\/]+:[^\s@\/]+@/i },
  // mysql -u root -pSecret / mysqldump --password=...
  { kind: '命令行密码', re: /\b(?:mysql|mariadb|mysqldump|psql|pg_dump|redis-cli)\b[^\n]*?(?:\s-p\S+|\s--password[= ]\S+)/i },
  // 中文里直接写出密码值：“我的密码是 8f3a9c2b” / “密钥：sk-xxx”。
  {
    kind: '中文凭据',
    re: /(?:密码|口令|密钥|秘钥|私钥|token|令牌)\s*(?:是|为|＝|=|：|:)\s*[A-Za-z0-9!@#$%^&*_.+\/-]{4,}/i,
  },
  { kind: 'Webhook', re: /https:\/\/hooks\.slack\.com\/services\/[A-Za-z0-9/_-]{8,}/i },
];

/** 返回疑似敏感内容类型；只返回类型，绝不把原始值回显到 UI 或日志。 */
export function findSensitiveContent(text) {
  const value = String(text || '');
  for (const item of SECRET_PATTERNS) {
    if (item.re.test(value)) return item.kind;
  }
  return '';
}

/** 常见命令行工具及其子命令：只有「工具名 + 参数/子命令/文件名」才当作命令。 */
const SHELL_TOOLS = new Set([
  'curl', 'wget', 'http', 'httpie', 'ssh', 'scp', 'docker', 'kubectl', 'npm', 'npx', 'pnpm', 'yarn',
  'pip', 'pip3', 'python', 'python3', 'node', 'git', 'export', 'set', 'mysql', 'mysqldump', 'psql',
  'pg_dump', 'redis-cli', 'mvn', 'gradle', 'cargo', 'go', 'java', 'javac', 'gcc', 'g++', 'clang',
]);
const SHELL_SUBCOMMANDS = {
  git: ['add', 'commit', 'push', 'pull', 'rebase', 'merge', 'checkout', 'switch', 'reset', 'status', 'log', 'diff', 'stash', 'clone', 'branch', 'remote', 'fetch', 'cherry-pick', 'revert', 'tag', 'config', 'init', 'show'],
  npm: ['install', 'i', 'add', 'remove', 'uninstall', 'run', 'test', 'init', 'publish', 'ci', 'exec', 'create', 'start', 'build', 'dev', 'update', 'audit'],
  npx: ['create', 'run', 'exec'],
  pnpm: ['install', 'i', 'add', 'remove', 'run', 'test', 'dlx', 'create'],
  yarn: ['install', 'add', 'remove', 'run', 'test', 'create', 'init', 'build'],
  docker: ['run', 'build', 'pull', 'push', 'ps', 'exec', 'compose', 'images', 'rmi', 'logs', 'start', 'stop', 'rm', 'tag', 'login', 'inspect'],
  kubectl: ['get', 'apply', 'describe', 'logs', 'exec', 'delete', 'create', 'port-forward', 'rollout', 'config', 'top'],
  cargo: ['build', 'run', 'test', 'add', 'new', 'init', 'publish', 'check', 'clippy', 'fmt'],
  mvn: ['clean', 'install', 'package', 'test', 'compile', 'deploy', 'spring-boot:run'],
  gradle: ['build', 'run', 'test', 'clean', 'assemble'],
  go: ['build', 'run', 'test', 'get', 'mod', 'install', 'vet', 'fmt'],
};

function looksLikeShellCommand(t) {
  const match = /^(?:[$#]\s*)?([a-z][\w.+-]*)\b(.*)$/i.exec(t);
  if (!match) return false;
  const tool = match[1].toLowerCase();
  if (!SHELL_TOOLS.has(tool)) return false;
  const rest = match[2].trim();
  if (!rest) return false; // 单个工具名（"docker"）当查询词仍有意义，不判为代码
  // 参数、路径、URL、文件名、重定向：几乎只出现在真实命令里。
  if (/^[-/]/.test(rest)) return true;
  if (/^(?:[^\s]+\.[a-z0-9]{1,4})\b/i.test(rest)) return true;
  if (/[|><]/.test(rest)) return true;
  const first = rest.split(/\s+/)[0].toLowerCase();
  if ((SHELL_SUBCOMMANDS[tool] || []).includes(first)) {
    // “npm install fails / git merge conflict resolution”是自然语言，不是命令。
    return !/\b(?:why|how|what|error|errors|fail|fails|failed|issue|problem|difference|conflict|resolution|explain|explained|tutorial|guide|meaning|purpose|vs|versus|crash|crashes|broken|example|examples|practice)\b/i.test(
      t,
    );
  }
  return false;
}

/**
 * 看着像代码/命令/配置的行。宁可不搜，也不把粘贴的凭据或命令外发。
 * 但「整行含中文」几乎只可能是自然语言（"Python 列表和元组的区别"），
 * 此时若仍按 shell 命令丢弃，会把一整类正常提问变成空查询。
 */
export function isCodeLine(line) {
  const raw = String(line || '');
  const t = raw.trim();
  if (!t) return false;
  // 环境变量/配置赋值：即使夹带中文也按代码处理（可能含密钥）。
  if (/^(?:export\s+)?[A-Z][A-Z0-9_]{2,}\s*=/.test(t)) return true;
  // 明显缩进的行（题面里的编号列表一般只有 1–3 个空格）。
  if (/^ {4,}\S/.test(raw)) return true;
  // 以 ; 或 { } 结尾，并且含有赋值、比较或调用等代码符号。
  if (/[;{}]\s*$/.test(t) && /[=()<>[\]]/.test(t)) return true;
  // 中文行按自然语言处理；真正的代码块/行内代码已在 stripForQuery 中移除。
  if (CJK_RE.test(t)) return false;
  if (
    /^(#include|#define|using\s+namespace|import\s|from\s+\S+\s+import|def\s|class\s|public\s|private\s|protected\s|static\s|const\s|let\s|var\s|function\s|package\s|template\s*<|struct\s|int\s+main|void\s+main|for\s*\(|while\s*\(|if\s*\(|return\b|print\s*\(|console\.log|System\.out)/.test(
      t,
    )
  ) {
    return true;
  }
  if (looksLikeShellCommand(t)) return true;
  if (/^[{[]\s*(?:["'][\w.-]+["']\s*:)?/.test(t)) return true;
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
  return CJK_RE.test(String(text || ''));
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
/** 纯客套/程序性行：作为查询词没有信息量，“能帮我看看吗”不该抢走真正的问题。 */
const POLITE_BOILERPLATE =
  /^(?:你|您)?(?:能|可不可以|可以|麻烦|帮忙|帮|请|帮忙看一下)?(?:帮我|帮我看看|看看|看一下|瞧瞧|检查|排查|分析|解答|回答|处理)?(?:看看|一下|下|这个问题|这个)?(?:吗|么|嘛)?[。！？!?～~]*$/;

/**
 * 从清洗后的多行文字中选最像真实问题的一行，避免“代码如下：”或“能帮我看看吗”抢走查询位。
 * 评分规则：疑问句加分；去掉停用词后仍有较多内容加分；纯客套/样板行重罚；
 * 同分时偏向靠后的行（多行消息通常最后一句才是问题）。不生成查询变体，也不把整段题面发给搜索服务。
 */
function pickQueryLine(cleaned) {
  const lines = String(cleaned || '')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length >= 2 && !QUERY_BOILERPLATE.test(line));
  if (!lines.length) return '';
  if (lines.length === 1) return lines[0];
  let best = lines[0];
  let bestScore = -Infinity;
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    let score = QUESTION_HINT.test(line) ? 2 : 0;
    if (POLITE_BOILERPLATE.test(line)) score -= 4;
    // 内容丰富度：去掉停用词后剩下的实际文本越长，越可能是真正的问题主题。
    const keywords = hasCjk(line) ? zhKeywords(line) : enKeywords(line);
    score += Math.min(6, keywords.length / 4);
    score += i * 0.01;
    if (score > bestScore + 1e-9) {
      bestScore = score;
      best = line;
    }
  }
  return best;
}

/**
 * 分析一次待检索消息。检测到疑似凭据时返回 blocked，而不是把敏感文本
 * 放进错误原因；调用方可继续正常模型回答，但不会联网。
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

/** 寒暄、致谢、确认类消息：联网检索没有意义，不应该产生任何请求。 */
const TRIVIAL_MESSAGE =
  /^(?:你好|您好|hi|hello|hey|谢谢|多谢|感谢|thanks|thank you|继续|好的|好|嗯|ok|okay|收到|明白|懂了|再见|bye|测试|test)[\s!！。.,，~～?？]*$/i;

/** 极短或纯寒暄的查询：调用方应显示「已跳过」，而不是「检索失败」。 */
export function isTrivialQuery(text) {
  const s = String(text || '').trim();
  if (!s) return true;
  if (TRIVIAL_MESSAGE.test(s)) return true;
  return s.replace(/[\s\p{P}\p{S}]/gu, '').length < 2;
}

/**
 * 这条消息值不值得联网：纯代码/空白/寒暄返回 false（0 请求），
 * 含疑似凭据仍返回 true，让 runWebSearch 给出 fail-closed 的明确提示。
 */
export function isSearchWorthwhile(text) {
  const analyzed = analyzeSearchInput(text);
  if (analyzed.blocked) return { ok: true, reason: '', query: '' };
  if (!analyzed.query) return { ok: false, reason: '没有可检索的文本（纯代码或空白）', query: '' };
  if (isTrivialQuery(analyzed.query)) return { ok: false, reason: '寒暄或极短消息，无需联网检索', query: analyzed.query };
  return { ok: true, reason: '', query: analyzed.query };
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

/**
 * 常见专有名词与刷题/工程术语的中英对照；没有命中时不硬翻。
 * 这张表同时承担两个职责：把中文概念翻成可检索的英文术语，以及
 * 让 SO/HN 等英文源拿到「保留问题意图」的查询（而不只是一个条目名）。
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
  人工智能对齐: 'AI alignment',
  对齐: 'alignment',
  对齐问题: 'AI alignment',
  人工智能: 'artificial intelligence',
  机器学习: 'machine learning',
  深度学习: 'deep learning',
  大语言模型: 'large language model',
  大模型: 'large language model',
  神经网络: 'neural network',
  强化学习: 'reinforcement learning',
  量子计算: 'quantum computing',
  区块链: 'blockchain',
  红黑树: 'red-black tree',
  平衡二叉树: 'balanced binary tree',
  平衡二叉搜索树: 'balanced binary search tree',
  二叉搜索树: 'binary search tree',
  二叉查找树: 'binary search tree',
  二叉树: 'binary tree',
  哈希表: 'hash table',
  一致性哈希: 'consistent hashing',
  链表: 'linked list',
  双向链表: 'doubly linked list',
  优先队列: 'priority queue',
  并查集: 'disjoint set union',
  线段树: 'segment tree',
  树状数组: 'binary indexed tree',
  后缀数组: 'suffix array',
  字符串匹配: 'string matching',
  最短路径: 'shortest path',
  最小生成树: 'minimum spanning tree',
  拓扑排序: 'topological sort',
  二分查找: 'binary search',
  二分搜索: 'binary search',
  快速排序: 'quicksort',
  归并排序: 'merge sort',
  堆排序: 'heapsort',
  动态规划: 'dynamic programming',
  贪心算法: 'greedy algorithm',
  回溯算法: 'backtracking',
  分治算法: 'divide and conquer',
  时间复杂度: 'time complexity',
  空间复杂度: 'space complexity',
  保持平衡: 'maintain balance',
  旋转操作: 'rotation',
  虚函数表: 'virtual method table',
  虚拟函数表: 'virtual method table',
  多态: 'polymorphism',
  循环依赖: 'circular dependency',
  死锁: 'deadlock',
  竞态条件: 'race condition',
  内存泄漏: 'memory leak',
  内存溢出: 'out of memory',
  栈溢出: 'stack overflow',
  段错误: 'segmentation fault',
  空指针: 'null pointer',
  野指针: 'dangling pointer',
  事件循环: 'event loop',
  协程: 'coroutine',
  线程池: 'thread pool',
  线程: 'thread',
  进程: 'process',
  垃圾回收: 'garbage collection',
  依赖注入: 'dependency injection',
  响应式原理: 'reactivity',
  中间件: 'middleware',
  微服务: 'microservice',
  连接池: 'connection pool',
  隔离级别: 'isolation level',
  分布式锁: 'distributed lock',
  消息队列: 'message queue',
  负载均衡: 'load balancing',
  反向代理: 'reverse proxy',
  跨域: 'CORS',
  同源策略: 'same-origin policy',
  持久化: 'persistence',
  主从复制: 'replication',
  分库分表: 'sharding',
  幂等: 'idempotent',
  序列化: 'serialization',
  正则表达式: 'regular expression',
  虚拟机: 'virtual machine',
  内存管理: 'memory management',
  引用传递: 'pass by reference',
  值传递: 'pass by value',
  闭包: 'closure',
  原型链: 'prototype chain',
  作用域: 'scope',
  柯里化: 'currying',
  装饰器: 'decorator',
  泛型: 'generics',
  异常处理: 'exception handling',
  单元测试: 'unit test',
  版本控制: 'version control',
  变基: 'rebase',
  回滚: 'rollback',
  有向图: 'directed graph',
  无向图: 'undirected graph',
  二分图: 'bipartite graph',
  状态压缩: 'bitmask DP',
  单调栈: 'monotonic stack',
  滑动窗口: 'sliding window',
  双指针: 'two pointers',
  前缀和: 'prefix sum',
  差分数组: 'difference array',
};

/** 中文里对检索没有帮助的词。 */
const ZH_STOPWORDS = [
  '请问', '想问一下', '想问', '想知道', '帮我查一下', '帮我查', '帮我', '告诉我', '介绍一下', '解释一下',
  '的获得者是谁', '的获得者', '获得者', '的得主', '得主是谁', '得主', '是谁', '是什么', '有哪些', '是哪个',
  '为什么', '怎么样', '怎么办', '如何获取', '多少', '是否', '能不能', '有没有', '一下',
  '这段代码', '我的代码', '这段', '我的', '一直', '怎么改', '怎么办', '怎么回事', '问题', '情况',
  '这', '这是', '怎么求', '如何求', '怎么写', '怎么回事',
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

/**
 * 中文疑问焦点 → 英文检索词。只映射能显著提升英文源命中的意图词，
 * 不做逐字翻译；翻译不完整时宁可退回条目名（见 composeEnQuery）。
 */
const INTENT_PATTERNS = [
  [/为什么|为何|什么原因|怎么回事/, 'why'],
  [/怎么|如何|怎样/, 'how'],
  [/区别|差异|不同|对比|哪个好|怎么选/, 'difference'],
  [/报错|错误|异常|失败|崩溃|闪退/, 'error'],
  [/超时|时间限制/, 'timeout'],
  [/优化|性能|更快|加速/, 'performance'],
  [/原理|机制|工作方式/, 'principle'],
  [/实现|写法|代码示例/, 'implementation'],
  [/复杂度/, 'complexity'],
  [/安装|配置|搭建/, 'setup'],
  [/用法|使用|怎么用/, 'usage'],
  [/保持平衡/, 'maintain balance'],
];

/** 把中文意图词替换成英文，返回替换后的文本与命中的意图词。 */
function replaceIntent(text) {
  let s = String(text || '');
  const hits = [];
  for (const [re, en] of INTENT_PATTERNS) {
    if (re.test(s)) {
      hits.push(en);
      s = s.replace(new RegExp(re.source, 'g'), ` ${en} `);
    }
  }
  return { text: s, hits };
}

/**
 * 把中文里的常见专有名词与疑问焦点换成英文；没有命中或翻不干净时返回空串。
 * 翻译结果用于 SO/HN 与英文维基：保留「问题意图」比只给一个条目名更有用。
 */
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
  const intent = replaceIntent(s);
  s = intent.text;
  s = s.replace(/[年月份]/g, ' ');
  // 术语替换后残留的结构虚词（“的”）对英文检索是噪音。
  s = s.replace(/的/g, ' ');
  s = tidy(stripStopwords(tidy(s), ZH_STOPWORDS));
  s = tidy(stripStopwords(s, EN_STOPWORDS));
  const cjkCount = (s.match(/[\u3400-\u9fff]/g) || []).length;
  const meaningful = s.replace(/[^A-Za-z0-9+#. -]/g, '').trim();
  if (!/[a-z]{2,}/i.test(meaningful)) return '';
  // 翻译不完整（中文仍占多数）时宁可放弃，也不把半中半英的查询发给英文源。
  if (cjkCount > meaningful.length * 0.35) return '';
  // 去掉残留的单字中文，得到纯英文关键词串。
  const cleaned = tidy(meaningful.replace(/[\u3400-\u9fff]/g, ' '));
  return cleaned.slice(0, 120);
}

/** 从中文问题里提取可直接用于英文源的技术片段（如 segmentation fault、CUDA out of memory）。 */
export function englishFragment(text) {
  const s = String(text || '');
  const runs = s.match(/[A-Za-z][A-Za-z0-9+#._-]*(?:\s+[A-Za-z0-9][A-Za-z0-9+#._-]*)+/g) || [];
  const singles = s.match(/\b[A-Za-z][A-Za-z0-9+#._-]{2,}\b/g) || [];
  const candidates = [...runs, ...singles]
    .map((x) => x.replace(/[.,;:!?]+$/, '').trim())
    .filter((x) => /[A-Za-z]{2,}/.test(x) && x.length >= 3 && x.length <= 60);
  if (!candidates.length) return '';
  // 优先「多词短语」，其次较长的单词；避免把 "the" / "is" 之类当技术片段。
  candidates.sort((a, b) => b.split(/\s+/).length - a.split(/\s+/).length || b.length - a.length);
  return candidates[0].replace(/\s+/g, ' ').trim();
}

/**
 * 给英文源使用的查询：优先完整翻译（保留意图），其次技术片段，最后才是条目名。
 * topic 由维基 langlinks 提供，仅在没有更好的选择时使用。
 */
export function composeEnQuery(query, { topic = '' } = {}) {
  const translated = translateQuery(query);
  if (translated) return translated;
  const fragment = englishFragment(query);
  if (fragment) return fragment;
  const t = String(topic || '').trim();
  return t ? t.slice(0, 120) : '';
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

/* ------------------------------------------------------------------ *
 * 分词与相关性打分
 * ------------------------------------------------------------------ */

/** 1–2 个字符但检索价值极高的技术词（分词器默认会丢掉它们）。 */
const SHORT_TERMS = new Set([
  'ai', 'ml', 'go', 'js', 'ts', 'gc', 'vm', 'os', 'io', 'db', 'ui', 'ux', 'ci', 'cd', 'r', 'c',
  '5g', 'vr', 'ar', 'ip', 'tcp', 'udp', 'dns', 'cpu', 'gpu', 'sql', 'api', 'orm', 'sdk', 'npm',
]);
const WORD_ALIASES = { 'c++': 'cpp', 'c#': 'csharp', 'f#': 'fsharp', '.net': 'dotnet', 'node.js': 'nodejs', 'vue.js': 'vuejs', 'next.js': 'nextjs' };

function stemVariants(word) {
  const w = String(word || '').toLowerCase();
  const out = new Set([w]);
  if (w.length > 4 && w.endsWith('ies')) out.add(`${w.slice(0, -3)}y`);
  if (w.length > 5 && w.endsWith('ing')) {
    out.add(w.slice(0, -3));
    out.add(`${w.slice(0, -3)}e`);
  }
  if (w.length > 4 && w.endsWith('ed')) {
    out.add(w.slice(0, -2));
    out.add(w.slice(0, -1));
  }
  if (w.length > 4 && w.endsWith('es')) out.add(w.slice(0, -2));
  if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) out.add(w.slice(0, -1));
  return out;
}

function addWord(set, raw) {
  const lower = String(raw || '').toLowerCase();
  if (!lower) return;
  const alias = WORD_ALIASES[lower] || lower;
  const isShort = alias.length <= 2;
  if (alias.length < 3 && !SHORT_TERMS.has(alias)) return;
  if (isShort && !SHORT_TERMS.has(alias)) return;
  for (const v of stemVariants(alias)) set.add(v);
}

/** 文本 → 检索 token 集合：英文单词（含词干与短技术词）+ 中文 bigram + 2 位以上数字。 */
export function textTokens(value) {
  const s = String(value || '').toLowerCase();
  const tokens = new Set();
  for (const word of s.match(/[a-z][a-z0-9+#.]*/g) || []) addWord(tokens, word);
  for (const num of s.match(/\d{2,}/g) || []) tokens.add(num);
  const cjk = s.match(/[\u3400-\u9fff\uf900-\ufaff]+/g) || [];
  for (const run of cjk) {
    if (run.length === 1) tokens.add(run);
    for (let i = 0; i + 1 < run.length; i += 1) tokens.add(run.slice(i, i + 2));
  }
  return tokens;
}

/** 术语权重：技术短词/缩写权重更高，避免被泛词稀释。 */
function termWeight(term) {
  if (/^\d+$/.test(term)) return 1;
  if (SHORT_TERMS.has(term) || /^[a-z]{2,3}$/.test(term)) return 2;
  return 1;
}

/** 把查询（含可靠翻译）展开成若干「接受变体」，每个变体带年份与加权术语表。 */
export function queryVariants(query, translations = []) {
  const safe = buildSearchQuery(query) || String(query || '');
  const keyword = hasCjk(safe) ? zhKeywords(safe) : enKeywords(safe);
  const translated = translateQuery(keyword);
  const raws = [keyword, translated, ...(Array.isArray(translations) ? translations : [])]
    .map((x) => String(x || '').trim())
    .filter(Boolean);
  return raws.map((text) => {
    const terms = new Map();
    for (const term of textTokens(text)) terms.set(term, termWeight(term));
    return { text, year: text.match(/(?:19|20)\d{2}/)?.[0] || '', terms };
  });
}

/**
 * 单条结果的相关性分数（0–1）。做法：
 * - 多个接受变体取最高分，避免「概念在英文世界有别的叫法」被整体误杀；
 * - 术语按权重累计覆盖率，短技术词（AI/Go/C++）不会被泛词淹没；
 * - 查询带年份时要求候选出现同一年（最低时效门槛，不冒充事实核查）；
 * - 标题命中给少量奖励，让权威条目的名次更靠前。
 */
export function scoreResult(query, result, { translations = [] } = {}) {
  const title = String(result?.title || '');
  const content = `${title} ${result?.snippet || ''} ${result?.extract || ''}`;
  const candidates = textTokens(content);
  const titleTokens = textTokens(title);
  const variants = queryVariants(query, translations);
  let best = 0;
  for (const variant of variants) {
    if (!variant.terms.size) continue;
    if (variant.year && !candidates.has(variant.year)) continue;
    let matched = 0;
    let weight = 0;
    let total = 0;
    for (const [term, w] of variant.terms) {
      total += w;
      if (candidates.has(term)) {
        matched += 1;
        weight += w;
      }
    }
    if (!matched) continue;
    let score = weight / total;
    // 3 个以上术语却只命中 1 个：降权但不直接判死（短查询仍需要一个精准词即可）。
    if (variant.terms.size >= 3 && matched < 2) score *= 0.6;
    let titleHits = 0;
    for (const term of variant.terms.keys()) if (titleTokens.has(term)) titleHits += 1;
    if (titleHits) score = Math.min(1, score + 0.1 * Math.min(1, titleHits / variant.terms.size));
    if (score > best) best = score;
  }
  return best;
}

/** 默认相关性门槛：与上一版「硬删」相比，0.3 仍能挡住无关页面，但放过短技术词命中的精准条目。 */
export const MIN_RELEVANCE = 0.28;

/**
 * 保守的相关性过滤：按 scoreResult 打分，低于门槛的丢弃。
 * 不做「宁杀错」的整源清空；调用方还可以用 keep-top 策略信任来源自己的排序。
 */
export function filterRelevantResults(query, results, { translations = [], minScore = MIN_RELEVANCE } = {}) {
  return (Array.isArray(results) ? results : []).filter(
    (result) => scoreResult(query, result, { translations }) >= minScore,
  );
}
