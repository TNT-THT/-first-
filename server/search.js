/**
 * 服务端网页搜索（软件版的“正经网页搜索”）。
 *
 * 策略与上市 AI 的联网搜索一致：先向搜索引擎拿候选，再抓取前几条网页正文，
 * 去重、按相关性排序后把带出处的资料交给模型，并要求回答标注 [编号]。
 *
 * 搜索源优先级（有哪个用哪个，可同时配置多个，失败自动降级）：
 * 1. Tavily（TAVILY_API_KEY）——专为 AI 搜索设计的接口，自带正文摘要；
 * 2. Brave Search（BRAVE_SEARCH_KEY）；
 * 3. Bing Web Search API（BING_SEARCH_KEY）；
 * 4. 免密钥兜底：Bing RSS + DuckDuckGo HTML（可用性随网络环境波动）。
 *
 * 抓正文只取 HTML 文本（去脚本/样式/标签），单条截断，避免把整页塞给模型。
 */

const TIMEOUT_MS = 9000;

function stripHtml(html) {
  return String(html || '')
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, ' ')
    .replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, ' ')
    .replace(/<head\b[^>]*>[\s\S]*?<\/head>/gi, ' ')
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<\/(?:p|div|li|h[1-6]|section|article|tr)>/gi, '\n')
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<[^>]*>/g, '')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .split('\n')
    .map((l) => l.replace(/[ \t]+/g, ' ').trim())
    .filter(Boolean)
    .join('\n');
}

async function fetchWithTimeout(fetchImpl, url, opts = {}, timeoutMs = TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('搜索请求超时')), timeoutMs);
  try {
    const res = await fetchImpl(url, { ...opts, signal: controller.signal, redirect: 'follow' });
    return res;
  } finally {
    clearTimeout(timer);
  }
}

export function parseTavily(payload) {
  const list = Array.isArray(payload?.results) ? payload.results : [];
  return list
    .map((r) => ({
      title: String(r.title || r.url || ''),
      url: String(r.url || ''),
      snippet: String(r.content || '').slice(0, 400),
      content: String(r.content || ''),
      publishedAt: r.published_date || '',
      provider: 'tavily',
      score: Number(r.score) || 0,
    }))
    .filter((r) => r.url);
}

export function parseBrave(payload) {
  const list = Array.isArray(payload?.web?.results) ? payload.web.results : [];
  return list
    .map((r) => ({
      title: String(r.title || r.url || ''),
      url: String(r.url || ''),
      snippet: String(r.description || '').slice(0, 400),
      content: '',
      publishedAt: r.age || '',
      provider: 'brave',
      score: 0,
    }))
    .filter((r) => r.url);
}

export function parseBingApi(payload) {
  const list = Array.isArray(payload?.webPages?.value) ? payload.webPages.value : [];
  return list
    .map((r) => ({
      title: String(r.name || r.url || ''),
      url: String(r.url || ''),
      snippet: String(r.snippet || '').slice(0, 400),
      content: '',
      publishedAt: r.dateLastCrawled || '',
      provider: 'bing',
      score: 0,
    }))
    .filter((r) => r.url);
}

/** Bing RSS（免密钥兜底）：解析 <item> 条目。 */
export function parseBingRss(xml) {
  const out = [];
  const items = String(xml || '').match(/<item>[\s\S]*?<\/item>/g) || [];
  for (const item of items) {
    const pick = (tag) => {
      const m = item.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`));
      return m ? stripHtml(m[1].replace(/<!\[CDATA\[|\]\]>/g, '')) : '';
    };
    const url = pick('link');
    if (!url) continue;
    out.push({ title: pick('title') || url, url, snippet: pick('description').slice(0, 400), content: '', publishedAt: pick('pubDate'), provider: 'bing-rss', score: 0 });
  }
  return out;
}

/** DuckDuckGo HTML（免密钥兜底）：解析 result__a 链接（uddg 跳转解码）。 */
export function parseDuckDuckGoHtml(html) {
  const out = [];
  const re = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g;
  let m;
  while ((m = re.exec(String(html || ''))) && out.length < 15) {
    let url = m[1].replace(/&amp;/g, '&');
    try {
      const u = new URL(url, 'https://duckduckgo.com');
      const target = u.searchParams.get('uddg');
      if (target) url = decodeURIComponent(target);
    } catch { /* 保持原链接 */ }
    if (!/^https?:\/\//.test(url)) continue;
    out.push({ title: stripHtml(m[2]), url, snippet: '', content: '', publishedAt: '', provider: 'duckduckgo-html', score: 0 });
  }
  return out;
}

/** 抓一条网页正文（只要 HTML 文本，失败返回空串不抛错）。 */
export async function fetchPageText(url, { fetchImpl = fetch, maxChars = 2000, timeoutMs = TIMEOUT_MS } = {}) {
  try {
    const res = await fetchWithTimeout(fetchImpl, url, {
      headers: { 'user-agent': 'Mozilla/5.0 (compatible; LLMHost-SkyBlue/1.0; +local-search)', accept: 'text/html' },
    }, timeoutMs);
    if (!res.ok) return '';
    const type = String(res.headers?.get?.('content-type') || '');
    if (type && !/text\/html|text\/plain|application\/xhtml/.test(type)) return '';
    const text = stripHtml(await res.text());
    return text.slice(0, maxChars);
  } catch {
    return '';
  }
}

function dedupeAndRank(results, query) {
  const terms = String(query || '').toLowerCase().split(/[\s,，。;；:：、]+/).filter((t) => t.length > 1);
  const seen = new Map();
  for (const r of results) {
    let host = '';
    try {
      host = new URL(r.url).hostname.replace(/^www\./, '');
    } catch {
      continue;
    }
    const key = `${host}|${r.title.toLowerCase().slice(0, 60)}`;
    if (seen.has(key)) continue;
    const hay = `${r.title} ${r.snippet} ${r.content}`.toLowerCase();
    let hits = 0;
    for (const t of terms) if (hay.includes(t)) hits += 1;
    const relevance = terms.length ? hits / terms.length : 0.5;
    seen.set(key, { ...r, relevance });
  }
  return [...seen.values()].sort((a, b) => (b.score || 0) + b.relevance - ((a.score || 0) + a.relevance));
}

/**
 * 执行一次网页搜索。env 里读三家密钥（TAVILY_API_KEY / BRAVE_SEARCH_KEY / BING_SEARCH_KEY）。
 * @returns {Promise<{provider: string, results: Array, errors: string[]}>}
 */
export async function searchWeb({ query, limit = 8, fetchImpl = fetch, env = process.env, timeoutMs = TIMEOUT_MS } = {}) {
  const q = String(query || '').trim();
  if (!q) throw Object.assign(new Error('搜索关键词不能为空。'), { status: 400 });
  const errors = [];
  let provider = '';
  let candidates = [];

  const tryJson = async (name, url, opts, parse) => {
    try {
      const res = await fetchWithTimeout(fetchImpl, url, opts, timeoutMs);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = parse(await res.json());
      if (data.length) {
        provider = name;
        candidates = data;
        return true;
      }
      errors.push(`${name}：没有结果`);
    } catch (err) {
      errors.push(`${name}：${err.message}`);
    }
    return false;
  };

  if (env.TAVILY_API_KEY) {
    await tryJson('tavily', 'https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ api_key: env.TAVILY_API_KEY, query: q, max_results: limit, search_depth: 'advanced', include_answer: false }),
    }, parseTavily);
  }
  if (!candidates.length && env.BRAVE_SEARCH_KEY) {
    const params = new URLSearchParams({ q, count: String(limit) });
    await tryJson('brave', `https://api.search.brave.com/res/v1/web/search?${params}`, {
      headers: { accept: 'application/json', 'x-subscription-token': env.BRAVE_SEARCH_KEY },
    }, parseBrave);
  }
  if (!candidates.length && env.BING_SEARCH_KEY) {
    const params = new URLSearchParams({ q, count: String(limit), responseFilter: 'WebPages' });
    await tryJson('bing', `https://api.bing.microsoft.com/v7.0/search?${params}`, {
      headers: { accept: 'application/json', 'ocp-apim-subscription-key': env.BING_SEARCH_KEY },
    }, parseBingApi);
  }
  if (!candidates.length) {
    // 免密钥兜底：两个来源都试，合并去重
    const fallback = [];
    try {
      const res = await fetchWithTimeout(fetchImpl, `https://www.bing.com/search?format=rss&q=${encodeURIComponent(q)}`, {}, timeoutMs);
      if (res.ok) fallback.push(...parseBingRss(await res.text()));
      else errors.push(`bing-rss：HTTP ${res.status}`);
    } catch (err) {
      errors.push(`bing-rss：${err.message}`);
    }
    try {
      const res = await fetchWithTimeout(fetchImpl, `https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`, {
        headers: { 'user-agent': 'Mozilla/5.0' },
      }, timeoutMs);
      if (res.ok) fallback.push(...parseDuckDuckGoHtml(await res.text()));
      else errors.push(`duckduckgo：HTTP ${res.status}`);
    } catch (err) {
      errors.push(`duckduckgo：${err.message}`);
    }
    if (fallback.length) {
      provider = 'keyless';
      candidates = fallback;
    }
  }

  if (!candidates.length) {
    throw Object.assign(new Error(`网页搜索没有拿到结果${errors.length ? `（${errors.slice(0, 3).join('；')}）` : ''}。可配置 TAVILY_API_KEY 等搜索密钥提升稳定性。`), { status: 502 });
  }

  const ranked = dedupeAndRank(candidates, q).slice(0, Math.max(1, limit));
  // 抓前几条正文（Tavily 自带 content 的不重复抓）
  const enriched = await Promise.all(
    ranked.map(async (r, i) => {
      if (r.content && r.content.length >= 300) return { ...r, content: r.content.slice(0, 2000) };
      if (i >= 5) return r;
      const text = await fetchPageText(r.url, { fetchImpl, timeoutMs });
      return { ...r, content: text || r.content || '' };
    }),
  );
  return {
    provider,
    results: enriched.map(({ score, relevance, ...rest }) => rest),
    errors,
  };
}
