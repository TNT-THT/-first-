/**
 * 搜索源的实现（纯解析逻辑 + 可注入 fetch，便于测试）。
 *
 * 这个功能的设计前提是：**没有 API Key，也没有独立服务器**。所有搜索都在浏览器里直接完成，
 * 因此只使用“公开、允许跨域、不需要密钥”的接口：
 *
 * | id | 接口 | 密钥 | 适用 |
 * |---|---|---|---|
 * | wikipedia | 维基百科 Action API（`origin=*`） | 不需要 | 概念、算法、数据结构 |
 * | stackexchange | Stack Exchange API 2.3（Stack Overflow） | 不需要 | 具体的编程问题、报错 |
 * | hackernews | HN Algolia Search API | 不需要 | 技术讨论、工具与经验 |
 * | duckduckgo | DuckDuckGo Instant Answer API | 不需要 | 常识性摘要（跨域支持不稳定，默认关闭） |
 * | searxng / custom | 自建服务 | 可选 | 想搜“整个网页”时，由管理员自己填地址（本机保存） |
 *
 * 前四个是默认的免费来源，会并行请求、合并去重；任何一个失败或跨域被拦都不影响其它来源。
 *
 * 智能搜索（smart）在快速搜索的基础上再补一段正文：维基百科用 Action API 的 extracts，
 * Stack Exchange 直接用答案正文，Hacker News 用讨论文本；
 * 其他来源可以在设置里填一个“正文抓取模板”（例如 `https://r.jina.ai/`），抓不到就算了，不影响回答。
 */

/** 搜索请求统一加超时；失败抛出可读的错误信息。 */
async function fetchJson(url, { fetchImpl = globalThis.fetch, signal, timeoutMs = 8000, headers } = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('当前环境不支持网络请求');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('搜索请求超时')), Math.max(1000, timeoutMs));
  const forward = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) forward();
    else signal.addEventListener?.('abort', forward, { once: true });
  }
  try {
    const res = await fetchImpl(url, {
      signal: controller.signal,
      mode: 'cors',
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
      headers,
    });
    if (!res.ok) throw new Error(`搜索服务返回 HTTP ${res.status}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener?.('abort', forward);
  }
}

/** 去掉搜索摘要里的 HTML 标签（维基百科的高亮是 <span class="searchmatch">）。 */
export function stripHtml(text) {
  return String(text || '')
    .replace(/<[^>]*>/g, '')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ')
    .trim();
}

function pick(obj, keys) {
  for (const key of keys) {
    const v = obj?.[key];
    if (typeof v === 'string' && v.trim()) return v.trim();
  }
  return '';
}

/** 维基百科：query.search[] → 统一结果。 */
export function parseWikipedia(payload, { lang = 'zh' } = {}) {
  const list = payload?.query?.search;
  if (!Array.isArray(list)) return [];
  return list
    .map((item) => {
      const title = stripHtml(item?.title);
      if (!title) return null;
      const page = String(title).replace(/ /g, '_');
      return {
        title,
        url: `https://${lang}.wikipedia.org/wiki/${encodeURIComponent(page)}`,
        snippet: stripHtml(item?.snippet),
        provider: 'wikipedia',
      };
    })
    .filter(Boolean);
}

/** SearXNG：results[] → 统一结果。 */
export function parseSearxng(payload) {
  const list = Array.isArray(payload?.results) ? payload.results : [];
  return list
    .map((item) => {
      const url = pick(item, ['url', 'link']);
      const title = pick(item, ['title']) || url;
      if (!url) return null;
      return {
        title: stripHtml(title),
        url,
        snippet: stripHtml(pick(item, ['content', 'snippet', 'summary'])),
        provider: 'searxng',
      };
    })
    .filter(Boolean);
}

/** 自定义接口：接受 {results:[]}、{items:[]} 或直接数组，字段名容错。 */
export function parseCustom(payload) {
  const list = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.results)
      ? payload.results
      : Array.isArray(payload?.items)
        ? payload.items
        : Array.isArray(payload?.data)
          ? payload.data
          : [];
  return list
    .map((item) => {
      const url = pick(item, ['url', 'link', 'href']);
      const title = pick(item, ['title', 'name', 'heading']) || url;
      if (!url) return null;
      return {
        title: stripHtml(title),
        url,
        snippet: stripHtml(pick(item, ['snippet', 'content', 'summary', 'description', 'text'])),
        provider: 'custom',
      };
    })
    .filter(Boolean);
}

/** Stack Exchange：items[] → 统一结果（正文是 HTML，按纯文本截断）。 */
export function parseStackExchange(payload, { site = 'stackoverflow', maxSnippetChars = 240 } = {}) {
  const list = Array.isArray(payload?.items) ? payload.items : [];
  return list
    .map((item) => {
      const url = pick(item, ['link']);
      const title = stripHtml(pick(item, ['title']));
      if (!url || !title) return null;
      const body = stripHtml(pick(item, ['body', 'body_markdown', 'excerpt']));
      return {
        title,
        url,
        snippet: body.slice(0, Math.max(0, maxSnippetChars)),
        provider: 'stackexchange',
        site,
      };
    })
    .filter(Boolean);
}

/** Hacker News（Algolia）：hits[] → 统一结果；没有外链的 Ask HN 用讨论页地址。 */
export function parseHackerNews(payload, { maxSnippetChars = 240 } = {}) {
  const list = Array.isArray(payload?.hits) ? payload.hits : [];
  return list
    .map((hit) => {
      const objectID = pick(hit, ['objectID']);
      const url =
        pick(hit, ['url', 'story_url']) ||
        (objectID ? `https://news.ycombinator.com/item?id=${encodeURIComponent(objectID)}` : '');
      const title = stripHtml(pick(hit, ['title', 'story_title']));
      if (!url || !title) return null;
      const text = stripHtml(pick(hit, ['story_text', 'comment_text', 'title']));
      return {
        title,
        url,
        snippet: text.slice(0, Math.max(0, maxSnippetChars)),
        provider: 'hackernews',
      };
    })
    .filter(Boolean);
}

/** DuckDuckGo 即时答案：摘要 + 相关主题（不是完整网页搜索，只适合常识性内容）。 */
export function parseDuckDuckGo(payload, { maxSnippetChars = 240 } = {}) {
  const out = [];
  const abstractUrl = pick(payload, ['AbstractURL']);
  const abstract = stripHtml(pick(payload, ['AbstractText', 'Answer']));
  if (abstractUrl && abstract) {
    out.push({
      title: stripHtml(pick(payload, ['Heading'])) || abstractUrl,
      url: abstractUrl,
      snippet: abstract.slice(0, Math.max(0, maxSnippetChars)),
      provider: 'duckduckgo',
    });
  }
  const topics = Array.isArray(payload?.RelatedTopics) ? payload.RelatedTopics : [];
  const flat = topics.flatMap((t) => (Array.isArray(t?.Topics) ? t.Topics : [t]));
  for (const t of flat) {
    const url = pick(t, ['FirstURL']);
    const text = stripHtml(pick(t, ['Text']));
    if (!url || !text) continue;
    out.push({ title: text.slice(0, 60), url, snippet: text.slice(0, Math.max(0, maxSnippetChars)), provider: 'duckduckgo' });
  }
  return out;
}

/** Stack Exchange 搜索（默认 Stack Overflow；也可填 superuser / askubuntu 等站点）。 */
export async function searchStackExchange({
  query,
  site = 'stackoverflow',
  limit = 5,
  maxSnippetChars = 240,
  fetchImpl,
  signal,
  timeoutMs,
} = {}) {
  const params = new URLSearchParams({
    order: 'desc',
    sort: 'relevance',
    q: query,
    site,
    pagesize: String(Math.min(10, Math.max(1, limit))),
    filter: 'withbody',
  });
  const payload = await fetchJson(`https://api.stackexchange.com/2.3/search/advanced?${params}`, {
    fetchImpl,
    signal,
    timeoutMs,
  });
  // 配额用完时接口返回 error_message（HTTP 200），这里当作“没有结果”，由上层换下一个来源
  if (payload?.error_message) throw new Error(String(payload.error_message).slice(0, 120));
  return parseStackExchange(payload, { site, maxSnippetChars }).slice(0, Math.max(1, limit));
}

/** Hacker News 搜索（Algolia，免费、允许跨域）。 */
export async function searchHackerNews({ query, limit = 5, maxSnippetChars = 240, fetchImpl, signal, timeoutMs } = {}) {
  const params = new URLSearchParams({
    query,
    tags: 'story',
    hitsPerPage: String(Math.min(10, Math.max(1, limit))),
  });
  const payload = await fetchJson(`https://hn.algolia.com/api/v1/search?${params}`, {
    fetchImpl,
    signal,
    timeoutMs,
  });
  return parseHackerNews(payload, { maxSnippetChars }).slice(0, Math.max(1, limit));
}

/** DuckDuckGo 即时答案（默认不启用：跨域支持不稳定）。 */
export async function searchDuckDuckGo({ query, limit = 5, maxSnippetChars = 240, fetchImpl, signal, timeoutMs } = {}) {
  const params = new URLSearchParams({
    q: query,
    format: 'json',
    no_html: '1',
    no_redirect: '1',
    skip_disambig: '1',
  });
  const payload = await fetchJson(`https://api.duckduckgo.com/?${params}`, {
    fetchImpl,
    signal,
    timeoutMs,
  });
  return parseDuckDuckGo(payload, { maxSnippetChars }).slice(0, Math.max(1, limit));
}

/** 维基百科搜索（可传多个语言，前一个没有结果时自动试下一个）。 */
export async function searchWikipedia({
  query,
  langs = ['zh'],
  limit = 5,
  fetchImpl,
  signal,
  timeoutMs,
} = {}) {
  const out = [];
  for (const lang of langs) {
    const params = new URLSearchParams({
      action: 'query',
      list: 'search',
      srsearch: query,
      srlimit: String(limit),
      srprop: 'snippet',
      format: 'json',
      origin: '*',
    });
    const payload = await fetchJson(`https://${lang}.wikipedia.org/w/api.php?${params}`, {
      fetchImpl,
      signal,
      timeoutMs,
    });
    const results = parseWikipedia(payload, { lang });
    if (results.length) return results;
    out.push(...results);
  }
  return out;
}

export async function searchSearxng({ query, baseUrl, limit = 5, fetchImpl, signal, timeoutMs } = {}) {
  if (!baseUrl) throw new Error('未配置 SearXNG 地址');
  const params = new URLSearchParams({ q: query, format: 'json', language: 'all', safesearch: '1' });
  const url = `${String(baseUrl).replace(/\/+$/, '')}/search?${params}`;
  const payload = await fetchJson(url, { fetchImpl, signal, timeoutMs });
  return parseSearxng(payload).slice(0, Math.max(1, limit));
}

export async function searchCustom({ query, endpoint, limit = 5, fetchImpl, signal, timeoutMs } = {}) {
  if (!endpoint) throw new Error('未配置自定义搜索接口');
  const base = String(endpoint);
  const sep = base.includes('?') ? '&' : '?';
  const url = `${base}${sep}q=${encodeURIComponent(query)}&limit=${encodeURIComponent(limit)}`;
  const payload = await fetchJson(url, { fetchImpl, signal, timeoutMs });
  return parseCustom(payload).slice(0, Math.max(1, limit));
}

/** 按 id 调用对应的搜索源；未知来源直接报错，避免静默失败。 */
export function runProvider(id, options = {}) {
  if (id === 'wikipedia') {
    return searchWikipedia({
      query: options.query,
      langs: options.langs || ['zh'],
      limit: options.limit,
      maxSnippetChars: options.maxSnippetChars,
      fetchImpl: options.fetchImpl,
      signal: options.signal,
      timeoutMs: options.timeoutMs,
    });
  }
  if (id === 'stackexchange') return searchStackExchange(options);
  if (id === 'hackernews') return searchHackerNews(options);
  if (id === 'duckduckgo') return searchDuckDuckGo(options);
  if (id === 'searxng') return searchSearxng(options);
  if (id === 'custom') return searchCustom(options);
  throw new Error(`未知的搜索源：${id}`);
}

/**
 * 智能搜索：给前几个结果补一段正文摘要。
 * - 维基百科：一次请求拿到多条的 extracts（纯文本引言），跨域允许；
 * - 其他来源：只有在设置了正文抓取模板时才尝试（模板 + 经过编码的原文地址）；
 * 抓取失败一律忽略，保留原来的摘要。
 */
export async function enrichWithExtracts(
  results,
  { maxPages = 3, readerTemplate = '', fetchImpl, signal, timeoutMs } = {},
) {
  const list = Array.isArray(results) ? results.slice(0, Math.max(0, maxPages)) : [];
  if (!list.length) return Array.isArray(results) ? results : [];
  const out = results.map((r) => ({ ...r }));
  const wikiIndexes = [];
  for (let i = 0; i < list.length; i += 1) {
    const m = /^https:\/\/([a-z-]+)\.wikipedia\.org\/wiki\//i.exec(list[i].url || '');
    if (m) wikiIndexes.push({ index: i, lang: m[1] });
  }
  if (wikiIndexes.length) {
    for (const lang of [...new Set(wikiIndexes.map((x) => x.lang))]) {
      const group = wikiIndexes.filter((x) => x.lang === lang);
      const titles = group.map((x) => decodeURIComponent(String(list[x.index].url).split('/wiki/')[1] || '').replace(/_/g, ' '));
      const params = new URLSearchParams({
        action: 'query',
        prop: 'extracts',
        explaintext: '1',
        exintro: '1',
        exlimit: String(titles.length),
        redirects: '1',
        format: 'json',
        origin: '*',
        titles: titles.join('|'),
      });
      try {
        const payload = await fetchJson(`https://${lang}.wikipedia.org/w/api.php?${params}`, {
          fetchImpl,
          signal,
          timeoutMs,
        });
        const pages = payload?.query?.pages;
        if (pages && typeof pages === 'object') {
          const byTitle = new Map();
          for (const page of Object.values(pages)) {
            if (!page || typeof page !== 'object') continue;
            const extract = stripHtml(page.extract || '');
            if (!extract) continue;
            byTitle.set(String(page.title || ''), extract);
            // 规范化 / 重定向后的标题也登记一份，便于匹配
            if (page.normalizedtitle) byTitle.set(String(page.normalizedtitle), extract);
          }
          for (const { index } of group) {
            const title = decodeURIComponent(String(list[index].url).split('/wiki/')[1] || '').replace(/_/g, ' ');
            const extract = byTitle.get(title);
            if (extract) out[index].extract = extract;
          }
        }
      } catch (err) {
        if (signal?.aborted) throw err;
        // 抓正文失败不影响结果：保留原来的摘要
      }
    }
  }
  if (readerTemplate) {
    for (let i = 0; i < list.length; i += 1) {
      const item = out[i];
      if (item.extract || !item.url) continue;
      try {
        const res = await fetchJsonText(`${readerTemplate}${encodeURIComponent(item.url)}`, {
          fetchImpl,
          signal,
          timeoutMs,
        });
        const text = stripHtml(res).slice(0, 800);
        if (text) out[i].extract = text;
      } catch (err) {
        if (signal?.aborted) throw err;
      }
    }
  }
  return out;
}

/** 抓取纯文本（正文提取模板返回的不是 JSON）。 */
async function fetchJsonText(url, { fetchImpl = globalThis.fetch, signal, timeoutMs = 8000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('正文抓取超时')), Math.max(1000, timeoutMs));
  const forward = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) forward();
    else signal.addEventListener?.('abort', forward, { once: true });
  }
  try {
    const res = await fetchImpl(url, {
      signal: controller.signal,
      mode: 'cors',
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
    });
    if (!res.ok) throw new Error(`正文抓取返回 HTTP ${res.status}`);
    return await res.text();
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener?.('abort', forward);
  }
}
