/**
 * 公开资料源的适配器（纯解析逻辑 + 可注入 fetch）。
 *
 * 默认来源是维基百科、Stack Overflow 和 Hacker News 的公开 API；它们不是
 * 通用网页索引。深入检索会补取：维基百科导言、Stack Overflow 回答、HN
 * 讨论摘录。其他网页正文只能通过部署者明确配置的受信任 reader 服务读取。
 */

import { readerUrlFor } from './config.js';
import { filterRelevantResults, queryForLang } from './query.js';

/**
 * 原生 fetch 会响应 AbortSignal，但可注入 fetch / 少数 polyfill 不一定会。
 * 以 race 强制结束等待，确保单请求和全局超时不被一个不合作的 promise 拖住。
 */
function abortError(signal) {
  const reason = signal?.reason;
  return Object.assign(new Error(reason?.message || '请求已取消'), { name: 'AbortError' });
}

function awaitWithSignal(value, signal) {
  if (!signal) return Promise.resolve(value);
  if (signal.aborted) return Promise.reject(abortError(signal));
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject(abortError(signal));
    };
    const cleanup = () => signal.removeEventListener?.('abort', onAbort);
    signal.addEventListener?.('abort', onAbort, { once: true });
    Promise.resolve(value).then(
      (result) => {
        cleanup();
        resolve(result);
      },
      (err) => {
        cleanup();
        reject(err);
      },
    );
  });
}

/** 搜索请求统一加超时；失败抛出可读错误。 */
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
    const res = await awaitWithSignal(
      fetchImpl(url, {
        signal: controller.signal,
        mode: 'cors',
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
        headers,
      }),
      controller.signal,
    );
    if (!res?.ok) throw new Error(`搜索服务返回 HTTP ${res?.status || '错误'}`);
    return await awaitWithSignal(res.json(), controller.signal);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener?.('abort', forward);
  }
}

async function fetchText(url, { fetchImpl = globalThis.fetch, signal, timeoutMs = 8000 } = {}) {
  if (typeof fetchImpl !== 'function') throw new Error('当前环境不支持网络请求');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('正文读取超时')), Math.max(1000, timeoutMs));
  const forward = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) forward();
    else signal.addEventListener?.('abort', forward, { once: true });
  }
  try {
    const res = await awaitWithSignal(
      fetchImpl(url, {
        signal: controller.signal,
        mode: 'cors',
        credentials: 'omit',
        referrerPolicy: 'no-referrer',
      }),
      controller.signal,
    );
    if (!res?.ok) throw new Error(`正文读取返回 HTTP ${res?.status || '错误'}`);
    return await awaitWithSignal(res.text(), controller.signal);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener?.('abort', forward);
  }
}

function isAbort(err, signal) {
  return !!signal?.aborted || err?.name === 'AbortError';
}

/** 去掉来源摘要里的 HTML 标签；仅用于文本展示，非提示注入防护。 */
export function stripHtml(text) {
  return String(text || '')
    // 网页 reader 可能返回整页 HTML；脚本/样式内容不是可引用的正文，也不该进入提示词。
    .replace(/<(?:script|style|noscript)\b[^>]*>[\s\S]*?<\/(?:script|style|noscript)>/gi, ' ')
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

function asInt(value) {
  const n = Number(value);
  return Number.isInteger(n) && n >= 0 ? n : null;
}

function sourceDate(obj) {
  return pick(obj, ['publishedAt', 'published_at', 'publishedDate', 'date', 'timestamp', 'created_at', 'updated_at']) || obj?.creation_date || obj?.last_activity_date || '';
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
        publishedAt: sourceDate(item),
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
        publishedAt: sourceDate(item),
        provider: 'searxng',
      };
    })
    .filter(Boolean);
}

/** 自定义接口：接受 {results:[]}、{items:[]}、{data:[]} 或直接数组。 */
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
        publishedAt: sourceDate(item),
        provider: 'custom',
      };
    })
    .filter(Boolean);
}

/** Stack Exchange 搜索结果是“问题”，快速模式明确只使用问题摘要。 */
export function parseStackExchange(payload, { site = 'stackoverflow', maxSnippetChars = 240 } = {}) {
  const list = Array.isArray(payload?.items) ? payload.items : [];
  return list
    .map((item) => {
      const url = pick(item, ['link']);
      const title = stripHtml(pick(item, ['title']));
      if (!url || !title) return null;
      const body = stripHtml(pick(item, ['body', 'body_markdown', 'excerpt']));
      const questionId = asInt(item?.question_id);
      const acceptedAnswerId = asInt(item?.accepted_answer_id);
      return {
        title,
        url,
        snippet: body.slice(0, Math.max(0, maxSnippetChars)),
        publishedAt: sourceDate(item),
        provider: 'stackexchange',
        site,
        ...(questionId !== null ? { questionId } : {}),
        ...(acceptedAnswerId !== null ? { acceptedAnswerId } : {}),
      };
    })
    .filter(Boolean);
}

/** HN story 搜索。快速模式仅用帖文文本；深入模式再读取讨论 API。 */
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
        publishedAt: sourceDate(hit),
        provider: 'hackernews',
        ...(objectID ? { storyId: objectID } : {}),
      };
    })
    .filter(Boolean);
}

/** DuckDuckGo Instant Answer：摘要 + 相关主题，不伪装成完整网页搜索。 */
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

/** Stack Exchange 搜索（默认 Stack Overflow）。 */
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
  if (payload?.error_message) throw new Error(String(payload.error_message).slice(0, 120));
  return parseStackExchange(payload, { site, maxSnippetChars }).slice(0, Math.max(1, limit));
}

/** Hacker News story 搜索（Algolia）。 */
export async function searchHackerNews({ query, limit = 5, maxSnippetChars = 240, fetchImpl, signal, timeoutMs } = {}) {
  const params = new URLSearchParams({ query, tags: 'story', hitsPerPage: String(Math.min(10, Math.max(1, limit))) });
  const payload = await fetchJson(`https://hn.algolia.com/api/v1/search?${params}`, { fetchImpl, signal, timeoutMs });
  return parseHackerNews(payload, { maxSnippetChars }).slice(0, Math.max(1, limit));
}

/** DuckDuckGo Instant Answer（默认关闭，跨域可用性不稳定）。 */
export async function searchDuckDuckGo({ query, limit = 5, maxSnippetChars = 240, fetchImpl, signal, timeoutMs } = {}) {
  const params = new URLSearchParams({ q: query, format: 'json', no_html: '1', no_redirect: '1', skip_disambig: '1' });
  const payload = await fetchJson(`https://api.duckduckgo.com/?${params}`, { fetchImpl, signal, timeoutMs });
  return parseDuckDuckGo(payload, { maxSnippetChars }).slice(0, Math.max(1, limit));
}

/** 维基百科搜索（可传多个语言，前一个没有结果时再试另一个）。 */
export async function searchWikipedia({ query, langs = ['zh'], limit = 5, fetchImpl, signal, timeoutMs } = {}) {
  for (const lang of langs) {
    const params = new URLSearchParams({
      action: 'query',
      list: 'search',
      srsearch: queryForLang(query, lang),
      srlimit: String(limit),
      srprop: 'snippet|timestamp',
      format: 'json',
      origin: '*',
    });
    const payload = await fetchJson(`https://${lang}.wikipedia.org/w/api.php?${params}`, { fetchImpl, signal, timeoutMs });
    const localizedQuery = queryForLang(query, lang);
    const results = filterRelevantResults(localizedQuery, parseWikipedia(payload, { lang }));
    if (results.length) return results;
  }
  return [];
}

export async function searchSearxng({ query, baseUrl, limit = 5, fetchImpl, signal, timeoutMs } = {}) {
  if (!baseUrl) throw new Error('未配置 SearXNG 地址');
  let endpoint;
  try {
    // baseUrl 是实例根/部署路径，不允许它已有的 query/hash 污染真正的 /search 请求。
    const base = new URL(baseUrl);
    base.search = '';
    base.hash = '';
    base.pathname = `${base.pathname.replace(/\/+$/, '')}/search`;
    endpoint = base.href;
  } catch {
    throw new Error('SearXNG 地址无效');
  }
  const params = new URLSearchParams({ q: query, format: 'json', language: 'all', safesearch: '1' });
  const url = `${endpoint}?${params}`;
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

/** 按 id 调用对应来源；未知来源直接报错。 */
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

function wikiEntry(url) {
  try {
    const u = new URL(String(url || ''));
    const match = /^([a-z-]+)\.wikipedia\.org$/i.exec(u.hostname);
    if (!match || !u.pathname.startsWith('/wiki/')) return null;
    return { lang: match[1], title: decodeURIComponent(u.pathname.slice('/wiki/'.length)).replace(/_/g, ' ') };
  } catch {
    return null;
  }
}

async function enrichWikipediaExtracts(out, list, { fetchImpl, signal, timeoutMs }) {
  const indexed = list
    .map((item, index) => ({ index, wiki: wikiEntry(item?.url) }))
    .filter((item) => item.wiki);
  const langs = [...new Set(indexed.map((item) => item.wiki.lang))];
  await Promise.all(
    langs.map(async (lang) => {
      const group = indexed.filter((item) => item.wiki.lang === lang);
      const titles = group.map((item) => item.wiki.title);
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
        const payload = await fetchJson(`https://${lang}.wikipedia.org/w/api.php?${params}`, { fetchImpl, signal, timeoutMs });
        const pages = payload?.query?.pages;
        if (!pages || typeof pages !== 'object') return;
        const byTitle = new Map();
        for (const page of Object.values(pages)) {
          if (!page || typeof page !== 'object') continue;
          const extract = stripHtml(page.extract || '');
          if (extract) byTitle.set(String(page.title || ''), extract);
        }
        const aliases = new Map();
        for (const item of [...(payload?.query?.normalized || []), ...(payload?.query?.redirects || [])]) {
          if (item?.from && item?.to) aliases.set(String(item.from), String(item.to));
        }
        for (const { index, wiki } of group) {
          const resolved = aliases.get(wiki.title) || wiki.title;
          const extract = byTitle.get(resolved) || byTitle.get(wiki.title);
          if (extract) {
            out[index].extract = extract;
            out[index].evidenceKind = '维基百科导言';
          }
        }
      } catch (err) {
        if (isAbort(err, signal)) throw err;
      }
    }),
  );
}

/** 深入模式一次批量读取 Stack Overflow 的高分/采纳答案，而不是谎称问题 body 是答案。 */
async function enrichStackExchangeAnswers(out, list, { site = 'stackoverflow', fetchImpl, signal, timeoutMs }) {
  const targets = list
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => item?.provider === 'stackexchange' && Number.isInteger(item.questionId));
  if (!targets.length) return;
  const ids = [...new Set(targets.map(({ item }) => item.questionId))].slice(0, 10);
  const params = new URLSearchParams({
    site,
    pagesize: String(ids.length * 3),
    order: 'desc',
    sort: 'votes',
    filter: 'withbody',
  });
  try {
    const payload = await fetchJson(`https://api.stackexchange.com/2.3/questions/${ids.join(';')}/answers?${params}`, {
      fetchImpl,
      signal,
      timeoutMs,
    });
    if (payload?.error_message) throw new Error(String(payload.error_message).slice(0, 120));
    const answers = Array.isArray(payload?.items) ? payload.items : [];
    for (const { item, index } of targets) {
      const candidates = answers.filter((answer) => Number(answer?.question_id) === item.questionId);
      candidates.sort((a, b) => {
        const aAccepted = Number(a?.answer_id) === item.acceptedAnswerId || a?.is_accepted ? 1 : 0;
        const bAccepted = Number(b?.answer_id) === item.acceptedAnswerId || b?.is_accepted ? 1 : 0;
        return bAccepted - aAccepted || Number(b?.score || 0) - Number(a?.score || 0);
      });
      const answer = candidates[0];
      const body = stripHtml(answer?.body || answer?.body_markdown || '');
      if (body) {
        const accepted = !!answer?.is_accepted || Number(answer?.answer_id) === item.acceptedAnswerId;
        out[index].extract = body;
        out[index].evidenceKind = accepted ? 'Stack Overflow 采纳答案' : 'Stack Overflow 高分答案';
      }
    }
  } catch (err) {
    if (isAbort(err, signal)) throw err;
  }
}

function collectHnCommentText(children, limit = 3, out = []) {
  for (const child of Array.isArray(children) ? children : []) {
    if (out.length >= limit) break;
    const text = stripHtml(child?.text || '');
    if (text) out.push(text);
    collectHnCommentText(child?.children, limit, out);
  }
  return out;
}

/** 深入模式读取 HN item API 的帖文/评论摘录；不假装抓到了外链文章全文。 */
async function enrichHackerNewsDiscussions(out, list, { fetchImpl, signal, timeoutMs }) {
  const targets = list
    .map((item, index) => ({ item, index }))
    .filter(({ item }) => item?.provider === 'hackernews' && /^\d+$/.test(String(item.storyId || '')));
  await Promise.all(
    targets.map(async ({ item, index }) => {
      try {
        const payload = await fetchJson(`https://hn.algolia.com/api/v1/items/${encodeURIComponent(item.storyId)}`, {
          fetchImpl,
          signal,
          timeoutMs,
        });
        const pieces = [stripHtml(payload?.text || ''), ...collectHnCommentText(payload?.children, 3)].filter(Boolean);
        if (pieces.length) {
          out[index].extract = pieces.join('\n\n');
          out[index].evidenceKind = 'Hacker News 帖文与评论摘录';
        }
        if (!out[index].publishedAt && payload?.created_at) out[index].publishedAt = payload.created_at;
      } catch (err) {
        if (isAbort(err, signal)) throw err;
      }
    }),
  );
}

async function mapWithConcurrency(items, concurrency, worker) {
  let next = 0;
  const runners = Array.from({ length: Math.min(Math.max(1, concurrency), items.length) }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      await worker(items[index]);
    }
  });
  await Promise.all(runners);
}

/**
 * 深入检索：并发补充可信 API 的正文/答案/讨论；可选 reader 只处理仍没有摘录的结果。
 * 抓取失败保留原摘要，用户停止或总超时则向上抛出取消。
 */
export async function enrichWithExtracts(
  results,
  { maxPages = 3, readerTemplate = '', stackexchangeSite = 'stackoverflow', fetchImpl, signal, timeoutMs } = {},
) {
  const list = Array.isArray(results) ? results.slice(0, Math.max(0, maxPages)) : [];
  if (!list.length) return Array.isArray(results) ? results : [];
  const out = results.map((r) => ({ ...r }));
  try {
    await Promise.all([
      enrichWikipediaExtracts(out, list, { fetchImpl, signal, timeoutMs }),
      enrichStackExchangeAnswers(out, list, { site: stackexchangeSite, fetchImpl, signal, timeoutMs }),
      enrichHackerNewsDiscussions(out, list, { fetchImpl, signal, timeoutMs }),
    ]);
  } catch (err) {
    if (isAbort(err, signal)) throw err;
  }

  const readerTargets = list
    .map((_item, index) => index)
    .filter((index) => !out[index]?.extract && out[index]?.url && readerUrlFor(readerTemplate, out[index].url));
  try {
    await mapWithConcurrency(readerTargets, 2, async (index) => {
      const item = out[index];
      const url = readerUrlFor(readerTemplate, item.url);
      try {
        const text = stripHtml(await fetchText(url, { fetchImpl, signal, timeoutMs })).slice(0, 800);
        if (text) {
          item.extract = text;
          item.evidenceKind = '正文读取服务摘录';
        }
      } catch (err) {
        if (isAbort(err, signal)) throw err;
      }
    });
  } catch (err) {
    if (isAbort(err, signal)) throw err;
  }
  return out;
}
