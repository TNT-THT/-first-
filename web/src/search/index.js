/**
 * 联网资料检索的编排（纯逻辑，fetch 与缓存可注入）。
 *
 * 设计要点（对应评估报告 2.x / 3.x 的整改）：
 * - 查询规划：先尝试术语表翻译（保留问题意图），再退到英文技术片段，最后才用维基条目名；
 *   三者都拿不到时不把中文原句发给看不懂中文的接口。
 * - 来源取舍：所有来源并发检索，结果按「相关性分数 × 来源权重」排序，
 *   并保证来源多样性；不再让 HN 标题和维基正文同权轮流占编号。
 * - 证据质量：快速模式也会补取前 N 条的维基导言 / SO 答案，
 *   让「搜到了」真的等于「有内容可读」；HN 只有讨论（smart）时才补，避免标题占位。
 * - 失败路径仍然显式：每个来源的状态、部分结果、缓存命中都带回来。
 */

import { DEFAULT_WEB_SEARCH, planProviders, resolveSearchPrefs } from './config.js';
import {
  analyzeSearchInput,
  composeEnQuery,
  hasCjk,
  isTrivialQuery,
  searchKeywords,
  translateQuery,
  wikipediaLangs,
} from './query.js';
import { enrichWithExtracts, resolveEnQuery, runProvider } from './providers.js';
import { mergeResults, rankResults } from './context.js';

/** 只接受英文查询的公开源：中文原句发过去必然空手，还暴露访客 IP。 */
const EN_ONLY_PROVIDERS = new Set(['stackexchange', 'hackernews', 'duckduckgo']);
const CJK_SKIP_MESSAGE = '中文查询没有可靠的英文翻译，本轮已跳过英文源';

function makeAbortError(signal) {
  return Object.assign(new Error(signal?.reason?.message || '已取消'), { name: 'AbortError' });
}

/** 不假设底层 fetch 一定兑现 AbortSignal，确保全局 deadline 不会无限等待。 */
function awaitWithSignal(value, signal) {
  if (!signal) return Promise.resolve(value);
  if (signal.aborted) return Promise.reject(makeAbortError(signal));
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject(makeAbortError(signal));
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

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function providerOptions(id, { query, enQuery, url, config, settings, fetchImpl, signal, timeoutMs }) {
  const cjk = hasCjk(query);
  const base = searchKeywords(query) || query;
  return {
    query: EN_ONLY_PROVIDERS.has(id) && cjk ? enQuery || '' : base,
    langs: id === 'wikipedia' ? wikipediaLangs(query, config.wikipedia) : undefined,
    ...(id === 'wikipedia' ? { enQuery, defaultLang: config.wikipedia?.lang || 'zh' } : {}),
    site: id === 'stackexchange' ? config.stackexchange?.site : undefined,
    acceptedOnly: id === 'stackexchange' ? config.stackexchange?.acceptedOnly !== false : undefined,
    baseUrl: url,
    endpoint: url,
    limit: config.maxResults,
    maxSnippetChars: config.maxSnippetChars,
    readerTemplate: settings.readerTemplate || config.smart?.readerTemplate || '',
    fetchImpl,
    signal,
    timeoutMs,
  };
}

function cacheKey(query, mode, config, settings, targets, enQuery) {
  return JSON.stringify({
    query,
    mode,
    enQuery,
    targets: targets.map((target) => [target.id, target.url || '']),
    maxResults: config.maxResults,
    maxSnippetChars: config.maxSnippetChars,
    maxPerProvider: config.maxPerProvider,
    minRelevance: config.minRelevance,
    wikipedia: config.wikipedia,
    stackexchange: config.stackexchange,
    pages: mode === 'smart' ? config.smart?.maxPages : config.quick?.maxPages,
    readerTemplate: settings.readerTemplate || config.smart?.readerTemplate || '',
  });
}

/** 创建仅当前页面存活的短缓存；调用方不应把它持久化到 localStorage。 */
export function createSearchCache() {
  return new Map();
}

function readCache(cache, key, ttlMs, nowMs) {
  if (!(cache instanceof Map) || !ttlMs) return null;
  const entry = cache.get(key);
  if (!entry || nowMs - entry.at > ttlMs) {
    cache.delete(key);
    return null;
  }
  return clone(entry.value);
}

function writeCache(cache, key, ttlMs, nowMs, value) {
  if (!(cache instanceof Map) || !ttlMs) return;
  cache.set(key, { at: nowMs, value: clone(value) });
  // 防止长会话无限增长；最近 30 个查询已足够覆盖“重试/重新生成”。
  while (cache.size > 30) cache.delete(cache.keys().next().value);
}

/** 每个来源的结果先按 snippet 长度归一化、去重，再统一排序。 */
function normalizeAll(groups, config, mode) {
  const all = [];
  for (const group of groups) {
    for (const item of group.results) {
      // 快速模式不读取 HN 讨论，只有标题的条目等于零证据：直接排除，
      // 避免它占掉一个 [编号] 和约 130 字符的元数据。
      if (mode === 'quick' && item.titleOnly) continue;
      all.push({ ...item, provider: item.provider || group.id });
    }
  }
  return mergeResults(all, { limit: 40, maxSnippetChars: config.maxSnippetChars });
}

function rankPlan(list, { query, enQuery, config }) {
  return rankResults(list, {
    query,
    translations: enQuery ? [enQuery] : [],
    limit: config.maxResults,
    maxPerProvider: config.maxPerProvider ?? 2,
    minScore: config.minRelevance ?? 0.28,
  });
}

/**
 * 执行一次联网资料检索。
 * @returns Promise<{ok, skipped?, query, mode, plan?, provider?, providers?, results?, attempts?, errors?, reason?, cached?}>
 */
export async function runWebSearch(text, options = {}) {
  const {
    config = DEFAULT_WEB_SEARCH,
    prefs = resolveSearchPrefs(config),
    settings = {},
    fetchImpl,
    signal: outerSignal,
    cache,
  } = options;
  // 即使短追问改用上一轮主题检索，也要先审查当前整条消息；不能因为它不会
  // 成为查询词，就绕过“含凭据时完全不联网”的隐私承诺。
  const currentAnalysis = analyzeSearchInput(text);
  const analyzed = analyzeSearchInput(options.queryText || text);
  const query = analyzed.query;
  const mode = prefs?.mode === 'smart' ? 'smart' : 'quick';
  if (currentAnalysis.blocked || analyzed.blocked) {
    return { ok: false, blocked: true, query: '', mode, reason: currentAnalysis.reason || analyzed.reason, attempts: [] };
  }
  // 纯代码 / 空白 / 寒暄：不发任何请求，界面应显示「已跳过」而不是「失败」。
  if (!query) {
    return { ok: false, skipped: true, query: '', mode, reason: '没有需要联网检索的文本（纯代码或空白）', attempts: [] };
  }
  if (isTrivialQuery(query)) {
    return { ok: false, skipped: true, query, mode, reason: '寒暄或极短消息，无需联网检索', attempts: [] };
  }
  if (!config?.enabled) return { ok: false, query, mode, reason: '站点已关闭联网资料检索', attempts: [] };
  if (!prefs?.enabled) return { ok: false, query, mode, reason: '联网资料检索未开启', attempts: [] };

  const targets = planProviders(config, settings);
  if (!targets.length) return { ok: false, query, mode, reason: '没有可用的资料源', attempts: [] };

  const nowMs = Date.now();
  const ttlMs = Math.max(0, Number(options.cacheTtlMs ?? config.cacheTtlMs ?? 0) || 0);
  const timeoutMs = options.timeoutMs ?? config.timeoutMs ?? DEFAULT_WEB_SEARCH.timeoutMs;
  const totalTimeoutMs = options.totalTimeoutMs ?? config.totalTimeoutMs ?? DEFAULT_WEB_SEARCH.totalTimeoutMs;

  // ---- 跨语言查询规划：翻译（保留意图）→ 技术片段 → 维基条目名 ----
  const cjkQuery = hasCjk(query);
  const keywordQuery = searchKeywords(query) || query;
  const needEn = cjkQuery && targets.some((t) => EN_ONLY_PROVIDERS.has(t.id) || t.id === 'wikipedia');
  let enQuery = cjkQuery ? composeEnQuery(keywordQuery) : '';
  let enSource = '';
  if (enQuery) enSource = translateQuery(keywordQuery) ? 'glossary' : 'fragment';

  const key = cacheKey(query, mode, config, settings, targets, enQuery);
  const cached = readCache(cache, key, ttlMs, nowMs);
  if (cached) return { ...cached, cached: true };

  const controller = new AbortController();
  let deadlineHit = false;
  const forwardOuterAbort = () => controller.abort(outerSignal?.reason);
  if (outerSignal) {
    if (outerSignal.aborted) throw makeAbortError(outerSignal);
    outerSignal.addEventListener?.('abort', forwardOuterAbort, { once: true });
  }
  const startedAt = Date.now();
  const deadline = setTimeout(() => {
    deadlineHit = true;
    controller.abort(new Error('联网资料检索总超时'));
  }, Math.max(1000, totalTimeoutMs));
  const signal = controller.signal;
  const remainingMs = () => Math.max(1000, Math.min(timeoutMs, totalTimeoutMs - (Date.now() - startedAt)));

  try {
    const errors = [];
    const attempts = [];

    // 术语表与技术片段都没有英文时，才用维基 langlinks 兜底（一次请求 + 页面缓存）。
    // langlinks 内部会校验首条命中的相关性，避免把无关条目的英文名发给所有英文源。
    if (needEn && !enQuery) {
      const langKey = JSON.stringify({ langlink: keywordQuery, lang: config.wikipedia?.lang || 'zh' });
      const langCached = readCache(cache, langKey, ttlMs, nowMs);
      if (langCached) {
        enQuery = langCached.en || '';
        if (enQuery) enSource = 'langlinks';
      } else {
        try {
          enQuery = await awaitWithSignal(
            resolveEnQuery({
              query: keywordQuery,
              lang: config.wikipedia?.lang || 'zh',
              fetchImpl,
              signal,
              timeoutMs: remainingMs(),
            }),
            signal,
          );
          if (enQuery) enSource = 'langlinks';
          writeCache(cache, langKey, ttlMs, nowMs, { en: enQuery });
        } catch (err) {
          if (outerSignal?.aborted) throw makeAbortError(outerSignal);
          enQuery = ''; // 翻译失败只影响英文源，不阻塞中文来源
        }
      }
    }

    // 所有来源并发，避免“自定义源超时两次才开始默认源”的串行卡顿。
    const settled = await Promise.allSettled(
      targets.map((target) => {
        if (EN_ONLY_PROVIDERS.has(target.id) && cjkQuery && !enQuery) {
          return Promise.resolve({ __skipped: CJK_SKIP_MESSAGE });
        }
        return awaitWithSignal(
          runProvider(
            target.id,
            providerOptions(target.id, { query, enQuery, url: target.url, config, settings, fetchImpl, signal, timeoutMs }),
          ),
          signal,
        );
      }),
    );
    if (outerSignal?.aborted) throw makeAbortError(outerSignal);

    const groups = [];
    settled.forEach((outcome, i) => {
      const id = targets[i].id;
      if (outcome.status === 'fulfilled') {
        if (outcome.value?.__skipped) {
          attempts.push({ id, status: 'skipped', count: 0, message: outcome.value.__skipped });
          return;
        }
        const results = Array.isArray(outcome.value) ? outcome.value : [];
        if (results.length) {
          attempts.push({ id, status: 'ok', count: results.length });
          groups.push({ id, results });
        } else {
          attempts.push({ id, status: 'empty', count: 0 });
          errors.push({ id, message: '没有相关结果' });
        }
      } else {
        const timedOut = deadlineHit || signal.aborted || outcome.reason?.name === 'AbortError';
        const message = deadlineHit || signal.aborted ? '本次检索总超时' : String(outcome.reason?.message || outcome.reason).slice(0, 120);
        attempts.push({ id, status: timedOut ? 'timeout' : 'error', count: 0, message });
        errors.push({ id, message });
      }
    });

    // 两种模式都补正文，差别只在页数与是否读 HN 讨论。
    const pages = Math.max(0, Number(mode === 'smart' ? config.smart?.maxPages : config.quick?.maxPages) || 0);

    // 统一归一化 → 相关性排序 → 取前 N（不再按来源轮流坐庄）。
    const ranked = rankPlan(normalizeAll(groups, config, mode), { query: keywordQuery, enQuery, config });
    if (!ranked.length) {
      // 报错文案去重：四个源同时“没有相关结果”不该复读四遍；
      // 来源自己的具体诊断（如 SearXNG 返回 HTTP 503）排在通用文案前面。
      const messages = [...new Set(errors.map((error) => error.message).filter(Boolean))];
      const ordered = [...messages.filter((m) => m !== '没有相关结果'), ...messages.filter((m) => m === '没有相关结果')];
      const reason = deadlineHit
        ? '联网资料检索总超时，未拿到可用结果'
        : groups.length
          ? '检索到的候选与问题相关性不足或没有可读正文，未达到资料门槛'
          : ordered.slice(0, 3).join('；') || '资料检索失败';
      return { ok: false, query, mode, plan: { keyword: keywordQuery, en: enQuery, enSource, providers: groups.map((g) => g.id), pages }, reason, attempts, errors };
    }

    let results = ranked;
    if (pages > 0 && !deadlineHit) {
      try {
        const enriched = await awaitWithSignal(
          enrichWithExtracts(ranked, {
            maxPages: pages,
            readerTemplate: settings.readerTemplate || config.smart?.readerTemplate || '',
            stackexchangeSite: config.stackexchange?.site,
            includeHackerNews: mode === 'smart',
            includeWikipedia: true,
            includeStackExchange: true,
            fetchImpl,
            signal,
            timeoutMs: remainingMs(),
          }),
          signal,
        );
        // 正文到手后重新算分：有正文的条目通常更值得占编号。
        const enrichedList = mergeResults(enriched, { limit: 40, maxSnippetChars: config.maxSnippetChars }).filter(
          // 补正文后仍然只有标题（HN 无帖文/评论）的条目没有证据价值，不占编号。
          (item) => !(item.titleOnly && !item.extract),
        );
        results = rankResults(
          enrichedList,
          {
            query: keywordQuery,
            translations: enQuery ? [enQuery] : [],
            limit: config.maxResults,
            maxPerProvider: config.maxPerProvider ?? 2,
            minScore: config.minRelevance ?? 0.28,
          },
        );
      } catch (err) {
        if (outerSignal?.aborted) throw makeAbortError(outerSignal);
        // 正文补充失败时保留已经检索到的摘要，不假装“深入资料”一定存在。
        errors.push({ id: 'enrichment', message: deadlineHit ? '正文补充超时，已保留搜索摘要' : String(err?.message || err).slice(0, 120) });
      }
    }

    // 补正文后可能把「只有标题」的候选全部排除：此时宁可明说没有可用资料，
    // 也不要返回一个 ok=true 但 results=[] 的“成功”结果。
    if (!results.length) {
      return {
        ok: false,
        query,
        mode,
        plan: { keyword: keywordQuery, en: enQuery, enSource, providers: groups.map((g) => g.id), pages },
        reason: '检索到的候选没有可读正文或摘要，本条回答未使用资料',
        attempts,
        errors,
      };
    }

    const result = {
      ok: true,
      query,
      mode,
      plan: {
        keyword: keywordQuery,
        en: enQuery,
        enSource,
        providers: groups.map((group) => group.id),
        pages,
      },
      provider: results[0]?.provider || groups[0]?.id,
      providers: [...new Set(results.map((item) => item.provider).filter(Boolean))],
      results,
      attempts,
      errors,
      now: new Date(nowMs).toISOString(),
      ...(deadlineHit ? { partial: true } : {}),
    };
    // 部分结果是超时快照，不缓存，避免下一次发送被旧的、不完整的来源列表固定住。
    if (!deadlineHit) writeCache(cache, key, ttlMs, nowMs, result);
    return result;
  } finally {
    clearTimeout(deadline);
    outerSignal?.removeEventListener?.('abort', forwardOuterAbort);
  }
}
