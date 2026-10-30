/**
 * 联网资料检索的编排（纯逻辑，fetch 与缓存可注入）。
 *
 * - 默认资料源与管理员配置源并行请求、合并去重；任一源不能独占答案；
 * - 深入模式补取维基导言、Stack Overflow 回答、HN 讨论摘录；
 * - 单个请求与全局检索各有超时；成功结果可使用页面内短缓存；
 * - 查询经过敏感信息检查，疑似凭据绝不外发。
 */

import { DEFAULT_WEB_SEARCH, planProviders, resolveSearchPrefs } from './config.js';
import { analyzeSearchInput, filterRelevantResults, searchKeywords, wikipediaLangs } from './query.js';
import { enrichWithExtracts, runProvider } from './providers.js';
import { interleaveByProvider, mergeResults } from './context.js';

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

function providerOptions(id, { query, url, config, settings, fetchImpl, signal, timeoutMs }) {
  return {
    query: searchKeywords(query) || query,
    langs: id === 'wikipedia' ? wikipediaLangs(query, config.wikipedia) : undefined,
    site: id === 'stackexchange' ? config.stackexchange?.site : undefined,
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

function cacheKey(query, mode, config, settings, targets) {
  return JSON.stringify({
    query,
    mode,
    targets: targets.map((target) => [target.id, target.url || '']),
    maxResults: config.maxResults,
    maxSnippetChars: config.maxSnippetChars,
    wikipedia: config.wikipedia,
    stackexchange: config.stackexchange,
    smart: mode === 'smart' ? { maxPages: config.smart?.maxPages, readerTemplate: settings.readerTemplate || config.smart?.readerTemplate || '' } : null,
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

/**
 * 执行一次联网资料检索。
 * @returns Promise<{ok, query, mode, provider?, providers?, results?, attempts?, errors?, reason?, cached?}>
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
    return { ok: false, query: '', mode, reason: currentAnalysis.reason || analyzed.reason, attempts: [] };
  }
  if (!query) return { ok: false, query: '', mode, reason: '没有可安全外发的问题文本可用于检索', attempts: [] };
  if (!config?.enabled) return { ok: false, query, mode, reason: '站点已关闭联网资料检索', attempts: [] };
  if (!prefs?.enabled) return { ok: false, query, mode, reason: '联网资料检索未开启', attempts: [] };

  const targets = planProviders(config, settings);
  if (!targets.length) return { ok: false, query, mode, reason: '没有可用的资料源', attempts: [] };

  const nowMs = Date.now();
  const ttlMs = Math.max(0, Number(options.cacheTtlMs ?? config.cacheTtlMs ?? 0) || 0);
  const key = cacheKey(query, mode, config, settings, targets);
  const cached = readCache(cache, key, ttlMs, nowMs);
  if (cached) return { ...cached, cached: true };

  const timeoutMs = options.timeoutMs ?? config.timeoutMs ?? DEFAULT_WEB_SEARCH.timeoutMs;
  const totalTimeoutMs = options.totalTimeoutMs ?? config.totalTimeoutMs ?? DEFAULT_WEB_SEARCH.totalTimeoutMs;
  const controller = new AbortController();
  let deadlineHit = false;
  const forwardOuterAbort = () => controller.abort(outerSignal?.reason);
  if (outerSignal) {
    if (outerSignal.aborted) throw makeAbortError(outerSignal);
    outerSignal.addEventListener?.('abort', forwardOuterAbort, { once: true });
  }
  const deadline = setTimeout(() => {
    deadlineHit = true;
    controller.abort(new Error('联网资料检索总超时'));
  }, Math.max(1000, totalTimeoutMs));
  const signal = controller.signal;

  try {
    const errors = [];
    const attempts = [];
    // 所有来源并发，避免“自定义源超时两次才开始默认源”的串行卡顿。
    const settled = await Promise.allSettled(
      targets.map((target) =>
        awaitWithSignal(
          runProvider(target.id, providerOptions(target.id, { query, url: target.url, config, settings, fetchImpl, signal, timeoutMs })),
          signal,
        ),
      ),
    );
    if (outerSignal?.aborted) throw makeAbortError(outerSignal);

    const groups = [];
    const relevanceQuery = searchKeywords(query) || query;
    settled.forEach((outcome, i) => {
      const id = targets[i].id;
      if (outcome.status === 'fulfilled') {
        const results = filterRelevantResults(
          relevanceQuery,
          mergeResults(outcome.value, { limit: config.maxResults, maxSnippetChars: config.maxSnippetChars }),
        );
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

    const merged = interleaveByProvider(groups, { limit: config.maxResults, maxSnippetChars: config.maxSnippetChars });
    if (!merged.length) {
      const reason = deadlineHit
        ? '联网资料检索总超时，未拿到可用结果'
        : errors.map((error) => error.message).filter(Boolean).slice(0, 2).join('；') || '资料检索失败';
      return { ok: false, query, mode, reason, attempts, errors };
    }

    let results = merged;
    if (mode === 'smart' && !deadlineHit) {
      try {
        results = mergeResults(
          await awaitWithSignal(
            enrichWithExtracts(merged, {
              maxPages: config.smart?.maxPages ?? 0,
              readerTemplate: settings.readerTemplate || config.smart?.readerTemplate || '',
              stackexchangeSite: config.stackexchange?.site,
              fetchImpl,
              signal,
              timeoutMs,
            }),
            signal,
          ),
          { limit: config.maxResults, maxSnippetChars: config.maxSnippetChars },
        );
      } catch (err) {
        if (outerSignal?.aborted) throw makeAbortError(outerSignal);
        // 正文补充失败时保留已经检索到的摘要，不假装“深入资料”一定存在。
        errors.push({ id: 'enrichment', message: deadlineHit ? '正文补充超时，已保留搜索摘要' : String(err?.message || err).slice(0, 120) });
      }
    }

    const result = {
      ok: true,
      query,
      mode,
      provider: groups[0]?.id,
      providers: groups.map((group) => group.id),
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
