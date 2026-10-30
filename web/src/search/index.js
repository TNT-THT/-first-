/**
 * 联网搜索的编排（纯逻辑，fetch 与存储可注入，便于测试）。
 *
 * **前提：没有 API Key，也没有独立服务器。**
 * - 免费来源（维基百科、Stack Overflow、Hacker News）并行请求、合并去重，任何一个失败都不影响其它；
 * - 管理员另填了 SearXNG / 自定义接口时，先用它们（能搜整个网页），拿不到结果再用免费来源；
 * - 智能搜索（smart）在结果上再补一段正文摘要。
 *
 * 隐私：搜索请求直接从浏览器发给搜索服务，只包含提取出的查询词，
 * 不发送对话历史、模型设置或任何密钥。
 */

import { DEFAULT_WEB_SEARCH, planAdvanced, planKeyless, resolveSearchPrefs } from './config.js';
import { buildSearchQuery, wikipediaLangs } from './query.js';
import { enrichWithExtracts, runProvider } from './providers.js';
import { interleaveByProvider, mergeResults } from './context.js';

function isAbort(err, signal) {
  return !!signal?.aborted || err?.name === 'AbortError';
}

function providerOptions(id, { query, url, config, settings, fetchImpl, signal, timeoutMs }) {
  return {
    query,
    // wikipedia 的 langs / stackexchange 的 site 都来自站点配置
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

/**
 * 执行一次联网搜索。
 * @param {string} text 用户的原消息
 * @param {{config?: object, prefs?: object, settings?: object, fetchImpl?: Function,
 *          signal?: AbortSignal, timeoutMs?: number}} options
 * @returns {Promise<{ok: boolean, query: string, mode: string, provider?: string, providers?: string[],
 *                    results?: object[], reason?: string, errors?: Array<{id: string, message: string}>}>}
 */
export async function runWebSearch(text, options = {}) {
  const {
    config = DEFAULT_WEB_SEARCH,
    prefs = resolveSearchPrefs(config),
    settings = {},
    fetchImpl,
    signal,
    now = new Date(),
  } = options;
  const timeoutMs = options.timeoutMs ?? config.timeoutMs ?? DEFAULT_WEB_SEARCH.timeoutMs;

  const query = buildSearchQuery(text);
  const mode = prefs.mode === 'smart' ? 'smart' : 'quick';
  if (!query) return { ok: false, query: '', mode, reason: '没有问题文本可用于搜索' };
  if (!config?.enabled) return { ok: false, query, mode, reason: '站点已关闭联网搜索' };

  const advanced = planAdvanced(config, settings);
  const keyless = planKeyless(config);
  if (!advanced.length && !keyless.length) {
    return { ok: false, query, mode, reason: '没有可用的搜索源' };
  }

  const errors = [];
  const finish = async (usable, providerIds) => {
    const enriched =
      mode === 'smart'
        ? mergeResults(
            await enrichWithExtracts(usable, {
              maxPages: config.smart?.maxPages ?? 0,
              readerTemplate: settings.readerTemplate || config.smart?.readerTemplate || '',
              fetchImpl,
              signal,
              timeoutMs,
            }),
            { limit: config.maxResults, maxSnippetChars: config.maxSnippetChars },
          )
        : usable;
    return {
      ok: true,
      query,
      mode,
      provider: providerIds[0],
      providers: providerIds,
      results: enriched,
      errors,
      now,
    };
  };

  // 1) 进阶来源（需要管理员配置）：按优先级依次尝试，失败换下一个
  const advancedResults = [];
  for (const target of advanced) {
    if (signal?.aborted) throw Object.assign(new Error('已取消'), { name: 'AbortError' });
    try {
      const raw = await runProvider(target.id, {
        ...providerOptions(target.id, { query, url: target.url, config, settings, fetchImpl, signal, timeoutMs }),
      });
      const results = mergeResults(raw, { limit: config.maxResults, maxSnippetChars: config.maxSnippetChars });
      if (results.length) {
        advancedResults.push({ id: target.id, results });
        break; // 进阶来源通常已经覆盖整个网页，不再重复请求其它进阶来源
      }
      errors.push({ id: target.id, message: '没有搜索结果' });
    } catch (err) {
      if (isAbort(err, signal)) throw err;
      errors.push({ id: target.id, message: err?.message || String(err) });
    }
  }
  if (advancedResults.length) {
    return finish(advancedResults[0].results, [advancedResults[0].id]);
  }

  // 2) 免费来源（免密钥、允许跨域）：并行请求，合并去重
  if (!keyless.length) {
    const reason = errors.map((e) => e.message).filter(Boolean).slice(0, 2).join('；');
    return { ok: false, query, mode, reason: reason || '搜索失败', errors };
  }
  const settled = await Promise.allSettled(
    keyless.map((target) =>
      runProvider(
        target.id,
        providerOptions(target.id, { query, url: '', config, settings, fetchImpl, signal, timeoutMs }),
      ),
    ),
  );
  if (signal?.aborted) throw Object.assign(new Error('已取消'), { name: 'AbortError' });
  const groups = [];
  settled.forEach((outcome, i) => {
    const id = keyless[i].id;
    if (outcome.status === 'fulfilled') {
      const results = mergeResults(outcome.value, {
        limit: config.maxResults,
        maxSnippetChars: config.maxSnippetChars,
      });
      if (results.length) groups.push({ id, results });
      else errors.push({ id, message: '没有搜索结果' });
    } else {
      errors.push({ id, message: outcome.reason?.message || String(outcome.reason) });
    }
  });

  // 各来源轮流取，避免某一来源占满全部名额
  const merged = interleaveByProvider(groups, { limit: config.maxResults });
  if (merged.length) {
    return finish(
      merged,
      groups.map((g) => g.id),
    );
  }
  const reason = errors.map((e) => e.message).filter(Boolean).slice(0, 2).join('；');
  return { ok: false, query, mode, reason: reason || '搜索失败', errors };
}
