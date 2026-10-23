/**
 * 文本模型权重的并行预下载（默认 12 路并发）。
 *
 * WebLLM 内置的下载固定为 4 路并行，无法从外部调整。这里在加载模型之前，
 * 先把缺少的分片以更高的并发下载到 WebLLM 使用的缓存中（Cache Storage，
 * 作用域 webllm/model，键为分片的完整地址）。WebLLM 随后发现全部分片已经缓存，
 * 直接从缓存加载。
 *
 * - 暂时性错误（网络中断、5xx、408、429）自动重试，直到成功或被取消。
 * - 永久错误（其他 4xx）记录后继续下载其他分片；不抛出，交给 WebLLM 加载时报告。
 * - 取消（AbortError）原样抛出。
 */
import { createLimiter } from '../image/model-files.js';

export const TEXT_CACHE_SCOPE = 'webllm/model';
export const TEXT_PREFETCH_CONCURRENCY = 12;
const MANIFEST = 'tensor-cache.json';
const RETRY_BASE_MS = 1000;
const RETRY_MAX_MS = 30000;

class PermanentFetchError extends Error {}

function abortError() {
  return Object.assign(new Error('已取消下载'), { name: 'AbortError' });
}

function sleepMs(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(abortError());
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * 与 WebLLM 的 cleanModelUrl 相同：补全末尾的 "/"，没有 "/resolve/…/" 时补上 "resolve/main/"。
 * 返回绝对地址；无法解析时返回 null。
 */
export function textModelBaseUrl(modelField) {
  let url = String(modelField || '');
  if (!url) return null;
  url += url.endsWith('/') ? '' : '/';
  if (!url.match(/.+\/resolve\/.+\//)) url += 'resolve/main/';
  try {
    return new URL(url).href;
  } catch {
    return null;
  }
}

/** 模型记录中的 model 字段是否为可以预下载的 https 地址（本地“新建 AI”等虚拟地址不下载）。 */
export function isPrefetchableRecord(record) {
  return typeof record?.model === 'string' && /^https:\/\//.test(record.model) && !record.model.includes('.invalid/');
}

/**
 * 预下载缺少的分片。
 * @returns {Promise<{total: number, cached: number, failed: string[]}>} failed 为永久失败的分片地址
 */
export async function prefetchTextModelShards({
  modelUrl,
  storage = globalThis.caches,
  fetchImpl = globalThis.fetch,
  concurrency = TEXT_PREFETCH_CONCURRENCY,
  onProgress,
  signal,
  sleep = sleepMs,
  retryBaseMs = RETRY_BASE_MS,
  retryMaxMs = RETRY_MAX_MS,
} = {}) {
  const cache = await storage.open(TEXT_CACHE_SCOPE);
  const manifestUrl = new URL(MANIFEST, modelUrl).href;

  const fetchWithRetry = async (url, handle) => {
    for (let attempt = 1; ; attempt += 1) {
      if (signal?.aborted) throw abortError();
      try {
        return await handle(url);
      } catch (err) {
        if (err?.name === 'AbortError' || signal?.aborted) throw err;
        if (err instanceof PermanentFetchError) throw err;
        const waitMs = Math.min(retryMaxMs, retryBaseMs * 2 ** Math.min(attempt - 1, 16));
        await sleep(waitMs, signal);
      }
    }
  };

  const fetchOk = async (url) => {
    const res = await fetchImpl(url, { signal });
    if (!res.ok) {
      const permanent = res.status >= 400 && res.status < 500 && res.status !== 408 && res.status !== 429;
      if (permanent) throw new PermanentFetchError(`下载失败（HTTP ${res.status}）：${url}`);
      throw new Error(`服务器返回 ${res.status}：${url}`);
    }
    return res;
  };

  // 清单：已缓存则直接读取，否则下载并保存（与 WebLLM 的做法一致）
  let manifestRes = await cache.match(manifestUrl);
  if (!manifestRes) {
    await fetchWithRetry(manifestUrl, async (url) => {
      const res = await fetchOk(url);
      await cache.put(url, res);
    });
    manifestRes = await cache.match(manifestUrl);
  }
  const manifest = await manifestRes.json();
  const shards = new Map();
  for (const rec of manifest.records || []) {
    const url = new URL(rec.dataPath, modelUrl).href;
    if (!shards.has(url)) shards.set(url, Number(rec.nbytes) || 0);
  }

  const cachedKeys = new Set((await cache.keys()).map((req) => req.url));
  const missing = [...shards.keys()].filter((url) => !cachedKeys.has(url));
  const totalBytes = [...shards.values()].reduce((a, b) => a + b, 0);
  let doneBytes = [...shards.entries()].filter(([url]) => cachedKeys.has(url)).reduce((a, [, b]) => a + b, 0);
  let doneCount = shards.size - missing.length;
  const failed = [];

  const report = (text) => {
    onProgress?.({ progress: totalBytes ? doneBytes / totalBytes : 1, text });
  };
  report(`正在下载模型文件：${doneCount}/${shards.size} 个分片（${concurrency} 路并行）…`);

  const limit = createLimiter(concurrency);
  await Promise.all(
    missing.map((url) =>
      fetchWithRetry(url, (u) =>
        limit(async () => {
          const res = await fetchOk(u);
          await cache.put(u, res);
          doneBytes += shards.get(u);
          doneCount += 1;
          report(`正在下载模型文件：${doneCount}/${shards.size} 个分片（${concurrency} 路并行）…`);
        }),
      ).catch((err) => {
        if (err?.name === 'AbortError' || signal?.aborted) throw err;
        if (err instanceof PermanentFetchError) {
          failed.push(url);
          return;
        }
        throw err;
      }),
    ),
  );

  return { total: shards.size, cached: doneCount, failed };
}
