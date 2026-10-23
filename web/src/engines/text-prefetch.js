/**
 * 文本模型权重的并行预下载（默认 12 路并发）。
 *
 * WebLLM 内置的下载固定为 4 路并行，无法从外部调整。这里在加载模型之前，
 * 先把缺少的分片以更高的并发下载到 WebLLM 使用的缓存中（Cache Storage，
 * 作用域 webllm/model，键为分片的完整地址）。WebLLM 随后发现全部分片已经缓存，
 * 直接从缓存加载。
 *
 * - 分片的下载与续传与生图模型相同（见 download/segmented.js）：网络中断、取消时已收到的字节也会保存，
 *   下次从断点继续；分片完整后才放入 webllm/model 缓存。
 * - 暂时性错误自动重试，直到成功或被取消。
 * - 永久错误（其他 4xx）记录后继续下载其他分片；不抛出，交给 WebLLM 加载时报告。
 * - 取消（AbortError）原样抛出。
 */
import { createLimiter } from '../image/model-files.js';
import {
  PermanentDownloadError,
  STALL_MS,
  abortError,
  downloadSegmentedFile,
  isPermanentStatus,
  segmentStatus,
  withRetry,
} from '../download/segmented.js';

export const TEXT_CACHE_SCOPE = 'webllm/model';
/** 分片的临时数据（与 WebLLM 的缓存分开；分片完整后移入 TEXT_CACHE_SCOPE 并删除）。 */
export const TEXT_PARTS_SCOPE = 'oj-ai-text-parts-v1';
export const TEXT_PREFETCH_CONCURRENCY = 12;
export const TEXT_PART_BYTES = 64 * 1024 * 1024;
const MANIFEST = 'tensor-cache.json';
const RETRY_BASE_MS = 1000;
const RETRY_MAX_MS = 30000;

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
  partBytes = TEXT_PART_BYTES,
  onProgress,
  signal,
  sleep,
  retryBaseMs = RETRY_BASE_MS,
  retryMaxMs = RETRY_MAX_MS,
  stallMs = STALL_MS,
} = {}) {
  const cache = await storage.open(TEXT_CACHE_SCOPE);
  const parts = await storage.open(TEXT_PARTS_SCOPE);
  const manifestUrl = new URL(MANIFEST, modelUrl).href;
  const retry = (task, onRetry) => withRetry(task, { signal, sleep, baseMs: retryBaseMs, maxMs: retryMaxMs, onRetry });

  // 清单：已缓存则直接读取，否则下载并保存（与 WebLLM 的做法一致）
  let manifestRes = await cache.match(manifestUrl);
  if (!manifestRes) {
    await retry(async () => {
      const res = await fetchImpl(manifestUrl, { signal });
      if (!res.ok) {
        if (isPermanentStatus(res.status)) throw new PermanentDownloadError(`下载失败（HTTP ${res.status}）：${manifestUrl}`);
        throw new Error(`服务器返回 ${res.status}：${manifestUrl}`);
      }
      await cache.put(manifestUrl, res);
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
  const totalBytes = [...shards.values()].reduce((a, b) => a + b, 0);
  // 需要下载的分片（大小未知的分片交给 WebLLM 自己下载）
  const missing = [...shards.entries()].filter(([url, nbytes]) => !cachedKeys.has(url) && nbytes > 0);
  let doneBytes = [...shards.entries()].filter(([url]) => cachedKeys.has(url)).reduce((a, [, b]) => a + b, 0);
  let doneCount = shards.size - missing.length;
  for (const [url, nbytes] of missing) {
    doneBytes += (await segmentStatus(parts, url, nbytes, partBytes)).bytes;
  }
  const failed = [];

  let note = null;
  const report = () => {
    onProgress?.({
      progress: totalBytes ? doneBytes / totalBytes : 1,
      text: note || `正在下载模型文件：${doneCount}/${shards.size} 个分片（${concurrency} 路并行）…`,
    });
  };
  report();

  const limit = createLimiter(concurrency);
  const addBytes = (n) => {
    doneBytes += n;
    report();
  };
  await Promise.all(
    missing.map(([url, nbytes]) =>
      downloadSegmentedFile({
        url,
        label: url,
        size: nbytes,
        finalCache: cache,
        parts,
        fetchImpl,
        signal,
        partBytes,
        concurrency,
        limit,
        retry: (task) => retry(task),
        addBytes,
        stallMs,
        onNote: (text) => {
          note = text;
          report();
        },
      }).then(
        () => {
          doneCount += 1;
          note = null;
          report();
        },
        (err) => {
          if (err?.name === 'AbortError' || signal?.aborted) throw err;
          if (err instanceof PermanentDownloadError) {
            failed.push(url);
            return;
          }
          throw err;
        },
      ),
    ),
  );
  if (signal?.aborted) throw abortError();

  return { total: shards.size, cached: doneCount, failed };
}
