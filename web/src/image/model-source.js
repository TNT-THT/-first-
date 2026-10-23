/**
 * 生图时读取一个模型文件（纯逻辑，存储与网络接口可注入，worker 与测试共用）。
 *
 * 顺序：
 * 1. 完整缓存（IMAGE_CACHE）有该文件 → 直接读取，不访问网络。
 * 2. 下载到一半（分段缓存 IMAGE_PART_CACHE 中已有字节）→ 只补下缺少的字节（Range 续传），
 *    已保存的字节不重下；补齐后写入完整缓存并读取。
 * 3. 什么都没有 → 从 Hugging Face 读取（与原来一致，不写入缓存）。
 */
import { IMAGE_CACHE, IMAGE_PART_CACHE, PART_BYTES, PART_CONCURRENCY, RETRY_BASE_MS, RETRY_MAX_MS, createLimiter } from './model-files.js';
import { downloadSegmentedFile, segmentStatus, withRetry } from '../download/segmented.js';

/**
 * @param {string} url 文件地址（与下载时的地址一致）
 * @param {number} expected 配置中的文件大小（字节）
 * @param {object} [opts]
 * @param {Cache API} [opts.storage] 为空时直接从网络读取
 * @param {Function} [opts.fetchImpl]
 * @param {Function} [opts.limit] 并发限制（createLimiter 返回的函数）；worker 中应传入同一个
 * @returns {Promise<Response>}
 */
export async function openModelFile(url, expected, { storage = null, fetchImpl = globalThis.fetch, limit } = {}) {
  if (!storage) return fetchImpl(url);
  const cache = await storage.open(IMAGE_CACHE);
  const hit = await cache.match(url);
  if (hit) return hit;

  if (expected > 0) {
    const parts = await storage.open(IMAGE_PART_CACHE);
    const st = await segmentStatus(parts, url, expected, PART_BYTES);
    if (st.bytes > 0) {
      // 只补下缺少的部分（已保存的字节不重下），补齐后整理为完整文件
      await downloadSegmentedFile({
        url,
        label: url,
        size: expected,
        finalCache: cache,
        parts,
        fetchImpl,
        partBytes: PART_BYTES,
        concurrency: PART_CONCURRENCY,
        limit: limit ?? createLimiter(PART_CONCURRENCY),
        retry: (task) => withRetry(task, { baseMs: RETRY_BASE_MS, maxMs: RETRY_MAX_MS }),
        addBytes: () => {},
      });
      const done = await cache.match(url);
      if (done) return done;
    }
  }
  return fetchImpl(url);
}
