/**
 * 生图模型的本机文件：下载到浏览器、删除、导出 / 导入单文件（纯逻辑，存储接口可注入）。
 *
 * - 下载：把 config.json 中该模型的全部文件（按原始 Hugging Face 地址）写入 Cache Storage 的独立空间
 *   IMAGE_CACHE。已缓存的文件会跳过，因此中断后可以续传。生图时 worker 会先读这里的缓存，离线也可以生成。
 * - 导出：把缓存中的文件打包成一个 .ojimage 文件（格式与 .ojmodel 相同的二进制容器，但 format 不同，
 *   因此两种文件不会被误导入）。
 * - 导入：只接受与站点配置一致的模型（同一个 id、同样的地址、同样的大小）；写入同一个缓存空间。
 *
 * 文件格式（小端序）：[0,4) 魔数 "OJMD"；[4,8) 头部长度 uint32；[8,8+N) 头部 JSON；其后为各文件的原始字节。
 */
import { isSelectable, sourceUrls } from './catalog.js';
import {
  PermanentDownloadError,
  abortError,
  STALL_MS,
  deleteSegments,
  downloadSegmentedFile,
  segmentStatus,
  sleepMs,
  withRetry,
} from '../download/segmented.js';

export { PermanentDownloadError };

export const IMAGE_CACHE = 'oj-ai-image-models-v1';
// 分段下载的临时数据（与完整文件分开保存；文件完整后移入 IMAGE_CACHE 并删除分段）
export const IMAGE_PART_CACHE = 'oj-ai-image-parts-v1';
export const IMAGE_MODEL_EXT = '.ojimage';
export const IMAGE_FILE_FORMAT = 'oj-ai-image-model';
export const IMAGE_FILE_VERSION = 1;
const MAGIC = 'OJMD';
const PREFIX_BYTES = 8;
const MAX_HEADER_BYTES = 1024 * 1024;
/** 超过此大小的文件按分段下载（每段 64 MB），中断后只需重下未完成的段。 */
export const PART_BYTES = 64 * 1024 * 1024;
/** 同一文件内同时下载的段数。 */
// 同时下载的请求数上限（文件与分段共用）。每段约 64 MB 会整段读入内存，峰值约 12 × 64 MB ≈ 768 MB。
export const PART_CONCURRENCY = 12;

function safeName(id) {
  return String(id).replace(/[^A-Za-z0-9._-]+/g, '_');
}

export function imageModelFileName(modelId) {
  return `${safeName(modelId)}${IMAGE_MODEL_EXT}`;
}

async function openCache(storage) {
  return storage.open(IMAGE_CACHE);
}

async function openParts(storage) {
  return storage.open(IMAGE_PART_CACHE);
}

async function cachedUrls(storage) {
  const cache = await openCache(storage);
  return new Set((await cache.keys()).map((req) => req.url));
}

/**
 * 并发上限：返回 limit(fn)，同时运行的 fn 不超过 max 个（全部文件与分段共用同一上限）。
 * 只包住单次网络请求（含读取响应体），重试等待不占用名额。
 */
export function createLimiter(max) {
  let active = 0;
  const queue = [];
  const pump = () => {
    while (active < Math.max(1, max) && queue.length) {
      const job = queue.shift();
      active += 1;
      Promise.resolve()
        .then(job.fn)
        .then(job.resolve, job.reject)
        .finally(() => {
          active -= 1;
          pump();
        });
    }
  };
  return (fn) =>
    new Promise((resolve, reject) => {
      queue.push({ fn, resolve, reject });
      pump();
    });
}

/**
 * 下载模型文件到本机缓存（与文本模型一致：先把全部文件下载完，再从缓存校验）。
 * - 暂时性错误自动重试，下载过程中不报错；进度中会显示重试信息。
 * - 已完整的文件与已保存的分段会跳过，中断（取消）后再次下载从断点继续。
 * - 大文件按 PART_BYTES 分段，段之间并行下载。
 * - 永久错误记录后继续下载其他文件；全部完成后从缓存校验，缺失时抛出错误（已保存的部分保留）。
 * @param {object} p
 * @param {(p: {done: number, total: number, key: string|null, bytes: number, totalBytes: number, retry?: object, note?: string}) => void} [p.onProgress]
 *   done/total 为文件数；bytes/totalBytes 为字节数（含上次已保存的部分）。开始时先报告一次（key 为 null）。
 *   重试时带 retry: {attempt, waitMs, message}；说明性提示带 note（字节数不变）。
 */
/** 重试间隔：1 s、2 s、4 s …，最长 30 s。下载过程中不会因网络问题而报错，直到用户取消。 */
export const RETRY_BASE_MS = 1000;
export const RETRY_MAX_MS = 30000;

/**
 * 模型文件的缓存状态。
 * - complete / cached / missing：完整文件的情况。
 * - partialBytes：未完成文件中已保存的字节（含已完整的分段；网络中断时已收到的部分也计入）。
 * - partsCached：已完整的分段数。
 * - doneBytes：已完成文件与未完成文件已保存字节的合计，用于显示进度。
 * @returns {Promise<{total: number, cached: number, complete: boolean, missing: string[], bytes: number, partialBytes: number, partsCached: number, doneBytes: number}>}
 */
export async function imageCacheStatus(model, { storage = globalThis.caches, partBytes = PART_BYTES } = {}) {
  const urls = sourceUrls(model);
  const keys = Object.keys(urls);
  const have = await cachedUrls(storage);
  const missing = keys.filter((k) => !have.has(urls[k]));
  const sizeOf = (k) => Number(model.files[k]?.size) || 0;
  const bytes = keys.reduce((n, k) => n + sizeOf(k), 0);
  const parts = await openParts(storage);
  let partialBytes = 0;
  let partsCached = 0;
  for (const k of missing) {
    const st = await segmentStatus(parts, urls[k], sizeOf(k), partBytes);
    partialBytes += st.bytes;
    partsCached += st.completeCount;
  }
  const doneBytes = keys.filter((k) => !missing.includes(k)).reduce((n, k) => n + sizeOf(k), 0) + partialBytes;
  return {
    total: keys.length,
    cached: keys.length - missing.length,
    complete: missing.length === 0,
    missing,
    bytes,
    partialBytes,
    partsCached,
    doneBytes,
  };
}

/**
 * 下载模型文件到本机缓存（与文本模型一致：先把全部文件下载完，再从缓存加载）。
 * - 下载中收到的字节随时保存（网络中断、取消时也保存），下次从断点继续，不重下已保存的部分。
 * - 暂时性错误自动重试，下载过程中不报错；进度中会显示重试信息。
 * - 文件并行下载，同时的请求数不超过 concurrency。
 * - 永久错误记录后继续下载其他文件；全部完成后从缓存校验，缺失时抛出错误（已保存的部分保留）。
 * @param {object} p
 * @param {(p: {done: number, total: number, key: string|null, bytes: number, totalBytes: number, retry?: object, note?: string}) => void} [p.onProgress]
 *   done/total 为文件数；bytes/totalBytes 为字节数（含上次已保存的部分）。开始时先报告一次（key 为 null）。
 *   重试时带 retry: {attempt, waitMs, message}；说明性提示带 note（字节数不变）。
 */
export async function downloadImageModel(
  model,
  {
    storage = globalThis.caches,
    fetchImpl = globalThis.fetch,
    onProgress,
    signal,
    partBytes = PART_BYTES,
    concurrency = PART_CONCURRENCY,
    sleep = sleepMs,
    retryBaseMs = RETRY_BASE_MS,
    retryMaxMs = RETRY_MAX_MS,
    stallMs = STALL_MS,
  } = {},
) {
  if (!isSelectable(model)) throw new Error('该模型暂不可用，无法下载');
  const urls = sourceUrls(model);
  const status = await imageCacheStatus(model, { storage, partBytes });
  const cache = await openCache(storage);
  const parts = await openParts(storage);
  const sizeOf = (k) => Number(model.files[k].size) || 0;
  const totalBytes = status.bytes;
  const total = status.total;
  let bytes = status.doneBytes;
  let done = status.cached;
  onProgress?.({ done, total, key: null, bytes, totalBytes });
  const limit = createLimiter(concurrency);

  // 每个文件各自的进度回调（文件并行下载时，进度要带上对应的文件名）
  const progressFor = (key) => ({
    addBytes: (n) => {
      bytes += n;
      onProgress?.({ done, total, key, bytes, totalBytes });
    },
    // 说明性提示（跳过已保存部分、拼接进度等），不改变字节数
    onNote: (note) => onProgress?.({ done, total, key, bytes, totalBytes, note }),
    retry: (task) =>
      withRetry(task, {
        signal,
        sleep,
        baseMs: retryBaseMs,
        maxMs: retryMaxMs,
        onRetry: (info) => onProgress?.({ done, total, key, bytes, totalBytes, retry: info }),
      }),
  });

  const failures = [];
  const keys = status.missing;
  let nextKey = 0;
  let fatal = null;

  const downloadOne = async (key) => {
    const { addBytes, retry, onNote } = progressFor(key);
    try {
      await downloadSegmentedFile({
        url: urls[key],
        label: model.files[key].path,
        size: sizeOf(key),
        finalCache: cache,
        parts,
        fetchImpl,
        signal,
        partBytes,
        concurrency,
        limit,
        retry,
        addBytes,
        stallMs,
        onNote,
      });
    } catch (err) {
      if (err.name === 'AbortError') throw err;
      if (!(err instanceof PermanentDownloadError)) throw err;
      failures.push(err.message);
    }
    done += 1;
    onProgress?.({ done, total, key, bytes, totalBytes });
  };

  // 文件并行下载（同时最多 concurrency 个文件）
  const worker = async () => {
    while (!fatal && nextKey < keys.length) {
      const key = keys[nextKey];
      nextKey += 1;
      try {
        await downloadOne(key);
      } catch (err) {
        fatal ??= err;
        return;
      }
    }
  };
  const workers = Array.from({ length: Math.min(Math.max(1, concurrency), keys.length) }, worker);
  await Promise.allSettled(workers);
  if (signal?.aborted) throw abortError();
  if (fatal) throw fatal;

  // 全部文件处理完后，再从缓存校验（与文本模型一致：下载完成后从缓存加载，问题在此阶段报告）
  const after = await imageCacheStatus(model, { storage, partBytes });
  if (!after.complete) {
    const detail = failures.length ? `：${failures.join('；')}` : '';
    throw new Error(`缓存校验未通过，缺少 ${after.missing.length} 个文件${detail}。已保存的部分会保留，可点击“继续下载”重试缺失的文件。`);
  }
  return after;
}

/** 删除该模型在本机缓存中的全部文件（含未完成下载的分段）。返回删除的完整文件数。 */
export async function deleteImageModelFiles(model, { storage = globalThis.caches } = {}) {
  const cache = await openCache(storage);
  const parts = await openParts(storage);
  let removed = 0;
  for (const url of Object.values(sourceUrls(model))) {
    if (await cache.delete(url)) removed += 1;
    await deleteSegments(parts, url);
  }
  return removed;
}

function encodePrefix(headerLength) {
  const buf = new ArrayBuffer(PREFIX_BYTES);
  const view = new DataView(buf);
  for (let i = 0; i < MAGIC.length; i += 1) view.setUint8(i, MAGIC.charCodeAt(i));
  view.setUint32(4, headerLength, true);
  return new Uint8Array(buf);
}

/**
 * 把缓存中的模型文件打包为 .ojimage（Blob）。文件必须已完整下载，且大小与站点配置一致。
 * @returns {Promise<{blob: Blob, filename: string, size: number, entryCount: number}>}
 */
export async function buildImageModelFile(model, { storage = globalThis.caches, onProgress, now = new Date() } = {}) {
  const status = await imageCacheStatus(model, { storage });
  if (!status.complete) {
    throw new Error(`模型文件尚未完整下载（缺少 ${status.missing.length} 个文件）。请先点击“下载到本机”。`);
  }
  const cache = await openCache(storage);
  const urls = sourceUrls(model);
  const entries = [];
  const blobs = [];
  let offset = 0;
  let i = 0;
  for (const key of Object.keys(urls)) {
    const res = await cache.match(urls[key]);
    if (!res) throw new Error(`缓存条目丢失：${model.files[key].path}`);
    const blob = await res.blob();
    const expected = Number(model.files[key].size);
    if (expected > 0 && blob.size !== expected) {
      throw new Error(`缓存文件大小与模型记录不一致（${model.files[key].path}），请删除后重新下载`);
    }
    entries.push({ key, path: model.files[key].path, url: urls[key], size: blob.size, offset });
    blobs.push(blob);
    offset += blob.size;
    i += 1;
    onProgress?.({ done: i, total: status.total });
  }
  const header = {
    format: IMAGE_FILE_FORMAT,
    version: IMAGE_FILE_VERSION,
    modelId: model.id,
    label: model.label || model.id,
    repo: model.repo,
    revision: model.revision || 'main',
    createdAt: now.toISOString(),
    entries,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const blob = new Blob([encodePrefix(headerBytes.byteLength), headerBytes, ...blobs], { type: 'application/octet-stream' });
  return { blob, filename: imageModelFileName(model.id), size: blob.size, entryCount: entries.length };
}

/** 校验头部结构。 */
export function validateImageHeader(h) {
  if (!h || typeof h !== 'object') throw new Error('模型文件头部无效');
  if (h.format !== IMAGE_FILE_FORMAT) throw new Error('文件格式不受支持，这不是本应用导出的生图模型');
  if (!Number.isInteger(h.version) || h.version > IMAGE_FILE_VERSION) {
    throw new Error(`生图模型文件版本 ${h.version} 高于当前支持的版本 ${IMAGE_FILE_VERSION}，请更新页面后再导入`);
  }
  if (typeof h.modelId !== 'string' || !h.modelId) throw new Error('生图模型文件缺少 modelId');
  if (!Array.isArray(h.entries) || h.entries.length === 0) throw new Error('生图模型文件没有任何数据条目');
  for (const e of h.entries) {
    if (typeof e.key !== 'string' || typeof e.url !== 'string' || !/^https:\/\//.test(e.url)) {
      throw new Error('生图模型文件中的地址无效');
    }
    if (!Number.isSafeInteger(e.size) || e.size < 0 || !Number.isSafeInteger(e.offset) || e.offset < 0) {
      throw new Error('生图模型文件中的偏移或大小无效');
    }
  }
  return h;
}

/** 读取并校验 .ojimage 头部（不读取文件内容）。 */
export async function readImageModelHeader(file) {
  if (!file || file.size < PREFIX_BYTES) throw new Error('模型文件不完整');
  const prefix = new DataView(await file.slice(0, PREFIX_BYTES).arrayBuffer());
  let magic = '';
  for (let i = 0; i < MAGIC.length; i += 1) magic += String.fromCharCode(prefix.getUint8(i));
  if (magic !== MAGIC) throw new Error('这不是本应用导出的生图模型文件');
  const headerLength = prefix.getUint32(4, true);
  if (headerLength === 0 || headerLength > MAX_HEADER_BYTES || PREFIX_BYTES + headerLength > file.size) {
    throw new Error('生图模型文件头部损坏或不完整');
  }
  let header;
  try {
    header = JSON.parse(await file.slice(PREFIX_BYTES, PREFIX_BYTES + headerLength).text());
  } catch {
    throw new Error('生图模型文件头部无法解析');
  }
  validateImageHeader(header);
  const dataStart = PREFIX_BYTES + headerLength;
  const dataSize = file.size - dataStart;
  for (const e of header.entries) {
    if (e.offset + e.size > dataSize) throw new Error(`生图模型文件数据不完整：${e.path || e.key}`);
  }
  return { header, dataStart };
}

/**
 * 导入 .ojimage：必须与站点配置中的同一个模型一致（id、文件键、地址、大小），否则拒绝。
 * @param {Array<object>} models config.json 中的 imageModels
 * @returns {Promise<{model: object, entryCount: number}>}
 */
export async function importImageModelFile(file, { models, storage = globalThis.caches, onProgress } = {}) {
  const { header, dataStart } = await readImageModelHeader(file);
  const model = (models || []).find((m) => m.id === header.modelId);
  if (!model) throw new Error(`这个文件对应的生图模型「${header.modelId}」不在本站配置中，无法导入`);
  if (!isSelectable(model)) throw new Error(`模型「${model.label || model.id}」暂不可用，无法导入`);

  const expected = sourceUrls(model);
  const keys = Object.keys(expected);
  const seen = new Set();
  for (const e of header.entries) {
    if (!(e.key in expected)) throw new Error(`生图模型文件包含未知的文件：${e.key}`);
    if (seen.has(e.key)) throw new Error(`生图模型文件中重复的文件：${e.key}`);
    seen.add(e.key);
    if (e.url !== expected[e.key]) throw new Error(`文件地址与站点配置不一致（${e.key}），文件可能已损坏或来自其他版本`);
    const want = Number(model.files[e.key].size);
    if (want > 0 && e.size !== want) throw new Error(`文件大小与站点配置不一致（${e.key}）`);
  }
  if (seen.size !== keys.length) throw new Error('生图模型文件缺少部分文件，无法导入');

  const cache = await openCache(storage);
  const written = [];
  try {
    let done = 0;
    for (const e of header.entries) {
      const body = file.slice(dataStart + e.offset, dataStart + e.offset + e.size);
      await cache.put(e.url, new Response(body, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } }));
      written.push(e.url);
      done += 1;
      onProgress?.({ done, total: header.entries.length });
    }
    const status = await imageCacheStatus(model, { storage });
    if (!status.complete) throw new Error('导入后文件仍不完整');
  } catch (err) {
    // 失败时清理本次写入的文件，避免留下半成品
    for (const url of written) {
      try {
        await cache.delete(url);
      } catch {
        /* 忽略清理错误 */
      }
    }
    throw err;
  }
  return { model, entryCount: header.entries.length };
}


/**
 * 浏览器中单个 ArrayBuffer 的上限（本项目测试的 Chromium 中，2 GB 的分配即失败；1.7 GB 可以成功）。
 * 模型的每个文件都要整体读入一个 ArrayBuffer 再交给 ONNX Runtime，超过上限的文件无法在网页中加载。
 */
export const MAX_BROWSER_FILE_BYTES = 2 ** 31 - 1;

/**
 * 返回大小超过浏览器单文件上限的文件名（sizes 为 { 文件键: 字节数 }）。
 * 只检查 sizes 中的数值；缺失或为 0 的项目视为未知，不报错。
 */
export function oversizedFiles(sizes, limit = MAX_BROWSER_FILE_BYTES) {
  return Object.entries(sizes || {})
    .filter(([, n]) => Number(n) > limit)
    .map(([key, n]) => ({ key, bytes: Number(n) }));
}
