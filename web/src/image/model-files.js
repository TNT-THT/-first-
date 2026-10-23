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
export const PART_CONCURRENCY = 4;
const PART_ORIGIN = 'https://cache.oj-ai.invalid/parts/';

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

function partPrefix(url) {
  return `${PART_ORIGIN}${encodeURIComponent(url)}/`;
}

function partKey(url, index) {
  return `${partPrefix(url)}${index}`;
}

function usesParts(size, partBytes) {
  return size > partBytes;
}

function partRange(size, partBytes, index) {
  const start = index * partBytes;
  const end = Math.min(size, start + partBytes) - 1; // 含端点
  return { start, end, length: end - start + 1 };
}

function partCount(size, partBytes) {
  return Math.ceil(size / partBytes);
}

/** 已缓存的分段（仅统计长度与配置一致的段）。返回 { indices: Set, bytes }。 */
async function cachedParts(parts, url, size, partBytes) {
  const indices = new Set();
  let bytes = 0;
  for (let i = 0; i < partCount(size, partBytes); i += 1) {
    const res = await parts.match(partKey(url, i));
    if (!res) continue;
    const len = Number(res.headers.get('content-length'));
    if (len !== partRange(size, partBytes, i).length) continue;
    indices.add(i);
    bytes += len;
  }
  return { indices, bytes };
}

/**
 * 模型文件的缓存状态。
 * - complete / cached / missing：完整文件的情况（与之前相同）。
 * - partialBytes / partsCached：未完成文件中已保存的分段（可续传，不会重下）。
 * - doneBytes：已完成文件与已保存分段的字节数合计，用于显示进度。
 * @returns {Promise<{total: number, cached: number, complete: boolean, missing: string[], bytes: number, partialBytes: number, partsCached: number, doneBytes: number}>}
 */
export async function imageCacheStatus(model, { storage = globalThis.caches, partBytes = PART_BYTES } = {}) {
  const urls = sourceUrls(model);
  const keys = Object.keys(urls);
  const have = await cachedUrls(storage);
  const missing = keys.filter((k) => !have.has(urls[k]));
  const sizeOf = (k) => Number(model.files[k]?.size) || 0;
  const bytes = keys.reduce((n, k) => n + sizeOf(k), 0);
  let partialBytes = 0;
  let partsCached = 0;
  const parts = await openParts(storage);
  for (const k of missing) {
    if (!usesParts(sizeOf(k), partBytes)) continue;
    const got = await cachedParts(parts, urls[k], sizeOf(k), partBytes);
    partialBytes += got.bytes;
    partsCached += got.indices.size;
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

/** 分段下载时服务器忽略了 Range 头（返回 200），改为整文件下载。 */
class RangeUnsupported extends Error {}

/**
 * 下载模型文件到本机缓存。
 * - 已完整的文件跳过；大文件的已完成分段也跳过，因此中断后再次点击会从断点继续（不会从头重下）。
 * - 大文件按 PART_BYTES 分段，段之间并行下载，提高吞吐量。
 * - 任一步失败时抛出错误；已完成的文件与分段保留。
 * @param {object} p
 * @param {(p: {done: number, total: number, key: string|null, bytes: number, totalBytes: number}) => void} [p.onProgress]
 *   done/total 为文件数；bytes/totalBytes 为字节数（包括上次已保存的部分）。开始时会先报告一次（key 为 null）。
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
  } = {},
) {
  if (!isSelectable(model)) throw new Error('该模型暂不可用，无法下载');
  const urls = sourceUrls(model);
  const status = await imageCacheStatus(model, { storage, partBytes });
  const cache = await openCache(storage);
  const parts = await openParts(storage);
  const sizeOf = (k) => Number(model.files[k].size) || 0;
  const totalBytes = status.bytes;
  let bytes = status.doneBytes;
  let done = status.cached;
  const total = status.total;
  onProgress?.({ done, total, key: null, bytes, totalBytes });

  // 每完成一段（或一个小文件）就报告一次字节进度
  let current = null;
  const addBytes = (n) => {
    bytes += n;
    onProgress?.({ done, total, key: current, bytes, totalBytes });
  };

  for (const key of status.missing) {
    const url = urls[key];
    const size = sizeOf(key);
    current = key;
    if (usesParts(size, partBytes)) {
      try {
        await downloadInParts({ url, path: model.files[key].path, size, parts, cache, fetchImpl, signal, partBytes, concurrency, addBytes });
      } catch (err) {
        if (!(err instanceof RangeUnsupported)) throw err;
        // 服务器不支持分段：整文件下载（此时无法断点续传，但仍然正确）
        await downloadWhole({ url, path: model.files[key].path, cache, fetchImpl, signal });
        addBytes(size);
      }
    } else {
      await downloadWhole({ url, path: model.files[key].path, cache, fetchImpl, signal });
      addBytes(size);
    }
    current = null;
    done += 1;
    onProgress?.({ done, total, key, bytes, totalBytes });
  }
  const after = await imageCacheStatus(model, { storage, partBytes });
  if (!after.complete) throw new Error(`下载不完整，缺少 ${after.missing.length} 个文件`);
  return after;
}

async function downloadWhole({ url, path, cache, fetchImpl, signal }) {
  const res = await fetchImpl(url, { signal });
  if (!res.ok) throw new Error(`下载失败（HTTP ${res.status}）：${path}`);
  await cache.put(url, res);
}

async function downloadInParts({ url, path, size, parts, cache, fetchImpl, signal, partBytes, concurrency, addBytes }) {
  const n = partCount(size, partBytes);
  const have = await cachedParts(parts, url, size, partBytes);
  const todo = [];
  for (let i = 0; i < n; i += 1) if (!have.indices.has(i)) todo.push(i);

  // 用户取消时中止所有进行中的请求；某段失败时不再启动新段，但进行中的段仍会完成并保存（不浪费已下载的数据）
  const inner = new AbortController();
  const onAbort = () => inner.abort();
  signal?.addEventListener('abort', onAbort);
  const fetchPart = async (i) => {
    const { start, end, length } = partRange(size, partBytes, i);
    const res = await fetchImpl(url, { signal: inner.signal, headers: { Range: `bytes=${start}-${end}` } });
    if (res.status === 200) {
      await res.body?.cancel();
      throw new RangeUnsupported();
    }
    if (res.status !== 206) throw new Error(`下载失败（HTTP ${res.status}）：${path}（第 ${i + 1}/${n} 段）`);
    const buf = new Uint8Array(await res.arrayBuffer());
    if (buf.length !== length) throw new Error(`分段大小不符（第 ${i + 1}/${n} 段）：${path}`);
    await parts.put(partKey(url, i), new Response(buf, { headers: { 'content-length': String(length) } }));
    addBytes(length);
  };
  let failure = null;
  let next = 0;
  const worker = async () => {
    while (!failure && next < todo.length) {
      const i = todo[next];
      next += 1;
      try {
        await fetchPart(i);
      } catch (err) {
        failure ??= err;
        return;
      }
    }
  };
  try {
    const workers = Array.from({ length: Math.min(Math.max(1, concurrency), todo.length) }, worker);
    await Promise.all(workers);
  } finally {
    signal?.removeEventListener('abort', onAbort);
  }
  if (failure) throw failure;

  // 全部分段就绪：按顺序拼接成完整文件放入 IMAGE_CACHE，然后删除分段
  let index = 0;
  let reader = null;
  const body = new ReadableStream({
    async pull(controller) {
      for (;;) {
        if (!reader) {
          if (index >= n) {
            controller.close();
            return;
          }
          const res = await parts.match(partKey(url, index));
          if (!res) {
            controller.error(new Error(`缓存分段丢失（第 ${index + 1}/${n} 段）：${path}`));
            return;
          }
          reader = res.body.getReader();
        }
        const { done, value } = await reader.read();
        if (done) {
          reader = null;
          index += 1;
          continue;
        }
        controller.enqueue(value);
        return;
      }
    },
    cancel() {
      reader?.cancel();
    },
  });
  await cache.put(url, new Response(body, { headers: { 'content-length': String(size) } }));
  for (let i = 0; i < n; i += 1) await parts.delete(partKey(url, i));
}

/** 删除该模型在本机缓存中的全部文件（含未完成下载的分段）。返回删除的完整文件数。 */
export async function deleteImageModelFiles(model, { storage = globalThis.caches } = {}) {
  const cache = await openCache(storage);
  const parts = await openParts(storage);
  let removed = 0;
  for (const url of Object.values(sourceUrls(model))) {
    if (await cache.delete(url)) removed += 1;
    const prefix = partPrefix(url);
    for (const req of await parts.keys()) {
      if (req.url.startsWith(prefix)) await parts.delete(req.url);
    }
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
