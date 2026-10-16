/**
 * 模型导出 / 导入（.ojmodel 单文件）。
 *
 * WebLLM 把模型缓存在浏览器的 Cache Storage 中，键是精确的 URL：
 *   webllm/model   权重分片（tensor-cache.json 列出的 params_shard_*.bin）与 tokenizer
 *   webllm/config  mlc-chat-config.json
 *   webllm/wasm    模型运行库（model_lib）
 * 导出时把这些条目按原 URL 打包；导入时按同样的 URL 写回 Cache Storage，
 * WebLLM 就会直接读取缓存，不再访问 Hugging Face / GitHub，因此离线可用。
 *
 * 文件格式（小端序）：
 *   [0, 4)        魔数 "OJMD"
 *   [4, 8)        头部 JSON 的字节长度（uint32）
 *   [8, 8 + N)    头部 JSON（UTF-8）：模型 id、来源地址、条目列表（scope / url / offset / size）
 *   其后          各条目的原始字节，按 offset 与 size 定位
 *
 * 存储接口与 Cache Storage 相同（open / keys / match / put），便于在 Node 中用内存实现测试。
 * 注意：Cache Storage 按站点（协议 + 域名 + 端口）隔离，导入必须在同一个站点上进行。
 */

const MAGIC = 'OJMD';
const PREFIX_BYTES = 8;
const MAX_HEADER_BYTES = 16 * 1024 * 1024;

export const MODEL_FILE_EXT = '.ojmodel';
export const FORMAT = 'oj-ai-model';
export const FORMAT_VERSION = 1;
export const CACHE_SCOPES = ['webllm/model', 'webllm/config', 'webllm/wasm'];

/** 与 WebLLM 内部 cleanModelUrl 保持一致：补齐结尾斜杠，并补上 resolve/main/。 */
export function cleanModelUrl(modelUrl) {
  let url = String(modelUrl);
  url += url.endsWith('/') ? '' : '/';
  if (!/.+\/resolve\/.+\//.test(url)) url += 'resolve/main/';
  return new URL(url).href;
}

function fileUrl(base, name) {
  return new URL(name, base).href;
}

function safeFileName(id) {
  return String(id).replace(/[^A-Za-z0-9._-]+/g, '_');
}

async function keysOf(storage, scope) {
  const cache = await storage.open(scope);
  return (await cache.keys()).map((req) => req.url);
}

/**
 * 收集某个模型在缓存中的全部条目。关键文件缺失时抛出可读错误。
 * @returns {Promise<Array<{scope: string, url: string}>>}
 */
export async function collectModelEntries(record, storage = globalThis.caches) {
  const base = cleanModelUrl(record.model);
  const [modelKeys, configKeys, wasmKeys] = await Promise.all(CACHE_SCOPES.map((s) => keysOf(storage, s)));

  const modelUrls = modelKeys.filter((u) => u.startsWith(base));
  const configUrls = configKeys.filter((u) => u.startsWith(base));
  const wasmUrls = wasmKeys.filter((u) => u === record.model_lib);

  const tensorCacheUrl = fileUrl(base, 'tensor-cache.json');
  const missing = [];
  if (!modelUrls.includes(tensorCacheUrl)) missing.push('权重索引 tensor-cache.json');
  if (!modelUrls.includes(fileUrl(base, 'tokenizer.json')) && !modelUrls.includes(fileUrl(base, 'tokenizer.model'))) {
    missing.push('tokenizer');
  }
  if (!configUrls.includes(fileUrl(base, 'mlc-chat-config.json'))) missing.push('mlc-chat-config.json');
  if (wasmUrls.length === 0) missing.push('模型运行库（wasm）');
  if (missing.length) {
    throw new Error(`模型尚未完整缓存（缺少：${missing.join('、')}）。请先发送一条消息加载该模型，等加载完成后再导出。`);
  }

  const indexRes = await (await storage.open('webllm/model')).match(tensorCacheUrl);
  if (!indexRes) throw new Error('权重索引读取失败，请重新加载该模型后再试。');
  const index = await indexRes.json();
  const shardUrls = [...new Set((index.records || []).map((r) => fileUrl(base, r.dataPath)))];
  const shardMissing = shardUrls.filter((u) => !modelUrls.includes(u));
  if (shardMissing.length) {
    throw new Error(`权重文件不完整（缺少 ${shardMissing.length} 个分片），请重新加载该模型后再试。`);
  }

  return [
    ...modelUrls.map((url) => ({ scope: 'webllm/model', url })),
    ...configUrls.map((url) => ({ scope: 'webllm/config', url })),
    ...wasmUrls.map((url) => ({ scope: 'webllm/wasm', url })),
  ];
}

function encodePrefix(headerLength) {
  const buf = new ArrayBuffer(PREFIX_BYTES);
  const view = new DataView(buf);
  for (let i = 0; i < MAGIC.length; i++) view.setUint8(i, MAGIC.charCodeAt(i));
  view.setUint32(4, headerLength, true);
  return new Uint8Array(buf);
}

/**
 * 把缓存中的模型打包成 .ojmodel Blob。
 * @returns {Promise<{blob: Blob, filename: string, size: number, entryCount: number}>}
 */
export async function buildModelFile(record, { storage = globalThis.caches, onProgress } = {}) {
  const entries = await collectModelEntries(record, storage);
  const blobs = [];
  const meta = [];
  let offset = 0;
  for (let i = 0; i < entries.length; i++) {
    const { scope, url } = entries[i];
    const res = await (await storage.open(scope)).match(url);
    if (!res) throw new Error(`缓存条目丢失：${url}`);
    const blob = await res.blob();
    meta.push({ scope, url, size: blob.size, offset });
    blobs.push(blob);
    offset += blob.size;
    onProgress?.({ done: i + 1, total: entries.length });
  }

  const header = {
    format: FORMAT,
    version: FORMAT_VERSION,
    modelId: record.model_id,
    source: { model: record.model, model_lib: record.model_lib },
    createdAt: new Date().toISOString(),
    entries: meta,
  };
  const headerBytes = new TextEncoder().encode(JSON.stringify(header));
  const blob = new Blob([encodePrefix(headerBytes.byteLength), headerBytes, ...blobs], {
    type: 'application/octet-stream',
  });
  return {
    blob,
    filename: `${safeFileName(record.model_id)}${MODEL_FILE_EXT}`,
    size: blob.size,
    entryCount: meta.length,
  };
}

/** 校验头部结构，不合法时抛出错误。 */
export function validateHeader(h) {
  if (!h || typeof h !== 'object') throw new Error('模型文件头部无效');
  if (h.format !== FORMAT) throw new Error('文件格式不受支持，这不是本应用导出的模型');
  if (!Number.isInteger(h.version) || h.version > FORMAT_VERSION) {
    throw new Error(`模型文件版本 ${h.version} 高于当前支持的版本 ${FORMAT_VERSION}，请更新页面后再导入`);
  }
  if (typeof h.modelId !== 'string' || !h.modelId) throw new Error('模型文件缺少 modelId');
  if (!Array.isArray(h.entries) || h.entries.length === 0) throw new Error('模型文件没有任何数据条目');
  for (const e of h.entries) {
    if (!CACHE_SCOPES.includes(e.scope)) throw new Error(`未知的缓存类别：${e.scope}`);
    if (typeof e.url !== 'string' || !/^https?:\/\//.test(e.url)) throw new Error('模型文件中的地址无效');
    if (!Number.isSafeInteger(e.size) || e.size < 0 || !Number.isSafeInteger(e.offset) || e.offset < 0) {
      throw new Error('模型文件中的偏移或大小无效');
    }
  }
  return h;
}

/**
 * 读取并校验 .ojmodel 文件头部（只读取头部，不读取权重）。
 * @returns {Promise<{header: object, dataStart: number}>}
 */
export async function readModelFileHeader(file) {
  if (!file || file.size < PREFIX_BYTES) throw new Error('模型文件不完整');
  const prefix = new DataView(await file.slice(0, PREFIX_BYTES).arrayBuffer());
  let magic = '';
  for (let i = 0; i < MAGIC.length; i++) magic += String.fromCharCode(prefix.getUint8(i));
  if (magic !== MAGIC) throw new Error('这不是本应用导出的模型文件');

  const headerLength = prefix.getUint32(4, true);
  if (headerLength === 0 || headerLength > MAX_HEADER_BYTES || PREFIX_BYTES + headerLength > file.size) {
    throw new Error('模型文件头部损坏或不完整');
  }
  let header;
  try {
    header = JSON.parse(await file.slice(PREFIX_BYTES, PREFIX_BYTES + headerLength).text());
  } catch {
    throw new Error('模型文件头部无法解析');
  }
  validateHeader(header);

  const dataStart = PREFIX_BYTES + headerLength;
  const dataSize = file.size - dataStart;
  for (const e of header.entries) {
    if (e.offset + e.size > dataSize) throw new Error(`模型文件数据不完整：${e.url}`);
  }
  return { header, dataStart };
}

/**
 * 把 .ojmodel 文件写回缓存，使 WebLLM 可以离线读取该模型。
 * @param {File} file 用户选择的文件
 * @param {object} record 当前配置中与文件 modelId 对应的模型记录（包含 model 与 model_lib）
 */
export async function importModelFile(file, record, { storage = globalThis.caches, onProgress } = {}) {
  const { header, dataStart } = await readModelFileHeader(file);
  if (header.modelId !== record.model_id) {
    throw new Error(`文件属于模型 ${header.modelId}，与所选模型 ${record.model_id} 不一致`);
  }
  // 缓存键必须与当前配置的下载地址一致，否则 WebLLM 找不到这些条目
  const base = cleanModelUrl(record.model);
  for (const e of header.entries) {
    const ok = e.scope === 'webllm/wasm' ? e.url === record.model_lib : e.url.startsWith(base);
    if (!ok) {
      throw new Error('模型文件的下载地址与当前配置不一致（可能是镜像设置不同），无法导入。请检查 config.json 中的 mirrorOwner。');
    }
  }

  let done = 0;
  for (const e of header.entries) {
    const cache = await storage.open(e.scope);
    const body = file.slice(dataStart + e.offset, dataStart + e.offset + e.size);
    await cache.put(e.url, new Response(body, { status: 200, headers: { 'Content-Type': 'application/octet-stream' } }));
    done += 1;
    onProgress?.({ done, total: header.entries.length });
  }

  // 导入后再校验一次，确保模型已完整可用
  await collectModelEntries(record, storage);
  return { modelId: header.modelId, entryCount: header.entries.length };
}

/**
 * 基本检查：每个模型的关键文件是否都在缓存中（不逐个校验权重分片）。
 * @param {Array<object>} records 模型记录列表
 * @returns {Promise<Set<string>>} 已缓存的 model_id 集合
 */
export async function listCachedModelIds(records, storage = globalThis.caches) {
  const sets = {};
  for (const scope of CACHE_SCOPES) sets[scope] = new Set(await keysOf(storage, scope));
  const cached = new Set();
  for (const r of records) {
    const base = cleanModelUrl(r.model);
    const hasTokenizer =
      sets['webllm/model'].has(fileUrl(base, 'tokenizer.json')) || sets['webllm/model'].has(fileUrl(base, 'tokenizer.model'));
    if (
      sets['webllm/model'].has(fileUrl(base, 'tensor-cache.json')) &&
      hasTokenizer &&
      sets['webllm/config'].has(fileUrl(base, 'mlc-chat-config.json')) &&
      sets['webllm/wasm'].has(r.model_lib)
    ) {
      cached.add(r.model_id);
    }
  }
  return cached;
}
