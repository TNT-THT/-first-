/**
 * Hugging Face 权重仓库访问：列文件、拼下载地址、断点续传下载。
 * 只在服务端运行（管理员触发），浏览器不直连这些大文件。
 */
import { createWriteStream } from 'node:fs';
import { mkdir, rename, stat, unlink } from 'node:fs/promises';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';

export const HF_BASE = 'https://huggingface.co';

/** HF 接口默认超时：慢/挂死的网络必须在有限时间内给出明确说法，不能让前端无限干等。 */
export const HF_TIMEOUT_MS = 30000;

function timeoutSignal(timeoutMs) {
  const ms = Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : HF_TIMEOUT_MS;
  return typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(ms) : undefined;
}

function isTimeoutError(err) {
  return err?.name === 'TimeoutError' || err?.name === 'AbortError';
}

/** 把网络层异常翻译成用户看得懂的中文（裸 fetch 只会抛英文 fetch failed）。 */
function hfNetworkError(err, timeoutMs) {
  if (isTimeoutError(err)) {
    return new Error(`连接 Hugging Face 超时（${Math.round((timeoutMs || HF_TIMEOUT_MS) / 1000)} 秒），网络可能较慢或不可达，请检查网络后重试。`);
  }
  return new Error(`无法连接 Hugging Face（${err?.message || '网络错误'}），请检查网络后重试。`);
}

/**
 * 列出仓库全部文件（递归）。
 * @returns {Promise<Array<{path: string, size: number, sha256: string}>>}
 */
export async function fetchRepoFiles(repo, { fetchImpl = fetch, timeoutMs = HF_TIMEOUT_MS } = {}) {
  const url = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
  let res;
  try {
    res = await fetchImpl(url, { headers: { accept: 'application/json' }, signal: timeoutSignal(timeoutMs) });
  } catch (err) {
    throw hfNetworkError(err, timeoutMs);
  }
  if (!res.ok) {
    throw new Error(`读取模型仓库 ${repo} 失败（HTTP ${res.status}），请确认仓库存在且网络可达 Hugging Face。`);
  }
  const tree = await res.json();
  if (!Array.isArray(tree)) throw new Error(`模型仓库 ${repo} 返回了无法识别的文件列表。`);
  return tree
    .filter((e) => e && e.type === 'file' && typeof e.path === 'string')
    .map((e) => ({
      path: e.path,
      size: Math.max(0, Number(e.lfs?.size ?? e.size) || 0),
      sha256: typeof e.lfs?.oid === 'string' ? e.lfs.oid : '',
    }));
}

/**
 * 搜索 Hugging Face 模型（服务端桥接 Hub API，小白不用自己拼地址）。
 * 返回归一化结果：格式标签先按 tags/仓库名粗判，点开详情时再按真实文件确认。
 */
export async function searchModels({ query = '', limit = 24, sort = 'downloads', fetchImpl = fetch } = {}) {
  const params = new URLSearchParams({
    limit: String(Math.min(50, Math.max(1, Math.floor(limit) || 24))),
    sort: ['downloads', 'likes', 'lastModified'].includes(sort) ? sort : 'downloads',
    direction: '-1',
  });
  const q = String(query || '').trim();
  if (q) params.set('search', q);
  let res;
  try {
    res = await fetchImpl(`${HF_BASE}/api/models?${params}`, { headers: { accept: 'application/json' }, signal: timeoutSignal(HF_TIMEOUT_MS) });
  } catch (err) {
    throw hfNetworkError(err, HF_TIMEOUT_MS);
  }
  if (!res.ok) throw new Error(`Hugging Face 搜索失败（HTTP ${res.status}），请检查网络后重试。`);
  const list = await res.json();
  if (!Array.isArray(list)) throw new Error('Hugging Face 搜索返回了无法识别的结果。');
  return list
    .filter((m) => m && typeof m.id === 'string' && !m.private)
    .map((m) => ({
      repo: m.id,
      name: m.id.split('/').pop() || m.id,
      author: m.author || m.id.split('/')[0] || '',
      downloads: Number(m.downloads) || 0,
      likes: Number(m.likes) || 0,
      pipeline: m.pipeline_tag || '',
      library: m.library_name || '',
      tags: Array.isArray(m.tags) ? m.tags.slice(0, 30) : [],
      gated: !!m.gated,
      updatedAt: m.lastModified || '',
    }));
}

/** 仓库详情：真实文件树 + 格式判定（搜索结果点开/一键使用时调用）。 */
export async function fetchModelInfo(repo, { fetchImpl = fetch, timeoutMs = HF_TIMEOUT_MS } = {}) {
  let res;
  try {
    res = await fetchImpl(`${HF_BASE}/api/models/${repo}`, { headers: { accept: 'application/json' }, signal: timeoutSignal(timeoutMs) });
  } catch (err) {
    throw hfNetworkError(err, timeoutMs);
  }
  if (!res.ok) {
    throw Object.assign(
      new Error(res.status === 404 ? `仓库不存在：${repo}` : `读取仓库 ${repo} 失败（HTTP ${res.status}）。`),
      { status: res.status === 404 ? 404 : 502 },
    );
  }
  const meta = await res.json();
  const files = await fetchRepoFiles(repo, { fetchImpl });
  return {
    repo,
    name: repo.split('/').pop() || repo,
    author: meta?.author || repo.split('/')[0] || '',
    downloads: Number(meta?.downloads) || 0,
    likes: Number(meta?.likes) || 0,
    pipeline: meta?.pipeline_tag || '',
    library: meta?.library_name || '',
    tags: Array.isArray(meta?.tags) ? meta.tags : [],
    gated: !!meta?.gated,
    files,
    totalBytes: files.reduce((n, f) => n + (f.size || 0), 0),
  };
}

/** 仓库内单个文件的下载地址（自动跟随重定向到 CDN）。 */
export function resolveFileUrl(repo, filePath) {
  const encoded = String(filePath).split('/').map(encodeURIComponent).join('/');
  return `${HF_BASE}/${repo}/resolve/main/${encoded}`;
}

/**
 * 下载单个文件到 dest（先写 .part，支持 Range 断点续传，完成后改名）。
 * @returns {Promise<{bytes: number, resumed: boolean}>}
 */
export async function downloadFile({ url, dest, fetchImpl = fetch, signal, onProgress, headers: extraHeaders = {} } = {}) {
  await mkdir(path.dirname(dest), { recursive: true });
  const part = `${dest}.part`;
  let offset = 0;
  try {
    offset = (await stat(part)).size;
  } catch {
    offset = 0;
  }
  const headers = { ...extraHeaders };
  if (offset > 0) headers.range = `bytes=${offset}-`;
  const res = await fetchImpl(url, { headers, signal, redirect: 'follow' });
  if (res.status === 416 && offset > 0) {
    // 服务器说范围无效：多半是 .part 已完整，直接收尾
    await rename(part, dest);
    return { bytes: offset, resumed: true };
  }
  if (!res.ok && res.status !== 206) {
    throw new Error(`下载失败（HTTP ${res.status}）：${url}`);
  }
  const resumed = res.status === 206 && offset > 0;
  if (!resumed) offset = 0;
  const totalHeader = Number(res.headers?.get?.('content-length') || 0);
  const total = totalHeader ? offset + totalHeader : 0;
  let done = offset;
  const out = createWriteStream(part, { flags: resumed ? 'a' : 'w' });
  const body = Readable.fromWeb(res.body);
  body.on('data', (chunk) => {
    done += chunk.length;
    onProgress?.({ downloaded: done, total });
  });
  await pipeline(body, out, { signal });
  const finalSize = (await stat(part)).size;
  if (total && finalSize !== total) {
    throw new Error(`文件下载不完整：期望 ${total} 字节，实际 ${finalSize} 字节。可重试续传。`);
  }
  await rename(part, dest);
  return { bytes: finalSize, resumed };
}

/** 删除残留的 .part（下载被明确取消/删除时调用）。 */
export async function removePart(dest) {
  await unlink(`${dest}.part`).catch(() => {});
}

/** 本地可用磁盘空间（字节）；statfs 不可用时返回 Infinity（不拦截）。 */
export async function freeDiskBytes(dir) {
  try {
    const { statfs } = await import('node:fs/promises');
    const s = await statfs(dir);
    return Number(s.bavail) * Number(s.bsize);
  } catch {
    return Infinity;
  }
}
