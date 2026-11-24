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
 * 解析 Link 响应头里的 rel="next" 地址（Hub 树接口按游标分页）。
 * 例如 `<https://huggingface.co/api/models/x/y/tree/main?recursive=true&cursor=abc>; rel="next"`。
 * 找不到返回 ''。
 */
export function parseLinkNext(linkHeader) {
  const header = String(linkHeader || '');
  for (const part of header.split(',')) {
    const m = part.match(/<([^>]+)>\s*;\s*rel\s*=\s*"?([^";]+)"?/i);
    if (m && String(m[2]).trim().toLowerCase() === 'next') return m[1];
  }
  return '';
}

/** 归一化一条树条目（只留判定/分组要用的字段）；不是文件返回 null。 */
function normalizeTreeEntry(e) {
  if (!e || e.type !== 'file' || typeof e.path !== 'string') return null;
  return {
    path: e.path,
    size: Math.max(0, Number(e.lfs?.size ?? e.size) || 0),
    sha256: typeof e.lfs?.oid === 'string' ? e.lfs.oid : '',
  };
}

/**
 * 列出仓库全部文件（递归），自动跟随 Link rel="next" 分页。
 *
 * Hub 的树接口一次最多返回 1000 条，并在 Link 头里给下一页游标。以前这里只读
 * 第一页：超大模型的仓库（几十个量化 × 多个分片，很容易过千条）会被静默截断，
 * 结果就是“量化列表少了几档 / 分片只下了一部分”，加载时才报文件缺失。
 * 现在一页一页跟到底，并做三项保护：最多 100 页、游标必须前进、结果按路径去重。
 *
 * @returns {Promise<Array<{path: string, size: number, sha256: string}>>}
 */
export async function fetchRepoFiles(repo, { fetchImpl = fetch, timeoutMs = HF_TIMEOUT_MS, headers = {}, maxPages = 100 } = {}) {
  const first = `${HF_BASE}/api/models/${repo}/tree/main?recursive=true`;
  const out = [];
  const seen = new Set();
  const visited = new Set();
  let url = first;
  for (let page = 0; url && page < maxPages; page += 1) {
    if (visited.has(url)) break; // 游标不前进（镜像/代理异常）：停止，不要死循环
    visited.add(url);
    let res;
    try {
      res = await fetchImpl(url, { headers: { accept: 'application/json', ...headers }, signal: timeoutSignal(timeoutMs) });
    } catch (err) {
      throw hfNetworkError(err, timeoutMs);
    }
    if (!res.ok) {
      throw new Error(`读取模型仓库 ${repo} 失败（HTTP ${res.status}），请确认仓库存在且网络可达 Hugging Face。`);
    }
    const tree = await res.json();
    if (!Array.isArray(tree)) throw new Error(`模型仓库 ${repo} 返回了无法识别的文件列表。`);
    for (const raw of tree) {
      const f = normalizeTreeEntry(raw);
      if (!f || seen.has(f.path)) continue;
      seen.add(f.path);
      out.push(f);
    }
    const next = parseLinkNext(typeof res.headers?.get === 'function' ? res.headers.get('link') : '');
    url = next && !visited.has(next) ? next : '';
  }
  return out;
}

/**
 * 搜索 Hugging Face 模型（服务端桥接 Hub API，小白不用自己拼地址）。
 * 返回归一化结果：格式标签先按 tags/仓库名粗判，点开详情时再按真实文件确认。
 */
export async function searchModels({ query = '', limit = 24, sort = 'downloads', fetchImpl = fetch, headers = {} } = {}) {
  const params = new URLSearchParams({
    limit: String(Math.min(50, Math.max(1, Math.floor(limit) || 24))),
    sort: ['downloads', 'likes', 'lastModified'].includes(sort) ? sort : 'downloads',
    direction: '-1',
  });
  const q = String(query || '').trim();
  if (q) params.set('search', q);
  let res;
  try {
    res = await fetchImpl(`${HF_BASE}/api/models?${params}`, { headers: { accept: 'application/json', ...headers }, signal: timeoutSignal(HF_TIMEOUT_MS) });
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
export async function fetchModelInfo(repo, { fetchImpl = fetch, timeoutMs = HF_TIMEOUT_MS, headers = {} } = {}) {
  let res;
  try {
    res = await fetchImpl(`${HF_BASE}/api/models/${repo}`, { headers: { accept: 'application/json', ...headers }, signal: timeoutSignal(timeoutMs) });
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
  const files = await fetchRepoFiles(repo, { fetchImpl, headers });
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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * 下载单个文件到 dest（先写 .part，支持 Range 断点续传，完成后改名）。
 *
 * 超大模型动辄几十个分片、单文件几十 GB，一次网络抖动就整档下载失败（以前直接
 * 抛给用户，要手动重按）。现在单个文件内部自带重试：失败后保留 .part、按已落盘
 * 偏移续传，默认最多 3 次；只有 404/403/磁盘不足这类重试也没用的错误才立即失败。
 *
 * 完整性以 expectedSize（Hub 给的 lfs.size）为准：以前只在响应带 content-length
 * 时校验，而 CDN 走分块传输时可能没有该头，截断的文件会被当成“完成”改名转正，
 * 加载时才报 GGUF 损坏。
 *
 * @returns {Promise<{bytes: number, resumed: boolean}>}
 */
export async function downloadFile({
  url, dest, fetchImpl = fetch, signal, onProgress,
  headers: extraHeaders = {}, expectedSize = 0, retries = 3, retryDelayMs = 800,
} = {}) {
  await mkdir(path.dirname(dest), { recursive: true });
  // 已转正的完整文件不要再下一遍：分片权重失败重试时，逐片检查能省下大量重复流量
  // （以前只认 .part，下完转正的分片在重试时会被从头再拉一次，用户看着就是“又从 0 开始”）。
  if (expectedSize > 0) {
    try {
      const st = await stat(dest);
      if (st.size === expectedSize) return { bytes: st.size, resumed: true, skipped: true };
      // 大小对不上的正式文件是坏文件（旧版本可能把截断文件转正过）：删掉重下
      await unlink(dest).catch(() => {});
    } catch { /* 正式文件不在，走下面的 .part 续传 */ }
  }
  const part = `${dest}.part`;
  let lastErr = null;
  const attempts = Math.max(1, Math.floor(retries) || 1);
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (signal?.aborted) {
      const err = new Error('下载已取消');
      err.name = 'AbortError';
      throw err;
    }
    let offset = 0;
    try {
      offset = (await stat(part)).size;
    } catch {
      offset = 0;
    }
    // .part 比目标还大（上个版本写坏/目标文件被替换）：丢掉重来
    if (expectedSize > 0 && offset > expectedSize) {
      await unlink(part).catch(() => {});
      offset = 0;
    }
    if (expectedSize > 0 && offset === expectedSize) {
      await rename(part, dest);
      return { bytes: offset, resumed: true };
    }
    const headers = { ...extraHeaders };
    if (offset > 0) headers.range = `bytes=${offset}-`;
    let res;
    try {
      res = await fetchImpl(url, { headers, signal, redirect: 'follow' });
    } catch (err) {
      if (err?.name === 'AbortError' || signal?.aborted) throw err;
      lastErr = hfNetworkError(err, HF_TIMEOUT_MS);
      if (attempt < attempts - 1) {
        await sleep(retryDelayMs * (attempt + 1));
        continue;
      }
      throw lastErr;
    }
    if (res.status === 416 && offset > 0) {
      // 服务器说范围无效：.part 恰好等于目标大小才算完整，否则是坏片段，丢掉重下
      const size = (await stat(part).catch(() => ({ size: -1 }))).size;
      if (!expectedSize || size === expectedSize) {
        await rename(part, dest);
        return { bytes: size, resumed: true };
      }
      await unlink(part).catch(() => {});
      lastErr = new Error('已下载的片段大小与原文件不符，已丢弃重下。');
      continue;
    }
    if (!res.ok && res.status !== 206) {
      const err = new Error(res.status === 401 || res.status === 403
        ? `下载失败（HTTP ${res.status}）：该仓库需要 Hugging Face Token 或访问授权，请在面板里保存 Token 后重试。`
        : `下载失败（HTTP ${res.status}）：${url}`);
      if (res.status === 408 || res.status === 429 || res.status >= 500) {
        lastErr = err;
        if (attempt < attempts - 1) {
          await sleep(retryDelayMs * (attempt + 1) * 2);
          continue;
        }
      }
      throw err;
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
      onProgress?.({ downloaded: done, total: total || expectedSize });
    });
    try {
      await pipeline(body, out, { signal });
    } catch (err) {
      if (err?.name === 'AbortError' || signal?.aborted) throw err;
      // 传到一半断线：.part 已落盘的部分保留，下次重试按偏移续传，不用从头来
      lastErr = new Error(`下载中断（${err?.message || '网络错误'}），已下载的部分已保留，重新下载会从断点续传。`);
      if (attempt < attempts - 1) {
        await sleep(retryDelayMs * (attempt + 1));
        continue;
      }
      throw lastErr;
    }
    const finalSize = (await stat(part)).size;
    // 以 Hub 声明的文件大小为准（没有 expectedSize 时退回 content-length）
    const want = expectedSize > 0 ? expectedSize : total;
    if (want && finalSize !== want) {
      lastErr = new Error(`文件下载不完整：期望 ${want} 字节，实际 ${finalSize} 字节。可重试续传。`);
      if (finalSize > want) await unlink(part).catch(() => {});
      if (attempt < attempts - 1) {
        await sleep(retryDelayMs * (attempt + 1));
        continue;
      }
      throw lastErr;
    }
    await rename(part, dest);
    return { bytes: finalSize, resumed };
  }
  throw lastErr || new Error(`下载失败：${url}`);
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
