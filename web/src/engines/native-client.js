/**
 * 本机大模型（llama.cpp 原生引擎）的网页端接口。
 *
 * 只有带服务端的部署（网站包、独立软件）才有 /api/native/*；纯静态网页版
 * 请求会 404，这里统一吞掉错误返回 null，界面据此隐藏大模型功能。
 */

/** 读取引擎状态与模型目录；不可用时返回 null。 */
export async function fetchNativeStatus() {
  try {
    const res = await fetch('/api/native/status', { cache: 'no-store' });
    if (!res.ok) return null;
    const data = await res.json();
    return data?.enabled ? data : null;
  } catch {
    return null;
  }
}

/** 管理员接口（下载/加载/卸载等）：带管理员令牌，失败抛 Error(含服务端消息)。 */
export async function nativeAdminRequest(path, { token, method = 'POST', body } = {}) {
  const headers = { 'content-type': 'application/json' };
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(path, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `请求失败（HTTP ${res.status}）`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/** HF 搜索（需网站登录令牌）：关键词或仓库链接都可以。 */
export async function hfSearch(query, token) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(`/api/native/hf/search?q=${encodeURIComponent(query || '')}`, { headers, cache: 'no-store' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `搜索失败（HTTP ${res.status}）`);
  return data;
}

/** 把用户输入（关键词、owner/repo、完整 HF 链接）解析成仓库 id；不是仓库返回 null。 */
export function parseRepoInput(text) {
  const raw = String(text || '').trim();
  if (!raw) return null;
  const urlMatch = raw.match(/huggingface\.co\/([A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*)/);
  if (urlMatch) return urlMatch[1];
  if (/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(raw)) return raw;
  return null;
}

/** 浏览器直连 HF 仓库元信息（与服务端 fetchModelInfo 取的是同一个接口）。 */
export async function fetchRepoMetaViaBrowser(repo, { fetchImpl = defaultFetch, timeoutMs = 30000 } = {}) {
  const signal = typeof AbortSignal?.timeout === 'function' && timeoutMs > 0
    ? AbortSignal.timeout(timeoutMs)
    : undefined;
  let res;
  try {
    res = await fetchImpl(`${HF_WEB_BASE}/api/models/${repo}`, {
      headers: { accept: 'application/json' },
      cache: 'no-store',
      signal,
    });
  } catch (err) {
    if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
      throw new Error('浏览器连接 Hugging Face 超时，请检查网络/代理后重试。');
    }
    throw new Error(`浏览器无法连接 Hugging Face（${err?.message || '网络错误'}）。`);
  }
  if (!res.ok) {
    throw new Error(res.status === 404 ? `仓库不存在：${repo}` : `读取仓库 ${repo} 失败（HTTP ${res.status}）。`);
  }
  const meta = await res.json().catch(() => null);
  if (!meta || typeof meta !== 'object') throw new Error(`仓库 ${repo} 返回了无法识别的信息。`);
  return meta;
}

/** 浏览器取数 + 本地判定：HF 搜索（关键词）和粘贴仓库链接两种输入都走这里。 */
export async function searchHfViaBrowser(query, token, { fetchImpl = defaultFetch } = {}) {
  const q = String(query || '').trim();
  const direct = parseRepoInput(q);
  if (direct) {
    // 粘贴的是仓库：直接取详情，结果列表里给一条，与原服务端行为一致
    const info = await fetchModelInfoViaBrowser(direct, token, { fetchImpl });
    return { direct, results: [{ repo: direct, name: info.name, author: info.repo.split('/')[0] || '', downloads: info.downloads, likes: info.likes, formats: info.formats }] };
  }
  const params = new URLSearchParams({ limit: '24', sort: 'downloads', direction: '-1' });
  if (q) params.set('search', q);
  let res;
  try {
    res = await fetchImpl(`${HF_WEB_BASE}/api/models?${params}`, { headers: { accept: 'application/json' }, cache: 'no-store' });
  } catch (err) {
    throw new Error(`浏览器无法连接 Hugging Face（${err?.message || '网络错误'}）。`);
  }
  if (!res.ok) throw new Error(`Hugging Face 搜索失败（HTTP ${res.status}），请检查网络后重试。`);
  const list = await res.json().catch(() => null);
  if (!Array.isArray(list)) throw new Error('Hugging Face 搜索返回了无法识别的结果。');
  const results = list
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
  // 格式粗判定在本地服务做，与服务端直连搜索同一套口径
  const headers = { 'content-type': 'application/json' };
  if (token) headers.authorization = `Bearer ${token}`;
  const judge = await fetchImpl('/api/native/hf/search-meta', {
    method: 'POST', headers, body: JSON.stringify({ results }), cache: 'no-store',
  });
  const data = await judge.json().catch(() => ({}));
  if (!judge.ok) throw new Error(data.error || `本地判定失败（HTTP ${judge.status}）`);
  return { results: data.results || [] };
}

/** 浏览器取数 + 本地判定：HF 仓库详情（格式路线 + 量化清单）。 */
export async function fetchModelInfoViaBrowser(repo, token, { fetchImpl = defaultFetch } = {}) {
  const [meta, files] = await Promise.all([
    fetchRepoMetaViaBrowser(repo, { fetchImpl }),
    fetchRepoTreeViaBrowser(repo, { fetchImpl }),
  ]);
  const headers = { 'content-type': 'application/json' };
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetchImpl('/api/native/hf/model-meta', {
    method: 'POST',
    headers,
    body: JSON.stringify({ repo, meta, files: slimFiles(files) }),
    cache: 'no-store',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `本地判定失败（HTTP ${res.status}）`);
  return data;
}

/**
 * 使用 HF 仓库（浏览器通道优先）：浏览器取详情 → 本地登记与路线判定 →
 * GGUF 本地路线直接走浏览器下载；其余路线（转换/Python/云端）沿用服务端流程。
 * 浏览器取数失败抛错，调用方回退服务端直连的 /use。
 * @returns {Promise<{out: object, downloadState: 'browser'|'fallback'|'none'}>}
 */
export async function useRepoViaBrowser({ repo, mode = 'auto', quant = '', adminToken, fetchImpl = defaultFetch, onStatus } = {}) {
  const note = (text) => { try { onStatus?.(text); } catch { /* 忽略回调错误 */ } };
  note('正在经浏览器读取仓库信息（和网页版同一条网络路）…');
  const [meta, files] = await Promise.all([
    fetchRepoMetaViaBrowser(repo, { fetchImpl }),
    fetchRepoTreeViaBrowser(repo, { fetchImpl }),
  ]);
  const out = await adminJson(fetchImpl, '/api/native/hf/use', {
    token: adminToken,
    body: { repo, mode, quant: quant || undefined, viaBrowser: true, info: { meta, files: slimFiles(files) } },
  });
  // 多档量化：服务端把清单交回来让用户选，这里不下载任何字节
  if (out.action === 'choose-quant') return { out, downloadState: 'none' };
  if (out.action !== 'download' || out.via !== 'browser') return { out, downloadState: 'none' };
  // 如果下载已在服务端完成（已下载过），downloadModelViaBrowser 会在 prepare 秒回 done
  try {
    await downloadModelViaBrowser({ id: out.id, repo, quant: out.quant || undefined, adminToken, fetchImpl, onStatus: note });
    return { out, downloadState: 'browser' };
  } catch (browserErr) {
    // 浏览器传字节失败：回退服务端直连（量化分组缓存已在 /use 时写入，选档不必再列树）。
    // 失败原因必须带出来——以前这里默默吞掉，用户最后只看到直连的一句 fetch failed，
    // 根本不知道浏览器通道到底死在哪一步。
    //
    // 但“文件本身拿不到”（401/403/404）时直连同样注定失败：门控仓库要 Token、
    // 仓库改名/删档要重新搜索。这时不再制造第二个更含糊的报错，直接把原因抛上去。
    const fatal = /HTTP (401|403|404)|Token|仓库不存在|找不到/.test(String(browserErr?.message || ''));
    if (fatal) throw browserErr;
    await adminJson(fetchImpl, `/api/native/models/${encodeURIComponent(out.id)}/download`, {
      token: adminToken,
      body: { quant: out.quant || undefined },
    });
    return { out, downloadState: 'fallback', downloadError: browserErr?.message || '未知原因' };
  }
}

/**
 * 本地导入：把浏览器已下载到电脑的文件传进软件（全程只走本机，碰不到外网，
 * 是三条下载路线里最稳的一条）。同一组文件会派生同一个导入编号，传一半
 * 关掉页面后重新选同一组文件能接着传，不用从头来。
 *
 * @param {{entries: Array<{file: File, path: string}>, name?: string,
 *          adminToken: string, onProgress?: (sent: number, total: number) => void,
 *          fetchImpl?: Function, sliceBytes?: number}} opts
 * @returns {Promise<{id: string, name: string, formats: string[], quants: string[], snapshot: boolean}>}
 */
export async function importLocalModel({ entries, name, adminToken, onProgress, fetchImpl = defaultFetch, sliceBytes = 8 * 1024 * 1024 } = {}) {
  const list = (Array.isArray(entries) ? entries : []).filter((e) => e?.file && typeof e.path === 'string');
  if (!list.length) throw new Error('没有选择要导入的文件。');
  const totalBytes = list.reduce((n, e) => n + (e.file.size || 0), 0);
  // 导入编号由文件清单派生：同一组文件永远同一个编号，天然断点续传
  let hash = 0x811c9dc5;
  const fingerprint = list.map((e) => `${e.path}:${e.file.size}`).join('|');
  for (let i = 0; i < fingerprint.length; i += 1) {
    hash ^= fingerprint.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  const importId = `imp-${hash.toString(16).padStart(8, '0')}`;
  const begin = await adminJson(fetchImpl, '/api/native/import/begin', {
    token: adminToken,
    body: { importId, files: list.map((e) => ({ path: e.path, size: e.file.size || 0 })) },
  });
  const receivedByPath = new Map((begin.files || []).map((f) => [f.path, Math.max(0, Number(f.received) || 0)]));
  let sentTotal = [...receivedByPath.values()].reduce((n, v) => n + v, 0);
  const report = () => { try { onProgress?.(sentTotal, totalBytes); } catch { /* 忽略回调错误 */ } };
  report();
  for (const entry of list) {
    let offset = receivedByPath.get(entry.path) || 0;
    while (offset < entry.file.size) {
      const slice = entry.file.slice(offset, Math.min(entry.file.size, offset + sliceBytes));
      const data = await slice.arrayBuffer();
      const res = await fetchImpl(`/api/native/import/${encodeURIComponent(importId)}/file`, {
        method: 'POST',
        headers: {
          authorization: `Bearer ${adminToken}`,
          'content-type': 'application/octet-stream',
          'x-import-path': encodeURIComponent(entry.path),
          'x-import-offset': String(offset),
        },
        body: data,
        cache: 'no-store',
      });
      const out = await res.json().catch(() => ({}));
      if (!res.ok) {
        if (Number.isFinite(out.received)) {
          // 与软件已落盘位置对齐后接着传（错位不会写坏文件）
          sentTotal += Math.max(0, out.received - offset);
          offset = out.received;
          report();
          continue;
        }
        throw new Error(out.error || `导入 ${entry.path} 失败（HTTP ${res.status}）。`);
      }
      sentTotal += data.byteLength;
      offset = out.received;
      report();
    }
  }
  return adminJson(fetchImpl, `/api/native/import/${encodeURIComponent(importId)}/finish`, {
    token: adminToken,
    body: { name: name || undefined },
  });
}

/** HF 仓库详情（格式与运行路线）。 */
export async function hfModelInfo(repo, token) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetch(`/api/native/hf/model?repo=${encodeURIComponent(repo)}`, { headers, cache: 'no-store' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `读取仓库失败（HTTP ${res.status}）`);
  return data;
}

/** 某模型在仓库里可下载的量化列表（需网站登录令牌由 api.js 的 getToken 提供）。 */
export async function fetchNativeQuants(id, token, { timeoutMs = 40000 } = {}) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  const signal = typeof AbortSignal?.timeout === 'function' && timeoutMs > 0
    ? AbortSignal.timeout(timeoutMs)
    : undefined;
  let res;
  try {
    res = await fetch(`/api/native/models/${encodeURIComponent(id)}/quants`, { headers, cache: 'no-store', signal });
  } catch (err) {
    if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
      throw new Error('读取量化列表超时：连接模型仓库太慢或不可达，请检查网络后重试。');
    }
    throw new Error(`读取量化列表失败（${err?.message || '网络错误'}），请检查网络后重试。`);
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
  return data.quants || [];
}

// ---------- 浏览器通道：和普通网页版完全同一条网络路 ----------
// 软件里的 Node 服务端是裸直连 Hugging Face，不走浏览器/系统代理，很多用户
// 电脑上连不通；浏览器却能下网页版模型。所以取数改由浏览器做（fetch HF），
// 本地服务只负责分组计算与落盘。浏览器通道失败时调用方回退服务端直连。

const HF_WEB_BASE = 'https://huggingface.co';

/**
 * 浏览器原生 fetch 有个著名的大坑：它必须以 window 的身份被调用。一旦被
 * “拎出来”当普通函数经对象转手调用（比如 ctx.fetchImpl(...)），Chrome 就
 * 当场抛 “Failed to execute 'fetch' on 'Window': Illegal invocation”。
 * 浏览器通道真实浏览器里每次拉权重文件都死在这一步，而 Node 的 fetch
 * 没有这道身份检查、自动化测试一直测不出来。统一用箭头转发：内部做裸
 * 调用 fetch(...)，身份永远是 window，怎么转手都不怕。
 */
const defaultFetch = (...args) => fetch(...args);

/**
 * 解析 Link 响应头里的 rel="next"（Hub 树接口按游标分页，每页最多 1000 条）。
 * 与服务端 parseLinkNext 同一套口径。找不到返回 ''。
 */
export function parseLinkNext(linkHeader) {
  for (const part of String(linkHeader || '').split(',')) {
    const m = part.match(/<([^>]+)>\s*;\s*rel\s*=\s*"?([^";]+)"?/i);
    if (m && String(m[2]).trim().toLowerCase() === 'next') return m[1];
  }
  return '';
}

/**
 * 浏览器直连 HF 文件树，归一化为与服务端 fetchRepoFiles 一致的清单。
 *
 * 必须跟随 Link rel="next"：Hub 一次最多给 1000 条，超大模型仓库（几十个量化
 * × 多个分片）轻松过千。以前只取第一页，于是量化列表少档、分片只下一部分，
 * 加载时才报缺文件——浏览器通道是大多数用户的实际下载路径，这个坑影响最大。
 */
export async function fetchRepoTreeViaBrowser(repo, { fetchImpl = defaultFetch, timeoutMs = 60000, maxPages = 100 } = {}) {
  let url = `${HF_WEB_BASE}/api/models/${repo}/tree/main?recursive=true`;
  const out = [];
  const seen = new Set();
  const visited = new Set();
  for (let page = 0; url && page < maxPages; page += 1) {
    if (visited.has(url)) break;
    visited.add(url);
    let res = null;
    let lastErr = null;
    // 大仓库的文件树不小、慢代理下 30 秒经常不够：放宽到 60 秒，网络失败自动重试一次
    for (let attempt = 0; attempt < 2 && !res; attempt += 1) {
      const signal = typeof AbortSignal?.timeout === 'function' && timeoutMs > 0
        ? AbortSignal.timeout(timeoutMs)
        : undefined;
      try {
        res = await fetchImpl(url, {
          headers: { accept: 'application/json' },
          cache: 'no-store',
          signal,
        });
      } catch (err) {
        lastErr = err;
        if (attempt === 0) await new Promise((r) => setTimeout(r, 1200));
      }
    }
    if (!res) {
      const err = lastErr;
      if (err?.name === 'TimeoutError' || err?.name === 'AbortError') {
        throw new Error('浏览器连接 Hugging Face 超时（仓库较大或网络较慢），请检查网络/代理后重试。');
      }
      throw new Error(`浏览器无法连接 Hugging Face（${err?.message || '网络错误'}）。`);
    }
    if (res.status === 401 || res.status === 403) {
      throw new Error(`读取模型仓库 ${repo} 需要授权（HTTP ${res.status}）：该仓库是门控/私有仓库，请先在软件里保存 Hugging Face Token 再试。`);
    }
    if (!res.ok) throw new Error(`读取模型仓库 ${repo} 失败（HTTP ${res.status}）。`);
    const tree = await res.json().catch(() => null);
    if (!Array.isArray(tree)) throw new Error(`模型仓库 ${repo} 返回了无法识别的文件列表。`);
    for (const e of tree) {
      if (!e || e.type !== 'file' || typeof e.path !== 'string' || seen.has(e.path)) continue;
      seen.add(e.path);
      out.push({
        path: e.path,
        size: Math.max(0, Number(e?.lfs?.size ?? e?.size) || 0),
        sha256: typeof e?.lfs?.oid === 'string' ? e.lfs.oid : '',
      });
    }
    const next = parseLinkNext(typeof res.headers?.get === 'function' ? res.headers.get('link') : '');
    url = next && !visited.has(next) ? next : '';
  }
  return out;
}

/** 管理员 JSON 请求（走指定 fetchImpl，便于整条浏览器通道统一替换/测试）。 */
async function adminJson(fetchImpl, path, { token, body } = {}) {
  const headers = { 'content-type': 'application/json' };
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetchImpl(path, {
    method: 'POST',
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || `请求失败（HTTP ${res.status}）`);
    err.status = res.status;
    throw err;
  }
  return data;
}

/**
 * 瘦身文件树：回传本地服务时只留判定/分组需要的字段。
 * 量化分组只需要 GGUF 文件的路径与大小；全量回传会让大仓库请求体膨胀。
 */
function slimFiles(files, { ggufOnly = false } = {}) {
  return (Array.isArray(files) ? files : [])
    .filter((f) => f && typeof f.path === 'string' && (!ggufOnly || /\.gguf$/i.test(f.path)))
    .map((f) => ({ path: f.path, size: Math.max(0, Number(f.size) || 0) }));
}

/** 量化列表（浏览器通道）：浏览器取文件树 → 本地服务分组 → 返回量化列表。 */
export async function fetchQuantsViaBrowser({ id, repo, token, fetchImpl = defaultFetch, timeoutMs = 60000 } = {}) {
  const files = slimFiles(await fetchRepoTreeViaBrowser(repo, { fetchImpl, timeoutMs }), { ggufOnly: true });
  const headers = { 'content-type': 'application/json' };
  if (token) headers.authorization = `Bearer ${token}`;
  const res = await fetchImpl(`/api/native/models/${encodeURIComponent(id)}/quants-preview`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ files }),
    cache: 'no-store',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `本地分组失败（HTTP ${res.status}）`);
  return data.quants || [];
}

/** 向软件写入一块权重字节；返回软件确认的落盘情况。 */
async function postDownloadChunk({ fetchImpl, id, quant, filePath, offset, data, adminToken }) {
  let res;
  try {
    res = await fetchImpl(`/api/native/models/${encodeURIComponent(id)}/browser-download/chunk`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${adminToken}`,
      'content-type': 'application/octet-stream',
      'x-bd-quant': String(quant || ''),
      'x-bd-file': encodeURIComponent(filePath),
      'x-bd-offset': String(offset),
    },
      body: data,
      cache: 'no-store',
    });
  } catch (err) {
    throw new Error(`把数据写入软件时连接中断（${err?.message || '网络错误'}），已落盘部分保留，重新点下载会续传。`);
  }
  const out = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(out.error || `写入软件失败（HTTP ${res.status}）`);
    err.status = res.status;
    if (Number.isFinite(out.received)) err.received = out.received;
    throw err;
  }
  return out;
}

/** 用一块空数据探软件当前的真实落盘偏移（校准续传位置）。 */
async function probeDownloadOffset(ctx, quant, filePath, guess) {
  try {
    const out = await postDownloadChunk({ ...ctx, quant, filePath, offset: guess, data: new Uint8Array(0) });
    return out.received;
  } catch (err) {
    if (Number.isFinite(err?.received)) return err.received;
    throw err;
  }
}

function mergeParts(parts, totalBytes) {
  if (parts.length === 1) return parts[0];
  const merged = new Uint8Array(totalBytes);
  let at = 0;
  for (const part of parts) {
    merged.set(part, at);
    at += part.length;
  }
  return merged;
}

/** 浏览器通道下载单个文件：Range 续传、分块上传、失败按落盘偏移重开（最多 3 次尝试）。 */
/**
 * 把浏览器/网络层的英文裸错（fetch failed、Failed to fetch、NetworkError…）
 * 归一化成小白能看懂的中文。用户两次反馈都只看到 fetch failed，就是因为这类
 * 裸错从某一层原样漏到了界面上；所有上抛前都过这一道。
 */
function friendlyDownloadError(err, filePath) {
  const raw = String(err?.message || err || '').trim();
  if (/fetch failed|failed to fetch|networkerror|network error|load failed|err_network|connection.*(reset|refused|closed)|socket/i.test(raw)) {
    return new Error(`下载 ${filePath || '文件'} 时网络中断（${raw}）：多半是网络抖动或代理不稳，已下载的部分都保留着，重新点下载会从断点续传，不会从头来。`);
  }
  return err instanceof Error ? err : new Error(raw || '下载失败，请重试。');
}

async function downloadOneFileViaBrowser(ctx) {
  const { file, onProgress } = ctx;
  let sent = Math.max(0, Number(file.offset) || 0);
  let lastError = null;
  // 大文件要传很久：重试放宽到 5 次、间隔指数拉长扛网络抖动；每次都按已落盘偏移续传
  for (let attempt = 0; attempt < 5; attempt += 1) {
    let reader = null;
    try {
      const headers = {};
      if (sent > 0) headers.range = `bytes=${sent}-`;
      let res;
      try {
        res = await ctx.fetchImpl(file.url, { headers, cache: 'no-store' });
      } catch (err) {
        // 浏览器底层只给英文裸错（Failed to fetch 之类）：换成用户能看懂的中文再上抛
        throw new Error(`从 Hugging Face 拉取 ${file.path} 时网络中断（${err?.message || '网络错误'}），已传部分保留，重新点下载会续传。`);
      }
      if (sent > 0 && res.status === 200) {
        throw new Error('模型文件没有按断点返回内容（Range 被忽略），浏览器通道不可用。');
      }
      if (res.status === 401 || res.status === 403) {
        // 门控/私有仓库：这些必须是浏览器请求带上 Token 才有权限，换直连也一样 403
        throw Object.assign(
          new Error(`该仓库是门控/私有仓库，或该文件需要授权（HTTP ${res.status}），请先保存 Hugging Face Token 再试。`),
          { status: res.status },
        );
      }
      if (res.status === 404) {
        throw Object.assign(
          new Error(`仓库里找不到 ${file.path}（HTTP 404）：仓库可能已改名/删档，或文件被移走，请重新搜索这个模型。`),
          { status: 404 },
        );
      }
      if (res.status !== 200 && res.status !== 206) {
        throw new Error(`从 Hugging Face 下载 ${file.path} 失败（HTTP ${res.status}）。`);
      }
      if (!res.body?.getReader) throw new Error('当前浏览器不支持流式读取，浏览器通道不可用。');
      if (ctx.shouldStop?.()) throw new Error('已有其他文件下载失败，本文件暂停（已传部分保留，可续传）。');
      reader = res.body.getReader();
      let parts = [];
      let partsBytes = 0;
      const flush = async () => {
        if (!partsBytes) return;
        const data = mergeParts(parts, partsBytes);
        parts = [];
        partsBytes = 0;
        // 同一块字节就地重试：写本地服务偶发失败（连接抖动、响应丢失）不该连累
        // 已经从 HF 拉到的这一块回炉重读。服务端按偏移落盘是幂等的：若字节其实
        // 已写入只是响应丢了，重试会收到 409 并带回真实落盘位置，对上账就算成功。
        let postError = null;
        for (let postAttempt = 0; postAttempt < 3; postAttempt += 1) {
          try {
            const out = await postDownloadChunk({ ...ctx, filePath: file.path, offset: sent, data });
            if (out.received !== sent + data.length) {
              const err = new Error('软件确认的落盘位置对不上，自动校准后重试。');
              err.received = out.received;
              throw err;
            }
            sent = out.received;
            onProgress?.(sent);
            return;
          } catch (err) {
            if (Number.isFinite(err?.received)) {
              if (err.received === sent + data.length) {
                // 字节已落盘、只是成功响应丢了：认账，不重传
                sent = err.received;
                onProgress?.(sent);
                return;
              }
              throw err; // 真错位：交给外层按真实偏移校准续传
            }
            postError = err;
            if (postAttempt < 2) await new Promise((r) => setTimeout(r, 500 * (postAttempt + 1)));
          }
        }
        throw postError || new Error('把数据写入软件失败，请重试。');
      };
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (ctx.shouldStop?.()) throw new Error('已有其他文件下载失败，本文件暂停（已传部分保留，可续传）。');
        if (value?.length) {
          parts.push(value);
          partsBytes += value.length;
          if (partsBytes >= ctx.chunkBytes) await flush();
        }
      }
      await flush();
      if (sent !== file.size) {
        throw new Error(`文件 ${file.path} 还差 ${Math.max(0, file.size - sent)} 字节没有传完。`);
      }
      return;
    } catch (err) {
      lastError = err;
      try { await reader?.cancel(); } catch { /* 忽略 */ }
      // 并行下载里已有别的文件失败：不用再重试探路，直接停，根因由失败的那个文件给出
      if (ctx.shouldStop?.()) throw err;
      // 权限/不存在类错误（401/403/404）重试多少次都一样，只会让用户多等几十秒
      if ([401, 403, 404].includes(Number(err?.status))) throw friendlyDownloadError(err, file.path);
      if (Number.isFinite(err?.received)) {
        sent = err.received;
      } else if (attempt < 4) {
        try {
          sent = await probeDownloadOffset(ctx, ctx.quant, file.path, sent);
        } catch { /* 探不到就按原偏移重试 */ }
        await new Promise((r) => setTimeout(r, ctx.retryDelayMs ?? Math.min(5000, 800 * (2 ** attempt))));
      }
    }
  }
  throw friendlyDownloadError(lastError, file.path);
}

/**
 * 浏览器通道下载权重：浏览器取文件树与字节流，本地服务按偏移落盘、核对转正。
 * 任何环节失败都会抛错，调用方据此回退到服务端直连下载。
 * @returns {Promise<{id: string, quant: string, state: string}>}
 */
export async function downloadModelViaBrowser({
  id, repo, quant, adminToken, onStatus,
  // 一块 2MB：足够大到不让本地请求次数爆炸，又足够细——慢网下也几秒就跳一次进度，
  // 不至于攒 8MB 才出现第一个百分比，用户误以为“没有下载进度”。
  fetchImpl = defaultFetch, treeTimeoutMs = 60000, chunkBytes = 2 * 1024 * 1024, retryDelayMs,
} = {}) {
  const note = (text) => { try { onStatus?.(text); } catch { /* 忽略回调错误 */ } };
  note('正在经浏览器读取仓库文件列表（和网页版同一条网络路）…');
  const files = slimFiles(await fetchRepoTreeViaBrowser(repo, { fetchImpl, timeoutMs: treeTimeoutMs }), { ggufOnly: true });
  const prep = await adminJson(fetchImpl, `/api/native/models/${encodeURIComponent(id)}/browser-download/prepare`, {
    token: adminToken,
    body: { quant, files },
  });
  if (prep.state === 'done') {
    note('这个量化已经下载完成，可以直接点“加载”使用。');
    return prep;
  }
  const ctx = { fetchImpl, id, quant: prep.quant, adminToken, chunkBytes, retryDelayMs };
  let moved = (prep.files || []).reduce((n, f) => n + (Number(f.offset) || 0), 0);
  // 还没传第一个字节就先给个 0%：别让界面停在“正在读取文件列表”，误以为卡住
  note(`正在下载（浏览器通道）：${(moved / 1e9).toFixed(2)} / ${((prep.total || 0) / 1e9).toFixed(2)} GB（${prep.total ? Math.floor((moved / prep.total) * 100) : 0}%），关掉这个页面会中断，重新点下载可续传。`);
  try {
    // 分片并行：大模型的量化通常有几十个分片文件，3 个一起传比排队逐个传快得多；
    // 并发只跨文件（服务端按文件分开落 .part），单个文件内部仍顺序传、偏移不会乱。
    // 一个文件失败时其余在飞的会尽快停下（已落盘的字节都保留，下次续传）。
    const pending = (prep.files || []).filter((f) => !(f.done || (Number(f.offset) || 0) >= f.size));
    const sentByPath = new Map(pending.map((f) => [f.path, Number(f.offset) || 0]));
    let firstError = null;
    let cursor = 0;
    const worker = async () => {
      for (;;) {
        if (firstError) return;
        const file = pending[cursor];
        cursor += 1;
        if (!file) return;
        try {
          await downloadOneFileViaBrowser({
            ...ctx,
            file,
            shouldStop: () => !!firstError,
            onProgress: (sentNow) => {
              sentByPath.set(file.path, sentNow);
              moved = (prep.files || []).reduce((n, f) => n
                + (sentByPath.has(f.path) ? sentByPath.get(f.path) : (Number(f.offset) || 0)), 0);
              const pct = prep.total ? Math.floor((moved / prep.total) * 100) : 0;
              note(`正在下载（浏览器通道）：${(moved / 1e9).toFixed(2)} / ${(prep.total / 1e9).toFixed(2)} GB（${pct}%），关掉这个页面会中断，重新点下载可续传。`);
            },
          });
        } catch (err) {
          if (!firstError) firstError = err;
          return;
        }
      }
    };
    await Promise.all(Array.from({ length: Math.min(3, Math.max(1, pending.length)) }, () => worker()));
    if (firstError) throw firstError;
    const out = await adminJson(fetchImpl, `/api/native/models/${encodeURIComponent(id)}/browser-download/finish`, {
      token: adminToken,
      body: { quant: prep.quant },
    });
    note('下载完成（浏览器通道）：现在可以点“加载”使用这个模型。');
    return out;
  } catch (err) {
    const friendly = friendlyDownloadError(err);
    // 尽力通知软件标记失败（保留 .part 续传），失败也无妨；标的也是归一化后的中文
    try {
      await adminJson(fetchImpl, `/api/native/models/${encodeURIComponent(id)}/browser-download/fail`, {
        token: adminToken,
        body: { quant: prep.quant, error: friendly.message },
      });
    } catch { /* 忽略 */ }
    throw friendly;
  }
}
