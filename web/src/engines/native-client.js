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

/** 浏览器直连 HF 文件树，归一化为与服务端 fetchRepoFiles 一致的清单。 */
export async function fetchRepoTreeViaBrowser(repo, { fetchImpl = fetch, timeoutMs = 30000 } = {}) {
  const signal = typeof AbortSignal?.timeout === 'function' && timeoutMs > 0
    ? AbortSignal.timeout(timeoutMs)
    : undefined;
  let res;
  try {
    res = await fetchImpl(`${HF_WEB_BASE}/api/models/${repo}/tree/main?recursive=true`, {
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
  if (!res.ok) throw new Error(`读取模型仓库 ${repo} 失败（HTTP ${res.status}）。`);
  const tree = await res.json().catch(() => null);
  if (!Array.isArray(tree)) throw new Error(`模型仓库 ${repo} 返回了无法识别的文件列表。`);
  return tree
    .filter((e) => e && e.type === 'file' && typeof e.path === 'string')
    .map((e) => ({
      path: e.path,
      size: Math.max(0, Number(e?.lfs?.size ?? e?.size) || 0),
      sha256: typeof e?.lfs?.oid === 'string' ? e.lfs.oid : '',
    }));
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

/** 量化列表（浏览器通道）：浏览器取文件树 → 本地服务分组 → 返回量化列表。 */
export async function fetchQuantsViaBrowser({ id, repo, token, fetchImpl = fetch, timeoutMs = 30000 } = {}) {
  const files = await fetchRepoTreeViaBrowser(repo, { fetchImpl, timeoutMs });
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
  const res = await fetchImpl(`/api/native/models/${encodeURIComponent(id)}/browser-download/chunk`, {
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
async function downloadOneFileViaBrowser(ctx) {
  const { file, onProgress } = ctx;
  let sent = Math.max(0, Number(file.offset) || 0);
  let lastError = null;
  for (let attempt = 0; attempt < 3; attempt += 1) {
    let reader = null;
    try {
      const headers = {};
      if (sent > 0) headers.range = `bytes=${sent}-`;
      const res = await ctx.fetchImpl(file.url, { headers, cache: 'no-store' });
      if (sent > 0 && res.status === 200) {
        throw new Error('模型文件没有按断点返回内容（Range 被忽略），浏览器通道不可用。');
      }
      if (res.status !== 200 && res.status !== 206) {
        throw new Error(`从 Hugging Face 下载 ${file.path} 失败（HTTP ${res.status}）。`);
      }
      if (!res.body?.getReader) throw new Error('当前浏览器不支持流式读取，浏览器通道不可用。');
      reader = res.body.getReader();
      let parts = [];
      let partsBytes = 0;
      const flush = async () => {
        if (!partsBytes) return;
        const data = mergeParts(parts, partsBytes);
        parts = [];
        partsBytes = 0;
        const out = await postDownloadChunk({ ...ctx, filePath: file.path, offset: sent, data });
        if (out.received !== sent + data.length) {
          const err = new Error('软件确认的落盘位置对不上，自动校准后重试。');
          err.received = out.received;
          throw err;
        }
        sent = out.received;
        onProgress?.(sent);
      };
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
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
      if (Number.isFinite(err?.received)) {
        sent = err.received;
      } else if (attempt < 2) {
        try {
          sent = await probeDownloadOffset(ctx, ctx.quant, file.path, sent);
        } catch { /* 探不到就按原偏移重试 */ }
        await new Promise((r) => setTimeout(r, 800));
      }
    }
  }
  throw lastError || new Error(`下载 ${file.path} 失败。`);
}

/**
 * 浏览器通道下载权重：浏览器取文件树与字节流，本地服务按偏移落盘、核对转正。
 * 任何环节失败都会抛错，调用方据此回退到服务端直连下载。
 * @returns {Promise<{id: string, quant: string, state: string}>}
 */
export async function downloadModelViaBrowser({
  id, repo, quant, adminToken, onStatus,
  fetchImpl = fetch, treeTimeoutMs = 30000, chunkBytes = 8 * 1024 * 1024,
} = {}) {
  const note = (text) => { try { onStatus?.(text); } catch { /* 忽略回调错误 */ } };
  note('正在经浏览器读取仓库文件列表（和网页版同一条网络路）…');
  const files = await fetchRepoTreeViaBrowser(repo, { fetchImpl, timeoutMs: treeTimeoutMs });
  const prep = await adminJson(fetchImpl, `/api/native/models/${encodeURIComponent(id)}/browser-download/prepare`, {
    token: adminToken,
    body: { quant, files },
  });
  if (prep.state === 'done') {
    note('这个量化已经下载完成，可以直接点“加载”使用。');
    return prep;
  }
  const ctx = { fetchImpl, id, quant: prep.quant, adminToken, chunkBytes };
  let moved = (prep.files || []).reduce((n, f) => n + (Number(f.offset) || 0), 0);
  try {
    for (const file of prep.files || []) {
      if (file.done || (Number(file.offset) || 0) >= file.size) continue;
      let fileBase = Number(file.offset) || 0;
      await downloadOneFileViaBrowser({
        ...ctx,
        file,
        onProgress: (sentNow) => {
          moved += sentNow - fileBase;
          fileBase = sentNow;
          const pct = prep.total ? Math.floor((moved / prep.total) * 100) : 0;
          note(`正在下载（浏览器通道）：${(moved / 1e9).toFixed(2)} / ${(prep.total / 1e9).toFixed(2)} GB（${pct}%），关掉这个页面会中断，重新点下载可续传。`);
        },
      });
    }
    const out = await adminJson(fetchImpl, `/api/native/models/${encodeURIComponent(id)}/browser-download/finish`, {
      token: adminToken,
      body: { quant: prep.quant },
    });
    note('下载完成（浏览器通道）：现在可以点“加载”使用这个模型。');
    return out;
  } catch (err) {
    // 尽力通知软件标记失败（保留 .part 续传），失败也无妨
    try {
      await adminJson(fetchImpl, `/api/native/models/${encodeURIComponent(id)}/browser-download/fail`, {
        token: adminToken,
        body: { quant: prep.quant, error: err?.message || '浏览器通道下载中断' },
      });
    } catch { /* 忽略 */ }
    throw err;
  }
}
