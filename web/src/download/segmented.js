/**
 * 可续传的分段下载（生图模型与文本模型共用）。
 *
 * - 文件按 partBytes 切成分段，分段之间并行下载（并发由调用方的 limit 控制）。
 * - 下载过程中已收到的字节会随时保存（包括网络中断、用户取消时），不会因为一段没下完就整段丢弃。
 *   分段的临时数据保存在独立的 parts 缓存中，键为 “地址 + 段号”，内容长度即已保存的字节数。
 * - 下次下载时，每段从已保存的位置用 Range 继续，只请求缺少的字节。
 * - 全部分段齐全后，按顺序拼接成完整文件放入最终缓存，然后删除分段。
 * - 服务器不支持 Range（返回 200）时，改为顺序读取整个文件，并把字节切入各段保存（仍然保留已下载的部分）。
 *
 * 错误约定：4xx（408、429 除外）为永久错误（PermanentDownloadError），其余视为暂时性错误，由 withRetry 重试。
 */

export const PART_ORIGIN = 'https://cache.oj-ai.invalid/parts/';

export class PermanentDownloadError extends Error {}

/** 服务器忽略了 Range（返回 200）。内部信号，调用方据此改用整文件读取。 */
export class RangeUnsupported extends Error {}

export function abortError() {
  return Object.assign(new Error('已取消下载'), { name: 'AbortError' });
}

/** 可被取消的等待。 */
export function sleepMs(ms, signal) {
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

/** 4xx（408、429 除外）为永久错误；5xx、408、429 与网络错误都视为暂时的。 */
export function isPermanentStatus(status) {
  return status >= 400 && status < 500 && status !== 408 && status !== 429;
}

/**
 * 执行一次下载任务，暂时性错误自动重试（退避 baseMs、2×baseMs …，最长 maxMs），直到成功或被取消。
 * 永久错误与 RangeUnsupported 直接抛出，由调用方处理。
 */
export async function withRetry(task, { signal, onRetry, sleep = sleepMs, baseMs, maxMs }) {
  for (let attempt = 1; ; attempt += 1) {
    if (signal?.aborted) throw abortError();
    try {
      return await task();
    } catch (err) {
      if (err?.name === 'AbortError' || signal?.aborted) throw err;
      if (err instanceof PermanentDownloadError || err instanceof RangeUnsupported) throw err;
      const waitMs = Math.min(maxMs, baseMs * 2 ** Math.min(attempt - 1, 16));
      onRetry?.({ attempt, waitMs, message: err?.message || String(err) });
      await sleep(waitMs, signal);
    }
  }
}

export function partCount(size, partBytes) {
  return Math.ceil(size / partBytes);
}

/** 第 index 段的字节范围（含端点）。 */
export function partRange(size, partBytes, index) {
  const start = index * partBytes;
  const end = Math.min(size, start + partBytes) - 1;
  return { start, end, length: end - start + 1 };
}

export function partPrefix(url) {
  return `${PART_ORIGIN}${encodeURIComponent(url)}/`;
}

export function partKey(url, index) {
  return `${partPrefix(url)}${index}`;
}

/**
 * 每段已保存的字节数（have）。段完整时 have === 段长度。
 * 已保存的内容长度（content-length）超过段长度的视为无效，按 0 处理。
 */
export async function segmentStatus(parts, url, size, partBytes) {
  const n = partCount(size, partBytes);
  const have = [];
  let bytes = 0;
  let completeCount = 0;
  for (let i = 0; i < n; i += 1) {
    const { length } = partRange(size, partBytes, i);
    const res = await parts.match(partKey(url, i));
    let h = 0;
    if (res) {
      const len = Number(res.headers.get('content-length'));
      if (Number.isFinite(len) && len > 0 && len <= length) h = len;
    }
    have.push(h);
    bytes += h;
    if (h === length) completeCount += 1;
  }
  return { n, have, bytes, completeCount, complete: completeCount === n };
}

function concatChunks(chunks, total) {
  const out = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

async function putPart(parts, url, index, bytes) {
  await parts.put(partKey(url, index), new Response(bytes, { headers: { 'content-length': String(bytes.length) } }));
}

/** 下载一个分段（或它的剩余部分）。已保存的字节不会重下；任何已收到的字节都会保存，包括出错时。 */
async function fetchSegment(ctx, i) {
  const { url, label, size, partBytes, parts, fetchImpl, signal, addBytes, n } = ctx;
  const { start, end, length } = partRange(size, partBytes, i);
  const key = partKey(url, i);
  let chunks = [];
  let have = 0;
  const old = await parts.match(key);
  if (old) {
    const buf = new Uint8Array(await old.arrayBuffer());
    if (buf.length === length) return; // 已完整
    if (buf.length > 0 && buf.length < length) {
      chunks = [buf];
      have = buf.length;
    }
  }
  let got = 0;
  try {
    const res = await fetchImpl(url, { signal, headers: { Range: `bytes=${start + have}-${end}` } });
    if (res.status === 200) {
      await res.body?.cancel();
      throw new RangeUnsupported();
    }
    if (res.status !== 206) {
      const where = `${label}（第 ${i + 1}/${n} 段）`;
      if (isPermanentStatus(res.status)) throw new PermanentDownloadError(`下载失败（HTTP ${res.status}）：${where}`);
      throw new Error(`服务器返回 ${res.status}：${where}`);
    }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      got += value.length;
      have += value.length;
      addBytes(value.length);
      if (have > length) throw new Error(`分段超出长度（第 ${i + 1}/${n} 段）：${label}`);
    }
    if (have !== length) throw new Error(`分段大小不符（第 ${i + 1}/${n} 段）：${label}`);
    await putPart(parts, url, i, concatChunks(chunks, have));
  } catch (err) {
    // 已收到的字节照常保存（网络中断、取消时也一样），下次从这里继续
    if (got > 0 && have <= length) await putPart(parts, url, i, concatChunks(chunks, have));
    throw err;
  }
}

/**
 * 服务器不支持 Range 时：顺序读取整个文件，把字节切入各段。
 * 已完整的段跳过；未完整的段从头收集，只有超过已保存长度的部分才会写入（不计入进度的字节不会重复计数）。
 */
async function fetchWhole(ctx, todo, have) {
  const { url, label, size, partBytes, parts, fetchImpl, signal, addBytes, n } = ctx;
  const res = await fetchImpl(url, { signal });
  if (!res.ok) {
    if (isPermanentStatus(res.status)) throw new PermanentDownloadError(`下载失败（HTTP ${res.status}）：${label}`);
    throw new Error(`服务器返回 ${res.status}：${label}`);
  }
  const segs = new Map(); // 段号 -> { chunks, got }
  let pos = 0;
  const savePartial = async () => {
    for (const [i, s] of segs) {
      if (s.got > have[i]) await putPart(parts, url, i, concatChunks(s.chunks, s.got));
    }
  };
  const reader = res.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      let off = 0;
      while (off < value.length) {
        const i = Math.floor(pos / partBytes);
        const { start, length } = partRange(size, partBytes, i);
        const segEnd = start + length;
        const take = Math.min(value.length - off, segEnd - pos);
        if (todo.has(i)) {
          let s = segs.get(i);
          if (!s) {
            s = { chunks: [], got: 0 };
            segs.set(i, s);
          }
          s.chunks.push(value.subarray(off, off + take));
          s.got += take;
          const newBytes = pos + take - Math.max(pos, have[i]);
          if (newBytes > 0) addBytes(newBytes);
          if (pos + take === segEnd) {
            await putPart(parts, url, i, concatChunks(s.chunks, length));
            segs.delete(i);
          }
        }
        pos += take;
        off += take;
      }
    }
    if (pos !== size) throw new Error(`下载不完整（已收到 ${pos} / ${size} 字节）：${label}`);
  } catch (err) {
    await savePartial();
    throw err;
  }
}

/** 并行运行 fn，最多 concurrency 个同时进行；出现错误后不再启动新的项目，进行中的项目完成后抛出第一个错误。 */
async function runPool(items, concurrency, fn) {
  let next = 0;
  let failure = null;
  const worker = async () => {
    while (!failure && next < items.length) {
      const item = items[next];
      next += 1;
      try {
        await fn(item);
      } catch (err) {
        failure ??= err;
        return;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(Math.max(1, concurrency), items.length) }, worker));
  if (failure) throw failure;
}

/** 把全部分段按顺序拼接成完整文件放入最终缓存，然后删除分段。分段不完整时抛出错误（已保存的部分保留）。 */
async function assemble({ finalCache, parts, url, size, partBytes, n, label }) {
  let index = 0;
  let reader = null;
  let bytesInSegment = 0;
  let expected = 0;
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
            controller.error(new Error(`缓存分段丢失（第 ${index + 1}/${n} 段）：${label}`));
            return;
          }
          expected = partRange(size, partBytes, index).length;
          bytesInSegment = 0;
          reader = res.body.getReader();
        }
        const { done, value } = await reader.read();
        if (done) {
          if (bytesInSegment !== expected) {
            controller.error(new Error(`分段不完整（第 ${index + 1}/${n} 段）：${label}`));
            return;
          }
          reader = null;
          index += 1;
          continue;
        }
        bytesInSegment += value.length;
        controller.enqueue(value);
        return;
      }
    },
    cancel() {
      reader?.cancel();
    },
  });
  await finalCache.put(url, new Response(body, { headers: { 'content-length': String(size) } }));
  for (let i = 0; i < n; i += 1) await parts.delete(partKey(url, i));
}

/**
 * 下载一个文件到 finalCache（键为 url）。已保存的分段与字节会跳过 / 续传。
 * @param {object} p
 * @param {string} p.url 文件地址（同时作为最终缓存的键）
 * @param {string} [p.label] 用于错误信息的文件名
 * @param {number} p.size 文件大小（字节）
 * @param {Cache} p.finalCache 最终缓存
 * @param {Cache} p.parts 分段缓存
 * @param {number} p.partBytes 分段大小
 * @param {number} p.concurrency 同一文件内的分段并发
 * @param {Function} p.limit 全局并发限制（createLimiter 返回的函数），只包住单次请求
 * @param {Function} p.retry 重试包装（withRetry 的绑定版本）
 * @param {(n: number) => void} p.addBytes 每收到 n 个新字节时调用（不包括已保存的字节）
 */
export async function downloadSegmentedFile({
  url,
  label = url,
  size,
  finalCache,
  parts,
  fetchImpl = globalThis.fetch,
  signal,
  partBytes,
  concurrency,
  limit,
  retry,
  addBytes,
}) {
  const n = partCount(size, partBytes);
  const ctx = { url, label, size, partBytes, parts, fetchImpl, signal, addBytes, n };
  const incomplete = (s) => {
    const list = [];
    for (let i = 0; i < s.n; i += 1) if (s.have[i] !== partRange(size, partBytes, i).length) list.push(i);
    return list;
  };

  // 单段文件且尚未保存任何字节：整文件一次请求（不需要 Range）
  if (n === 1) {
    const s0 = await segmentStatus(parts, url, size, partBytes);
    if (s0.have[0] === 0) {
      await retry(() =>
        limit(async () => {
          const s = await segmentStatus(parts, url, size, partBytes);
          if (s.have[0] === 0) await fetchWhole(ctx, new Set([0]), s.have);
        }),
      );
    }
  }

  // 其余未完成的段：分段并行下载（续传用 Range）
  const todo = incomplete(await segmentStatus(parts, url, size, partBytes));
  if (todo.length > 0) {
    try {
      await runPool(todo, concurrency, (i) => retry(() => limit(() => fetchSegment(ctx, i))));
    } catch (err) {
      if (!(err instanceof RangeUnsupported)) throw err;
      if (signal?.aborted) throw abortError();
      // 服务器不支持 Range：顺序读取整个文件（每次尝试前重新读取已保存的状态）
      await retry(() =>
        limit(async () => {
          const s = await segmentStatus(parts, url, size, partBytes);
          const set = new Set(incomplete(s));
          if (set.size > 0) await fetchWhole(ctx, set, s.have);
        }),
      );
    }
  }

  const after = await segmentStatus(parts, url, size, partBytes);
  if (!after.complete) {
    throw new Error(`分段未齐全（缺少 ${after.n - after.completeCount} 段）：${label}`);
  }
  await assemble({ finalCache, parts, url, size, partBytes, n: after.n, label });
}

/** 删除某文件的全部分段。 */
export async function deleteSegments(parts, url) {
  const prefix = partPrefix(url);
  for (const req of await parts.keys()) {
    if (req.url.startsWith(prefix)) await parts.delete(req.url);
  }
}
