/**
 * 可续传的分段下载（生图模型与文本模型共用）。
 *
 * - 文件按 partBytes 切成分段，分段之间并行下载（并发由调用方的 limit 控制）。
 * - 每个分段再切成 BLOCK_BYTES（1 MB）的块。块每写满就保存一次，网络中断、用户取消时未写满的块也会保存；
 *   因此下载中的数据最多只有一个未写满的块（每个连接不超过 1 MB）只存在于内存中。
 * - 块保存在独立的 parts 缓存中，键为 “地址 + 段号/块号/长度”，长度即该块已保存的字节数。
 * - 下次下载时，从已保存的字节位置用 Range 继续，只请求缺少的字节；已保存的字节不会重下。
 * - 全部分段齐全后，按顺序拼接成完整文件放入最终缓存，然后删除分块。
 * - 服务器不支持 Range（返回 200）时，改为顺序读取整个文件，并把字节写入各段的块（仍然保留已下载的部分）。
 *
 * 错误约定：4xx（408、429 除外）为永久错误（PermanentDownloadError），其余视为暂时性错误，由 withRetry 重试。
 */

export const PART_ORIGIN = 'https://cache.oj-ai.invalid/parts/';
/** 块大小：每写满一块即保存。越小越频繁写入（刷新页面时丢失的字节越少），但缓存条目越多。 */
export const BLOCK_BYTES = 1024 * 1024;

export class PermanentDownloadError extends Error {}

/** 连接超过这么久没有收到任何数据，视为卡住：中止本次请求，已收到的字节保存后自动重试。 */
export const STALL_MS = 30 * 1000;

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

/**
 * 卡住检测：只在等待数据时计时（写入缓存的时间不计入）。
 * arm() 开始等待，disarm() 收到数据后停止计时；超时后中止本次请求（signal），failure() 把结果转换为可重试的错误。
 */
export function stallGuard(parentSignal, stallMs) {
  const ctl = new AbortController();
  let timer = null;
  let stalled = false;
  const onParent = () => ctl.abort();
  parentSignal?.addEventListener('abort', onParent, { once: true });
  return {
    signal: ctl.signal,
    arm() {
      clearTimeout(timer);
      timer = setTimeout(() => {
        stalled = true;
        ctl.abort();
      }, stallMs);
    },
    disarm() {
      clearTimeout(timer);
    },
    stop() {
      clearTimeout(timer);
      parentSignal?.removeEventListener('abort', onParent);
    },
    /** 把请求或读取时的错误转换：用户取消保持 AbortError；卡住则变为暂时性错误（会自动重试）。 */
    failure(err) {
      if (parentSignal?.aborted) return abortError();
      if (stalled) return new Error(`网络超过 ${Math.round(stallMs / 1000)} 秒没有数据，将自动重试`);
      return err;
    },
  };
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

/** 块大小：不超过 BLOCK_BYTES，也不超过分段大小。 */
function blockBytesOf(partBytes) {
  return Math.min(BLOCK_BYTES, partBytes);
}

function blockCount(segLength, blk) {
  return Math.ceil(segLength / blk);
}

/** 段内第 j 块的长度（最后一块可能较短）。 */
function blockLength(segLength, blk, j) {
  return Math.min(blk, segLength - j * blk);
}

export function partPrefix(url) {
  return `${PART_ORIGIN}${encodeURIComponent(url)}/`;
}

/** 块的缓存键：地址 + 段号 / 块号 / 已保存长度。长度写在键里，列出键即可得知进度，不必读取内容。 */
function blockKey(url, i, j, len) {
  return `${partPrefix(url)}${i}/${j}/${len}`;
}

function parseBlockKey(prefix, reqUrl) {
  if (!reqUrl.startsWith(prefix)) return null;
  const m = /^(\d+)\/(\d+)\/(\d+)$/.exec(reqUrl.slice(prefix.length));
  return m ? { i: Number(m[1]), j: Number(m[2]), len: Number(m[3]) } : null;
}

/**
 * 各段已保存的字节数（have）与各块的记录。段完整时 have === 段长度。
 * blocks：Map “段号/块号” -> { len, key }（同一块有多条记录时取较长的一条）。
 */
export async function segmentStatus(parts, url, size, partBytes) {
  const n = partCount(size, partBytes);
  const blk = blockBytesOf(partBytes);
  const prefix = partPrefix(url);
  const blocks = new Map();
  for (const req of await parts.keys()) {
    const e = parseBlockKey(prefix, req.url);
    if (!e) continue;
    const id = `${e.i}/${e.j}`;
    const cur = blocks.get(id);
    if (!cur || e.len > cur.len) blocks.set(id, { len: e.len, key: req.url });
  }
  const have = [];
  let bytes = 0;
  let completeCount = 0;
  for (let i = 0; i < n; i += 1) {
    const { length } = partRange(size, partBytes, i);
    const nb = blockCount(length, blk);
    let h = 0;
    for (let j = 0; j < nb; j += 1) {
      const e = blocks.get(`${i}/${j}`);
      const want = blockLength(length, blk, j);
      if (!e || e.len > want) break;
      h += e.len;
      if (e.len < want) break;
    }
    have.push(h);
    bytes += h;
    if (h === length) completeCount += 1;
  }
  return { n, have, bytes, completeCount, complete: completeCount === n, blocks };
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

/**
 * 某段的写入器：按顺序接收字节，写满一块就保存一块；flushPartial 保存未写满的块。
 * - startBlock / prefix：从第 startBlock 块开始，prefix 为该块已保存的前缀（可为空）。
 * - skip：接收的字节中，前 skip 个已经保存过（例如整文件读取时，前面的段或块已完整），直接丢弃。
 * - prevKey：该块原有的记录键；写入新记录后删除，避免重复。
 */
function createSegmentWriter({ url, parts, i, size, partBytes, startBlock, prefix, prevKey, skip, blocks, label }) {
  const { length } = partRange(size, partBytes, i);
  const blk = blockBytesOf(partBytes);
  const nb = blockCount(length, blk);
  let j = startBlock;
  let cur = [];
  let curLen = 0;
  if (prefix && prefix.length > 0) {
    cur = [prefix];
    curLen = prefix.length;
  }
  let toSkip = skip;
  let prev = prevKey;
  let wrote = false;
  const blen = (jj) => blockLength(length, blk, jj);

  const save = async () => {
    const key = blockKey(url, i, j, curLen);
    await parts.put(key, new Response(concatChunks(cur, curLen), { headers: { 'content-length': String(curLen) } }));
    if (prev && prev !== key) await parts.delete(prev);
  };

  return {
    get complete() {
      return j === nb && curLen === 0;
    },
    /** 接收一段字节，返回其中新增（未保存过）的字节数。 */
    async feed(chunk) {
      let off = 0;
      if (toSkip > 0) {
        const d = Math.min(toSkip, chunk.length);
        toSkip -= d;
        off = d;
      }
      let accepted = 0;
      while (off < chunk.length) {
        if (j >= nb) throw new Error(`分段超出长度（第 ${i + 1} 段）：${label}`);
        const take = Math.min(blen(j) - curLen, chunk.length - off);
        cur.push(chunk.subarray(off, off + take));
        curLen += take;
        off += take;
        accepted += take;
        if (curLen === blen(j)) {
          await save();
          j += 1;
          cur = [];
          curLen = 0;
          prev = j < nb ? (blocks.get(`${i}/${j}`)?.key ?? null) : null;
        }
      }
      if (accepted > 0) wrote = true;
      return accepted;
    },
    /** 保存未写满的块（网络中断、取消时调用）。没有新字节时不写。 */
    async flushPartial() {
      if (wrote && curLen > 0 && j < nb) await save();
    },
  };
}

/**
 * 计算某段从哪里开始：已保存的完整块跳过；未写满的块读出已保存的前缀，继续写入。
 * base 为请求 / 跳过的起点（段内字节偏移）。
 */
async function planSegment(parts, size, partBytes, i, st, url) {
  const blk = blockBytesOf(partBytes);
  const have = st.have[i];
  const startBlock = Math.floor(have / blk);
  const fill = have - startBlock * blk;
  const e = st.blocks.get(`${i}/${startBlock}`) ?? null;
  let base = startBlock * blk;
  let prefix = null;
  if (fill > 0 && e && e.len === fill) {
    const res = await parts.match(e.key);
    if (res) {
      const b = new Uint8Array(await res.arrayBuffer());
      if (b.length === fill) {
        prefix = b;
        base = have;
      }
    }
  }
  return { startBlock, prefix, base, prevKey: e?.key ?? null, blocks: st.blocks, url, size, partBytes, i };
}

/** 下载一个分段（或它的剩余部分）。已保存的字节不会重下；每写满一块即保存，出错时保存未写满的块。 */
async function fetchSegment(ctx, i) {
  const { url, label, size, partBytes, parts, fetchImpl, signal, addBytes, n, stallMs } = ctx;
  const { start, end, length } = partRange(size, partBytes, i);
  const st = await segmentStatus(parts, url, size, partBytes);
  if (st.have[i] === length) return;
  const plan = await planSegment(parts, size, partBytes, i, st, url);
  const writer = createSegmentWriter({
    url,
    parts,
    i,
    size,
    partBytes,
    startBlock: plan.startBlock,
    prefix: plan.prefix,
    prevKey: plan.prevKey,
    skip: 0,
    blocks: plan.blocks,
    label,
  });
  const guard = stallGuard(signal, stallMs);
  try {
    guard.arm();
    const res = await fetchImpl(url, { signal: guard.signal, headers: { Range: `bytes=${start + plan.base}-${end}` } });
    guard.disarm();
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
      guard.arm();
      const { done, value } = await reader.read();
      guard.disarm();
      if (done) break;
      addBytes(await writer.feed(value));
    }
    if (!writer.complete) throw new Error(`分段大小不符（第 ${i + 1}/${n} 段）：${label}`);
  } catch (err) {
    // 已收到的字节照常保存（网络中断、卡住、取消时也一样），下次从这里继续
    const e = guard.failure(err);
    await writer.flushPartial();
    throw e;
  } finally {
    guard.stop();
  }
}

/**
 * 服务器不支持 Range 时：顺序读取整个文件，把字节写入各未完整的段。
 * 已完整的段跳过；未写满的块只写入超过已保存长度的部分（不计入进度的字节不会重复计数）。
 */
async function fetchWhole(ctx) {
  const { url, label, size, partBytes, parts, fetchImpl, signal, addBytes, stallMs, onNote } = ctx;
  const st = await segmentStatus(parts, url, size, partBytes);
  const todo = [];
  for (let i = 0; i < st.n; i += 1) if (st.have[i] !== partRange(size, partBytes, i).length) todo.push(i);
  if (todo.length === 0) return;

  const guard = stallGuard(signal, stallMs);
  let res;
  try {
    guard.arm();
    res = await fetchImpl(url, { signal: guard.signal });
    guard.disarm();
  } catch (err) {
    guard.stop();
    throw guard.failure(err);
  }
  if (!res.ok) {
    guard.stop();
    if (isPermanentStatus(res.status)) throw new PermanentDownloadError(`下载失败（HTTP ${res.status}）：${label}`);
    throw new Error(`服务器返回 ${res.status}：${label}`);
  }
  const writers = [];
  for (const i of todo) {
    const plan = await planSegment(parts, size, partBytes, i, st, url);
    const { start, length } = partRange(size, partBytes, i);
    writers.push({
      start,
      length,
      skip: plan.base,
      w: createSegmentWriter({
        url,
        parts,
        i,
        size,
        partBytes,
        startBlock: plan.startBlock,
        prefix: plan.prefix,
        prevKey: plan.prevKey,
        skip: plan.base,
        blocks: plan.blocks,
        label,
      }),
    });
  }

  if (writers.some((x) => x.skip > 0)) {
    onNote?.('服务器不支持断点续传，正在从头读取，并跳过已保存的部分…');
  }
  let pos = 0;
  try {
    const reader = res.body.getReader();
    for (;;) {
      guard.arm();
      const { done, value } = await reader.read();
      guard.disarm();
      if (done) break;
      const cEnd = pos + value.length;
      for (const { w, start, length } of writers) {
        const lo = Math.max(pos, start);
        const hi = Math.min(cEnd, start + length);
        if (hi > lo) addBytes(await w.feed(value.subarray(lo - pos, hi - pos)));
      }
      pos = cEnd;
    }
    if (pos !== size) throw new Error(`下载不完整（已收到 ${pos} / ${size} 字节）：${label}`);
    for (const { w } of writers) if (!w.complete) throw new Error(`分段不完整：${label}`);
  } catch (err) {
    const e = guard.failure(err);
    for (const { w } of writers) await w.flushPartial();
    throw e;
  } finally {
    guard.stop();
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

/** 把全部块按顺序拼接成完整文件放入最终缓存，然后删除分块。块不完整时抛出错误（已保存的部分保留）。 */
async function assemble({ finalCache, parts, url, size, partBytes, n, label, blocks, onNote }) {
  const blk = blockBytesOf(partBytes);
  const list = [];
  for (let i = 0; i < n; i += 1) {
    const { length } = partRange(size, partBytes, i);
    const nb = blockCount(length, blk);
    for (let j = 0; j < nb; j += 1) {
      const e = blocks.get(`${i}/${j}`);
      if (!e) throw new Error(`分块缺失（第 ${i + 1} 段 第 ${j + 1} 块）：${label}`);
      list.push({ key: e.key, len: blockLength(length, blk, j) });
    }
  }
  let idx = 0;
  let reader = null;
  let got = 0;
  // 拼接是复制整个文件，较慢；每 5% 报告一次，避免进度条看起来停住
  let copied = 0;
  let lastPct = -1;
  const report = () => {
    const pct = Math.floor((copied / size) * 20) * 5;
    if (pct !== lastPct) {
      lastPct = pct;
      onNote?.(`正在整理文件（${label}）：${pct}%，请稍候…`);
    }
  };
  onNote?.(`正在整理文件（${label}）…`);
  const body = new ReadableStream({
    async pull(controller) {
      for (;;) {
        if (!reader) {
          if (idx >= list.length) {
            controller.close();
            return;
          }
          const res = await parts.match(list[idx].key);
          if (!res) {
            controller.error(new Error(`缓存分块丢失（第 ${idx + 1}/${list.length} 块）：${label}`));
            return;
          }
          reader = res.body.getReader();
          got = 0;
        }
        const { done, value } = await reader.read();
        if (done) {
          if (got !== list[idx].len) {
            controller.error(new Error(`分块不完整（第 ${idx + 1}/${list.length} 块）：${label}`));
            return;
          }
          reader = null;
          idx += 1;
          continue;
        }
        got += value.length;
        copied += value.length;
        report();
        controller.enqueue(value);
        return;
      }
    },
    cancel() {
      reader?.cancel();
    },
  });
  await finalCache.put(url, new Response(body, { headers: { 'content-length': String(size) } }));
  await deleteSegments(parts, url);
}

/**
 * 下载一个文件到 finalCache（键为 url）。已保存的块与字节会跳过 / 续传。
 * @param {object} p
 * @param {string} p.url 文件地址（同时作为最终缓存的键）
 * @param {string} [p.label] 用于错误信息的文件名
 * @param {number} p.size 文件大小（字节）
 * @param {Cache} p.finalCache 最终缓存
 * @param {Cache} p.parts 分块缓存
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
  stallMs = STALL_MS,
  onNote,
}) {
  const n = partCount(size, partBytes);
  const ctx = { url, label, size, partBytes, parts, fetchImpl, signal, addBytes, n, stallMs, onNote };
  const incomplete = (s) => {
    const list = [];
    for (let i = 0; i < s.n; i += 1) if (s.have[i] !== partRange(size, partBytes, i).length) list.push(i);
    return list;
  };

  // 单段文件：尚未保存任何字节时整文件一次请求；已有断点时用 Range 续传（服务器不支持则整文件）
  if (n === 1) {
    await retry(() =>
      limit(async () => {
        const s = await segmentStatus(parts, url, size, partBytes);
        if (s.complete) return;
        if (s.have[0] === 0) {
          await fetchWhole(ctx);
          return;
        }
        try {
          await fetchSegment(ctx, 0);
        } catch (err) {
          if (!(err instanceof RangeUnsupported)) throw err;
          await fetchWhole(ctx);
        }
      }),
    );
  } else {
    // 多段文件：分段并行下载（续传用 Range）
    const todo = incomplete(await segmentStatus(parts, url, size, partBytes));
    if (todo.length > 0) {
      try {
        await runPool(todo, concurrency, (i) => retry(() => limit(() => fetchSegment(ctx, i))));
      } catch (err) {
        if (!(err instanceof RangeUnsupported)) throw err;
        if (signal?.aborted) throw abortError();
        // 服务器不支持 Range：顺序读取整个文件
        await retry(() => limit(() => fetchWhole(ctx)));
      }
    }
  }

  const after = await segmentStatus(parts, url, size, partBytes);
  if (!after.complete) {
    throw new Error(`分段未齐全（缺少 ${after.n - after.completeCount} 段）：${label}`);
  }
  await assemble({ finalCache, parts, url, size, partBytes, n: after.n, label, blocks: after.blocks, onNote });
}

/** 删除某文件的全部分块（含未完成下载的部分）。 */
export async function deleteSegments(parts, url) {
  const prefix = partPrefix(url);
  for (const req of await parts.keys()) {
    if (req.url.startsWith(prefix)) await parts.delete(req.url);
  }
}
