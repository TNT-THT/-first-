/**
 * GGUF 文件头解析（只读元数据，不碰权重张量）。
 *
 * 用途：数出模型有多少层（<架构>.block_count），这样界面可以告诉用户
 * “每层约占多少显存”，加载失败疑似显存不足时也能有依据地自动回退层数。
 *
 * 关键点：真实 GGUF 的元数据区不小——分词器词表（`tokenizer.ggml.tokens`）
 * 就有十几万个字符串、几 MB 到几十 MB。以前一次性只读前 1 MB，解析到词表数组
 * 就越界返回 null，于是超大模型（词表更大）几乎总是“层数未知”，界面上算不出
 * 每层显存。现在改成分块按需读取：解析到哪读到哪，不设硬上限，也不会把整个
 * 元数据读进内存。解析不完整或不是 GGUF 时返回 null，调用方按“未知”处理。
 */
import { open } from 'node:fs/promises';

// GGUF 元数据值类型（见 ggml GGUF 规范）
const T = { UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12 };
const FIXED_SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

/** 元数据解析的保护上限：KV 数与字符串长度异常时立即放弃，避免读到坏的/伪造的文件时死循环。 */
const MAX_KV = 1_000_000;
const MAX_STRING = 256 * 1024 * 1024;
const MAX_ARRAY = 500_000_000;

/**
 * 顺序读取器：内部按块从文件读取，对外提供 read(n) / skip(n)。
 * 只为解析 GGUF 头服务，不持有整份文件。
 */
class ByteReader {
  constructor(handle, { chunkBytes = 1024 * 1024, limitBytes = Infinity } = {}) {
    this.handle = handle;
    this.chunkBytes = chunkBytes;
    this.limitBytes = limitBytes;
    this.buf = Buffer.alloc(0);
    this.pos = 0; // 在 buf 内的偏移
    this.base = 0; // buf 起点在文件中的偏移
    this.eof = false;
  }

  get fileOffset() {
    return this.base + this.pos;
  }

  async _fill(need) {
    if (this.buf.length - this.pos >= need) return true;
    // 新数据必须从“缓冲区已覆盖到的文件位置”往后读：base+pos 是已消费位置，
    // 直接从那里读会把 buf 里还没消费的尾部再读一遍（重复数据 → 解析错位）。
    const at = this.base + this.buf.length;
    const have = this.buf.length - this.pos;
    const want = Math.max(need - have, this.chunkBytes);
    if (at >= this.limitBytes) return false;
    const size = Math.min(want, this.limitBytes - at);
    if (size <= 0) return false;
    const chunk = Buffer.alloc(size);
    let bytesRead = 0;
    try {
      ({ bytesRead } = await this.handle.read(chunk, 0, size, at));
    } catch {
      return false;
    }
    if (bytesRead <= 0) {
      this.eof = true;
      return false;
    }
    // 丢掉已消费的部分，只保留未读尾部 + 新读到的数据。
    // 新缓冲区从“尾部第一个字节”开始，所以 pos 归零、base 前移尾部长度。
    const rest = this.buf.subarray(this.pos);
    this.buf = Buffer.concat([rest, chunk.subarray(0, bytesRead)]);
    this.base = at - rest.length;
    this.pos = 0;
    return this.buf.length >= need;
  }

  async read(n) {
    if (n === 0) return Buffer.alloc(0);
    if (!(await this._fill(n))) return null;
    const out = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return out;
  }

  async skip(n) {
    if (n <= 0) return true;
    if (n <= this.buf.length - this.pos) {
      this.pos += n;
      return true;
    }
    // 大跨度（如几 MB 的词表数组）：直接移动文件游标，不把数据读进内存
    const remaining = this.buf.length - this.pos;
    const need = n - remaining;
    const at = this.base + this.pos + remaining;
    if (at + need > this.limitBytes) return false;
    this.pos = this.buf.length;
    this.base = at;
    // 探测目标位置是否在文件内：读 1 字节（EOF 之前都算成功）
    if (need > 0) {
      const probe = Buffer.alloc(1);
      try {
        const { bytesRead } = await this.handle.read(probe, 0, 1, at + need - 1);
        if (bytesRead <= 0) {
          this.eof = true;
          return false;
        }
      } catch {
        return false;
      }
    }
    // 游标跳过后，之前的 buf 数据作废，从头缓冲
    this.buf = Buffer.alloc(0);
    this.base = at + need;
    return true;
  }
}

/**
 * 从一段已读入内存的 GGUF 头字节里找 <架构>.block_count（层数）。找不到返回 null。
 * 纯函数，便于测试。只适合“整个头都在这一段字节里”的情况（测试与小文件）；
 * 大文件请用 readGgufBlockCount，它会按需分块读取。
 */
export function parseGgufBlockCount(buf) {
  const r = new SyncReader(buf);
  return parseBlockCount(r);
}

/** 内存字节数组上的同步读取器（供 parseGgufBlockCount 用）。 */
class SyncReader {
  constructor(buf) {
    this.buf = Buffer.isBuffer(buf) ? buf : Buffer.from(buf || []);
    this.pos = 0;
  }

  read(n) {
    if (this.pos + n > this.buf.length) return null;
    const out = this.buf.subarray(this.pos, this.pos + n);
    this.pos += n;
    return out;
  }

  skip(n) {
    if (n < 0 || this.pos + n > this.buf.length) return false;
    this.pos += n;
    return true;
  }
}

function u64(r, v) {
  const b = r.read(8);
  return b ? Number(b.readBigUInt64LE(0)) : null;
}

function readStr(r) {
  const len = u64(r);
  if (len === null || !Number.isSafeInteger(len) || len < 0 || len > MAX_STRING) return null;
  const b = r.read(len);
  return b ? b.toString('utf8') : null;
}

/** 跳过（或读出）一个元数据值；skipValue 返回数值型值本身，其余返回 true / false。 */
function skipValue(r, type) {
  if (type === T.STRING) return readStr(r) !== null;
  if (type === T.ARRAY) {
    const head = r.read(12);
    if (!head) return false;
    const elemType = head.readUInt32LE(0);
    const count = Number(head.readBigUInt64LE(4));
    if (!Number.isSafeInteger(count) || count < 0 || count > MAX_ARRAY) return false;
    if (elemType === T.STRING) {
      for (let i = 0; i < count; i += 1) if (readStr(r) === null) return false;
      return true;
    }
    const size = FIXED_SIZE[elemType];
    if (!size) return false;
    return r.skip(size * count) !== false;
  }
  const size = FIXED_SIZE[type];
  if (!size) return false;
  const b = r.read(size);
  if (!b) return false;
  if (type === T.UINT32) return b.readUInt32LE(0);
  if (type === T.INT32) return b.readInt32LE(0);
  if (type === T.UINT64) return Number(b.readBigUInt64LE(0));
  if (type === T.INT64) return Number(b.readBigInt64LE(0));
  if (type === T.UINT16) return b.readUInt16LE(0);
  if (type === T.INT16) return b.readInt16LE(0);
  if (type === T.UINT8) return b.readUInt8(0);
  if (type === T.INT8) return b.readInt8(0);
  if (type === T.BOOL) return b.readUInt8(0) !== 0;
  if (type === T.FLOAT32) return b.readFloatLE(0);
  if (type === T.FLOAT64) return b.readDoubleLE(0);
  return true;
}

/** 解析头部并返回 block_count；reader 只需实现 read(n)/skip(n)（同步）。 */
function parseBlockCount(r) {
  const head = r.read(24);
  if (!head || head.length < 24) return null;
  if (head.toString('latin1', 0, 4) !== 'GGUF') return null;
  const version = head.readUInt32LE(4);
  if (version !== 2 && version !== 3) return null;
  const kvCount = Number(head.readBigUInt64LE(16));
  if (!Number.isSafeInteger(kvCount) || kvCount < 0 || kvCount > MAX_KV) return null;
  for (let i = 0; i < kvCount; i += 1) {
    const key = readStr(r);
    if (key === null) return null;
    const tb = r.read(4);
    if (!tb) return null;
    const type = tb.readUInt32LE(0);
    const v = skipValue(r, type);
    if (v === false) return null;
    if (key.endsWith('.block_count')) {
      const n = typeof v === 'number' ? v : null;
      return Number.isInteger(n) && n > 0 && n < 100000 ? n : null;
    }
  }
  return null;
}

/**
 * 读取 GGUF 文件开头的元数据并返回层数；读不到返回 null（不抛错）。
 * 分块按需读取：词表数组再大也能越过，不会因为“头太大”而放弃。
 * maxBytes 只是“最多往后读多少”的保护上限，默认 512MB（任何正常 GGUF 的
 * 元数据都远小于此；超过说明文件损坏，直接按未知处理）。
 */
export async function readGgufBlockCount(filePath, { maxBytes = 512 * 1024 * 1024, chunkBytes = 1024 * 1024 } = {}) {
  let handle = null;
  try {
    handle = await open(filePath, 'r');
    const reader = new ByteReader(handle, { chunkBytes, limitBytes: maxBytes });
    return await parseBlockCountAsync(reader);
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => {});
  }
}

/** ByteReader 是异步的：把 parseBlockCount 的逻辑跑在异步读取器上。 */
async function parseBlockCountAsync(r) {
  const head = await r.read(24);
  if (!head || head.length < 24) return null;
  if (head.toString('latin1', 0, 4) !== 'GGUF') return null;
  const version = head.readUInt32LE(4);
  if (version !== 2 && version !== 3) return null;
  const kvCount = Number(head.readBigUInt64LE(16));
  if (!Number.isSafeInteger(kvCount) || kvCount < 0 || kvCount > MAX_KV) return null;

  const readStrAsync = async () => {
    const lb = await r.read(8);
    if (!lb) return null;
    const len = Number(lb.readBigUInt64LE(0));
    if (!Number.isSafeInteger(len) || len < 0 || len > MAX_STRING) return null;
    const b = await r.read(len);
    return b ? b.toString('utf8') : null;
  };
  const skipValueAsync = async (type) => {
    if (type === T.STRING) return (await readStrAsync()) !== null;
    if (type === T.ARRAY) {
      const h = await r.read(12);
      if (!h) return false;
      const elemType = h.readUInt32LE(0);
      const count = Number(h.readBigUInt64LE(4));
      if (!Number.isSafeInteger(count) || count < 0 || count > MAX_ARRAY) return false;
      if (elemType === T.STRING) {
        for (let i = 0; i < count; i += 1) if ((await readStrAsync()) === null) return false;
        return true;
      }
      const size = FIXED_SIZE[elemType];
      if (!size) return false;
      return r.skip(size * count) !== false;
    }
    const size = FIXED_SIZE[type];
    if (!size) return false;
    const b = await r.read(size);
    if (!b) return false;
    if (type === T.UINT32) return b.readUInt32LE(0);
    if (type === T.INT32) return b.readInt32LE(0);
    if (type === T.UINT64) return Number(b.readBigUInt64LE(0));
    if (type === T.INT64) return Number(b.readBigInt64LE(0));
    if (type === T.UINT16) return b.readUInt16LE(0);
    if (type === T.INT16) return b.readInt16LE(0);
    if (type === T.UINT8) return b.readUInt8(0);
    if (type === T.INT8) return b.readInt8(0);
    if (type === T.BOOL) return b.readUInt8(0) !== 0;
    if (type === T.FLOAT32) return b.readFloatLE(0);
    if (type === T.FLOAT64) return b.readDoubleLE(0);
    return true;
  };

  for (let i = 0; i < kvCount; i += 1) {
    const key = await readStrAsync();
    if (key === null) return null;
    const tb = await r.read(4);
    if (!tb) return null;
    const type = tb.readUInt32LE(0);
    const v = await skipValueAsync(type);
    if (v === false) return null;
    if (key.endsWith('.block_count')) {
      const n = typeof v === 'number' ? v : null;
      return Number.isInteger(n) && n > 0 && n < 100000 ? n : null;
    }
  }
  return null;
}
