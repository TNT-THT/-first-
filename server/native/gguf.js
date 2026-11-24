/**
 * GGUF 文件头解析（只读元数据，不碰权重张量）。
 *
 * 用途：数出模型有多少层（<架构>.block_count），这样界面可以告诉用户
 * “每层约占多少显存”，加载失败疑似显存不足时也能有依据地自动回退层数。
 * 只读取文件开头一段字节；解析不完整或不是 GGUF 时返回 null，调用方按“未知”处理。
 */
import { open } from 'node:fs/promises';

// GGUF 元数据值类型（见 ggml GGUF 规范）
const T = { UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12 };
const FIXED_SIZE = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

/**
 * 从一段 GGUF 文件头字节里找 <架构>.block_count（层数）。找不到返回 null。
 * 纯函数，便于测试。
 */
export function parseGgufBlockCount(buf) {
  if (!Buffer.isBuffer(buf) || buf.length < 24) return null;
  if (buf.toString('latin1', 0, 4) !== 'GGUF') return null;
  let off = 4;
  const version = buf.readUInt32LE(off); off += 4;
  if (version !== 2 && version !== 3) return null;
  // tensor_count 与 metadata_kv_count 都是 uint64；层数只需后者，前者跳过
  if (off + 16 > buf.length) return null;
  off += 8;
  const kvCount = Number(buf.readBigUInt64LE(off)); off += 8;
  if (!Number.isSafeInteger(kvCount) || kvCount < 0 || kvCount > 100000) return null;

  const readStr = () => {
    if (off + 8 > buf.length) return null;
    const len = Number(buf.readBigUInt64LE(off)); off += 8;
    if (!Number.isSafeInteger(len) || len < 0 || len > 1024 * 1024 || off + len > buf.length) return null;
    const s = buf.toString('utf8', off, off + len); off += len;
    return s;
  };
  const skipValue = (type) => {
    if (type === T.STRING) return readStr() !== null;
    if (type === T.ARRAY) {
      if (off + 12 > buf.length) return false;
      const elemType = buf.readUInt32LE(off); off += 4;
      const count = Number(buf.readBigUInt64LE(off)); off += 8;
      if (!Number.isSafeInteger(count) || count < 0 || count > 1e9) return false;
      if (elemType === T.STRING) {
        for (let i = 0; i < count; i += 1) if (readStr() === null) return false;
        return true;
      }
      const size = FIXED_SIZE[elemType];
      if (!size || off + size * count > buf.length) return false;
      off += size * count;
      return true;
    }
    const size = FIXED_SIZE[type];
    if (!size || off + size > buf.length) return false;
    if (type === T.UINT32) { const v = buf.readUInt32LE(off); off += size; return v; }
    if (type === T.UINT64) { const v = Number(buf.readBigUInt64LE(off)); off += size; return v; }
    off += size;
    return true;
  };

  for (let i = 0; i < kvCount; i += 1) {
    const key = readStr();
    if (key === null || off + 4 > buf.length) return null;
    const type = buf.readUInt32LE(off); off += 4;
    if (key.endsWith('.block_count')) {
      const v = skipValue(type);
      return typeof v === 'number' && v > 0 ? v : null;
    }
    if (skipValue(type) === false) return null;
  }
  return null;
}

/** 读取 GGUF 文件开头的元数据并返回层数；读不到返回 null（不抛错）。 */
export async function readGgufBlockCount(filePath, { maxBytes = 1024 * 1024 } = {}) {
  let handle = null;
  try {
    handle = await open(filePath, 'r');
    const buf = Buffer.alloc(maxBytes);
    const { bytesRead } = await handle.read(buf, 0, maxBytes, 0);
    return parseGgufBlockCount(buf.subarray(0, bytesRead));
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => {});
  }
}
