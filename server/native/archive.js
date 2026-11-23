/**
 * 最小归档解压（zip / tar.gz），供 llama.cpp 运行库下载与独立软件解压使用。
 *
 * 只实现需要的子集：zip 的 stored/deflate 条目、ustar tar（含 GNU 长文件名 'L'）。
 * 解压时做路径穿越防护：拒绝 `..`、绝对路径与盘符路径。
 */
import { gunzipSync, inflateRawSync } from 'node:zlib';
import { mkdir, writeFile, chmod } from 'node:fs/promises';
import path from 'node:path';

export function safeJoin(base, name) {
  const p = String(name || '').replace(/\\/g, '/');
  if (!p || p.startsWith('/') || /^[a-zA-Z]:/.test(p) || p.split('/').includes('..')) {
    throw new Error(`归档中包含不安全路径：${name}`);
  }
  return path.join(base, ...p.split('/').filter(Boolean));
}

/** 解析 zip 中央目录，返回 [{name, method, compressedSize, size, localOffset}]。 */
export function listZipEntries(buf) {
  if (buf.length < 22) throw new Error('zip 文件过小或已损坏。');
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 65536); i -= 1) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('找不到 zip 中央目录（文件可能不完整）。');
  const count = buf.readUInt16LE(eocd + 10);
  let off = buf.readUInt32LE(eocd + 16);
  const entries = [];
  for (let n = 0; n < count; n += 1) {
    if (buf.readUInt32LE(off) !== 0x02014b50) throw new Error('zip 中央目录损坏。');
    const method = buf.readUInt16LE(off + 10);
    const compressedSize = buf.readUInt32LE(off + 20);
    const size = buf.readUInt32LE(off + 24);
    const nameLen = buf.readUInt16LE(off + 28);
    const extraLen = buf.readUInt16LE(off + 30);
    const commentLen = buf.readUInt16LE(off + 32);
    const localOffset = buf.readUInt32LE(off + 42);
    const name = buf.toString('utf8', off + 46, off + 46 + nameLen);
    entries.push({ name, method, compressedSize, size, localOffset });
    off += 46 + nameLen + extraLen + commentLen;
  }
  return entries;
}

export function extractZipEntry(buf, entry) {
  const off = entry.localOffset;
  if (buf.readUInt32LE(off) !== 0x04034b50) throw new Error(`zip 条目损坏：${entry.name}`);
  const nameLen = buf.readUInt16LE(off + 26);
  const extraLen = buf.readUInt16LE(off + 28);
  const start = off + 30 + nameLen + extraLen;
  const data = buf.subarray(start, start + entry.compressedSize);
  if (entry.method === 0) return Buffer.from(data);
  if (entry.method === 8) return inflateRawSync(data);
  throw new Error(`不支持的 zip 压缩方式 ${entry.method}：${entry.name}`);
}

/** 把 zip 解压到目录；返回写出的相对路径列表。 */
export async function extractZip(buf, destDir, { maxBytes = 2 * 1024 * 1024 * 1024 } = {}) {
  const entries = listZipEntries(buf);
  const written = [];
  let total = 0;
  for (const entry of entries) {
    if (entry.name.endsWith('/')) continue;
    total += entry.size;
    if (total > maxBytes) throw new Error('归档解压后体积超过上限，已中止。');
    const data = extractZipEntry(buf, entry);
    const target = safeJoin(destDir, entry.name);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, data);
    written.push(entry.name);
  }
  return written;
}

/** 解析 tar（已 gunzip 的 Buffer），返回 [{name, size, typeflag, mode, offset}]。 */
export function listTarEntries(buf) {
  const entries = [];
  let off = 0;
  let longName = '';
  while (off + 512 <= buf.length) {
    const header = buf.subarray(off, off + 512);
    if (header.every((b) => b === 0)) break;
    let name = header.toString('utf8', 0, 100).replace(/\0.*$/, '');
    const prefix = header.toString('utf8', 345, 500).replace(/\0.*$/, '');
    if (prefix) name = `${prefix}/${name}`;
    if (longName) {
      name = longName;
      longName = '';
    }
    const sizeText = header.toString('utf8', 124, 136).replace(/\0.*$/, '').trim();
    const size = sizeText ? parseInt(sizeText, 8) : 0;
    const typeflag = String.fromCharCode(header[156] || 48);
    const modeText = header.toString('utf8', 100, 108).replace(/\0.*$/, '').trim();
    const mode = modeText ? parseInt(modeText, 8) : 0o644;
    if (typeflag === 'L') {
      longName = buf.toString('utf8', off + 512, off + 512 + size).replace(/\0.*$/, '');
    } else if (typeflag === '0' || typeflag === '' || typeflag === '5') {
      entries.push({ name, size, typeflag, mode, dataOffset: off + 512 });
    }
    off += 512 + Math.ceil(size / 512) * 512;
  }
  return entries;
}

/** 把 tar.gz 解压到目录；返回写出的相对路径列表。 */
export async function extractTarGz(buf, destDir, { maxBytes = 2 * 1024 * 1024 * 1024 } = {}) {
  const tar = gunzipSync(buf);
  const entries = listTarEntries(tar);
  const written = [];
  let total = 0;
  for (const entry of entries) {
    if (entry.typeflag === '5' || entry.name.endsWith('/')) continue;
    total += entry.size;
    if (total > maxBytes) throw new Error('归档解压后体积超过上限，已中止。');
    const target = safeJoin(destDir, entry.name);
    await mkdir(path.dirname(target), { recursive: true });
    await writeFile(target, tar.subarray(entry.dataOffset, entry.dataOffset + entry.size));
    if (entry.mode & 0o111) await chmod(target, 0o755).catch(() => {});
    written.push(entry.name);
  }
  return written;
}
