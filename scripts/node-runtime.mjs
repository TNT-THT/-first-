/**
 * 网站包内置 Node 运行时：打包时把官方 Node 二进制按平台打进各自的网站包
 * （site-win-x64.zip、site-darwin-arm64.zip 等，每个包的 runtime/ 里只放对应平台的一个），
 * 用户按自己电脑的系统下载一个，解压双击即可使用，不必再单独下载安装 Node.js。
 *
 * - 运行时按平台分开存放，启动器（serve-api.ps1 / start-mac.command）按当前系统选用；
 * - 首次下载后缓存在 .node-runtime-cache/（已 gitignore），之后打包直接复用；
 * - 若某平台下载失败（例如无网络），打包仍会继续并在日志中明确警告：
 *   缺失平台的启动器会回退到系统安装的 Node.js。
 */
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { gunzipSync, inflateRawSync } from 'node:zlib';

export const NODE_RUNTIME_VERSION = process.env.OJ_NODE_VERSION || 'v22.22.3';
export const NODE_RUNTIME_CACHE_DIR = '.node-runtime-cache';

/** 网站包内置的平台运行时（文件名位于 zip 根目录的 runtime/ 下）。 */
export const NODE_RUNTIME_TARGETS = [
  {
    id: 'win-x64',
    file: 'runtime/node-win-x64.exe',
    archive: (v) => `node-${v}-win-x64.zip`,
    inner: (v) => `node-${v}-win-x64/node.exe`,
    mode: 0o644,
  },
  {
    id: 'darwin-arm64',
    file: 'runtime/node-darwin-arm64',
    archive: (v) => `node-${v}-darwin-arm64.tar.gz`,
    inner: (v) => `node-${v}-darwin-arm64/bin/node`,
    mode: 0o755,
  },
  {
    id: 'darwin-x64',
    file: 'runtime/node-darwin-x64',
    archive: (v) => `node-${v}-darwin-x64.tar.gz`,
    inner: (v) => `node-${v}-darwin-x64/bin/node`,
    mode: 0o755,
  },
  {
    id: 'linux-x64',
    file: 'runtime/node-linux-x64',
    archive: (v) => `node-${v}-linux-x64.tar.gz`,
    inner: (v) => `node-${v}-linux-x64/bin/node`,
    mode: 0o755,
  },
];

/** 允许用 OJ_NODE_RUNTIMES=win-x64,darwin-arm64 只打包部分平台（默认全部）。 */
export function selectedNodeRuntimeTargets(env = process.env) {
  const wanted = String(env.OJ_NODE_RUNTIMES || '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (!wanted.length) return NODE_RUNTIME_TARGETS;
  return NODE_RUNTIME_TARGETS.filter((t) => wanted.includes(t.id));
}

/** 从 ZIP 压缩包中取出指定条目（支持 stored / deflate）。 */
export function extractZipEntry(buf, entryName) {
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i -= 1) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('ZIP 格式无效：找不到中央目录');
  const count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  for (let n = 0; n < count; n += 1) {
    if (buf.readUInt32LE(p) !== 0x02014b50) break;
    const method = buf.readUInt16LE(p + 10);
    const compSize = buf.readUInt32LE(p + 20);
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    const localOffset = buf.readUInt32LE(p + 42);
    const name = buf.toString('utf8', p + 46, p + 46 + nameLen);
    if (name === entryName) {
      const localName = buf.readUInt16LE(localOffset + 26);
      const localExtra = buf.readUInt16LE(localOffset + 28);
      const start = localOffset + 30 + localName + localExtra;
      const comp = buf.subarray(start, start + compSize);
      if (method === 0) return Buffer.from(comp);
      if (method === 8) return inflateRawSync(comp);
      throw new Error(`不支持的 ZIP 压缩方式：${method}`);
    }
    p += 46 + nameLen + extraLen + commentLen;
  }
  throw new Error(`ZIP 中找不到文件：${entryName}`);
}

function readTarString(buf, offset, length) {
  const raw = buf.toString('utf8', offset, offset + length);
  const zero = raw.indexOf('\0');
  return zero >= 0 ? raw.slice(0, zero) : raw;
}

/** 从 .tar.gz 中取出指定条目（ustar 格式，官方 Node 发布包即此格式）。 */
export function extractTarGzEntry(buf, entryName) {
  const tar = gunzipSync(buf);
  let offset = 0;
  while (offset + 512 <= tar.length) {
    const header = tar.subarray(offset, offset + 512);
    if (header.every((b) => b === 0)) break;
    const name = readTarString(header, 0, 100);
    const prefix = readTarString(header, 345, 155);
    const fullName = prefix ? `${prefix}/${name}` : name;
    const size = parseInt(readTarString(header, 124, 12).trim() || '0', 8) || 0;
    const type = String.fromCharCode(header[156]);
    const dataStart = offset + 512;
    if ((type === '0' || type === '\0') && fullName === entryName) {
      return Buffer.from(tar.subarray(dataStart, dataStart + size));
    }
    offset = dataStart + Math.ceil(size / 512) * 512;
  }
  throw new Error(`TAR 中找不到文件：${entryName}`);
}

function extractInner(archiveBuf, target, version) {
  return target.archive(version).endsWith('.zip')
    ? extractZipEntry(archiveBuf, target.inner(version))
    : extractTarGzEntry(archiveBuf, target.inner(version));
}

/**
 * 确保每个平台的内置运行时可用：优先读缓存，其次从 nodejs.org 下载并解压出 node 二进制。
 * 返回 { files: [{name, data, mode}], missing: [platformId] }；下载失败只记入 missing，不抛错。
 */
export async function ensureNodeRuntimes({
  projectRoot,
  cacheDir = path.join(projectRoot, NODE_RUNTIME_CACHE_DIR),
  version = NODE_RUNTIME_VERSION,
  targets = selectedNodeRuntimeTargets(),
  offline = process.env.OJ_NODE_RUNTIME_OFFLINE === '1',
  fetchImpl = fetch,
  log = console,
} = {}) {
  const files = [];
  const missing = [];
  for (const target of targets) {
    const cacheFile = path.join(cacheDir, `${version}-${target.id}`);
    if (existsSync(cacheFile)) {
      files.push({ targetId: target.id, name: target.file, data: await readFile(cacheFile), mode: target.mode });
      continue;
    }
    if (offline) {
      missing.push(target.id);
      continue;
    }
    try {
      const url = `https://nodejs.org/dist/${version}/${target.archive(version)}`;
      log.log(`正在下载内置 Node 运行时（${target.id}）：${url}`);
      const res = await fetchImpl(url);
      if (!res?.ok) throw new Error(`HTTP ${res?.status ?? 'unknown'}`);
      const data = extractInner(Buffer.from(await res.arrayBuffer()), target, version);
      await mkdir(cacheDir, { recursive: true });
      await writeFile(cacheFile, data);
      files.push({ targetId: target.id, name: target.file, data, mode: target.mode });
    } catch (err) {
      log.warn(`[warn] 获取 ${target.id} 的内置 Node 运行时失败（${err.message}），该平台将回退到系统 Node。`);
      missing.push(target.id);
    }
  }
  return { files, missing };
}
