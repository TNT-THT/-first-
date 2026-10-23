/**
 * 浏览器中的最小 zip 读写（纯逻辑，无第三方依赖）。
 *
 * - 写入：只使用 STORE（不压缩）。生图导出的 PNG 本身已经压缩，再压缩几乎没有收益。
 * - 读取：支持 STORE 与 DEFLATE（通过 DecompressionStream('deflate-raw')），校验 CRC32 与文件名。
 * - 不支持：加密、ZIP64（超过 4 GB 或超过 65535 个条目）、分卷。遇到时报错，而不是静默读错。
 * - 文件名必须是相对路径：拒绝绝对路径、反斜杠以外的盘符，以及包含 .. 的路径（防止路径穿越）。
 */

const encoder = new TextEncoder();
const decoder = new TextDecoder();

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/** 文件名的安全检查：返回规范化后的名称；不安全时抛出错误。 */
export function safeEntryName(name) {
  const p = String(name || '').replace(/\\/g, '/');
  if (!p || p.startsWith('/') || /^[A-Za-z]:/.test(p) || p.includes('\0')) {
    throw new Error(`压缩包中的文件路径不安全：${name}`);
  }
  const parts = p.split('/');
  if (parts.some((s) => s === '..' || s === '')) throw new Error(`压缩包中的文件路径不安全：${name}`);
  return parts.join('/');
}

function dosDateTime(date = new Date()) {
  const time = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
  const day = ((date.getFullYear() - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { time, day };
}

/**
 * @param {Array<{name: string, data: Uint8Array}>} files
 * @param {{date?: Date}} [options]
 * @returns {Uint8Array}
 */
export function buildStoredZip(files, { date = new Date() } = {}) {
  if (files.length > 65535) throw new Error('文件数量超过 zip 的上限');
  const { time, day } = dosDateTime(date);
  const locals = [];
  const centrals = [];
  let offset = 0;
  for (const file of files) {
    const name = encoder.encode(safeEntryName(file.name));
    const data = file.data instanceof Uint8Array ? file.data : new Uint8Array(file.data);
    if (data.length > 0xffffffff) throw new Error('单个文件超过 4 GB，无法写入 zip');
    const crc = crc32(data);

    const local = new DataView(new ArrayBuffer(30));
    local.setUint32(0, 0x04034b50, true);
    local.setUint16(4, 20, true);
    local.setUint16(6, 0x0800, true); // 文件名为 UTF-8
    local.setUint16(8, 0, true); // STORE
    local.setUint16(10, time, true);
    local.setUint16(12, day, true);
    local.setUint32(14, crc, true);
    local.setUint32(18, data.length, true);
    local.setUint32(22, data.length, true);
    local.setUint16(26, name.length, true);
    local.setUint16(28, 0, true);
    locals.push(new Uint8Array(local.buffer), name, data);

    const central = new DataView(new ArrayBuffer(46));
    central.setUint32(0, 0x02014b50, true);
    central.setUint16(4, 20, true);
    central.setUint16(6, 20, true);
    central.setUint16(8, 0x0800, true);
    central.setUint16(10, 0, true);
    central.setUint16(12, time, true);
    central.setUint16(14, day, true);
    central.setUint32(16, crc, true);
    central.setUint32(20, data.length, true);
    central.setUint32(24, data.length, true);
    central.setUint16(28, name.length, true);
    central.setUint16(30, 0, true);
    central.setUint16(32, 0, true);
    central.setUint16(34, 0, true);
    central.setUint16(36, 0, true);
    central.setUint32(38, 0o644 << 16 >>> 0, true); // 外部属性：Unix 权限 0644
    central.setUint32(42, offset, true);
    centrals.push(new Uint8Array(central.buffer), name);

    offset += 30 + name.length + data.length;
  }
  const centralSize = centrals.reduce((n, p) => n + p.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);

  const parts = [...locals, ...centrals, new Uint8Array(end.buffer)];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

async function inflateRaw(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * 读取 zip。返回 [{name, data}]。
 * @param {Uint8Array} bytes
 * @param {{maxEntries?: number, maxEntryBytes?: number}} [limits]
 */
export async function readZip(bytes, { maxEntries = 2000, maxEntryBytes = 64 * 1024 * 1024 } = {}) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  // 从末尾向前查找 End Of Central Directory（注释最长 65535 字节）
  let eocd = -1;
  for (let i = buf.length - 22; i >= Math.max(0, buf.length - 22 - 65535); i -= 1) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('不是有效的 zip 文件（找不到目录结构）');
  const count = view.getUint16(eocd + 10, true);
  const centralOffset = view.getUint32(eocd + 16, true);
  if (count === 0xffff || centralOffset === 0xffffffff) throw new Error('暂不支持 ZIP64 格式的压缩包');
  if (count > maxEntries) throw new Error(`压缩包中的文件过多（${count} 个，上限 ${maxEntries}）`);

  const files = [];
  let p = centralOffset;
  for (let n = 0; n < count; n += 1) {
    if (p + 46 > buf.length || view.getUint32(p, true) !== 0x02014b50) throw new Error('压缩包目录损坏');
    const flags = view.getUint16(p + 8, true);
    const method = view.getUint16(p + 10, true);
    const crc = view.getUint32(p + 16, true);
    const compSize = view.getUint32(p + 20, true);
    const size = view.getUint32(p + 24, true);
    const nameLen = view.getUint16(p + 28, true);
    const extraLen = view.getUint16(p + 30, true);
    const commentLen = view.getUint16(p + 32, true);
    const localOffset = view.getUint32(p + 42, true);
    if (flags & 0x0001) throw new Error('暂不支持加密的压缩包');
    if (compSize === 0xffffffff || size === 0xffffffff) throw new Error('暂不支持 ZIP64 格式的压缩包');
    const name = safeEntryName(decoder.decode(buf.subarray(p + 46, p + 46 + nameLen)));
    p += 46 + nameLen + extraLen + commentLen;

    if (size > maxEntryBytes) throw new Error(`压缩包中的文件过大：${name}`);
    if (localOffset + 30 > buf.length || view.getUint32(localOffset, true) !== 0x04034b50) {
      throw new Error(`压缩包中的文件记录损坏：${name}`);
    }
    const lNameLen = view.getUint16(localOffset + 26, true);
    const lExtraLen = view.getUint16(localOffset + 28, true);
    const start = localOffset + 30 + lNameLen + lExtraLen;
    const raw = buf.subarray(start, start + compSize);
    if (raw.length !== compSize) throw new Error(`压缩包数据不完整：${name}`);

    let data;
    if (method === 0) data = raw;
    else if (method === 8) data = await inflateRaw(raw);
    else throw new Error(`不支持的压缩方式（${method}）：${name}`);
    if (data.length !== size) throw new Error(`文件大小与目录不一致：${name}`);
    if (crc32(data) !== crc) throw new Error(`文件校验失败（CRC 不一致）：${name}`);
    files.push({ name, data });
  }
  return files;
}
