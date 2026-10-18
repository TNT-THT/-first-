/**
 * 最小的 zip 生成器（无第三方依赖）。
 * 每个条目使用 deflate 压缩；只支持不超过 4 GB、文件数不超过 65535 的归档，
 * 足以打包网站源码与构建产物。
 */
import { deflateRawSync } from 'node:zlib';

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32(buf) {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) crc = CRC_TABLE[(crc ^ buf[i]) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

// 固定时间戳（1980-01-01 00:00），使相同输入生成相同的归档
const DOS_TIME = 0;
const DOS_DATE = (0 << 9) | (1 << 5) | 1;

/**
 * @param {Array<{name: string, data: Buffer|Uint8Array|string, mode?: number}>} files 相对路径使用 / 分隔；mode 为 Unix 权限位（默认 0o644，启动脚本用 0o755）
 * @returns {Buffer}
 */
export function buildZip(files) {
  const locals = [];
  const centrals = [];
  let offset = 0;

  for (const file of files) {
    const name = Buffer.from(file.name.replace(/\\/g, '/'), 'utf8');
    const mode = file.mode ?? 0o644;
    const raw = typeof file.data === 'string' ? Buffer.from(file.data, 'utf8') : Buffer.from(file.data);
    const compressed = deflateRawSync(raw, { level: 9 });
    const crc = crc32(raw);

    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0); // 本地文件头签名
    local.writeUInt16LE(20, 4); // 解压所需版本
    local.writeUInt16LE(0x0800, 6); // 通用标志：文件名为 UTF-8
    local.writeUInt16LE(8, 8); // 压缩方法：deflate
    local.writeUInt16LE(DOS_TIME, 10);
    local.writeUInt16LE(DOS_DATE, 12);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(compressed.length, 18);
    local.writeUInt32LE(raw.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);
    const localPart = Buffer.concat([local, name, compressed]);
    locals.push(localPart);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0); // 中央目录签名
    central.writeUInt16LE(0x031e, 4); // 制作版本：Unix 3.0（以便保存权限位）
    central.writeUInt16LE(20, 6); // 解压所需版本
    central.writeUInt16LE(0x0800, 8);
    central.writeUInt16LE(8, 10);
    central.writeUInt16LE(DOS_TIME, 12);
    central.writeUInt16LE(DOS_DATE, 14);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(compressed.length, 20);
    central.writeUInt32LE(raw.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt16LE(0, 30); // 额外字段长度
    central.writeUInt16LE(0, 32); // 注释长度
    central.writeUInt16LE(0, 34); // 磁盘编号
    central.writeUInt16LE(0, 36); // 内部属性
    central.writeUInt32LE((0o100000 | mode) * 0x10000 >>> 0, 38); // 外部属性：Unix 权限位在高 16 位
    central.writeUInt32LE(offset, 42); // 本地头的偏移
    centrals.push(Buffer.concat([central, name]));

    offset += localPart.length;
  }

  const centralBuf = Buffer.concat(centrals);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); // 中央目录结束签名
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(files.length, 8);
  end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(centralBuf.length, 12);
  end.writeUInt32LE(offset, 16);
  end.writeUInt16LE(0, 20);

  return Buffer.concat([...locals, centralBuf, end]);
}
