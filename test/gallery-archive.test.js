// 生图画廊导出 / 导入（zip）测试：往返一致、CRC 校验、DEFLATE 读取、路径安全、非法条目跳过。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deflateRawSync } from 'node:zlib';
import { buildStoredZip, crc32, readZip, safeEntryName } from '../web/src/image/zip.js';
import {
  GALLERY_FORMAT,
  buildGalleryArchive,
  isPng,
  readGalleryArchive,
  safePngName,
} from '../web/src/image/gallery-archive.js';

const enc = new TextEncoder();
const dec = new TextDecoder();

/** 一个假的 PNG：正确的文件头加若干字节（不需要是可解码的图片）。 */
function fakePng(seed, n = 64) {
  const out = new Uint8Array(8 + n);
  out.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
  for (let i = 8; i < out.length; i += 1) out[i] = (seed * 31 + i) & 0xff;
  return out;
}

/** 手工构造一个使用 DEFLATE 的单文件 zip（不经过我们的写入代码）。 */
function deflateZip(name, data) {
  const nameBytes = enc.encode(name);
  const comp = deflateRawSync(data);
  const crc = crc32(data);
  const local = new DataView(new ArrayBuffer(30));
  local.setUint32(0, 0x04034b50, true);
  local.setUint16(8, 8, true);
  local.setUint32(14, crc, true);
  local.setUint32(18, comp.length, true);
  local.setUint32(22, data.length, true);
  local.setUint16(26, nameBytes.length, true);
  const central = new DataView(new ArrayBuffer(46));
  central.setUint32(0, 0x02014b50, true);
  central.setUint16(10, 8, true);
  central.setUint32(16, crc, true);
  central.setUint32(20, comp.length, true);
  central.setUint32(24, data.length, true);
  central.setUint16(28, nameBytes.length, true);
  central.setUint32(42, 0, true);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, 1, true);
  end.setUint16(10, 1, true);
  end.setUint32(12, 46 + nameBytes.length, true);
  end.setUint32(16, 30 + nameBytes.length + comp.length, true);
  const parts = [new Uint8Array(local.buffer), nameBytes, comp, new Uint8Array(central.buffer), nameBytes, new Uint8Array(end.buffer)];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

test('zip：写入后读回的文件名、内容一致（含中文与空文件）', async () => {
  const files = [
    { name: 'manifest.json', data: enc.encode('{"a":1}') },
    { name: 'images/001-猫.png', data: fakePng(1) },
    { name: 'empty.txt', data: new Uint8Array(0) },
  ];
  const zip = buildStoredZip(files);
  const back = await readZip(zip);
  assert.deepEqual(back.map((f) => f.name), files.map((f) => f.name));
  for (let i = 0; i < files.length; i += 1) assert.deepEqual(back[i].data, files[i].data);
});

test('zip：CRC32 标准向量', () => {
  assert.equal(crc32(enc.encode('123456789')), 0xcbf43926);
});

test('zip：读取 DEFLATE 压缩的条目', async () => {
  const data = enc.encode('hello hello hello hello 你好');
  const back = await readZip(deflateZip('a/b.txt', data));
  assert.equal(back[0].name, 'a/b.txt');
  assert.equal(dec.decode(back[0].data), 'hello hello hello hello 你好');
});

test('zip：CRC 不一致（内容被改动）时报错', async () => {
  const zip = buildStoredZip([{ name: 'x.bin', data: enc.encode('abcdef') }]);
  const bad = zip.slice();
  bad[30 + 5 + 2] ^= 0xff; // 改动文件内容中的一个字节
  await assert.rejects(readZip(bad), /CRC/);
});

test('zip：不是 zip 的数据报错', async () => {
  await assert.rejects(readZip(enc.encode('not a zip at all')), /不是有效的 zip/);
});

test('zip：路径穿越与绝对路径被拒绝', () => {
  assert.throws(() => safeEntryName('../evil.txt'), /不安全/);
  assert.throws(() => safeEntryName('/etc/passwd'), /不安全/);
  assert.throws(() => safeEntryName('C:/x.txt'), /不安全/);
  assert.throws(() => safeEntryName('a//b'), /不安全/);
  assert.equal(safeEntryName('images\\001.png'), 'images/001.png');
  assert.throws(() => buildStoredZip([{ name: '../x', data: new Uint8Array(1) }]), /不安全/);
});

test('zip：读取时同样拒绝恶意的目录条目名', async () => {
  const zip = deflateZip('../../evil.txt', enc.encode('x'));
  await assert.rejects(readZip(zip), /不安全/);
});

test('画廊：导出后再导入，提示词、尺寸、种子与图片字节完全一致', async () => {
  const items = [
    { data: fakePng(1), filename: 'sd15-1-512x512.png', prompt: 'a cat, 水彩', width: 512, height: 512, steps: 20, guidance: 7.5, seed: 42, model: 'sd15', createdAt: 1700000000000 },
    { data: fakePng(2), filename: '图 2?.png', prompt: '城市夜景', width: 768, height: 432, steps: 25, guidance: 6, seed: 4294967295, model: 'sd15', createdAt: 1700000001000 },
  ];
  const zip = buildGalleryArchive(items, { now: new Date('2026-10-23T00:00:00Z') });
  const { items: back, skipped } = await readGalleryArchive(zip);
  assert.deepEqual(skipped, []);
  assert.equal(back.length, 2);
  for (let i = 0; i < 2; i += 1) {
    assert.deepEqual(back[i].data, items[i].data);
    for (const k of ['prompt', 'width', 'height', 'steps', 'guidance', 'seed', 'model', 'createdAt']) {
      assert.equal(back[i][k], items[i][k], k);
    }
  }
  assert.match(back[1].filename, /\.png$/);
});

test('画廊：清单格式与版本校验', async () => {
  const notGallery = buildStoredZip([{ name: 'a.txt', data: enc.encode('x') }]);
  await assert.rejects(readGalleryArchive(notGallery), /缺少 manifest/);
  const wrongFormat = buildStoredZip([{ name: 'manifest.json', data: enc.encode(JSON.stringify({ format: 'other', version: 1, items: [] })) }]);
  await assert.rejects(readGalleryArchive(wrongFormat), /不是本应用导出的画廊/);
  const future = buildStoredZip([{ name: 'manifest.json', data: enc.encode(JSON.stringify({ format: GALLERY_FORMAT, version: 99, items: [] })) }]);
  await assert.rejects(readGalleryArchive(future), /高于当前支持的版本/);
});

test('画廊：缺失、非 PNG、尺寸无效的条目被跳过并说明原因，其余照常导入', async () => {
  const manifest = {
    format: GALLERY_FORMAT,
    version: 1,
    items: [
      { file: 'images/001.png', prompt: 'ok', width: 512, height: 512, steps: 20, guidance: 7.5, seed: 1, model: 'm', createdAt: 1 },
      { file: 'images/002.png', prompt: 'missing', width: 512, height: 512 },
      { file: 'images/003.png', prompt: 'not png', width: 512, height: 512 },
      { file: 'images/004.png', prompt: 'bad size', width: 0, height: 512 },
      { file: '../escape.png', prompt: 'escape', width: 512, height: 512 },
    ],
  };
  const zip = buildStoredZip([
    { name: 'manifest.json', data: enc.encode(JSON.stringify(manifest)) },
    { name: 'images/001.png', data: fakePng(5) },
    { name: 'images/003.png', data: enc.encode('plain text, not a png') },
    { name: 'images/004.png', data: fakePng(6) },
  ].map((f) => (f.name.includes('..') ? f : f)));
  const { items, skipped } = await readGalleryArchive(zip);
  assert.equal(items.length, 1);
  assert.equal(items[0].prompt, 'ok');
  assert.equal(skipped.length, 4);
  const reasons = Object.fromEntries(skipped.map((s) => [s.file, s.reason]));
  assert.match(reasons['images/002.png'], /缺少/);
  assert.match(reasons['images/003.png'], /PNG/);
  assert.match(reasons['images/004.png'], /尺寸/);
  assert.match(reasons['../escape.png'], /不安全/);
});

test('画廊：文件名清理与 PNG 文件头判断', () => {
  assert.equal(safePngName('a b/c?.PNG'), 'a_b_c_.png');
  assert.equal(safePngName(''), 'image.png');
  assert.equal(isPng(fakePng(1)), true);
  assert.equal(isPng(enc.encode('GIF89a')), false);
});
