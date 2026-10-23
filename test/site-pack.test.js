// 网站包：白名单、排除规则与 zip 生成器（读回校验 CRC 与内容）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { inflateRawSync } from 'node:zlib';
import { buildZip, crc32 } from '../scripts/lib/zip.mjs';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { exclusionReason, selectSiteFiles, MAX_SITE_FILE_BYTES, LAUNCHER_DIR, LAUNCHER_FILES } from '../scripts/site-files.mjs';

// 读回中央目录中每个条目的 Unix 权限位（外部属性高 16 位）
function unixModes(buf) {
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  const count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  const modes = {};
  for (let n = 0; n < count; n++) {
    const nameLen = buf.readUInt16LE(p + 28);
    const ext = buf.readUInt32LE(p + 38);
    modes[buf.toString('utf8', p + 46, p + 46 + nameLen)] = ext >>> 16;
    p += 46 + nameLen;
  }
  return modes;
}

// 读回 zip（只支持本生成器写出的归档），返回 {name: Buffer}
function readZip(buf) {
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  assert.ok(eocd >= 0, '找不到中央目录结束记录');
  const count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  const out = {};
  for (let n = 0; n < count; n++) {
    assert.equal(buf.readUInt32LE(p), 0x02014b50);
    const method = buf.readUInt16LE(p + 10);
    const crc = buf.readUInt32LE(p + 16);
    const compSize = buf.readUInt32LE(p + 20);
    const nameLen = buf.readUInt16LE(p + 28);
    const localOffset = buf.readUInt32LE(p + 42);
    const name = buf.toString('utf8', p + 46, p + 46 + nameLen);
    assert.equal(buf.readUInt32LE(localOffset), 0x04034b50);
    const localName = buf.readUInt16LE(localOffset + 26);
    const dataStart = localOffset + 30 + localName;
    const comp = buf.subarray(dataStart, dataStart + compSize);
    assert.equal(method, 8);
    const data = inflateRawSync(comp);
    assert.equal(crc32(data), crc, `CRC 不一致：${name}`);
    out[name] = data;
    p += 46 + nameLen;
  }
  return out;
}

test('zip：往返读回的文件名、内容与 CRC 一致（含中文与空文件）', () => {
  const zip = buildZip([
    { name: 'a.txt', data: 'hello' },
    { name: 'web/说明.md', data: '你好，世界' },
    { name: 'empty.bin', data: Buffer.alloc(0) },
  ]);
  const files = readZip(zip);
  assert.equal(files['a.txt'].toString('utf8'), 'hello');
  assert.equal(files['web/说明.md'].toString('utf8'), '你好，世界');
  assert.equal(files['empty.bin'].length, 0);
});

test('zip：CRC32 标准向量', () => {
  assert.equal(crc32(Buffer.from('123456789')), 0xcbf43926);
});

test('白名单：只收录网站源码、构建产物与说明文件', () => {
  const { included } = selectSiteFiles([
    { path: 'package.json', size: 10 },
    { path: 'README.md', size: 10 },
    { path: 'web/index.html', size: 10 },
    { path: 'web/src/main.js', size: 10 },
    { path: 'web/public/config.json', size: 10 },
    { path: 'web/dist/index.html', size: 10 },
    { path: 'server/index.js', size: 10 },
    { path: 'Dockerfile', size: 10 },
    { path: 'training/models/base/model.bin', size: 10 },
  ]);
  assert.deepEqual(included, ['README.md', 'package.json', 'web/dist/index.html', 'web/index.html', 'web/public/config.json', 'web/src/main.js']);
});

test('排除：环境变量、模型与权重、依赖目录、网站包本身、超大文件', () => {
  assert.match(exclusionReason('.env'), /密钥/);
  assert.match(exclusionReason('web/.env.local'), /密钥/);
  assert.match(exclusionReason('web/public/my.ojmodel'), /模型或权重/);
  assert.match(exclusionReason('web/public/model.safetensors'), /模型或权重/);
  assert.match(exclusionReason('web/node_modules/x/index.js'), /依赖或版本库/);
  assert.match(exclusionReason('web/dist/site.zip'), /网站包本身/);
  assert.match(exclusionReason('web/public/big.json', MAX_SITE_FILE_BYTES + 1), /文件过大/);
  assert.equal(exclusionReason('web/src/main.js', 100), null);
  const { included, excluded } = selectSiteFiles([
    { path: 'web/.env', size: 1 },
    { path: 'web/public/x.ojmodel', size: 1 },
    { path: 'web/dist/site.zip', size: 1 },
  ]);
  assert.deepEqual(included, []);
  assert.equal(excluded.length, 3);
});

test('启动文件：源文件存在，且中文文件名与权限位正确写入 zip', () => {
  for (const f of LAUNCHER_FILES) {
    assert.ok(existsSync(join(LAUNCHER_DIR, f.from)), `缺少启动文件：${f.from}`);
  }
  const zip = buildZip(LAUNCHER_FILES.map((f) => ({ name: f.to, data: 'x', mode: f.mode })));
  const modes = unixModes(zip);
  assert.equal(modes['使用说明.txt'], 0o100644);
  assert.equal(modes['start-mac.command'], 0o100755);
  assert.equal(modes['serve-site.py'], 0o100755);
  assert.equal(modes['start-windows.bat'], 0o100644);
  assert.equal(readZip(zip)['使用说明.txt'].toString('utf8'), 'x');
});

test('网站包：生图模块源码在打包范围内，模型文件被排除', () => {
  const entries = [
    { path: 'web/src/image/sd15-worker.js', size: 1000 },
    { path: 'web/src/image/model-files.js', size: 1000 },
    { path: 'web/src/image/gallery-store.js', size: 1000 },
    { path: 'web/src/image/zip.js', size: 1000 },
    { path: 'web/src/image/gallery-archive.js', size: 1000 },
    { path: 'web/dist/assets/sd15-worker.js', size: 1000 },
    { path: 'web/public/models/example.ojimage', size: 1000 },
    { path: 'web/public/models/example.ojmodel', size: 1000 },
  ];
  const { included, excluded } = selectSiteFiles(entries);
  assert.ok(included.includes('web/src/image/sd15-worker.js'));
  assert.ok(included.includes('web/src/image/model-files.js'));
  assert.ok(included.includes('web/src/image/gallery-store.js'));
  assert.ok(included.includes('web/src/image/zip.js'));
  assert.ok(included.includes('web/src/image/gallery-archive.js'));
  assert.ok(included.includes('web/dist/assets/sd15-worker.js'));
  assert.equal(exclusionReason('web/public/models/example.ojimage'), '模型或权重文件');
  assert.equal(exclusionReason('web/public/models/example.ojmodel'), '模型或权重文件');
  assert.deepEqual(excluded.map((e) => e.path).sort(), ['web/public/models/example.ojimage', 'web/public/models/example.ojmodel']);
});

test('网站包：超过 20 MB 的运行库（ORT wasm）不打包', () => {
  assert.equal(exclusionReason('web/public/ort/ort-wasm-simd-threaded.jsep.wasm', 28_300_000), '文件过大');
});
