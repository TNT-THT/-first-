import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { oversizedFiles, MAX_BROWSER_FILE_BYTES } from '../web/src/image/model-files.js';
import { sourceSizes } from '../web/src/image/catalog.js';

const config = JSON.parse(fs.readFileSync(new URL('../web/public/config.json', import.meta.url), 'utf8'));
const model = (id) => config.imageModels.find((m) => m.id === id);

test('浏览器单文件上限为 2 GB 减 1 字节', () => {
  assert.equal(MAX_BROWSER_FILE_BYTES, 2147483647);
});

test('fp32 的 UNet 权重（3.4 GB）超过上限，会被识别出来', () => {
  const over = oversizedFiles(sourceSizes(model('sd15-nmkd-fp32')));
  assert.deepEqual(over.map((o) => o.key), ['unetWeights']);
});

test('fp16 的全部文件都在上限以内（最大为 UNet 权重 1.72 GB）', () => {
  assert.deepEqual(oversizedFiles(sourceSizes(model('sd15-nmkd-fp16'))), []);
});

test('缺失或为 0 的大小视为未知，不报错', () => {
  assert.deepEqual(oversizedFiles({ a: undefined, b: 0, c: '' }), []);
  assert.deepEqual(oversizedFiles(null), []);
});

test('按自定义上限判断，边界值本身不超过', () => {
  assert.deepEqual(oversizedFiles({ x: 10, y: 11 }, 10), [{ key: 'y', bytes: 11 }]);
});
