// 模型文件没有大小上限：不再按大小事先拒绝；只有浏览器真正分配不下时才给出说明。
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { fileTooLargeMessage } from '../web/src/image/model-files.js';

test('没有单文件大小上限：model-files 不再导出上限常量或预检查函数', async () => {
  const mod = await import('../web/src/image/model-files.js');
  assert.equal('MAX_BROWSER_FILE_BYTES' in mod, false);
  assert.equal('oversizedFiles' in mod, false);
  const worker = fs.readFileSync(new URL('../web/src/image/sd15-worker.js', import.meta.url), 'utf8');
  assert.doesNotMatch(worker, /超过浏览器单文件上限/);
  assert.doesNotMatch(worker, /oversizedFiles/);
});

test('内存分配失败（RangeError）时给出说明，包含文件地址与大小', () => {
  const msg = fileTooLargeMessage(new RangeError('Array buffer allocation failed'), 'https://h/unet/weights.pb', 3.4e9);
  assert.match(msg, /无法在内存中装下该文件（约 3\.40 GB）/);
  assert.match(msg, /https:\/\/h\/unet\/weights\.pb/);
  assert.match(msg, /sd15_local\.py/);
});

test('其他错误不改写（返回 null）', () => {
  assert.equal(fileTooLargeMessage(new Error('网络错误'), 'u', 10), null);
  assert.match(fileTooLargeMessage(new RangeError('x'), 'u', 0), /无法在内存中装下该文件：u/);
});
