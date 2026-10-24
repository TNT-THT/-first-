// SD 1.5 UNet 计算图补丁：配置指向本站的修正图，大小与文件一致；打包规则只放行这一个图文件。
// 背景：原图的注意力 LayerNorm 使用 float64，浏览器中的 ONNX Runtime 没有 float64 内核（Mul/Pow/ReduceMean 报错）。
// 补丁只把 double 改为 float（48 处 Cast、96 处常量），权重文件不变。见 scripts/image/patch_sd15_unet_fp32_ln.py。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileUrl, localFileUrl, sourceUrls } from '../web/src/image/catalog.js';
import { exclusionReason } from '../scripts/site-files.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(here, '..');
const cfg = JSON.parse(fs.readFileSync(path.join(root, 'web', 'public', 'config.json'), 'utf8'));
const fp16 = cfg.imageModels.find((m) => m.id === 'sd15-nmkd-fp16');
const GRAPH = path.join(root, 'web', 'public', 'sd15-unet', 'model.onnx');

test('sd15-nmkd-fp16 的 UNet 使用本站的修正图，大小与文件一致', () => {
  assert.equal(fp16.files.unet.local, 'sd15-unet/model.onnx');
  assert.equal(fs.statSync(GRAPH).size, fp16.files.unet.size);
  assert.equal(fp16.files.unet.size, 1217320);
});

test('修正图的外部权重仍指向 Hugging Face 上的原始 weights.pb', () => {
  assert.equal(fp16.files.unetWeights.path, 'unet/weights.pb');
  assert.equal(fp16.files.unetWeights.size, 1718976000);
  assert.equal(fp16.files.unetWeights.local, undefined);
  assert.match(fileUrl(fp16, 'unet/weights.pb', fp16.files.unetWeights), /^https:\/\/huggingface\.co\/nmkd\/stable-diffusion-1\.5-onnx-fp16\/resolve\/main\/unet\/weights\.pb$/);
});

test('localFileUrl 按页面地址解析本站文件', () => {
  const prev = globalThis.document;
  globalThis.document = { baseURI: 'https://example.github.io/-first-/' };
  try {
    assert.equal(localFileUrl('sd15-unet/model.onnx'), 'https://example.github.io/-first-/sd15-unet/model.onnx');
    assert.equal(fileUrl(fp16, 'unet/model.onnx', fp16.files.unet), 'https://example.github.io/-first-/sd15-unet/model.onnx');
    assert.equal(sourceUrls(fp16).unet, 'https://example.github.io/-first-/sd15-unet/model.onnx');
  } finally {
    if (prev === undefined) delete globalThis.document;
    else globalThis.document = prev;
  }
});

test('站点打包：只放行修正图，其它 .onnx 仍按模型文件排除', () => {
  assert.equal(exclusionReason('web/public/sd15-unet/model.onnx', 1217320), null);
  assert.match(exclusionReason('web/public/sd15-unet/weights.bin', 100), /模型或权重文件/);
  assert.match(exclusionReason('web/public/other/model.onnx', 100), /模型或权重文件/);
});
