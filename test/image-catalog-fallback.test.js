// 文本编码器备选文件（fp32）：作为 fp16 模型的正式文件下载、缓存、离线读取。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileUrl, imageModelsOf, isSelectable, sourceSizes, sourceUrls, totalBytes } from '../web/src/image/catalog.js';

const here = path.dirname(fileURLToPath(import.meta.url));

const cfg = JSON.parse(fs.readFileSync(path.join(here, '..', 'web', 'public', 'config.json'), 'utf-8'));
const sd15 = imageModelsOf(cfg).find((m) => m.id === 'sd15-nmkd-fp16');
const fp32 = imageModelsOf(cfg).find((m) => m.id === 'sd15-nmkd-fp32');

test('备选文件是 fp16 模型的正式文件：大小与 Hugging Face 元数据一致，并计入总大小', () => {
  const fb = sd15.files.textEncoderFallback;
  assert.equal(fb.repo, 'nmkd/stable-diffusion-1.5-onnx');
  assert.equal(fb.path, 'text_encoder/model.onnx');
  assert.equal(fb.size, 492582967);
  assert.equal(totalBytes(sd15), Object.values(sd15.files).reduce((n, f) => n + f.size, 0));
});

test('备选文件的地址指向其自身仓库（不是 fp16 仓库）', () => {
  assert.equal(sourceUrls(sd15).textEncoderFallback, 'https://huggingface.co/nmkd/stable-diffusion-1.5-onnx/resolve/main/text_encoder/model.onnx');
  assert.equal(sourceSizes(sd15).textEncoderFallback, 492582967);
  assert.equal(sourceUrls(sd15).textEncoder, fileUrl(sd15, 'text_encoder/model.onnx'));
});

test('其余文件仍使用模型自己的仓库', () => {
  assert.match(sourceUrls(sd15).unetWeights, /^https:\/\/huggingface\.co\/nmkd\/stable-diffusion-1\.5-onnx-fp16\/resolve\/main\/unet\/weights\.pb$/);
});

test('fp32 模型不需要备选文件（它的文本编码器本身就是 fp32）', () => {
  assert.equal(fp32.files.textEncoderFallback, undefined);
  assert.equal(isSelectable(fp32), true);
});
