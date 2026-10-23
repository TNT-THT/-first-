// 文本编码器备选文件（fp32）的目录逻辑：只在生成时使用，不参与下载、缓存状态与导出。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fallbackFile, imageModelsOf, isSelectable, runtimeSizes, runtimeUrls, sourceUrls, totalBytes } from '../web/src/image/catalog.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(fs.readFileSync(path.join(here, '..', 'web', 'public', 'config.json'), 'utf-8'));
const sd15 = imageModelsOf(cfg).find((m) => m.id === 'sd15-nmkd-fp16');

test('配置：SD 1.5 备选文件为 nmkd 的 fp32 文本编码器，大小与 Hugging Face 元数据一致', () => {
  assert.equal(sd15.textEncoderFallback.repo, 'nmkd/stable-diffusion-1.5-onnx');
  assert.equal(sd15.textEncoderFallback.path, 'text_encoder/model.onnx');
  assert.equal(sd15.textEncoderFallback.size, 492582967);
});

test('备选文件的地址指向其自身仓库', () => {
  assert.deepEqual(fallbackFile(sd15), {
    url: 'https://huggingface.co/nmkd/stable-diffusion-1.5-onnx/resolve/main/text_encoder/model.onnx',
    size: 492582967,
  });
});

test('备选文件不参与下载：sourceUrls 与 totalBytes 不包含它', () => {
  assert.ok(!Object.values(sourceUrls(sd15)).some((u) => u.includes('nmkd/stable-diffusion-1.5-onnx/resolve/')));
  assert.equal(totalBytes(sd15), Object.values(sd15.files).reduce((n, f) => n + f.size, 0));
});

test('生成时：runtimeUrls 与 runtimeSizes 包含备选文件，其余地址与大小不变', () => {
  const urls = runtimeUrls(sd15);
  assert.equal(urls.textEncoderFallback, fallbackFile(sd15).url);
  assert.equal(urls.textEncoder, sourceUrls(sd15).textEncoder);
  assert.equal(runtimeSizes(sd15).textEncoderFallback, 492582967);
  assert.equal(runtimeSizes(sd15).textEncoder, sd15.files.textEncoder.size);
});

test('没有备选文件的模型：runtimeUrls 与 sourceUrls 相同，无 textEncoderFallback', () => {
  const model = { ...sd15, textEncoderFallback: undefined };
  assert.equal(fallbackFile(model), null);
  assert.deepEqual(runtimeUrls(model), sourceUrls(model));
  assert.equal('textEncoderFallback' in runtimeSizes(model), false);
});

test('fp32 版本：独立条目，可选择，地址指向 nmkd 仓库的 fp32 文件，fp16 条目不变', () => {
  const fp32 = imageModelsOf(cfg).find((m) => m.id === 'sd15-nmkd-fp32');
  assert.ok(fp32, '缺少 sd15-nmkd-fp32 条目');
  assert.equal(isSelectable(fp32), true);
  assert.equal(fp32.repo, 'nmkd/stable-diffusion-1.5-onnx');
  for (const key of ['textEncoder', 'unet', 'unetWeights', 'vaeDecoder', 'vocab', 'merges', 'scheduler']) {
    assert.ok(fp32.files[key].size > 0, `${key} 大小无效`);
  }
  assert.equal(fp32.files.unetWeights.size, 3438083840);
  assert.equal(totalBytes(fp32), 4131528349);
  assert.match(sourceUrls(fp32).textEncoder, /^https:\/\/huggingface\.co\/nmkd\/stable-diffusion-1\.5-onnx\/resolve\/main\/text_encoder\/model\.onnx$/);
  assert.equal(sd15.repo, 'nmkd/stable-diffusion-1.5-onnx-fp16');
  assert.equal(isSelectable(sd15), true);
  assert.equal(sd15.files.textEncoder.size, 246476214);
});
