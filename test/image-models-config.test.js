// 生图模型配置检查：新加入的 SD 1.x 结构模型（BK-SDM、Tiny SD、Small SD、SDXS）的文件清单、大小与地址。
// 只检查配置本身；真实文件的接口与画质需在浏览器中测试（见各条目的 note）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { imageModelsOf, isSelectable, sourceSizes, sourceUrls, totalBytes } from '../web/src/image/catalog.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const cfg = JSON.parse(fs.readFileSync(path.join(here, '..', 'web', 'public', 'config.json'), 'utf8'));
const models = imageModelsOf(cfg);
const NEW = ['bk-sdm-small-2m-onnx-webgpu', 'tiny-sd-onnx-webgpu', 'small-sd-onnx-webgpu', 'sdxs-512-dreamshaper-onnx-webgpu'];

test('新模型条目：可选、有完整的 SD 1.x 文件清单（无外部权重）', () => {
  for (const id of NEW) {
    const m = models.find((x) => x.id === id);
    assert.ok(m, `缺少 ${id}`);
    assert.ok(isSelectable(m), `${id} 应为 available`);
    for (const key of ['textEncoder', 'unet', 'vaeDecoder', 'vocab', 'merges', 'scheduler']) {
      assert.ok(m.files[key]?.path, `${id} 缺少 ${key}`);
      assert.ok(m.files[key].size > 0, `${id} 的 ${key} 缺少大小`);
    }
    assert.equal(m.files.unetWeights, undefined, `${id} 不应有外部权重`);
    assert.ok(totalBytes(m) < 4e9, `${id} 总大小异常`);
  }
});

test('新模型条目：地址指向各自的仓库与标准路径', () => {
  const m = models.find((x) => x.id === 'sdxs-512-dreamshaper-onnx-webgpu');
  const urls = sourceUrls(m);
  assert.equal(urls.unet, 'https://huggingface.co/danyalmalik/sdxs-512-dreamshaper-ONNX-WebGPU/resolve/main/unet/model.onnx');
  assert.equal(urls.scheduler, 'https://huggingface.co/danyalmalik/sdxs-512-dreamshaper-ONNX-WebGPU/resolve/main/scheduler/scheduler_config.json');
  assert.equal(m.latentScale, 1);
  assert.deepEqual(m.defaults, { width: 512, height: 512, steps: 1, guidance: 1 });
});

test('每个新模型的 note 都说明了未在浏览器中验证', () => {
  for (const id of NEW) {
    const m = models.find((x) => x.id === id);
    assert.match(m.note, /未核对：真实 ONNX/);
  }
});

test('SDXS 之外的新模型使用 SD 1.5 的潜变量缩放（缺省）', () => {
  for (const id of NEW.filter((x) => !x.startsWith('sdxs'))) {
    const m = models.find((x) => x.id === id);
    assert.equal(m.latentScale, undefined, id);
  }
});

// 三个新加入的动漫模型：只在选择器中显示，暂不可选、不下载；备注说明暂不可用的原因。
const ANIME_UNAVAILABLE = ['wd14-anime-sd15-onnx-fp16', 'cyberpunk-anime-sd15-onnx-fp32', 'wai-illustrious-sdxl-onnx-webgpu'];

test('动漫模型条目：状态为 unavailable，不可选，备注写明暂不可用的原因', () => {
  for (const id of ANIME_UNAVAILABLE) {
    const m = models.find((x) => x.id === id);
    assert.ok(m, `缺少 ${id}`);
    assert.equal(m.status, 'unavailable', id);
    assert.equal(isSelectable(m), false, `${id} 不应可选`);
    assert.match(m.note, /暂不可用/, `${id} 备注应说明暂不可用`);
    assert.match(m.note, /未在浏览器中验证/, `${id} 备注应说明未验证`);
  }
});

test('WD 1.4 动漫模型：单文件 UNet 无外部权重，文件大小均已配置', () => {
  const m = models.find((x) => x.id === 'wd14-anime-sd15-onnx-fp16');
  assert.equal(m.files.unetWeights, undefined);
  for (const key of ['textEncoder', 'unet', 'vaeDecoder', 'vocab', 'merges', 'scheduler']) {
    assert.ok(m.files[key]?.size > 0, `缺少 ${key} 的大小`);
  }
  assert.equal(sourceUrls(m).unet, 'https://huggingface.co/RanaLLC/waifu-diffusion-onnx-v1-4/resolve/main/unet/model.onnx');
});

test('Cyberpunk 动漫模型：fp32 外部权重大小已配置（超过 2 GB，浏览器能否分配由运行时决定）', () => {
  const m = models.find((x) => x.id === 'cyberpunk-anime-sd15-onnx-fp32');
  assert.ok(m.files.unetWeights.size > 2e9, '外部权重应记录为超过 2 GB');
});
