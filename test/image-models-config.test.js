// 生图模型配置检查：新加入的 SD 1.x 结构模型（BK-SDM、Tiny SD、Small SD、SDXS）的文件清单、大小与地址。
// 只检查配置本身；真实文件的接口与画质需在浏览器中测试（见各条目的 note）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { imageModelsOf, isSelectable, sourceSizes, sourceUrls, totalBytes } from '../web/src/image/catalog.js';
import { oversizedFiles } from '../web/src/image/model-files.js';

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
    assert.equal(oversizedFiles(sourceSizes(m)).length, 0, `${id} 有文件超过 2 GB`);
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
