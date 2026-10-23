// 生图模型加载进度：读取文件与创建会话交错进行，整体进度必须单调递增，
// 且不能因为开始某个会话就跳到 90%（回归：文本编码器会话开始时进度条直接到 90%）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { loadFraction, LOAD_SHARE } from '../web/src/image/load-progress.js';

test('整体进度：读取占 90%，会话占 10%', () => {
  assert.equal(loadFraction({ loaded: 0, total: 100, sessionsDone: 0, sessionTotal: 3 }), 0);
  assert.ok(Math.abs(loadFraction({ loaded: 100, total: 100, sessionsDone: 0, sessionTotal: 3 }) - LOAD_SHARE) < 1e-12);
  assert.ok(Math.abs(loadFraction({ loaded: 100, total: 100, sessionsDone: 3, sessionTotal: 3 }) - 1) < 1e-12);
  assert.equal(loadFraction({ loaded: 0, total: 0, sessionsDone: 0, sessionTotal: 3 }), 0);
});

test('开始文本编码器会话时（约读完 0.25 GB / 2.56 GB）进度不超过 20%', () => {
  const total = 2559933000; // 与 fp16 配置的文件总大小（不含备选文本编码器）相近
  const f = loadFraction({ loaded: 246476214 + 1500000, total, sessionsDone: 0, sessionTotal: 3 });
  assert.ok(f < 0.2, `进度 ${f} 过高`);
});

test('按真实加载顺序模拟：进度单调不减，且 0.28 GB 时远低于 80%', () => {
  const cfg = JSON.parse(fs.readFileSync(new URL('../web/public/config.json', import.meta.url), 'utf8'));
  const model = cfg.imageModels.find((m) => m.id === 'sd15-nmkd-fp16');
  const size = (k) => Number(model.files[k].size);
  // 与 sd15-worker.js 相同的顺序：先读小文件，然后 文本编码器 → UNet(+权重) → VAE；每个会话在读完其文件后创建
  const reads = [
    ['vocab', size('vocab')], ['merges', size('merges')], ['scheduler', size('scheduler')],
    ['textEncoder', size('textEncoder')], ['session'],
    ['unet', size('unet')], ['unetWeights', size('unetWeights')], ['session'],
    ['vaeDecoder', size('vaeDecoder')], ['session'],
  ];
  const total = ['vocab', 'merges', 'scheduler', 'textEncoder', 'unet', 'unetWeights', 'vaeDecoder'].reduce((n, k) => n + size(k), 0);
  let loaded = 0;
  let sessionsDone = 0;
  let last = 0;
  let maxAtStart = 0;
  for (const step of reads) {
    if (step[0] === 'session') {
      const f = loadFraction({ loaded, total, sessionsDone, sessionTotal: 3 });
      if (sessionsDone === 0) maxAtStart = f;
      assert.ok(f >= last - 1e-12, '进度不应后退');
      last = f;
      sessionsDone += 1;
      continue;
    }
    loaded += step[1];
    const f = loadFraction({ loaded, total, sessionsDone, sessionTotal: 3 });
    assert.ok(f >= last - 1e-12, '进度不应后退');
    last = f;
  }
  assert.ok(maxAtStart < 0.2, `文本编码器会话开始时进度 ${maxAtStart} 过高`);
  assert.ok(last <= 1 + 1e-12);
});
