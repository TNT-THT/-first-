// ONNX 图改写：Pow(x, 2) → Mul(x, x)（见 web/src/image/onnx-pow.js）
// 夹具由 test/fixtures/onnx-pow/make_fixture.py 生成（模拟分解后的 LayerNorm，fp16）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { rewritePowToMul } from '../web/src/image/onnx-pow.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = (p) => new Uint8Array(fs.readFileSync(path.join(here, 'fixtures', p)));

test('改写：指数为 2 的 Pow 改为 Mul；指数为 3 的 Pow 保留', () => {
  const src = fixture('onnx-pow/ln_decomposed_fp16.onnx');
  const r = rewritePowToMul(src);
  assert.equal(r.rewritten, 2);
  assert.equal(r.kept, 1);
  assert.notEqual(r.bytes, src);
});

test('没有可改写节点时返回原字节（同一对象）', () => {
  const src = fixture('sd15-tiny/text_encoder/model.onnx');
  const r = rewritePowToMul(src);
  assert.equal(r.rewritten, 0);
  assert.equal(r.bytes, src);
});

test('改写前后 ORT（wasm/CPU）的输出一致', async () => {
  const src = fixture('onnx-pow/ln_decomposed_fp16.onnx');
  const rewritten = rewritePowToMul(src).bytes;
  const x = new Uint16Array(32);
  for (let i = 0; i < 32; i++) x[i] = 0x3c00 + ((i * 37) % 200);
  const feeds = { x: new ort.Tensor('float16', x, [1, 4, 8]) };
  const run = async (bytes) => {
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    const out = await session.run(feeds);
    return Array.from(out.y.data);
  };
  const a = await run(src);
  const b = await run(rewritten);
  assert.deepEqual(b, a);
});
