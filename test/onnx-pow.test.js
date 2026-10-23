// ONNX 图改写：指数为常量 2 的 Pow(x, 2) → Mul(x, x)（见 web/src/image/onnx-pow.js）
// 夹具由 test/fixtures/onnx-pow/make_fixture.py 生成（模拟分解后的 LayerNorm，以及指数 2 的各种存放方式）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-web';
import { rewritePowToMul } from '../web/src/image/onnx-pow.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = (p) => new Uint8Array(fs.readFileSync(path.join(here, 'fixtures', 'onnx-pow', p)));

const xFeed = () => new ort.Tensor('float16', Uint16Array.from({ length: 32 }, (_, i) => 0x3c00 + ((i * 37) % 200)), [1, 4, 8]);

async function runCpu(bytes, feeds) {
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  const out = await session.run(feeds);
  return Array.from(out.y.data);
}

test('分解的 LayerNorm：指数 2（初始化器、Constant）改为 Mul；指数 3 保留并列出节点名', async () => {
  const src = fixture('ln_decomposed_fp16.onnx');
  const r = rewritePowToMul(src);
  assert.equal(r.rewritten, 2);
  assert.equal(r.kept, 1);
  assert.deepEqual(r.keptNodes, ['pow3/Pow']);
  assert.notEqual(r.bytes, src);
  const feeds = { x: xFeed() };
  assert.deepEqual(await runCpu(r.bytes, feeds), await runCpu(src, feeds));
});

for (const name of [
  'enc_fp16_int32data',
  'enc_fp32_floatdata',
  'enc_const_valuefloat',
  'enc_const_tensor_int32data',
  'enc_identity_chain',
  'enc_cast_chain',
]) {
  test(`指数 2 的存放方式：${name} 可以识别并改写，输出不变`, async () => {
    const src = fixture(`${name}.onnx`);
    const r = rewritePowToMul(src);
    assert.equal(r.rewritten, 1, `${name} 应当被改写`);
    assert.equal(r.kept, 0);
    const feeds = { x: xFeed() };
    assert.deepEqual(await runCpu(r.bytes, feeds), await runCpu(src, feeds));
  });
}

test('指数来自图输入（非常量）：保持不变并列出', () => {
  const src = fixture('enc_dynamic_exp.onnx');
  const r = rewritePowToMul(src);
  assert.equal(r.rewritten, 0);
  assert.equal(r.kept, 1);
  assert.deepEqual(r.keptNodes, ['Pow_dynamic']);
  assert.equal(r.bytes, src);
});

test('没有 Pow 的图：返回原字节（同一对象）', () => {
  const src = new Uint8Array(fs.readFileSync(path.join(here, 'fixtures', 'sd15-tiny', 'text_encoder', 'model.onnx')));
  const r = rewritePowToMul(src);
  assert.equal(r.rewritten, 0);
  assert.equal(r.kept, 0);
  assert.equal(r.bytes, src);
});
