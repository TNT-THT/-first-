// SD 1.5 端到端测试：用 scripts/image/make-tiny-sd15.py 生成的微型 ONNX 模型（接口与真实模型一致），
// 通过 onnxruntime-web 的 wasm 后端真正加载并运行。它验证的是加载、名称、dtype、外部权重与流程，
// 不验证画质（微型模型没有学到任何东西）。真实模型需要在有 WebGPU 的设备上另行测试。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';
import { createSd15Runtime } from '../web/src/image/sd15-runtime.js';
import { createClipTokenizer, bytesToUnicode } from '../web/src/image/clip-tokenizer.js';
import { createPndmScheduler } from '../web/src/image/pndm.js';
import { chwToRgba, generateSd15 } from '../web/src/image/sd15.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.join(here, 'fixtures');
const SCHEDULER_CONFIG = {
  num_train_timesteps: 1000,
  beta_start: 0.00085,
  beta_end: 0.012,
  beta_schedule: 'scaled_linear',
  skip_prk_steps: true,
  steps_offset: 1,
  prediction_type: 'epsilon',
  timestep_spacing: 'leading',
  set_alpha_to_one: false,
};

/** 小词表：字节字符 + 几条合并规则（足以覆盖测试提示词）。 */
function tinyTokenizer() {
  const chars = [...new Set(bytesToUnicode().values())];
  const vocab = {};
  let id = 0;
  for (const c of chars) {
    vocab[c] = id;
    id += 1;
    vocab[`${c}</w>`] = id;
    id += 1;
  }
  vocab['<|startoftext|>'] = id++;
  vocab['<|endoftext|>'] = id++;
  vocab.ca = id++;
  vocab['cat</w>'] = id++;
  const merges = '#version: 0.2\nc a\nca t</w>\n';
  return createClipTokenizer({ vocab, merges, maxLength: 77 });
}

function fixtureSources(folder) {
  const base = path.join(FIXTURES, folder);
  const read = (p) => new Uint8Array(fs.readFileSync(path.join(base, p)));
  return {
    textEncoder: read('text_encoder/model.onnx'),
    unet: read('unet/model.onnx'),
    unetExternal: { path: 'weights.pb', data: read('unet/weights.pb') },
    vaeDecoder: read('vae_decoder/model.onnx'),
  };
}

async function loadRuntime(folder) {
  return createSd15Runtime({ ort, sources: fixtureSources(folder), executionProviders: ['wasm'] });
}

function scheduler() {
  return createPndmScheduler(SCHEDULER_CONFIG);
}

test('运行时：fp32 接口加载，名称与 dtype 与 SD 1.5 约定一致', async () => {
  const rt = await loadRuntime('sd15-tiny');
  const ids = tinyTokenizer().encode('a cat');
  const emb = await rt.encode(ids);
  assert.equal(emb.length, 77 * 8);
  assert.ok(emb.every((v) => Number.isFinite(v)));
  rt.release();
});

test('运行时：fp16 接口（timestep 为 int64）同样可以加载与运行', async () => {
  const rt = await loadRuntime('sd15-tiny-fp16');
  const ids = tinyTokenizer().encode('a cat');
  const emb = await rt.encode(ids);
  assert.equal(emb.length, 77 * 8);
  const latent = new Float32Array(4 * 4 * 4).fill(0.25);
  const noise = await rt.unet({
    sample: new Float32Array(2 * 4 * 4 * 4).fill(0.1),
    timestep: 500,
    context: new Float32Array(2 * 77 * 8).fill(0.2),
    batch: 2,
    h8: 4,
    w8: 4,
  });
  assert.equal(noise.length, 2 * 4 * 4 * 4);
  assert.ok(noise.every((v) => Number.isFinite(v)));
  assert.equal(latent.length, 64);
  rt.release();
});

test('完整流程：256×256、4 步，输出 [3, H, W]，取值在 [-1, 1] 附近', async () => {
  const rt = await loadRuntime('sd15-tiny');
  const steps = [];
  const out = await generateSd15({
    runtime: rt,
    tokenizer: tinyTokenizer(),
    scheduler: scheduler(),
    prompt: 'a cat',
    width: 256,
    height: 256,
    steps: 4,
    guidance: 7.5,
    seed: 42,
    onStep: (p) => steps.push(p),
  });
  assert.equal(out.width, 256);
  assert.equal(out.height, 256);
  assert.equal(out.chw.length, 3 * 256 * 256);
  assert.ok(out.chw.every((v) => Number.isFinite(v) && v >= -1.001 && v <= 1.001));
  assert.deepEqual(steps.map((s) => s.step), [1, 2, 3, 4, 5]);
  assert.equal(steps.at(-1).total, 5);
  const rgba = chwToRgba(out.chw, 256, 256);
  assert.equal(rgba.length, 256 * 256 * 4);
  assert.equal(rgba[3], 255);
  rt.release();
});

test('确定性：相同种子与提示词得到相同结果；不同种子得到不同结果', async () => {
  const rt = await loadRuntime('sd15-tiny');
  const tok = tinyTokenizer();
  const run = (seed) =>
    generateSd15({
      runtime: rt,
      tokenizer: tok,
      scheduler: scheduler(),
      prompt: 'a cat',
      width: 256,
      height: 256,
      steps: 3,
      guidance: 7.5,
      seed,
    });
  const a = await run(7);
  const b = await run(7);
  const c = await run(8);
  assert.deepEqual(Array.from(a.chw), Array.from(b.chw));
  assert.notDeepEqual(Array.from(a.chw.slice(0, 64)), Array.from(c.chw.slice(0, 64)));
  rt.release();
});

test('fp16 与 fp32 接口的结果接近（验证 float16 的编码与解码）', async () => {
  const tok = tinyTokenizer();
  const args = { tokenizer: tok, prompt: 'a cat', width: 256, height: 256, steps: 3, guidance: 7.5, seed: 11 };
  const rt32 = await loadRuntime('sd15-tiny');
  const rt16 = await loadRuntime('sd15-tiny-fp16');
  const a = await generateSd15({ ...args, runtime: rt32, scheduler: scheduler() });
  const b = await generateSd15({ ...args, runtime: rt16, scheduler: scheduler() });
  let maxErr = 0;
  for (let i = 0; i < a.chw.length; i += 1) maxErr = Math.max(maxErr, Math.abs(a.chw[i] - b.chw[i]));
  assert.ok(maxErr < 0.02, `fp16 与 fp32 的最大误差为 ${maxErr}`);
  rt32.release();
  rt16.release();
});

test('无分类器引导（guidance ≤ 1）时只调用一次 UNet（batch=1）', async () => {
  const calls = [];
  const fake = {
    async encode() {
      return new Float32Array(77 * 8);
    },
    async unet(args) {
      calls.push(args.batch);
      return new Float32Array(args.sample.length);
    },
    async decode(latent, h8, w8) {
      return new Float32Array(3 * h8 * 8 * w8 * 8);
    },
  };
  await generateSd15({
    runtime: fake,
    tokenizer: tinyTokenizer(),
    scheduler: scheduler(),
    prompt: 'a cat',
    width: 256,
    height: 256,
    steps: 2,
    guidance: 1,
    seed: 1,
  });
  assert.deepEqual(calls, [1, 1, 1]);
});

test('取消：中途取消会抛出 AbortError，并且不再调用 UNet', async () => {
  const controller = new AbortController();
  let unetCalls = 0;
  const fake = {
    async encode() {
      return new Float32Array(77 * 8);
    },
    async unet(args) {
      unetCalls += 1;
      if (unetCalls === 1) controller.abort();
      return new Float32Array(args.sample.length);
    },
    async decode() {
      return new Float32Array(0);
    },
  };
  await assert.rejects(
    generateSd15({
      runtime: fake,
      tokenizer: tinyTokenizer(),
      scheduler: scheduler(),
      prompt: 'a cat',
      width: 256,
      height: 256,
      steps: 5,
      seed: 1,
      signal: controller.signal,
    }),
    (err) => err.name === 'AbortError',
  );
  assert.equal(unetCalls, 1);
});

test('输入名称不符时给出明确错误（而不是静默出错）', async () => {
  // 把 VAE 解码器（无外部权重）当作文本编码器加载：会话可以创建，但输入名称是 latent_sample，必然不符
  const sources = fixtureSources('sd15-tiny');
  await assert.rejects(
    createSd15Runtime({
      ort,
      sources: { ...sources, textEncoder: sources.vaeDecoder },
      executionProviders: ['wasm'],
    }),
    /文本编码器的输入名称与预期不符/,
  );
});
