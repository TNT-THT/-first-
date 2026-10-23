// 长提示词分段（CLIP 每段 75 个内容 token，最多 4 段）与回退到单段。
// - 纯函数：splitPromptIntoChunks 的布局、截断与补齐；
// - 假运行时：UNet 收到的 context 长度与段顺序、CFG 下正负提示词的补齐；
// - 真实 ONNX（微型模型）：其 encoder_hidden_states 的序列长度写死为 77，分段时应失败并回退到单段。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import * as ort from 'onnxruntime-web';
import { createClipTokenizer, parseMerges, bytesToUnicode } from '../web/src/image/clip-tokenizer.js';
import { splitPromptIntoChunks } from '../web/src/image/long-prompt.js';
import { createPndmScheduler } from '../web/src/image/pndm.js';
import { generateSd15, generateSd15WithFallback, ChunkedPromptError } from '../web/src/image/sd15.js';
import { createSd15Runtime } from '../web/src/image/sd15-runtime.js';
import {
  checkPromptBudget,
  MAX_PROMPT_CHUNKS,
  CLIP_CONTENT_TOKENS,
} from '../web/src/image/prompt-budget.js';

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
const D = 2; // 假运行时的嵌入维度
const SIZE = 256; // 生图尺寸下限为 256

// 每个单词 'a' 编码为 'a</w>'（id 1）；与 image-prompt-budget.test.js 相同的小词表
function tinyTokenizer() {
  const vocab = { '<|startoftext|>': 49406, '<|endoftext|>': 49407, 'a</w>': 1, a: 2 };
  return createClipTokenizer({ vocab, merges: parseMerges(''), maxLength: 77 });
}

const A = (n) => Array(n).fill('a').join(' ');

/** 假运行时：嵌入的第 k 维取 token id，便于检查段的顺序；记录每次 UNet 调用。 */
function fakeRuntime({ failUnlessSeq = null, failWith = null } = {}) {
  const calls = { encode: [], unet: [] };
  return {
    calls,
    async encode(ids) {
      calls.encode.push(Int32Array.from(ids));
      const out = new Float32Array(77 * D);
      for (let k = 0; k < 77; k += 1) {
        out[k * D] = ids[k];
        out[k * D + 1] = -1;
      }
      return out;
    },
    async unet({ sample, context, batch, seqLength = 77 }) {
      calls.unet.push({ batch, seqLength, contextLength: context.length, context: Float32Array.from(context) });
      if (failWith) throw failWith;
      if (failUnlessSeq !== null && seqLength !== failUnlessSeq) {
        throw new Error(`Got invalid dimensions for input: encoder_hidden_states, expected 77, got ${seqLength}`);
      }
      return new Float32Array(sample.length);
    },
    async decode(latent, h8, w8) {
      return new Float32Array(3 * h8 * 8 * w8 * 8);
    },
  };
}

function scheduler() {
  return createPndmScheduler(SCHEDULER_CONFIG);
}

test('分段：短提示词只有一段，与 tokenizer.encode 逐位相同', () => {
  const tok = tinyTokenizer();
  const r = splitPromptIntoChunks(tok, 'a a a', { maxChunks: MAX_PROMPT_CHUNKS });
  assert.equal(r.chunks.length, 1);
  assert.deepEqual(Array.from(r.chunks[0]), Array.from(tok.encode('a a a')));
  assert.equal(r.truncatedTokens, 0);
});

test('分段：空文本仍然是一段（与 encode("") 相同）', () => {
  const tok = tinyTokenizer();
  const r = splitPromptIntoChunks(tok, '', { maxChunks: 4 });
  assert.equal(r.chunks.length, 1);
  assert.deepEqual(Array.from(r.chunks[0]), Array.from(tok.encode('')));
});

test('分段：75 个内容 token 恰好一段；76 个 token 分为两段', () => {
  const tok = tinyTokenizer();
  assert.equal(splitPromptIntoChunks(tok, A(75), { maxChunks: 4 }).chunks.length, 1);
  const r = splitPromptIntoChunks(tok, A(76), { maxChunks: 4 });
  assert.equal(r.chunks.length, 2);
  assert.equal(r.contentTokens, 76);
});

test('分段：每段布局为 [BOS, 内容, EOS, PAD…]，长度 77', () => {
  const tok = tinyTokenizer();
  const { chunks } = splitPromptIntoChunks(tok, A(100), { maxChunks: 4 });
  assert.equal(chunks.length, 2);
  for (const c of chunks) {
    assert.equal(c.length, 77);
    assert.equal(c[0], tok.bos);
  }
  // 第一段：75 个内容 token 后紧跟 EOS
  assert.equal(chunks[0][76], tok.eos);
  // 第二段：25 个内容 token，之后是 EOS 与 PAD
  assert.equal(chunks[1][26], tok.eos);
  assert.equal(chunks[1][27], tok.pad);
});

test('分段：超过 maxChunks × 75 的部分截断，并报告截断数', () => {
  const tok = tinyTokenizer();
  const r = splitPromptIntoChunks(tok, A(400), { maxChunks: MAX_PROMPT_CHUNKS });
  assert.equal(r.chunks.length, MAX_PROMPT_CHUNKS);
  assert.equal(r.contentTokens, 400);
  assert.equal(r.truncatedTokens, 400 - MAX_PROMPT_CHUNKS * CLIP_CONTENT_TOKENS);
});

test('分段：maxChunks 为 1 时与旧行为一致（只保留 75 个）', () => {
  const tok = tinyTokenizer();
  const r = splitPromptIntoChunks(tok, A(100), { maxChunks: 1 });
  assert.equal(r.chunks.length, 1);
  assert.equal(r.truncatedTokens, 25);
});

test('生成：长提示词分段时 UNet 的 seqLength 为 77 × 段数，context 按段顺序拼接', async () => {
  const tok = tinyTokenizer();
  const rt = fakeRuntime();
  const out = await generateSd15({
    runtime: rt,
    tokenizer: tok,
    scheduler: scheduler(),
    prompt: A(100),
    width: SIZE,
    height: SIZE,
    steps: 2,
    guidance: 1, // 无 CFG：batch 为 1
    seed: 1,
    maxChunks: 4,
  });
  assert.equal(out.promptChunks, 2);
  assert.ok(rt.calls.unet.length > 0);
  const u = rt.calls.unet[0];
  assert.equal(u.seqLength, 154);
  assert.equal(u.batch, 1);
  assert.equal(u.contextLength, 154 * D);
  // 第一段的第一个位置是 BOS；第二段的第一个位置（第 77 个）也是 BOS
  assert.equal(u.context[0], tok.bos);
  assert.equal(u.context[77 * D], tok.bos);
});

test('生成：CFG 下正负提示词段数不同时，较短一方用空段补齐，两者长度一致', async () => {
  const tok = tinyTokenizer();
  const rt = fakeRuntime();
  await generateSd15({
    runtime: rt,
    tokenizer: tok,
    scheduler: scheduler(),
    prompt: A(100), // 条件：2 段
    negative: 'a', // 无条件：1 段，需补一个空段
    width: SIZE,
    height: SIZE,
    steps: 2,
    guidance: 7.5,
    seed: 1,
    maxChunks: 4,
  });
  const u = rt.calls.unet[0];
  assert.equal(u.batch, 2);
  assert.equal(u.seqLength, 154);
  assert.equal(u.contextLength, 2 * 154 * D);
  const half = 154 * D;
  // 无条件部分（前半）：第一段为 'a'，第二段是补齐的空段 [BOS, EOS, PAD…]
  assert.equal(u.context[77 * D], tok.bos);
  assert.equal(u.context[78 * D], tok.eos);
  // 条件部分（后半）：第二段为真实内容（每个单词 'a' 编码为 'a</w>'，id 为 1，不是 EOS）
  assert.equal(u.context[half + 77 * D], tok.bos);
  assert.equal(u.context[half + 78 * D], 1);
});

test('生成：单段（默认）时 context 长度与旧行为相同', async () => {
  const tok = tinyTokenizer();
  const rt = fakeRuntime();
  const out = await generateSd15({
    runtime: rt,
    tokenizer: tok,
    scheduler: scheduler(),
    prompt: 'a a',
    negative: '',
    width: SIZE,
    height: SIZE,
    steps: 2,
    guidance: 7.5,
    seed: 1,
  });
  assert.equal(out.promptChunks, 1);
  assert.equal(rt.calls.unet[0].seqLength, 77);
  assert.equal(rt.calls.unet[0].contextLength, 2 * 77 * D);
});

test('回退：分段时 UNet 失败，自动改用单段重做并标记 chunkFallback', async () => {
  const tok = tinyTokenizer();
  const rt = fakeRuntime({ failUnlessSeq: 77 });
  const out = await generateSd15WithFallback({
    runtime: rt,
    tokenizer: tok,
    scheduler: scheduler(),
    prompt: A(100),
    width: SIZE,
    height: SIZE,
    steps: 2,
    guidance: 1,
    seed: 1,
    maxChunks: 4,
  });
  assert.equal(out.chunkFallback, true);
  assert.match(out.chunkError, /invalid dimensions/);
  assert.equal(out.promptChunks, 1);
  assert.equal(out.chw.length, 3 * SIZE * SIZE);
  // 最后一次 UNet 调用使用单段
  assert.equal(rt.calls.unet.at(-1).seqLength, 77);
});

test('回退：单段时 UNet 失败不回退，直接抛出原错误', async () => {
  const tok = tinyTokenizer();
  const rt = fakeRuntime({ failWith: new Error('GPU 内存不足') });
  await assert.rejects(
    generateSd15WithFallback({
      runtime: rt,
      tokenizer: tok,
      scheduler: scheduler(),
      prompt: 'a',
      width: SIZE,
      height: SIZE,
      steps: 2,
      guidance: 1,
      seed: 1,
      maxChunks: 4,
    }),
    (err) => !(err instanceof ChunkedPromptError) && /GPU 内存不足/.test(err.message),
  );
});

test('回退：分段时取消（AbortError）不回退', async () => {
  const tok = tinyTokenizer();
  const abortErr = Object.assign(new Error('已取消生图'), { name: 'AbortError' });
  const rt = fakeRuntime({ failWith: abortErr });
  await assert.rejects(
    generateSd15WithFallback({
      runtime: rt,
      tokenizer: tok,
      scheduler: scheduler(),
      prompt: A(100),
      width: SIZE,
      height: SIZE,
      steps: 2,
      guidance: 1,
      seed: 1,
      maxChunks: 4,
    }),
    (err) => err.name === 'AbortError',
  );
  assert.equal(rt.calls.unet.length, 1, '不应重做');
});

test('预算：允许 4 段时容量为 300，超出部分才报告截断；管理员内容仍以单段 75 计', () => {
  const tok = tinyTokenizer();
  const r = checkPromptBudget(tok, { prompt: A(320), maxChunks: MAX_PROMPT_CHUNKS });
  assert.equal(r.capacity, 300);
  assert.equal(r.truncatedTokens, 20);
  const one = checkPromptBudget(tok, { prompt: A(320), maxChunks: 1 });
  assert.equal(one.capacity, 75);
  assert.equal(one.truncatedTokens, 245);
  // 管理员前置内容超过单段 75 个 token 时仍拒绝，与段数无关
  assert.throws(() => checkPromptBudget(tok, { lead: A(76), prompt: 'a', maxChunks: 4 }), /75/);
});

// ---- 真实 ONNX（微型模型，固定序列长度 77）----

function tinyVocabTokenizer() {
  // 与 image-sd15.test.js 相同的小词表：字节字符 + 合并规则
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
  return createClipTokenizer({ vocab, merges: '#version: 0.2\nc a\nca t</w>\n', maxLength: 77 });
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

test('真实 ONNX：微型模型（序列长度固定为 77）分段失败后回退到单段并出图', async () => {
  const rt = await createSd15Runtime({
    ort,
    sources: fixtureSources('sd15-tiny'),
    executionProviders: ['wasm'],
  });
  const tok = tinyVocabTokenizer();
  const long = Array(120).fill('cat').join(' ');
  const out = await generateSd15WithFallback({
    runtime: rt,
    tokenizer: tok,
    scheduler: scheduler(),
    prompt: long,
    negative: 'cat',
    width: SIZE,
    height: SIZE,
    steps: 2,
    guidance: 7.5,
    seed: 3,
    maxChunks: MAX_PROMPT_CHUNKS,
  });
  assert.equal(out.chunkFallback, true, '固定长度模型应拒绝两段输入，从而回退');
  assert.ok(out.chunkError.length > 0);
  assert.equal(out.promptChunks, 1);
  assert.equal(out.chw.length, 3 * SIZE * SIZE);
  rt.release();
});

test('真实 ONNX：短提示词不分段，不回退', async () => {
  const rt = await createSd15Runtime({
    ort,
    sources: fixtureSources('sd15-tiny'),
    executionProviders: ['wasm'],
  });
  const out = await generateSd15WithFallback({
    runtime: rt,
    tokenizer: tinyVocabTokenizer(),
    scheduler: scheduler(),
    prompt: 'cat',
    negative: '',
    width: SIZE,
    height: SIZE,
    steps: 2,
    guidance: 7.5,
    seed: 3,
    maxChunks: MAX_PROMPT_CHUNKS,
  });
  assert.equal(out.chunkFallback, false);
  assert.equal(out.promptChunks, 1);
  rt.release();
});
