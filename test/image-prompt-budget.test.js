// 生图提示词预算：管理员的更正与知识库必须完整放入 CLIP 的 75 个可用 token，超出则拒绝生成（不截断）；
// 用户描述超出时只报告截断的 token 数。使用真实的 CLIP 分词器（小词表）计数。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createClipTokenizer, parseMerges } from '../web/src/image/clip-tokenizer.js';
import { checkPromptBudget, countContentTokens, joinLeadTexts, CLIP_CONTENT_TOKENS } from '../web/src/image/prompt-budget.js';
import { buildImagePrompt, effectiveInputLimit, parseImageParams, setImageParams, getImageSettings } from '../web/src/image/params.js';
import { DEFAULT_INPUT_TOKENS, MAX_INPUT_TOKENS } from '../web/src/image/prompt-budget.js';

// 每个 'a' 是一个 token（没有合并规则，词表中只有 a</w>；其余字符映射为 unk，同样计为一个 token）
function tinyTokenizer() {
  const vocab = { '<|startoftext|>': 49406, '<|endoftext|>': 49407, 'a</w>': 1, a: 2 };
  return createClipTokenizer({ vocab, merges: parseMerges(''), maxLength: 77 });
}

test('占用计数：不含 BOS 与 EOS；rawTokenize 不截断，tokenize 仍截断到 77', () => {
  const tok = tinyTokenizer();
  assert.equal(countContentTokens(tok, 'aaaa'), 4);
  assert.equal(countContentTokens(tok, '   '), 0);
  const long = 'a '.repeat(100); // 100 个词
  assert.equal(tok.rawTokenize(long).length, 102);
  const cut = tok.tokenize(long);
  assert.equal(cut.length, 77);
  assert.equal(cut[76], 49407, '截断后仍以 EOS 结尾');
});

test('预算：前置内容与描述都能放下时不截断', () => {
  const tok = tinyTokenizer();
  const r = checkPromptBudget(tok, { lead: 'a a', prompt: 'a a a' });
  assert.equal(r.leadTokens, 2);
  assert.equal(r.promptTokens, 3);
  assert.equal(r.capacity, CLIP_CONTENT_TOKENS);
  assert.equal(r.truncatedTokens, 0);
});

test('预算：前置内容恰好 75 个 token 时允许生成', () => {
  const tok = tinyTokenizer();
  const lead = Array(75).fill('a').join(' ');
  const r = checkPromptBudget(tok, { lead, prompt: lead });
  assert.equal(r.leadTokens, 75);
  assert.equal(r.truncatedTokens, 0);
});

test('预算：前置内容超过 75 个 token 时拒绝生成（不静默截断）', () => {
  const tok = tinyTokenizer();
  const lead = Array(76).fill('a').join(' ');
  assert.throws(() => checkPromptBudget(tok, { lead, prompt: 'cat' }), (err) => {
    assert.match(err.message, /已拒绝生成/);
    assert.match(err.message, /76 个 token/);
    assert.match(err.message, /75 个可用/);
    return true;
  });
});

test('预算：描述过长时仍生成，并报告末尾被截断的 token 数', () => {
  const tok = tinyTokenizer();
  const lead = 'a a';
  // 逗号本身也是一个 token（映射为 unk）：2 + 1 + 80 = 83
  const prompt = `${lead}, ${Array(80).fill('a').join(' ')}`;
  const r = checkPromptBudget(tok, { lead, prompt });
  assert.equal(r.promptTokens, 83);
  assert.equal(r.truncatedTokens, 8);
});

test('前置内容拼接与提示词构造一致：空项被忽略，lead 位于提示词最前面', () => {
  assert.equal(joinLeadTexts(['  更正甲 ', '', '知识乙']), '更正甲, 知识乙');
  const prompt = buildImagePrompt('猫', [{ text: '水彩' }], ['更正甲', '知识乙']);
  assert.equal(prompt, '更正甲, 知识乙, 猫, 水彩');
  assert.ok(prompt.startsWith(joinLeadTexts(['更正甲', '知识乙'])));
});

// ---- 描述输入上限（默认 4096，管理员可调，硬上限 16384）----

test('输入上限：默认 4096，硬上限 16384', () => {
  assert.equal(DEFAULT_INPUT_TOKENS, 4096);
  assert.equal(MAX_INPUT_TOKENS, 16384);
  assert.equal(effectiveInputLimit({}), 4096);
  assert.equal(effectiveInputLimit({ inputTokens: 8192 }), 8192);
  assert.equal(effectiveInputLimit({ inputTokens: 99999 }), 16384, '超过硬上限时夹到 16384');
  assert.equal(effectiveInputLimit({ inputTokens: 0 }), 1, '下限为 1');
});

test('输入上限：描述超过上限时拒绝生成，并说明可调上限', () => {
  const tok = tinyTokenizer();
  const input = Array(10).fill('a').join(' ');
  assert.throws(() => checkPromptBudget(tok, { input, maxInputTokens: 9 }), (err) => {
    assert.match(err.message, /描述共 10 个 token/);
    assert.match(err.message, /输入上限 9 个/);
    assert.match(err.message, /已拒绝生成/);
    return true;
  });
});

test('输入上限：描述在上限内时通过；超过 75 个 token 只报告截断（不拒绝）', () => {
  const tok = tinyTokenizer();
  const input = Array(100).fill('a').join(' ');
  const r = checkPromptBudget(tok, { input, maxInputTokens: 4096, prompt: input });
  assert.equal(r.inputTokens, 100);
  assert.equal(r.maxInputTokens, 4096);
  assert.equal(r.truncatedTokens, 25, 'SD 1.5 的 CLIP 只读取 75 个');
});

test('输入上限：未传入 input 时不检查（兼容旧调用）', () => {
  const tok = tinyTokenizer();
  const r = checkPromptBudget(tok, { prompt: 'a a' });
  assert.equal(r.inputTokens, 0);
});

test('管理员参数：inputTokens 可由参数行设置，超出范围时夹到 16384，仅保留合法键', () => {
  assert.deepEqual(parseImageParams('参数：inputTokens=8192, steps=25'), { inputTokens: 8192, steps: 25 });
  assert.equal(parseImageParams('参数：inputTokens=99999').inputTokens, 16384);
  const storage = { store: null, getItem() { return this.store; }, setItem(_k, v) { this.store = v; } };
  const params = setImageParams('tiny-sd-onnx-webgpu', { inputTokens: 50000, bogus: 1 }, { storage });
  assert.deepEqual(params, { inputTokens: 16384 });
  assert.equal(effectiveInputLimit(getImageSettings('tiny-sd-onnx-webgpu', { storage }).params), 16384);
});
