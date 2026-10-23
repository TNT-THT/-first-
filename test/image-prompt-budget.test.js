// 生图提示词预算：管理员的更正与知识库必须完整放入 CLIP 的 75 个可用 token，超出则拒绝生成（不截断）；
// 用户描述超出时只报告截断的 token 数。使用真实的 CLIP 分词器（小词表）计数。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createClipTokenizer, parseMerges } from '../web/src/image/clip-tokenizer.js';
import { checkPromptBudget, countContentTokens, joinLeadTexts, CLIP_CONTENT_TOKENS } from '../web/src/image/prompt-budget.js';
import { buildImagePrompt } from '../web/src/image/params.js';

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
