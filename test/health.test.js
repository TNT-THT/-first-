// 输出自检（fp16 异常时切换 fp32）的纯逻辑测试。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { decideVariant, diagnoseOutput, fp32VariantOf, HEALTH_PROMPT } from '../web/src/engines/health.js';

test('fp32VariantOf：q4f16 版本映射到 q4f32，其他版本返回 null', () => {
  assert.equal(fp32VariantOf('Qwen2.5-3B-Instruct-q4f16_1-MLC'), 'Qwen2.5-3B-Instruct-q4f32_1-MLC');
  assert.equal(fp32VariantOf('Qwen3-8B-q4f16_1-MLC'), 'Qwen3-8B-q4f32_1-MLC');
  assert.equal(fp32VariantOf('Qwen3-8B-q4f32_1-MLC'), null);
  assert.equal(fp32VariantOf(''), null);
  assert.equal(fp32VariantOf(undefined), null);
});

test('diagnoseOutput：正常回答不报异常', () => {
  assert.equal(diagnoseOutput('你好！有什么可以帮你的吗？'), null);
  assert.equal(diagnoseOutput('Hello! How can I help?'), null);
  assert.equal(diagnoseOutput('ok ok ok'), null);
});

test('diagnoseOutput：空输出、替换字符、连续重复', () => {
  assert.equal(diagnoseOutput(''), 'empty');
  assert.equal(diagnoseOutput('   \n'), 'empty');
  assert.equal(diagnoseOutput('你好\uFFFD'), 'replacement');
  assert.equal(diagnoseOutput(',!!!!!!!!!!!!!!!!!!!!'), 'repeat');
  assert.equal(diagnoseOutput('好'.repeat(10)), 'repeat');
  // 连续 7 次还不算异常，阈值是 8
  assert.equal(diagnoseOutput('a'.repeat(7)), null);
});

test('decideVariant：正常保持；异常且有 fp32 切换；异常无 fp32 只提示', () => {
  assert.deepEqual(
    decideVariant({ modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', healthy: true, fp32Available: true }),
    { action: 'keep' },
  );
  assert.deepEqual(
    decideVariant({ modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', healthy: false, fp32Available: true }),
    { action: 'fallback', to: 'Qwen2.5-3B-Instruct-q4f32_1-MLC' },
  );
  assert.deepEqual(
    decideVariant({ modelId: 'Qwen2.5-3B-Instruct-q4f16_1-MLC', healthy: false, fp32Available: false }),
    { action: 'warn' },
  );
  // 本身就是 fp32 或没有 fp32 对应版本的模型，异常时不会切换
  assert.deepEqual(
    decideVariant({ modelId: 'Qwen3-8B-q4f32_1-MLC', healthy: false, fp32Available: true }),
    { action: 'warn' },
  );
});

test('自检提示词固定为“你好”', () => {
  assert.equal(HEALTH_PROMPT, '你好');
});
