import test from 'node:test';
import assert from 'node:assert/strict';
import { bridgeErrorMessage, mapBridgeParams } from '../web/src/local-model-bridge.js';

test('桥接错误说明：保留错误类型与真实原因，并带上模型 ID', () => {
  const gpu = new Error('device lost: GPUBuffer mapAsync failed');
  gpu.name = 'GpuResetError';
  assert.equal(
    bridgeErrorMessage(gpu, 'Qwen2.5-3B-Instruct-q4f16_1-MLC'),
    '本机模型生成失败（模型 Qwen2.5-3B-Instruct-q4f16_1-MLC）：GpuResetError: device lost: GPUBuffer mapAsync failed',
  );

  // DOMException 等带 name/message 的非 Error 对象
  assert.match(bridgeErrorMessage({ name: 'AbortError', message: '已停止' }, 'm1'), /AbortError: 已停止/);
  // 字符串错误
  assert.match(bridgeErrorMessage('模型尚未加载', 'm1'), /模型 m1.*模型尚未加载/);
  // 没有 message 的普通对象：序列化内容，而不是只剩一句通用提示
  assert.match(bridgeErrorMessage({ code: 'OOM', detail: 'out of memory' }, 'm1'), /OOM/);
  // 完全没有信息的错误也要落到可读文本
  assert.match(bridgeErrorMessage(undefined, 'm1'), /本机模型生成失败（模型 m1）：/);
  // 超长原因会被截断，避免把整段堆栈塞进 API 响应
  const long = bridgeErrorMessage(new Error('x'.repeat(5000)), 'm1');
  assert.ok(long.length < 800);
});

test('桥接生成参数映射：支持的参数转成引擎选项，非法值明确报错而不是静默丢弃', () => {
  assert.deepEqual(mapBridgeParams(null), {});
  assert.deepEqual(mapBridgeParams({}), {});
  assert.deepEqual(
    mapBridgeParams({ top_p: 0.9, stop: '<end>', frequency_penalty: 0.5, seed: 7, response_format: { type: 'json_object' } }),
    { topP: 0.9, stop: ['<end>'], frequencyPenalty: 0.5, seed: 7, responseFormat: { type: 'json_object' } },
  );
  assert.throws(() => mapBridgeParams({ top_p: 3 }), /top_p 不合法/);
  assert.throws(() => mapBridgeParams({ frequency_penalty: -9 }), /frequency_penalty 不合法/);
  assert.throws(() => mapBridgeParams({ seed: 1.5 }), /seed 不合法/);
  assert.throws(() => mapBridgeParams({ response_format: { type: 'json_schema' } }), /仅支持 response_format/);
});
