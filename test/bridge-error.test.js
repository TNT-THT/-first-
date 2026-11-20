import test from 'node:test';
import assert from 'node:assert/strict';
import { bridgeErrorMessage } from '../web/src/local-model-bridge.js';

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
