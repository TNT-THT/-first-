/**
 * 模型输出自检（纯逻辑，便于测试）。
 *
 * 背景：部分 q4f16 模型在某些显卡 / 驱动上会输出大量无意义字符（例如 "!!!!!!"），
 * 原因通常是 fp16 数值溢出。默认加载 fp16 版本；加载后用固定提示词做一次短测试，
 * 若输出异常且存在对应的 fp32 版本，则自动切换到 fp32。
 */

/** 固定自检提示词与参数：确定性（temperature 0）、输出很短，耗时可忽略。 */
export const HEALTH_PROMPT = '你好';
export const HEALTH_MAX_TOKENS = 32;
/** 自检最长等待时间。超时则跳过自检（不阻塞加载），并中断这次生成。 */
export const HEALTH_TIMEOUT_MS = 90_000;

const FP16_SUFFIX = 'q4f16_1-MLC';
const FP32_SUFFIX = 'q4f32_1-MLC';
const REPEAT_LIMIT = 8;

/** 与 fp16 版本对应的 fp32 模型 ID；不是 q4f16 版本时返回 null。 */
export function fp32VariantOf(modelId) {
  const id = String(modelId || '');
  if (!id.endsWith(FP16_SUFFIX)) return null;
  return id.slice(0, -FP16_SUFFIX.length) + FP32_SUFFIX;
}

/**
 * 判断自检输出是否异常。异常返回原因字符串，正常返回 null。
 * - empty：没有任何输出
 * - replacement：出现 Unicode 替换字符（解码失败的迹象）
 * - repeat：同一个非空白字符连续出现 REPEAT_LIMIT 次以上（例如 "!!!!!!!!"）
 */
export function diagnoseOutput(text) {
  const s = String(text ?? '');
  if (!s.trim()) return 'empty';
  if (s.includes('\uFFFD')) return 'replacement';
  const re = /(\S)\1*/gu;
  for (const m of s.matchAll(re)) {
    if (m[0].length >= REPEAT_LIMIT) return 'repeat';
  }
  return null;
}

/**
 * 决定加载后的处理：
 *   { action: 'keep' }                     fp16 正常，继续使用
 *   { action: 'fallback', to }             fp16 异常且存在 fp32 版本，切换到 to
 *   { action: 'warn' }                     异常但没有可用的 fp32 版本，只提示
 */
export function decideVariant({ modelId, healthy, fp32Available }) {
  if (healthy) return { action: 'keep' };
  const to = fp32VariantOf(modelId);
  if (to && fp32Available) return { action: 'fallback', to };
  return { action: 'warn' };
}
