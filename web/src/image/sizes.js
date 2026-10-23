/**
 * 生图尺寸（纯逻辑）。SD 1.5 的 VAE 下采样为 8 倍，因此宽高必须是 8 的倍数。
 * 范围上限 768：SD 1.5 以 512 为原生分辨率，明显超过时画质会下降，并占用更多显存。
 */
export const SIZE_LIMITS = { min: 256, max: 768, step: 8 };
export const DEFAULT_SIZE = { width: 512, height: 512 };

export const RATIOS = [
  { id: '1:1', w: 1, h: 1 },
  { id: '4:3', w: 4, h: 3 },
  { id: '3:4', w: 3, h: 4 },
  { id: '16:9', w: 16, h: 9 },
  { id: '9:16', w: 9, h: 16 },
];

/** 把任意数值夹到 [min, max] 并取到最近的 step 倍数。非数值时返回 null。 */
export function snapSize(value, { min = SIZE_LIMITS.min, max = SIZE_LIMITS.max, step = SIZE_LIMITS.step } = {}) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  const clamped = Math.min(max, Math.max(min, n));
  return Math.round(clamped / step) * step;
}

/** 宽高组合：两者都夹到范围内并取整到 8 的倍数；非数值时使用默认尺寸。 */
export function normalizeSize(width, height) {
  const w = snapSize(width);
  const h = snapSize(height);
  if (w === null || h === null) return { ...DEFAULT_SIZE };
  return { width: w, height: h };
}

/** 按比例与长边计算宽高，例如 (16:9, 768) → 768 × 432。结果取整到 8 的倍数。 */
export function sizeForRatio(ratioId, longSide = DEFAULT_SIZE.width) {
  const ratio = RATIOS.find((r) => r.id === ratioId);
  if (!ratio) return { ...DEFAULT_SIZE };
  const long = snapSize(longSide);
  if (long === null) return { ...DEFAULT_SIZE };
  if (ratio.w >= ratio.h) return normalizeSize(long, (long * ratio.h) / ratio.w);
  return normalizeSize((long * ratio.w) / ratio.h, long);
}

/** 检查宽高是否合法；合法时返回 null，否则返回中文说明。 */
export function sizeProblem(width, height) {
  for (const [name, v] of [
    ['宽', width],
    ['高', height],
  ]) {
    if (!Number.isInteger(v) || v < SIZE_LIMITS.min || v > SIZE_LIMITS.max) {
      return `${name}必须是 ${SIZE_LIMITS.min}–${SIZE_LIMITS.max} 之间的整数`;
    }
    if (v % SIZE_LIMITS.step !== 0) return `${name}必须是 ${SIZE_LIMITS.step} 的倍数`;
  }
  return null;
}
