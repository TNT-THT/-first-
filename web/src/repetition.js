/**
 * 重复输出的检测与裁剪（纯逻辑，便于测试）。
 *
 * 小模型有时会陷入循环，反复输出同一段文字直到达到输出上限。这里在流式输出的末尾检测
 * “同一段文字连续重复”的情况：对每个候选周期 P，检查末尾是否存在周期为 P 的连续重复，
 * 不依赖输出停在哪个位置。检测不使用时间。
 */

/** 重复单元的最短长度（字符）。更短的重复（如连续的分隔符、短句）不视为循环。 */
export const REPEAT_MIN_UNIT = 30;
/** 连续重复的最少次数。 */
export const REPEAT_MIN_COUNT = 3;
/** 重复单元的最长长度（字符），限制检测的计算量。 */
export const REPEAT_MAX_UNIT = 400;
/** 流式输出中每累计这么多字符检查一次循环（检查有计算开销，不必每个字符都做）。 */
export const LOOP_CHECK_EVERY = 50;
/** 重复单元中至少要有这么多字母或汉字，避免把 “=====” 之类的分隔线当作循环。 */
const MIN_LETTERS = 10;

function letterCount(s) {
  const m = s.match(/\p{L}/gu);
  return m ? m.length : 0;
}

/**
 * 检测末尾是否存在连续重复。返回 {period, count, start}：
 *   period：重复单元长度；count：重复次数（估计）；start：重复区域在文本中的起点。
 * 没有重复时返回 null。
 */
export function findTailRepeat(
  text,
  { minUnit = REPEAT_MIN_UNIT, minCount = REPEAT_MIN_COUNT, maxUnit = REPEAT_MAX_UNIT } = {},
) {
  const s = String(text || '');
  const n = s.length;
  for (let p = minUnit; p <= maxUnit && p * minCount <= n; p += 1) {
    // run：从末尾向前，满足 s[k] === s[k-p] 的连续位置数
    let k = n - 1;
    while (k - p >= 0 && s[k] === s[k - p]) k -= 1;
    const run = n - 1 - k;
    const span = run + p; // 周期 p 覆盖的长度
    if (span < p * minCount) continue;
    const unit = s.slice(n - p);
    if (letterCount(unit) < MIN_LETTERS) continue;
    return { period: p, count: Math.floor(span / p), start: n - span };
  }
  return null;
}

/** 去掉连续重复：只保留第一份重复单元，之后的重复全部删除。没有重复时原样返回。 */
export function trimRepetition(text, options) {
  const s = String(text || '');
  const hit = findTailRepeat(s, options);
  if (!hit) return s;
  return s.slice(0, hit.start + hit.period);
}
