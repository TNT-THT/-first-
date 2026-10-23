/**
 * 可复现的随机数（纯逻辑）：相同种子得到相同的初始噪声。
 * 使用 mulberry32 + Box-Muller 生成标准正态分布。注意：这与 PyTorch 的随机数序列不同，
 * 因此同一种子在本项目与 diffusers 中得到的图像不会逐像素相同。
 */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 长度为 n 的标准正态样本（Float32Array）。 */
export function gaussianNoise(seed, n) {
  const uniform = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(uniform(), 1e-12);
    const u2 = uniform();
    const r = Math.sqrt(-2 * Math.log(u1));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

/** 生成一个新的随机种子（无符号 32 位整数）。 */
export function randomSeed() {
  return Math.floor(Math.random() * 4294967296) >>> 0;
}
