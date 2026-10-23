/**
 * Stable Diffusion 1.5 文生图流程（纯逻辑，与运行时无关）。
 * 与 diffusers 的 OnnxStableDiffusionPipeline 对齐：
 *   - 文本编码：先编码提示词、再编码负面提示词（各自 batch=1）；
 *   - 无分类器引导（CFG）：guidance > 1 时 UNet 输入为 [无条件, 条件] 两份，输出 noise = u + g·(c − u)；
 *   - 调度器：由 createScheduler 决定（PNDM、DPM-Solver++、DEIS、Euler，见 schedulers.js）；
 *     初始噪声乘以 initNoiseSigma，UNet 输入经 scaleModelInput 缩放（Euler 需要）；
 *   - 解码：latent / latentScale（SD 1.x 为 0.18215；TAESD 为 1.0），输出约在 [-1, 1]。
 * 运行时（ONNX 会话）通过 runtime 参数注入，便于用假模型测试。
 */
import { gaussianNoise } from './rng.js';
import { sizeProblem } from './sizes.js';

export const SD15_LATENT_SCALE = 0.18215;
export const SD15_MAX_STEPS = 50;
/** 调度器未声明 minSteps 时的下限（PNDM 需要至少 2 步）。 */
const DEFAULT_MIN_STEPS = 2;

/**
 * @param {object} p
 * @param {{encode(ids:Int32Array):Promise<Float32Array>, unet(args:object):Promise<Float32Array>, decode(latent:Float32Array,h8:number,w8:number):Promise<Float32Array>}} p.runtime
 * @param {{encode(text:string):Int32Array}} p.tokenizer
 * @param {{setTimesteps(n:number):number[], step(out:ArrayLike<number>,t:number,sample:ArrayLike<number>):Float64Array, initNoiseSigma:number, minSteps?:number, scaleModelInput?(sample:ArrayLike<number>,t:number):ArrayLike<number>}} p.scheduler
 * @param {number} [p.latentScale] 解码前除以的潜变量缩放（默认 SD 1.x 的 0.18215；SDXS 的 TAESD 为 1.0）
 * @returns {Promise<{width:number,height:number,chw:Float32Array}>} chw 为 [3, H, W]，取值约在 [-1, 1]
 */
export async function generateSd15({
  runtime,
  tokenizer,
  scheduler,
  prompt,
  negative = '',
  width,
  height,
  steps = 20,
  guidance = 7.5,
  seed,
  latentScale = SD15_LATENT_SCALE,
  onStep,
  signal,
}) {
  const problem = sizeProblem(width, height);
  if (problem) throw new Error(problem);
  const minSteps = scheduler.minSteps ?? DEFAULT_MIN_STEPS;
  if (!Number.isInteger(steps) || steps < minSteps || steps > SD15_MAX_STEPS) {
    throw new Error(`步数必须是 ${minSteps}–${SD15_MAX_STEPS} 之间的整数`);
  }
  if (!(latentScale > 0)) throw new Error('潜变量缩放必须是正数');
  if (!Number.isFinite(guidance) || guidance < 0) throw new Error('引导系数必须是非负数字');
  if (!Number.isInteger(seed) || seed < 0) throw new Error('种子必须是非负整数');
  const h8 = height / 8;
  const w8 = width / 8;
  const latentLen = 4 * h8 * w8;
  const useCfg = guidance > 1;

  throwIfAborted(signal);
  const condition = await runtime.encode(tokenizer.encode(prompt));
  let context = condition;
  if (useCfg) {
    throwIfAborted(signal);
    const uncondition = await runtime.encode(tokenizer.encode(negative));
    context = concat(uncondition, condition);
  }

  // 先设置时间步：Euler 等调度器的初始噪声乘数取决于时间步
  const timesteps = scheduler.setTimesteps(steps);
  let latents = Float64Array.from(gaussianNoise(seed, latentLen), (v) => v * scheduler.initNoiseSigma);
  for (let i = 0; i < timesteps.length; i += 1) {
    throwIfAborted(signal);
    const t = timesteps[i];
    const model = scheduler.scaleModelInput ? scheduler.scaleModelInput(latents, t) : latents;
    const sample = useCfg ? concat(model, model) : model;
    const noise = await runtime.unet({
      sample: Float32Array.from(sample),
      timestep: t,
      context,
      batch: useCfg ? 2 : 1,
      h8,
      w8,
    });
    let guided = noise;
    if (useCfg) {
      const half = noise.length / 2;
      guided = new Float64Array(half);
      for (let j = 0; j < half; j += 1) {
        const u = noise[j];
        guided[j] = u + guidance * (noise[half + j] - u);
      }
    }
    latents = scheduler.step(guided, t, latents);
    onStep?.({ step: i + 1, total: timesteps.length });
  }

  throwIfAborted(signal);
  const scaled = Float32Array.from(latents, (v) => v / latentScale);
  const chw = await runtime.decode(scaled, h8, w8);
  if (chw.length !== 3 * height * width) throw new Error(`解码结果尺寸不正确：${chw.length}`);
  return { width, height, chw };
}

/** [3, H, W] 的 [-1, 1] 数据 → RGBA（Uint8ClampedArray，alpha 为 255）。与 diffusers 的后处理一致：clip(x/2+0.5, 0, 1)。 */
export function chwToRgba(chw, width, height) {
  const plane = width * height;
  const out = new Uint8ClampedArray(plane * 4);
  for (let p = 0; p < plane; p += 1) {
    for (let c = 0; c < 3; c += 1) {
      const v = Math.min(1, Math.max(0, chw[c * plane + p] / 2 + 0.5));
      out[p * 4 + c] = Math.round(v * 255);
    }
    out[p * 4 + 3] = 255;
  }
  return out;
}

function concat(a, b) {
  const out = new Float32Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

function throwIfAborted(signal) {
  if (signal?.aborted) {
    const err = new Error('已取消生图');
    err.name = 'AbortError';
    throw err;
  }
}
