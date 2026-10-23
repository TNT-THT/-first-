/**
 * 扩散调度器（纯逻辑），与 diffusers 0.41.0 对齐，供 SD 1.x 结构的模型使用：
 *   - DPMSolverMultistepScheduler：algorithm_type = dpmsolver++，2 阶，solver_type = midpoint 或 heun，
 *     时间步 linspace 或 trailing，epsilon 预测，lower_order_final；
 *   - DEISMultistepScheduler：algorithm_type = deis（logrho），2 阶，时间步 linspace，epsilon 预测；
 *   - EulerDiscreteScheduler：时间步 trailing，epsilon 预测，无随机性（s_churn = 0）。
 * PNDM 由 pndm.js 实现。不支持的配置直接报错，而不是静默给出不同的结果。
 * 逐步结果与 diffusers 的对照见 test/image-schedulers.test.js（数据由 scripts/image/scheduler_fixtures.py 生成）。
 *
 * 统一接口（与 pndm.js 相同，另有 minSteps 与 scaleModelInput）：
 *   initNoiseSigma        初始噪声的乘数（Euler 为最大 sigma，其余为 1）
 *   minSteps              允许的最少推理步数
 *   setTimesteps(n)       设置推理步数并返回时间步（降序）；会重置内部状态
 *   scaleModelInput(s, t) UNet 输入的缩放（Euler 需要，其余原样返回）
 *   step(out, t, sample)  返回上一时刻的样本（Float64Array）
 */
import { createPndmScheduler } from './pndm.js';

/**
 * 解析 scheduler_config.json。Hugging Face 上的配置由 Python 写成，可能含有 -Infinity（JSON 中不合法），
 * 这里把它替换为 null；lambda_min_clipped 的 null 即默认值 -∞。
 */
export function parseSchedulerConfig(text) {
  return JSON.parse(String(text).replace(/-?Infinity|NaN/g, 'null'));
}

/** 按 _class_name 创建调度器。缺少 _class_name 时按 PNDM 处理（与原有配置一致）。 */
export function createScheduler(config = {}) {
  const cls = config._class_name || 'PNDMScheduler';
  switch (cls) {
    case 'PNDMScheduler':
      return createPndmScheduler(config);
    case 'DPMSolverMultistepScheduler':
      return createDpmSolverScheduler(config);
    case 'DEISMultistepScheduler':
      return createDeisScheduler(config);
    case 'EulerDiscreteScheduler':
      return createEulerScheduler(config);
    default:
      throw new Error(`不支持的调度器：${cls}`);
  }
}

// ---------- 公共部分 ----------

/** numpy 的 round：四舍五入到偶数（0.5 → 0，1.5 → 2，2.5 → 2）。 */
export function roundHalfEven(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/** 与 diffusers 相同的 betas 与 alphas_cumprod（linear 或 scaled_linear）。 */
function alphasCumprodOf(c) {
  const N = c.num_train_timesteps ?? 1000;
  if (c.trained_betas) throw new Error('调度器：不支持 trained_betas');
  if (c.rescale_betas_zero_snr) throw new Error('调度器：不支持 rescale_betas_zero_snr');
  const b0 = c.beta_start ?? 0.0001;
  const b1 = c.beta_end ?? 0.02;
  const alphas = new Float64Array(N);
  let acc = 1;
  for (let i = 0; i < N; i += 1) {
    let beta;
    const schedule = c.beta_schedule ?? 'linear'; // diffusers 的默认值
    if (schedule === 'scaled_linear') {
      const a = Math.sqrt(b0);
      const b = Math.sqrt(b1);
      beta = (a + ((b - a) * i) / (N - 1)) ** 2;
    } else if (schedule === 'linear') {
      beta = b0 + ((b1 - b0) * i) / (N - 1);
    } else {
      throw new Error(`调度器：不支持 beta_schedule=${schedule}`);
    }
    acc *= 1 - beta;
    alphas[i] = acc;
  }
  return { N, alphasCumprod: alphas };
}

/** 每个离散时间步的 sigma：sqrt((1 - ᾱ) / ᾱ)。 */
function sigmasAllOf(alphasCumprod) {
  return Float64Array.from(alphasCumprod, (a) => Math.sqrt((1 - a) / a));
}

/** linspace 时间步（last_timestep = N）：linspace(0, N−1, n+1).round()[::-1][:-1]。 */
function linspaceTimesteps(N, n) {
  const step = (N - 1) / n;
  const out = [];
  for (let k = n; k >= 1; k -= 1) out.push(roundHalfEven(k * step));
  return out;
}

/** trailing 时间步：arange(N, 0, −N/n).round() − 1（arange 的长度与 numpy 相同）。 */
function trailingTimesteps(N, n) {
  const ratio = N / n;
  const len = Math.ceil(N / ratio);
  const out = [];
  for (let i = 0; i < len; i += 1) out.push(roundHalfEven(N - i * ratio) - 1);
  return out;
}

function requireSteps(steps, min) {
  if (!Number.isInteger(steps) || steps < min) throw new Error(`推理步数必须是不小于 ${min} 的整数`);
}

/** diffusers 的 _sigma_to_alpha_sigma_t（非 flow 模式）：alpha = 1/√(σ²+1)，sigma_t = σ·alpha。 */
function alphaSigma(sigma) {
  const alpha = 1 / Math.sqrt(sigma * sigma + 1);
  return { alpha, sigmaT: sigma * alpha };
}

function unsupported(c, keys) {
  for (const k of keys) {
    if (c[k]) throw new Error(`调度器：不支持 ${k}=${JSON.stringify(c[k])}`);
  }
}

// ---------- DPM-Solver++（多步） ----------

export function createDpmSolverScheduler(raw = {}) {
  const c = {
    solver_order: 2,
    algorithm_type: 'dpmsolver++',
    solver_type: 'midpoint',
    prediction_type: 'epsilon',
    lower_order_final: true,
    euler_at_final: false,
    final_sigmas_type: 'zero',
    lambda_min_clipped: null,
    timestep_spacing: 'linspace',
    ...raw,
  };
  if (c.algorithm_type !== 'dpmsolver++') throw new Error(`DPM-Solver：不支持 algorithm_type=${c.algorithm_type}`);
  if (c.solver_order !== 2) throw new Error(`DPM-Solver：本实现只支持 solver_order = 2（当前 ${c.solver_order}）`);
  if (c.prediction_type !== 'epsilon') throw new Error(`DPM-Solver：不支持 prediction_type=${c.prediction_type}`);
  if (!['midpoint', 'heun'].includes(c.solver_type)) throw new Error(`DPM-Solver：不支持 solver_type=${c.solver_type}`);
  if (!['zero', 'sigma_min'].includes(c.final_sigmas_type)) {
    throw new Error(`DPM-Solver：不支持 final_sigmas_type=${c.final_sigmas_type}`);
  }
  if (c.lambda_min_clipped !== null && c.lambda_min_clipped !== undefined) {
    throw new Error('DPM-Solver：本实现只支持默认的 lambda_min_clipped（-∞）');
  }
  if (!['linspace', 'trailing'].includes(c.timestep_spacing)) {
    throw new Error(`DPM-Solver：不支持 timestep_spacing=${c.timestep_spacing}`);
  }
  if (c.thresholding) throw new Error('DPM-Solver：不支持 thresholding');
  unsupported(c, ['use_karras_sigmas', 'use_exponential_sigmas', 'use_beta_sigmas', 'use_lu_lambdas', 'use_flow_sigmas', 'variance_type']);

  const { N, alphasCumprod } = alphasCumprodOf(c);
  const sigmasAll = sigmasAllOf(alphasCumprod);
  const finalSigma = c.final_sigmas_type === 'zero' ? 0 : sigmasAll[0];

  let timesteps = [];
  let sigmas = [];
  let n = 0;
  let stepIndex = 0;
  let lowerOrderNums = 0;
  let prevOutputs = [null, null];

  /** 数据预测：x0 = (sample − σ_t·ε) / α_t。 */
  function convert(modelOutput, sample, i) {
    const { alpha, sigmaT } = alphaSigma(sigmas[i]);
    const out = new Float64Array(sample.length);
    for (let j = 0; j < sample.length; j += 1) out[j] = (sample[j] - sigmaT * modelOutput[j]) / alpha;
    return out;
  }

  function firstOrder(m, sample, i) {
    const t = alphaSigma(sigmas[i + 1]);
    const s = alphaSigma(sigmas[i]);
    const lambdaT = Math.log(t.alpha) - Math.log(t.sigmaT);
    const lambdaS = Math.log(s.alpha) - Math.log(s.sigmaT);
    const h = lambdaT - lambdaS;
    const ratio = t.sigmaT / s.sigmaT;
    const coef = t.alpha * (Math.exp(-h) - 1);
    const out = new Float64Array(sample.length);
    for (let j = 0; j < sample.length; j += 1) out[j] = ratio * sample[j] - coef * m[j];
    return out;
  }

  function secondOrder(sample, i) {
    const t = alphaSigma(sigmas[i + 1]);
    const s0 = alphaSigma(sigmas[i]);
    const s1 = alphaSigma(sigmas[i - 1]);
    const lambdaT = Math.log(t.alpha) - Math.log(t.sigmaT);
    const lambdaS0 = Math.log(s0.alpha) - Math.log(s0.sigmaT);
    const lambdaS1 = Math.log(s1.alpha) - Math.log(s1.sigmaT);
    const m0 = prevOutputs[1];
    const m1 = prevOutputs[0];
    const h = lambdaT - lambdaS0;
    const h0 = lambdaS0 - lambdaS1;
    const r0 = h0 / h;
    const e = Math.exp(-h);
    const ratio = t.sigmaT / s0.sigmaT;
    const out = new Float64Array(sample.length);
    for (let j = 0; j < sample.length; j += 1) {
      const d0 = m0[j];
      const d1 = (1 / r0) * (m0[j] - m1[j]);
      if (c.solver_type === 'midpoint') {
        out[j] = ratio * sample[j] - t.alpha * (e - 1) * d0 - 0.5 * (t.alpha * (e - 1)) * d1;
      } else {
        out[j] = ratio * sample[j] - t.alpha * (e - 1) * d0 + t.alpha * ((e - 1) / h + 1) * d1;
      }
    }
    return out;
  }

  return {
    get initNoiseSigma() {
      return 1.0;
    },
    minSteps: 1,
    get timesteps() {
      return timesteps;
    },
    setTimesteps(steps) {
      requireSteps(steps, 1);
      n = steps;
      timesteps = c.timestep_spacing === 'trailing' ? trailingTimesteps(N, steps) : linspaceTimesteps(N, steps);
      sigmas = [...timesteps.map((t) => sigmasAll[t]), finalSigma];
      stepIndex = 0;
      lowerOrderNums = 0;
      prevOutputs = [null, null];
      return timesteps;
    },
    scaleModelInput(sample) {
      return sample;
    },
    step(modelOutput, timestep, sample) {
      if (n === 0) throw new Error('请先调用 setTimesteps');
      const i = stepIndex;
      const lowerFinal = i === n - 1 && (c.euler_at_final || (c.lower_order_final && n < 15) || c.final_sigmas_type === 'zero');
      const x0 = convert(modelOutput, sample, i);
      prevOutputs = [prevOutputs[1], x0];
      let out;
      // solver_order = 2：与 diffusers 一致，非首步且非末步时进入二阶分支
      if (lowerOrderNums < 1 || lowerFinal) out = firstOrder(x0, sample, i);
      else out = secondOrder(sample, i);
      if (lowerOrderNums < c.solver_order) lowerOrderNums += 1;
      stepIndex += 1;
      return out;
    },
  };
}

// ---------- DEIS（logrho，多步） ----------

export function createDeisScheduler(raw = {}) {
  const c = {
    solver_order: 2,
    algorithm_type: 'deis',
    solver_type: 'logrho',
    prediction_type: 'epsilon',
    lower_order_final: true,
    timestep_spacing: 'linspace',
    ...raw,
  };
  if (c.algorithm_type !== 'deis') throw new Error(`DEIS：不支持 algorithm_type=${c.algorithm_type}`);
  if (c.solver_order !== 2) throw new Error(`DEIS：本实现只支持 solver_order = 2（当前 ${c.solver_order}）`);
  if (c.solver_type !== 'logrho') throw new Error(`DEIS：不支持 solver_type=${c.solver_type}`);
  if (c.prediction_type !== 'epsilon') throw new Error(`DEIS：不支持 prediction_type=${c.prediction_type}`);
  if (c.timestep_spacing !== 'linspace') throw new Error(`DEIS：不支持 timestep_spacing=${c.timestep_spacing}`);
  if (c.thresholding) throw new Error('DEIS：不支持 thresholding');
  unsupported(c, ['use_karras_sigmas', 'use_exponential_sigmas', 'use_beta_sigmas', 'use_flow_sigmas', 'lambda_min_clipped']);

  const { N, alphasCumprod } = alphasCumprodOf(c);
  const sigmasAll = sigmasAllOf(alphasCumprod);
  const finalSigma = sigmasAll[0]; // DEIS 的末端 sigma 固定为 sigma_min

  let timesteps = [];
  let sigmas = [];
  let n = 0;
  let stepIndex = 0;
  let lowerOrderNums = 0;
  let prevOutputs = [null, null];

  function convert(modelOutput, sample, i) {
    const { alpha, sigmaT } = alphaSigma(sigmas[i]);
    const out = new Float64Array(sample.length);
    for (let j = 0; j < sample.length; j += 1) {
      const x0 = (sample[j] - sigmaT * modelOutput[j]) / alpha;
      out[j] = (sample[j] - alpha * x0) / sigmaT;
    }
    return out;
  }

  function firstOrder(m, sample, i) {
    const t = alphaSigma(sigmas[i + 1]);
    const s = alphaSigma(sigmas[i]);
    const lambdaT = Math.log(t.alpha) - Math.log(t.sigmaT);
    const lambdaS = Math.log(s.alpha) - Math.log(s.sigmaT);
    const h = lambdaT - lambdaS;
    const out = new Float64Array(sample.length);
    for (let j = 0; j < sample.length; j += 1) {
      out[j] = (t.alpha / s.alpha) * sample[j] - t.sigmaT * (Math.exp(h) - 1) * m[j];
    }
    return out;
  }

  function secondOrder(sample, i) {
    const t = alphaSigma(sigmas[i + 1]);
    const s0 = alphaSigma(sigmas[i]);
    const s1 = alphaSigma(sigmas[i - 1]);
    const rhoT = t.sigmaT / t.alpha;
    const rhoS0 = s0.sigmaT / s0.alpha;
    const rhoS1 = s1.sigmaT / s1.alpha;
    // ∫ (log t − log c)/(log b − log c) 的原函数（diffusers 中的 ind_fn）
    const ind = (x, b, cc) => (x * (-Math.log(cc) + Math.log(x) - 1)) / (Math.log(b) - Math.log(cc));
    const coef1 = ind(rhoT, rhoS0, rhoS1) - ind(rhoS0, rhoS0, rhoS1);
    const coef2 = ind(rhoT, rhoS1, rhoS0) - ind(rhoS0, rhoS1, rhoS0);
    const m0 = prevOutputs[1];
    const m1 = prevOutputs[0];
    const out = new Float64Array(sample.length);
    for (let j = 0; j < sample.length; j += 1) {
      out[j] = t.alpha * (sample[j] / s0.alpha + coef1 * m0[j] + coef2 * m1[j]);
    }
    return out;
  }

  return {
    get initNoiseSigma() {
      return 1.0;
    },
    minSteps: 1,
    get timesteps() {
      return timesteps;
    },
    setTimesteps(steps) {
      requireSteps(steps, 1);
      n = steps;
      timesteps = linspaceTimesteps(N, steps);
      sigmas = [...timesteps.map((t) => sigmasAll[t]), finalSigma];
      stepIndex = 0;
      lowerOrderNums = 0;
      prevOutputs = [null, null];
      return timesteps;
    },
    scaleModelInput(sample) {
      return sample;
    },
    step(modelOutput, timestep, sample) {
      if (n === 0) throw new Error('请先调用 setTimesteps');
      const i = stepIndex;
      const lowerFinal = i === n - 1 && c.lower_order_final && n < 15;
      const eps = convert(modelOutput, sample, i);
      prevOutputs = [prevOutputs[1], eps];
      let out;
      // solver_order = 2：与 DPM-Solver++ 相同，非首步且非末步时进入二阶分支
      if (lowerOrderNums < 1 || lowerFinal) out = firstOrder(eps, sample, i);
      else out = secondOrder(sample, i);
      if (lowerOrderNums < c.solver_order) lowerOrderNums += 1;
      stepIndex += 1;
      return out;
    },
  };
}

// ---------- Euler（离散，trailing） ----------

export function createEulerScheduler(raw = {}) {
  const c = {
    prediction_type: 'epsilon',
    timestep_spacing: 'trailing',
    interpolation_type: 'linear',
    ...raw,
  };
  if (c.prediction_type !== 'epsilon') throw new Error(`Euler：不支持 prediction_type=${c.prediction_type}`);
  if (c.timestep_spacing !== 'trailing') throw new Error(`Euler：不支持 timestep_spacing=${c.timestep_spacing}`);
  if (c.interpolation_type !== 'linear') throw new Error(`Euler：不支持 interpolation_type=${c.interpolation_type}`);
  unsupported(c, ['use_karras_sigmas', 'use_exponential_sigmas', 'use_beta_sigmas', 'use_flow_sigmas', 'sigma_max', 'sigma_min']);

  const { N, alphasCumprod } = alphasCumprodOf(c);
  const sigmasAll = sigmasAllOf(alphasCumprod);

  let timesteps = [];
  let sigmas = [];
  let n = 0;
  let stepIndex = 0;

  return {
    get initNoiseSigma() {
      return n ? sigmas[0] : 1.0; // 最大 sigma（sigma 随时间步递减）
    },
    minSteps: 1,
    get timesteps() {
      return timesteps;
    },
    setTimesteps(steps) {
      requireSteps(steps, 1);
      n = steps;
      timesteps = trailingTimesteps(N, steps);
      sigmas = [...timesteps.map((t) => sigmasAll[t]), 0];
      stepIndex = 0;
      return timesteps;
    },
    /** 输入按 1/√(σ²+1) 缩放（diffusers 的 scale_model_input）。 */
    scaleModelInput(sample) {
      const sigma = sigmas[stepIndex];
      const k = 1 / Math.sqrt(sigma * sigma + 1);
      return Float64Array.from(sample, (v) => v * k);
    },
    step(modelOutput, timestep, sample) {
      if (n === 0) throw new Error('请先调用 setTimesteps');
      const i = stepIndex;
      const sigma = sigmas[i];
      const out = new Float64Array(sample.length);
      for (let j = 0; j < sample.length; j += 1) {
        const pred = sample[j] - sigma * modelOutput[j];
        const derivative = (sample[j] - pred) / sigma;
        const dt = sigmas[i + 1] - sigma;
        out[j] = sample[j] + derivative * dt;
      }
      stepIndex += 1;
      return out;
    },
  };
}
