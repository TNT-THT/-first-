/**
 * PNDM 调度器（纯逻辑），对应 diffusers 的 PNDMScheduler，仅实现 SD 1.5 用到的配置：
 *   skip_prk_steps = true、prediction_type = epsilon、timestep_spacing = leading。
 * 其他配置会直接报错，而不是静默给出不同的结果。
 * 时间步、step_plms 与 _get_prev_sample 的公式与 diffusers 0.41.0 逐项对应（见 test/image-scheduler.test.js）。
 */
export function createPndmScheduler(config = {}) {
  const c = {
    num_train_timesteps: 1000,
    beta_start: 0.0001,
    beta_end: 0.02,
    beta_schedule: 'linear',
    skip_prk_steps: false,
    set_alpha_to_one: false,
    prediction_type: 'epsilon',
    timestep_spacing: 'leading',
    steps_offset: 0,
    ...config,
  };
  if (!c.skip_prk_steps) throw new Error('PNDM：本实现只支持 skip_prk_steps = true（SD 1.5 的配置）');
  if (c.prediction_type !== 'epsilon') throw new Error(`PNDM：不支持 prediction_type=${c.prediction_type}`);
  if (c.timestep_spacing !== 'leading') throw new Error(`PNDM：不支持 timestep_spacing=${c.timestep_spacing}`);

  const N = c.num_train_timesteps;
  const betas = new Float64Array(N);
  if (c.beta_schedule === 'scaled_linear') {
    // diffusers：linspace(sqrt(b0), sqrt(b1))^2
    const a = Math.sqrt(c.beta_start);
    const b = Math.sqrt(c.beta_end);
    for (let i = 0; i < N; i += 1) betas[i] = (a + ((b - a) * i) / (N - 1)) ** 2;
  } else if (c.beta_schedule === 'linear') {
    for (let i = 0; i < N; i += 1) betas[i] = c.beta_start + ((c.beta_end - c.beta_start) * i) / (N - 1);
  } else {
    throw new Error(`PNDM：不支持 beta_schedule=${c.beta_schedule}`);
  }
  const alphasCumprod = new Float64Array(N);
  let acc = 1;
  for (let i = 0; i < N; i += 1) {
    acc *= 1 - betas[i];
    alphasCumprod[i] = acc;
  }
  const finalAlpha = c.set_alpha_to_one ? 1 : alphasCumprod[0];

  let timesteps = [];
  let numInference = null;
  let ets = [];
  let counter = 0;
  let curSample = null;

  function prevSample(sample, t, prevT, modelOutput) {
    const aT = alphasCumprod[t];
    const aPrev = prevT >= 0 ? alphasCumprod[prevT] : finalAlpha;
    const bT = 1 - aT;
    const bPrev = 1 - aPrev;
    const sampleCoeff = Math.sqrt(aPrev / aT);
    const denom = aT * Math.sqrt(bPrev) + Math.sqrt(aT * bT * aPrev);
    const k = (aPrev - aT) / denom;
    const out = new Float64Array(sample.length);
    for (let i = 0; i < sample.length; i += 1) out[i] = sampleCoeff * sample[i] - k * modelOutput[i];
    return out;
  }

  return {
    initNoiseSigma: 1.0,
    get timesteps() {
      return timesteps;
    },
    /** 设置推理步数，返回时间步（降序）。与 diffusers 的 set_timesteps（skip_prk 模式）一致。 */
    setTimesteps(n) {
      if (!Number.isInteger(n) || n < 1) throw new Error('推理步数必须是正整数');
      numInference = n;
      const ratio = Math.floor(N / n);
      const base = [];
      for (let i = 0; i < n; i += 1) base.push(Math.round(i * ratio) + c.steps_offset);
      // plms = concat(ts[:-1], ts[-2:-1], ts[-1:])[::-1]
      timesteps = [...base.slice(0, -1), base[base.length - 2], base[base.length - 1]].reverse();
      ets = [];
      counter = 0;
      curSample = null;
      return timesteps;
    },
    /**
     * 一步去噪（step_plms）。modelOutput 与 sample 为等长数组，返回新的 Float64Array。
     * 前几步使用低阶公式，之后使用 4 阶线性多步公式。
     */
    step(modelOutput, timestep, sample) {
      if (numInference === null) throw new Error('请先调用 setTimesteps');
      const ratio = Math.floor(N / numInference);
      let prev = timestep - ratio;
      let t = timestep;
      if (counter !== 1) {
        ets = ets.slice(-3);
        ets.push(modelOutput);
      } else {
        prev = timestep;
        t = timestep + ratio;
      }
      let mo;
      let s = sample;
      if (ets.length === 1 && counter === 0) {
        mo = modelOutput;
        curSample = sample;
      } else if (ets.length === 1 && counter === 1) {
        mo = combine([modelOutput, ets[0]], (v) => (v[0] + v[1]) / 2);
        s = curSample;
        curSample = null;
      } else if (ets.length === 2) {
        mo = combine([ets[1], ets[0]], (v) => (3 * v[0] - v[1]) / 2);
      } else if (ets.length === 3) {
        mo = combine([ets[2], ets[1], ets[0]], (v) => (23 * v[0] - 16 * v[1] + 5 * v[2]) / 12);
      } else {
        mo = combine([ets[3], ets[2], ets[1], ets[0]], (v) => (55 * v[0] - 59 * v[1] + 37 * v[2] - 9 * v[3]) / 24);
      }
      const out = prevSample(s, t, prev, mo);
      counter += 1;
      return out;
    },
  };
}

/** 逐元素组合多个等长数组：f 接收每个位置上的值数组。 */
function combine(arrays, f) {
  const len = arrays[0].length;
  const out = new Float64Array(len);
  const v = new Array(arrays.length);
  for (let i = 0; i < len; i += 1) {
    for (let j = 0; j < arrays.length; j += 1) v[j] = arrays[j][i];
    out[i] = f(v);
  }
  return out;
}
