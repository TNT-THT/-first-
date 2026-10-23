// 调度器测试：DPM-Solver++、DEIS、Euler 与 diffusers 0.41.0 逐步对照（数据见 test/fixtures/image-schedulers.json，
// 由 scripts/image/scheduler_fixtures.py 生成）。另测试 -Infinity 的解析、按 _class_name 分派与不支持配置的报错。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import {
  createScheduler,
  createDpmSolverScheduler,
  createDeisScheduler,
  createEulerScheduler,
  parseSchedulerConfig,
  roundHalfEven,
} from '../web/src/image/schedulers.js';
import { createPndmScheduler } from '../web/src/image/pndm.js';
import { generateSd15 } from '../web/src/image/sd15.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(fs.readFileSync(path.join(here, 'fixtures', 'image-schedulers.json'), 'utf8'));

const FACTORY = {
  'DPMSolverMultistepScheduler': createDpmSolverScheduler,
  'DEISMultistepScheduler': createDeisScheduler,
  'EulerDiscreteScheduler': createEulerScheduler,
};

function close(actual, expected, tol, label) {
  assert.equal(actual.length, expected.length, `${label}：长度不同`);
  for (let i = 0; i < expected.length; i += 1) {
    const e = expected[i];
    const a = actual[i];
    const err = Math.abs(a - e) / Math.max(1, Math.abs(e));
    assert.ok(err <= tol, `${label}[${i}]：实际 ${a}，期望 ${e}，相对误差 ${err}`);
  }
}

for (const c of fixture.cases) {
  test(`与 diffusers 一致：${c.scheduler}，${c.steps} 步`, () => {
    const s = FACTORY[c.className](parseSchedulerConfig(JSON.stringify(c.config)));
    const timesteps = s.setTimesteps(c.steps);
    assert.deepEqual(timesteps, c.timesteps, '时间步');
    assert.ok(Math.abs(s.initNoiseSigma - c.initNoiseSigma) < 1e-6 * c.initNoiseSigma, '初始噪声乘数');
    // 与 diffusers 的 float32 计算相比，误差应在 1e-5 量级
    let latents = Float64Array.from(c.sample0, (v) => v * s.initNoiseSigma);
    c.perStep.forEach((st, i) => {
      assert.equal(st.timestep, timesteps[i], `第 ${i} 步时间步`);
      const scaled = s.scaleModelInput(latents, st.timestep);
      close(scaled, st.scaledInput, 1e-5, `第 ${i} 步 UNet 输入`);
      const next = s.step(Float64Array.from(st.output), st.timestep, latents);
      close(next, st.next, 1e-5, `第 ${i} 步输出`);
      latents = next;
    });
  });
}

test('roundHalfEven 与 numpy 的 round 一致', () => {
  assert.equal(roundHalfEven(0.5), 0);
  assert.equal(roundHalfEven(1.5), 2);
  assert.equal(roundHalfEven(2.5), 2);
  assert.equal(roundHalfEven(499.5), 500);
  assert.equal(roundHalfEven(998.4), 998);
  assert.equal(roundHalfEven(998.6), 999);
});

test('parseSchedulerConfig 可以读取含 -Infinity 的配置（tiny-sd 的配置即如此）', () => {
  const cfg = parseSchedulerConfig('{"_class_name":"DPMSolverMultistepScheduler","lambda_min_clipped": -Infinity,"x": 1}');
  assert.equal(cfg.lambda_min_clipped, null);
  assert.equal(cfg._class_name, 'DPMSolverMultistepScheduler');
  assert.equal(cfg.x, 1);
});

test('createScheduler 按 _class_name 分派；缺少 _class_name 时按 PNDM', () => {
  const dpm = createScheduler({ _class_name: 'DPMSolverMultistepScheduler', solver_order: 2 });
  assert.equal(dpm.minSteps, 1);
  const euler = createScheduler({ _class_name: 'EulerDiscreteScheduler', timestep_spacing: 'trailing' });
  assert.equal(euler.minSteps, 1);
  const pndm = createScheduler({ skip_prk_steps: true, steps_offset: 1 });
  assert.equal(pndm.minSteps === undefined ? 2 : pndm.minSteps, 2);
  assert.throws(() => createScheduler({ _class_name: 'LCMScheduler' }), /不支持的调度器：LCMScheduler/);
});

test('不支持的配置直接报错，而不是静默给出不同结果', () => {
  assert.throws(() => createDpmSolverScheduler({ algorithm_type: 'sde-dpmsolver++' }), /不支持 algorithm_type/);
  assert.throws(() => createDpmSolverScheduler({ solver_order: 3 }), /solver_order/);
  assert.throws(() => createDpmSolverScheduler({ use_karras_sigmas: true }), /use_karras_sigmas/);
  assert.throws(() => createDpmSolverScheduler({ thresholding: true }), /thresholding/);
  assert.throws(() => createDpmSolverScheduler({ prediction_type: 'v_prediction' }), /prediction_type/);
  assert.throws(() => createDeisScheduler({ timestep_spacing: 'trailing' }), /timestep_spacing/);
  assert.throws(() => createEulerScheduler({ timestep_spacing: 'linspace' }), /timestep_spacing/);
});

test('步数过少时报错；未设置时间步就调用 step 报错', () => {
  const s = createDpmSolverScheduler({});
  assert.throws(() => s.setTimesteps(0), /不小于 1/);
  const fresh = createEulerScheduler({ timestep_spacing: 'trailing' });
  assert.throws(() => fresh.step(new Float64Array(2), 999, new Float64Array(2)), /setTimesteps/);
});

test('Euler 的 initNoiseSigma 在设置时间步之后等于最大 sigma', () => {
  const c = fixture.cases.find((x) => x.scheduler === 'euler-trailing' && x.steps === 1);
  const s = createEulerScheduler(c.config);
  assert.equal(s.initNoiseSigma, 1.0); // 设置时间步之前
  s.setTimesteps(1);
  assert.ok(Math.abs(s.initNoiseSigma - c.initNoiseSigma) < 1e-5);
});

// ---------- 生成流程（假模型，只检查流程与形状） ----------

function fakeRuntime(seen) {
  return {
    async encode() {
      return new Float32Array(77 * 8);
    },
    async unet({ sample }) {
      return Float32Array.from(sample, (v) => v * 0.1);
    },
    async decode(latent, h8, w8) {
      seen?.push(latent);
      return new Float32Array(3 * h8 * 8 * w8 * 8);
    },
  };
}
const fakeTokenizer = { encode: () => new Int32Array(77) };

test('生成流程可以使用 DPM-Solver++、DEIS、Euler（含一步模型）', async () => {
  for (const c of fixture.cases.filter((x) => x.steps <= 4)) {
    const scheduler = FACTORY[c.className](c.config);
    const res = await generateSd15({
      runtime: fakeRuntime(),
      tokenizer: fakeTokenizer,
      scheduler,
      prompt: 'x',
      width: 256,
      height: 256,
      steps: c.steps,
      guidance: 1,
      seed: 7,
      latentScale: 1,
    });
    assert.equal(res.chw.length, 3 * 256 * 256, c.scheduler);
    assert.ok(res.chw.every((v) => Number.isFinite(v)), `${c.scheduler} 输出应为有限数`);
  }
});

test('解码前除以 latentScale：SDXS 的 TAESD 为 1.0，SD 1.x 为 0.18215', async () => {
  const cfg = fixture.cases.find((x) => x.scheduler === 'euler-trailing' && x.steps === 1).config;
  const run = async (latentScale) => {
    const seen = [];
    await generateSd15({
      runtime: fakeRuntime(seen),
      tokenizer: fakeTokenizer,
      scheduler: createEulerScheduler(cfg),
      prompt: 'x',
      width: 256,
      height: 256,
      steps: 1,
      guidance: 1,
      seed: 7,
      latentScale,
    });
    return seen[0];
  };
  const a = await run(1);
  const b = await run(0.18215);
  // 同一随机种子：两次 decode 的输入应相差 1/0.18215 倍
  assert.ok(Math.abs(b[0] / a[0] - 1 / 0.18215) < 1e-4);
});

test('步数下限：PNDM 为 2，Euler/DPM/DEIS 为 1', async () => {
  const base = { runtime: fakeRuntime(), tokenizer: fakeTokenizer, prompt: 'x', width: 256, height: 256, guidance: 1, seed: 1 };
  await assert.rejects(
    generateSd15({ ...base, scheduler: createPndmScheduler({ skip_prk_steps: true }), steps: 1 }),
    /步数必须是 2–50/,
  );
  await generateSd15({ ...base, scheduler: createEulerScheduler(fixture.cases[7].config), steps: 1 });
  await generateSd15({ ...base, scheduler: createDpmSolverScheduler(fixture.cases[0].config), steps: 1 });
  await assert.rejects(
    generateSd15({ ...base, scheduler: createDeisScheduler(fixture.cases[4].config), steps: 60 }),
    /步数必须是 1–50/,
  );
});
