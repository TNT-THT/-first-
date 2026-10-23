// SD 1.5 会话创建的回退逻辑：WebGPU（basic 图优化）失败时退回 CPU；数据只读取一次；错误信息完整。
// 使用假的 ort 模块，不需要真实模型或浏览器。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createSd15Runtime, sessionPlans, SD15_SESSION_NAMES } from '../web/src/image/sd15-runtime.js';

const INPUTS = {
  te: ['input_ids'],
  unet: ['sample', 'timestep', 'encoder_hidden_states'],
  vae: ['latent_sample'],
};

/** failOn: 形如 {'te|WebGPU': 'Pow 算子错误'} 的失败表。 */
function fakeOrt(failOn = {}) {
  const created = [];
  let released = 0;
  return {
    created,
    get released() {
      return released;
    },
    InferenceSession: {
      async create(src, opts) {
        const plan = opts.executionProviders[0] === 'webgpu' ? 'WebGPU' : 'CPU';
        created.push({ src, opts, plan });
        const key = `${src}|${plan}`;
        if (failOn[key]) throw new Error(failOn[key]);
        return {
          inputNames: INPUTS[src],
          inputMetadata: [],
          outputNames: ['out'],
          async run() {
            return { out: { type: 'float32', data: new Float32Array(1) } };
          },
          release() {
            released += 1;
          },
        };
      },
    },
    Tensor: class {
      constructor(type, data, dims) {
        Object.assign(this, { type, data, dims });
      }
    },
  };
}

const SOURCES = (counter) => ({
  textEncoder: () => {
    counter.te += 1;
    return 'te';
  },
  unet: 'unet',
  unetExternal: { path: 'weights.pb', data: 'weights-data' },
  vaeDecoder: 'vae',
});

test('执行方案：WebGPU 使用 basic 图优化，CPU 为最后的退路', () => {
  assert.deepEqual(sessionPlans(['webgpu']), [
    { label: 'WebGPU', executionProviders: ['webgpu'], graphOptimizationLevel: 'basic' },
    { label: 'CPU', executionProviders: ['wasm'], graphOptimizationLevel: 'all' },
  ]);
  assert.deepEqual(sessionPlans(['wasm']), [{ label: 'CPU', executionProviders: ['wasm'], graphOptimizationLevel: 'all' }]);
});

test('WebGPU 全部成功：三个会话都使用 WebGPU，且不混用 wasm', async () => {
  const ort = fakeOrt();
  const counter = { te: 0 };
  const events = [];
  const rt = await createSd15Runtime({ ort, sources: SOURCES(counter), executionProviders: ['webgpu'], onSession: (e) => events.push(e) });
  assert.deepEqual(rt.plans, { textEncoder: 'WebGPU', unet: 'WebGPU', vaeDecoder: 'WebGPU' });
  assert.equal(counter.te, 1, '文本编码器数据只读取一次');
  for (const c of ort.created) {
    assert.deepEqual(c.opts.executionProviders, ['webgpu']);
    assert.equal(c.opts.graphOptimizationLevel, 'basic');
  }
  const unetOpts = ort.created.find((c) => c.src === 'unet').opts;
  assert.deepEqual(unetOpts.externalData, [{ path: 'weights.pb', data: 'weights-data' }]);
  assert.deepEqual(
    events.filter((e) => e.phase === 'done').map((e) => e.key),
    ['textEncoder', 'unet', 'vaeDecoder'],
  );
  assert.equal(events[0].label, SD15_SESSION_NAMES.textEncoder);
});

test('文本编码器在 WebGPU 上失败：退回 CPU，数据不重新读取，并报告失败原因', async () => {
  const ort = fakeOrt({ 'te|WebGPU': 'Provider type for Pow node is not set' });
  const counter = { te: 0 };
  const events = [];
  const rt = await createSd15Runtime({ ort, sources: SOURCES(counter), executionProviders: ['webgpu'], onSession: (e) => events.push(e) });
  assert.equal(rt.plans.textEncoder, 'CPU');
  assert.equal(rt.plans.unet, 'WebGPU');
  assert.equal(counter.te, 1, '失败后复用已读取的数据');
  const teCalls = ort.created.filter((c) => c.src === 'te');
  assert.deepEqual(teCalls.map((c) => c.plan), ['WebGPU', 'CPU']);
  assert.deepEqual(teCalls[1].opts.executionProviders, ['wasm']);
  const failed = events.find((e) => e.phase === 'failed');
  assert.equal(failed.key, 'textEncoder');
  assert.match(failed.message, /Pow node/);
});

test('两种方案都失败：错误信息同时列出 WebGPU 与 CPU 的原因，并释放已创建的会话', async () => {
  const ort = fakeOrt({ 'te|WebGPU': 'GPU 错误', 'te|CPU': 'CPU 错误' });
  await assert.rejects(
    createSd15Runtime({ ort, sources: SOURCES({ te: 0 }), executionProviders: ['webgpu'] }),
    (err) => {
      assert.match(err.message, /无法创建文本编码器会话/);
      assert.match(err.message, /WebGPU：GPU 错误/);
      assert.match(err.message, /CPU：CPU 错误/);
      return true;
    },
  );
});

test('UNet 失败时，已创建的文本编码器会话被释放', async () => {
  const ort = fakeOrt({ 'unet|WebGPU': 'u1', 'unet|CPU': 'u2' });
  await assert.rejects(createSd15Runtime({ ort, sources: SOURCES({ te: 0 }), executionProviders: ['webgpu'] }), /无法创建 ?UNet/);
  assert.equal(ort.released, 1);
});

test('只使用 wasm 时（Node 测试）：没有 WebGPU 方案', async () => {
  const ort = fakeOrt();
  const rt = await createSd15Runtime({ ort, sources: SOURCES({ te: 0 }), executionProviders: ['wasm'] });
  assert.deepEqual(rt.plans, { textEncoder: 'CPU', unet: 'CPU', vaeDecoder: 'CPU' });
  assert.ok(ort.created.every((c) => c.plan === 'CPU'));
});
