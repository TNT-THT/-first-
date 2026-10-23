/**
 * SD 1.5 的 ONNX Runtime 适配层：加载三个会话（文本编码器、UNet、VAE 解码器），
 * 按每个输入的实际数据类型（float32 / float16 / int32 / int64）创建张量，并检查输入名称。
 * ort 通过参数注入：浏览器 worker 使用 onnxruntime-web/webgpu，Node 测试使用同一个包的 wasm 后端。
 *
 * 会话创建的执行方案（sessionPlans）：
 *   1. WebGPU，图优化级别 basic。WebGPU 执行提供程序在 extended / all 级别的图优化下有已知问题
 *      （融合后的算子在 WebGPU 上输出异常或无法创建会话，见 microsoft/onnxruntime#32145）。
 *   2. 失败时退回 CPU（wasm，图优化 all）。CPU 较慢，但能让同一个模型继续工作。
 * 每个会话只使用单一执行提供程序，不再把 webgpu 与 wasm 混合（混合时 ORT 需要插入跨设备拷贝，
 * 这一步在部分模型上会因算子未分配执行提供程序而失败）。
 */
import { decodeFromType, encodeForType } from './tensor-types.js';

export const SD15_INPUTS = {
  textEncoder: ['input_ids'],
  unet: ['sample', 'timestep', 'encoder_hidden_states'],
  vaeDecoder: ['latent_sample'],
};
export const SD15_CONTEXT_LENGTH = 77;
export const SD15_SESSION_NAMES = { textEncoder: '文本编码器', unet: 'UNet', vaeDecoder: 'VAE 解码器' };

/**
 * 某个会话的执行方案（按顺序尝试）。executionProviders 含 'webgpu' 时先尝试 WebGPU，始终以 CPU 结尾。
 * @returns {Array<{label: string, executionProviders: string[], graphOptimizationLevel: string}>}
 */
export function sessionPlans(executionProviders = ['wasm']) {
  const plans = [];
  if (executionProviders.includes('webgpu')) {
    plans.push({ label: 'WebGPU', executionProviders: ['webgpu'], graphOptimizationLevel: 'basic' });
  }
  plans.push({ label: 'CPU', executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  return plans;
}

/**
 * @param {object} p
 * @param {object} p.ort            onnxruntime-web 模块（需提供 InferenceSession 与 Tensor）
 * @param {object} p.sources        {textEncoder, unet, unetExternal?, vaeDecoder}
 *                                  每项可以是 URL 字符串、Uint8Array，或返回其中之一的函数（按需读取，读取一次）。
 *                                  unetExternal 形如 {path: 'weights.pb', data: url|Uint8Array|function}
 * @param {object} [p.sources.textEncoderFallback] 文本编码器的备选文件（如 fp32）。仅当主文件的全部方案都无法创建会话时使用。
 * @param {string[]} [p.executionProviders]
 * @param {(e: object) => void} [p.onSession] 会话进度：{phase: 'start'|'done'|'failed', key, label, plan, index, total, message?}
 * @param {(e: {message: string}) => void} [p.onFallback] 文本编码器改用备选文件时调用（message 为主文件的失败原因）
 */
export async function createSd15Runtime({ ort, sources, executionProviders = ['wasm'], onSession, onFallback }) {
  const sessions = [];
  const plansUsed = {};
  const keys = ['textEncoder', 'unet', 'vaeDecoder'];
  const resolve = async (v) => (typeof v === 'function' ? v() : v);
  const open = async (key, src, extra = {}, index) => {
    const label = SD15_SESSION_NAMES[key];
    const errors = [];
    // 读取一次，之后的方案复用同一份数据
    const data = await resolve(src);
    for (const plan of sessionPlans(executionProviders)) {
      onSession?.({ phase: 'start', key, label, plan: plan.label, index, total: keys.length });
      try {
        const s = await ort.InferenceSession.create(data, {
          executionProviders: plan.executionProviders,
          graphOptimizationLevel: plan.graphOptimizationLevel,
          ...extra,
        });
        sessions.push(s);
        plansUsed[key] = plan.label;
        onSession?.({ phase: 'done', key, label, plan: plan.label, index, total: keys.length });
        return s;
      } catch (err) {
        const message = err?.message || String(err);
        errors.push(`${plan.label}：${message}`);
        onSession?.({ phase: 'failed', key, label, plan: plan.label, index, total: keys.length, message });
      }
    }
    throw new Error(`无法创建${label}会话。${errors.join('；')}`);
  };

  let textEncoder;
  let unet;
  let vaeDecoder;
  let textEncoderSource = 'primary';
  try {
    try {
      textEncoder = await open('textEncoder', sources.textEncoder, {}, 0);
    } catch (err) {
      if (!sources.textEncoderFallback) throw err;
      onFallback?.({ message: err?.message || String(err) });
      textEncoder = await open('textEncoder', sources.textEncoderFallback, {}, 0);
      textEncoderSource = 'fallback';
    }
    expectInputs(textEncoder, SD15_INPUTS.textEncoder, '文本编码器');
    const ext = sources.unetExternal;
    const unetExtra = ext ? { externalData: [{ path: ext.path, data: await resolve(ext.data) }] } : {};
    unet = await open('unet', sources.unet, unetExtra, 1);
    expectInputs(unet, SD15_INPUTS.unet, 'UNet');
    vaeDecoder = await open('vaeDecoder', sources.vaeDecoder, {}, 2);
    expectInputs(vaeDecoder, SD15_INPUTS.vaeDecoder, 'VAE 解码器');
  } catch (err) {
    for (const s of sessions) s.release?.();
    throw err;
  }

  const typeOf = (session, name) => session.inputMetadata.find((m) => m.name === name)?.type || 'float32';
  const tensor = (type, values, dims) => new ort.Tensor(type, encodeForType(type, values), dims);
  const firstOutput = (session, out) => out[session.outputNames[0]];
  const readFloat = (t) => decodeFromType(t.type, t.data);

  return {
    /** 每个会话实际使用的方案（WebGPU 或 CPU）。 */
    plans: plansUsed,
    /** 文本编码器使用的文件：'primary'（主文件）或 'fallback'（备选文件）。 */
    textEncoderSource,
    /** 文本编码：ids 为 [77] 的 int32，返回 [77 × D] 的 Float32Array。 */
    async encode(ids) {
      const type = typeOf(textEncoder, 'input_ids');
      const out = await textEncoder.run({ input_ids: tensor(type, ids, [1, SD15_CONTEXT_LENGTH]) });
      return readFloat(firstOutput(textEncoder, out));
    },
    /**
     * UNet 单步：sample 为 [batch, 4, h8, w8]，context 为 [batch, seqLength, D]，返回同形状的噪声预测。
     * seqLength 默认 77（单段）；分段编码时为 77 × 段数。固定形状的模型（如测试用微型模型）会拒绝不同的 seqLength，
     * 由调用方回退到单段。
     */
    async unet({ sample, timestep, context, batch, h8, w8, seqLength = SD15_CONTEXT_LENGTH }) {
      const hidden = context.length / (batch * seqLength);
      const feeds = {
        sample: tensor(typeOf(unet, 'sample'), sample, [batch, 4, h8, w8]),
        timestep: tensor(typeOf(unet, 'timestep'), [timestep], [1]),
        encoder_hidden_states: tensor(typeOf(unet, 'encoder_hidden_states'), context, [
          batch,
          seqLength,
          hidden,
        ]),
      };
      const out = await unet.run(feeds);
      return readFloat(firstOutput(unet, out));
    },
    /** VAE 解码：latent 为已除以 0.18215 的 [4, h8, w8]，返回 [3, 8·h8, 8·w8]，取值约在 [-1, 1]。 */
    async decode(latent, h8, w8) {
      const out = await vaeDecoder.run({
        latent_sample: tensor(typeOf(vaeDecoder, 'latent_sample'), latent, [1, 4, h8, w8]),
      });
      return readFloat(firstOutput(vaeDecoder, out));
    },
    /** 释放全部会话（释放显存）。 */
    release() {
      for (const s of sessions) s.release?.();
      sessions.length = 0;
    },
  };
}

function expectInputs(session, expected, label) {
  const got = session.inputNames;
  const missing = expected.filter((n) => !got.includes(n));
  if (missing.length) {
    throw new Error(`${label}的输入名称与预期不符：缺少 ${missing.join('、')}（实际为 ${got.join('、')}）`);
  }
}
