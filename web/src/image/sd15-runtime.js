/**
 * SD 1.5 的 ONNX Runtime 适配层：加载三个会话（文本编码器、UNet、VAE 解码器），
 * 按每个输入的实际数据类型（float32 / float16 / int32 / int64）创建张量，并检查输入名称。
 * ort 通过参数注入：浏览器 worker 使用 onnxruntime-web/webgpu，Node 测试使用同一个包的 wasm 后端。
 */
import { decodeFromType, encodeForType } from './tensor-types.js';

export const SD15_INPUTS = {
  textEncoder: ['input_ids'],
  unet: ['sample', 'timestep', 'encoder_hidden_states'],
  vaeDecoder: ['latent_sample'],
};
export const SD15_CONTEXT_LENGTH = 77;

/**
 * @param {object} p
 * @param {object} p.ort            onnxruntime-web 模块（需提供 InferenceSession 与 Tensor）
 * @param {object} p.sources        {textEncoder, unet, unetExternal?, vaeDecoder}：URL 字符串或 Uint8Array
 *                                  unetExternal 形如 {path: 'weights.pb', data: url|Uint8Array}
 * @param {string[]} [p.executionProviders]
 */
export async function createSd15Runtime({ ort, sources, executionProviders = ['wasm'] }) {
  const base = { executionProviders, graphOptimizationLevel: 'all' };
  const sessions = [];
  const open = async (src, extra = {}) => {
    const s = await ort.InferenceSession.create(src, { ...base, ...extra });
    sessions.push(s);
    return s;
  };
  let textEncoder;
  let unet;
  let vaeDecoder;
  try {
    textEncoder = await open(sources.textEncoder);
    unet = await open(sources.unet, sources.unetExternal ? { externalData: [sources.unetExternal] } : {});
    vaeDecoder = await open(sources.vaeDecoder);
    expectInputs(textEncoder, SD15_INPUTS.textEncoder, '文本编码器');
    expectInputs(unet, SD15_INPUTS.unet, 'UNet');
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
    /** 文本编码：ids 为 [77] 的 int32，返回 [77 × D] 的 Float32Array。 */
    async encode(ids) {
      const type = typeOf(textEncoder, 'input_ids');
      const out = await textEncoder.run({ input_ids: tensor(type, ids, [1, SD15_CONTEXT_LENGTH]) });
      return readFloat(firstOutput(textEncoder, out));
    },
    /** UNet 单步：sample 为 [batch, 4, h8, w8]，context 为 [batch, 77, D]，返回同形状的噪声预测。 */
    async unet({ sample, timestep, context, batch, h8, w8 }) {
      const hidden = context.length / (batch * SD15_CONTEXT_LENGTH);
      const feeds = {
        sample: tensor(typeOf(unet, 'sample'), sample, [batch, 4, h8, w8]),
        timestep: tensor(typeOf(unet, 'timestep'), [timestep], [1]),
        encoder_hidden_states: tensor(typeOf(unet, 'encoder_hidden_states'), context, [
          batch,
          SD15_CONTEXT_LENGTH,
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
