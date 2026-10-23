/**
 * 生图 Worker（module worker）：加载 ONNX Runtime（WebGPU）、分词器、调度器与模型，并执行生成。
 * 放在独立的 worker 中：界面不会卡顿；取消时主线程直接终止 worker，从而释放 GPU 显存
 * （ONNX Runtime 的 WebGPU 后端在会话运行后不会立即释放显存）。
 *
 * 消息协议：
 *   主线程 → worker：{ id, type: 'generate', urls, request }
 *   worker → 主线程：{ id, type: 'progress', stage, step?, total?, message? }
 *                    { id, type: 'result', width, height, rgba }   （rgba 通过 transfer 传输）
 *                    { id, type: 'error', message }
 */
import * as ort from 'onnxruntime-web/webgpu';
import { createSd15Runtime } from './sd15-runtime.js';
import { createClipTokenizer, parseMerges } from './clip-tokenizer.js';
import { createPndmScheduler } from './pndm.js';
import { chwToRgba, generateSd15 } from './sd15.js';

// 运行库的 wasm 文件从 CDN 加载（版本与 package.json 中的 onnxruntime-web 一致）
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

let loadedKey = null;
let state = null;

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`下载失败（${res.status}）：${url}`);
  return res.text();
}

async function ensureLoaded(id, urls) {
  const key = urls.textEncoder;
  if (state && loadedKey === key) return state;
  release();
  post({ id, type: 'progress', stage: 'load', message: '正在读取分词器与调度器配置…' });
  const [vocabText, mergesText, schedulerText] = await Promise.all([
    fetchText(urls.vocab),
    fetchText(urls.merges),
    fetchText(urls.scheduler),
  ]);
  const vocab = JSON.parse(vocabText);
  const tokenizer = createClipTokenizer({ vocab, merges: parseMerges(mergesText), maxLength: 77 });
  const scheduler = createPndmScheduler(JSON.parse(schedulerText));
  post({ id, type: 'progress', stage: 'load', message: '正在加载文本编码器与 UNet（首次需要下载约 2 GB）…' });
  const runtime = await createSd15Runtime({
    ort,
    sources: {
      textEncoder: urls.textEncoder,
      unet: urls.unet,
      unetExternal: { path: 'weights.pb', data: urls.unetWeights },
      vaeDecoder: urls.vaeDecoder,
    },
    executionProviders: ['webgpu', 'wasm'],
  });
  state = { runtime, tokenizer, scheduler };
  loadedKey = key;
  return state;
}

function release() {
  state?.runtime?.release();
  state = null;
  loadedKey = null;
}

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

self.onmessage = async (event) => {
  const { id, type, urls, request } = event.data || {};
  if (type !== 'generate') return;
  try {
    const { runtime, tokenizer, scheduler } = await ensureLoaded(id, urls);
    post({ id, type: 'progress', stage: 'denoise', step: 0, total: request.steps, message: '开始生成' });
    const { width, height, chw } = await generateSd15({
      runtime,
      tokenizer,
      scheduler,
      prompt: request.prompt,
      negative: request.negative || '',
      width: request.width,
      height: request.height,
      steps: request.steps,
      guidance: request.guidance,
      seed: request.seed,
      onStep: ({ step, total }) => post({ id, type: 'progress', stage: 'denoise', step, total }),
    });
    post({ id, type: 'progress', stage: 'decode', message: '正在解码图像…' });
    const rgba = chwToRgba(chw, width, height);
    post({ id, type: 'result', width, height, rgba }, [rgba.buffer]);
  } catch (err) {
    post({ id, type: 'error', message: err?.message || String(err) });
  }
};
