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
import { IMAGE_CACHE } from './model-files.js';

// 运行库的 wasm 文件从 CDN 加载（版本与 package.json 中的 onnxruntime-web 一致）
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

let loadedKey = null;
let state = null;

// 本机已下载（或已导入）的模型文件优先从缓存读取，离线也可以生成；缓存中没有的才从 Hugging Face 读取。
// 缓存中的文件以 blob URL 交给 ONNX Runtime，会话创建后立即释放。
let objectUrls = [];

async function sourceFor(url) {
  if (typeof caches === 'undefined') return url;
  const cache = await caches.open(IMAGE_CACHE);
  const hit = await cache.match(url);
  if (!hit) return url;
  const objectUrl = URL.createObjectURL(await hit.blob());
  objectUrls.push(objectUrl);
  return objectUrl;
}

function revokeSources() {
  for (const u of objectUrls) URL.revokeObjectURL(u);
  objectUrls = [];
}

async function fetchText(url) {
  const res = await fetch(await sourceFor(url));
  if (!res.ok) throw new Error(`读取失败（${res.status}）：${url}`);
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
  let runtime;
  try {
    runtime = await createSd15Runtime({
      ort,
      sources: {
        textEncoder: await sourceFor(urls.textEncoder),
        unet: await sourceFor(urls.unet),
        unetExternal: { path: 'weights.pb', data: await sourceFor(urls.unetWeights) },
        vaeDecoder: await sourceFor(urls.vaeDecoder),
      },
      executionProviders: ['webgpu', 'wasm'],
    });
  } finally {
    // 会话创建时已读取全部数据，之后可以释放 blob URL
    revokeSources();
  }
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
