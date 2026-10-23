/**
 * 生图 Worker（module worker）：加载 ONNX Runtime（WebGPU，失败时退回 CPU）、分词器、调度器与模型，并执行生成。
 * 放在独立的 worker 中：界面不会卡顿；取消时主线程直接终止 worker，从而释放 GPU 显存
 * （ONNX Runtime 的 WebGPU 后端在会话运行后不会立即释放显存）。
 *
 * 消息协议：
 *   主线程 → worker：{ id, type: 'generate', urls, sizes, request }
 *   worker → 主线程：{ id, type: 'progress', stage: 'load', loaded, total, message }      模型文件读取进度（字节）
 *                    { id, type: 'progress', stage: 'session', fraction, message }        会话创建进度（0–1）
 *                    { id, type: 'progress', stage: 'denoise', step, total }
 *                    { id, type: 'progress', stage: 'decode' | 'load', message }
 *                    { id, type: 'result', width, height, rgba }   （rgba 通过 transfer 传输）
 *                    { id, type: 'error', message }
 */
import * as ort from 'onnxruntime-web/webgpu';
import { createSd15Runtime } from './sd15-runtime.js';
import { createClipTokenizer, parseMerges } from './clip-tokenizer.js';
import { createPndmScheduler } from './pndm.js';
import { chwToRgba, generateSd15 } from './sd15.js';
import { IMAGE_CACHE } from './model-files.js';
import { rewritePowToMul } from './onnx-pow.js';

// ONNX Runtime 的 wasm 运行库由构建时随网站一起输出到 dist/ort/（见 vite.config.js），
// 因此网站包离线也能运行；路径相对于本 worker 文件计算（dist/assets/ → dist/ort/）。
ort.env.wasm.wasmPaths = new URL('../ort/', self.location.href).href;

const LOAD_SHARE = 0.9; // 进度条中，模型读取占 90%，会话创建占 10%

let loadedKey = null;
let state = null;

/**
 * 读取一个模型文件（字节）。优先使用本机缓存（下载或导入的文件），没有则从 Hugging Face 读取。
 * 按字节上报进度；expected 为配置中的大小，用于预分配内存。
 */
async function loadBytes(url, expected, onBytes) {
  let res;
  if (typeof caches !== 'undefined') {
    const cache = await caches.open(IMAGE_CACHE);
    res = await cache.match(url);
  }
  if (!res) res = await fetch(url);
  if (!res.ok) throw new Error(`读取失败（${res.status}）：${url}`);
  if (!res.body) {
    const buf = new Uint8Array(await res.arrayBuffer());
    onBytes(buf.length);
    return buf;
  }
  const reader = res.body.getReader();
  let out = new Uint8Array(expected > 0 ? expected : 1 << 20);
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (n + value.length > out.length) {
      const grown = new Uint8Array(Math.max(out.length * 2, n + value.length));
      grown.set(out.subarray(0, n));
      out = grown;
    }
    out.set(value, n);
    n += value.length;
    onBytes(value.length);
  }
  return n === out.length ? out : out.slice(0, n);
}

/**
 * 读取一个计算图文件（ONNX 图，不含外部权重），把指数为 2 的 Pow 改写为 Mul(x, x)。
 * 原因与改写方法见 onnx-pow.js：分解后的 LayerNorm 中的 fp16 Pow 在 WebGPU 与 CPU 上可能没有内核。
 */
async function loadGraph(url, expected, onBytes) {
  const bytes = await loadBytes(url, expected, onBytes);
  const r = rewritePowToMul(bytes);
  if (r.rewritten) console.info(`[sd15] ${url.split('/').slice(-2).join('/')}：已把 ${r.rewritten} 个 Pow(x, 2) 改写为 Mul(x, x)`);
  if (r.kept) console.warn(`[sd15] ${url.split('/').slice(-2).join('/')}：仍有 ${r.kept} 个 Pow 的指数不是常量 2，保持不变`);
  return r.bytes;
}

async function loadText(url, expected, onBytes) {
  return new TextDecoder().decode(await loadBytes(url, expected, onBytes));
}

async function ensureLoaded(id, urls, sizes) {
  const key = urls.textEncoder;
  if (state && loadedKey === key) return state;
  release();

  const total = Object.values(sizes).reduce((a, b) => a + (Number(b) || 0), 0);
  let loaded = 0;
  let lastPost = 0;
  const onBytes = (n) => {
    loaded += n;
    const now = Date.now();
    if (now - lastPost < 120 && loaded < total) return;
    lastPost = now;
    const pct = total ? Math.floor((loaded / total) * 100) : 0;
    post({
      id,
      type: 'progress',
      stage: 'load',
      loaded,
      total,
      message: `正在读取模型文件 ${pct}%（${(loaded / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB）`,
    });
  };

  post({ id, type: 'progress', stage: 'load', loaded: 0, total, message: '正在读取分词器与调度器配置…' });
  const vocab = JSON.parse(await loadText(urls.vocab, sizes.vocab, onBytes));
  const merges = parseMerges(await loadText(urls.merges, sizes.merges, onBytes));
  const scheduler = createPndmScheduler(JSON.parse(await loadText(urls.scheduler, sizes.scheduler, onBytes)));
  const tokenizer = createClipTokenizer({ vocab, merges, maxLength: 77 });

  // 会话按顺序创建：每个会话创建后即释放其文件数据，峰值内存只取决于最大的一个文件。
  const sessionTotal = 3;
  const runtime = await createSd15Runtime({
    ort,
    executionProviders: ['webgpu'],
    sources: {
      textEncoder: () => loadGraph(urls.textEncoder, sizes.textEncoder, onBytes),
      unet: () => loadGraph(urls.unet, sizes.unet, onBytes),
      unetExternal: { path: 'weights.pb', data: () => loadBytes(urls.unetWeights, sizes.unetWeights, onBytes) },
      vaeDecoder: () => loadGraph(urls.vaeDecoder, sizes.vaeDecoder, onBytes),
    },
    onSession: (e) => {
      const done = e.phase === 'done' ? e.index + 1 : e.index;
      const fraction = LOAD_SHARE + ((1 - LOAD_SHARE) * done) / sessionTotal;
      if (e.phase === 'start') {
        post({
          id,
          type: 'progress',
          stage: 'session',
          fraction,
          message: `正在创建${e.label}会话（${e.plan === 'WebGPU' ? 'WebGPU，首次需要编译 GPU 程序，可能要几十秒' : 'CPU，速度较慢'}）…`,
        });
      } else if (e.phase === 'failed') {
        post({
          id,
          type: 'progress',
          stage: 'session',
          fraction,
          message: `${e.label}在 ${e.plan} 上无法创建，${e.plan === 'WebGPU' ? '改用 CPU 继续（较慢）' : '加载失败'}：${e.message}`,
        });
      }
    },
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
  const { id, type, urls, sizes, wasmBase, request } = event.data || {};
  if (type !== 'generate') return;
  try {
    // 运行库目录由页面给出（开发服务器与网站包的路径不同），优先于上面的默认值
    if (wasmBase) ort.env.wasm.wasmPaths = wasmBase;
    const { runtime, tokenizer, scheduler } = await ensureLoaded(id, urls, sizes || {});
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
