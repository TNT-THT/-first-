/**
 * 生图 Worker（module worker）：加载 ONNX Runtime（WebGPU，失败时退回 CPU）、分词器、调度器与模型，并执行生成。
 * 放在独立的 worker 中：界面不会卡顿；取消时主线程直接终止 worker，从而释放 GPU 显存
 * （ONNX Runtime 的 WebGPU 后端在会话运行后不会立即释放显存）。
 *
 * 消息协议：
 *   主线程 → worker：{ id, type: 'generate', urls, sizes, request }
 *   worker → 主线程：{ id, type: 'progress', stage: 'load', loaded, total, fraction, message }  模型文件读取进度（字节）
 *                    { id, type: 'progress', stage: 'session', fraction, message }             会话创建进度
 *                    （两者的 fraction 均为整体进度 0–1，见 load-progress.js）
 *                    { id, type: 'progress', stage: 'denoise', step, total }
 *                    { id, type: 'progress', stage: 'decode' | 'load', message }
 *                    { id, type: 'result', width, height, rgba, truncatedTokens, promptChunks, chunkFallback, chunkError }
 *                    （rgba 通过 transfer 传输；promptChunks 为实际分段数；chunkFallback 表示分段失败后改用单段）
 *   request.input：用户在文本框中的描述，超过 request.maxInputTokens（默认 4096）时在下载大文件之前报错。
 *   request.lead：管理员前置内容（更正与知识库）。超出单段 CLIP 预算时在下载大文件之前报错，不截断。
 *   描述最多分 MAX_PROMPT_CHUNKS 段编码（约 300 个 token），超出部分截断并在 truncatedTokens 中报告。
 *                    { id, type: 'error', message }
 */
import * as ort from 'onnxruntime-web/webgpu';
import { createSd15Runtime } from './sd15-runtime.js';
import { createClipTokenizer, parseMerges } from './clip-tokenizer.js';
import { createScheduler, parseSchedulerConfig } from './schedulers.js';
import { chwToRgba, generateSd15WithFallback } from './sd15.js';
import { oversizedFiles, createLimiter, PART_CONCURRENCY } from './model-files.js';
import { openModelFile } from './model-source.js';
import { rewritePowToMul } from './onnx-pow.js';
import { loadFraction } from './load-progress.js';
import { checkPromptBudget, MAX_PROMPT_CHUNKS } from './prompt-budget.js';

// ONNX Runtime 的 wasm 运行库由构建时随网站一起输出到 dist/ort/（见 vite.config.js），
// 因此网站包离线也能运行；路径相对于本 worker 文件计算（dist/assets/ → dist/ort/）。
ort.env.wasm.wasmPaths = new URL('../ort/', self.location.href).href;

let loadedKey = null;
let state = null;

// 所有文件的读取（包括生成时补下缺少的部分）共用同一并发上限
const workerLimit = createLimiter(PART_CONCURRENCY);

/**
 * 读取一个模型文件（字节）。优先使用本机完整缓存；下载到一半的文件只补下缺少的字节（不重下已保存的部分）；
 * 什么都没有则从 Hugging Face 读取。按字节上报进度；expected 为配置中的大小，用于预分配内存。
 */
async function loadBytes(url, expected, onBytes) {
  const res = await openModelFile(url, expected, {
    storage: typeof caches !== 'undefined' ? caches : null,
    limit: workerLimit,
  });
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

/**
 * 读取文本编码器的备选文件（如 fp32）。仅在主文件的会话创建全部失败时调用。
 * 大小必须与配置一致，否则报错（避免用错误文件创建会话）。
 */
async function loadFallbackGraph(url, expected, onBytes, onStart) {
  onStart?.(expected);
  const bytes = await loadBytes(url, expected, onBytes);
  if (expected > 0 && bytes.length !== expected) {
    throw new Error(`备选文本编码器大小与配置不一致（实际 ${bytes.length}，应为 ${expected}）：${url}`);
  }
  const r = rewritePowToMul(bytes);
  if (r.rewritten) console.info(`[sd15] 备选文本编码器：已把 ${r.rewritten} 个 Pow(x, 2) 改写为 Mul(x, x)`);
  if (r.kept) console.warn(`[sd15] 备选文本编码器：仍有 ${r.kept} 个 Pow 的指数不是常量 2，保持不变`);
  return r.bytes;
}

async function loadText(url, expected, onBytes) {
  return new TextDecoder().decode(await loadBytes(url, expected, onBytes));
}

// 分词器（词表与合并规则，约 1.6 MB）单独缓存：生成前先检查提示词预算，超出时不去读取大模型文件。
let tokenizerCache = null;

async function ensureTokenizer(urls, sizes) {
  const key = `${urls.vocab}|${urls.merges}`;
  if (tokenizerCache && tokenizerCache.key === key) return tokenizerCache.tokenizer;
  const vocab = JSON.parse(await loadText(urls.vocab, sizes.vocab, () => {}));
  const merges = parseMerges(await loadText(urls.merges, sizes.merges, () => {}));
  const tokenizer = createClipTokenizer({ vocab, merges, maxLength: 77 });
  tokenizerCache = { key, tokenizer };
  return tokenizer;
}

let loadingKey = null;
let loadingPromise = null;

/**
 * 创建（或复用）当前模型的运行时。同一模型的加载只进行一次（预加载与生成可以同时请求）；
 * 另一个模型的加载会先等待正在进行的加载结束，避免两个模型同时占用显存。
 */
async function ensureLoaded(id, urls, sizes, tokenizer) {
  const key = urls.textEncoder;
  for (;;) {
    if (state && loadedKey === key) return state;
    if (loadingPromise && loadingKey === key) return loadingPromise;
    if (!loadingPromise) break;
    await loadingPromise.catch(() => {});
  }
  release();
  loadingKey = key;
  const p = loadRuntime(id, urls, sizes, tokenizer, key);
  loadingPromise = p;
  try {
    return await p;
  } finally {
    if (loadingPromise === p) {
      loadingPromise = null;
      loadingKey = null;
    }
  }
}

async function loadRuntime(id, urls, sizes, tokenizer, key) {

  // 超过浏览器单文件上限的模型（如 fp32 的 3.4 GB UNet 权重）无法在网页中加载：尽早报错，不去读取
  const big = oversizedFiles(sizes);
  if (big.length) {
    const names = big.map((b) => `${b.key}（${(b.bytes / 1e9).toFixed(2)} GB）`).join('、');
    throw new Error(
      `该模型的文件超过浏览器单文件上限（约 2 GB）：${names}。网页版无法加载，请改用 fp16 版，或使用本机 Python 脚本（scripts/image/sd15_local.py）。`,
    );
  }

  // 进度总量只计入一定会读取的文件；备选文本编码器仅在主文件失败时读取，开始读取时再计入（见 onStart 回调）。
  // 词表与合并规则由 ensureTokenizer 读取，这里不再计入。
  const sessionTotal = 3;
  let sessionsDone = 0;
  let total = Object.entries(sizes).reduce(
    (a, [k, b]) => (k === 'textEncoderFallback' || k === 'vocab' || k === 'merges' ? a : a + (Number(b) || 0)),
    0,
  );
  let loaded = 0;
  let lastPost = 0;
  const overall = () => loadFraction({ loaded, total, sessionsDone, sessionTotal });
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
      fraction: overall(),
      message: `正在读取模型文件 ${pct}%（${(loaded / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB）`,
    });
  };

  post({ id, type: 'progress', stage: 'load', loaded: 0, total, fraction: 0, message: '正在读取调度器配置…' });
  // 调度器由配置的 _class_name 决定（PNDM、DPM-Solver++、DEIS、Euler）；不支持的配置直接报错
  const scheduler = createScheduler(parseSchedulerConfig(await loadText(urls.scheduler, sizes.scheduler, onBytes)));

  // 会话按顺序创建：每个会话创建后即释放其文件数据，峰值内存只取决于最大的一个文件。
  const runtime = await createSd15Runtime({
    ort,
    executionProviders: ['webgpu'],
    sources: {
      textEncoder: () => loadGraph(urls.textEncoder, sizes.textEncoder, onBytes),
      // 备选文本编码器（仅在主文件失败时读取，见 sd15-runtime.js）
      textEncoderFallback: urls.textEncoderFallback
        ? () =>
            loadFallbackGraph(urls.textEncoderFallback, sizes.textEncoderFallback, onBytes, (n) => {
              total += Number(n) || 0; // 备选文件开始读取时才计入进度总量
            })
        : undefined,
      unet: () => loadGraph(urls.unet, sizes.unet, onBytes),
      // 外部权重可选：单文件 UNet（如 bk-sdm、tiny-sd）没有 weights.pb
      unetExternal: urls.unetWeights
        ? { path: 'weights.pb', data: () => loadBytes(urls.unetWeights, sizes.unetWeights, onBytes) }
        : undefined,
      vaeDecoder: () => loadGraph(urls.vaeDecoder, sizes.vaeDecoder, onBytes),
    },
    onFallback: ({ message }) => {
      console.warn(`[sd15] 主文本编码器无法创建会话，改用备选文件（fp32）：${message}`);
      post({ id, type: 'progress', stage: 'session', fraction: overall(), message: '主文本编码器无法创建会话，正在改用备选文件（约 0.5 GB）…' });
    },
    onSession: (e) => {
      if (e.phase === 'done') sessionsDone += 1;
      if (e.phase === 'start' || e.phase === 'failed') {
        const fraction = overall();
        if (e.phase === 'start') {
          post({
            id,
            type: 'progress',
            stage: 'session',
            fraction,
            message: `正在创建${e.label}会话（${e.plan === 'WebGPU' ? 'WebGPU，首次需要编译 GPU 程序，可能要几十秒' : 'CPU，速度较慢'}）…`,
          });
        } else {
          post({
            id,
            type: 'progress',
            stage: 'session',
            fraction,
            message: `${e.label}在 ${e.plan} 上无法创建，${e.plan === 'WebGPU' ? '改用 CPU 继续（较慢）' : '加载失败'}：${e.message}`,
          });
        }
      }
    },
  });
  if (runtime.textEncoderSource === 'fallback') console.info('[sd15] 文本编码器使用备选文件（fp32）');
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
  const { id, type, urls, sizes, wasmBase, request, latentScale } = event.data || {};
  if (type !== 'generate' && type !== 'load') return;
  try {
    // 运行库目录由页面给出（开发服务器与网站包的路径不同），优先于上面的默认值
    if (wasmBase) ort.env.wasm.wasmPaths = wasmBase;
    // 先读取分词器并检查预算（管理员内容超出时直接报错，不下载大文件）
    const tokenizer = await ensureTokenizer(urls, sizes || {});
    if (type === 'load') {
      // 只加载（预加载）：不检查提示词、不生成
      await ensureLoaded(id, urls, sizes || {}, tokenizer);
      post({ id, type: 'loaded' });
      return;
    }
    const budgetInput = {
      input: typeof request.input === 'string' ? request.input : null,
      maxInputTokens: request.maxInputTokens,
      lead: request.lead || '',
      prompt: request.prompt,
    };
    checkPromptBudget(tokenizer, { ...budgetInput, maxChunks: MAX_PROMPT_CHUNKS });
    const { runtime, scheduler } = await ensureLoaded(id, urls, sizes || {}, tokenizer);
    post({ id, type: 'progress', stage: 'denoise', step: 0, total: request.steps, message: '开始生成' });
    const { width, height, chw, promptChunks, chunkFallback, chunkError } = await generateSd15WithFallback({
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
      latentScale,
      maxChunks: MAX_PROMPT_CHUNKS,
      onStep: ({ step, total }) => post({ id, type: 'progress', stage: 'denoise', step, total }),
    });
    // 单段回退时，实际参与生图的只有第一段，截断数需按单段重新计算
    const truncatedTokens = checkPromptBudget(tokenizer, {
      ...budgetInput,
      maxChunks: chunkFallback ? 1 : MAX_PROMPT_CHUNKS,
    }).truncatedTokens;
    post({ id, type: 'progress', stage: 'decode', message: '正在解码图像…' });
    const rgba = chwToRgba(chw, width, height);
    post(
      { id, type: 'result', width, height, rgba, truncatedTokens, promptChunks, chunkFallback, chunkError },
      [rgba.buffer],
    );
  } catch (err) {
    post({ id, type: 'error', message: err?.message || String(err) });
  }
};
