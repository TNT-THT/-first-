/**
 * AI 模型格式识别与运行路线判定（软件版多格式支持的核心）。
 *
 * Hugging Face 上的模型仓库格式五花八门，软件版按下面的路线对齐处理：
 * - GGUF：llama.cpp 本机直接运行（下载 → 加载 → 聊天/API）。
 * - Transformers（Safetensors / PyTorch bin）：用 llama.cpp 官方
 *   convert_hf_to_gguf.py 在本机自动转成 GGUF 后运行（绝大多数常见架构可转，
 *   不支持的架构会在转换时明确报错，不会假装能跑）。
 * - MLC（WebLLM 格式）：浏览器 WebGPU 运行，静态版模型记录路线，不在本机引擎里跑。
 * - ONNX / MLX / CoreML 等其他格式：本机暂无对应运行时，统一走 Hugging Face
 *   云端推理（Inference Providers），配好 HF Token 后点一下就能用；
 *   也可以点“找 GGUF 版”跳到同名 GGUF 仓库改走本机路线。
 *
 * 这里只做纯判定，不碰网络与文件系统，便于测试。
 */

/** 从仓库文件列表 + 元信息里识别格式。files: [{path}]，meta: {tags, library_name}。 */
export function detectFormats(files, meta = {}) {
  const paths = (Array.isArray(files) ? files : []).map((f) => String(f?.path || f || '')).filter(Boolean);
  const lower = paths.map((p) => p.toLowerCase());
  const tags = new Set((meta.tags || []).map((t) => String(t).toLowerCase()));
  const formats = new Set();

  if (lower.some((p) => p.endsWith('.gguf'))) formats.add('gguf');
  if (lower.some((p) => p.endsWith('.safetensors'))) formats.add('safetensors');
  if (lower.some((p) => /(^|\/)(pytorch_model|model)(-\d+-of-\d+)?\.bin$/.test(p))) formats.add('pytorch');
  if (lower.some((p) => p.endsWith('.onnx'))) formats.add('onnx');
  if (lower.some((p) => p.includes('mlc-chat-config.json') || p.includes('mlc_llm') || /(^|\/)params_shard_\d+\.bin$/.test(p))) formats.add('mlc');
  if (tags.has('gguf')) formats.add('gguf');
  if (tags.has('mlc-llm') || String(meta.library_name || '').toLowerCase() === 'mlc-llm') formats.add('mlc');
  // MLX 权重通常也是 safetensors，但带 mlx 标签/库名时优先按 MLX 识别（仅 Apple 芯片生态）
  if (tags.has('mlx') || String(meta.library_name || '').toLowerCase().includes('mlx')) formats.add('mlx');
  if (lower.some((p) => p.endsWith('.mlmodel') || p.endsWith('.mlpackage'))) formats.add('coreml');
  if (lower.some((p) => p.endsWith('.tflite'))) formats.add('tflite');
  if (lower.some((p) => p.endsWith('.npz') || p.endsWith('.pth') || p.endsWith('.pt'))) formats.add('pytorch');
  return [...formats];
}

/** 量化/部署变体（不是独立格式，影响能不能被 llama.cpp 转换）。 */
export function detectVariants(files, meta = {}) {
  const paths = (Array.isArray(files) ? files : []).map((f) => String(f?.path || f || '').toLowerCase());
  const tags = new Set((meta.tags || []).map((t) => String(t).toLowerCase()));
  const variants = new Set();
  const configBlob = `${paths.join(' ')} ${[...tags].join(' ')}`;
  if (/awq/.test(configBlob)) variants.add('AWQ');
  if (/gptq/.test(configBlob)) variants.add('GPTQ');
  if (/exl2/.test(configBlob)) variants.add('EXL2');
  if (/bnb|bitsandbytes|nf4|int4/.test(configBlob)) variants.add('BNB');
  return [...variants];
}

/**
 * 给一个模型列出全部可用的运行方式（云端只是其中一个可切换选项，
 * 任何格式都保证有一条“下载到本地运行”的路）：
 * - local-gguf：GGUF 直接下载，llama.cpp 本机运行；
 * - local-convert：Transformers 权重转 GGUF 后本机运行；
 * - local-python：原格式下载到本地，用 Python 运行时（Transformers/ONNX Runtime）直接运行；
 * - browser：MLC 在浏览器 WebGPU 运行（也是本地）；
 * - cloud：Hugging Face 云端推理（需 HF Token）。
 * 每项 {type, label, local: 是否本地}，第一项为默认推荐。
 */
export function optionsForFormats(formats, variants = [], platform = process.platform) {
  const has = (f) => formats.includes(f);
  const options = [];
  const quantized = variants.filter((v) => ['AWQ', 'GPTQ', 'EXL2', 'BNB'].includes(v));
  if (has('gguf')) {
    options.push({ type: 'local-gguf', label: '下载到本地运行（GGUF）', local: true });
  }
  if (has('safetensors') || has('pytorch')) {
    // 普通权重与多数 AWQ/GPTQ 权重都能被 llama.cpp 官方脚本转成 GGUF（含反量化）
    options.push({ type: 'local-convert', label: '下载并转成 GGUF 本地运行', local: true });
    options.push({ type: 'local-python', label: '下载原格式本地运行（Python）', local: true });
  }
  if (has('onnx')) {
    options.push({ type: 'local-python', label: '下载原格式本地运行（ONNX）', local: true });
  }
  if (has('mlx')) {
    if (platform === 'darwin') options.push({ type: 'local-python', label: '下载原格式本地运行（MLX）', local: true });
  }
  if (has('mlc')) {
    options.push({ type: 'browser', label: '浏览器本地运行（WebGPU）', local: true });
  }
  if (has('coreml') || has('tflite')) {
    options.push({ type: 'local-python', label: '下载原格式尝试本地运行（Python）', local: true });
  }
  if (!options.length && quantized.length) {
    options.push({ type: 'local-python', label: '下载原格式本地运行（Python）', local: true });
  }
  options.push({ type: 'cloud', label: '云端使用（不下载）', local: false });
  return options;
}

/**
 * 给格式判定默认运行路线（“一键使用”时用；用户仍可切换到其他方式）：
 * route: 'local'（GGUF 直跑）| 'convert'（转 GGUF 后跑）| 'browser'（MLC 浏览器）|
 *        'cloud'（HF 云端推理）| 'unsupported'（明确不支持）
 */
export function routeForFormats(formats, variants = []) {
  const has = (f) => formats.includes(f);
  if (has('gguf')) {
    return {
      route: 'local',
      engine: 'llama.cpp（本机）',
      label: '可直接在本机运行',
      action: 'download',
      detail: '仓库含 GGUF 权重，下载后用 llama.cpp 在本机运行。',
    };
  }
  if (has('safetensors') || has('pytorch')) {
    const quantized = variants.filter((v) => ['AWQ', 'GPTQ', 'EXL2', 'BNB'].includes(v));
    if (quantized.length) {
      return {
        route: 'convert',
        engine: '自动转换 → llama.cpp（本机）',
        label: '可转换后在本机运行',
        action: 'convert',
        detail: `${quantized.join('/')} 量化权重可由 llama.cpp 官方脚本反量化转成 GGUF 本地运行（少数架构转不了时可改用 Python 本地运行或云端）。`,
      };
    }
    return {
      route: 'convert',
      engine: '自动转换 → llama.cpp（本机）',
      label: '可转换后在本机运行',
      action: 'convert',
      detail: 'Transformers 权重可自动转成 GGUF 后在本机运行（首次需准备转换工具）。',
    };
  }
  if (has('mlc')) {
    return {
      route: 'browser',
      engine: '浏览器 WebGPU（WebLLM）',
      label: '浏览器格式',
      action: 'browser',
      detail: 'MLC 格式在浏览器里运行；软件版聊天同样可用浏览器模型路线。',
    };
  }
  if (has('onnx') || has('mlx') || has('coreml') || has('tflite')) {
    return {
      route: 'python',
      engine: 'Python 运行时（本机）',
      label: '可下载到本地运行',
      action: 'python',
      detail: '原格式下载到本地用 Python 运行时（ONNX Runtime / Transformers）运行；也可以切换为云端使用。',
    };
  }
  return {
    route: 'unsupported',
    engine: '—',
    label: '暂未识别格式',
    action: 'none',
    detail: '没有找到可识别的权重文件；可能是数据集/代码仓库，或格式较新。',
  };
}

/** 把各种输入（关键词、owner/repo、完整 HF 链接）解析成仓库 id；不是仓库时返回 null。 */
export function parseHfRepoInput(text) {
  const raw = String(text || '').trim();
  if (!raw) return null;
  const urlMatch = raw.match(/huggingface\.co\/([A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*)/);
  if (urlMatch) return urlMatch[1];
  if (/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(raw)) return raw;
  return null;
}

/** 模型 id（聊天/API 里用）：hf/<owner>/<repo>，与内置目录 id 区分。 */
export function hfModelId(repo) {
  return `hf/${repo}`;
}

/** 模型在本地磁盘上的安全目录名（id 里含 / 与 :，不能直接当目录）。 */
export function safeDirName(id) {
  return String(id).replace(/[^A-Za-z0-9._-]+/g, '_');
}
