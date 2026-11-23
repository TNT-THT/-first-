/**
 * 本机大模型目录（GGUF / llama.cpp 原生引擎）。
 *
 * 这些模型太大，浏览器 WebGPU（WebLLM）跑不了：本站在有服务端程序的环境
 * （网站包、独立软件）中通过 llama.cpp 的 llama-server 在本机原生运行它们，
 * 权重按需从 Hugging Face 下载到本机，不包含在任何安装包里。
 *
 * - 每个条目只登记已被核实存在的 GGUF 仓库；具体量化文件在运行时从仓库
 *   文件树解析（不同仓库布局不同：根目录文件、按量化分子目录、分片文件），
 *   因此这里不写死文件名，避免仓库改名/增删量化后失效。
 * - paramsB 为总参数量（单位：十亿）；MoE 模型另注 activeB（每 token 激活量）。
 * - 显存/内存估算以实际下载文件大小为准（见 estimateRamGB），目录里不写死。
 */

export const NATIVE_MODELS = [
  {
    id: 'Qwen3.8-27B',
    name: 'Qwen3.8 27B',
    vendor: 'Qwen（阿里）',
    paramsB: 27,
    activeB: null,
    reasoning: true,
    repo: 'bartowski/Qwen3.8-27B-GGUF',
    license: 'Apache-2.0',
    context: 262144,
    note: '27B 稠密视觉语言模型；本站只接纯文本对话（不加载视觉编码器）。原为网页版“待转换（暂不可用）”登记模型，在本机引擎中可直接运行。',
  },
  {
    id: 'Qwen3-32B',
    name: 'Qwen3 32B',
    vendor: 'Qwen（阿里）',
    paramsB: 32,
    activeB: null,
    reasoning: true,
    repo: 'bartowski/Qwen3-32B-GGUF',
    license: 'Apache-2.0',
    context: 131072,
    note: 'Qwen3 稠密 32B，中文与代码能力强。',
  },
  {
    id: 'Qwen3-30B-A3B',
    name: 'Qwen3 30B-A3B',
    vendor: 'Qwen（阿里）',
    paramsB: 30,
    activeB: 3,
    reasoning: true,
    repo: 'bartowski/Qwen3-30B-A3B-GGUF',
    license: 'Apache-2.0',
    context: 262144,
    note: 'MoE：总参 30B、每 token 只激活约 3B，内存占用按 30B 算、速度接近小模型。',
  },
  {
    id: 'DeepSeek-R1-Distill-Qwen-32B',
    name: 'DeepSeek-R1 蒸馏 Qwen 32B',
    vendor: 'DeepSeek',
    paramsB: 32,
    activeB: null,
    reasoning: true,
    repo: 'bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF',
    license: 'MIT（基座 Qwen 部分另行遵循其许可）',
    context: 131072,
    note: 'R1 蒸馏推理模型，数学/代码推理强，输出含思考过程。',
  },
  {
    id: 'Llama-3.3-70B',
    name: 'Llama 3.3 70B',
    vendor: 'Meta',
    paramsB: 70,
    activeB: null,
    reasoning: false,
    repo: 'bartowski/Llama-3.3-70B-Instruct-GGUF',
    license: 'Llama 3.3 Community License',
    context: 131072,
    note: 'Meta 旗舰稠密 70B（仅次于 405B 的常用尺寸）。',
  },
  {
    id: 'Qwen2.5-72B',
    name: 'Qwen2.5 72B',
    vendor: 'Qwen（阿里）',
    paramsB: 72,
    activeB: null,
    reasoning: false,
    repo: 'bartowski/Qwen2.5-72B-Instruct-GGUF',
    license: 'Qwen Research License（商用需另行授权）',
    context: 131072,
    note: 'Qwen2.5 旗舰 72B，中文能力强。',
  },
  {
    id: 'Llama-4-Scout-109B',
    name: 'Llama 4 Scout 109B',
    vendor: 'Meta',
    paramsB: 109,
    activeB: 17,
    reasoning: false,
    repo: 'unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF',
    license: 'Llama 4 Community License',
    context: 1048576,
    note: 'MoE：总参 109B、激活约 17B。本站只加载语言部分（不下载视觉 mmproj），按纯文本使用。',
  },
  {
    id: 'Mixtral-8x22B-141B',
    name: 'Mixtral 8x22B 141B',
    vendor: 'Mistral AI',
    paramsB: 141,
    activeB: 39,
    reasoning: false,
    repo: 'MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-GGUF',
    license: 'Apache-2.0',
    context: 65536,
    note: '8 专家 MoE，总参 141B。',
  },
  {
    id: 'Qwen3-235B-A22B',
    name: 'Qwen3 235B-A22B',
    vendor: 'Qwen（阿里）',
    paramsB: 235,
    activeB: 22,
    reasoning: true,
    repo: 'unsloth/Qwen3-235B-A22B-GGUF',
    license: 'Apache-2.0',
    context: 262144,
    note: 'MoE 旗舰：总参 235B、激活约 22B。',
  },
  {
    id: 'Llama-3.1-405B',
    name: 'Llama 3.1 405B',
    vendor: 'Meta',
    paramsB: 405,
    activeB: null,
    reasoning: false,
    repo: 'ThomasBaruzier/Meta-Llama-3.1-405B-Instruct-GGUF',
    license: 'Llama 3.1 Community License',
    context: 131072,
    note: 'Meta 最大开源权重 405B；低比特量化也需要约百 GB 级内存/显存。',
  },
  {
    id: 'DeepSeek-R1-671B',
    name: 'DeepSeek-R1 671B',
    vendor: 'DeepSeek',
    paramsB: 671,
    activeB: 37,
    reasoning: true,
    repo: 'unsloth/DeepSeek-R1-GGUF',
    license: 'MIT',
    context: 163840,
    note: 'MoE 旗舰推理模型：总参 671B、激活约 37B。UD 动态量化为 unsloth 优化版本。',
  },
];

/** 量化档位偏好（自动推荐时从前到后取第一个可用的）。 */
export const QUANT_PREFERENCE = [
  'Q4_K_M',
  'Q5_K_M',
  'Q4_K_S',
  'UD-Q4_K_XL',
  'Q3_K_M',
  'IQ4_XS',
  'Q2_K',
  'UD-IQ2_XXS',
  'IQ2_M',
  'IQ1_M',
];

const BY_ID = new Map(NATIVE_MODELS.map((m) => [m.id, m]));

export function nativeModelById(id) {
  return BY_ID.get(String(id || '')) || null;
}

export function isNativeModelId(id) {
  return BY_ID.has(String(id || ''));
}

/**
 * 从 GGUF 文件路径中提取量化标记。
 * 兼容三种真实布局：根目录文件（`Model-Q4_K_M.gguf`、点分隔 `Model.Q4_K_M-00001-of-00002.gguf`）、
 * 按量化分子目录（`Q4_K_M/Model-Q4_K_M-00001-of-00003.gguf`、`DeepSeek-R1-Q2_K/...`）。
 * 找不到返回 null（如 mmproj、imatrix 等非权重文件由调用方先过滤）。
 */
export function extractQuant(filePath) {
  const p = String(filePath || '');
  const m = p.match(/(?:^|[-./])(UD-[A-Z0-9]+(?:_[A-Z0-9]+)*|IQ\d(?:_[A-Z0-9]+)+|Q\d(?:_[A-Z0-9]+)+|MXFP4|BF16|F16|F32)(?=[-./]|$)/i);
  return m ? m[1].toUpperCase() : null;
}

/** 分片序号（`...-00002-of-00003.gguf`），无分片时为 1。 */
function splitIndex(filePath) {
  const m = String(filePath).match(/-(\d{1,5})-of-\d{1,5}\.gguf$/i);
  return m ? Number(m[1]) : 1;
}

/**
 * 把仓库文件列表按量化分组（只收 .gguf 权重，排除 mmproj 视觉投影文件）。
 * @param {Array<{path: string, size?: number}>} files
 * @returns {Array<{quant: string, files: Array<{path,size}>, totalBytes: number}>} 按总大小升序
 */
export function groupFilesByQuant(files) {
  const groups = new Map();
  for (const f of Array.isArray(files) ? files : []) {
    const p = String(f?.path || '');
    if (!/\.gguf$/i.test(p)) continue;
    if (/mmproj/i.test(p)) continue;
    const quant = extractQuant(p);
    if (!quant) continue;
    if (!groups.has(quant)) groups.set(quant, []);
    groups.get(quant).push({ path: p, size: Math.max(0, Number(f.size) || 0) });
  }
  const out = [];
  for (const [quant, list] of groups) {
    list.sort((a, b) => splitIndex(a.path) - splitIndex(b.path) || a.path.localeCompare(b.path));
    out.push({ quant, files: list, totalBytes: list.reduce((n, f) => n + f.size, 0) });
  }
  out.sort((a, b) => a.totalBytes - b.totalBytes);
  return out;
}

/** 从分组结果里挑推荐量化：偏好表命中优先，否则取体积中位的一档。 */
export function pickDefaultQuant(grouped) {
  if (!grouped?.length) return null;
  for (const want of QUANT_PREFERENCE) {
    const hit = grouped.find((g) => g.quant === want);
    if (hit) return hit.quant;
  }
  return grouped[Math.floor(grouped.length / 2)].quant;
}

/**
 * 估算运行所需内存（内存+显存合计，GB）：权重 + KV 缓存与运行时开销。
 * llama.cpp 可把权重放在系统内存、把一部分层卸载到显存，两者合计不低于此数。
 */
export function estimateRamGB(totalBytes, contextTokens = 8192) {
  const weightsGB = (Number(totalBytes) || 0) / 1e9;
  const kvGB = Math.max(0.5, (Number(contextTokens) || 8192) / 16384);
  return Math.ceil((weightsGB * 1.12 + kvGB + 1.5) * 10) / 10;
}

/** 给网页/API 展示用的简短状态名。 */
export function modelStateLabel(model) {
  if (!model) return '';
  const parts = [`${model.paramsB}B`];
  if (model.activeB) parts.push(`激活约 ${model.activeB}B`);
  parts.push(model.reasoning ? '推理' : '非推理');
  return parts.join('·');
}
