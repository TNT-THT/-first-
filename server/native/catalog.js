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

/** 是否为 GGUF 分片文件（`...-00002-of-00005.gguf`）。 */
export function isShardFile(filePath) {
  return /-\d{1,5}-of-\d{1,5}\.gguf$/i.test(String(filePath || ''));
}

/**
 * 在一档量化（或清单）里挑出“加载入口”文件——llama.cpp 的 `-m` 要指向第 1 个分片，
 * 它自己会顺着 `#00002` 之类的顺序把其余分片拉起来。
 *
 * 以前调用方直接用 `files.sort()[0]`：字典序在两位以上分片号（`-00002-` 排在
 * `-00001-` 后面没问题，但 `-10-of-` 会排到 `-2-of-` 前面）或按量化分子目录布局时
 * 会挑到中段分片，超大模型于是“下载成功、加载即崩”。这里按分片号排序取值。
 * mmproj（视觉投影）永远不能当入口文件。
 */
export function pickEntryFile(files) {
  const list = (Array.isArray(files) ? files : [])
    .filter((f) => f && typeof f.path === 'string' && /\.gguf$/i.test(f.path) && !/mmproj/i.test(f.path));
  if (!list.length) return null;
  const shards = list.filter((f) => isShardFile(f.path));
  if (shards.length) {
    // 只要编号最小的那一个；同号再按路径稳定排序
    return [...shards].sort((a, b) => splitIndex(a.path) - splitIndex(b.path) || a.path.localeCompare(b.path))[0];
  }
  return [...list].sort((a, b) => a.path.localeCompare(b.path))[0];
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

/**
 * 给量化档位做小白可读的标注：类型、每权重位宽、是否可直接使用、质量一句话。
 *
 * 本软件下载的量化全部是 GGUF：llama.cpp 原生格式，下载完点“加载”即可运行，
 * 不需要任何转换（与 Safetensors 需要转 GGUF 的路线不同，这里必须说清楚）。
 * 位宽按档位名的名义位数估算（K 量化/i-quant 的实际平均位宽会略有出入，
 * 只用于给用户一个量级概念，不参与任何下载/加载逻辑）。
 */
export function describeQuant(quant) {
  const q = String(quant || '').toUpperCase();
  // 两个特殊“量化名”是登记口径，不是真量化档位，单独给说法别误导
  if (q === 'SNAPSHOT') {
    return {
      typeLabel: '原格式快照（还没转成 GGUF）',
      format: 'Snapshot',
      bitsPerWeight: null,
      dynamic: false,
      directUse: false,
      useNote: '需要“转换并使用”转成 GGUF，或用“本地运行（Python）”原格式运行',
      qualityNote: '仓库原始权重整包，体积通常最大。',
    };
  }
  if (q === 'CONVERTED') {
    return {
      typeLabel: '已转换的 GGUF',
      format: 'GGUF',
      bitsPerWeight: null,
      dynamic: false,
      directUse: true,
      useNote: '已在本地转成 GGUF，可直接加载使用',
      qualityNote: '由原始权重本地转换而来。',
    };
  }
  const dynamic = q.startsWith('UD-');
  const base = dynamic ? q.slice(3) : q;
  let bits = null;
  if (base === 'F32') bits = 32;
  else if (base === 'F16' || base === 'BF16') bits = 16;
  else if (base === 'MXFP4') bits = 4;
  else {
    const m = base.match(/^(?:I?Q)(\d)/);
    if (m) bits = Number(m[1]);
  }
  const family = dynamic
    ? 'GGUF 动态量化（按层重要性分配位数）'
    : base.startsWith('IQ')
      ? 'GGUF i-quant 量化（重要性矩阵）'
      : bits !== null && bits >= 16
        ? 'GGUF 未压缩权重'
        : 'GGUF K 量化';
  let qualityNote;
  if (bits === null) qualityNote = '特殊档位，质量以仓库说明为准。';
  else if (bits >= 16) qualityNote = '未压缩，质量最好但体积最大、速度最慢。';
  else if (bits >= 8) qualityNote = '质量几乎无损，体积仍较大。';
  else if (bits >= 5) qualityNote = '质量很好，体积适中。';
  else if (bits >= 4) qualityNote = '质量与体积最均衡，最推荐。';
  else if (bits >= 3) qualityNote = '体积更小，质量略有损失。';
  else qualityNote = '体积最小，质量损失明显，只适合内存非常紧张时。';
  return {
    // 类型：明确是 GGUF（本机引擎原生格式），不是需要再转换的原始权重
    typeLabel: family,
    format: 'GGUF',
    bitsPerWeight: bits,
    dynamic,
    // GGUF 是 llama.cpp 原生格式：下载后直接加载运行，不需要转换
    directUse: true,
    useNote: '下载后可直接加载使用（GGUF 是本机引擎的原生格式，不用转换）',
    qualityNote,
  };
}

/**
 * 在“这台机器装得下什么”的前提下挑量化。**不靠型号表，靠本机实测内存。**
 *
 * 背景（用户真实机器的教训：16.6 GB 总内存 / 7.4 GB 可用，HF 一键使用
 * 自动挑了 Q4_K_M，15 GB 下完了才发现根本跑不动）：
 *
 * 原来的 pickDefaultQuant 只看偏好表，完全不管本机装不装得下——
 * 省了用户 10 秒，可能浪费 10 小时下载和几十 GB 磁盘。
 *
 * 这里按“装得下 → 装得下但不宽裕 → 只能靠页面文件硬顶 → 都装不下”分档，
 * 在每一档里再尽量选质量高的那一个。判断依据是 estimateRamGB（按真实文件
 * 大小算），所以任何机器、任何仓库都适用，不需要任何 CPU/机型假设。
 *
 * @param {Array<{quant:string,totalBytes:number}>} grouped 仓库里的量化分组
 * @param {object} opts
 * @param {number} [opts.totalGB]          物理内存（GB），读不到传 0
 * @param {number} [opts.freeGB]           当前可用内存（GB），读不到传 0
 * @param {number} [opts.commitLimitGB]    可提交上限（物理内存+页面文件，GB）
 * @param {number} [opts.pagefileGB]       页面文件/swap 大小（GB），0 表示没配
 * @param {number} [opts.contextTokens]    KV 缓存按多少上下文估算
 * @returns {{quant:string|null, fits:'free'|'total'|'spill'|'never'|'unknown',
 *            neededGB:number, candidates:Array, reason:string}}
 */
export function pickQuantForBudget(grouped, {
  totalGB = 0, freeGB = 0, commitLimitGB = 0, pagefileGB = 0, contextTokens = 8192,
} = {}) {
  const list = (Array.isArray(grouped) ? grouped : [])
    .filter((g) => g && g.quant && Number(g.totalBytes) > 0);
  if (!list.length) return { quant: null, fits: 'unknown', neededGB: 0, candidates: [], reason: '没有可下载的量化文件。' };

  const round = (n) => Math.round(n * 10) / 10;
  const ctx = Math.max(1024, Number(contextTokens) || 8192);
  // 与 engine._memoryReport 同一套估算（权重 + KV + 运行时），保证界面上
  // 量化列表里显示的内存需求和这里的判定永远一致，不会出现两套数字打架。
  const need = (bytes) => round((Number(bytes) / 1e9) * 1.12 + Math.max(0.5, ctx / 16384) + 1.5);

  // 系统预留：总内存不能全给模型。Windows 自身 + 驱动 + 其它进程在 16 GB 机器上
  // 通常就要吃掉 4~5 GB，所以“刚好等于总内存”其实装不下——以前那种判法会给出
  // “装得下”的假结论（15.4 GB 塞进 16.6 GB），用户照做才发现还是崩。
  const osReserveGB = Math.max(2, totalGB * 0.15);
  const ceilingTotal = totalGB > 0 ? totalGB - osReserveGB : 0;
  // 当前可用内存留 10% 余量：刚好卡在边界上照样会在加载过程中失败
  const ceilingFree = freeGB > 0 ? freeGB * 0.9 : 0;

  const candidates = list
    .map((g) => {
      const neededGB = need(g.totalBytes);
      const fitsFree = ceilingFree > 0 && neededGB <= ceilingFree;
      const fitsTotal = ceilingTotal > 0 && neededGB <= ceilingTotal;
      // spill：物理内存装不下，但**确实配了页面文件**能兜住 → 能跑，只是慢到不可用。
      // 没配页面文件时不存在这条退路，不能拿 commitLimit（此时等于物理内存）糊弄成“能跑”。
      const fitsCommit = pagefileGB > 0 && commitLimitGB > 0 && neededGB <= commitLimitGB;
      const fits = fitsFree ? 'free' : fitsTotal ? 'total' : fitsCommit ? 'spill' : 'never';
      return { quant: g.quant, totalBytes: g.totalBytes, neededGB, fits };
    })
    // 从大到小排：挑选时“在装得下的里面选最大的”，就是质量最高的那一档
    .sort((a, b) => b.totalBytes - a.totalBytes);

  const order = ['free', 'total', 'spill'];
  // 没有内存信息时（容器、受限环境）不能瞎猜“装不下”，退回纯粹的质量偏好
  const hasBudget = totalGB > 0 || freeGB > 0 || commitLimitGB > 0;
  for (const tier of hasBudget ? order : []) {
    const pool = candidates.filter((c) => c.fits === tier);
    if (!pool.length) continue;
    // 同一档里优先命中偏好表（质量/体积平衡的经典档位），否则取最大的
    for (const want of QUANT_PREFERENCE) {
      const hit = pool.find((c) => c.quant === want);
      if (hit) {
        return {
          quant: hit.quant, fits: tier, neededGB: hit.neededGB, candidates,
          reason: describeFit(tier, hit.quant, hit.neededGB, { totalGB, freeGB, commitLimitGB }),
        };
      }
    }
    return {
      quant: pool[0].quant, fits: tier, neededGB: pool[0].neededGB, candidates,
      reason: describeFit(tier, pool[0].quant, pool[0].neededGB, { totalGB, freeGB, commitLimitGB }),
    };
  }
  // 全都不装下：仍给出最小的一档，但把实话说明白——用户有权自己决定要不要试
  const smallest = candidates[candidates.length - 1];
  return {
    quant: smallest.quant, fits: 'never', neededGB: smallest.neededGB, candidates,
    reason: describeFit('never', smallest.quant, smallest.neededGB, { totalGB, freeGB, commitLimitGB }),
  };
}

/** 给用户看的一句话，说明为什么选这一档（以及这台机器的真实处境）。 */
function describeFit(tier, quant, neededGB, { totalGB, freeGB, commitLimitGB }) {
  const mem = totalGB > 0 ? `本机总内存 ${totalGB} GB、当前可用 ${freeGB} GB` : '本机内存未知';
  if (tier === 'free') return `已选 ${quant}（约需 ${neededGB} GB），当前可用内存装得下，可以正常速度运行。`;
  if (tier === 'total') return `已选 ${quant}（约需 ${neededGB} GB），超过当前可用内存但装得进总内存——${mem}，关掉其它占内存的程序再跑即可。`;
  if (tier === 'spill') {
    return `已选 ${quant}（约需 ${neededGB} GB），物理内存装不下，只能靠页面文件硬顶——${mem}，可提交上限 ${commitLimitGB} GB。`
      + '能跑但会频繁换页到硬盘，速度会慢到几乎不可用；建议选更小的一档。';
  }
  return `仓库里最小的一档 ${quant} 也需要 ${neededGB} GB，超过这台机器的内存上限（${mem}${commitLimitGB > 0 ? `，可提交上限 ${commitLimitGB} GB` : ''}）。`
    + '下载下来很可能加载失败或慢到不可用，建议换更小的模型。';
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
