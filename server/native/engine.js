/**
 * 本机大模型引擎（llama.cpp）：模型目录 + 下载管理 + 进程管理 + 对话转发。
 *
 * 权重存放：<dataDir>/models/<模型目录>/<量化>/（含 manifest.json 记录文件清单）。
 * 同一时间只运行一个模型（单机单卡/内存约束）；切换模型会自动卸载旧的。
 *
 * 模型来源有三类，聊天/API 用法完全一致：
 * 1. 内置目录（catalog.js 的 11 个 27B~671B 大模型）；
 * 2. Hugging Face 自定义模型：搜索/粘贴链接一键添加的任意仓库——GGUF 直接下载
 *    本机运行，Transformers 格式自动转 GGUF 后运行；
 * 3. HF 云端模型：本机跑不了的格式（ONNX/MLX/AWQ 等）在配好 HF Token 后
 *    走 Hugging Face 云端推理，模型 id 不变、用量照常统计。
 */
import { existsSync, readFileSync } from 'node:fs';
import os from 'node:os';
import { appendFile, mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  describeQuant,
  estimateRamGB,
  groupFilesByQuant,
  nativeModelById,
  pickDefaultQuant,
  pickEntryFile,
  pickQuantForBudget,
  NATIVE_MODELS,
} from './catalog.js';
import { detectFormats, detectVariants, hfModelId, optionsForFormats, routeForFormats, safeDirName } from './formats.js';
import { downloadFile, fetchModelInfo, fetchRepoFiles, freeDiskBytes, resolveFileUrl, searchModels } from './hf.js';
import {
  buildNumber, downloadLlamaBinary, findLlamaBinary, LlamaProcess, minBuildForArch, matchUnsupportedArch,
} from './llama.js';
import { extractZip } from './archive.js';
import { readGgufBlockCount, readGgufMeta } from './gguf.js';
import { findConverter, prepareConverter, runConvert } from './convert.js';
import { findPythonRuntime, preparePythonRuntime, PythonProcess } from './python-runner.js';
import { planPagefile, readMemoryBudget } from './sysmem.js';
import { CPU_LEVELS, detectCpuLevel, parseCpuFeatures } from './cpu.js';

const QUANT_CACHE_MS = 10 * 60 * 1000;

/** 失败定性的统一文案：内外层共用，避免两处各写一套导致结论互相矛盾。 */
const DIAG_TITLES = {
  architecture: {
    title: '这个 llama.cpp 版本不认识该模型的架构',
    advice: '模型文件是好的，是 llama-server 太旧。点“准备运行库”更新到更新的版本后重试。',
  },
  memory: { title: '内存/显存不足', advice: '换更小的量化，或关掉其它占用内存/显存的程序后重试。' },
  runtime: {
    title: '运行库或程序本身有问题',
    advice: '点“准备运行库”重装 llama.cpp；若装的是 CUDA/Vulkan 版，把 OJ_LLAMA_VARIANT 设为 cpu 改用纯 CPU 版；并确认已安装 VC++ 运行库。',
  },
  corrupt: { title: '权重文件损坏或不完整', advice: '删除这一档量化后重新下载（会断点续传），再点加载。' },
  silent: {
    title: 'llama-server 启动即退出且没有任何日志',
    advice: '这通常不是内存不够（内存不足时会先打印加载进度）。请点“准备运行库”重装 llama.cpp；'
      + '若仍失败，把 OJ_LLAMA_VARIANT 设为 cpu 改用纯 CPU 版，或用系统命令行直接运行 llama-server 看系统报错。',
  },
  unknown: { title: 'llama-server 启动失败', advice: '请把状态面板里的日志尾部发给开发者。' },
};

/** 一组加载参数的指纹：用于去重/判断“这一组是不是刚试过”。 */
function attemptKey(a) {
  return `${a.layers}|${a.contextTokens}|${a.flashAttn}|${a.cacheReuse}|${a.mmap}|${a.batch}`;
}

/**
 * 粗略估算一段对话要占多少上下文 token。
 * 中文约 1 token/字、英文约 4 字符/token，模板与特殊符号另有开销，统一按
 * “每字符 0.6 token”偏保守估算——宁可少留一点历史，也不要让 llama.cpp 直接 400。
 */
function estimateTokens(text) {
  const s = String(text || '');
  if (!s) return 0;
  let wide = 0;
  for (const ch of s) {
    const code = ch.codePointAt(0);
    if (code > 0x2e80) wide += 1; // CJK 等宽字符：约 1 token/字
  }
  const narrow = s.length - wide;
  return Math.ceil(wide * 1 + narrow / 3.2 + 4);
}

/**
 * 把消息裁剪到上下文窗口内：保留最前面的 system（系统提示/检索证据）与最新的消息。
 * 超出时从最早的非 system 消息开始丢；仍然装不下再截断超长单条消息的中间部分。
 * @returns {{messages: Array, estimatedTokens: number, contextTokens: number, droppedMessages: number}}
 */
export function fitMessagesToContext(messages, { contextTokens = 8192, reserveForOutput = 1024 } = {}) {
  const list = Array.isArray(messages) ? messages : [];
  const ctx = Math.max(512, Math.floor(contextTokens) || 8192);
  // 至少给输出留 25% 空间：很多模型的 max_tokens 请求值本身就接近窗口
  const outputReserve = Math.max(256, Math.min(Math.floor(ctx / 2), Math.floor(reserveForOutput) || 1024));
  const budget = Math.max(256, ctx - outputReserve);
  const tokensOf = (m) => estimateTokens(m?.content) + 8;
  const systems = list.filter((m) => m?.role === 'system');
  const rest = list.filter((m) => m?.role !== 'system');
  const systemTokens = systems.reduce((n, m) => n + tokensOf(m), 0);
  // 从最新往回收，直到装不下；再保证最后一条一定是用户消息
  const kept = [];
  let used = systemTokens;
  for (let i = rest.length - 1; i >= 0; i -= 1) {
    const cost = tokensOf(rest[i]);
    if (used + cost > budget && kept.length) break;
    used += cost;
    kept.unshift(rest[i]);
    if (used >= budget) break;
  }
  let dropped = rest.length - kept.length;
  // 单条消息本身就超预算（例如一整段长文）：保留头尾，中间省略，明确告知模型。
  // 字符数用估算函数二分逼近，而不是按固定系数换算——中英文的 token/字符比差 3 倍。
  if (kept.length === 1 && used > budget) {
    const only = kept[0];
    const text = String(only.content || '');
    const room = budget - systemTokens - 16;
    const fits = (chars) => estimateTokens(
      `${text.slice(0, Math.ceil(chars * 0.6))}${text.slice(-Math.floor(chars * 0.4))}`,
    ) <= room;
    if (!fits(text.length)) {
      let lo = 200;
      let hi = text.length;
      while (hi - lo > 64) {
        const mid = Math.floor((lo + hi) / 2);
        if (fits(mid)) lo = mid; else hi = mid;
      }
      const head = Math.ceil(lo * 0.6);
      const tail = lo - head;
      kept[0] = {
        ...only,
        content: `${text.slice(0, head)}\n\n…（内容过长，中间约 ${text.length - lo} 个字符已省略，请基于首尾内容回答）…\n\n${text.slice(-tail)}`,
      };
    }
  }
  // 最后一条不是用户消息时（历史以助手消息结尾会长这样），补回来的那条已在上面的循环里
  const out = [...systems, ...kept];
  return {
    messages: out,
    estimatedTokens: out.reduce((n, m) => n + tokensOf(m), 0),
    contextTokens: ctx,
    droppedMessages: dropped,
  };
}

/**
 * 两份量化分组是否等价（档位名、分片路径、分片大小全一致）。
 * 用来判断缓存能不能直接用：仓库重新上传过文件时大小会变，继续用缓存会让
 * 下载端以旧的 expectedSize 校验，永远报“文件下载不完整”。
 */
function sameQuantSizes(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  const byQuant = new Map(b.map((g) => [g.quant, g]));
  for (const g of a) {
    const other = byQuant.get(g.quant);
    if (!other || other.files.length !== g.files.length) return false;
    const sizes = new Map(other.files.map((f) => [f.path, f.size]));
    for (const f of g.files) if (sizes.get(f.path) !== f.size) return false;
  }
  return true;
}
/**
 * 浏览器通道任务“已中断”判定：超过这个时长没有收到任何新数据块，就认为
 * 浏览器页面已关/已断（关页面时客户端的失败通知发不出来，服务端不能干等）。
 * 中断后任务转为 error，前端按钮随之恢复可点，重新点下载会按偏移续传。
 */
const BROWSER_JOB_STALE_MS = 180 * 1000;

/** 下发给界面前的错误兜底：任何一层漏出的英文裸错（fetch failed 等）都换成中文。 */
function visibleDownloadError(text) {
  const raw = String(text || '').trim();
  if (/^fetch failed$/i.test(raw) || /failed to fetch|networkerror/i.test(raw)) {
    return '连接中断（网络或代理不稳）：已下载的部分都保留着，重新点下载会从断点续传。若反复失败，先在浏览器里打开 huggingface.co 确认能访问。';
  }
  return raw;
}
export const HF_ROUTER_URL = 'https://router.huggingface.co/v1/chat/completions';

export class NativeEngine {
  constructor({
    dataDir,
    fetchImpl = fetch,
    spawnImpl,
    llama = null,
    variant = 'cpu',
    env = process.env,
    logger = console,
    execImpl = null,
  } = {}) {
    if (!dataDir) throw new Error('NativeEngine 需要 dataDir。');
    this.dataDir = dataDir;
    this.modelsDir = path.join(dataDir, 'models');
    this.fetchImpl = fetchImpl;
    this.variant = variant;
    this.env = env;
    this.logger = logger;
    this.execImpl = execImpl; // 转换/准备工具时注入（测试用）
    this.llama = llama || new LlamaProcess({ fetchImpl, spawnImpl, logger });
    this.downloads = new Map(); // key -> {id, quant, state, downloaded, total, error}
    this.importSessions = new Map(); // 本地导入会话：importId -> {id, at, declared}
    this.quantCache = new Map(); // repo -> {at, grouped}
    this.binaryJob = null;
    this.conversions = new Map(); // id -> {id, state, error}
    this.converterJob = null;
    this._converterCache = null;
    // 本地 Python 运行时（特殊格式原格式本地运行）
    this.python = new PythonProcess({ fetchImpl, logger });
    this.pythonJobs = new Map(); // id -> {id, state, stage, error, promise}
    this.runnerJob = null;
    this._runtimeCache = null;
    // 自定义 HF 模型与云端模型注册表（同步读，启动即可用）
    this.custom = new Map();
    this.remote = new Set();
    this.preferCloud = new Set(); // 用户明确点了“云端使用”的模型：聊天优先走云端
    this.hfToken = env.HF_TOKEN || '';
    this._loadRegistries();
  }

  _registryPath(name) {
    return path.join(this.dataDir, name);
  }

  _loadRegistries() {
    try {
      const list = JSON.parse(readFileSync(this._registryPath('custom-models.json'), 'utf8'));
      for (const e of Array.isArray(list) ? list : []) {
        // 本地导入的条目没有 HF 仓库（repo 为空），靠 source 标记识别，别被过滤掉
        if (e?.id && (e?.repo || e?.source === 'import')) this.custom.set(e.id, e);
      }
    } catch { /* 首次运行没有注册表 */ }
    try {
      const saved = JSON.parse(readFileSync(this._registryPath('hf-remote.json'), 'utf8'));
      const remoteList = Array.isArray(saved) ? saved : saved?.remote || [];
      const preferList = Array.isArray(saved) ? [] : saved?.preferCloud || [];
      for (const id of remoteList) if (id) this.remote.add(id);
      for (const id of preferList) if (id) this.preferCloud.add(id);
    } catch { /* 同上 */ }
    if (!this.hfToken) {
      try {
        const saved = JSON.parse(readFileSync(this._registryPath('hf-token.json'), 'utf8'));
        if (typeof saved?.token === 'string') this.hfToken = saved.token;
      } catch { /* 未配置云端 Token */ }
    }
  }

  async _persistCustom() {
    await mkdir(this.dataDir, { recursive: true });
    await writeFile(this._registryPath('custom-models.json'), `${JSON.stringify([...this.custom.values()], null, 2)}\n`);
  }

  async _persistRemote() {
    await mkdir(this.dataDir, { recursive: true });
    await writeFile(
      this._registryPath('hf-remote.json'),
      `${JSON.stringify({ remote: [...this.remote], preferCloud: [...this.preferCloud] }, null, 2)}\n`,
    );
  }

  /** 用户切换到本地路线时取消云端优先（云端登记保留，随时可再切回）。 */
  async _preferLocal(id) {
    if (this.preferCloud.delete(id)) await this._persistRemote();
  }

  /** 保存 HF 云端 Token（只写不读回明文；文件权限 0600）。 */
  async setHfToken(token) {
    const value = String(token || '').trim();
    this.hfToken = value;
    await mkdir(this.dataDir, { recursive: true });
    if (value) {
      await writeFile(this._registryPath('hf-token.json'), `${JSON.stringify({ token: value })}\n`, { mode: 0o600 });
    } else {
      await rm(this._registryPath('hf-token.json'), { force: true });
    }
    return { hasToken: !!value };
  }

  /** 解析模型：内置目录或 HF 自定义注册表；找不到返回 null。 */
  resolveModel(id) {
    const key = String(id || '');
    const builtin = nativeModelById(key);
    if (builtin) return { ...builtin, kind: 'catalog' };
    const custom = this.custom.get(key);
    if (custom) {
      const imported = custom.source === 'import';
      return {
        id: custom.id,
        name: custom.name || custom.repo,
        vendor: imported ? '本地导入' : 'Hugging Face',
        paramsB: custom.paramsB || null,
        activeB: null,
        reasoning: false,
        license: imported ? '见原始发布说明' : '见仓库说明',
        context: custom.context || 32768,
        note: imported ? '从本机文件导入的模型（浏览器下载后导入）。' : '从 Hugging Face 添加的模型。',
        repo: custom.repo || '',
        formats: custom.formats || [],
        kind: imported ? 'import' : 'hf',
      };
    }
    return null;
  }

  hasModel(id) {
    return !!this.resolveModel(id);
  }

  /** 全部可用模型条目（内置 + 自定义），供状态与 /v1/models 使用。 */
  listModels() {
    return [
      ...NATIVE_MODELS.map((m) => ({ ...m, kind: 'catalog' })),
      ...[...this.custom.keys()].map((id) => this.resolveModel(id)).filter(Boolean),
    ];
  }

  modelDir(id, quant) {
    const base = path.join(this.modelsDir, safeDirName(id));
    return quant ? path.join(base, quant) : base;
  }

  /** 已下载的量化（读 manifest 并核对文件大小）。 */
  async downloadedQuants(id) {
    const base = this.modelDir(id);
    let quants = [];
    try {
      quants = await readdir(base);
    } catch {
      return [];
    }
    const out = [];
    for (const quant of quants) {
      const manifest = await this.readManifest(id, quant);
      if (!manifest) continue;
      let ok = true;
      for (const f of manifest.files) {
        try {
          const s = await stat(path.join(base, quant, f.path));
          if (s.size !== f.size) {
            ok = false;
            break;
          }
        } catch {
          ok = false;
          break;
        }
      }
      if (ok) {
        out.push({
          quant,
          totalBytes: manifest.totalBytes,
          files: manifest.files.length,
          // 层数（读 GGUF 文件头得到）：界面据此提示“每层约占多少显存”；读不到为 null。
          // 分片模型必须读第 1 个分片的文件头（元数据只写在首片里），
          // 以前读 manifest.files[0]，按量化分子目录布局时可能读到中段分片而显示“层数未知”。
          layers: quant === 'SNAPSHOT' || quant === 'CONVERTED'
            ? null
            : await this._ggufLayers(path.join(base, quant, pickEntryFile(manifest.files)?.path || ''), manifest.totalBytes),
          ...describeQuant(quant),
        });
      }
    }
    return out;
  }

  /**
   * 磁盘上有量化目录、但没通过完整性核对（缺片/截断/大小不一致）的档位。
   *
   * 这类“下了一半”的量化以前对上层完全不可见：`downloadedQuants` 只返回完整的，
   * 于是点加载得到的是“尚未下载”，用户只能把几十 GB 删掉重下——而其实 .part 与
   * 已下完的分片都在，续传一次就能用。
   *
   * @returns {Promise<Array<{quant: string, summary: string, files: number}>>}
   */
  async _brokenQuants(id) {
    const base = this.modelDir(id);
    let dirs = [];
    try {
      dirs = await readdir(base);
    } catch {
      return [];
    }
    const out = [];
    for (const quant of dirs) {
      if (quant === 'SNAPSHOT' || quant === 'CONVERTED') continue;
      const manifest = await this.readManifest(id, quant);
      if (!manifest) {
        // 没有清单通常意味着“还没下完就中断了”：清单只在全部文件到位后才写。
        // 这里按 .part 的存在与否给出可续传的提示，同样不要报成“尚未下载”。
        const parts = await this._countPartFiles(path.join(base, quant));
        if (parts > 0) {
          out.push({ quant, files: parts, summary: `有 ${parts} 个文件只下到一半（.part）` });
        }
        continue;
      }
      let present = 0;
      let missing = 0;
      let short = 0;
      for (const f of manifest.files) {
        const dest = path.join(base, quant, f.path);
        const want = Number(f.size) || 0;
        let size = -1;
        try {
          size = (await stat(dest)).size;
        } catch {
          // 正式文件不在：看有没有 .part（有就是下到一半，续传即可）
          try {
            size = (await stat(`${dest}.part`)).size;
          } catch {
            size = -1;
          }
        }
        if (size < 0) missing += 1;
        else {
          present += 1;
          if (want && size < want) short += 1;
        }
      }
      if (!missing && !short) continue; // 完整档位由 downloadedQuants 负责
      const bits = [];
      if (missing) bits.push(`缺 ${missing} 个分片`);
      if (short) bits.push(`${short} 个分片还没下完`);
      out.push({
        quant,
        files: manifest.files.length,
        summary: `${bits.join('、')}（共 ${manifest.files.length} 个，已就位 ${present} 个）`,
      });
    }
    return out;
  }

  /** 统计一个量化目录下还没转正的 .part 文件数（递归，最多看两层子目录）。 */
  async _countPartFiles(dir, depth = 0) {
    if (depth > 1) return 0;
    let entries = [];
    try {
      entries = await readdir(dir, { withFileTypes: true });
    } catch {
      return 0;
    }
    let n = 0;
    for (const e of entries) {
      if (e.isDirectory()) n += await this._countPartFiles(path.join(dir, e.name), depth + 1);
      else if (e.isFile() && /\.part$/i.test(e.name)) n += 1;
    }
    return n;
  }

  /** GGUF 层数（按“文件+大小”缓存，避免状态刷新时反复读文件头）。 */
  async _ggufLayers(filePath, totalBytes) {
    if (!filePath) return null;
    const key = `${filePath}|${totalBytes || 0}`;
    if (this._layerCache?.has(key)) return this._layerCache.get(key);
    const layers = await readGgufBlockCount(filePath).catch(() => null);
    if (!this._layerCache) this._layerCache = new Map();
    if (this._layerCache.size > 200) this._layerCache.clear();
    this._layerCache.set(key, layers);
    return layers;
  }

  async readManifest(id, quant) {
    try {
      const raw = await readFile(path.join(this.modelDir(id, quant), 'manifest.json'), 'utf8');
      const data = JSON.parse(raw);
      if (!Array.isArray(data.files)) return null;
      return data;
    } catch {
      return null;
    }
  }

  /** 仓库里实际可下载的量化（带缓存）。 */
  async availableQuants(id) {
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    const cached = this.quantCache.get(model.repo);
    // 新鲜缓存直接返回：大仓库的文件树是分页拉取的（几千个文件、十几页），
    // 状态面板每次刷新都重拉一遍会明显拖慢界面。缓存过期后再核对大小（见下）。
    if (cached && Date.now() - cached.at < QUANT_CACHE_MS) return cached.grouped;
    // 带 Token：门控/私有仓库没有鉴权头连文件树都读不到（会误报“没有 GGUF”）。
    const files = await fetchRepoFiles(model.repo, { fetchImpl: this.fetchImpl, headers: this._hfAuthHeaders() });
    const grouped = groupFilesByQuant(files);
    if (cached && sameQuantSizes(cached.grouped, grouped)) {
      // 仓库没变：沿用旧分组（顺序稳定），只更新时间
      cached.at = Date.now();
      return cached.grouped;
    }
    if (!grouped.length) {
      throw new Error(`仓库 ${model.repo} 里没有找到可用的 GGUF 量化文件。`);
    }
    this.quantCache.set(model.repo, { at: Date.now(), grouped });
    return grouped;
  }

  converterStatus() {
    if (this._converterCache && Date.now() - this._converterCache.at < 30000) return this._converterCache.value;
    const value = findConverter({ dataDir: this.dataDir, env: this.env });
    this._converterCache = { at: Date.now(), value };
    return value;
  }

  runtimeStatus() {
    if (this._runtimeCache && Date.now() - this._runtimeCache.at < 30000) return this._runtimeCache.value;
    const value = findPythonRuntime({ dataDir: this.dataDir, env: this.env });
    this._runtimeCache = { at: Date.now(), value };
    return value;
  }

  /**
   * 把“浏览器页面已关/已断但失败通知没发出来”的浏览器通道任务转成中断状态。
   * 不处理服务端直连任务（它有自己的后台流程会给出成败）。
   */
  _expireStaleBrowserJobs() {
    const now = Date.now();
    for (const job of this.downloads.values()) {
      if (job.via !== 'browser' || job.state !== 'downloading') continue;
      const last = job.lastChunkAt || job.startedAt || 0;
      if (now - last > BROWSER_JOB_STALE_MS) {
        job.state = 'error';
        job.error = '下载已中断（页面关闭或网络断开超过 3 分钟），重新点下载会从断点续传。';
      }
    }
  }

  async status() {
    this._expireStaleBrowserJobs();
    const binary = findLlamaBinary({ env: this.env, dataDir: this.dataDir });
    const models = [];
    for (const m of this.listModels()) {
      const downloaded = await this.downloadedQuants(m.id);
      // 下了但没下完的档位也要让界面知道：否则用户在量化下拉里根本看不到它，
      // 点加载只会得到“还没下载完”，却不知道是哪一档、还差多少。
      const incomplete = await this._brokenQuants(m.id);
      // 只下发展示需要的字段：内部 entry（含整份文件清单）、receivedByFile（Map）不外传
      const dl = [...this.downloads.values()]
        .filter((d) => d.id === m.id)
        .map((d) => ({
          id: d.id, quant: d.quant, state: d.state,
          downloaded: d.downloaded, total: d.total, error: visibleDownloadError(d.error),
          via: d.via || 'server',
        }));
      models.push({
        id: m.id,
        name: m.name,
        vendor: m.vendor,
        paramsB: m.paramsB ?? null,
        activeB: m.activeB ?? null,
        reasoning: !!m.reasoning,
        license: m.license,
        context: m.context,
        note: m.note,
        repo: m.repo,
        kind: m.kind,
        formats: m.formats || ['gguf'],
        downloaded,
        incomplete,
        downloads: dl,
        remote: this.remote.has(m.id),
        conversion: this.conversions.get(m.id) || null,
        pythonJob: this.pythonJobs.get(m.id) ? { state: this.pythonJobs.get(m.id).state, stage: this.pythonJobs.get(m.id).stage, error: this.pythonJobs.get(m.id).error } : null,
        running: (this.llama.info?.modelId === m.id && this.llama.state !== 'idle')
          || (this.python.info?.modelId === m.id && this.python.state !== 'idle'),
      });
    }
    return {
      enabled: true,
      binaryReady: !!binary,
      binarySource: binary?.source || '',
      binaryJob: this.binaryJob
        ? {
            state: this.binaryJob.state,
            error: this.binaryJob.error || '',
            tag: this.binaryJob.tag || '',
            build: this.binaryJob.build || null,
            runnable: this.binaryJob.runnable ?? null,
            probeMessage: this.binaryJob.probeMessage || '',
          }
        : null,
      // 运行库健康度：让界面能提前警告“这个 llama.cpp 在本机跑不起来 / 版本太旧”，
      // 而不是等用户点了加载、等半天、再看到一个看不懂的崩溃码。
      binaryHealth: this.binaryHealth || null,
      // 本机 CPU 指令集（由 llama-server --version 自己报告，不查型号表）
      cpu: this.cpuInfo || null,
      running: this.llama.status(),
      // GPU 卸载自动回退的说明（请求层数装不下时降到了多少层），界面据此提示用户
      loadNote: this.loadNote || null,
      // 最近一次生成是否因上下文不够而省略了历史（面板据此提示用户）
      contextTrim: this.contextTrim || null,
      // 最近一次加载前算出的内存需求 vs 本机实际（面板显示，避免用户对着崩溃码猜）
      memory: this.memoryReport || null,
      pythonRunning: this.python.status(),
      models,
      hf: {
        hasToken: !!this.hfToken,
        remote: [...this.remote],
        converter: this.converterStatus(),
        converterJob: this.converterJob ? { state: this.converterJob.state, error: this.converterJob.error || '' } : null,
        runner: this.runtimeStatus(),
        runnerJob: this.runnerJob ? { state: this.runnerJob.state, error: this.runnerJob.error || '' } : null,
      },
    };
  }

  /** 管理员：准备 llama.cpp 运行库（已存在则直接返回，否则下载）。 */
  /**
   * 准备 llama.cpp 运行库。
   *
   * @param {object} [opts]
   * @param {boolean} [opts.force] 强制重新下载。
   *   以前这里是「找到就返回」，导致“准备运行库”在已有二进制时变成空操作 ——
   *   用户遇到版本太旧（不认新的模型架构）时，点多少次都还是那个旧版本。
   * @param {string} [opts.tag] 指定要装的构建号（如 b8421）。用于锁定已知可用的版本；
   *   新版在老系统上静默崩溃时，这是唯一能让软件跑起来的办法。
   */
  async ensureBinary({ force = false, tag = '' } = {}) {
    const wantTag = String(tag || this.env.OJ_LLAMA_TAG || '').trim();
    if (!force) {
      const found = findLlamaBinary({ env: this.env, dataDir: this.dataDir });
      if (found) return found;
    }
    if (this.binaryJob?.state === 'downloading') return this.binaryJob.promise;
    this.binaryJob = { state: 'downloading', error: '', tag: wantTag };
    this.binaryJob.promise = (async () => {
      try {
        // 装完必须实测能不能跑。旧版 llama.cpp 按指令集分包（avx512/avx2/avx/noavx），
        // 挑错档位就是“非法指令”或直接崩；新版是单包（运行时按 CPU 分发 ggml-cpu-*.dll）。
        // 无论哪种，都让 CPU 自己回答：跑不了就降一档重来，直到能跑或试完。
        // 这样不需要任何 CPU 型号表 —— 换任何一台机器都适用。
        const attempts = [];
        let got = null;
        let probe = null;
        const levels = this.variant === 'cpu' && process.platform === 'win32' && process.arch === 'x64'
          ? CPU_LEVELS
          : [''];
        for (const level of levels) {
          const candidate = await downloadLlamaBinary({
            dataDir: this.dataDir,
            fetchImpl: this.fetchImpl,
            variant: this.variant,
            tag: wantTag,
            cpuLevel: level || 'avx2',
            onProgress: (p) => { this.binaryJob.progress = p; },
          });
          got = candidate;
          probe = this.llama._probeRun(candidate.path, path.dirname(candidate.path));
          attempts.push({ level: candidate.cpuLevel || level || 'cpu', asset: candidate.asset, ok: probe.ok !== false });
          if (probe.ok !== false) break;
          // 单包（新版）没有档位可降，实测失败就直接收手，不要反复下载同一个包
          if (!candidate.cpuLevel || candidate.cpuLevel === 'cpu') break;
          this.logger.warn?.(`[native] ${candidate.cpuLevel} 版在本机跑不了，自动降一档重试：${(probe.message || '').split('\n')[0]}`);
        }
        this.binaryJob = {
          state: 'done',
          error: '',
          tag: got?.version || wantTag,
          build: buildNumber(got?.version || wantTag),
          runnable: probe?.ok !== false,
          probeMessage: probe?.ok === false ? probe.message : '',
          // 实际装上的指令集档位 + 试过哪些：界面据此告诉用户“你的 CPU 用的是 avx2 版”
          cpuLevel: got?.cpuLevel || '',
          cpuAttempts: attempts,
        };
        if (probe?.ok === false) {
          this.logger.warn?.(`[native] 新装的 llama.cpp 在本机启动不了：${probe.message.split('\n')[0]}`);
        }
        return { path: got.path, source: 'downloaded', version: got?.version || '', cpuLevel: got?.cpuLevel || '' };
      } catch (err) {
        this.binaryJob = { state: 'error', error: err.message, tag: wantTag };
        throw err;
      }
    })();
    return this.binaryJob.promise;
  }

  /**
   * 读出这档量化的模型架构（如 qwen35moe / llama / qwen2）。
   * 读不到就返回空字符串：架构未知时不能拦加载，只能交给 llama.cpp 自己判断。
   */
  async _modelArch(id, quant) {
    try {
      const manifest = await this.readManifest(id, quant);
      const entry = pickEntryFile(manifest.files);
      if (!entry) return '';
      const file = path.join(this.modelDir(id, quant), entry.path);
      if (!existsSync(file)) return '';
      const meta = await readGgufMeta(file);
      return String(meta?.architecture || meta?.general?.architecture || '').trim();
    } catch {
      return '';
    }
  }

  /**
   * 评估当前 llama.cpp 运行库在本机是否可用，并给出可执行的建议。
   *
   * 回答两个用户真实问过的问题：
   *   ①“我的系统支持不了最新的 llama 吗？”—— 实测一次 --version，零输出崩溃就是不支持；
   *   ②“当前的 llama 是不是太旧？”—— 读它的构建号，与已知需要的最低版本比较。
   *
   * @param {object} [opts]
   * @param {string} [opts.forArch] 目标模型架构（如 qwen35moe），用来判断“够不够新”
   */
  async checkBinaryHealth({ forArch = '' } = {}) {
    const found = findLlamaBinary({ env: this.env, dataDir: this.dataDir });
    if (!found) {
      return {
        state: 'missing',
        ok: false,
        message: '还没有安装 llama.cpp 运行库。点“准备运行库”下载（纯 CPU 版约 30–50 MB）。',
      };
    }
    const probe = this.llama._probeRun(found.path, path.dirname(found.path));
    // 顺便记下 CPU 指令集：新版的 --version 会打印 "CPU : AVX = 1 | AVX2 = 1 ..."。
    // 读不到就保持 null（不猜、不编）。
    if (probe.ok !== false) {
      const feats = parseCpuFeatures(probe.output || probe.version || '');
      if (feats) this.cpuInfo = feats;
    }
    if (probe.ok === false) {
      // 文件在、但真的启动不了：这正是“系统不支持这个 llama 构建”的情形
      return {
        state: 'unrunnable',
        ok: false,
        build: null,
        message: probe.message,
        hint: '可以把 OJ_LLAMA_TAG 设成一个较早的构建号（例如 b4000）后点“准备运行库”强制重装；'
          + '较早的构建对老系统兼容性更好。',
      };
    }
    const build = buildNumber(probe.version);
    const minBuild = forArch ? minBuildForArch(forArch) : 0;
    const tooOld = !!(minBuild && build && build < minBuild);
    return {
      state: tooOld ? 'tooOld' : 'ok',
      ok: !tooOld,
      build: build || null,
      version: probe.version || '',
      // 这个构建用的是哪一档指令集 / 本机 CPU 支持什么：
      // 老版按 avx512/avx2/avx/noavx 分包，挑错档位就跑不起来，所以让用户看得见。
      cpuLevel: this.binaryJob?.cpuLevel || '',
      cpu: this.cpuInfo || null,
      minBuild: minBuild || null,
      forArch: forArch || '',
      message: tooOld
        ? `当前 llama.cpp 是 b${build}，但 ${forArch} 架构需要 b${minBuild} 或更新，加载这个模型会失败。`
        : (build ? `llama.cpp 运行库正常（b${build}）。` : 'llama.cpp 运行库可以启动。'),
      hint: tooOld ? '点“准备运行库”更新 llama.cpp 后重试。' : '',
    };
  }

  /** 给搜索结果粗判定格式（标签/库名/仓库名级别，详情时再按真实文件确认）。 */
  _annotateSearchResults(results) {
    return (Array.isArray(results) ? results : []).map((r) => {
      const guess = detectFormats([], { tags: [...(r.tags || []), String(r.repo || '').toLowerCase().includes('gguf') ? 'gguf' : ''], library_name: r.library });
      return { ...r, formats: guess };
    });
  }

  /** HF 搜索（桥接 Hub API，服务端直连；浏览器通道不可用时的回退）。 */
  async searchHf(query, opts = {}) {
    const results = await searchModels({ query, ...opts, fetchImpl: this.fetchImpl, headers: this._hfAuthHeaders() });
    return this._annotateSearchResults(results);
  }

  /**
   * HF 搜索（浏览器通道）：浏览器已从 Hub 取回搜索结果，这里只在本地做
   * 格式粗判定，与普通网页版同一条网络路，不再要求软件直连 huggingface.co。
   */
  searchFromMeta(results) {
    return this._annotateSearchResults(results);
  }

  /** 把仓库元信息拼成详情（格式/路线/可运行方式/量化清单）。 */
  _buildInfo(repo, meta, files) {
    const cleanFiles = (Array.isArray(files) ? files : [])
      .filter((f) => f && typeof f.path === 'string')
      .slice(0, 50000)
      .map((f) => ({ path: f.path, size: Math.max(0, Number(f.size) || 0), sha256: typeof f.sha256 === 'string' ? f.sha256 : '' }));
    const info = {
      repo,
      name: repo.split('/').pop() || repo,
      author: meta?.author || repo.split('/')[0] || '',
      downloads: Number(meta?.downloads) || 0,
      likes: Number(meta?.likes) || 0,
      pipeline: meta?.pipeline_tag || '',
      library: meta?.library_name || '',
      tags: Array.isArray(meta?.tags) ? meta.tags : [],
      gated: !!meta?.gated,
      files: cleanFiles,
      totalBytes: cleanFiles.reduce((n, f) => n + (f.size || 0), 0),
    };
    const formats = detectFormats(info.files, info);
    const variants = detectVariants(info.files, info);
    const route = routeForFormats(formats, variants);
    const options = optionsForFormats(formats, variants);
    // GGUF 仓库顺带给出量化清单（类型/是否可直接使用），供详情页选档
    let quants = [];
    if (formats.includes('gguf')) {
      quants = groupFilesByQuant(cleanFiles).map((g) => ({
        quant: g.quant,
        files: g.files.length,
        totalBytes: g.totalBytes,
        ...describeQuant(g.quant),
      }));
    }
    return { ...info, formats, variants, route, options, quants };
  }

  /** HF 仓库详情 + 格式路线（服务端直连；浏览器通道不可用时的回退）。 */
  async hfInfo(repo) {
    const info = await fetchModelInfo(repo, { fetchImpl: this.fetchImpl, headers: this._hfAuthHeaders() });
    return this._buildInfo(repo, info, info.files);
  }

  /** HF 仓库详情（浏览器通道）：浏览器取回的 meta+文件树在本地判定。 */
  hfInfoFromMeta(repo, meta, files) {
    return this._buildInfo(repo, meta || {}, files);
  }

  /** 把一个 HF 仓库登记为自定义模型（幂等），返回模型 id。 */
  async registerCustom(repo, { formats = [], name } = {}) {
    const id = hfModelId(repo);
    if (!this.custom.has(id)) {
      this.custom.set(id, {
        id,
        repo,
        name: name || repo.split('/').pop() || repo,
        formats,
        context: 32768,
        addedAt: new Date().toISOString(),
      });
      await this._persistCustom();
    } else if (formats.length) {
      this.custom.get(id).formats = formats;
      await this._persistCustom();
    }
    return id;
  }

  /**
   * 使用一个 HF 仓库（运行方式可切换，云端只是其中一个选项）：
   * - mode='auto'（一键使用）：GGUF 下载本地跑；Transformers 转 GGUF；ONNX 等特殊
   *   格式走本地 Python 运行时；MLC 提示浏览器路线；
   * - mode='local'：强制本地（同 auto 的本地路线）；
   * - mode='python'：强制原格式本地 Python 运行；
   * - mode='cloud'：云端使用（需 HF Token，不下载）。
   * 任何格式都保证有本地路线（见 formats.optionsForFormats）。
   */
  /**
   * @param {{info?: object, viaBrowser?: boolean}} [opts]
   *   info：浏览器通道已取回并在本地判定好的详情（有它就不再直连 HF）；
   *   viaBrowser：下载路线改由浏览器负责传字节，这里只登记与推荐量化。
   */
  async useRepo(repo, mode = 'auto', { info: preloaded = null, viaBrowser = false, quant = '' } = {}) {
    const info = preloaded || await this.hfInfo(repo);
    const id = await this.registerCustom(repo, { formats: info.formats, name: info.name });
    // 浏览器通道顺手把 GGUF 量化分组写入缓存：后续下载/回退直连选档不必再列树
    if (preloaded && (info.formats || []).includes('gguf') && Array.isArray(info.files) && info.files.length) {
      try { this.groupQuantsFromFiles(id, info.files); } catch { /* 无 GGUF 文件时不阻塞使用流程 */ }
    }
    const { route, options } = info;
    const want = ['auto', 'local', 'python', 'cloud'].includes(mode) ? mode : 'auto';

    if (want === 'cloud') {
      if (!this.hfToken) return { id, action: 'needs-token', route, options, formats: info.formats };
      this.remote.add(id);
      this.preferCloud.add(id);
      await this._persistRemote();
      return { id, action: 'remote', route, options, formats: info.formats };
    }
    if (want === 'python') {
      if (options.some((o) => o.type === 'local-python')) {
        return { id, action: 'python', route, options, runnerReady: this.runtimeStatus().ready, formats: info.formats };
      }
      // 没有 Python 路线的格式（如纯 GGUF）回落到它的本地路线
    }
    // local / auto / python 回落：按默认路线走本地
    if (route.route === 'local') {
      if (viaBrowser) {
        // 浏览器通道：只返回推荐量化，字节由浏览器逐块传（和网页版同一条路）；
        // 不在这里触发服务端直连下载，否则用户电脑上又变成立即失败。
        const grouped = this.quantCache.get(repo)?.grouped || [];
        if (!grouped.length) {
          return { id, action: 'download', via: 'browser', route, options, quant: null, formats: info.formats };
        }
        // **多档量化时不再替用户决定。**
        // 以前这里直接 pickDefaultQuant（偏好表里 Q4_K_M 排第一），完全不管本机
        // 内存装不装得下：16 GB 机器上挑了 15.5 GB 那档，用户下完一整晚才发现跑不动。
        // 现在默认把选择权交回去；只有“用户明确指定了量化”或“仓库只有一档”才直接开下。
        const chosen = String(quant || '').trim().toUpperCase();
        if (!chosen && grouped.length > 1) {
          const pick = this.quantChoice(id, grouped);
          return {
            id, action: 'choose-quant', via: 'browser', route, options,
            formats: info.formats,
            // 整张清单都交给界面（带大小/内存/装不装得下/推荐理由），
            // 用户看得到“这档要 19.4 GB、你的机器装不下”才不会再走进坑里。
            quants: pick.quants,
            recommended: pick.recommended,
            reason: pick.reason,
            budget: pick.budget,
          };
        }
        const entry = this._pickQuantEntry(grouped, chosen || undefined);
        return { id, action: 'download', via: 'browser', route, options, quant: entry?.quant || null, formats: info.formats };
      }
      // 服务端直连：同样先让用户选（多档时），避免又替用户下错一大坨
      const groupedDirect = await this.availableQuants(id).catch(() => []);
      const chosenDirect = String(quant || '').trim().toUpperCase();
      if (!chosenDirect && groupedDirect.length > 1) {
        const pick = this.quantChoice(id, groupedDirect);
        return {
          id, action: 'choose-quant', route, options, formats: info.formats,
          quants: pick.quants, recommended: pick.recommended, reason: pick.reason, budget: pick.budget,
        };
      }
      const job = await this.download(id, chosenDirect || undefined);
      return { id, action: 'download', route, options, quant: job.quant, formats: info.formats };
    }
    if (route.route === 'convert') {
      return { id, action: 'convert', route, options, converterReady: this.converterStatus().ready, formats: info.formats };
    }
    if (route.route === 'python') {
      return { id, action: 'python', route, options, runnerReady: this.runtimeStatus().ready, formats: info.formats };
    }
    if (route.route === 'browser') {
      return { id, action: 'browser', route, options, formats: info.formats };
    }
    if (route.route === 'cloud') {
      if (this.hfToken) {
        this.remote.add(id);
        await this._persistRemote();
        return { id, action: 'remote', route, options, formats: info.formats };
      }
      return { id, action: 'needs-token', route, options, formats: info.formats };
    }
    return { id, action: 'unsupported', route, options, formats: info.formats };
  }

  /** 下载仓库全部文件到指定目录（快照），返回文件清单。 */
  /** HF 下载鉴权头（门控仓库需要 Token；无 Token 时为空）。 */
  _hfAuthHeaders() {
    return this.hfToken ? { authorization: `Bearer ${this.hfToken}` } : {};
  }

  async _downloadSnapshotTo(model, dir) {
    const files = await fetchRepoFiles(model.repo, { fetchImpl: this.fetchImpl, headers: this._hfAuthHeaders() });
    await mkdir(dir, { recursive: true });
    const saved = [];
    for (const file of files) {
      if (/^(\.gitattributes|README)/i.test(file.path)) continue;
      // 只要是 GGUF 就别算进“原格式快照”：超大模型仓库常常同时放着 GGUF 和其它格式，
      // 那些文件动辄几十 GB，下载了 Python 路线也用不上（GGUF 直接走 llama.cpp）。
      if (/\.gguf$/i.test(file.path)) continue;
      const dest = path.join(dir, file.path);
      await mkdir(path.dirname(dest), { recursive: true });
      // expectedSize 让已下完的文件直接跳过：快照中断后重试不从头来
      await downloadFile({ url: resolveFileUrl(model.repo, file.path), dest, fetchImpl: this.fetchImpl, headers: this._hfAuthHeaders(), expectedSize: file.size });
      saved.push({ path: file.path, size: file.size });
    }
    return saved;
  }

  /** 确保原格式快照已下载到 models/<id>/SNAPSHOT 并写好清单，返回快照目录。 */
  async _ensureSnapshot(id) {
    const model = this.resolveModel(id);
    const downloaded = await this.downloadedQuants(id);
    const dir = this.modelDir(id, 'SNAPSHOT');
    if (downloaded.some((d) => d.quant === 'SNAPSHOT')) return dir;
    const saved = await this._downloadSnapshotTo(model, dir);
    const totalBytes = saved.reduce((n, f) => n + (f.size || 0), 0);
    await writeFile(
      path.join(dir, 'manifest.json'),
      `${JSON.stringify({ id, quant: 'SNAPSHOT', repo: model.repo, files: saved, totalBytes, downloadedAt: new Date().toISOString() }, null, 2)}\n`,
    );
    return dir;
  }

  /** 一键准备本地 Python 运行时（异步任务）。 */
  async prepareRunnerTool() {
    if (this.runnerJob?.state === 'running') return this.runnerJob;
    const job = { state: 'running', error: '' };
    this.runnerJob = job;
    (async () => {
      try {
        await preparePythonRuntime({ dataDir: this.dataDir, env: this.env, exec: this.execImpl || undefined });
        this._runtimeCache = null;
        job.state = 'done';
      } catch (err) {
        job.state = 'error';
        job.error = err.message || '本地运行时准备失败';
      }
    })();
    return job;
  }

  /**
   * 原格式本地运行（异步任务）：下载快照 → 启动 runner.py → 就绪。
   * 已在运行同一模型时直接返回。
   */
  async loadPython(id) {
    await this._preferLocal(id);
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    if (this.python.info?.modelId === id && this.python.state === 'ready') {
      return this.pythonJobs.get(id) || { id, state: 'done', stage: '运行中', error: '' };
    }
    const existing = this.pythonJobs.get(id);
    if (existing?.state === 'running') return existing;
    const runtime = this.runtimeStatus();
    if (!runtime.ready) {
      throw Object.assign(new Error(`${runtime.detail}（准备好后会自动继续）`), { status: 503 });
    }
    const job = { id, state: 'running', stage: '下载模型文件', error: '' };
    job.promise = (async () => {
      try {
        const snapshotDir = await this._ensureSnapshot(id);
        job.stage = '正在加载模型（首次较久）';
        const port = 19100 + Math.floor(Math.random() * 800);
        await this.python.start({ python: runtime.python, modelDir: snapshotDir, modelId: id, port, script: runtime.script });
        job.state = 'done';
        job.stage = '运行中';
        return job;
      } catch (err) {
        job.state = 'error';
        job.error = err.message || '本地运行启动失败';
        this.logger.warn?.(`[native] Python 运行 ${id} 失败：${job.error}`);
        throw err;
      }
    })();
    job.promise.catch(() => {});
    this.pythonJobs.set(id, job);
    return job;
  }

  /** 聊天前确保 Python 模型就绪（未启动则启动并等）。 */
  async _ensurePythonLoaded(id) {
    if (this.python.info?.modelId === id && this.python.state === 'ready') return;
    const job = await this.loadPython(id);
    if (job.state === 'done') return;
    await job.promise;
  }

  /** 一键准备转换工具（异步任务，状态从 status().hf.converterJob 读）。 */
  async prepareConverterTool() {
    if (this.converterJob?.state === 'running') return this.converterJob;
    const job = { state: 'running', error: '' };
    this.converterJob = job;
    (async () => {
      try {
        await prepareConverter({ dataDir: this.dataDir, env: this.env, exec: this.execImpl || undefined });
        this._converterCache = null;
        job.state = 'done';
      } catch (err) {
        job.state = 'error';
        job.error = err.message || '转换工具准备失败';
      }
    })();
    return job;
  }

  /**
   * 把已登记的 Transformers 模型转成 GGUF（异步任务）：
   * 下载仓库快照 → convert_hf_to_gguf.py → 写 CONVERTED 量化清单。
   */
  async convert(id) {
    await this._preferLocal(id);
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    const existing = this.conversions.get(id);
    if (existing?.state === 'running') return existing;
    const conv = this.converterStatus();
    if (!conv.ready) {
      throw Object.assign(new Error(`转换工具尚未就绪：${conv.detail}请先点“一键准备转换工具”。`), { status: 503 });
    }
    const job = { id, state: 'running', error: '', stage: '下载模型文件' };
    this.conversions.set(id, job);
    (async () => {
      try {
        // 本地已有快照（HF 下载过的、或本机导入的）直接用，不碰网络；
        // 导入模型没有 repo 地址，必须走这条本地短路
        const snapshotDir = await this._ensureSnapshot(id);
        job.stage = '正在转换（大模型可能需要较久）';
        const outfile = path.join(this.modelDir(id, 'CONVERTED'), 'converted.gguf');
        await runConvert({
          dataDir: this.dataDir,
          snapshotDir,
          outfile,
          env: this.env,
          exec: this.execImpl || undefined,
        });
        const size = (await stat(outfile)).size;
        await writeFile(
          path.join(this.modelDir(id, 'CONVERTED'), 'manifest.json'),
          `${JSON.stringify({
            id,
            quant: 'CONVERTED',
            repo: model.repo,
            files: [{ path: 'converted.gguf', size }],
            totalBytes: size,
            downloadedAt: new Date().toISOString(),
            converted: true,
          }, null, 2)}\n`,
        );
        await rm(path.join(this.dataDir, 'convert', safeDirName(id)), { recursive: true, force: true }).catch(() => {});
        job.state = 'done';
        job.stage = '完成';
      } catch (err) {
        job.state = 'error';
        job.error = err.message || '转换失败';
        this.logger.warn?.(`[native] 转换 ${id} 失败：${job.error}`);
      }
    })();
    return job;
  }

  /** 管理员：下载某模型的某量化（断点续传；同一量化重复调用直接返回进行中的任务）。 */
  async download(id, quant) {
    await this._preferLocal(id);
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    const grouped = await this.availableQuants(id);
    const entry = grouped.find((g) => g.quant === String(quant || '').toUpperCase())
      || grouped.find((g) => g.quant === pickDefaultQuant(grouped));
    if (!entry) throw Object.assign(new Error(`仓库里没有量化：${quant}`), { status: 400 });
    const key = `${id}@${entry.quant}`;
    await this._dropMismatchedPartFiles(id, entry);
    this._expireStaleBrowserJobs();
    const existing = this.downloads.get(key);
    if (existing?.state === 'downloading' && existing.via === 'browser') {
      // 浏览器通道还在活跃传数据：不能同时再开一条直连（会互相写坏 .part）。
      // 注意不能把浏览器任务原样返回假装直连已开始——那会显示假“下载中”。
      throw Object.assign(new Error('该量化正在经浏览器通道下载中，等它完成或中断后再试；急用可先关掉正在下载的页面。'), { status: 409 });
    }
    if (existing?.state === 'downloading') return existing;
    const already = await this.downloadedQuants(id);
    if (already.some((d) => d.quant === entry.quant)) {
      const done = { id, quant: entry.quant, state: 'done', downloaded: entry.totalBytes, total: entry.totalBytes, error: '' };
      this.downloads.set(key, done);
      return done;
    }
    const free = await freeDiskBytes(this.modelsDir);
    // 续传场景下已落盘的字节不会再占新空间：可用空间只需覆盖“还没下完的部分”
    const remaining = await this._remainingBytes(id, entry);
    const needBytes = Math.max(0, entry.totalBytes - remaining);
    if (Number.isFinite(free) && free < needBytes * 1.05) {
      throw Object.assign(new Error(`磁盘空间不足：该量化还需下载约 ${(needBytes / 1e9).toFixed(1)} GB，可用约 ${(free / 1e9).toFixed(1)} GB。`), { status: 507 });
    }
    const job = {
      id, quant: entry.quant, state: 'downloading', downloaded: remaining, total: entry.totalBytes,
      error: '', startedAt: Date.now(), updatedAt: Date.now(),
    };
    this.downloads.set(key, job);
    const dir = this.modelDir(id, entry.quant);
    await mkdir(dir, { recursive: true });
    (async () => {
      const ac = new AbortController();
      try {
        // 大模型通常有几十个分片：3 路并行一起拉，比排队逐个下快得多；
        // 并发只按文件分工，单个文件内部仍是一条流顺序写，不会写乱。
        // 已下完的分片由 downloadFile 按大小直接跳过，失败重试不从头来。
        //
        // perFile 必须按“下标”精确赋值：以前在文件完成时写的是
        // Math.max(perFile[i], done.bytes)——而 done.bytes 是该文件的完整大小，
        // 这样一个分片下完就把进度多算了一份，进度条会先涨到 150% 再回落。
        const perFile = new Array(entry.files.length).fill(0);
        const bump = () => {
          job.downloaded = perFile.reduce((n, v) => n + v, 0);
          job.updatedAt = Date.now();
        };
        // 续传时先把已落盘的分片算进进度（重新打开面板也能看到真实起点，而不是 0%）
        for (let i = 0; i < entry.files.length; i += 1) {
          try {
            const st = await stat(path.join(dir, entry.files[i].path));
            if (entry.files[i].size && st.size === entry.files[i].size) perFile[i] = st.size;
            else perFile[i] = Math.min(st.size, entry.files[i].size || st.size);
          } catch { perFile[i] = 0; }
        }
        bump();
        let cursor = 0;
        let firstError = null;
        const worker = async () => {
          for (;;) {
            if (firstError) return;
            const i = cursor;
            cursor += 1;
            if (i >= entry.files.length) return;
            const file = entry.files[i];
            const dest = path.join(dir, file.path);
            await mkdir(path.dirname(dest), { recursive: true });
            try {
              const done = await downloadFile({
                url: resolveFileUrl(model.repo, file.path),
                dest,
                fetchImpl: this.fetchImpl,
                headers: this._hfAuthHeaders(),
                signal: ac.signal,
                expectedSize: file.size,
                onProgress: ({ downloaded }) => { perFile[i] = downloaded; bump(); },
              });
              perFile[i] = Number(done?.bytes) || file.size;
              bump();
            } catch (err) {
              if (!firstError) {
                firstError = err;
                ac.abort(); // 一片失败就停掉其余在飞的，别让注定要重试的下载白占带宽
              }
              return;
            }
          }
        };
        await Promise.all(Array.from({ length: Math.min(3, Math.max(1, entry.files.length)) }, () => worker()));
        if (firstError) throw firstError;
        await writeFile(
          path.join(dir, 'manifest.json'),
          `${JSON.stringify({
            id,
            quant: entry.quant,
            repo: model.repo,
            files: entry.files,
            totalBytes: entry.totalBytes,
            downloadedAt: new Date().toISOString(),
          }, null, 2)}\n`,
        );
        job.state = 'done';
      } catch (err) {
        job.state = 'error';
        job.error = err.message || '下载失败';
        this.logger.warn?.(`[native] 下载 ${key} 失败：${job.error}`);
      }
    })();
    return job;
  }

  /**
   * 浏览器通道：用浏览器已经取到的仓库文件树在本地分组（计算规则与服务端直连
   * 完全一致，只是取数的人换成浏览器——浏览器能走代理/系统网络，服务端直连
   * 在很多用户电脑上连不上 Hugging Face）。返回分组并写入量化缓存。
   */
  groupQuantsFromFiles(id, files) {
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    const clean = [];
    for (const f of Array.isArray(files) ? files.slice(0, 50000) : []) {
      if (!f || typeof f.path !== 'string') continue;
      clean.push({ path: f.path, size: Math.max(0, Number(f.size) || 0), sha256: typeof f.sha256 === 'string' ? f.sha256 : '' });
    }
    const grouped = groupFilesByQuant(this._safeListFiles(clean, model.repo));
    if (!grouped.length) {
      throw Object.assign(new Error(`仓库 ${model.repo} 里没有找到可用的 GGUF 量化文件。`), { status: 400 });
    }
    this.quantCache.set(model.repo, { at: Date.now(), grouped });
    return grouped;
  }

  /** 过滤掉会逃出模型目录的文件路径（浏览器传回的清单不可盲信）。 */
  _safeListFiles(files, repo) {
    const out = [];
    for (const f of files) {
      const p = String(f.path || '');
      if (!p || p.length > 512 || p.includes('\\') || p.includes('\0')) continue;
      if (p.startsWith('/') || /^[a-zA-Z]:/.test(p)) continue;
      const parts = p.split('/');
      if (parts.some((seg) => seg === '..' || seg === '')) continue;
      out.push({ ...f, path: p });
    }
    return out;
  }

  /** 已落盘的字节数（完整分片 + 各分片的 .part）：续传时用来算真实起点与剩余空间。 */
  async _remainingBytes(id, entry) {
    const dir = this.modelDir(id, entry.quant);
    let sum = 0;
    for (const file of entry.files) {
      const want = Number(file.size) || 0;
      let got = 0;
      try {
        const st = await stat(path.join(dir, file.path));
        if (want && st.size === want) { sum += st.size; continue; }
        got = st.size;
      } catch { /* 完整文件不在，看 .part */ }
      try {
        const st = await stat(path.join(dir, `${file.path}.part`));
        got = Math.max(got, st.size);
      } catch { /* 没有 .part */ }
      sum += want ? Math.min(got, want) : got;
    }
    return sum;
  }

  /**
   * 从磁盘重建“浏览器通道”下载任务：软件重启、页面刷新导致内存任务丢失时，
   * 只要该量化目录里还有 .part 或未完成的分片，就按磁盘现状恢复进度。
   * 找不到任何半成品则返回 null（那就该走 prepare 正常开始）。
   */
  async _rebuildBrowserJob(id, quant) {
    const model = this.resolveModel(id);
    if (!model) return null;
    const q = String(quant || '').toUpperCase();
    let manifest = null;
    try {
      manifest = JSON.parse(await readFile(path.join(this.modelDir(id, q), 'manifest.json'), 'utf8'));
    } catch { /* 没有清单就现列一次仓库文件树 */ }
    let files = Array.isArray(manifest?.files) && manifest.files.length
      ? manifest.files.map((f) => ({ path: f.path, size: Number(f.size) || 0 }))
      : null;
    if (!files) {
      try {
        const grouped = await this.availableQuants(id);
        files = this._pickQuantEntry(grouped, q)?.files || null;
      } catch { files = null; }
    }
    if (!Array.isArray(files) || !files.length) return null;
    const dir = this.modelDir(id, q);
    const received = new Map();
    let anyProgress = false;
    let total = 0;
    for (const file of files) {
      total += Number(file.size) || 0;
      let got = 0;
      try {
        const st = await stat(path.join(dir, file.path));
        if (!file.size || st.size === file.size) got = st.size;
      } catch { /* 没有正式文件 */ }
      if (!got) {
        try { got = (await stat(path.join(dir, `${file.path}.part`))).size; } catch { got = 0; }
      }
      if (got > 0) anyProgress = true;
      received.set(file.path, got);
    }
    if (!anyProgress) return null;
    return {
      id, quant: q, state: 'downloading', via: 'browser',
      downloaded: [...received.values()].reduce((n, v) => n + v, 0),
      total, error: '', files, entry: { quant: q, files, totalBytes: total },
      receivedByFile: received, startedAt: Date.now(), lastChunkAt: Date.now(), updatedAt: Date.now(),
    };
  }

  /**
   * 仓库里的文件换了大小（作者重新上传/重新量化）后，本地 .part 是按旧大小写的，
   * 续传会永远差一截或写超，最后卡在“文件下载不完整”。这里把“已落盘字节比新目标
   * 还大”的 .part 直接丢掉，让下一次下载从头写这一片（其余分片不受影响）。
   */
  async _dropMismatchedPartFiles(id, entry) {
    const dir = this.modelDir(id, entry.quant);
    for (const file of entry.files) {
      const size = Number(file.size) || 0;
      if (!size) continue;
      const part = path.join(dir, `${file.path}.part`);
      try {
        const st = await stat(part);
        if (st.size > size) await rm(part, { force: true });
      } catch { /* 没有 .part 就不用管 */ }
    }
  }

  /** 单个仓库相对路径是否安全（不逃出量化目录）。 */
  _isSafeRepoPath(p) {
    if (typeof p !== 'string' || !p || p.length > 512) return false;
    if (p.includes('\\') || p.includes('\0') || p.startsWith('/') || /^[a-zA-Z]:/.test(p)) return false;
    return p.split('/').every((seg) => seg !== '..' && seg !== '');
  }

  /**
   * 给界面用的量化清单：每档带上“多大 / 要多少内存 / 这台机器装不装得下”。
   *
   * 这是 HF 一键使用选量化的数据源。关键点是 **fits** 字段：它不是让用户自己去
   * 心算 7.4 GB 可用内存能塞下哪档，而是直接把结论摆出来（装得下 / 只能靠页面
   * 文件硬顶 / 装不下），并且推荐项是按本机真实内存算出来的，不是拍脑袋的偏好表。
   *
   * @returns {{quants:Array, recommended:string|null, reason:string, budget:object}}
   */
  quantChoice(id, grouped) {
    const round = (n) => Math.round(n * 10) / 10;
    const model = this.resolveModel(id);
    // 上下文按模型登记值算：KV 缓存估算对 48~64 层的 MoE 差别很大
    const contextTokens = Number(model?.context) || 8192;
    let budget = { totalGB: 0, freeGB: 0, commitLimitGB: 0, pagefileGB: null, commitFreeGB: 0, source: 'os-only' };
    try {
      const b = readMemoryBudget();
      budget = {
        totalGB: round((Number(b.totalBytes) || 0) / 1e9),
        freeGB: round((Number(b.freeBytes) || 0) / 1e9),
        commitLimitGB: round((Number(b.commitLimitBytes) || 0) / 1e9),
        // pagefileGB 保留 null 语义：null = 查不到（不能断言“没有兜底”），
        // 0 = 确认没配。以前这里 `|| 0` 会把两者糊成一个 0，
        // 导致配了页面文件但探测失败的机器被一律判成“装不下”。
        pagefileGB: b.pagefileBytes === null || b.pagefileBytes === undefined
          ? null
          : round(b.pagefileBytes / 1e9),
        commitFreeGB: round((Number(b.commitFreeBytes) || 0) / 1e9),
        source: b.source,
      };
    } catch { /* 受限环境读不到内存：budget 全 0，pickQuantForBudget 会退回纯偏好 */ }

    const pick = pickQuantForBudget(grouped, { ...budget, contextTokens });
    const byQuant = new Map((pick.candidates || []).map((c) => [c.quant, c]));
    const quants = grouped.map((g) => {
      const c = byQuant.get(g.quant);
      const neededGB = c ? c.neededGB : null;
      return {
        quant: g.quant,
        files: g.files.length,
        totalBytes: g.totalBytes,
        sizeGB: round((g.totalBytes || 0) / 1e9),
        ramGB: neededGB ?? estimateRamGB(g.totalBytes, contextTokens),
        // fits：free（现在就能跑）/ total（关点程序能跑）/ spill（只能靠硬盘顶，很慢）/ never（装不下）
        fits: c ? c.fits : 'unknown',
        recommended: g.quant === pick.quant,
        ...describeQuant(g.quant),
      };
    });
    // 让界面不用自己排序：装得下的排前面，同档内大的（质量高的）在前
    const rank = { free: 0, total: 1, spill: 2, never: 3, unknown: 4 };
    quants.sort((a, b) => (rank[a.fits] - rank[b.fits]) || (b.totalBytes - a.totalBytes));
    return { quants, recommended: pick.quant, reason: pick.reason, budget };
  }

  _pickQuantEntry(grouped, quant) {
    return grouped.find((g) => g.quant === String(quant || '').toUpperCase())
      || grouped.find((g) => g.quant === pickDefaultQuant(grouped));
  }

  /**
   * 浏览器通道下载-准备：返回每个文件的下载地址与已落盘偏移（断点续传）。
   * files 是浏览器取到的文件树（可省，省时走服务端直连列树，兜底用）。
   */
  async prepareBrowserDownload(id, quant, files) {
    await this._preferLocal(id);
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    const grouped = Array.isArray(files) && files.length
      ? this.groupQuantsFromFiles(id, files)
      : await this.availableQuants(id);
    const entry = this._pickQuantEntry(grouped, quant);
    if (!entry) throw Object.assign(new Error(`仓库里没有量化：${quant}`), { status: 400 });
    const key = `${id}@${entry.quant}`;
    await this._dropMismatchedPartFiles(id, entry);
    const downloaded = await this.downloadedQuants(id);
    if (downloaded.some((d) => d.quant === entry.quant)) {
      const done = { id, quant: entry.quant, state: 'done', downloaded: entry.totalBytes, total: entry.totalBytes, error: '' };
      this.downloads.set(key, done);
      return { id, quant: entry.quant, state: 'done', total: entry.totalBytes, files: [] };
    }
    let existing = this.downloads.get(key);
    if (existing?.state === 'downloading' && existing.via !== 'browser') {
      // 只有“真的还在传”才拒绝；进程重启/中断留下的旧任务不能把续传永久卡死，
      // 因为 .part 本身就是断点账本，下面会按磁盘上的实际字节重新对齐。
      const quietMs = Date.now() - Number(existing.updatedAt || existing.startedAt || 0);
      if (quietMs < 120000) {
        throw Object.assign(new Error('该量化已有软件直连下载在进行中，等它完成或失败后再试。'), { status: 409 });
      }
      this.downloads.delete(key);
      existing = null;
    }
    // 续传：已落盘的字节不会再占新空间，按“还需下载的部分”判断
    const alreadyOnDisk = await this._remainingBytes(id, entry);
    const needBytes = Math.max(0, entry.totalBytes - alreadyOnDisk);
    const free = await freeDiskBytes(this.modelsDir);
    if (Number.isFinite(free) && free < needBytes * 1.05) {
      throw Object.assign(new Error(`磁盘空间不足：该量化还需约 ${(needBytes / 1e9).toFixed(1)} GB，可用约 ${(free / 1e9).toFixed(1)} GB。`), { status: 507 });
    }
    const dir = this.modelDir(id, entry.quant);
    this._expireStaleBrowserJobs();
    const job = existing?.via === 'browser' && existing.state === 'downloading'
      ? existing
      : {
          id, quant: entry.quant, state: 'downloading', downloaded: alreadyOnDisk, total: entry.totalBytes,
          error: '', via: 'browser', entry, receivedByFile: new Map(),
          startedAt: Date.now(), lastChunkAt: Date.now(),
        };
    // 新一轮 prepare 说明浏览器还活着并在续传：刷新活跃时间，避免被陈旧清理误杀
    job.lastChunkAt = Date.now();
    job.startedAt = job.startedAt || job.lastChunkAt;
    job.entry = entry;
    if (!(job.receivedByFile instanceof Map)) job.receivedByFile = new Map();
    this.downloads.set(key, job);
    const outFiles = [];
    let sum = 0;
    for (const file of entry.files) {
      if (!this._isSafeRepoPath(file.path)) {
        throw Object.assign(new Error(`仓库文件名不安全，已拒绝：${file.path}`), { status: 400 });
      }
      const dest = path.join(dir, file.path);
      await mkdir(path.dirname(dest), { recursive: true });
      const part = `${dest}.part`;
      let offset = 0;
      try {
        const st = await stat(dest);
        if (st.size === file.size) offset = file.size;
      } catch { /* 正式文件不在，看 .part */ }
      if (offset < file.size) {
        try {
          const st = await stat(part);
          offset = st.size;
          if (offset > file.size) {
            // 半成品比目标还大：损坏，重下
            await rm(part, { force: true });
            offset = 0;
          } else if (offset === file.size) {
            offset = file.size;
          }
        } catch { offset = 0; }
      }
      job.receivedByFile.set(file.path, offset);
      sum += offset;
      outFiles.push({ path: file.path, size: file.size, url: resolveFileUrl(model.repo, file.path), offset, done: offset >= file.size });
    }
    job.downloaded = sum;
    return { id, quant: entry.quant, state: 'downloading', total: entry.totalBytes, files: outFiles };
  }

  /**
   * 浏览器通道下载-收一块：把浏览器传来的原始字节顺序追加到 .part。
   * expectedOffset 必须与服务端已落盘大小一致，防止错位写坏权重。
   */
  async receiveBrowserChunk(id, quant, filePath, expectedOffset, buffer) {
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    const key = `${id}@${String(quant || '').toUpperCase()}`;
    let job = this.downloads.get(key);
    if (!job?.entry || job.state !== 'downloading' || job.via !== 'browser') {
      // 软件重启/会话过期后内存里的任务没了，但磁盘上的 .part 还在：此时让用户
      // “重新点下载”会重新走 prepare（多半又因为旧任务报 409），最后只能删了重下。
      // 这里按磁盘上的真实文件清单重建任务，直接接着传。
      job = await this._rebuildBrowserJob(id, quant);
      if (job) this.downloads.set(key, job);
    }
    if (!job || job.state !== 'downloading' || job.via !== 'browser' || !job.entry) {
      throw Object.assign(new Error('没有进行中的浏览器通道下载，请重新点下载。'), { status: 409 });
    }
    if (!this._isSafeRepoPath(filePath) || !job.entry.files.some((f) => f.path === filePath)) {
      throw Object.assign(new Error(`文件不在本次下载清单里：${filePath}`), { status: 400 });
    }
    const part = path.join(this.modelDir(id, job.quant), `${filePath}.part`);
    await mkdir(path.dirname(part), { recursive: true });
    let current = 0;
    try { current = (await stat(part)).size; } catch { current = 0; }
    if (Number.isFinite(expectedOffset) && expectedOffset !== current) {
      const err = Object.assign(new Error(`写入位置对不上（软件已存 ${current} 字节，你以为 ${expectedOffset} 字节），请重新点下载自动续传。`), { status: 409 });
      err.received = current;
      throw err;
    }
    if (buffer?.length) await appendFile(part, buffer);
    const received = current + (buffer?.length || 0);
    job.receivedByFile.set(filePath, received);
    job.downloaded = [...job.receivedByFile.values()].reduce((n, v) => n + v, 0);
    job.lastChunkAt = Date.now();
    job.updatedAt = job.lastChunkAt;
    return { received, downloaded: job.downloaded, total: job.total };
  }

  /** 浏览器通道下载-收尾：核对每个文件大小，.part 转正并写清单。 */
  async finishBrowserDownload(id, quant) {
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    const key = `${id}@${String(quant || '').toUpperCase()}`;
    const job = this.downloads.get(key);
    if (!job?.entry) throw Object.assign(new Error('没有这次下载的清单，请重新点下载。'), { status: 409 });
    const dir = this.modelDir(id, job.quant);
    for (const file of job.entry.files) {
      const dest = path.join(dir, file.path);
      const part = `${dest}.part`;
      let ok = false;
      try { ok = (await stat(dest)).size === file.size; } catch { ok = false; }
      if (!ok) {
        let partSize = -1;
        try { partSize = (await stat(part)).size; } catch { partSize = -1; }
        if (partSize === file.size) {
          await mkdir(path.dirname(dest), { recursive: true });
          await rename(part, dest);
          ok = true;
        }
      }
      if (!ok) {
        job.state = 'error';
        job.error = `文件 ${file.path} 还没传完整，重新点下载会自动续传。`;
        throw Object.assign(new Error(job.error), { status: 409 });
      }
    }
    await writeFile(
      path.join(dir, 'manifest.json'),
      `${JSON.stringify({
        id,
        quant: job.quant,
        repo: model.repo,
        files: job.entry.files,
        totalBytes: job.entry.totalBytes,
        downloadedAt: new Date().toISOString(),
      }, null, 2)}\n`,
    );
    job.state = 'done';
    job.downloaded = job.total;
    job.error = '';
    return { id, quant: job.quant, state: 'done' };
  }

  // ---------- 本地导入：浏览器先把文件下到电脑，再导入软件（最稳的一条路） ----------
  // 浏览器自带下载器下大文件最稳（断点续传、重试都由浏览器管），下完把文件
  // 交给软件：暂存区收文件（按偏移续传），收齐后自动识别类型落地登记——
  // GGUF 按量化分档落地、其它权重按快照落地、zip 先解压再识别，用户不用懂格式。

  _importDir(importId) {
    return path.join(this.dataDir, 'import-staging', safeDirName(importId));
  }

  _importSession(importId) {
    const session = this.importSessions?.get(String(importId || ''));
    if (!session) throw Object.assign(new Error('导入会话不存在或已结束，请重新选择文件导入。'), { status: 404 });
    return session;
  }

  /** 开始一次导入：登记文件清单并建暂存区；同一组文件重复开始会接着上次的进度。 */
  async beginImport(importId, files) {
    const id = String(importId || '');
    if (!/^[A-Za-z0-9-]{8,64}$/.test(id)) throw Object.assign(new Error('导入编号不合法。'), { status: 400 });
    // 顺手清掉一天前的残留会话，别让暂存区越堆越大
    for (const [key, s] of this.importSessions) {
      if (Date.now() - s.at > 24 * 3600 * 1000) {
        this.importSessions.delete(key);
        await rm(this._importDir(key), { recursive: true, force: true }).catch(() => {});
      }
    }
    const clean = this._safeListFiles(Array.isArray(files) ? files.slice(0, 2000) : []);
    if (!clean.length) throw Object.assign(new Error('没有收到要导入的文件清单。'), { status: 400 });
    const dir = this._importDir(id);
    await mkdir(dir, { recursive: true });
    const session = { id, at: Date.now(), declared: new Map(clean.map((f) => [f.path, f.size])) };
    this.importSessions.set(id, session);
    return this.importSessionStatus(id);
  }

  /** 导入进度：每个文件已收到的字节（.part 的大小），供断点续传对齐。 */
  async importSessionStatus(importId) {
    const session = this._importSession(importId);
    const dir = this._importDir(session.id);
    const out = [];
    for (const [rel, size] of session.declared) {
      let received = 0;
      try { received = (await stat(path.join(dir, `${rel}.part`))).size; } catch { received = 0; }
      out.push({ path: rel, size, received });
    }
    return { importId: session.id, files: out };
  }

  /** 收一片导入数据：偏移必须与已落盘大小一致（409 带真实偏移），防错位写坏文件。 */
  async writeImportSlice(importId, relPath, expectedOffset, buffer) {
    const session = this._importSession(importId);
    if (!this._isSafeRepoPath(relPath) || !session.declared.has(relPath)) {
      throw Object.assign(new Error(`文件不在本次导入清单里：${relPath}`), { status: 400 });
    }
    const part = path.join(this._importDir(session.id), `${relPath}.part`);
    await mkdir(path.dirname(part), { recursive: true });
    let current = 0;
    try { current = (await stat(part)).size; } catch { current = 0; }
    if (Number.isFinite(expectedOffset) && expectedOffset !== current) {
      const err = Object.assign(new Error(`写入位置对不上（软件已存 ${current} 字节），已自动按真实位置续传。`), { status: 409 });
      err.received = current;
      throw err;
    }
    if (buffer?.length) await appendFile(part, buffer);
    session.at = Date.now();
    return { received: current + (buffer?.length || 0) };
  }

  /** 取消导入：清掉暂存文件。 */
  async cancelImport(importId) {
    const id = String(importId || '');
    this.importSessions?.delete(id);
    await rm(this._importDir(id), { recursive: true, force: true });
    return { ok: true };
  }

  /**
   * 收尾导入：核对完整性 → zip 先解压 → 按内容自动分类落地并登记。
   * - GGUF：按量化分档（一档一目录 + 清单），下载后可直接加载；
   * - 其它权重（Safetensors/PyTorch/ONNX…）：整包落为快照，走转换/Python 路线；
   * - 混合仓库两种都落，两条路线都在。
   * 全部识别不了才报错，不会把垃圾文件登记成模型。
   */
  async finishImport(importId, { name } = {}) {
    const session = this._importSession(importId);
    const dir = this._importDir(session.id);
    // 完整性核对 + .part 转正（清单外多出来的文件一并收编，如 zip 解压产物会再走一轮收集）
    const landed = [];
    for (const [rel, size] of session.declared) {
      const part = path.join(dir, `${rel}.part`);
      let got = -1;
      try { got = (await stat(part)).size; } catch { got = -1; }
      if (size === 0 && got !== 0) {
        // 零字节文件客户端不会真的发片，直接在暂存区落一个空文件
        await mkdir(path.dirname(part), { recursive: true });
        await writeFile(part, '');
        got = 0;
      }
      if (got !== size) {
        throw Object.assign(new Error(`文件 ${rel} 还没传完整（已收 ${Math.max(0, got)} / ${size} 字节），重新导入同一组文件会接着传。`), { status: 409 });
      }
      await rename(part, path.join(dir, rel));
      landed.push({ path: rel, size });
    }
    // zip 先解压再识别（浏览器下的是整包时最常见）：解到暂存根，zip 本身不进模型目录
    const files = [];
    for (const f of landed) {
      if (/\.zip$/i.test(f.path)) {
        if (f.size > 2 * 1024 * 1024 * 1024) {
          throw Object.assign(new Error(`压缩包 ${f.path} 超过 2GB，请先在电脑上解压，再用“导入文件夹”导入。`), { status: 413 });
        }
        const buf = await readFile(path.join(dir, f.path));
        await extractZip(buf, dir);
        await rm(path.join(dir, f.path), { force: true });
      } else {
        files.push(f);
      }
    }
    if (landed.some((f) => /\.zip$/i.test(f.path))) {
      // 解压产物重新收集（跳过苹果系统的 __MACOSX 之类杂物）
      const walk = async (base, prefix = '') => {
        for (const entry of await readdir(base, { withFileTypes: true })) {
          const rel = prefix ? `${prefix}/${entry.name}` : entry.name;
          if (rel.startsWith('__MACOSX/') || entry.name === '.DS_Store') continue;
          if (entry.isDirectory()) await walk(path.join(base, entry.name), rel);
          else if (entry.isFile()) files.push({ path: rel, size: (await stat(path.join(base, entry.name))).size });
        }
      };
      const have = new Set(files.map((f) => f.path));
      const before = files.length;
      await walk(dir);
      for (let i = files.length - 1; i >= before; i -= 1) {
        if (have.has(files[i].path)) files.splice(i, 1);
      }
    }

    const cleanFiles = this._safeListFiles(files);
    const ggufFiles = cleanFiles.filter((f) => /\.gguf$/i.test(f.path) && !/mmproj/i.test(f.path));
    const mmprojFiles = cleanFiles.filter((f) => /\.gguf$/i.test(f.path) && /mmproj/i.test(f.path));
    const formats = detectFormats(cleanFiles, {});
    // 导入的 id/名字：优先用用户填写，其次从文件名推导（去掉量化与分片后缀）
    const stemOf = (p) => path.basename(p).replace(/\.gguf$/i, '').replace(/-\d{1,5}-of-\d{1,5}$/i, '').replace(/[-.](UD-[A-Z0-9_]+|IQ\d[A-Z0-9_]*|Q\d[A-Z0-9_]*|BF16|F16|F32)$/i, '');
    const guessName = String(name || '').trim()
      || (ggufFiles.length ? stemOf(ggufFiles.map((f) => f.path).sort()[0]) : '')
      || (cleanFiles.find((f) => f.path.includes('/'))?.path.split('/')[0] || '')
      || (cleanFiles[0] ? path.basename(cleanFiles[0].path).replace(/\.[^.]+$/, '') : '')
      || 'imported-model';
    const slug = guessName.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'imported-model';
    let newId = `local/${slug}`;
    for (let n = 2; this.custom.has(newId) || nativeModelById(newId); n += 1) newId = `local/${slug}-${n}`;
    const displayName = String(name || '').trim() || guessName;

    const placedQuants = [];
    let snapshot = false;
    const moveInto = async (targetDir, list) => {
      await mkdir(targetDir, { recursive: true });
      for (const f of list) {
        const dest = path.join(targetDir, f.path);
        await mkdir(path.dirname(dest), { recursive: true });
        await rename(path.join(dir, f.path), dest);
      }
    };
    if (ggufFiles.length) {
      // 按量化分组；文件名认不出量化标记的按文件名归档，保证一个不落
      const groups = groupFilesByQuant(ggufFiles);
      const groupedPaths = new Set(groups.flatMap((g) => g.files.map((f) => f.path)));
      for (const f of ggufFiles) {
        if (groupedPaths.has(f.path)) continue;
        const stem = path.basename(f.path).replace(/\.gguf$/i, '').toUpperCase().replace(/[^A-Z0-9]+/g, '_').slice(0, 40) || 'GGUF';
        groups.push({ quant: stem, files: [f], totalBytes: f.size });
      }
      for (const g of groups) {
        const qdir = this.modelDir(newId, g.quant);
        await moveInto(qdir, g.files);
        await writeFile(
          path.join(qdir, 'manifest.json'),
          `${JSON.stringify({ id: newId, quant: g.quant, repo: '', source: 'import', files: g.files, totalBytes: g.totalBytes, downloadedAt: new Date().toISOString() }, null, 2)}\n`,
        );
        placedQuants.push(g.quant);
      }
      // 视觉投影 mmproj 不进清单（本软件只加载语言部分），但文件跟第一档走，别丢用户下好的东西
      if (mmprojFiles.length && groups.length) {
        await moveInto(this.modelDir(newId, groups[0].quant), mmprojFiles);
      }
    }
    const restFiles = cleanFiles.filter((f) => !ggufFiles.includes(f) && !mmprojFiles.includes(f) && !/^(README|\.gitattributes)/i.test(path.basename(f.path)));
    const weightFiles = restFiles.filter((f) => /\.(safetensors|bin|onnx|pt|pth|npz|mlmodel)$/i.test(f.path) || /\.mlpackage\//i.test(f.path));
    if (!ggufFiles.length || weightFiles.length) {
      // 没有 GGUF 时整包做快照；有 GGUF 又混了其它权重时也把其余落成快照（两条路线都在）
      const snapFiles = ggufFiles.length ? restFiles : cleanFiles.filter((f) => !/^(README|\.gitattributes)/i.test(path.basename(f.path)));
      // 必须真有非 GGUF 权重才落快照：只有 config/tokenizer 的目录不能登记成模型
      if (snapFiles.length && weightFiles.length) {
        const sdir = this.modelDir(newId, 'SNAPSHOT');
        await moveInto(sdir, snapFiles);
        const totalBytes = snapFiles.reduce((n, f) => n + f.size, 0);
        await writeFile(
          path.join(sdir, 'manifest.json'),
          `${JSON.stringify({ id: newId, quant: 'SNAPSHOT', repo: '', source: 'import', files: snapFiles, totalBytes, downloadedAt: new Date().toISOString() }, null, 2)}\n`,
        );
        snapshot = true;
      }
    }
    if (!placedQuants.length && !snapshot) {
      throw Object.assign(new Error('在这些文件里没有找到可识别的模型权重（需要 .gguf、.safetensors、.onnx 等权重文件；只有说明文档/配置文件没法当模型用）。'), { status: 400 });
    }
    const finalFormats = [...new Set([...(placedQuants.length ? ['gguf'] : []), ...formats])];
    this.custom.set(newId, {
      id: newId, repo: '', name: displayName, formats: finalFormats,
      context: 32768, source: 'import', addedAt: new Date().toISOString(),
    });
    await this._persistCustom();
    this.importSessions.delete(session.id);
    await rm(dir, { recursive: true, force: true });
    return { id: newId, name: displayName, kind: 'import', formats: finalFormats, quants: placedQuants, snapshot };
  }

  /** 浏览器通道下载-失败标记：保留 .part，下次点下载按偏移续传。 */
  failBrowserDownload(id, quant, message) {
    const key = `${id}@${String(quant || '').toUpperCase()}`;
    const job = this.downloads.get(key);
    if (job && job.state === 'downloading') {
      job.state = 'error';
      job.error = String(message || '浏览器通道下载中断').slice(0, 300);
    }
    return { ok: true };
  }

  /** 管理员：删除已下载权重（不删正在运行的）。整模型删除时同时注销自定义/云端登记。 */
  async deleteModel(id, quant) {
    if ((this.llama.info?.modelId === id && this.llama.state !== 'idle')
      || (this.python.info?.modelId === id && this.python.state !== 'idle')) {
      throw Object.assign(new Error('该模型正在运行，请先卸载再删除权重。'), { status: 409 });
    }
    // 正在下载的量化不许删：后台任务会继续往已被删除的目录里写，最后留下半截
    // 文件与一条“下载失败”的怪错；浏览器通道任务更是会一直往空气里传字节。
    const busyKeys = [...this.downloads.entries()]
      .filter(([key, job]) => job?.state === 'downloading'
        && (quant ? key === `${id}@${quant}` : key.startsWith(`${id}@`)))
      .map(([key]) => key);
    if (busyKeys.length) {
      throw Object.assign(
        new Error('该量化正在下载中，请等下载完成或中断后再删除（中断后重新点下载会续传）。'),
        { status: 409 },
      );
    }
    if (quant) {
      await rm(this.modelDir(id, quant), { recursive: true, force: true });
      this.downloads.delete(`${id}@${quant}`);
    } else {
      await rm(this.modelDir(id), { recursive: true, force: true });
      for (const key of [...this.downloads.keys()]) {
        if (key.startsWith(`${id}@`)) this.downloads.delete(key);
      }
      if (this.custom.has(id)) {
        this.custom.delete(id);
        await this._persistCustom();
      }
      if (this.remote.delete(id)) await this._persistRemote();
      this.conversions.delete(id);
    }
    return { deleted: true };
  }

  /**
   * 记住“管理员上次显式加载时用的上下文/GPU 层数”。
   * 聊天自动加载（用户没点“加载”，直接发消息）以前一律按 8192/0 起，
   * 面板里设成 32768 或卸载 40 层完全不起作用，用户以为设置没生效。
   */
  _rememberLoadPrefs(id, { contextTokens, gpuLayers }) {
    this.loadPrefs = {
      contextTokens: Math.max(1024, Math.floor(contextTokens) || 8192),
      gpuLayers: Math.max(0, Math.floor(gpuLayers) || 0),
    };
    this.loadPrefsFor = id;
  }

  /**
   * 加载入口：并发请求合并、其余排队。
   *
   * llama-server 是单实例，而“点加载”与聊天自动加载同时到达很常见：以前它们
   * 会各自 start() 一次——后一次把前一次的进程停掉，前一个请求收到
   * “llama-server 已被停止或替换”，磁盘上还要把几十 GB 权重读两遍。
   * - 完全相同参数的加载：直接共享同一个 promise；
   * - 不同参数/不同模型：按顺序排队执行（后一个总会停掉前一个的进程）。
   *
   * 实际干活的是 _load（管理员/聊天自动共用：加载已下载的量化，必要时先卸载当前模型）。
   */
  load(id, opts = {}) {
    const key = `${id}|${opts.quant || ''}|${opts.contextTokens ?? ''}|${opts.gpuLayers ?? ''}|${opts.safe ? 'safe' : ''}`;
    if (!this._loadPendingByKey) this._loadPendingByKey = new Map();
    const running = this._loadPendingByKey.get(key);
    if (running) return running;
    const run = async () => {
      try {
        return await this._load(id, opts);
      } finally {
        this._loadPendingByKey.delete(key);
      }
    };
    const prev = this._loadChain || Promise.resolve();
    const promise = prev.catch(() => {}).then(run);
    this._loadChain = promise.catch(() => {});
    this._loadPendingByKey.set(key, promise);
    return promise;
  }

  async _load(id, { quant, contextTokens, gpuLayers, explicit = true, safe = false } = {}) {
    // 未显式给值时沿用“同一个模型上一次加载”的参数（聊天自动加载走这条）。
    // 只认同一个模型：把 70B 的 32K 上下文/40 层套到 8B 小模型上只会白白触发回退重试。
    const prefs = (this.loadPrefsFor === id && this.loadPrefs)
      ? this.loadPrefs
      : { contextTokens: 8192, gpuLayers: 0 };
    // safe：崩溃后的“救命加载”，不读也不写用户偏好，避免把安全值 4096/0 变成默认
    if (!safe) {
      if (!Number.isFinite(Number(contextTokens))) contextTokens = prefs.contextTokens;
      if (!Number.isFinite(Number(gpuLayers))) gpuLayers = prefs.gpuLayers;
      if (explicit) this._rememberLoadPrefs(id, { contextTokens, gpuLayers });
    }
    // 每次加载都重算说明：上一次加载的回退提示不能残留到这一次（否则界面上会
    // 一直挂着一条与当前模型无关的“已自动回退”）。
    this.loadNote = null;
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    if (this.llama.info?.modelId === id && this.llama.state === 'ready') {
      // 同一模型、同一档、参数也没变：直接返回（重复点“加载”不折腾）
      const sameQuant = !quant || String(this.llama.info.quant || '').toUpperCase() === String(quant).toUpperCase();
      const sameCtx = Number(this.llama.info.contextTokens || 0) === Math.max(1024, Math.floor(contextTokens) || 8192);
      const sameLayers = Number(this.llama.info.gpuLayers ?? -1) === Math.max(0, Math.floor(gpuLayers) || 0);
      if (sameQuant && sameCtx && sameLayers) return this.llama.status();
      // 换了量化/上下文/层数：必须重新加载，否则界面上选了 Q2_K、实际还在跑 Q4_K_M
      await this.llama.stop().catch(() => {});
    }
    const downloaded = await this.downloadedQuants(id);
    // 只能加载 GGUF（SNAPSHOT 是原格式快照、CONVERTED 是本地转换产物）。
    // 以前回落到 downloaded[0]：只下过快照的模型会被当成 GGUF 去加载，报一堆
    // “不是 GGUF”的怪错；现在明确告诉用户该走转换/原格式运行。
    const loadable = downloaded.filter((d) => d.quant !== 'SNAPSHOT');
    const wanted = String(quant || '').toUpperCase();
    // 指名要哪一档就必须加载哪一档：以前找不到会静默换成另一档已下载的量化，
    // 用户以为跑的是 Q2_K、其实跑的是 Q4_K_M（显存/速度全对不上）。
    if (wanted && loadable.length && !loadable.some((d) => d.quant === wanted)) {
      throw Object.assign(
        new Error(`量化 ${wanted} 还没有下载（已下载：${loadable.map((d) => d.quant).join('、')}）。请先下载这一档，或在“量化”下拉里改选已下载的档位。`),
        { status: 409 },
      );
    }
    const pick = loadable.find((d) => d.quant === wanted) || loadable[0];
    if (!pick) {
      // 关键：下载中断留下 .part / 被截断的分片，以前在这里就被判成“尚未下载”——
      // 用户明明看到几十 GB 已经躺在硬盘上，系统却说没下载，只能删了重下。
      // 现在先看磁盘上有没有量化目录，有就用“不完整”的口吻给出续传指引。
      const broken = await this._brokenQuants(id);
      if (broken.length) {
        const main = broken.find((b) => b.quant === wanted) || broken[0];
        throw Object.assign(
          new Error(
            `量化 ${main.quant} 还没下载完（${main.summary}）。\n`
            + '这不是内存不够，是文件不完整。请回到“本机大模型”点一次下载，软件会按已下载的字节续传补齐（不会从头开始），再点加载。',
          ),
          { status: 409, corrupt: true },
        );
      }
      const why = downloaded.length
        ? `这个模型目前只下载了原格式权重（SNAPSHOT），请点“转换并使用”转成 GGUF，或用“本地运行（Python）”直接运行原格式。`
        : `模型 ${model.name} 尚未下载。请在“API 接入 → 本机大模型”中先下载权重。`;
      throw Object.assign(new Error(why), { status: 409 });
    }
    const binary = findLlamaBinary({ env: this.env, dataDir: this.dataDir });
    if (!binary) {
      throw Object.assign(
        new Error('llama.cpp 运行库尚未准备。请管理员在“API 接入 → 本机大模型”中点“准备运行库”。'),
        { status: 503 },
      );
    }
    // 架构预检：这个权重需要更新的 llama.cpp 时，在启动进程之前就说清楚。
    // 否则 llama-server 会先跑起来、读完整份元数据、再报 "unknown model architecture"，
    // 用户等半天只看到一个被包装成“内存不足”的失败。
    const wantArch = await this._modelArch(id, pick.quant);
    if (wantArch) {
      const health = await this.checkBinaryHealth({ forArch: wantArch });
      this.binaryHealth = health;
      if (health.state === 'tooOld') {
        throw Object.assign(new Error(health.message + (health.hint ? `\n${health.hint}` : '')), {
          status: 409,
          loadFailed: true,
          diag: 'architecture',
          arch: wantArch,
          minBuild: health.minBuild,
          currentBuild: health.build,
        });
      }
    }
    const manifest = await this.readManifest(id, pick.quant);
    // 分片模型的入口文件必须是第 1 个分片（`...-00001-of-00005.gguf`）。
    // 以前按字典序取第一个：`-00002-of-` 排在 `1-of-` 前面，或者按量化分子目录
    // 布局时排到别的文件上，llama.cpp 会因为拿到中段分片而加载失败——这正是
    // 超大模型（几十个分片）最常见的“下载都正常、一加载就崩”原因。
    const entryFile = pickEntryFile(manifest.files);
    if (!entryFile) {
      throw Object.assign(new Error(`模型 ${model.name} 的权重清单是空的，请删除后重新下载。`), { status: 409 });
    }
    const modelFile = path.join(this.modelDir(id, pick.quant), entryFile.path);
    if (!existsSync(modelFile)) throw new Error(`权重文件缺失：${entryFile.path}`);
    // 分片必须齐全：缺片的模型让 llama.cpp 去加载只会得到一句含糊的崩溃/读取错误
    const missing = manifest.files.filter((f) => !existsSync(path.join(this.modelDir(id, pick.quant), f.path)));
    if (missing.length) {
      throw Object.assign(
        new Error(`权重文件不完整：缺少 ${missing.length} 个分片（如 ${missing[0].path}）。请重新点下载补齐（会从断点续传，不会从头下）。`),
        { status: 409 },
      );
    }
    // 开跑前先自己做一次“权重完整性体检”：分片齐全、每个分片大小与清单一致、
    // 入口分片的 GGUF 头能解析（架构/层数读得到）。
    // 以前所有这些只在 llama.cpp 启动失败后由一句含糊的“代码 3221225477”反推，
    // 用户根本分不清是内存不够还是权重缺片/截断——而这正是 0xC0000005 的两种
    // 最常见来源。体检失败直接给出“哪个文件坏、怎么修”，不再走十几分钟的阶梯。
    const verdict = await this._verifyWeights(id, pick.quant, manifest, entryFile);
    if (!verdict.ok) {
      throw Object.assign(new Error(verdict.message), { status: verdict.status || 500, corrupt: !!verdict.corrupt });
    }
    // 内存体检（只警告、不拦截）：0xC0000005 报告里最多的其实是“权重本身没坏，
    // 但这台机器内存装不下 / 剩余内存不够”。以前系统只丢一个崩溃码给用户，
    // 现在把“需要多少、现在有多少”直接写进日志与状态面板，用户一眼就知道是换量化还是关程序。
    const reportCtx = Math.min(Math.max(1024, Math.floor(contextTokens) || 8192), Number(model.context) > 0 ? Number(model.context) : 8192);
    // 用体检读到的真实层数算 KV 缓存：以前完全不看层数，48 层 MoE 的 KV 被低估 6~8 倍，
    // 于是给出“16.6 GB 总内存 < 15.5 GB 需求”却自相矛盾的结论。
    const ram = this._memoryReport({ totalBytes: manifest.totalBytes, blockCount: verdict.blockCount }, gpuLayers, reportCtx);
    this.memoryReport = ram;
    if (ram.tight) this.logger.warn?.(`[native] 内存可能不够：${ram.summary}`);
    const wantCtx = Math.max(1024, Math.floor(contextTokens) || 8192);
    const maxCtx = Number(model.context) > 0 ? Number(model.context) : 8192;
    const ctx = Math.min(wantCtx, maxCtx);
    const contextNote = wantCtx > maxCtx
      ? { requested: wantCtx, used: ctx, reason: `该模型最大上下文 ${maxCtx}，已按上限设置` }
      : null;
    const port = 18100 + Math.floor(Math.random() * 800);
    const requested = Math.max(0, Math.floor(gpuLayers) || 0);
    const planned = this._buildLoadAttempts({ requested, ctx, safe });
    // 这台机器上“已经试过且失败”的组合要记住（同一模型同一档位内有效）：
    // 聊天路径的救命重载若原样再跑一遍刚失败的配置，用户只会多等几分钟。
    // 成功一次就清空记录——之前失败可能只是因为别的程序占着内存。
    const failKey = `${id}@${pick.quant}`;
    if (!this._failedAttempts) this._failedAttempts = new Map();
    const failed = safe ? (this._failedAttempts.get(failKey) || new Set()) : null;
    const attempts = safe && failed ? planned.filter((a) => !failed.has(attemptKey(a))) : planned;
    const extraArgs = String(this.env.OJ_LLAMA_EXTRA_ARGS || '').trim().split(/\s+/).filter(Boolean);
    let lastErr = null;
    if (safe && !attempts.length) {
      const diag = this._diagnoseFailure(lastErr);
      const mem = this.memoryReport;
      const err = Object.assign(
        new Error(
          '这组权重在本机把能试的启动配置都试过了（纯 CPU、关 flash-attention、关 KV 缓存复用、不用 mmap、小上下文），仍然起不来。\n'
          + `判断：${diag.title}。${diag.advice}`
          // 只有真判定为内存问题、且内存确实不够时才摆数字；够用就直说够用，
          // 免得出现“需求 2.3 GB、可用 3.8 GB，内存够用”却仍提示换量化的怪结论。
          + (diag.kind === 'memory' && mem?.summary ? `\n内存情况：${mem.summary}` : '')
          // 日志说内存不够、但估算说够：这在显存/提交限制/碎片化时很常见，必须点明，
          // 否则用户只会觉得“软件自己都前后矛盾”。
          + (diag.kind === 'memory' && mem && mem.tight === false
            ? '\n说明：上面的内存估算是按权重体积粗算的；实际可能是显存不够、系统提交内存上限（页面文件）不够，'
              + '或其它程序正占着大块连续内存。仍然建议先换更小的量化，或把页面文件调大后重试。'
            : ''),
        ),
        { status: 502, loadFailed: true, diag: diag.kind, rawMessage: String(lastErr?.message || '') },
      );
      throw err;
    }
    for (const [i, attempt] of attempts.entries()) {
      try {
        await this.llama.start({
          binary: binary.path,
          modelFile,
          modelId: id,
          quant: pick.quant,
          contextTokens: attempt.contextTokens,
          gpuLayers: attempt.layers,
          port,
          // 高级参数（可选）：MoE 模型常需要 --n-cpu-moe / --override-tensor 之类把专家张量
          // 留在内存，否则 -ngl 拉满必然显存不足。不写死任何值，由管理员通过
          // OJ_LLAMA_EXTRA_ARGS 显式指定，避免把大模型推到更慢的配置上。
          extraArgs,
          // 参数级逃生开关（见 _buildLoadAttempts）：崩溃/内存不足时逐档收敛到最保险的组合
          flashAttn: attempt.flashAttn,
          cacheReuse: attempt.cacheReuse,
          mmap: attempt.mmap,
          batch: attempt.batch,
        });
        // 加载成功：之前的失败记录作废（那次可能只是内存被别的程序占着）
        this._failedAttempts?.delete(failKey);
        const reduced = attempt.layers < requested
          || attempt.contextTokens < ctx
          || attempt.flashAttn !== 'auto'
          || attempt.cacheReuse !== 256
          || attempt.mmap !== true
          || attempt.batch !== 0;
        this.loadNote = reduced
          ? {
              requested,
              used: attempt.layers,
              contextTokens: attempt.contextTokens,
              // 把“实际用了哪组参数”带出去：用户看到 0xC0000005 之后至少知道现在跑的是什么
              flags: { flashAttn: attempt.flashAttn, cacheReuse: attempt.cacheReuse, mmap: attempt.mmap, batch: attempt.batch },
              reason: `加载失败（疑似显存/内存不足或程序崩溃），已自动回退后加载成功（第 ${i + 1} 组配置）`,
            }
          : contextNote;
        return this.llama.status();
      } catch (err) {
        lastErr = err;
        const kinds = this._looksLikeContextOrModelFailure(err);
        // llama.cpp 的内存不足也印成 “error loading model: …”，别把它当权重损坏：
        // 那种情况降层数就能跑，判成“文件坏了”会让用户白删白下几十 GB。
        if (kinds.corrupt && this._looksLikeMemoryFailure(err)) kinds.corrupt = false;
        if (kinds.corrupt) {
          // 权重本身坏了：换层数/上下文都没用，直接给可执行的建议
          throw Object.assign(
            new Error(`${err.message}\n\n判断：模型文件不完整或已损坏（不是显存问题）。请回到模型列表删除该量化后重新下载；下载会保留已完成的分片，不会从头开始。`),
            { status: 500, corrupt: true },
          );
        }
        if (!this._failedAttempts) this._failedAttempts = new Map();
        if (!this._failedAttempts.has(failKey)) this._failedAttempts.set(failKey, new Set());
        this._failedAttempts.get(failKey).add(attemptKey(attempt));
        const retryable = kinds.context || this._looksLikeMemoryFailure(err) || this._looksLikeCrash(err);
        if (i === attempts.length - 1 || !retryable) throw err;
        // 日志只写“哪一组、为什么短一句话”，完整诊断留在状态面板与最终报错里。
        // 以前把整段长解释每档打印一次，日志被同一段话刷了 7 遍，反而更难读。
        const why = this.llama.exitInfo?.silent ? 'llama-server 无任何输出即退出' : String(err?.message || '').split('\n')[0].slice(0, 160);
        this.logger.warn?.(`[native] 加载失败（${attempt.layers} 层 / 上下文 ${attempt.contextTokens}${attempt.flashAttn !== 'auto' ? ` / fa=${attempt.flashAttn}` : ''}${attempt.batch ? ` / batch=${attempt.batch}` : ''}${attempt.mmap === false ? ' / no-mmap' : ''}），自动回退重试：${why}`);
        await this.llama.stop().catch(() => {});
      }
    }
    throw lastErr;
  }

  /**
   * 生成加载退避阶梯。
   *
   * 以前只有“层数 × 上下文”两个维度，而 0xC0000005 的真实成因远不止这两项：
   * - 强制 `-fa on`：部分模型/量化在该构建上没有 flash-attention 内核，一开就崩；
   * - `--cache-reuse`：混合注意力/SWA 模型（很多新版 MoE）上复用 KV 会越界；
   * - mmap：权重放在云同步盘/网络盘/加密盘时，访问违例直接崩；
   * - 批大小：内存很紧时峰值占用把系统压爆。
   * 现在每一步只动一个维度，保证“回退后能跑起来”的那一步是可解释、可复现的：
   *   1) 原配置（fa=auto、cache-reuse 256、mmap）
   *   2) 关掉 flash-attention（fa=off）
   *   3) 关掉 cache-reuse + 小批（-b 512）
   *   4) 层数减半 / 上下文 4096
   *   5) 纯 CPU（0 层）+ `--no-mmap` + 小批 + fa=off + 上下文 4096
   * 同一层数只保留一次，不重复尝试。
   */
  _buildLoadAttempts({ requested = 0, ctx = 8192, safe = false } = {}) {

    const half = Math.floor(requested / 2);
    const quarter = Math.floor(requested / 4);
    const smallCtx = Math.min(ctx, 4096);
    const raw = [
      { layers: requested, contextTokens: ctx, flashAttn: 'auto', cacheReuse: 256, mmap: true, batch: 0 },
      { layers: requested, contextTokens: ctx, flashAttn: 'off', cacheReuse: 256, mmap: true, batch: 0 },
      { layers: requested, contextTokens: ctx, flashAttn: 'off', cacheReuse: 0, mmap: true, batch: 512 },
      { layers: half, contextTokens: smallCtx, flashAttn: 'off', cacheReuse: 0, mmap: true, batch: 512 },
      { layers: quarter, contextTokens: smallCtx, flashAttn: 'off', cacheReuse: 0, mmap: true, batch: 512 },
      { layers: 0, contextTokens: smallCtx, flashAttn: 'off', cacheReuse: 0, mmap: true, batch: 0 },
      { layers: 0, contextTokens: smallCtx, flashAttn: 'off', cacheReuse: 0, mmap: false, batch: 256 },
      { layers: 0, contextTokens: 2048, flashAttn: null, cacheReuse: 0, mmap: false, batch: 128 },
    ];
    const seen = new Set();
    let out = [];
    for (const a of raw) {
      const key = attemptKey(a);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(a);
    }
    // safe（聊天崩溃后的救命重载）：只留“最保守的 CPU 组合”及之后的档位。
    // 到底哪些组合“上一次已经试过”，由 _load 拿着它自己记录的指纹再筛一遍
    // （这里只做位置收敛，不能反过来递归调用自己）。
    if (safe) {
      const idx = out.findIndex((a) => a.layers === 0 && a.mmap === true && a.batch === 0 && a.cacheReuse === 0);
      if (idx > 0) out = out.slice(idx);
    }
    return out;
  }

  /**
   * 内存体检：估算这份权重跑起来需要多少内存，并和本机总内存/当前可用内存对比。
   *
   * 只用于“提前说清楚”，不做拦截：Windows 下进程分配的虚拟内存超过可用物理内存
   * 时，llama.cpp 往往不是抛 out of memory，而是直接以 0xC0000005 崩掉——
   * 这正是用户看到的那个退出码最常见的成因。所以这里必须在启动前把数摆出来。
   *
   * @returns {{neededGB: number, totalGB: number, freeGB: number, tight: boolean, summary: string}}
   */
  _memoryReport(weights, gpuLayers = 0, contextTokens = 8192, { mmap = true, blockCount = null, budget: budgetOverride = null } = {}) {
    const round = (n) => Math.round(n * 10) / 10;
    // weights 可以是总字节数，也可以是 {totalBytes, blockCount}（有层数时算得更准）
    const w = typeof weights === 'object' && weights ? weights : { totalBytes: weights, blockCount: null };
    const totalBytes = Number(w.totalBytes) || 0;
    const layers = Number(w.blockCount) || Number(blockCount) || 0;
    const ctx = Math.max(1024, Number(contextTokens) || 8192);
    const weightsGB = totalBytes / 1e9;
    // KV 缓存：按“每 token 每层”估算，默认 llama.cpp 是 f16（2 字节）+ 8 头 GQA、
    // 每头 128 维。以前拍成“每 16K 上下文 1 GB”，对 48~64 层的 MoE 会低估 6~8 倍，
    // 于是出现“明明超了却说够 / 明明不够却说超”的离谱结论。
    const perLayerPerToken = 2 * 2 * 8 * 128; // K 与 V、2 字节、8 个 KV 头、每头 128 维
    const kvBytes = layers ? perLayerPerToken * layers * ctx : 0;
    const kvGB = layers ? kvBytes / 1e9 : Math.max(0.3, ctx / 16384);
    // 权重常驻内存：mmap 时专家张量仍会被读进内存，resident 只取一部分；
    // 不用 mmap 时整份权重都必须先读进来。
    const residentGB = weightsGB * (mmap ? 0.06 : 1.0);
    const runtimeGB = 1.5; // llama.cpp 自身缓冲、模板、tokenizer、日志
    const neededGB = round(weightsGB + residentGB + kvGB + runtimeGB);
    let totalGB = 0;
    let freeGB = 0;
    // 提交上限（= 物理内存 + 页面文件/swap）。这是“硬盘能不能兜底”的唯一真实指标：
    // 需求超过物理内存但没超过提交上限 → 能跑，只是慢；超过提交上限 → 申请内存直接失败。
    let budget = { commitLimitBytes: null, commitFreeBytes: null, canSpill: false, pagefileBytes: 0, source: 'os-only' };
    try {
      if (budgetOverride) {
        // budget 可注入：测试要能模拟“用户的机器”（16.6 GB / 7.4 GB 可用 / 配了多大页面文件），
        // 否则只能依赖跑测试那台机器本身的配置，边界情形根本测不到。
        budget = budgetOverride;
        totalGB = budgetOverride.totalBytes ? round(budgetOverride.totalBytes / 1e9) : 0;
        freeGB = budgetOverride.freeBytes ? round(budgetOverride.freeBytes / 1e9) : 0;
      } else {
        totalGB = round(os.totalmem() / 1e9);
        freeGB = round(os.freemem() / 1e9);
        budget = readMemoryBudget();
      }
    } catch { /* 某些受限环境拿不到内存信息 */ }
    const commitLimitGB = budget.commitLimitBytes ? round(budget.commitLimitBytes / 1e9) : 0;
    const commitFreeGB = budget.commitFreeBytes ? round(budget.commitFreeBytes / 1e9) : 0;
    // null（查不到）与 0（确实没配）必须分开：下面 spillOnly 的判断依赖它。
    // 「没配页面文件」可以断言“硬盘顶不上来”，“查不到”不行——那时只能说“不知道”。
    const pagefileKnown = budget.pagefileBytes !== null && budget.pagefileBytes !== undefined;
    const pagefileGB = pagefileKnown ? round(budget.pagefileBytes / 1e9) : 0;
    // 三四档判断，各自给不同的建议（以前把“物理内存不够”和“页面文件也不够”混成一句）：
    //  - exceedsTotal：连物理内存都能装下 → 没事
    //  - exceedsFree ：放不进当前可用物理内存，但装得进物理内存总量 → 关程序即可
    //  - exceedsCommit：连“物理内存 + 页面文件”都不够 → 必须换量化/换机器
    //  - spillOnly   ：物理内存装不下，但页面文件能兜住 → 能跑，会慢很多
    const exceedsFree = totalGB > 0 && freeGB > 0 && neededGB > freeGB;
    const exceedsTotal = totalGB > 0 && neededGB > totalGB;
    const exceedsCommit = commitLimitGB > 0 && neededGB > commitLimitGB;
    // spillOnly 的语义是“物理内存装不下，但确实有硬盘兜底、能跑只是慢”。
    // 必须要求 pagefileKnown：否则探测不到页面文件的机器会被误报成“能跑”，
    // 把用户引到一个不存在的方案上。
    const spillOnly = exceedsTotal && !exceedsCommit && pagefileKnown && pagefileGB > 0;
    // tight 只表示“这份模型值得提醒用户”：装不进物理内存总量就算提醒。
    // 不能把 exceedsCommit 也算进来——某些环境（容器、禁用 overcommit 的内核）
    // 提交上限比物理内存还小，那样连 1.5 GB 的小模型都会被报“装不下”。
    const tight = exceedsTotal;
    const parts = [`这份权重约需 ${neededGB} GB 内存（权重 ${round(weightsGB)} GB + KV 缓存 ${round(kvGB)} GB + 运行时 ${runtimeGB} GB），本机总内存 ${totalGB} GB、当前可用 ${freeGB} GB`];
    if (commitLimitGB > 0) {
      parts.push(`（含页面文件/swap 可提交上限 ${commitLimitGB} GB${
        pagefileGB > 0 ? `，其中页面文件 ${pagefileGB} GB`
          : pagefileKnown ? '，未配页面文件' : '，页面文件大小未能探测'}）`);
    }
    if (totalGB > 0) {
      if (exceedsCommit && exceedsTotal) {
        parts.push('：连“物理内存 + 页面文件”的上限都超过了，系统无法再分配这么多内存，必须换更小的量化或换机器。');
      } else if (exceedsCommit && !exceedsTotal) {
        // 提交上限比物理内存还小（容器/禁用 overcommit）：说明这台机器限制很紧
        parts.push('：物理内存本身够，但系统的可提交内存上限比物理内存还小，可能需要调大页面文件/放宽限制。');
      } else if (spillOnly) {
        parts.push(`：物理内存装不下，但页面文件还能兜住，理论上可以跑——只是会频繁换页到硬盘（SSD 慢几十倍、机械硬盘更慢），`
          + '生成速度会低到几乎不可用，还容易超时。建议换更小的量化，装进物理内存才是正常速度。');
      } else if (exceedsFree) {
        parts.push('：当前可用内存不够，关掉其它占内存的程序后可以再试（总内存足够）。');
      } else {
        parts.push('，内存够用。');
      }
    }
    const summary = totalGB > 0 ? parts.join('') : `这份权重约需 ${neededGB} GB 内存（含上下文 ${ctx} 的 KV 缓存）。`;
    // 真的要“拿硬盘顶”时到底该开多大页面文件：给出具体数字，用户才能照着做
    const pagefilePlan = totalGB > 0 ? planPagefile({ neededGB, totalGB, pagefileGB, commitLimitGB }) : null;
    return {
      neededGB,
      weightsGB: round(weightsGB),
      kvGB: round(kvGB),
      totalGB,
      freeGB,
      // 硬盘兜底相关（界面据此解释“为什么不能靠 C/D 盘”，而不是让用户以为软件不会用硬盘）
      commitLimitGB,
      commitFreeGB,
      pagefileGB,
      // 页面文件是否真的探测到了：界面据此区分“确实没配”（可以断言没有硬盘兜底）
      // 和“没探测到”（只能说不知道），别再把两者说成同一句话。
      pagefileKnown,
      canSpill: !!budget.canSpill,
      memSource: budget.source,
      tight,
      exceedsTotal,
      exceedsFree,
      exceedsCommit,
      spillOnly,
      pagefilePlan,
      gpuLayers: Math.max(0, Math.floor(gpuLayers) || 0),
      contextTokens: ctx,
      blockCount: layers || null,
      summary,
    };
  }

  /**
   * 加载前的权重体检：分片是否齐全、大小是否与清单一致、入口分片的 GGUF 头能否解析。
   *
   * 返回 { ok: true, blockCount, architecture } 或 { ok: false, message, status, corrupt }。
   * 只在“入口分片连 GGUF 头都读不出来”时判 corrupt——仅仅是文件大小对不上时，
   * 有的仓库 LFS 记录的 size 与实际字节略有出入（重新上传/转换工具差异），
   * 那种情况仍让 llama.cpp 去试，避免把好模型误判成坏文件让用户白下几十 GB。
   */
  async _verifyWeights(id, quant, manifest, entryFile) {
    const dir = this.modelDir(id, quant);
    const shortNames = (list) => list.slice(0, 3).join('、') + (list.length > 3 ? ` 等 ${list.length} 个` : '');
    const missing = [];
    const truncated = [];
    const oversized = [];
    let entrySize = 0;
    for (const f of manifest.files) {
      const p = path.join(dir, f.path);
      let st = null;
      try {
        st = await stat(p);
      } catch {
        missing.push(f.path);
        continue;
      }
      const want = Number(f.size) || 0;
      if (f.path === entryFile.path) entrySize = st.size;
      if (!want || st.size === want) continue;
      if (st.size < want) truncated.push(`${f.path}（应 ${want} 字节，实际 ${st.size} 字节）`);
      else oversized.push(`${f.path}（清单 ${want} 字节，实际 ${st.size} 字节）`);
    }
    if (missing.length) {
      return {
        ok: false,
        corrupt: true,
        message: `权重文件不完整：缺少 ${missing.length} 个分片（${shortNames(missing)}）。\n`
          + '这不是内存不够，是文件没下全。请在“本机大模型”里重新点下载补齐（会从断点续传，不会从头开始），再点加载。',
      };
    }
    if (truncated.length) {
      return {
        ok: false,
        corrupt: true,
        message: `权重文件没下完：${shortNames(truncated)} 的字节数比清单少（下载中断/文件被截断）。\n`
          + '请重新点一次下载，软件会按已下载的字节续传补齐，再点加载。',
      };
    }
    const meta = await readGgufMeta(path.join(dir, entryFile.path));
    if (!meta.ok) {
      // 小文件（< 4KB）几乎不可能是真的权重：那更像是测试占位/空壳文件，
      // 只警告不拦，交给 llama.cpp 自己判断；真实体积的分片头读不出来就是坏文件。
      if (entrySize >= 4096) {
        return {
          ok: false,
          corrupt: true,
          message: `入口分片 ${entryFile.path} 的 GGUF 文件头读不出来：${meta.reason}。\n`
            + '这个文件多半已损坏或根本不是 GGUF。请在“本机大模型”里删除这一档量化后重新下载（其它量化不受影响）。',
        };
      }
      this.logger.warn?.(`[native] 入口分片 ${entryFile.path} 太小（${entrySize} 字节）且读不出 GGUF 头，仍尝试加载。`);
      return { ok: true, blockCount: null, architecture: '' };
    }
    if (meta.splitCount && meta.splitCount > manifest.files.length) {
      return {
        ok: false,
        corrupt: true,
        message: `权重分片不全：文件头声明共 ${meta.splitCount} 个分片，本地只有 ${manifest.files.length} 个。`
          + '请重新点下载补齐（会从断点续传）。',
      };
    }
    if (oversized.length) {
      this.logger.warn?.(`[native] 权重大小与清单不一致（仍尝试加载）：${shortNames(oversized)}`);
    }
    return { ok: true, blockCount: meta.blockCount, architecture: meta.architecture || '' };
  }

  /**
   * 失败是否疑似显存/内存不足（决定要不要自动减少 GPU 卸载层数重试）。
   *
   * 注意：以前这里把"llama-server 已退出"一律当成内存不足，于是静默退出（缺 DLL、
   * 参数不认、程序损坏）也被套上"内存不够"的结论，用户被彻底误导。
   * 现在只有日志里真的出现分配失败/显存不足的措辞才算；一行日志都没有的静默退出
   * 属于另一类故障（见 _diagnoseFailure）。
   */
  _looksLikeMemoryFailure(err) {
    // 架构不支持不是内存问题：日志里的 "failed to load model" 会命中下面的内存正则，
    // 但换层数/上下文/关 mmap 都不可能让不认识的架构变得认识 —— 让它走回退阶梯
    // 只会让用户多等 7 轮，最后还是那句错的“内存不足”。
    if (matchUnsupportedArch((this.llama.logTail || []).join('\n'))) return false;
    // 证据只认 llama.cpp 自己的输出（logTail）与未包装的原始错误。
    // 绝不能把我们自己生成的解释文本（例如“这不是内存不足”）也算成证据——
    // 那样会自我命中原生提示里的“内存不足”四个字，把结论判反。
    const own = (this.llama.logTail || []).join('\n');
    const raw = String(err?.rawMessage ?? err?.message ?? '');
    const text = `${raw}\n${own}`;
    // 我们包装过的长句（含“判断：”“建议：”等）不参与判定
    if (/判断：|建议：|内存情况：/.test(raw)) return /out of memory|failed to allocate|bad_alloc|not enough memory|cannot allocate/i.test(own);
    return /out of memory|failed to allocate|failed to load model|cuda error|vulkan.*(error|failed)|bad_alloc|内存不足|显存|not enough memory|cannot allocate/i.test(text);
  }

  /**
   * 根据日志与退出码给失败定性，并给出一句用户能照做的结论。
   *
   * 这是本次修复的重点：用户拿到的报错必须能区分这几种完全不同的情况——
   *   silent   ：一句日志都没有 → 缺 DLL / 参数不认 / 运行库不匹配（不是内存问题）
   *   memory   ：日志里有分配失败 → 内存/显存不够
   *   corrupt  ：读权重时报错 → 文件损坏/不完整
   *   unknown  ：有日志但没有已知特征 → 原文给用户看
   */
  _diagnoseFailure(err) {
    // 内层已经定过性（我们自己包装过的 502）：直接沿用，避免拿解释文本再判一次把结论判反
    if (err?.diag && DIAG_TITLES[err.diag]) {
      return { kind: err.diag, ...DIAG_TITLES[err.diag] };
    }
    const text = `${err?.message || ''}\n${(this.llama.logTail || []).join('\n')}`;
    // 架构不支持要排在内存之前：日志里那句 "failed to load model" 同时会命中内存正则，
    // 先判内存就会把“版本太旧”说成“内存不足”，用户于是去换量化/调页面文件，全走错。
    const arch = matchUnsupportedArch(text);
    if (arch) {
      const minBuild = minBuildForArch(arch);
      const curBuild = buildNumber(this.binaryJob?.tag || '');
      return {
        kind: 'architecture',
        title: `这个 llama.cpp 版本不支持 ${arch} 架构`,
        advice: `模型文件本身没有问题，是 llama-server 太旧${minBuild ? `（需要 b${minBuild} 或更新）` : ''}`
          + `${curBuild && minBuild && curBuild < minBuild ? `；当前是 b${curBuild}` : ''}。`
          + '点“准备运行库”更新 llama.cpp 后重试。',
        arch,
        minBuild: minBuild || null,
        currentBuild: curBuild || null,
      };
    }
    const silent = !!this.llama.exitInfo?.silent || /一句话都没输出/.test(err?.message || '');
    const missingDll = /missing|not found|cannot open shared object|找不到|\b0xc0000135\b/i.test(text);
    const badFlag = /invalid argument|unrecognized|unknown argument|invalid option|error: (unrecognized|invalid)/i.test(text);
    if (missingDll || badFlag) {
      return {
        kind: 'runtime',
        title: missingDll ? '运行库不完整（缺 DLL/so）' : 'llama.cpp 不认某个命令行参数',
        advice: missingDll
          ? '点“准备运行库”重装；若装的是 CUDA/Vulkan 版，把 OJ_LLAMA_VARIANT 设为 cpu 改用纯 CPU 版。'
          : '点“准备运行库”更新到新版 llama.cpp（当前版本缺少这些参数）。',
      };
    }
    if (this._looksLikeMemoryFailure(err)) {
      return { kind: 'memory', title: '内存/显存不足', advice: '换更小的量化，或关掉其它占用内存/显存的程序后重试。' };
    }
    if (this._looksLikeContextOrModelFailure(err).corrupt) {
      return { kind: 'corrupt', title: '权重文件损坏或不完整', advice: '删除这一档量化后重新下载（会断点续传），再点加载。' };
    }
    if (silent) {
      return {
        kind: 'silent',
        title: 'llama-server 启动即退出且没有任何日志',
        advice: '这通常不是内存不够（内存不足时会先打印加载进度）。请点“准备运行库”重装 llama.cpp；'
          + '若仍失败，把 OJ_LLAMA_VARIANT 设为 cpu 改用纯 CPU 版，或用系统命令行直接运行 llama-server 看系统报错。',
      };
    }
    return { kind: 'unknown', title: 'llama-server 启动失败', advice: '请把状态面板里的日志尾部发给开发者。' };
  }

  /** 失败是否为原生程序崩溃（如 Windows 0xC0000005 内存访问冲突、段错误）。 */
  _looksLikeCrash(err) {
    const text = `${err?.message || ''}\n${(this.llama.logTail || []).join('\n')}`;
    return /3221225477|0xC0000005|-1073741819|access violation|segmentation|segfault/i.test(text);
  }

  /**
   * 失败是否因为“上下文开太大”或“权重文件本身坏了/不完整”。
   *
   * 这两类以前完全没有识别（只认显存不足与崩溃），于是：
   * - 超大模型把上下文设成 256K，llama.cpp 一启动就因 KV cache 分配失败退出，
   *   降一半上下文本来就能跑起来，却直接把错误甩给用户；
   * - 分片缺失/GGUF 截断，llama.cpp 报 "unknown model architecture"/"invalid magic"
   *   之类，重试多少次都没用——应该明确告诉用户“重新下载补齐分片”，而不是
   *   在层数/上下文阶梯里白白重试十几分钟。
   */
  _looksLikeContextOrModelFailure(err) {
    const text = `${err?.message || ''}\n${(this.llama.logTail || []).join('\n')}`;
    return {
      context: /context (size|length).*(too|exceed)|n_ctx|KV cache|kv_cache|failed to create context|context shift/i.test(text),
      corrupt: /invalid magic|unknown model architecture|unsupported model architecture|wrong magic|tensor.*(not found|missing)|failed to read (magic|tensor)|corrupted|truncated|invalid gguf|error loading model/i.test(text),
    };
  }

  async unload() {
    // 加载中就点卸载：加载流程还在跑，卸载后它仍会把新进程拉起来，
    // 用户看到“点了卸载，模型还在加载/最后又跑起来了”。这里明确拒绝。
    if (this.llama.state === 'loading') {
      throw Object.assign(
        new Error('模型正在加载中，暂时不能卸载。等加载完成（或加载失败）后再点卸载，或直接点另一个模型的“加载”来替换它。'),
        { status: 409 },
      );
    }
    await this.llama.stop();
    await this.python.stop();
    return this.llama.status();
  }

  /** 当前是否正在加载（路由层用它挡住会互相打架的操作）。 */
  get loading() {
    return this.llama.state === 'loading';
  }

  /** 云端聊天：走 Hugging Face Inference Providers（OpenAI 兼容），返回原始 Response。 */
  async remoteChatCompletion({ entry, messages, stream, temperature, maxTokens, signal, thinking }) {
    if (!this.hfToken) {
      throw Object.assign(new Error('尚未配置 Hugging Face Token：请在“本机大模型”面板里填入 Token 后再使用云端模型（免费额度即可）。'), { status: 503 });
    }
    const body = { model: entry.repo, messages, stream: !!stream };
    if (thinking === false) body.chat_template_kwargs = { enable_thinking: false };
    if (Number.isFinite(temperature)) body.temperature = temperature;
    if (Number.isInteger(maxTokens) && maxTokens > 0) body.max_tokens = maxTokens;
    if (stream) body.stream_options = { include_usage: true };
    const res = await this.fetchImpl(HF_ROUTER_URL, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${this.hfToken}` },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw Object.assign(new Error(`Hugging Face 云端生成失败（HTTP ${res.status}）${text ? `：${text.slice(0, 200)}` : ''}`), { status: 502 });
    }
    return res;
  }

  /**
   * 对话转发：本地已下载 → llama.cpp；已登记云端 → HF 云端推理；
   * 都没有 → 交给 load() 抛出“尚未下载”的 409 提示。返回原始 Response。
   */
  async chatCompletion({ model, messages, stream = false, temperature, maxTokens, signal, thinking } = {}) {
    const entry = this.resolveModel(model);
    if (!entry) throw Object.assign(new Error(`未知模型：${model}`), { status: 404 });
    // 思考开关：明确关闭时才下发 enable_thinking=false（开启/未指定沿用模型模板默认，
    // 不多塞参数，避免不认该字段的后端报错）。此前服务端从不下发这个开关，
    // 界面开关关着也照样思考，就是这里缺了透传。
    const thinkOff = thinking === false ? { chat_template_kwargs: { enable_thinking: false } } : {};
    const downloaded = await this.downloadedQuants(model);
    const hasGguf = downloaded.some((d) => d.quant !== 'SNAPSHOT');
    const hasSnapshot = downloaded.some((d) => d.quant === 'SNAPSHOT');
    // 用户明确切到云端时云端优先；否则本地优先（GGUF → 原格式 Python → 云端兜底）
    if (this.preferCloud.has(model) && this.remote.has(model)) {
      return this.remoteChatCompletion({ entry, messages, stream, temperature, maxTokens, signal, thinking });
    }
    if (!hasGguf && !hasSnapshot && this.remote.has(model)) {
      return this.remoteChatCompletion({ entry, messages, stream, temperature, maxTokens, signal, thinking });
    }
    // 原格式本地运行（Python）：只有快照、没有 GGUF 时走这条路
    if (!hasGguf && hasSnapshot) {
      await this._ensurePythonLoaded(model);
      const body = { model: entry.repo, messages, stream: !!stream, ...thinkOff };
      if (Number.isFinite(temperature)) body.temperature = temperature;
      if (Number.isInteger(maxTokens) && maxTokens > 0) body.max_tokens = maxTokens;
      const res = await this.fetchImpl(`${this.python.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${this.python.apiKey}` },
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw Object.assign(new Error(`本地 Python 运行生成失败（HTTP ${res.status}）${text ? `：${text.slice(0, 200)}` : ''}`), { status: 502 });
      }
      return res;
    }
    // 需要（重新）加载：模型换了、进程空闲、或上次已崩（state==='error'）。
    // 以前只判 idle：崩溃后 state==='error'，聊天请求会直接对着一具尸体转发，
    // 报“llama-server 尚未运行”，永远恢复不了，只能手动点一次加载。
    const needLoad = this.llama.info?.modelId !== model
      || this.llama.state === 'idle'
      || this.llama.state === 'error';
    let loadErr = null;
    if (needLoad) {
      try {
        // 自动加载沿用管理员上次显式设置的上下文/层数（不再固定 8192/0）
        await this.load(model, { explicit: false });
      } catch (err) {
        // 加载本身失败（崩溃/内存不足）：聊天请求不要变成 500 的“unexpected”，
        // 先把它转成一次“安全配置重载”，失败也给出可执行的中文原因。
        if (!this._looksLikeCrash(err) && !this._looksLikeMemoryFailure(err)) throw err;
        loadErr = err;
      }
    }
    try {
      if (loadErr) throw loadErr;
      await this.llama.whenReady();
    } catch (err) {
      // 加载途中 llama-server 崩溃/内存失败（如 0xC0000005）：以安全配置
      // （纯 CPU + 上下文 4096 + 精简参数）自动重新加载一次再试，仍失败则给出明确建议。
      if (!this._looksLikeCrash(err) && !this._looksLikeMemoryFailure(err)) throw err;
      // 权重本身的问题（缺片/截断/头读不出来）已经由 _verifyWeights 给了精确原因，
      // 再拿安全配置重试一次毫无意义，只会让用户多等一轮。
      if (err?.corrupt) throw err;
      this.logger.warn?.(`[native] llama-server 加载失败，聊天自动以安全配置重新加载：${err.message}`);
      try {
        // safe: 这次只是“先把服务救活”，不覆盖管理员原本设定的上下文/层数，
        // 否则一次崩溃会把 32K 上下文悄悄降成 4096，之后用户以为设置没生效。
        await this.load(model, { contextTokens: 4096, gpuLayers: 0, explicit: false, safe: true });
        await this.llama.whenReady();
      } catch (retryErr) {
        // 权重坏了下游还会走到这里（safe 重载会再体检一次）：把真实原因原样带出去，
        // 不要再包一层“纯 CPU 也起不来”让人误以为要换更小的量化。
        if (retryErr?.corrupt) throw retryErr;
        // 连纯 CPU + 4096 上下文都起不来：不是配置问题，而是权重/内存/驱动问题，
        // 直接告诉用户下一步该干什么，别让他们对着退出码猜。
        const diag = this._diagnoseFailure(retryErr);
        const mem = this.memoryReport;
        const inner = String(retryErr?.message || '');
        // 内层已经给出完整判断与建议时，外层只补"这次是降级后的第二次尝试"，
        // 不再把同样的内存数字和建议重复一遍（用户看到的就是一大坨重复文字）。
        const alreadyExplained = /判断：/.test(inner);
        // 按证据给建议，而不是每次都把三条建议全倒给用户：
        // “换更小的量化”对缺 DLL 的故障毫无用处，只会让人白折腾一次。
        // 内存类故障：如果有可操作的页面文件数字，就一起给出来——
        // 这才是“用硬盘顶内存”唯一真实可行的做法，比空口说“内存不够”有用得多。
        const pf = mem?.pagefilePlan;
        const pfTip = pf && !pf.enough ? `③ 或把页面文件调到约 ${pf.requiredPagefileGB} GB（能用硬盘顶住不崩，但会很慢）。` : '';
        const tips = {
          architecture: '① 点“准备运行库”把 llama.cpp 更新到更新版本（内置目录里的模型大多早已支持）；'
            + '② 若已是最新版仍不支持，改用一个架构更成熟的模型（如 Qwen2.5 / Qwen3 系列）。',
          memory: `① 在“本机大模型”里换更小的量化（如 Q4_K_M → Q2_K）；② 关闭其它占用内存的程序后重试。${pfTip}`,
          runtime: '① 点“准备运行库”重装 llama.cpp 运行库；② 若装的是 CUDA/Vulkan 版，把 OJ_LLAMA_VARIANT 设为 cpu；③ 确认已安装 VC++ 运行库。',
          corrupt: '① 删除这一档量化后重新下载（会断点续传）；② 再点加载。',
          silent: '① 点“准备运行库”重装 llama.cpp 运行库；② 把 OJ_LLAMA_VARIANT 设为 cpu 改用纯 CPU 版；③ 仍失败时用系统命令行直接运行 llama-server，把系统报错发给开发者。',
          unknown: '请把状态面板里的日志尾部发给开发者。',
        };
        const message = alreadyExplained
          ? `llama-server 无法启动（自动降级到纯 CPU + 最小上下文后仍然失败）。\n\n${inner}`
          : `llama-server 无法启动（已自动降级并逐档重试，仍然失败）。\n`
            + `最后一次错误：${inner.slice(0, 300)}\n`
            + `\n判断：${diag.title}。\n`
            + (diag.kind === 'memory' && mem?.summary ? `内存情况：${mem.summary}\n` : '')
            + `\n建议：${tips[diag.kind] || tips.unknown}`;
        throw Object.assign(new Error(message), { status: 502, loadFailed: true, diag: diag.kind, rawMessage: inner });
      }
    }
    // 超大模型的上下文窗口往往只有 4K~32K，而网页/API 一次可以送来 8 万字符的历史：
    // 直接整包发给 llama.cpp，轻则被它悄悄截断（前面的系统提示/关键信息丢了），
    // 重则直接 400 “exceeds context”。这里按实际上下文预算裁剪历史：
    // 保留最前面的 system 消息（系统提示、检索证据）与最近的消息，
    // 被裁掉的部分在日志里说明，用户不会以为模型“忘了”。
    const fitted = fitMessagesToContext(messages, {
      contextTokens: this.llama.info?.contextTokens || 8192,
      reserveForOutput: Number.isInteger(maxTokens) && maxTokens > 0 ? maxTokens : 1024,
    });
    if (fitted.droppedMessages > 0) {
      this.logger.warn?.(`[native] 对话历史超出上下文（约 ${fitted.estimatedTokens} tokens > ${fitted.contextTokens}），已省略最早的 ${fitted.droppedMessages} 条消息以留出输出空间。`);
      // 面板上给一句说明：不然用户只会觉得“模型怎么忘了我前面说的”
      this.contextTrim = {
        droppedMessages: fitted.droppedMessages,
        estimatedTokens: fitted.estimatedTokens,
        contextTokens: fitted.contextTokens,
        at: new Date().toISOString(),
      };
    } else {
      // 这次没裁：清掉上一次的提示，避免面板一直挂着旧记录
      this.contextTrim = null;
    }
    const body = {
      model: entry.id,
      messages: fitted.messages,
      stream: !!stream,
      ...thinkOff,
    };
    if (Number.isFinite(temperature)) body.temperature = temperature;
    if (Number.isInteger(maxTokens) && maxTokens > 0) body.max_tokens = maxTokens;
    if (stream) body.stream_options = { include_usage: true };
    const send = () => this.fetchImpl(`${this.llama.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${this.llama.apiKey}`,
      },
      body: JSON.stringify(body),
      signal,
    });
    let res = await send();
    if (!res.ok) {
      let text = await res.text().catch(() => '');
      // llama.cpp 判定输入超窗（上下文估算比实际小，中文/代码最容易差这一截）：
      // 不要再把 400 直接甩给用户，把历史按一半预算再裁一次重发，用户几乎无感。
      if (/exceed|too (long|large)|context (size|length)|n_ctx/i.test(text) && fitted.droppedMessages >= 0) {
        const tight = fitMessagesToContext(messages, {
          contextTokens: Math.max(1024, Math.floor((this.llama.info?.contextTokens || 8192) / 2)),
          reserveForOutput: Number.isInteger(maxTokens) && maxTokens > 0 ? maxTokens : 1024,
        });
        if (tight.messages.length !== fitted.messages.length || tight.estimatedTokens < fitted.estimatedTokens) {
          this.logger.warn?.('[native] llama.cpp 提示输入超窗，已按更小的预算重新裁剪对话历史后重试一次。');
          this.contextTrim = {
            droppedMessages: tight.droppedMessages,
            estimatedTokens: tight.estimatedTokens,
            contextTokens: tight.contextTokens,
            at: new Date().toISOString(),
          };
          body.messages = tight.messages;
          res = await send();
          if (!res.ok) text = await res.text().catch(() => '');
        }
      }
      if (!res.ok) {
        throw Object.assign(
          new Error(`llama.cpp 生成失败（HTTP ${res.status}）${text ? `：${text.slice(0, 200)}` : ''}`),
          { status: 502 },
        );
      }
    }
    return res;
  }
}
