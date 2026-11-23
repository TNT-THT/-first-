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
import { appendFile, mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  describeQuant,
  groupFilesByQuant,
  nativeModelById,
  pickDefaultQuant,
  NATIVE_MODELS,
} from './catalog.js';
import { detectFormats, detectVariants, hfModelId, optionsForFormats, routeForFormats, safeDirName } from './formats.js';
import { downloadFile, fetchModelInfo, fetchRepoFiles, freeDiskBytes, resolveFileUrl, searchModels } from './hf.js';
import { downloadLlamaBinary, findLlamaBinary, LlamaProcess } from './llama.js';
import { findConverter, prepareConverter, runConvert } from './convert.js';
import { findPythonRuntime, preparePythonRuntime, PythonProcess } from './python-runner.js';

const QUANT_CACHE_MS = 10 * 60 * 1000;
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
    this.quantCache = new Map(); // repo -> {at, grouped}
    this.binaryJob = null;
    this.conversions = new Map(); // id -> {id, state, error}
    this.converterJob = null;
    this._converterCache = null;
    // 本地 Python 运行时（特殊格式原格式本地运行）
    this.python = new PythonProcess({ fetchImpl });
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
        if (e?.id && e?.repo) this.custom.set(e.id, e);
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
      return {
        id: custom.id,
        name: custom.name || custom.repo,
        vendor: 'Hugging Face',
        paramsB: custom.paramsB || null,
        activeB: null,
        reasoning: false,
        license: '见仓库说明',
        context: custom.context || 32768,
        note: '从 Hugging Face 添加的模型。',
        repo: custom.repo,
        formats: custom.formats || [],
        kind: 'hf',
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
      if (ok) out.push({ quant, totalBytes: manifest.totalBytes, files: manifest.files.length, ...describeQuant(quant) });
    }
    return out;
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
    if (cached && Date.now() - cached.at < QUANT_CACHE_MS) return cached.grouped;
    const files = await fetchRepoFiles(model.repo, { fetchImpl: this.fetchImpl });
    const grouped = groupFilesByQuant(files);
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
      binaryJob: this.binaryJob ? { state: this.binaryJob.state, error: this.binaryJob.error || '' } : null,
      running: this.llama.status(),
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
  async ensureBinary() {
    const found = findLlamaBinary({ env: this.env, dataDir: this.dataDir });
    if (found) return found;
    if (this.binaryJob?.state === 'downloading') return this.binaryJob.promise;
    this.binaryJob = { state: 'downloading', error: '' };
    this.binaryJob.promise = (async () => {
      try {
        const got = await downloadLlamaBinary({
          dataDir: this.dataDir,
          fetchImpl: this.fetchImpl,
          variant: this.variant,
        });
        this.binaryJob = { state: 'done', error: '' };
        return { path: got.path, source: 'downloaded' };
      } catch (err) {
        this.binaryJob = { state: 'error', error: err.message };
        throw err;
      }
    })();
    return this.binaryJob.promise;
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
    const results = await searchModels({ query, ...opts, fetchImpl: this.fetchImpl });
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
    const info = await fetchModelInfo(repo, { fetchImpl: this.fetchImpl });
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
  async useRepo(repo, mode = 'auto', { info: preloaded = null, viaBrowser = false } = {}) {
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
        const entry = this._pickQuantEntry(grouped, undefined);
        return { id, action: 'download', via: 'browser', route, options, quant: entry?.quant || null, formats: info.formats };
      }
      const job = await this.download(id);
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
    const files = await fetchRepoFiles(model.repo, { fetchImpl: this.fetchImpl });
    await mkdir(dir, { recursive: true });
    const saved = [];
    for (const file of files) {
      if (/^(\.gitattributes|README)/i.test(file.path)) continue;
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
        const snapshotDir = path.join(this.dataDir, 'convert', safeDirName(id), 'snapshot');
        await this._downloadSnapshotTo(model, snapshotDir);
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
    if (Number.isFinite(free) && free < entry.totalBytes * 1.05) {
      throw Object.assign(new Error(`磁盘空间不足：该量化需要约 ${(entry.totalBytes / 1e9).toFixed(1)} GB，可用约 ${(free / 1e9).toFixed(1)} GB。`), { status: 507 });
    }
    const job = { id, quant: entry.quant, state: 'downloading', downloaded: 0, total: entry.totalBytes, error: '' };
    this.downloads.set(key, job);
    const dir = this.modelDir(id, entry.quant);
    await mkdir(dir, { recursive: true });
    (async () => {
      const ac = new AbortController();
      try {
        // 大模型通常有几十个分片：3 路并行一起拉，比排队逐个下快得多；
        // 并发只按文件分工，单个文件内部仍是一条流顺序写，不会写乱。
        // 已下完的分片由 downloadFile 按大小直接跳过，失败重试不从头来。
        const perFile = new Array(entry.files.length).fill(0);
        const bump = () => { job.downloaded = perFile.reduce((n, v) => n + v, 0); };
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
              perFile[i] = Math.max(perFile[i], Number(done?.bytes) || file.size);
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

  /** 单个仓库相对路径是否安全（不逃出量化目录）。 */
  _isSafeRepoPath(p) {
    if (typeof p !== 'string' || !p || p.length > 512) return false;
    if (p.includes('\\') || p.includes('\0') || p.startsWith('/') || /^[a-zA-Z]:/.test(p)) return false;
    return p.split('/').every((seg) => seg !== '..' && seg !== '');
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
    const downloaded = await this.downloadedQuants(id);
    if (downloaded.some((d) => d.quant === entry.quant)) {
      const done = { id, quant: entry.quant, state: 'done', downloaded: entry.totalBytes, total: entry.totalBytes, error: '' };
      this.downloads.set(key, done);
      return { id, quant: entry.quant, state: 'done', total: entry.totalBytes, files: [] };
    }
    const existing = this.downloads.get(key);
    if (existing?.state === 'downloading' && existing.via !== 'browser') {
      throw Object.assign(new Error('该量化已有软件直连下载在进行中，等它完成或失败后再试。'), { status: 409 });
    }
    const free = await freeDiskBytes(this.modelsDir);
    if (Number.isFinite(free) && free < entry.totalBytes * 1.05) {
      throw Object.assign(new Error(`磁盘空间不足：该量化需要约 ${(entry.totalBytes / 1e9).toFixed(1)} GB，可用约 ${(free / 1e9).toFixed(1)} GB。`), { status: 507 });
    }
    const dir = this.modelDir(id, entry.quant);
    this._expireStaleBrowserJobs();
    const job = existing?.via === 'browser' && existing.state === 'downloading'
      ? existing
      : {
          id, quant: entry.quant, state: 'downloading', downloaded: 0, total: entry.totalBytes,
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
    const job = this.downloads.get(key);
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

  /** 管理员/聊天自动：加载模型（已下载的量化；会先卸载当前模型）。 */
  async load(id, { quant, contextTokens = 8192, gpuLayers = 0 } = {}) {
    const model = this.resolveModel(id);
    if (!model) throw Object.assign(new Error(`未知模型：${id}`), { status: 404 });
    if (this.llama.info?.modelId === id && this.llama.state === 'ready') {
      return this.llama.status();
    }
    const downloaded = await this.downloadedQuants(id);
    const pick = downloaded.find((d) => d.quant === String(quant || '').toUpperCase()) || downloaded[0];
    if (!pick) {
      throw Object.assign(
        new Error(`模型 ${model.name} 尚未下载。请在“API 接入 → 本机大模型”中先下载权重。`),
        { status: 409 },
      );
    }
    const binary = findLlamaBinary({ env: this.env, dataDir: this.dataDir });
    if (!binary) {
      throw Object.assign(
        new Error('llama.cpp 运行库尚未准备。请管理员在“API 接入 → 本机大模型”中点“准备运行库”。'),
        { status: 503 },
      );
    }
    const manifest = await this.readManifest(id, pick.quant);
    const firstFile = manifest.files.map((f) => f.path).sort()[0];
    const modelFile = path.join(this.modelDir(id, pick.quant), firstFile);
    if (!existsSync(modelFile)) throw new Error(`权重文件缺失：${firstFile}`);
    const ctx = Math.min(Math.max(1024, Math.floor(contextTokens) || 8192), model.context || 8192);
    const port = 18100 + Math.floor(Math.random() * 800);
    await this.llama.start({
      binary: binary.path,
      modelFile,
      modelId: id,
      quant: pick.quant,
      contextTokens: ctx,
      gpuLayers,
      port,
    });
    return this.llama.status();
  }

  async unload() {
    await this.llama.stop();
    await this.python.stop();
    return this.llama.status();
  }

  /** 云端聊天：走 Hugging Face Inference Providers（OpenAI 兼容），返回原始 Response。 */
  async remoteChatCompletion({ entry, messages, stream, temperature, maxTokens, signal }) {
    if (!this.hfToken) {
      throw Object.assign(new Error('尚未配置 Hugging Face Token：请在“本机大模型”面板里填入 Token 后再使用云端模型（免费额度即可）。'), { status: 503 });
    }
    const body = { model: entry.repo, messages, stream: !!stream };
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
  async chatCompletion({ model, messages, stream = false, temperature, maxTokens, signal } = {}) {
    const entry = this.resolveModel(model);
    if (!entry) throw Object.assign(new Error(`未知模型：${model}`), { status: 404 });
    const downloaded = await this.downloadedQuants(model);
    const hasGguf = downloaded.some((d) => d.quant !== 'SNAPSHOT');
    const hasSnapshot = downloaded.some((d) => d.quant === 'SNAPSHOT');
    // 用户明确切到云端时云端优先；否则本地优先（GGUF → 原格式 Python → 云端兜底）
    if (this.preferCloud.has(model) && this.remote.has(model)) {
      return this.remoteChatCompletion({ entry, messages, stream, temperature, maxTokens, signal });
    }
    if (!hasGguf && !hasSnapshot && this.remote.has(model)) {
      return this.remoteChatCompletion({ entry, messages, stream, temperature, maxTokens, signal });
    }
    // 原格式本地运行（Python）：只有快照、没有 GGUF 时走这条路
    if (!hasGguf && hasSnapshot) {
      await this._ensurePythonLoaded(model);
      const body = { model: entry.repo, messages, stream: !!stream };
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
    if (this.llama.info?.modelId !== model || this.llama.state === 'idle') {
      await this.load(model, {});
    }
    await this.llama.whenReady();
    const body = {
      model: entry.id,
      messages,
      stream: !!stream,
    };
    if (Number.isFinite(temperature)) body.temperature = temperature;
    if (Number.isInteger(maxTokens) && maxTokens > 0) body.max_tokens = maxTokens;
    if (stream) body.stream_options = { include_usage: true };
    const res = await this.fetchImpl(`${this.llama.baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${this.llama.apiKey}`,
      },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw Object.assign(new Error(`llama.cpp 生成失败（HTTP ${res.status}）${text ? `：${text.slice(0, 200)}` : ''}`), { status: 502 });
    }
    return res;
  }
}
