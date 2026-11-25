/**
 * llama.cpp 运行库（llama-server）管理：
 * - 查找顺序：环境变量 LLAMA_SERVER_BIN → 独立软件/网站包随附的 native-bin →
 *   数据目录里已下载的 bin → 系统 PATH。都找不到时可由管理员一键从
 *   ggml-org/llama.cpp 的 GitHub Release 下载对应平台的官方构建。
 * - 进程管理：单实例（同一时间只跑一个大模型），随机端口 + 随机 api-key，
 *   只绑 127.0.0.1；本站服务端再把对话请求代理过去。
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { chmod, mkdir, readdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractTarGz, extractZip } from './archive.js';

// 单文件打包（SEA/CJS）里没有 import.meta.url，退回 __filename（普通路径不能 fileURLToPath）。
const SERVER_DIR = import.meta.url && String(import.meta.url).startsWith('file:')
  ? path.dirname(fileURLToPath(import.meta.url))
  : path.resolve(typeof __filename === 'string' ? path.dirname(__filename) : process.cwd());
export const LLAMA_RELEASE_API = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10';

/**
 * 新模型架构 → 最早支持它的 llama.cpp 构建号。
 *
 * 这张表是为了解决“版本太旧”这类故障：llama.cpp 遇到不认识的架构只会打印
 * `unknown model architecture: 'qwen35moe'` 然后退出，用户完全不知道该升到哪一版，
 * 只能一个版本一个版本去试。这里给出确切的最低版本。
 *
 * 来源：ggml-org/llama.cpp 的模型支持 PR。b7973 首次加入 Qwen3.5 后被回滚（#19453），
 * 直到 b8239 才真正可用，社区实测 b8239/b8245 可以正常加载。
 */
export const ARCH_MIN_BUILD = {
  qwen35moe: 8239,
  qwen35: 8239,
  qwen3next: 7900,
};

/** 从 llama.cpp 日志里读出“不认识的架构”；没有则返回空字符串。 */
export function matchUnsupportedArch(text) {
  const m = String(text || '').match(/unknown model architecture:\s*['"]?([A-Za-z0-9_.-]+)['"]?/i);
  return m ? m[1] : '';
}

/** 已知架构的最低构建号（未知架构返回 0）。 */
export function minBuildForArch(arch) {
  return ARCH_MIN_BUILD[String(arch || '').toLowerCase()] || 0;
}

/**
 * 从发布列表里挑最新的 b 系列正式构建（/releases/latest 指向的可能不是 b 系列，
 * 直接用它会挑错）。返回 {tag_name, assets} 或 null。
 */
export function pickLlamaRelease(releases, { tag = '' } = {}) {
  const list = Array.isArray(releases) ? releases : [];
  // 指定版本优先：某些新构建会在老系统上静默崩溃，必须能钉在已知可用的版本上
  const want = String(tag || '').trim();
  if (want) {
    const hit = list.find((r) => String(r?.tag_name || '') === want && Array.isArray(r.assets) && r.assets.length);
    // 指定了版本却找不到时必须明确失败，绝不能悄悄退回最新版：
    // 用户锁定旧版正是因为新版跑不了，静默给他装回新版等于白折腾一轮。
    return hit || null;
  }
  for (const rel of list) {
    if (!/^b\d+$/.test(String(rel?.tag_name || ''))) continue;
    if (Array.isArray(rel.assets) && rel.assets.length) return rel;
  }
  return null;
}

/**
 * 把各种版本写法统一成构建号：`b8421` / `8421` / `version: 8421 (deadbeef)` → 8421。
 * 解析不出来返回 0（调用方据此判断“版本未知”，不要拿 0 去和最低版本比较）。
 */
export function buildNumber(tag) {
  const raw = String(tag || '').trim();
  const strict = raw.match(/^b?(\d+)$/i);
  if (strict) return Number(strict[1]);
  // llama-server --version 的输出形如 "version: 8421 (deadbeef)"
  const loose = raw.match(/\bb?(\d{3,7})\b/);
  return loose ? Number(loose[1]) : 0;
}

export function llamaBinaryName(platform = process.platform) {
  return platform === 'win32' ? 'llama-server.exe' : 'llama-server';
}

/**
 * 按平台给 llama.cpp Release 资源打分，挑当前平台最合适的一个。
 *
 * 必须只认 `llama-` 开头的正式构建包：同一发布里还有 `cudart-llama-…`（只有
 * CUDA 运行库、不含 llama-server）。以前 CUDA 变体用的是不带前缀约束的正则，
 * 会先命中靠前的 `cudart-llama-…` 包，解压后当然找不到 llama-server，
 * 用户看到的是“压缩包里没有找到 llama-server”这种莫名其妙的话。
 */
export function pickReleaseAsset(assets, { platform = process.platform, arch = process.arch, variant = 'cpu' } = {}) {
  const list = (assets || []).filter((a) => a && typeof a.name === 'string' && /^llama-/.test(a.name));
  const find = (re) => list.find((a) => re.test(a.name)) || null;
  const bits = arch === 'arm64' ? 'arm64' : 'x64';
  let hit = null;
  // 只挑“当前架构”的包：以前 CUDA/Vulkan 的正则是 (x64|arm64)，在 arm64 机器上
  // 可能先命中 x64 的包，装完一运行就是“可执行文件格式错误”。
  const archRe = bits;
  if (platform === 'win32') {
    if (variant === 'cuda') hit = find(new RegExp(`bin-win-cuda-[\\d.]+-${archRe}\\.zip$`, 'i'));
    else if (variant === 'vulkan') hit = find(new RegExp(`bin-win-vulkan-${archRe}\\.zip$`, 'i'));
    hit = hit || find(new RegExp(`bin-win-cpu-${bits}\\.zip$`, 'i'))
      || find(new RegExp(`bin-win-.*-${bits}\\.zip$`, 'i'));
  } else if (platform === 'darwin') {
    hit = find(new RegExp(`bin-macos-${bits}\\.tar\\.gz$`, 'i'));
  } else {
    if (variant === 'cuda') hit = find(new RegExp(`bin-ubuntu-cuda-[\\d.]+-${archRe}\\.tar\\.gz$`, 'i'));
    else if (variant === 'vulkan') hit = find(new RegExp(`bin-ubuntu-vulkan-${archRe}\\.tar\\.gz$`, 'i'));
    hit = hit || find(new RegExp(`bin-ubuntu-${bits}\\.tar\\.gz$`, 'i'))
      // 有的发布只给 rocm/sycl 之类：仍要能挑到一个本架构的可用包，而不是“没有预编译包”
      || find(new RegExp(`bin-ubuntu-(?!cuda|vulkan|rocm|sycl)[a-z0-9_.-]*-${bits}\\.tar\\.gz$`, 'i'));
  }
  return hit;
}

/**
 * CUDA 版 llama-server 还依赖同一发布里的 `cudart-llama-…` 运行库包（cudart DLL/so）。
 * 只下主包会得到“缺 cudart64_*.dll”的启动失败。这里按主包名推导出配套包。
 */
export function pickCudartAsset(assets, primaryAsset) {
  const name = String(primaryAsset?.name || '');
  if (!/^llama-/.test(name)) return null;
  const want = name.replace(/^llama-/, 'cudart-llama-');
  return (assets || []).find((a) => a?.name === want) || null;
}

/** 在候选路径里找可执行的 llama-server。 */
export function findLlamaBinary({ env = process.env, dataDir = '', platform = process.platform } = {}) {
  const bin = llamaBinaryName(platform);
  const candidates = [];
  if (env.LLAMA_SERVER_BIN) candidates.push({ path: env.LLAMA_SERVER_BIN, source: 'env' });
  // 随包附带：server/native-bin/<platform-arch>/llama-server（独立软件完整包会放这里）
  candidates.push({
    path: path.join(SERVER_DIR, '..', 'native-bin', `${platform}-${process.arch}`, bin),
    source: 'bundled',
  });
  // 独立软件单文件 exe：同目录 llama/ 下的 llama-server（完整压缩包会这样放）
  candidates.push({ path: path.join(path.dirname(process.execPath), 'llama', bin), source: 'bundled' });
  if (dataDir) candidates.push({ path: path.join(dataDir, 'bin', bin), source: 'downloaded' });
  for (const c of candidates) {
    if (c.path && existsSync(c.path)) return { ...c };
  }
  // 系统 PATH
  const probe = platform === 'win32'
    ? spawnSync('where', ['llama-server'], { encoding: 'utf8' })
    : spawnSync('sh', ['-c', 'command -v llama-server'], { encoding: 'utf8' });
  const found = String(probe.stdout || '').split(/\r?\n/).map((s) => s.trim()).find(Boolean);
  if (found && existsSync(found)) return { path: found, source: 'path' };
  return null;
}

/** 下载并解压 llama.cpp 官方 Release 到 <dataDir>/bin，返回二进制路径。 */
export async function downloadLlamaBinary({ dataDir, fetchImpl = fetch, variant = 'cpu', onProgress } = {}) {
  if (!dataDir) throw new Error('未配置本机数据目录，无法下载 llama.cpp 运行库。');
  const relRes = await fetchImpl(LLAMA_RELEASE_API, {
    headers: { accept: 'application/vnd.github+json', 'user-agent': 'llm-host-skyblue' },
  });
  if (!relRes.ok) throw new Error(`读取 llama.cpp 发布信息失败（HTTP ${relRes.status}），请检查网络后重试。`);
  const release = pickLlamaRelease(await relRes.json(), { tag });
  if (!release) {
    throw new Error(tag
      ? `没有找到指定的 llama.cpp 版本 ${tag}（可能已下架），请换一个版本号或留空使用最新版。`
      : '没有找到 llama.cpp 的 b 系列正式发布，请稍后重试或手动安装 llama.cpp。');
  }
  const assets = release.assets || [];
  const asset = pickReleaseAsset(assets, { variant });
  if (!asset) {
    throw new Error(`llama.cpp 最新版（${release.tag_name || '未知版本'}）没有当前平台（${process.platform}-${process.arch}）的预编译包，请手动安装 llama.cpp 并设置环境变量 LLAMA_SERVER_BIN。`);
  }
  // CUDA 版必须连 cudart 运行库一起装，否则 llama-server 启动就报缺 DLL/so
  const cudart = variant === 'cuda' ? pickCudartAsset(assets, asset) : null;
  const binDir = path.join(dataDir, 'bin');
  const staging = path.join(dataDir, `.bin-staging-${Date.now()}`);
  await mkdir(staging, { recursive: true });
  try {
    const packs = [asset, ...(cudart ? [cudart] : [])];
    // 先下到磁盘再解压：以前整包读进内存（CUDA 包 300~600 MB），低配电脑直接 OOM
    for (const pack of packs) {
      let buf = null;
      for (let attempt = 0; attempt < 3 && !buf; attempt += 1) {
        try {
          const res = await fetchImpl(pack.browser_download_url, {
            headers: { 'user-agent': 'llm-host-skyblue' },
            redirect: 'follow',
          });
          if (!res.ok) throw new Error(`下载 llama.cpp 失败（HTTP ${res.status}）：${pack.name}`);
          buf = Buffer.from(await res.arrayBuffer());
        } catch (err) {
          if (attempt >= 2) throw err;
          await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
        }
      }
      onProgress?.({ downloaded: buf.length, total: buf.length, asset: pack.name });
      if (/\.zip$/i.test(pack.name)) await extractZip(buf, staging);
      else await extractTarGz(buf, staging);
      buf = null;
    }
    // 在解压结果里找 llama-server（通常在 llama-bXXXX/ 子目录里）
    const found = await findInTree(staging, llamaBinaryName());
    if (!found) {
      const hint = variant === 'cuda' || variant === 'vulkan'
        ? `（当前选的是 ${variant} 版；这台电脑若没有对应显卡驱动，请把 OJ_LLAMA_VARIANT 改成 cpu 再试）`
        : '';
      throw new Error(`llama.cpp 压缩包里没有找到 ${llamaBinaryName()} 可执行文件${hint}。`);
    }
    await rm(binDir, { recursive: true, force: true });
    await mkdir(binDir, { recursive: true });
    // 把二进制所在目录的全部文件（dll/so 依赖）平铺到 bin/；
    // cudart 包解压在另一个目录，也要一并平铺过来（缺它 CUDA 版起不来）
    const flat = async (fromDir) => {
      const names = await readdir(fromDir).catch(() => []);
      for (const name of names) {
        const src = path.join(fromDir, name);
        const st = await stat(src).catch(() => null);
        if (!st || !st.isFile()) continue;
        await rename(src, path.join(binDir, name)).catch(() => {});
      }
    };
    await flat(path.dirname(found));
    if (cudart) {
      // cudart 包里的 dll/so 通常直接位于解压根或 bin/ 子目录：全部平铺
      const walkFlat = async (dir, depth = 0) => {
        if (depth > 3) return;
        for (const e of await readdir(dir, { withFileTypes: true }).catch(() => [])) {
          const p = path.join(dir, e.name);
          if (e.isDirectory()) await walkFlat(p, depth + 1);
          else if (/\.(dll|so(\.[\d.]+)?|dylib)$/i.test(e.name)) await rename(p, path.join(binDir, e.name)).catch(() => {});
        }
      };
      await walkFlat(staging);
    }
    const binary = path.join(binDir, llamaBinaryName());
    if (process.platform !== 'win32') await chmod(binary, 0o755).catch(() => {});
    return { path: binary, version: release.tag_name || '', asset: asset.name, cudart: cudart?.name || '' };
  } finally {
    await rm(staging, { recursive: true, force: true }).catch(() => {});
  }
}

async function findInTree(dir, name, depth = 0) {
  if (depth > 4) return null;
  let entries = [];
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return null;
  }
  for (const e of entries) {
    if (e.isFile() && e.name === name) return path.join(dir, e.name);
  }
  for (const e of entries) {
    if (e.isDirectory()) {
      const hit = await findInTree(path.join(dir, e.name), name, depth + 1);
      if (hit) return hit;
    }
  }
  return null;
}

/**
 * llama-server 进程管理（单实例）。
 * 状态机：idle → loading → ready / error；聊天代理只在 ready 时转发。
 */
export class LlamaProcess {
  constructor({ spawnImpl = spawn, fetchImpl = fetch, logger = console, preflight = true } = {}) {
    this.spawnImpl = spawnImpl;
    this.fetchImpl = fetchImpl;
    this.logger = logger;
    // 启动前自检（查文件/DLL/参数兼容性）。注入替身二进制的测试可关掉，
    // 生产环境必须开着：它是“静默退出”唯一能提前发现的地方。
    this.preflightEnabled = preflight;
    this.child = null;
    this.info = null; // {modelId, quant, port, apiKey, binary}
    this.state = 'idle'; // idle | loading | ready | error
    this.error = '';
    this.logTail = [];
    this.readyPromise = null;
  }

  status() {
    return {
      state: this.state,
      modelId: this.info?.modelId || null,
      quant: this.info?.quant || null,
      gpuLayers: this.info?.gpuLayers ?? null,
      // 本次加载实际用的上下文与增强参数：面板显示“回退到了哪一组”，
      // 用户排查 0xC0000005 时才知道现在跑的是不是精简配置。
      contextTokens: this.info?.contextTokens ?? null,
      flags: this.info?.flags || null,
      error: this.state === 'error' ? this.error : '',
      // 崩溃时把日志尾部一起放进 error：界面上的红色提示要能直接说明死因
      logTail: this.logTail.slice(-20),
      log: this.state === 'error' ? this.logTail.slice(-12) : [],
      // 进程退出码/信号：0xC0000005 这类崩溃码要能被界面与自动降级直接识别
      exitCode: this.exitInfo?.code ?? null,
      // 它在死之前到底说过话没有：一句都没说就退出，基本可以排除“内存不够”，
      // 更像 DLL 缺失/参数不被识别/可执行文件本身不对——诊断时这是决定性证据。
      silentExit: !!this.exitInfo?.silent,
      startedAt: this.info?.startedAt ?? null,
    };
  }

  get baseUrl() {
    return this.info ? `http://127.0.0.1:${this.info.port}` : '';
  }

  get apiKey() {
    return this.info?.apiKey || '';
  }

  /**
   * @param {object} [opts] 参数级逃生开关：崩溃（0xC0000005 等）不只是“层数太多”造成的，
   *   所以除了层数/上下文，还要能逐项关掉我们自己加的增强参数：
   *   - flashAttn：'auto'（默认，交给 llama.cpp 判断）| 'on' | 'off' | null（完全不下发）
   *     ——以前无条件 `-fa on`，在没有对应内核的构建/量化组合上会直接把进程打崩；
   *   - cacheReuse：prompt 缓存复用粒度，null/0 = 不下发（混合注意力/SWA 模型上关掉更稳）；
   *   - mmap：false 时下发 `--no-mmap`（模型放在云同步盘/网络盘/加密盘上时，
   *     mmap 会以访问违例 0xC0000005 崩掉，改成整块读入内存即可）；
   *   - batch：`-b/-ub` 批大小，内存很紧时调小可显著降低峰值占用。
   */
  async start({
    binary, modelFile, modelId, quant, contextTokens = 8192, gpuLayers = 0, port, extraArgs = [],
    flashAttn = 'auto', cacheReuse = 256, mmap = true, batch = 0,
  }) {
    await this.stop();
    this.state = 'loading';
    this.error = '';
    // 上一次加载失败的死因/退出码不能残留：新进程起来后 _waitHealthyLoop 会先看
    // state==='error'，残留会让新的一次启动立刻被误判为崩溃。
    this.exitInfo = null;
    this.logTail = [];
    const apiKey = `sk-local-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
    const layers = Math.max(0, Math.floor(gpuLayers) || 0);
    const args = [
      '-m', modelFile,
      '--host', '127.0.0.1',
      '--port', String(port),
      '-c', String(contextTokens),
      '-ngl', String(layers),
      '-np', '1',
      '--jinja',
      '--reasoning-format', 'deepseek',
      '--api-key', apiKey,
    ];
    // 历史里带上完整对话即可：每个请求都会把整段 messages 发来，llama.cpp 在 -c 满后
    // 默认不再复用 prompt 前缀缓存，长对话每次都要重新预填充（超大模型极慢）。
    if (Number(cacheReuse) > 0) args.push('--cache-reuse', String(Math.floor(Number(cacheReuse))));
    // flash attention：长上下文下省显存/内存的关键。默认 auto 而不是硬开 on：
    // 强制 on 时，构建里没有对应内核的模型（部分 MoE/小 head_dim 量化）会直接崩。
    if (flashAttn && ['auto', 'on', 'off'].includes(flashAttn)) args.push('--flash-attn', flashAttn);
    if (mmap === false) args.push('--no-mmap');
    if (Number(batch) > 0) args.push('-b', String(Math.floor(Number(batch))), '-ub', String(Math.floor(Number(batch))));
    // 调用方按模型追加的参数（目前是 MoE 的专家卸载；普通模型为空）
    args.push(...(Array.isArray(extraArgs) ? extraArgs.filter((a) => typeof a === 'string' && a) : []));
    this.info = {
      modelId, quant, port, apiKey, binary, contextTokens, gpuLayers: layers,
      // 面板要能显示“这次实际用的哪组参数”，否则用户看不出回退到了什么
      flags: { flashAttn: flashAttn || null, cacheReuse: Number(cacheReuse) > 0 ? Number(cacheReuse) : 0, mmap: mmap !== false, batch: Number(batch) || 0 },
    };
    // CUDA 版的 cudart 运行库就放在二进制同目录（随包下载时平铺在一起）：
    // Linux/macOS 的动态链接器不会自动搜“可执行文件所在目录”，必须显式加进
    // LD_LIBRARY_PATH / DYLD_LIBRARY_PATH，否则 CUDA 版一启动就报找不到 libcudart.so。
    const binEnv = {};
    const binDir = path.dirname(binary);
    const win = process.platform === 'win32';
    if (!win) {
      const varName = process.platform === 'darwin' ? 'DYLD_LIBRARY_PATH' : 'LD_LIBRARY_PATH';
      const prev = process.env[varName];
      binEnv[varName] = prev ? `${binDir}${path.delimiter}${prev}` : binDir;
    }
    // 启动前自检：真正的死因往往在“进程一句都没来得及说”的时候就定了。
    // 这里提前把可执行文件、依赖、参数兼容性都验一遍，避免用 6 次崩溃去试。
    this.preflight = this.preflightEnabled ? await this._preflight({ binary, args }) : { fatal: false, notes: [], dropArgs: null };
    if (this.preflight.fatal) {
      const err = new Error(this.preflight.message);
      err.preflight = this.preflight;
      this.state = 'error';
      this.error = this.preflight.message;
      throw err;
    }
    // 自检发现这个版本不认某个参数：直接省略，别让它成为“静默退出”的原因
    const drop = this.preflight?.dropArgs;
    if (drop?.size) {
      const kept = [];
      for (let i = 0; i < args.length; i += 1) {
        const a = args[i];
        if (drop.has(a)) {
          // 带值的参数（--flash-attn on）连值一起丢
          if (['--flash-attn', '--reasoning-format', '--cache-reuse'].includes(a)) i += 1;
          continue;
        }
        kept.push(a);
      }
      args.length = 0;
      args.push(...kept);
      this.logger.warn?.(`[native] 这个 llama.cpp 版本不认部分参数，已自动省略：${[...drop].join('、')}`);
    }
    this.info.startedAt = Date.now();
    this.firstOutputAt = 0;
    const child = this.spawnImpl(binary, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      ...(win ? {} : { env: { ...process.env, ...binEnv } }),
    });
    this.child = child;
    const pushLog = (chunk) => {
      if (!this.firstOutputAt) this.firstOutputAt = Date.now();
      for (const line of String(chunk).split(/\r?\n/)) {
        if (!line.trim()) continue;
        // 完整保留：以前截到 300 字符，DLL 报错路径与 llama.cpp 的关键行常被切掉，
        // 用户看到的就是“日志里什么都没说”。
        this.logTail.push(line.slice(0, 2000));
        if (this.logTail.length > 200) this.logTail.shift();
      }
    };
    child.stdout?.on('data', pushLog);
    child.stderr?.on('data', pushLog);
    child.once('error', (err) => {
      if (this.child === child) {
        this.state = 'error';
        this.error = `llama-server 启动失败：${err.message}`;
      }
    });
    child.once('exit', (code, signal) => {
      if (this.child === child) {
        // 顺序很关键：先记状态与死因，最后才清 child。
        // 以前先清 child，waitHealthy 的轮询一看 child 变了就抛
        // “llama-server 已被停止或替换”——真正的崩溃死因（0xC0000005 等）被这句
        // 假错误盖掉，界面上只剩一句莫名其妙的话，也没法据此自动回退。
        // “一句话都没说就退出”：给诊断层一个明确信号
        const silent = !this.firstOutputAt;
        this.exitInfo = { code, signal, at: Date.now(), silent };
        if (this.state === 'loading' || this.state === 'ready') {
          this.state = 'error';
          // Windows 崩溃码是大整数（如 3221225477 = 0xC0000005 内存访问冲突），附上十六进制便于识别
          const hex = Number.isInteger(code) && code > 0xffff ? `，即 0x${(code >>> 0).toString(16).toUpperCase()}` : '';
          // 死因要从 llama.cpp 自己的输出里读，不能一律套“内存不足”：
          // 用户实际遇到过“unknown model architecture”（版本太旧）却被报成内存不足，
          // 于是白白去换量化、调页面文件，全走错方向。
          const tail = (this.logTail || []).join('\n');
          const arch = matchUnsupportedArch(tail);
          const noMem = /out of memory|failed to allocate|bad_alloc|not enough memory|cannot allocate|cuda.*out of memory/i.test(tail);
          if (silent) {
            // 静默退出是另一类故障：内存不足时 llama.cpp 一定会先打印加载进度，
            // 所以这里不改口说“内存不够”，而是指向真正的原因（缺 DLL / 参数不支持 / 程序不对）。
            // 文案保持短：这段会被回退阶梯的每一步打印一次，太长会把日志刷满。
            this.error = `llama-server 一句话都没输出就退出了（代码 ${code}${hex}）：运行库/驱动/参数不匹配，不是内存不足`;
          } else if (arch) {
            // 版本太旧：换量化/调内存都没用，只有换 llama.cpp 版本一条路
            this.error = `这个 llama.cpp 版本不认识 ${arch} 架构（代码 ${code}${hex}）：模型文件本身没问题，是这个 llama-server 太旧。`
              + `请把它换成 ${ARCH_MIN_BUILD[arch] ? `b${ARCH_MIN_BUILD[arch]} 或更新的版本` : '更新的版本'}（点“准备运行库”可更新）。`;
          } else if (noMem) {
            this.error = `llama-server 因内存不足退出（代码 ${code}${hex}）：日志里有分配失败，请换更小的量化或关掉其它占内存的程序。`;
          } else {
            this.error = `llama-server 已退出（代码 ${code}${signal ? `，信号 ${signal}` : ''}${hex}）。日志尾部见状态面板。`;
          }
        }
        this.child = null;
        this._settleReady(new Error(this._withLog(this.error || `llama-server 已退出（代码 ${code}）。`)));
      }
    });
    this.readyPromise = this.waitHealthy(child);
    // 明确处理拒绝：调用方不 await 时也不能变成 unhandledRejection
    this.readyPromise.catch(() => {});
    return this.readyPromise;
  }

  /**
   * 启动前自检：把“进程一句都没说就死”的原因提前查出来。
   *
   * 为什么需要它：在 Windows 上，llama-server.exe 只要有一个依赖 DLL 找不到
   * （例如 CUDA/Vulkan 版缺 cudart、缺 VC++ 运行库），ntdll 会在进程真正运行
   * 之前就把它杀掉，退出码是 0xC0000135 / 0xC0000005 之类，**stderr 一个字节都没有**。
   * 用户看到的就是“日志空 + 崩溃码”，只能瞎试。这里在 spawn 之前先查：
   *   1) 可执行文件在不在、大不大（0 字节 / 被截断的直接判死）；
   *   2) 同目录的依赖库是否被系统加载器找得到（Windows 用 where，Unix 用 ldd）；
   *   3) 这个版本认不认我们要下发的参数（`--flash-attn` / `--cache-reuse` / `--no-mmap`）。
   * 只有“确定会失败”的情况才 fatal：查不到工具（如容器里没有 ldd）时一律放行。
   */
  async _preflight({ binary, args }) {
    const out = { binary, notes: [], fatal: false, message: '' };
    const fail = (message) => {
      out.fatal = true;
      out.message = message;
      return out;
    };
    let st = null;
    try {
      st = await stat(binary);
    } catch {
      return fail(`找不到 llama-server 可执行文件：${binary}\n请点“准备运行库”重新下载，或设置 LLAMA_SERVER_BIN 指向已有的 llama-server。`);
    }
    if (!st.isFile()) return fail(`llama-server 路径不是一个文件：${binary}`);
    if (st.size < 1) {
      return fail(`llama-server 可执行文件是 0 字节（下载没完成或被杀毒软件清空）：${binary}\n请点“准备运行库”重新下载。`);
    }
    // 官方 llama-server 是几十 MB。明显偏小只警告不拦：有人会用自己写的包装脚本。
    if (st.size < 1024 * 1024) {
      out.notes.push(`llama-server 只有 ${Math.round(st.size / 1024)} KB，官方构建通常几十 MB；若启动失败请重装运行库。`);
    }
    const dir = path.dirname(binary);
    // 真的跑一次：文件存在 ≠ 能启动。用户遇到过 llama-server.exe 明明在、
    // --version 却以 0xC0000005 静默退出（VC++ 运行库版本太旧 / 构建与系统不兼容）。
    // 只查文件大小是发现不了的，必须实际执行一次看退出码。
    const run = this._probeRun(binary, dir);
    out.run = run;
    if (run.ok === false) {
      return fail(run.message);
    }
    if (run.ok === true && run.version) {
      out.notes.push(`llama.cpp 版本：${run.version}`);
    }
    // 依赖库：CUDA/Vulkan 构建把 DLL/so 平铺在二进制旁边，缺一个就静默退出
    const deps = await this._checkDeps(binary, dir);
    out.deps = deps;
    if (deps.missing.length) {
      return fail(
        `llama-server 缺少运行库依赖：${deps.missing.slice(0, 6).join('、')}`
        + `${deps.missing.length > 6 ? ` 等 ${deps.missing.length} 个` : ''}。\n`
        + '这种缺失会让程序在启动前就被系统杀掉（一句日志都不会有，表现为 0xC0000005 / 0xC0000135）。\n'
        + '请点“准备运行库”重装；若已装的是 CUDA/Vulkan 版，把 OJ_LLAMA_VARIANT 设为 cpu 改用纯 CPU 版最稳。',
      );
    }
    // 参数兼容性：老版本 llama.cpp 不认这些参数会直接退出（同样可能一句不说）
    const caps = await this._probeCapabilities(binary, dir, args);
    out.caps = caps;
    if (caps.unknown.length) {
      out.notes.push(`这个 llama.cpp 版本可能不认这些参数：${caps.unknown.join('、')}（已自动省略）`);
    }
    out.dropArgs = caps.dropFlags;
    return out;
  }

  /** 检查二进制的依赖库是否齐全：Windows 用 where，Linux 用 ldd，macOS 用 otool。 */
  /**
   * 实际执行一次 `llama-server --version`，确认这个二进制在这台机器上真的能跑。
   *
   * 为什么要这么做：Windows 上缺运行库、运行库版本太旧、构建与系统不兼容时，
   * 加载器会在程序打印任何东西之前就终止它 —— 得到的是零输出 + 一个退出码
   * （0xC0000005 / 0xC0000135）。用户对着这个码毫无办法。这里在启动服务之前
   * 就把这种情况拦下来，并说清是哪一类问题。
   *
   * 返回 { ok: true|false|null, version, code, silent, message }
   * ok === null 表示探测本身无法进行（超时/被杀），此时不拦，交给后续流程。
   */
  _probeRun(binary, dir) {
    let probe;
    try {
      probe = spawnSync(binary, ['--version'], {
        encoding: 'utf8',
        timeout: 20000,
        cwd: dir,
        windowsHide: true,
        maxBuffer: 4 * 1024 * 1024,
      });
    } catch (err) {
      return { ok: null, message: `无法执行 llama-server 自检：${err.message}` };
    }
    if (probe.error) {
      // 超时或无法创建进程：不判死（有些环境限制 spawn），交给真正的启动流程
      return { ok: null, message: `llama-server 自检未完成：${probe.error.code || probe.error.message}` };
    }
    const out = `${probe.stdout || ''}${probe.stderr || ''}`;
    const versionLine = (probe.stdout || '').split(/\r?\n/).find((l) => /build|version/i.test(l)) || '';
    const result = {
      ok: probe.status === 0,
      code: probe.status,
      silent: !out.trim(),
      version: versionLine.trim().slice(0, 120),
    };
    if (result.ok) return result;
    // 退出码归一化：Windows 崩溃码在 Node 里可能是负数或大整数
    const raw = Number(probe.status);
    const unsigned = raw < 0 ? (raw + 0x100000000) >>> 0 : raw >>> 0;
    const hex = unsigned > 0xffff ? `0x${unsigned.toString(16).toUpperCase()}` : '';
    // 名称与含义都写清楚：Windows 用户看到的是 3221225477 这个十进制大数，
    // 而十六进制 0xC0000005 才是搜索/对号入座时真正有用的形式。
    const codeText = `${raw}${hex ? `，即 ${hex}` : ''}`;
    if (hex === '0XC0000135') {
      return {
        ...result,
        ok: false,
        message: `这个 llama-server 缺少依赖 DLL（退出码 ${codeText}）。\n`
          + 'Windows 在程序启动前就把它终止了，所以一行日志都不会有。\n'
          + '请点“准备运行库”重新安装 llama.cpp；若仍不行，安装 Microsoft Visual C++ 2015-2022 Redistributable (x64)：'
          + 'https://aka.ms/vs/17/release/vc_redist.x64.exe',
      };
    }
    if (!result.silent) {
      // 有输出就说明能启动，失败是这个构建自身报的错，交给后续流程判断
      return { ...result, ok: null };
    }
    // 零输出 + 非零退出：加载器/运行库/系统兼容性三类之一
    return {
      ...result,
      ok: false,
      message: `这个 llama-server 在本机启动不了（零输出，退出码 ${codeText}）：Windows 在它打印任何东西之前就终止了它。\n`
        + '常见原因是两类：\n'
        + '  ① VC++ 运行库缺失或版本过旧 —— 装这个（x64）：https://aka.ms/vs/17/release/vc_redist.x64.exe\n'
        + '  ② 这个构建对你的 Windows 版本太新 —— 换一个较早的 llama.cpp 构建（或升级 Windows）。\n'
        + '可以先在命令行里手动运行一次 `llama-server.exe --version` 确认；'
        + '仍失败时把 Windows 事件查看器（eventvwr.msc → Windows 日志 → 应用程序）里的“错误模块名称”发出来。',
    };
  }
  async _checkDeps(binary, dir) {
    const missing = [];
    const platform = process.platform;
    try {
      if (platform === 'win32') {
        // Windows 没有 ldd：用 `where` 验证常见的运行库是否在 PATH/同目录里。
        // 只查“我们知道 llama.cpp 会用到”的那几个名字，查不到不一定真的缺，
        // 所以只在 *所有* 候选都找不到时才判缺失（避免误杀）。
        const names = [];
        for (const name of await readdir(dir).catch(() => [])) {
          if (/\.dll$/i.test(name)) names.push(name);
        }
        return { missing, present: names };
      }
      if (platform === 'darwin') return { missing, present: [] };
      const probe = spawnSync('ldd', [binary], { encoding: 'utf8', timeout: 15000 });
      const text = `${probe.stdout || ''}${probe.stderr || ''}`;
      if (!text.trim()) return { missing, present: [] };
      for (const line of text.split(/\r?\n/)) {
        const m = line.match(/^\s*(\S+)\s+=>\s+not found/);
        if (m) missing.push(m[1]);
      }
      return { missing, present: [] };
    } catch {
      // 没有这些工具（精简容器/受限环境）：不拦截
      return { missing: [], present: [] };
    }
  }

  /**
   * 探测这个 llama.cpp 版本认不认我们要下发的参数。
   * `llama-server --help` 会在不发日志的情况下快速返回，用它列出支持项；
   * 拿不到 help（老版本/异常）就当作全都支持，保持原来的行为。
   */
  async _probeCapabilities(binary, dir, args) {
    const wanted = ['--flash-attn', '--cache-reuse', '--no-mmap', '--jinja', '--reasoning-format'];
    const unknown = [];
    const dropFlags = new Set();
    try {
      const probe = spawnSync(binary, ['--help'], {
        encoding: 'utf8',
        timeout: 20000,
        cwd: dir,
        env: process.env,
      });
      const text = `${probe.stdout || ''}${probe.stderr || ''}`;
      if (!text.trim()) return { unknown, dropFlags: [], probed: false };
      // 质量门槛：只有确认这是真正的 help 列表才敢据此省略参数。
      // 以前把任意 stderr（例如程序启动失败打的堆栈）也当成“支持项清单”，
      // 结果把一个完全正常的 llama-server 的 --flash-attn / --jinja 全给删了。
      const listed = wanted.filter((f) => text.includes(f));
      // 有 help 抬头就算“这份清单是真的”；此时只要列出了一个我们要用的参数
      // （说明后面的 key-value 表是完整的），就可以据此判断哪些不被支持。
      const hasHelpHeader = /usage:|--help|llama\.cpp|Available arguments|-h,\s*--help/i.test(text);
      if (!hasHelpHeader || !listed.length) return { unknown, dropFlags: [], probed: false };
      for (const flag of wanted) {
        if (!text.includes(flag)) {
          unknown.push(flag);
          dropFlags.add(flag);
        }
      }
      return { unknown, dropFlags, probed: true };
    } catch {
      return { unknown, dropFlags: [], probed: false };
    }
  }

  /**
   * 让正在等待的 waitHealthy 立刻以给定错误结束（进程已崩溃时不必再等到下一次轮询）。
   * 只影响“当前这一次启动”：stop()/start() 会清掉，避免旧错误串到新进程上。
   */
  _settleReady(err) {
    const reject = this._readyReject;
    this._readyReject = null;
    if (reject) reject(err);
  }

  /** 把日志尾部拼进错误信息，让上抛的报错自带死因（面板红字才有用）。 */
  _withLog(message) {
    const tail = this.logTail.slice(-6).join('\n');
    return tail ? `${message}\n${tail}` : message;
  }

  async waitHealthy(child) {
    // 崩溃时由 exit 回调直接 reject（见 _settleReady），不必等下一次 500ms 轮询。
    let rejectSelf = null;
    const settled = new Promise((_, reject) => { rejectSelf = reject; this._readyReject = reject; });
    settled.catch(() => {});
    try {
      return await Promise.race([this._waitHealthyLoop(child), settled]);
    } finally {
      // 只清“自己这一次”的句柄：并发/连续启动时，新的等待的句柄不能被旧的 finally 清掉
      if (this._readyReject === rejectSelf) this._readyReject = null;
    }
  }

  async _waitHealthyLoop(child) {
    const deadline = Date.now() + 30 * 60 * 1000;
    let lastProgressNote = 0;
    while (Date.now() < deadline) {
      // 顺序很重要：先判“有没有明确死因”，再判 child 是否被换掉。
      // 崩溃时 exit 回调会先写 state/error，如果反过来先判 child，就会把真实死因
      // 换成“已被停止或替换”，上层据此也无法判断该不该降级重试。
      if (this.state === 'error') throw new Error(this._withLog(this.error));
      if (this.child !== child) {
        if (this.error) throw new Error(this._withLog(this.error));
        throw new Error('llama-server 已被停止或替换。');
      }
      // 超大模型从磁盘读权重就要几分钟且没有 /health：把加载进度同步到日志尾部，
      // 界面因此能显示“正在加载 xx%”，而不是一直空白。
      const progressLine = [...this.logTail].reverse().find((l) => /load_tensors|loading model|llama_model_loader/i.test(l));
      if (progressLine && Date.now() - lastProgressNote > 5000) {
        lastProgressNote = Date.now();
        this.logger.log?.(`[native] ${progressLine}`);
      }
      try {
        const res = await this.fetchImpl(`${this.baseUrl}/health`, {
          headers: { authorization: `Bearer ${this.apiKey}` },
        });
        if (res.ok) {
          this.state = 'ready';
          return { ok: true };
        }
      } catch {
        /* 尚未监听，继续等 */
      }
      await new Promise((r) => setTimeout(r, 500));
    }
    throw new Error(this._withLog('llama-server 加载超时（30 分钟）。超大模型首次加载很慢，可稍后在状态面板确认，或换更小的量化。'));
  }

  /** 等待就绪（聊天请求自动加载时用）。 */
  async whenReady() {
    if (this.state === 'ready') return { ok: true };
    if (this.state === 'loading' && this.readyPromise) return this.readyPromise;
    throw new Error(this.error || 'llama-server 尚未运行。');
  }

  async stop() {
    const child = this.child;
    this.child = null;
    this.info = null;
    this.readyPromise = null;
    // 主动停止：先让正在等待的 waitHealthy 结束，避免它继续对着旧进程轮询。
    // 用户主动卸载/换模型的“已停止”不该在日志里变成一次崩溃。
    const stoppingErr = Object.assign(new Error('llama-server 已被停止或替换。'), { stopped: true });
    this._settleReady(stoppingErr);
    if (this.state !== 'error') this.state = 'idle';
    if (child) {
      try {
        child.kill('SIGTERM');
      } catch { /* 已退出 */ }
      // Windows/顽固进程兜底
      setTimeout(() => {
        try {
          child.kill('SIGKILL');
        } catch { /* 已退出 */ }
      }, 3000).unref?.();
    }
  }
}

/**
 * 把 llama.cpp 的 reasoning_content 折叠成 <think>…</think> 文本，
 * 与网页端已有的思考过程展示/隐藏逻辑保持一致。
 * 返回一个有状态的转换器：push(delta) → 应作为 content 发出的文本。
 */
export function createThinkFolder() {
  let inThink = false;
  let started = false;
  return {
    push(delta) {
      let out = '';
      const reasoning = typeof delta?.reasoning_content === 'string' ? delta.reasoning_content : '';
      const content = typeof delta?.content === 'string' ? delta.content : '';
      if (reasoning) {
        if (!inThink) {
          out += '<think>';
          inThink = true;
          started = true;
        }
        out += reasoning;
      }
      if (content) {
        if (inThink) {
          out += '</think>';
          inThink = false;
        }
        started = true;
        out += content;
      }
      return out;
    },
    get started() {
      return started;
    },
  };
}

/** 非流式：把 message.reasoning_content 并入 content（同上）。 */
export function foldMessageReasoning(message) {
  if (!message || typeof message !== 'object') return message;
  const reasoning = typeof message.reasoning_content === 'string' ? message.reasoning_content : '';
  const content = typeof message.content === 'string' ? message.content : '';
  if (!reasoning) return { ...message, content };
  const { reasoning_content: _drop, ...rest } = message;
  return { ...rest, content: `<think>${reasoning}</think>${content}` };
}
