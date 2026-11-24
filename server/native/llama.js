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
import { chmod, mkdir, readdir, rename, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { extractTarGz, extractZip } from './archive.js';

// 单文件打包（SEA/CJS）里没有 import.meta.url，退回 __filename（普通路径不能 fileURLToPath）。
const SERVER_DIR = import.meta.url && String(import.meta.url).startsWith('file:')
  ? path.dirname(fileURLToPath(import.meta.url))
  : path.resolve(typeof __filename === 'string' ? path.dirname(__filename) : process.cwd());
export const LLAMA_RELEASE_API = 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10';

/**
 * 从发布列表里挑最新的 b 系列正式构建（/releases/latest 指向的可能不是 b 系列，
 * 直接用它会挑错）。返回 {tag_name, assets} 或 null。
 */
export function pickLlamaRelease(releases) {
  for (const rel of Array.isArray(releases) ? releases : []) {
    if (!/^b\d+$/.test(String(rel?.tag_name || ''))) continue;
    if (Array.isArray(rel.assets) && rel.assets.length) return rel;
  }
  return null;
}

export function llamaBinaryName(platform = process.platform) {
  return platform === 'win32' ? 'llama-server.exe' : 'llama-server';
}

/** 按平台给 llama.cpp Release 资源打分，挑当前平台最合适的一个（CPU 优先于 GPU 变体之外的精确匹配）。 */
export function pickReleaseAsset(assets, { platform = process.platform, arch = process.arch, variant = 'cpu' } = {}) {
  const names = (assets || []).map((a) => a?.name).filter(Boolean);
  const find = (re) => names.find((n) => re.test(n));
  let hit = null;
  if (platform === 'win32') {
    if (variant === 'cuda') hit = find(/bin-win-cuda-[\d.]+-x64\.zip$/i);
    else if (variant === 'vulkan') hit = find(/bin-win-vulkan-x64\.zip$/i);
    hit = hit || find(/bin-win-cpu-x64\.zip$/i) || find(/bin-win-.*x64\.zip$/i);
  } else if (platform === 'darwin') {
    hit = find(new RegExp(`bin-macos-${arch === 'arm64' ? 'arm64' : 'x64'}\\.tar\\.gz$`, 'i'));
  } else {
    hit = find(new RegExp(`bin-ubuntu-${arch === 'arm64' ? 'arm64' : 'x64'}\\.tar\\.gz$`, 'i'));
  }
  if (!hit) return null;
  return (assets || []).find((a) => a.name === hit) || null;
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
  const release = pickLlamaRelease(await relRes.json());
  if (!release) throw new Error('没有找到 llama.cpp 的 b 系列正式发布，请稍后重试或手动安装 llama.cpp。');
  const asset = pickReleaseAsset(release.assets || [], { variant });
  if (!asset) {
    throw new Error(`llama.cpp 最新版（${release.tag_name || '未知版本'}）没有当前平台的预编译包，请手动安装 llama.cpp 并设置 LLAMA_SERVER_BIN。`);
  }
  const binDir = path.join(dataDir, 'bin');
  const staging = path.join(dataDir, `.bin-staging-${Date.now()}`);
  await mkdir(staging, { recursive: true });
  try {
    const res = await fetchImpl(asset.browser_download_url, {
      headers: { 'user-agent': 'llm-host-skyblue' },
      redirect: 'follow',
    });
    if (!res.ok) throw new Error(`下载 llama.cpp 失败（HTTP ${res.status}）。`);
    const buf = Buffer.from(await res.arrayBuffer());
    onProgress?.({ downloaded: buf.length, total: buf.length });
    if (/\.zip$/i.test(asset.name)) await extractZip(buf, staging);
    else await extractTarGz(buf, staging);
    // 在解压结果里找 llama-server（可能在子目录）
    const found = await findInTree(staging, llamaBinaryName());
    if (!found) throw new Error('llama.cpp 压缩包里没有找到 llama-server 可执行文件。');
    await rm(binDir, { recursive: true, force: true });
    await mkdir(binDir, { recursive: true });
    // 把二进制所在目录的全部文件（dll/so 依赖）平铺到 bin/
    const srcDir = path.dirname(found);
    for (const name of await readdir(srcDir)) {
      await rename(path.join(srcDir, name), path.join(binDir, name)).catch(() => {});
    }
    const binary = path.join(binDir, llamaBinaryName());
    if (process.platform !== 'win32') await chmod(binary, 0o755).catch(() => {});
    return { path: binary, version: release.tag_name || '', asset: asset.name };
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
  constructor({ spawnImpl = spawn, fetchImpl = fetch, logger = console } = {}) {
    this.spawnImpl = spawnImpl;
    this.fetchImpl = fetchImpl;
    this.logger = logger;
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
      error: this.state === 'error' ? this.error : '',
      logTail: this.logTail.slice(-20),
    };
  }

  get baseUrl() {
    return this.info ? `http://127.0.0.1:${this.info.port}` : '';
  }

  get apiKey() {
    return this.info?.apiKey || '';
  }

  async start({ binary, modelFile, modelId, quant, contextTokens = 8192, gpuLayers = 0, port }) {
    await this.stop();
    this.state = 'loading';
    this.error = '';
    this.logTail = [];
    const apiKey = `sk-local-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
    const args = [
      '-m', modelFile,
      '--host', '127.0.0.1',
      '--port', String(port),
      '-c', String(contextTokens),
      '-ngl', String(Math.max(0, Math.floor(gpuLayers) || 0)),
      '-np', '1',
      '--jinja',
      '--reasoning-format', 'deepseek',
      '--api-key', apiKey,
    ];
    this.info = { modelId, quant, port, apiKey, binary, gpuLayers: Math.max(0, Math.floor(gpuLayers) || 0) };
    const child = this.spawnImpl(binary, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    this.child = child;
    const pushLog = (chunk) => {
      for (const line of String(chunk).split(/\r?\n/)) {
        if (!line.trim()) continue;
        this.logTail.push(line.slice(0, 300));
        if (this.logTail.length > 60) this.logTail.shift();
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
    child.once('exit', (code) => {
      if (this.child === child) {
        this.child = null;
        if (this.state === 'loading' || this.state === 'ready') {
          this.state = 'error';
          // Windows 崩溃码是大整数（如 3221225477 = 0xC0000005 内存访问冲突），附上十六进制便于识别
          const hex = Number.isInteger(code) && code > 0xffff ? `，即 0x${(code >>> 0).toString(16).toUpperCase()}` : '';
          this.error = `llama-server 已退出（代码 ${code}${hex}）。常见原因：内存不足、权重文件损坏或显卡驱动不支持。日志尾部见状态面板。`;
        }
      }
    });
    this.readyPromise = this.waitHealthy(child);
    return this.readyPromise;
  }

  async waitHealthy(child) {
    const deadline = Date.now() + 30 * 60 * 1000;
    while (Date.now() < deadline) {
      if (this.child !== child) throw new Error('llama-server 已被停止或替换。');
      if (this.state === 'error') throw new Error(this.error);
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
    throw new Error('llama-server 加载超时（30 分钟）。超大模型首次加载很慢，可稍后在状态面板确认，或换更小的量化。');
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
    if (this.state !== 'error') this.state = 'idle';
    this.readyPromise = null;
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
