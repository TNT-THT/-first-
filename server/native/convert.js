/**
 * Transformers → GGUF 自动转换（llama.cpp 官方 convert_hf_to_gguf.py）。
 *
 * Safetensors / PyTorch 格式的模型先下载快照，再用官方脚本转成 GGUF，
 * 之后与原生 GGUF 模型走同一套 llama.cpp 加载/聊天/API 路线。
 * 转换工具（llama.cpp 源码 + Python 虚拟环境 + CPU 版 PyTorch 等）按需
 * 下载到数据目录 tools/ 下，不进安装包；准备一次约数百 MB，只需一次。
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, rm } from 'node:fs/promises';
import path from 'node:path';

export function toolsDir(dataDir) {
  return path.join(dataDir, 'tools');
}

export function convertScriptPath(dataDir, env = process.env) {
  if (env.LLAMA_CONVERT_SCRIPT) return env.LLAMA_CONVERT_SCRIPT;
  return path.join(toolsDir(dataDir), 'llama.cpp', 'convert_hf_to_gguf.py');
}

export function venvPython(dataDir, platform = process.platform) {
  return platform === 'win32'
    ? path.join(toolsDir(dataDir), 'venv', 'Scripts', 'python.exe')
    : path.join(toolsDir(dataDir), 'venv', 'bin', 'python');
}

export function systemPython(env = process.env, platform = process.platform) {
  return env.PYTHON_BIN || (platform === 'win32' ? 'python' : 'python3');
}

/** 转换工具是否就绪：脚本存在 + 虚拟环境 Python 可运行（或显式指定的脚本+系统 Python）。 */
export function findConverter({ dataDir, env = process.env, spawnSyncImpl = spawnSync, platform = process.platform } = {}) {
  const script = convertScriptPath(dataDir, env);
  if (!existsSync(script)) {
    return { ready: false, script, python: '', detail: '尚未准备转换工具（点“一键准备转换工具”自动下载，约数百 MB，只需一次）。' };
  }
  const venvPy = venvPython(dataDir, platform);
  const python = existsSync(venvPy) ? venvPy : systemPython(env, platform);
  try {
    const probe = spawnSyncImpl(python, ['--version'], { encoding: 'utf8', timeout: 10000 });
    if (probe.status !== 0) throw new Error('exit');
  } catch {
    return { ready: false, script, python, detail: `找不到可用的 Python（${python}），请先安装 Python 3.10+ 后重试。` };
  }
  return { ready: true, script, python, detail: '转换工具已就绪。' };
}

/** 默认命令执行器：跑完收集输出尾部，失败抛带尾部的错误。 */
export function runCommand(command, args, { cwd, env = process.env, timeoutMs = 1000 * 60 * 60 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = '';
    const push = (c) => {
      tail = (tail + String(c)).slice(-4000);
    };
    child.stdout?.on('data', push);
    child.stderr?.on('data', push);
    const timer = setTimeout(() => {
      try {
        child.kill('SIGKILL');
      } catch { /* 已退出 */ }
      reject(new Error(`命令超时：${command} ${args.slice(0, 2).join(' ')}…`));
    }, timeoutMs);
    child.once('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`无法运行 ${command}：${err.message}`));
    });
    child.once('exit', (code) => {
      clearTimeout(timer);
      if (code === 0) resolve({ output: tail });
      else reject(new Error(`命令失败（代码 ${code}）：${command} ${args.slice(0, 3).join(' ')}…\n${tail.slice(-800)}`));
    });
  });
}

/**
 * 一键准备转换工具：clone llama.cpp → 建 venv → 装转换依赖（CPU 版 PyTorch）。
 * 已就绪时直接返回。exec 可注入便于测试。
 */
export async function prepareConverter({ dataDir, env = process.env, exec = runCommand, spawnSyncImpl = spawnSync, platform = process.platform } = {}) {
  const before = findConverter({ dataDir, env, spawnSyncImpl, platform });
  if (before.ready) return before;
  const tools = toolsDir(dataDir);
  const repoDir = path.join(tools, 'llama.cpp');
  const script = convertScriptPath(dataDir, env);
  await mkdir(tools, { recursive: true });
  if (!existsSync(script)) {
    if (existsSync(repoDir)) await rm(repoDir, { recursive: true, force: true });
    await exec('git', ['clone', '--depth', '1', 'https://github.com/ggml-org/llama.cpp', repoDir], { timeoutMs: 15 * 60 * 1000 });
  }
  if (!existsSync(script)) throw new Error('llama.cpp 源码里没有找到 convert_hf_to_gguf.py，转换工具准备失败。');
  const venvPy = venvPython(dataDir, platform);
  if (!existsSync(venvPy)) {
    await exec(systemPython(env, platform), ['-m', 'venv', path.join(tools, 'venv')], { timeoutMs: 10 * 60 * 1000 });
  }
  if (!existsSync(venvPy)) throw new Error('Python 虚拟环境创建失败，请确认已安装 Python 3.10+。');
  const req = path.join(repoDir, 'requirements', 'requirements-convert_hf_to_gguf.txt');
  await exec(venvPy, ['-m', 'pip', 'install', '--upgrade', 'pip'], { timeoutMs: 10 * 60 * 1000 });
  await exec(venvPy, ['-m', 'pip', 'install', '-r', req], { timeoutMs: 60 * 60 * 1000 });
  const after = findConverter({ dataDir, env, spawnSyncImpl, platform });
  if (!after.ready) throw new Error(after.detail || '转换工具准备失败。');
  return after;
}

/**
 * 执行一次转换：snapshotDir 是已下载的 HF 仓库快照，输出单个 GGUF 文件。
 * 转换脚本对不支持的架构会非零退出，这里把输出尾部带回给用户看。
 */
export async function runConvert({ dataDir, snapshotDir, outfile, env = process.env, exec = runCommand, spawnSyncImpl = spawnSync, platform = process.platform } = {}) {
  const conv = findConverter({ dataDir, env, spawnSyncImpl, platform });
  if (!conv.ready) throw new Error(conv.detail);
  await mkdir(path.dirname(outfile), { recursive: true });
  await exec(conv.python, [conv.script, snapshotDir, '--outfile', outfile, '--outtype', 'f16'], {
    cwd: path.dirname(conv.script),
    timeoutMs: 6 * 60 * 60 * 1000,
  });
  if (!existsSync(outfile)) throw new Error('转换脚本执行完成但没有生成 GGUF 文件。');
  return { outfile, python: conv.python, script: conv.script };
}
