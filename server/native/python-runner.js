/**
 * 本地 Python 运行时管理（特殊格式原格式本地运行）。
 *
 * 与转换工具共用同一个 Python 虚拟环境（tools/venv，含 torch/transformers），
 * 额外装 optimum[onnxruntime] 后即可跑 ONNX。其余格式（Safetensors/PyTorch/
 * AWQ/GPTQ/BNB）由 transformers 按仓库 config 直接加载。
 *
 * 进程管理与 llama-server 一致：单实例、随机端口 + 随机 api-key、只绑
 * 127.0.0.1；runner.py 提供 /health 与 /v1/chat/completions，Node 侧转发。
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { prepareConverter, runCommand, venvPython } from './convert.js';

// 单文件打包（SEA/CJS）里没有 import.meta.url，退回 __filename（与 llama.js 一致）。
const HERE = import.meta.url && String(import.meta.url).startsWith('file:')
  ? path.dirname(fileURLToPath(import.meta.url))
  : path.resolve(typeof __filename === 'string' ? path.dirname(__filename) : process.cwd());
export const RUNNER_SCRIPT = path.join(HERE, 'python', 'runner.py');

/**
 * 定位 runner.py：开发/源码运行在模块旁；独立软件（SEA）是从启动器解压出的
 * <数据目录>/app/native/runner.py（构建时打进 payload）；可用 OJ_RUNNER_SCRIPT 覆盖。
 */
export function resolveRunnerScript({ dataDir, env = process.env } = {}) {
  const candidates = [];
  if (env.OJ_RUNNER_SCRIPT) candidates.push(path.resolve(env.OJ_RUNNER_SCRIPT));
  candidates.push(RUNNER_SCRIPT);
  if (dataDir) {
    // 引擎数据目录是 <数据根>/native，payload 解压在同级 <数据根>/app；
    // 也兼容直接把数据根传进来的情况。
    candidates.push(path.join(dataDir, '..', 'app', 'native', 'runner.py'));
    candidates.push(path.join(dataDir, 'app', 'native', 'runner.py'));
  }
  candidates.push(path.join(process.cwd(), 'server', 'native', 'python', 'runner.py'));
  for (const c of candidates) if (c && existsSync(c)) return c;
  return candidates[0] || RUNNER_SCRIPT;
}

export function pythonCandidates({ dataDir, env = process.env, platform = process.platform } = {}) {
  const list = [];
  if (env.PYTHON_BIN) list.push(env.PYTHON_BIN);
  if (dataDir) list.push(venvPython(dataDir, platform));
  list.push(platform === 'win32' ? 'python' : 'python3');
  return [...new Set(list)];
}

/** 运行时是否就绪：能找到 Python 且 transformers+torch 可导入（探测结果由调用方缓存）。 */
export function findPythonRuntime({ dataDir, env = process.env, spawnSyncImpl = spawnSync, platform = process.platform } = {}) {
  const script = resolveRunnerScript({ dataDir, env });
  if (!existsSync(script)) return { ready: false, python: '', script, detail: '运行时脚本缺失。' };
  for (const python of pythonCandidates({ dataDir, env, platform })) {
    try {
      const probe = spawnSyncImpl(python, ['-c', 'import transformers, torch'], { encoding: 'utf8', timeout: 30000 });
      if (probe.status === 0) return { ready: true, python, script, detail: 'Python 运行时已就绪。' };
    } catch { /* 试下一个候选 */ }
  }
  return {
    ready: false,
    python: pythonCandidates({ dataDir, env, platform })[0] || '',
    detail: 'Python 运行时未就绪：点“一键准备本地运行时”自动安装（与格式转换共用环境，约数百 MB，只装一次）。',
  };
}

/** 一键准备：先备好转换环境（venv+torch/transformers），再装 ONNX 等本地运行依赖。 */
export async function preparePythonRuntime({ dataDir, env = process.env, exec = runCommand, spawnSyncImpl = spawnSync, platform = process.platform } = {}) {
  await prepareConverter({ dataDir, env, exec, spawnSyncImpl, platform });
  const python = venvPython(dataDir, platform);
  await exec(python, ['-m', 'pip', 'install', 'optimum[onnxruntime]', 'accelerate', 'sentencepiece', 'protobuf'], {
    timeoutMs: 60 * 60 * 1000,
  });
  const after = findPythonRuntime({ dataDir, env, spawnSyncImpl, platform });
  if (!after.ready) throw new Error(after.detail || 'Python 运行时准备失败。');
  return after;
}

/** Python 模型进程（单实例），接口形状与 LlamaProcess 对齐。 */
export class PythonProcess {
  constructor({ spawnImpl = spawn, fetchImpl = fetch } = {}) {
    this.spawnImpl = spawnImpl;
    this.fetchImpl = fetchImpl;
    this.child = null;
    this.info = null;
    this.state = 'idle';
    this.error = '';
    this.logTail = [];
    this.readyPromise = null;
  }

  status() {
    return {
      state: this.state,
      modelId: this.info?.modelId || null,
      quant: this.info?.quant || null,
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

  async start({ python, modelDir, modelId, port, script }) {
    await this.stop();
    this.state = 'loading';
    this.error = '';
    this.logTail = [];
    const apiKey = `sk-local-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
    this.info = { modelId, quant: 'PYTHON', port, apiKey, binary: python };
    const child = this.spawnImpl(python, [script || RUNNER_SCRIPT, '--model-dir', modelDir, '--port', String(port), '--api-key', apiKey], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
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
        this.error = `Python 运行时启动失败：${err.message}`;
      }
    });
    child.once('exit', (code) => {
      if (this.child === child) {
        this.child = null;
        if (this.state === 'loading' || this.state === 'ready') {
          this.state = 'error';
          this.error = `Python 运行时已退出（代码 ${code}）。常见原因：依赖缺失、内存不足或该格式需要 GPU。日志尾部见状态面板。`;
        }
      }
    });
    this.readyPromise = this.waitHealthy(child);
    return this.readyPromise;
  }

  async waitHealthy(child) {
    const deadline = Date.now() + 30 * 60 * 1000;
    while (Date.now() < deadline) {
      if (this.child !== child) throw new Error('Python 运行时已被停止或替换。');
      if (this.state === 'error') throw new Error(this.error);
      try {
        const res = await this.fetchImpl(`${this.baseUrl}/health`, {
          headers: { authorization: `Bearer ${this.apiKey}` },
        });
        if (res.ok) {
          this.state = 'ready';
          return { ok: true };
        }
      } catch { /* 尚未监听或模型加载中 */ }
      await new Promise((r) => setTimeout(r, 500));
    }
    throw new Error('Python 运行时加载超时（30 分钟）。大模型在 CPU 上首次加载很慢，可稍后再试或改用 GGUF 路线。');
  }

  async whenReady() {
    if (this.state === 'ready') return { ok: true };
    if (this.state === 'loading' && this.readyPromise) return this.readyPromise;
    throw new Error(this.error || 'Python 运行时尚未运行。');
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
      setTimeout(() => {
        try {
          child.kill('SIGKILL');
        } catch { /* 已退出 */ }
      }, 3000).unref?.();
    }
  }
}
