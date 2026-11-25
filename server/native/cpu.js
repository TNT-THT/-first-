/**
 * CPU 指令集探测：决定该装哪个 llama.cpp 构建包。
 *
 * 为什么需要它：llama.cpp 的 Windows 构建在 2024 年之前是按指令集分包的 ——
 *   bin-win-avx512-x64.zip / bin-win-avx2-x64.zip / bin-win-avx-x64.zip / bin-win-noavx-x64.zip
 * 之后改成单个 bin-win-cpu-x64.zip，运行时再加载 ggml-cpu-*.dll 分发。
 *
 * 于是装旧版时有个真实的坑：如果随便挑一个包（比如 cuda 版或 avx512 版），
 * 装上一运行就是“非法指令”或直接崩。必须按这台机器**实际支持**的指令集
 * 从高到低挑。
 *
 * 关键设计：**不写任何 CPU 型号表**。
 *   - 型号表永远会过时，而且用户机器千差万别；
 *   - 这里的依据是“一个只用了该指令集的官方二进制能不能跑”——
 *     让 CPU 自己回答，不需要我们猜。
 */
import { spawnSync } from 'node:child_process';
import { readdir } from 'node:fs/promises';
import path from 'node:path';

/** 指令集由高到低。越靠前跑得越快，但对 CPU 要求越高。 */
export const CPU_LEVELS = ['avx512', 'avx2', 'avx', 'noavx'];

/**
 * 一个二进制能不能在这台机器上跑。
 *
 * 判据：退出码为 0 或有正常输出 → 能跑。
 * 反之“非零退出 + 零输出”是 Windows 上“非法指令 / 缺依赖 / 构建不兼容”的典型表现
 * （加载器在 main() 之前就终止进程）。这与 llama.js 里 _probeRun 用的是同一判据。
 *
 * @returns {{ok:boolean, code:number|null, silent:boolean, reason:string}}
 */
export function binaryRuns(binary) {
  let probe;
  try {
    probe = spawnSync(binary, ['--version'], {
      encoding: 'utf8',
      timeout: 20000,
      cwd: path.dirname(binary),
      windowsHide: true,
      maxBuffer: 4 * 1024 * 1024,
    });
  } catch (err) {
    return { ok: false, code: null, silent: true, reason: err.message };
  }
  if (probe.error) {
    return { ok: false, code: null, silent: true, reason: probe.error.code || probe.error.message };
  }
  const out = `${probe.stdout || ''}${probe.stderr || ''}`;
  return {
    ok: probe.status === 0 || !!out.trim(),
    code: probe.status,
    silent: !out.trim(),
    reason: out.trim() ? '' : '零输出退出',
  };
}

/**
 * 从 llama.cpp 的 `--version` 输出里读 CPU 特性行。
 * 新版会打印类似：`CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | ...`
 * 读不到返回 null（不能据此判断，交给实测）。
 */
export function parseCpuFeatures(text) {
  const line = String(text || '').split(/\r?\n/).find((l) => /\bCPU\b.*AVX/i.test(l)) || '';
  if (!line) return null;
  const has = (name) => new RegExp(`\\b${name}\\s*=\\s*1\\b`, 'i').test(line);
  return {
    avx: has('AVX'),
    avx2: has('AVX2'),
    avx512: has('AVX512') || has('AVX512F') || has('AVX512f'),
  };
}

/**
 * 用“官方二进制实测”的方式确定这台机器最高支持到哪一档。
 *
 * 传入的是若干 `{level, path}` 候选（已解压到本地），从高到低逐个试跑：
 * 第一个能跑的就是上限。全部跑不了则返回 'unknown'。
 *
 * 不传候选时退回读 --version 的 CPU 特性行（只能判断 AVX/AVX2，AVX-512 经常不打印）。
 */
export function detectCpuLevel(candidates = []) {
  const sorted = [...candidates].sort(
    (a, b) => CPU_LEVELS.indexOf(a.level) - CPU_LEVELS.indexOf(b.level),
  );
  const results = [];
  for (const c of sorted) {
    if (!c?.path) continue;
    const r = binaryRuns(c.path);
    results.push({ level: c.level, ok: r.ok, code: r.code });
    if (r.ok) return { level: c.level, probed: true, results };
  }
  if (results.length) return { level: 'unknown', probed: true, results };
  return { level: 'unknown', probed: false, results: [] };
}

/**
 * 在已解压的目录树里找 llma.cpp 的各个指令集版本（b4000 那种分发方式）。
 * 目录结构通常是 `<root>/llama-b4000-bin-win-avx2-x64/llama-server.exe`。
 *
 * @returns {Promise<Array<{level:string, path:string}>>}
 */
export async function findCpuVariants(rootDir, binaryName = 'llama-server.exe', depth = 0) {
  const out = [];
  if (depth > 3) return out;
  let entries = [];
  try {
    entries = await readdir(rootDir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    const full = path.join(rootDir, e.name);
    if (e.isDirectory()) {
      // 目录名里带指令集：llama-b4000-bin-win-avx2-x64
      const m = e.name.match(/-(avx512|avx2|avx|noavx)(?:-|$)/i);
      const level = m ? m[1].toLowerCase() : '';
      const bin = path.join(full, binaryName);
      if (level) {
        out.push({ level, path: bin, unverified: true });
      }
      out.push(...await findCpuVariants(full, binaryName, depth + 1));
    }
  }
  return out;
}

/**
 * 给“该下载哪个指令集包”做决策（不依赖任何 CPU 型号表）。
 *
 * 策略：从高到低逐个尝试，**装完立刻实测能不能跑**，跑不了就降一档重来。
 * 这样无论用户是什么 CPU，最终都能落到一个真正能用的版本上。
 *
 * @param {string[]} available 发布里实际存在的指令集档位（如 ['avx512','avx2','avx','noavx']）
 * @param {(level:string) => Promise<boolean>} tryInstall 装上并实测，能跑返回 true
 */
export async function installBestCpuVariant(available, tryInstall) {
  const order = CPU_LEVELS.filter((l) => available.includes(l));
  const tried = [];
  for (const level of order) {
    // eslint-disable-next-line no-await-in-loop
    const ok = await tryInstall(level);
    tried.push({ level, ok });
    if (ok) return { level, ok: true, tried };
  }
  return { level: null, ok: false, tried };
}
