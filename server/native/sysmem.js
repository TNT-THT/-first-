/**
 * 内存上限探测：把“物理内存”和“系统提交上限（页面文件）”分开报告。
 *
 * 为什么需要这个模块：用户问“能不能用硬盘/SSD 当内存”。答案在 Windows 上是
 * “能，但只有一个机制——页面文件（pagefile），而且它慢几十倍”，而那个机制对应的
 * 指标不是物理内存总量，而是**提交限制（Commit Limit）**：
 *
 *   进程要的内存 > 物理内存 但 <= 提交限制  → 可以跑，多出来的部分换页到硬盘（很慢）
 *   进程要的内存 > 提交限制            → 申请内存直接失败（ENOMEM / bad_alloc）
 *
 * 以前代码只读 os.totalmem()/os.freemem()，于是：
 *   - 明明页面文件还留了 20 GB 可以顶，软件却说“必须换量化”；
 *   - 或者反过来说“内存够用”，而实际上是提交限制不够，用户看到的是崩溃。
 *
 * 拿不到时一律返回 null（而不是编一个数字）：宁可少说，不能说错。
 */
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import os from 'node:os';

const MB = 1024 * 1024;

/** 跑一个命令并返回 stdout；失败/不存在时返回 null。 */
function run(cmd, args, opts = {}) {
  try {
    const r = spawnSync(cmd, args, {
      encoding: 'utf8',
      timeout: opts.timeout || 4000,
      windowsHide: true,
      maxBuffer: 4 * MB,
    });
    if (r.error || r.status !== 0 || !r.stdout) return null;
    return r.stdout;
  } catch {
    return null;
  }
}

/**
 * Windows：从 PowerShell 的 Win32_OperatingSystem 读提交上限与页面文件用量。
 * 这是唯一跨 Windows 版本都可靠的来源（性能计数器名字各语言版本不同，不能用）。
 */
function windowsCommit() {
  const script = [
    '$ErrorActionPreference="Stop"',
    'try { $os = Get-CimInstance Win32_OperatingSystem } catch { $os = Get-WmiObject Win32_OperatingSystem }',
    // TotalVirtualMemorySize / FreeVirtualMemory 单位是 KB，语义即“提交限制 / 可提交余量”
    '$o = [ordered]@{',
    '  commitLimitKB = [double]$os.TotalVirtualMemorySize',
    '  commitFreeKB  = [double]$os.FreeVirtualMemory',
    '  physKB        = [double]$os.TotalVisibleMemorySize',
    '  physFreeKB    = [double]$os.FreePhysicalMemory',
    '}',
    '$o | ConvertTo-Json -Compress',
  ].join('; ');
  const out = run('powershell', ['-NoProfile', '-NonInteractive', '-Command', script], { timeout: 8000 });
  if (!out) return null;
  try {
    const j = JSON.parse(out.trim());
    const commitLimitBytes = Math.round((Number(j.commitLimitKB) || 0) * 1024);
    const commitFreeBytes = Math.round((Number(j.commitFreeKB) || 0) * 1024);
    if (!commitLimitBytes) return null;
    return {
      commitLimitBytes,
      commitFreeBytes: commitFreeBytes || null,
      pagefileBytes: null, // 由调用方用“提交上限 - 物理内存”推算，避免再跑一次查询
      source: 'windows-cim',
    };
  } catch {
    return null;
  }
}

/**
 * Linux：/proc/meminfo 的 CommitLimit / Committed_AS 就是提交上限与已提交量。
 * SwapTotal/SwapFree 用来说明“到底有没有可换出的空间”。
 */
function linuxCommit() {
  try {
    // 同步读：这个模块只在算内存体检时调用一次，代价可以忽略
    const txt = readFileSync('/proc/meminfo', 'utf8');
    const kv = (k) => {
      const m = txt.match(new RegExp(`^${k}:\\s+(\\d+)\\s+kB`, 'm'));
      return m ? Number(m[1]) * 1024 : null;
    };
    const commitLimitBytes = kv('CommitLimit');
    const committedBytes = kv('Committed_AS');
    const swapTotalBytes = kv('SwapTotal');
    const swapFreeBytes = kv('SwapFree');
    if (!commitLimitBytes) {
      // 没开 overcommit accounting 的内核没有 CommitLimit：退回 swap 视角
      if (swapTotalBytes === null) return null;
      return {
        commitLimitBytes: (os.totalmem() || 0) + swapTotalBytes || null,
        commitFreeBytes: swapFreeBytes === null ? null : (os.freemem() || 0) + swapFreeBytes,
        pagefileBytes: swapTotalBytes,
        source: 'linux-meminfo-swap',
      };
    }
    const limit = commitLimitBytes + (swapTotalBytes || 0);
    const used = committedBytes || 0;
    return {
      commitLimitBytes: limit,
      commitFreeBytes: Math.max(0, limit - used),
      pagefileBytes: swapTotalBytes ?? Math.max(0, commitLimitBytes - (os.totalmem() || 0)),
      source: 'linux-meminfo',
    };
  } catch {
    return null;
  }
}

/** macOS：sysctl 的 vm.swapusage 给出 swap 总量/已用；没有提交限制的概念。 */
function darwinCommit() {
  const out = run('sysctl', ['-n', 'vm.swapusage']);
  if (!out) return null;
  // 形如：total = 2048.00M  used = 512.00M  free = 1536.00M  (encrypted)
  const num = (label) => {
    const m = out.match(new RegExp(`${label}\\s*=\\s*([\\d.]+)([MG])`, 'i'));
    if (!m) return null;
    const v = Number(m[1]) * (m[2].toUpperCase() === 'G' ? 1e9 : 1e6);
    return Number.isFinite(v) ? Math.round(v) : null;
  };
  const swapTotal = num('total');
  if (!swapTotal) return null;
  const swapFree = num('free');
  return {
    commitLimitBytes: (os.totalmem() || 0) + swapTotal,
    commitFreeBytes: swapFree === null ? null : (os.freemem() || 0) + swapFree,
    pagefileBytes: swapTotal,
    source: 'darwin-swapusage',
  };
}

/**
 * 本机的“能用上限”报告。
 *
 * 返回：
 *   totalBytes        物理内存
 *   freeBytes         当前可用物理内存
 *   commitLimitBytes  提交上限（= 物理内存 + 页面文件/swap 能顶的量）
 *   commitFreeBytes   当前还能提交多少（拿不到为 null）
 *   pagefileBytes     页面文件/swap 总量（0 表示没配，硬盘帮不上忙）
 *   canSpill          是否有“硬盘兜底”可用
 *   source            数据来源，便于排查
 */
export function readMemoryBudget({ forcePlatform = null } = {}) {
  const platform = forcePlatform || process.platform;
  let commit = null;
  try {
    if (platform === 'win32') commit = windowsCommit();
    else if (platform === 'linux') commit = linuxCommit();
    else if (platform === 'darwin') commit = darwinCommit();
  } catch {
    commit = null;
  }
  const totalBytes = Number(os.totalmem?.()) || 0;
  const freeBytes = Number(os.freemem?.()) || 0;
  const commitLimitBytes = Number(commit?.commitLimitBytes) || 0;
  // 页面文件没配时 commitLimit 往往就等于物理内存
  const pagefileBytes = commit?.pagefileBytes != null
    ? Number(commit.pagefileBytes)
    : (commitLimitBytes > totalBytes ? commitLimitBytes - totalBytes : 0);
  return {
    totalBytes,
    freeBytes,
    commitLimitBytes: commitLimitBytes || null,
    commitFreeBytes: commit?.commitFreeBytes ?? null,
    pagefileBytes,
    canSpill: !!(commitLimitBytes && pagefileBytes > 0),
    source: commit?.source || 'os-only',
  };
}

/**
 * 为什么“用 SSD/机械硬盘当内存”不是随便就能开的——给界面/manual 用的一段实话。
 * 这段文字会被原样展示给用户，所以写得尽量具体、可操作。
 */
export const STORAGE_AS_MEMORY_NOTE = [
  '硬盘不能直接当内存用。Windows/Linux 只提供一个机制：页面文件（Windows 的 pagefile.sys / Linux 的 swap），',
  '系统在物理内存不够时把最久没用的内存页写到硬盘上，需要时再读回来。',
  '代价是随机访问速度从内存的约 10 GB/s 掉到 SSD 的约 0.5–3 GB/s、机械硬盘的约 0.1 GB/s，',
  '而大模型加载时恰恰是成千上万次随机读——所以把模型“跑在页面文件上”通常会慢到不可用，甚至因为超时被当成崩溃。',
  '真正有用的做法只有两个：① 把页面文件调大（能让程序不崩，但仍然很慢）；② 换更小的量化，让它真的装进物理内存。',
].join('');

/**
 * 需要多大页面文件才够跑这份权重（Windows 主管可以直接拿去用）。
 *
 * 只有在“物理内存装不下、但硬件本身没坏”时才有意义：它不是让硬盘变成内存，
 * 而是让系统不要在申请内存时就拒绝（否则就是 0xC0000005 / bad_alloc）。
 *
 * @param {object} p
 * @param {number} p.neededGB    这份权重需要的总内存（含 KV 缓存与运行时）
 * @param {number} p.totalGB     物理内存
 * @param {number} p.pagefileGB  现有页面文件
 * @returns {{enough:boolean, requiredCommitGB:number, requiredPagefileGB:number, tip:string}|null}
 */
export function planPagefile({ neededGB, totalGB, pagefileGB = 0 } = {}) {
  const need = Number(neededGB) || 0;
  const total = Number(totalGB) || 0;
  if (!need || !total) return null;
  // 提交上限要留 20% 余量给系统与其它进程，否则刚好卡在边界上照样失败
  const requiredCommitGB = Math.round(need * 1.2 * 10) / 10;
  const requiredPagefileGB = Math.max(0, Math.round((requiredCommitGB - total) * 10) / 10);
  if (requiredPagefileGB <= 0) {
    return {
      enough: true,
      requiredCommitGB,
      requiredPagefileGB: 0,
      tip: '物理内存本身够用，不需要靠页面文件兜底。',
    };
  }
  if (pagefileGB >= requiredPagefileGB) {
    return {
      enough: true,
      requiredCommitGB,
      requiredPagefileGB,
      tip: `现有页面文件 ${pagefileGB} GB 已经够兜住这份权重（需要约 ${requiredPagefileGB} GB），但换页会让生成明显变慢。`,
    };
  }
  return {
    enough: false,
    requiredCommitGB,
    requiredPagefileGB,
    tip: `若一定要在这台机器上跑，需要把页面文件调到约 ${requiredPagefileGB} GB（并留出至少 ${requiredCommitGB} GB 可提交空间）。`
      + '这能让程序不崩，但生成会慢到几乎没法用；更换更小的量化才是正解。',
  };
}
