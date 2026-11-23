/**
 * 软件版的窗口打开策略。
 *
 * 软件本体是本地服务端，界面复用系统浏览器的渲染能力，但呈现方式必须是
 * “软件窗口”而不是“网页”：优先用 Edge/Chrome 的 --app 模式打开独立窗口
 * （没有地址栏和标签页，任务栏里是单独的软件图标）；电脑上没有
 * Chromium 内核浏览器时，才回退到系统默认浏览器打开普通页面。
 *
 * 可用环境变量调整：
 * - OJ_OPEN_MODE=app（默认）|browser|none
 * - OJ_APP_BROWSER=<浏览器可执行文件路径> 显式指定（自动探测失败时用）
 * - OJ_NO_OPEN_BROWSER=1 等价于 OJ_OPEN_MODE=none（兼容旧用法）
 */
import { spawn, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

/** 展开 %VAR%（Windows 风格）为 env 中的值，变量名不区分大小写。 */
export function expandEnvVars(value, env = process.env) {
  return String(value).replace(/%([^%]+)%/g, (_, name) => {
    const hit = Object.keys(env).find((k) => k.toLowerCase() === name.toLowerCase());
    return hit ? env[hit] : '';
  });
}

/** 候选的 Chromium 内核浏览器（找到任何一个即可用 --app 模式）。 */
export function appBrowserCandidates({ platform = process.platform, env = process.env } = {}) {
  if (platform === 'win32') {
    const roots = [
      expandEnvVars('%ProgramFiles(x86)%', env),
      expandEnvVars('%ProgramFiles%', env),
      expandEnvVars('%LOCALAPPDATA%', env),
    ].filter(Boolean);
    const rels = [
      'Microsoft\\Edge\\Application\\msedge.exe',
      'Google\\Chrome\\Application\\chrome.exe',
      'Chromium\\Application\\chrome.exe',
    ];
    const out = [];
    for (const rel of rels) for (const root of roots) out.push(path.join(root, rel));
    return out;
  }
  if (platform === 'darwin') {
    return [
      '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Chromium.app/Contents/MacOS/Chromium',
    ];
  }
  return ['microsoft-edge', 'google-chrome', 'chromium', 'chromium-browser', 'chrome'];
}

/** 找到可用的 Chromium 内核浏览器可执行文件；找不到返回 null。 */
export function findAppBrowser({ platform = process.platform, env = process.env, exists = existsSync, spawnSyncImpl = spawnSync } = {}) {
  if (env.OJ_APP_BROWSER) return env.OJ_APP_BROWSER;
  const candidates = appBrowserCandidates({ platform, env });
  if (platform === 'linux') {
    for (const name of candidates) {
      try {
        const r = spawnSyncImpl('which', [name], { encoding: 'utf8' });
        if (r?.status === 0) return name;
      } catch { /* 继续找下一个 */ }
    }
    return null;
  }
  for (const file of candidates) {
    try {
      if (exists(file)) return file;
    } catch { /* 继续找下一个 */ }
  }
  return null;
}

/** Edge/Chrome 应用窗口参数：无边框浏览器外壳，只剩软件本身。 */
export function appWindowArgs(url, { width = 1280, height = 860 } = {}) {
  return [`--app=${url}`, `--window-size=${width},${height}`, '--no-first-run'];
}

/** 回退方案：用系统默认浏览器打开普通页面。 */
export function defaultBrowserPlan(url, platform = process.platform) {
  if (platform === 'win32') return { kind: 'browser', command: 'cmd', args: ['/c', 'start', '', url] };
  if (platform === 'darwin') return { kind: 'browser', command: 'open', args: [url] };
  return { kind: 'browser', command: 'xdg-open', args: [url] };
}

/** 决定怎么开窗口：none / app（独立软件窗口）/ browser（默认浏览器）。 */
export function pickOpenPlan(url, { mode, platform = process.platform, env = process.env, exists, spawnSyncImpl } = {}) {
  const normalized = String(mode ?? env.OJ_OPEN_MODE ?? '').toLowerCase();
  if (env.OJ_NO_OPEN_BROWSER === '1' || normalized === 'none') return { kind: 'none' };
  if (normalized === 'browser') return defaultBrowserPlan(url, platform);
  const browser = findAppBrowser({ platform, env, exists, spawnSyncImpl });
  if (browser) return { kind: 'app', command: browser, args: appWindowArgs(url) };
  return defaultBrowserPlan(url, platform);
}

function launch(command, args, spawnImpl) {
  const child = spawnImpl(command, args, { detached: true, stdio: 'ignore' });
  child.once?.('error', () => {});
  child.unref?.();
  return child;
}

/**
 * 按策略打开窗口；永不抛错（开窗失败不影响服务本身）。
 * 返回实际采用的方案，方便日志与测试。
 */
export function openAppWindow(url, opts = {}) {
  const spawnImpl = opts.spawnImpl || spawn;
  const plan = pickOpenPlan(url, opts);
  if (plan.kind === 'none') return plan;
  try {
    launch(plan.command, plan.args, spawnImpl);
    return plan;
  } catch {
    if (plan.kind === 'app') {
      const fallback = defaultBrowserPlan(url, opts.platform || process.platform);
      try {
        launch(fallback.command, fallback.args, spawnImpl);
        return { ...fallback, fallback: true };
      } catch { /* 保持安静：用户可手动打开地址 */ }
    }
    return plan;
  }
}
