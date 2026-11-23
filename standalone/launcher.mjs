#!/usr/bin/env node
/**
 * 独立软件「LLM托管平台——天蓝色1.0」启动器。
 *
 * 双击 exe（或 `node standalone/launcher.mjs`）后：
 * 1. 准备数据目录（凭据、模型权重、API Key、用量统计都放这里，不写安装目录）；
 * 2. 单文件 exe 模式下把内置网页解压到数据目录（版本变化时自动更新）；
 * 3. 启动与网站包同一套服务端（网页聊天小模型桥接 + llama.cpp 本机大模型引擎
 *    + OpenAI 兼容 /v1 API + 管理员面板与用量统计），功能与原网站包完全一致；
 * 4. 自动打开浏览器。关闭窗口即停止服务。
 *
 * 端口默认 3210（被占用时自动顺延）；OJ_PORT / PORT 可指定，OJ_NO_OPEN_BROWSER=1
 * 时不自动打开浏览器，OJ_DATA_DIR 可指定数据目录。
 */
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import os from 'node:os';
import path from 'node:path';
import { randomBytes } from 'node:crypto';

// 必须在加载服务端模块之前设置：内嵌模式下 server/index.js 不自动监听，由本启动器接管。
process.env.OJ_EMBEDDED_SERVER = '1';

const PRODUCT = 'LLM托管平台——天蓝色1.0';

function seaAssets() {
  try {
    // SEA 里没有 import.meta / __filename；用 execPath 作锚点即可 require 内置模块。
    const require = createRequire(path.resolve(process.execPath));
    const sea = require('node:sea');
    if (sea?.isSea()) return sea;
  } catch {
    /* 非单文件模式（node 直接运行） */
  }
  return null;
}

/** 数据目录：优先程序同目录（便携），不可写时退回用户目录。 */
async function resolveDataDir() {
  if (process.env.OJ_DATA_DIR) return path.resolve(process.env.OJ_DATA_DIR);
  const beside = path.join(path.dirname(process.execPath), 'LLMHost-SkyBlue-Data');
  try {
    await mkdir(beside, { recursive: true });
    const probe = path.join(beside, '.write-probe');
    await writeFile(probe, 'ok');
    await rm(probe, { force: true });
    return beside;
  } catch {
    const base = process.env.LOCALAPPDATA || process.env.APPDATA || path.join(os.homedir(), '.local', 'share');
    return path.join(base, 'LLMHost-SkyBlue');
  }
}

/** 读取或生成本机凭据（只保存在数据目录，首次生成时打印一次）。 */
async function loadCredentials(dataDir) {
  const file = path.join(dataDir, 'credentials.json');
  try {
    const data = JSON.parse(await readFile(file, 'utf8'));
    if (data.APP_PASSWORD && data.API_ADMIN_PASSWORD && data.SESSION_SECRET) return { data, created: false };
  } catch {
    /* 首次运行或文件损坏：重新生成 */
  }
  const data = {
    APP_PASSWORD: randomBytes(9).toString('base64url'),
    API_ADMIN_PASSWORD: randomBytes(15).toString('base64url'),
    SESSION_SECRET: randomBytes(32).toString('hex'),
  };
  await writeFile(file, `${JSON.stringify(data, null, 2)}\n`, { mode: 0o600 });
  return { data, created: true };
}

/** 单文件模式：把内置网页（payload.zip）解压到 <dataDir>/app；返回应用根目录。 */
async function prepareAppRoot(dataDir, sea) {
  if (!sea) {
    // 开发/解压模式：直接使用仓库（或解压包）里的文件。
    // 不能依赖 import.meta/__filename：单文件打包成 CJS 后它们不存在；
    // 用启动脚本路径定位（node standalone/launcher.mjs 或 app/standalone/launcher.mjs）。
    const launcherPath = path.resolve(process.argv[1] || process.execPath);
    return path.resolve(path.dirname(launcherPath), '..');
  }
  const appDir = path.join(dataDir, 'app');
  const version = Buffer.from(sea.getAsset('version.txt')).toString('utf8').trim();
  const marker = path.join(appDir, '.payload-version');
  let current = '';
  try {
    current = (await readFile(marker, 'utf8')).trim();
  } catch {
    current = '';
  }
  if (current !== version || !existsSync(path.join(appDir, 'web', 'dist-api', 'index.html'))) {
    const { extractZip } = await import('../server/native/archive.js');
    const zip = Buffer.from(sea.getAsset('payload.zip'));
    await rm(appDir, { recursive: true, force: true });
    await mkdir(appDir, { recursive: true });
    await extractZip(zip, appDir);
    await writeFile(marker, `${version}\n`);
    console.log(`[standalone] 已准备内置网页（版本 ${version}）。`);
  }
  return appDir;
}

async function openBrowser(url) {
  if (process.env.OJ_NO_OPEN_BROWSER === '1') return;
  const [command, args] = process.platform === 'win32'
    ? ['cmd', ['/c', 'start', '', url]]
    : process.platform === 'darwin'
      ? ['open', [url]]
      : ['xdg-open', [url]];
  try {
    const child = spawn(command, args, { detached: true, stdio: 'ignore' });
    child.once('error', () => {});
    child.unref();
  } catch {
    /* 打不开浏览器不影响服务，用户可手动访问 */
  }
}

async function main() {
  const { createApp } = await import('../server/index.js');
  const { loadConfig } = await import('../server/config.js');
  const { BrowserModelBridge } = await import('../server/browser-model-bridge.js');
  const { NativeEngine } = await import('../server/native/engine.js');
  const { createServer } = await import('node:http');

  const sea = seaAssets();
  const dataDir = await resolveDataDir();
  await mkdir(dataDir, { recursive: true });
  const appRoot = await prepareAppRoot(dataDir, sea);
  const { data: credentials, created } = await loadCredentials(dataDir);

  const startPort = Number(process.env.OJ_PORT || process.env.PORT || 3210);
  const env = {
    ...process.env,
    HOST: process.env.OJ_HOST || '127.0.0.1',
    PORT: String(startPort),
    APP_PASSWORD: credentials.APP_PASSWORD,
    API_ADMIN_PASSWORD: credentials.API_ADMIN_PASSWORD,
    SESSION_SECRET: credentials.SESSION_SECRET,
    OJ_WEB_DIST: path.join(appRoot, 'web', 'dist-api'),
    OJ_NATIVE_DATA_DIR: path.join(dataDir, 'native'),
    API_KEYS_FILE: path.join(dataDir, 'api-keys.json'),
    API_SETTINGS_FILE: path.join(dataDir, 'api-settings.json'),
    USAGE_STATS_FILE: path.join(dataDir, 'usage-stats.json'),
  };
  const cfg = loadConfig(env);
  const browserBridge = new BrowserModelBridge();
  const nativeEngine = new NativeEngine({ dataDir: cfg.nativeDataDir, variant: cfg.llamaVariant });
  const server = createServer(createApp(cfg, { browserBridge, nativeEngine }));
  browserBridge.attach(server);

  const port = await new Promise((resolve, reject) => {
    let attempt = startPort;
    const tryListen = () => {
      server.once('error', (err) => {
        if (err.code === 'EADDRINUSE' && attempt < startPort + 20) {
          attempt += 1;
          tryListen();
        } else reject(err);
      });
      server.listen(attempt, cfg.host, () => resolve(attempt));
    };
    tryListen();
  });

  const url = `http://127.0.0.1:${port}/`;
  console.log('');
  console.log(`${PRODUCT} 已启动：${url}`);
  console.log('  网页聊天：浏览器打开上面的地址，用下面的“网站口令”登录；小模型在浏览器里运行，大模型在“API 接入 → 本机大模型”里下载后运行。');
  console.log(`  API：http://127.0.0.1:${port}/v1（OpenAI 兼容，登录后在“API 接入”里申请 Key）`);
  console.log(`  网站口令：${credentials.APP_PASSWORD}`);
  console.log(`  管理口令：${credentials.API_ADMIN_PASSWORD}（API 管理/大模型下载加载用，验证一次即可）`);
  console.log(`  数据目录：${dataDir}（凭据 credentials.json、模型权重 native/ 都在这里）`);
  if (created) console.log('  以上口令为首次运行自动生成并已保存，之后启动不会变化。');
  console.log('  关闭本窗口即停止服务。');
  await openBrowser(url);

  const shutdown = async () => {
    await nativeEngine.unload().catch(() => {});
    server.close(() => process.exit(0));
    setTimeout(() => process.exit(0), 3000).unref();
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((err) => {
  console.error(`${PRODUCT} 启动失败：${err.stack || err.message}`);
  process.exit(1);
});
