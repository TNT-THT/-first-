#!/usr/bin/env node
import { chmod, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { resolveLocalApiConfig, serializeLocalConfig } from './local-api-config.js';

// 注意：本脚本必须在尚未安装 node_modules 时也能运行（网站包解压后的首次启动），
// 因此不能 import 任何第三方包；读取 .env 用下面这个最小解析器即可。

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const localEnvPath = path.join(root, '.local-api.json');
const baseEnvPath = path.join(root, '.env');

/** 最小 .env 解析（KEY=VALUE、# 注释、可选引号），避免在依赖未安装时就需要 dotenv。 */
function parseEnv(text) {
  const out = {};
  for (const rawLine of String(text).split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const match = /^(?:export\s+)?([A-Za-z_][A-Za-z0-9_.-]*)\s*=\s*(.*)$/.exec(line);
    if (!match) continue;
    let value = match[2].trim();
    const quoted = (value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"));
    if (quoted && value.length >= 2) {
      const doubleQuoted = value.startsWith('"');
      value = value.slice(1, -1);
      if (doubleQuoted) value = value.replace(/\\n/g, '\n').replace(/\\r/g, '\r');
    }
    out[match[1]] = value;
  }
  return out;
}

async function readEnv(filePath) {
  try {
    return parseEnv(await readFile(filePath, 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') return {};
    throw error;
  }
}

async function readLocalConfig(filePath) {
  try {
    const value = JSON.parse(await readFile(filePath, 'utf8'));
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('本机配置必须是 JSON 对象。');
    }
    return value;
  } catch (error) {
    if (error.code === 'ENOENT') return {};
    throw new Error(`无法读取 ${path.basename(filePath)}：${error.message}`);
  }
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      cwd: root,
      stdio: 'inherit',
      shell: process.platform === 'win32',
      ...options,
    });
    child.once('error', reject);
    child.once('exit', (code, signal) => {
      if (signal) return reject(new Error(`${command} 已被信号 ${signal} 停止。`));
      if (code !== 0) return reject(new Error(`${command} 退出码为 ${code ?? 1}。`));
      return resolve();
    });
  });
}

function runServer(env) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['server/index.js'], {
      cwd: root,
      env,
      stdio: 'inherit',
    });
    const stop = (signal) => {
      if (child.exitCode === null) child.kill(signal);
    };
    const onInterrupt = () => stop('SIGINT');
    const onTerminate = () => stop('SIGTERM');
    process.on('SIGINT', onInterrupt);
    process.on('SIGTERM', onTerminate);
    child.once('error', reject);
    child.once('exit', (code, signal) => {
      process.off('SIGINT', onInterrupt);
      process.off('SIGTERM', onTerminate);
      if (signal === 'SIGINT' || signal === 'SIGTERM') return resolve();
      if (signal) return reject(new Error(`本机服务已被信号 ${signal} 停止。`));
      if (code !== 0) return reject(new Error(`本机服务退出码为 ${code ?? 1}。`));
      return resolve();
    });
  });
}

async function saveLocalConfig(values) {
  await writeFile(localEnvPath, serializeLocalConfig(values), { encoding: 'utf8', mode: 0o600 });
  await chmod(localEnvPath, 0o600).catch(() => {});
}

/** 服务就绪后自动打开浏览器（新手无需手动输入网址）；OJ_NO_OPEN_BROWSER=1 时跳过。 */
async function openBrowserWhenReady(port) {
  if (process.env.OJ_NO_OPEN_BROWSER === '1') return;
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 60; i += 1) {
    try {
      const res = await fetch(`${base}/healthz`);
      if (res.ok) break;
    } catch {
      // 服务尚未监听，稍后重试
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  const url = `${base}/`;
  const [command, args] = process.platform === 'win32'
    ? ['cmd', ['/c', 'start', '', url]]
    : process.platform === 'darwin'
      ? ['open', [url]]
      : ['xdg-open', [url]];
  try {
    const child = spawn(command, args, { detached: true, stdio: 'ignore' });
    // spawn 找不到命令（如无桌面环境的 xdg-open）是异步 error 事件：
    // 必须监听，否则未处理的 error 会把整个服务进程带崩
    child.once('error', () => {});
    child.unref();
  } catch {
    // 打不开浏览器不影响服务运行，用户可手动访问打印出的网址
  }
}

async function main() {
  const [baseEnv, existingLocalEnv] = await Promise.all([
    readEnv(baseEnvPath),
    readLocalConfig(localEnvPath),
  ]);
  const local = resolveLocalApiConfig({
    baseEnv,
    localEnv: existingLocalEnv,
    processEnv: process.env,
  });
  await saveLocalConfig(local.localEnv);

  // 网站包已把服务端依赖打入 server/node_modules/，此时无需再联网安装。
  const depsInstalled = existsSync(path.join(root, 'node_modules'))
    || existsSync(path.join(root, 'server', 'node_modules'));
  if (!depsInstalled) {
    console.log('首次运行：正在安装 Node.js 依赖…');
    await run('npm', ['ci']);
  }

  // 网站包内置预构建的服务版网页（web/dist-api）：直接使用，免去在用户电脑上重新构建。
  const usePrebuilt = process.env.OJ_LOCAL_PREBUILT === '1'
    && existsSync(path.join(root, 'web', 'dist-api', 'index.html'));
  if (usePrebuilt) {
    local.env.OJ_WEB_DIST = path.join('web', 'dist-api');
    console.log('\n使用网站包内置的服务版网页（跳过构建）。');
  } else {
    console.log('\n正在构建本机远程服务版…');
    await run('npm', ['run', 'build:remote'], { env: local.env });
  }

  console.log('\n本机 API 服务地址：');
  console.log(`  网页：http://127.0.0.1:${local.port}`);
  console.log(`  API：http://127.0.0.1:${local.port}/v1`);
  console.log('推理方式：本机网页里的 WebGPU 模型（无需 Ollama）。');
  console.log('仅本机可访问。浏览器会自动打开网页（请用 Chrome/Edge 并保持网页打开）；');
  console.log('网页打开后，登录并在“API 接入”里点“申请 API Key”即可使用（首次使用某模型时会下载权重）。按 Ctrl+C 停止服务。');

  if (Object.keys(local.generatedCredentials).length) {
    console.log('\n首次本机运行已生成并保存本机凭据（之后可在 .local-api.json 中查看）：');
    if (local.generatedCredentials.APP_PASSWORD) {
      console.log(`  网站口令 APP_PASSWORD：${local.generatedCredentials.APP_PASSWORD}`);
    }
    if (local.generatedCredentials.API_ADMIN_PASSWORD) {
      console.log(`  管理口令 API_ADMIN_PASSWORD：${local.generatedCredentials.API_ADMIN_PASSWORD}`);
    }
    console.log('请妥善保存；不要分享这些口令或提交 .local-api.json。');
  } else {
    console.log('\n已有凭据未更改。若忘记本机口令，请在 .env 或 .local-api.json 中查看。');
  }

  const serverDone = runServer(local.env);
  void openBrowserWhenReady(local.port);
  await serverDone;
}

main().catch((error) => {
  console.error(`本机 API 启动失败：${error.message}`);
  process.exitCode = 1;
});
