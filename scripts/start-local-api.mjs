#!/usr/bin/env node
import { chmod, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'dotenv';
import { resolveLocalApiConfig, serializeLocalConfig } from './local-api-config.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const localEnvPath = path.join(root, '.local-api.json');
const baseEnvPath = path.join(root, '.env');

async function readEnv(filePath) {
  try {
    return parse(await readFile(filePath, 'utf8'));
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

  if (!existsSync(path.join(root, 'node_modules'))) {
    console.log('首次运行：正在安装 Node.js 依赖…');
    await run('npm', ['ci']);
  }

  console.log('\n正在构建本机远程服务版…');
  await run('npm', ['run', 'build:remote'], { env: local.env });

  console.log('\n本机 API 服务地址：');
  console.log(`  网页：http://127.0.0.1:${local.port}`);
  console.log(`  API：http://127.0.0.1:${local.port}/v1`);
  console.log(`  模型上游：${local.llmBaseUrl}`);
  console.log(`  模型名称：${local.llmModel}`);
  console.log('仅本机可访问。保持此终端运行，并在同一台电脑打开网页；按 Ctrl+C 停止服务。');

  if (/^https?:\/\/(?:127\.0\.0\.1|localhost):11434(?:\/|$)/i.test(local.llmBaseUrl)) {
    console.log(`\n本机模型提示：请先启动 Ollama，并确保已安装模型 ${local.llmModel}。首次下载可运行：ollama pull ${local.llmModel}`);
    console.log('网页浏览器内置的 WebLLM 与此服务器 API 是两个独立的推理引擎；本机 API 默认通过 Ollama 生成回答。');
  }

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

  await runServer(local.env);
}

main().catch((error) => {
  console.error(`本机 API 启动失败：${error.message}`);
  process.exitCode = 1;
});
