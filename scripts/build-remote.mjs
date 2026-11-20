#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const viteCli = path.join(root, 'node_modules', 'vite', 'bin', 'vite.js');
// OJ_REMOTE_OUT_DIR 可指定输出目录（相对项目根目录）。网站包打包时会构建到
// web/dist-api，与静态版 web/dist 共存；默认仍构建到 web/dist。
const outDir = path.resolve(root, process.env.OJ_REMOTE_OUT_DIR || 'web/dist');
const result = spawnSync(
  process.execPath,
  [viteCli, 'build', '--config', 'web/vite.config.js', '--outDir', outDir],
  {
    cwd: root,
    env: { ...process.env, VITE_ENGINE: 'remote' },
    stdio: 'inherit',
  },
);

if (result.error) {
  console.error(`无法启动 Vite 远程构建：${result.error.message}`);
  process.exitCode = 1;
} else if (result.signal) {
  console.error(`Vite 远程构建被信号 ${result.signal} 终止。`);
  process.exitCode = 1;
} else {
  process.exitCode = result.status ?? 1;
}
