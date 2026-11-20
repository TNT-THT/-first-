#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const viteCli = path.join(root, 'node_modules', 'vite', 'bin', 'vite.js');
const result = spawnSync(
  process.execPath,
  [viteCli, 'build', '--config', 'web/vite.config.js'],
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
