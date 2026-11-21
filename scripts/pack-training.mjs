#!/usr/bin/env node
/**
 * 把一站式训练功能打包为 release-assets/oj-training.zip，供 GitHub Release 分发。
 *
 * 包含 training/ 下的训练脚本、模型清单、示例与说明；不包含本地配置、数据、
 * 训练产物与 Python 缓存（与 .gitignore 的训练运行时规则一致）。
 * 用户解压后按 training/README.md 操作：pip install -r requirements.txt 后
 * 运行 python3 ojtrain.py models 查看可训练模型。
 *
 * 用法：node scripts/pack-training.mjs
 * 输出路径可用 OJ_TRAINING_OUT 覆盖（默认 release-assets/oj-training.zip）。
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildZip } from './lib/zip.mjs';

export const TRAINING_PACKAGE_NAME = 'oj-training.zip';

/** 训练包白名单：根目录脚本/清单/说明 + examples/；其余（tests、数据、产物）一律不带。 */
export function selectTrainingFiles(entries) {
  const included = [];
  const excluded = [];
  for (const path of entries) {
    const p = String(path).replace(/\\/g, '/');
    const base = p.split('/').pop();
    const ok =
      !p.includes('/__pycache__/') &&
      !p.endsWith('.pyc') &&
      (p.startsWith('examples/') ||
        (!p.includes('/') &&
          (p.endsWith('.py') || p.endsWith('.mjs') || ['catalog.json', 'config.example.json', 'requirements.txt', 'README.md'].includes(base))));
    (ok ? included : excluded).push(p);
  }
  return { included, excluded };
}

function walk(dir, projectRoot, out = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (name === '__pycache__' || name === 'node_modules') continue;
      walk(full, projectRoot, out);
    } else if (st.isFile()) {
      out.push(relative(projectRoot, full).split(sep).join('/'));
    }
  }
  return out;
}

export function packTraining({ projectRoot, outPath } = {}) {
  const trainingDir = join(projectRoot, 'training');
  if (!existsSync(join(trainingDir, 'ojtrain.py'))) {
    throw new Error('找不到 training/ojtrain.py，训练包未生成。');
  }
  const all = walk(trainingDir, projectRoot).map((p) => p.slice('training/'.length));
  const { included } = selectTrainingFiles(all);
  if (!included.length) throw new Error('训练包没有任何文件，训练包未生成。');
  const files = included.map((name) => ({
    name: `training/${name}`,
    data: readFileSync(join(trainingDir, name)),
  }));
  const zip = buildZip(files);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, zip);
  return { outPath, fileCount: files.length, size: zip.length };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  const projectRoot = join(dirname(fileURLToPath(import.meta.url)), '..');
  const outPath = process.env.OJ_TRAINING_OUT
    ? join(projectRoot, process.env.OJ_TRAINING_OUT)
    : join(projectRoot, 'release-assets', TRAINING_PACKAGE_NAME);
  const { fileCount, size } = packTraining({ projectRoot, outPath });
  console.log(`已生成训练包 ${relative(projectRoot, outPath)}（${fileCount} 个文件，${(size / 1024 / 1024).toFixed(2)} MB）`);
}
