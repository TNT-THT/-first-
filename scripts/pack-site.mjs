#!/usr/bin/env node
/**
 * 生成网站包 web/dist/site.zip（静态版）。通常在 `npm run build` 之后自动执行。
 *
 * 包含：网站源码（web/）、构建产物（web/dist/）、package.json、package-lock.json、README.md。
 * 不包含：任何 AI 模型、.ojmodel、训练数据、.env 与密钥。见 scripts/site-files.mjs。
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildZip } from './lib/zip.mjs';
import { SITE_ROOT_FILES, SITE_ZIP_NAME, selectSiteFiles } from './site-files.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      if (name === 'node_modules' || name === '.git') continue;
      walk(full, out);
    } else if (st.isFile()) {
      out.push(full);
    }
  }
  return out;
}

export function packSite({ projectRoot = root } = {}) {
  if (!existsSync(join(projectRoot, 'web', 'dist', 'index.html'))) {
    throw new Error('找不到 web/dist/index.html，请先运行 npm run build。');
  }
  const candidates = [
    ...SITE_ROOT_FILES.filter((f) => existsSync(join(projectRoot, f))),
    ...walk(join(projectRoot, 'web')).map((f) => relative(projectRoot, f).split(sep).join('/')),
  ];
  const entries = candidates.map((path) => ({
    path,
    size: statSync(join(projectRoot, path)).size,
  }));
  const { included, excluded } = selectSiteFiles(entries);
  const files = included.map((path) => ({ name: path, data: readFileSync(join(projectRoot, path)) }));
  const zip = buildZip(files);
  const outPath = join(projectRoot, SITE_ZIP_NAME);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, zip);
  return { outPath, fileCount: included.length, size: zip.length, excluded };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  try {
    const { outPath, fileCount, size, excluded } = packSite();
    const mb = (size / 1024 / 1024).toFixed(2);
    console.log(`已生成网站包 ${relative(root, outPath)}（${fileCount} 个文件，${mb} MB）`);
    if (excluded.length) console.log(`已排除 ${excluded.length} 个文件（如 .env、模型文件）`);
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
