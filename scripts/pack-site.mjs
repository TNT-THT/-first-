#!/usr/bin/env node
/**
 * 生成按平台拆分的网站包（web/dist/site-<平台>.zip）。通常在 `npm run build`（静态版）之后自动执行。
 *
 * 除静态版网页外，还会把服务版网页构建到 web/dist-api 并一并打包，
 * 同时包含 Node 服务端（server/）、本机启动脚本（scripts/）与服务端运行依赖
 * （打入 server/node_modules/）。最后按平台分别生成四个网站包
 * （site-win-x64.zip、site-darwin-arm64.zip、site-darwin-x64.zip、site-linux-x64.zip），
 * 每个包的 runtime/ 里只放对应平台的一个内置 Node 运行时：
 * 用户按自己电脑的系统下载一个，解压双击即可启动本机 API，
 * 在网页里点“申请 API Key”就能使用——不需要预先安装 Node.js，也不用一次下载全部平台。
 * 包含：网站源码（web/）、构建产物（web/dist/、web/dist-api/）、package.json、package-lock.json、README.md，
 * 以及根目录的本机启动文件（start-windows.bat、start-mac.command 等）。
 * 不包含：任何 AI 模型、.ojmodel、训练数据、.env 与密钥。见 scripts/site-files.mjs。
 */
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildZip } from './lib/zip.mjs';
import { LAUNCHER_DIR, LAUNCHER_FILES, SITE_ROOT_FILES, siteZipPath, selectSiteFiles } from './site-files.mjs';
import { ensureNodeRuntimes, selectedNodeRuntimeTargets } from './node-runtime.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
/** 服务端运行时真正需要的第三方包（与 package.json 的版本一致）；其余依赖只供浏览器构建使用。 */
const SERVER_DEP_NAMES = ['express', 'ws', 'dotenv'];

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

/** 把服务版网页（远程构建）输出到 web/dist-api，随网站包一起分发，供本机 API 服务使用。 */
export function buildRemoteIntoPackage(projectRoot) {
  const result = spawnSync(process.execPath, [join(projectRoot, 'scripts', 'build-remote.mjs')], {
    cwd: projectRoot,
    env: { ...process.env, OJ_REMOTE_OUT_DIR: join('web', 'dist-api') },
    stdio: 'inherit',
  });
  if (result.error || result.status !== 0) {
    throw new Error('构建服务版网页（web/dist-api）失败，网站包未生成：缺少它，本机 API 功能将无法使用。');
  }
}

/**
 * 把服务端运行依赖安装到暂存区，返回打入 server/node_modules/ 的 zip 条目。
 * 这样用户解压后不必再运行 npm install，真正做到“下载一次、双击即用”。
 */
export function stageServerDeps(projectRoot) {
  const staging = join(projectRoot, '.package-staging');
  rmSync(staging, { recursive: true, force: true });
  mkdirSync(staging, { recursive: true });
  const rootPkg = JSON.parse(readFileSync(join(projectRoot, 'package.json'), 'utf8'));
  const dependencies = {};
  for (const name of SERVER_DEP_NAMES) {
    if (!rootPkg.dependencies?.[name]) throw new Error(`package.json 缺少服务端依赖：${name}`);
    dependencies[name] = rootPkg.dependencies[name];
  }
  writeFileSync(
    join(staging, 'package.json'),
    `${JSON.stringify({ name: 'oj-ai-api-runtime', private: true, version: '1.0.0', dependencies }, null, 2)}\n`,
  );
  console.log('正在准备本机 API 的运行依赖（打入网站包，用户无需再下载）…');
  const result = spawnSync('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund'], {
    cwd: staging,
    stdio: 'inherit',
    shell: process.platform === 'win32',
  });
  if (result.error || result.status !== 0) {
    throw new Error('准备服务端运行依赖失败，网站包未生成。');
  }
  const nmRoot = join(staging, 'node_modules');
  return walk(nmRoot).map((full) => ({
    name: `server/node_modules/${relative(nmRoot, full).split(sep).join('/')}`,
    data: readFileSync(full),
  }));
}

export async function packSite({ projectRoot = root } = {}) {
  if (!existsSync(join(projectRoot, 'web', 'dist', 'index.html'))) {
    throw new Error('找不到 web/dist/index.html，请先运行 npm run build。');
  }
  buildRemoteIntoPackage(projectRoot);
  const depFiles = stageServerDeps(projectRoot);
  const runtimes = await ensureNodeRuntimes({ projectRoot });
  if (runtimes.missing.length) {
    console.warn(
      `[warn] 网站包将缺少这些平台的内置 Node 运行时：${runtimes.missing.join(', ')}。` +
        '对应平台的启动器会回退到系统安装的 Node.js；如需完整包，请联网后重新打包。',
    );
  }
  const candidates = [
    ...SITE_ROOT_FILES.filter((f) => existsSync(join(projectRoot, f))),
    ...walk(join(projectRoot, 'web')).map((f) => relative(projectRoot, f).split(sep).join('/')),
    ...walk(join(projectRoot, 'server')).map((f) => relative(projectRoot, f).split(sep).join('/')),
    ...walk(join(projectRoot, 'scripts')).map((f) => relative(projectRoot, f).split(sep).join('/')),
  ];
  const entries = candidates.map((path) => ({
    path,
    size: statSync(join(projectRoot, path)).size,
  }));
  const { included, excluded } = selectSiteFiles(entries);
  const files = included.map((path) => ({ name: path, data: readFileSync(join(projectRoot, path)) }));
  // 本机启动文件放在包的根目录，双击即可运行（见 scripts/launcher/）
  for (const f of LAUNCHER_FILES) {
    files.push({ name: f.to, data: readFileSync(join(projectRoot, LAUNCHER_DIR, f.from)), mode: f.mode });
  }
  // 服务端运行依赖（不走白名单筛选，直接装入固定位置）
  files.push(...depFiles);
  // 按平台分别打包：每个 zip 只放入对应平台的内置 Node 运行时，用户按需下载一个。
  const packages = [];
  for (const target of selectedNodeRuntimeTargets()) {
    const runtime = runtimes.files.find((f) => f.targetId === target.id);
    if (!runtime) continue; // 运行时缺失的平台已在上方警告，这里跳过，不生成残缺包
    const zip = buildZip([...files, runtime]);
    const outPath = join(projectRoot, siteZipPath(target.id));
    mkdirSync(dirname(outPath), { recursive: true });
    writeFileSync(outPath, zip);
    packages.push({ target: target.id, outPath, size: zip.length });
  }
  if (!packages.length) {
    // 一个运行时都没拿到时（例如打包机完全无网络且无缓存）：仍生成一个不带运行时的通用包，
    // 保证静态网站可以发布，同时明确警告它需要用户自备 Node.js 才能使用本机 API。
    console.warn('[warn] 未能获取任何平台的内置 Node 运行时，仅生成不含运行时的 site.zip；该包需要用户自行安装 Node.js。');
    const zip = buildZip(files);
    const outPath = join(projectRoot, siteZipPath('').replace(/-\.zip$/, '.zip'));
    mkdirSync(dirname(outPath), { recursive: true });
    writeFileSync(outPath, zip);
    packages.push({ target: 'generic', outPath, size: zip.length });
  }
  return {
    packages,
    fileCount: files.length,
    excluded,
    serverDepFiles: depFiles.length,
    runtimeFiles: runtimes.files.map((f) => f.name),
    missingRuntimes: runtimes.missing,
  };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  try {
    const { packages, fileCount, excluded, serverDepFiles, runtimeFiles, missingRuntimes } = await packSite();
    for (const pkg of packages) {
      const mb = (pkg.size / 1024 / 1024).toFixed(2);
      console.log(`已生成网站包 ${relative(root, pkg.outPath)}（平台：${pkg.target}，${fileCount} 个文件，${mb} MB）`);
    }
    console.log(`  服务端运行依赖：${serverDepFiles} 个文件（server/node_modules/）`);
    console.log(`  内置 Node 运行时：${runtimeFiles.length ? runtimeFiles.join('、') : '无'}`);
    if (missingRuntimes.length) console.log(`  缺失运行时（对应平台包未生成）：${missingRuntimes.join('、')}`);
    if (excluded.length) console.log(`已排除 ${excluded.length} 个文件（如 .env、模型文件）`);
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
