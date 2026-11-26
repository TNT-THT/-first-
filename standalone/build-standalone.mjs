#!/usr/bin/env node
/**
 * 构建独立软件「LLM托管平台——天蓝色1.0」：
 *
 * 产物（release-assets/standalone/）：
 * - `LLM托管平台-天蓝色1.0.exe`：Windows 单文件可执行程序（Node SEA：
 *   服务端代码 + 内置服务版网页打进一个 exe，双击即用，不需要装 Node）。
 *   在非 Windows 系统构建时产物无 .exe 后缀，仅供本机校验。
 * - `LLM托管平台-天蓝色1.0.zip`：完整压缩包（exe + 解压版程序 app/ +
 *   使用说明；--with-llama 时另含 llama.cpp 运行库 llama/ 与 app/server/native-bin/）。
 *
 * 本机大模型权重不进任何安装包：运行时在软件内按需下载到数据目录。
 *
 * 用法：node standalone/build-standalone.mjs [--skip-web] [--with-llama]
 */
import { spawnSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildZip } from '../scripts/lib/zip.mjs';
import { extractZip } from '../server/native/archive.js';
import { LLAMA_RELEASE_API, pickLlamaRelease, pickReleaseAsset } from '../server/native/llama.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const buildDir = join(root, 'standalone', '.build');
const payloadDir = join(buildDir, 'payload');
const outDir = join(root, 'release-assets', 'standalone');
// Release 资产文件名使用单个 ASCII 连字符，必须与 web/src/release-downloads.js、
// web/index.html 和 .github/workflows/release.yml 完全一致（展示名中的双破折号不能进文件名）。
const PRODUCT_FILE_NAME = 'LLM托管平台-天蓝色1.0';
const args = new Set(process.argv.slice(2));
const skipWeb = args.has('--skip-web');
const withLlama = args.has('--with-llama');

function run(command, argv, options = {}) {
  const result = spawnSync(command, argv, { cwd: root, stdio: 'inherit', ...options });
  if (result.error) throw new Error(`无法运行 ${command}：${result.error.message}`);
  if (result.status !== 0) throw new Error(`${command} ${argv.join(' ')} 退出码 ${result.status}`);
}

function walk(dir, base = dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) out.push(...walk(full, base));
    else if (st.isFile()) out.push({ name: relative(base, full).split(sep).join('/'), data: readFileSync(full) });
  }
  return out;
}

/** 构建服务版网页到 web/dist-api（独立软件只用服务版：功能全集）。 */
function buildWeb() {
  if (skipWeb && existsSync(join(root, 'web', 'dist-api', 'index.html'))) {
    console.log('[standalone] 跳过网页构建（--skip-web 且已有 web/dist-api）。');
    return;
  }
  console.log('[standalone] 构建服务版网页（web/dist-api）…');
  run(process.execPath, [join(root, 'scripts', 'build-remote.mjs')], {
    env: { ...process.env, OJ_REMOTE_OUT_DIR: join('web', 'dist-api') },
  });
}

/** 组装 exe 内置载荷（网页 + 版本 + 说明），并打成 payload.zip 供 SEA 嵌入。 */
function buildPayload(version) {
  rmSync(payloadDir, { recursive: true, force: true });
  mkdirSync(join(payloadDir, 'web'), { recursive: true });
  cpSync(join(root, 'web', 'dist-api'), join(payloadDir, 'web', 'dist-api'), { recursive: true });
  writeFileSync(join(payloadDir, 'version.txt'), `${version}\n`);
  cpSync(join(root, 'standalone', '使用说明.txt'), join(payloadDir, '使用说明.txt'));
  mkdirSync(join(payloadDir, 'native'), { recursive: true });
  cpSync(join(root, 'server', 'native', 'python', 'runner.py'), join(payloadDir, 'native', 'runner.py'));
  const files = walk(payloadDir);
  const zip = buildZip(files);
  const zipPath = join(buildDir, 'payload.zip');
  writeFileSync(zipPath, zip);
  console.log(`[standalone] 内置载荷：${files.length} 个文件，${(zip.length / 1024 / 1024).toFixed(1)} MB。`);
  return zipPath;
}

/** 用 esbuild 把启动器 + 服务端打成单个 CommonJS 文件（SEA 只接受单文件）。 */
async function bundleLauncher() {
  let esbuild;
  try {
    esbuild = await import('esbuild');
  } catch {
    throw new Error('缺少 esbuild，请先运行 npm install（它是 devDependency）。');
  }
  const outfile = join(buildDir, 'bundle.cjs');
  await esbuild.build({
    entryPoints: [join(root, 'standalone', 'launcher.mjs')],
    bundle: true,
    platform: 'node',
    format: 'cjs',
    target: 'node20',
    outfile,
    external: ['bufferutil', 'utf-8-validate'],
    logLevel: 'warning',
  });
  console.log(`[standalone] 服务端单文件打包完成：${(statSync(outfile).size / 1024 / 1024).toFixed(1)} MB。`);
  return outfile;
}

/** 生成 SEA 可执行文件，返回产物路径。 */
function buildExe(bundlePath, payloadZipPath, version) {
  const seaConfig = {
    main: bundlePath,
    output: join(buildDir, 'sea-prep.blob'),
    disableExperimentalSEAWarning: true,
    assets: {
      'payload.zip': payloadZipPath,
      'version.txt': join(payloadDir, 'version.txt'),
    },
  };
  const seaConfigPath = join(buildDir, 'sea-config.json');
  writeFileSync(seaConfigPath, JSON.stringify(seaConfig, null, 2));
  run(process.execPath, ['--experimental-sea-config', seaConfigPath]);
  const exeName = process.platform === 'win32' ? `${PRODUCT_FILE_NAME}.exe` : PRODUCT_FILE_NAME;
  const exePath = join(outDir, exeName);
  mkdirSync(outDir, { recursive: true });
  cpSync(process.execPath, exePath, { force: true });
  // postject 注入 SEA blob（devDependency）
  const postjectPkg = JSON.parse(readFileSync(join(root, 'node_modules', 'postject', 'package.json'), 'utf8'));
  const postjectBin = join(root, 'node_modules', 'postject', typeof postjectPkg.bin === 'string' ? postjectPkg.bin : postjectPkg.bin.postject);
  const fuseArgs = [
    postjectBin,
    exePath,
    'NODE_SEA_BLOB',
    seaConfig.output,
    '--sentinel-fuse',
    'NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2',
  ];
  if (process.platform === 'darwin') fuseArgs.push('--macho-segment-name', 'NODE_SEA');
  run(process.execPath, fuseArgs);
  console.log(`[standalone] 单文件程序已生成：${exeName}（${(statSync(exePath).size / 1024 / 1024).toFixed(1)} MB，版本 ${version}）。`);
  return exePath;
}

/** 给解压版 app/ 准备服务端运行依赖（与网站包同样的三个包）。 */
function stageServerDeps() {
  const staging = join(buildDir, 'deps-staging');
  rmSync(staging, { recursive: true, force: true });
  mkdirSync(staging, { recursive: true });
  const rootPkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
  const dependencies = {};
  for (const name of ['express', 'ws', 'dotenv']) dependencies[name] = rootPkg.dependencies[name];
  writeFileSync(join(staging, 'package.json'), `${JSON.stringify({ name: 'skyblue-runtime', private: true, version: '1.0.0', dependencies }, null, 2)}\n`);
  run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund'], {
    cwd: staging,
    shell: process.platform === 'win32',
  });
  return walk(join(staging, 'node_modules'), join(staging, 'node_modules'));
}

/**
 * 带超时与重试的 GET：构建机（尤其 GitHub Actions 的 Windows runner）到
 * objects.githubusercontent.com 的链路会偶发抽风，裸 fetch 一次失败就让整个
 * Release 少两个产物。
 *
 * 这不是假想：v1.0.20 和 v1.0.22 两次都在这一步失败，而单独重跑同样的脚本
 * 又完全成功（48 秒 vs 失败那次只用了 23 秒 —— 连下载都没开始）。失败发生在
 * `fetch` 本身，一个字节都没收到。
 *
 * @param {string} url
 * @param {object} opts
 * @param {number} [opts.timeoutMs] 单次尝试的超时
 * @param {number} [opts.attempts]  总尝试次数
 */
export async function fetchWithRetry(url, {
  timeoutMs = 180000, attempts = 4, headers = {},
  // 只为测试可注入；生产路径就是全局 fetch
  fetchImpl = fetch, retryDelayMs = 2000, sleep = (ms) => new Promise((r) => setTimeout(r, ms)),
} = {}) {
  let lastErr = null;
  for (let i = 0; i < attempts; i += 1) {
    if (i > 0) {
      // 退避：2s / 4s / 8s。构建机的时间不宝贵，把事情做完更重要。
      const wait = retryDelayMs * 2 ** (i - 1);
      console.log(`[standalone] 第 ${i} 次重试（${wait / 1000}s 后）：${lastErr?.message || ''}`);
      await sleep(wait);
    }
    try {
      // AbortSignal.timeout 需要 Node 17.3+。构建机上通常够新，但有人用旧 Node 跑
      // 打包脚本时它会直接是 undefined → TypeError，报错信息完全对不上真正原因。
      // 拿不到就退回自己造的 AbortController + 定时器（Node 15+ 都支持）。
      let signal;
      let timer;
      if (typeof AbortSignal?.timeout === 'function') {
        signal = AbortSignal.timeout(timeoutMs);
      } else {
        const ac = new AbortController();
        signal = ac.signal;
        timer = setTimeout(() => ac.abort(new Error(`请求超时（${timeoutMs} ms）`)), timeoutMs);
      }
      let res;
      try {
        res = await fetchImpl(url, { headers, signal });
      } finally {
        if (timer) clearTimeout(timer);
      }
      // 5xx / 429 属于“值得再试一次”；4xx 重试也没用，直接抛出
      if (!res.ok && (res.status >= 500 || res.status === 429)) {
        lastErr = new Error(`HTTP ${res.status}`);
        continue;
      }
      return res;
    } catch (err) {
      // fetch 本身失败（DNS/连接被重置/超时）——这正是那两次 Release 的失败形态
      lastErr = err;
    }
  }
  throw lastErr || new Error(`请求失败：${url}`);
}

/** --with-llama：下载 llama.cpp 官方 Windows CPU 构建到 .build/llama。 */
async function fetchLlamaWindows() {
  const res = await fetchWithRetry(LLAMA_RELEASE_API, {
    headers: { accept: 'application/vnd.github+json', 'user-agent': 'llm-host-skyblue' },
  });
  if (!res.ok) throw new Error(`读取 llama.cpp 发布失败（HTTP ${res.status}）`);
  // /releases/latest 可能指向非 b 系列的特殊发布；必须从发布列表挑最新 b 系列构建。
  const release = pickLlamaRelease(await res.json());
  if (!release) throw new Error('没有找到 llama.cpp 的 b 系列正式发布。');
  const asset = pickReleaseAsset(release.assets || [], { platform: 'win32', arch: 'x64', variant: 'cpu' });
  if (!asset) throw new Error(`llama.cpp ${release.tag_name} 没有 Windows CPU 构建。`);
  console.log(`[standalone] 下载 llama.cpp ${release.tag_name}（${asset.name}）…`);
  const dl = await fetchWithRetry(asset.browser_download_url, {
    headers: { 'user-agent': 'llm-host-skyblue' },
  });
  if (!dl.ok) throw new Error(`下载 llama.cpp 失败（HTTP ${dl.status}）`);
  const dir = join(buildDir, 'llama');
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
  await extractZip(Buffer.from(await dl.arrayBuffer()), dir);
  return dir;
}

/** 组装完整压缩包。 */
function buildStandaloneZip(exePath, llamaDir) {
  const files = [{ name: exePath.split(sep).pop(), data: readFileSync(exePath) }];
  files.push({ name: '使用说明.txt', data: readFileSync(join(root, 'standalone', '使用说明.txt')) });
  // 解压版 app/：系统装有 Node 18+ 时可用 node standalone/launcher.mjs 运行（与 exe 同一套代码）
  const appRoot = join(buildDir, 'app');
  rmSync(appRoot, { recursive: true, force: true });
  mkdirSync(appRoot, { recursive: true });
  cpSync(join(root, 'web', 'dist-api'), join(appRoot, 'web', 'dist-api'), { recursive: true });
  cpSync(join(root, 'server'), join(appRoot, 'server'), { recursive: true });
  cpSync(join(root, 'standalone', 'launcher.mjs'), join(appRoot, 'standalone', 'launcher.mjs'));
  cpSync(join(root, 'package.json'), join(appRoot, 'package.json'));
  for (const f of walk(appRoot)) files.push({ name: `app/${f.name}`, data: f.data });
  for (const dep of stageServerDeps()) files.push({ name: `app/server/node_modules/${dep.name}`, data: dep.data });
  if (llamaDir) {
    for (const f of walk(llamaDir)) {
      files.push({ name: `llama/${f.name}`, data: f.data });
      files.push({ name: `app/server/native-bin/win32-x64/${f.name}`, data: f.data });
    }
  }
  const zip = buildZip(files);
  const zipPath = join(outDir, `${PRODUCT_FILE_NAME}.zip`);
  writeFileSync(zipPath, zip);
  console.log(`[standalone] 完整压缩包已生成：${PRODUCT_FILE_NAME}.zip（${(zip.length / 1024 / 1024).toFixed(1)} MB，${files.length} 个文件）。`);
  return zipPath;
}

/**
 * 只有“直接运行这个文件”才执行构建。
 *
 * 以前这里是无条件执行的顶层语句：测试为了验证 fetchWithRetry 而 import 本模块，
 * 结果会**真的把整个构建跑一遍**（几十秒、写出上百 MB 产物），还依赖本机有 vite。
 * 被导入和被执行必须分开。
 */
async function main() {
  const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
  const version = `${pkg.version}+${new Date().toISOString().slice(0, 10)}`;
  mkdirSync(buildDir, { recursive: true });
  buildWeb();
  const payloadZipPath = buildPayload(version);
  const bundlePath = await bundleLauncher();
  const exePath = buildExe(bundlePath, payloadZipPath, version);
  const llamaDir = withLlama ? await fetchLlamaWindows() : null;
  buildStandaloneZip(exePath, llamaDir);
  console.log('[standalone] 构建完成。');
}

// 入口判定：只有被 node 直接执行时才构建（被 import 时什么都不做）
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(`[standalone] 构建失败：${err?.message || err}`);
    process.exitCode = 1;
  });
}
