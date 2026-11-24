#!/usr/bin/env node
/**
 * 生成局域网服务器包 web/dist/server-win-x64.zip（先只出 Windows）。
 *
 * 与网站包的区别：网站包是“每台电脑自己跑一份”；服务器包只在一台 Windows
 * 电脑上运行，局域网内其他人用浏览器打开 http://服务器IP:3000 即可使用，
 * 模型由服务器本地计算，聊天结果传回各用户网页。包内包含：
 *   - Windows 版 Node 运行时（runtime/node-win-x64.exe，免安装）；
 *   - 服务端（server/，含打好的 server/node_modules 运行依赖）；
 *   - 服务版网页（web/dist-api，浏览器打开即为可用的聊天界面）；
 *   - start-server.bat / serve-server.ps1 / 服务器使用说明.txt。
 * 不包含任何模型权重、.env 与密钥；账号与模型数据运行时生成在 data/ 中。
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildZip } from './lib/zip.mjs';
import { buildRemoteIntoPackage, stageServerDeps } from './pack-site.mjs';
import { ensureNodeRuntimes, NODE_RUNTIME_TARGETS } from './node-runtime.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
export const SERVER_ZIP_PATH = 'web/dist/server-win-x64.zip';

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

export async function packServer({ projectRoot = root } = {}) {
  if (!existsSync(join(projectRoot, 'web', 'dist', 'index.html'))) {
    throw new Error('找不到 web/dist/index.html，请先运行 npm run build。');
  }
  buildRemoteIntoPackage(projectRoot);
  const depFiles = stageServerDeps(projectRoot);
  const winTarget = NODE_RUNTIME_TARGETS.find((t) => t.id === 'win-x64');
  const runtimes = await ensureNodeRuntimes({ projectRoot, targets: [winTarget] });
  const runtime = runtimes.files.find((f) => f.targetId === 'win-x64');
  if (!runtime) {
    throw new Error('未能取得 Windows 版 Node 运行时，服务器包未生成（请联网后重试）。');
  }
  const files = [
    ...walk(join(projectRoot, 'server')).map((full) => ({
      name: `server/${relative(join(projectRoot, 'server'), full).split(sep).join('/')}`,
      data: readFileSync(full),
    })),
    ...walk(join(projectRoot, 'web', 'dist-api')).map((full) => ({
      name: `web/dist-api/${relative(join(projectRoot, 'web', 'dist-api'), full).split(sep).join('/')}`,
      data: readFileSync(full),
    })),
    ...depFiles,
    { name: runtime.name, data: runtime.data, mode: runtime.mode },
    { name: 'start-server.bat', data: readFileSync(join(projectRoot, 'scripts', 'launcher-server', 'start-server.bat')) },
    { name: 'serve-server.ps1', data: readFileSync(join(projectRoot, 'scripts', 'launcher-server', 'serve-server.ps1')) },
    { name: '服务器使用说明.txt', data: readFileSync(join(projectRoot, 'scripts', 'launcher-server', '服务器使用说明.txt')) },
    {
      name: 'server.env.example',
      data: Buffer.from(
        [
          '# 复制本文件为 server.env 后修改；改完重启 start-server.bat 生效。',
          '# 管理员口令（下载/加载模型与 API 控制台用，至少 16 位）：',
          '# API_ADMIN_PASSWORD=换成你自己的长口令',
          '# 固定登录签名密钥（不设则每次重启后账号需重新登录）：',
          '# SESSION_SECRET=换成一串随机长字符串',
          '# 云端上游模型（与“本机大模型”二选一；用本机模型可不填）：',
          '# LLM_BASE_URL=https://example.com/v1',
          '# LLM_API_KEY=',
          '# LLM_MODEL=',
          '# 端口（如被占用可改）：',
          '# PORT=3000',
          '',
        ].join('\r\n'),
        'utf8',
      ),
    },
  ];
  const zip = buildZip(files);
  const outPath = join(projectRoot, SERVER_ZIP_PATH);
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, zip);
  return { outPath, size: zip.length, fileCount: files.length };
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) {
  try {
    const { outPath, size, fileCount } = await packServer();
    console.log(`已生成服务器包 ${relative(root, outPath)}（${fileCount} 个文件，${(size / 1024 / 1024).toFixed(2)} MB）`);
  } catch (err) {
    console.error(err.message);
    process.exit(1);
  }
}
