/**
 * 网站包（静态版）包含哪些文件：纯逻辑，便于测试。
 *
 * 采用白名单：只收录网站源码、构建产物和说明文件。
 * 模型权重、导出的 .ojmodel、训练数据、环境变量与密钥一律排除。
 * 网站本身不包含任何 AI 模型：模型由浏览器从 Hugging Face 下载并缓存，
 * 或由管理员单独导出一个 .ojmodel（聊天）/ .ojimage（生图）文件。
 * 生图模块的源码（web/src/image/）属于网站源码，包含在网站包中。
 */

export const SITE_ROOT_FILES = ['package.json', 'package-lock.json', 'README.md'];
export const SITE_DIRS = ['web/'];
export const SITE_ZIP_NAME = 'web/dist/site.zip';
export const MAX_SITE_FILE_BYTES = 20 * 1024 * 1024;
/**
 * 生图运行库（ONNX Runtime 的 wasm 文件，见 vite.config.js）允许更大：
 * ort-wasm-simd-threaded.asyncify.wasm 约 27 MB，随网站包分发后，生图功能不需要联网下载运行库。
 */
export const RUNTIME_DIR = 'web/dist/ort/';
export const MAX_RUNTIME_FILE_BYTES = 40 * 1024 * 1024;

/**
 * 网站包根目录的本机启动文件（双击即可运行）。来源：scripts/launcher/。
 * mode 为 Unix 权限位，macOS / Linux 的启动脚本需要可执行权限。
 */
export const LAUNCHER_DIR = 'scripts/launcher';
export const LAUNCHER_FILES = [
  { from: 'start-windows.bat', to: 'start-windows.bat', mode: 0o644 },
  { from: 'serve-site.ps1', to: 'serve-site.ps1', mode: 0o644 },
  { from: 'start-mac.command', to: 'start-mac.command', mode: 0o755 },
  { from: 'serve-site.py', to: 'serve-site.py', mode: 0o755 },
  { from: '使用说明.txt', to: '使用说明.txt', mode: 0o644 },
];

const EXCLUDED_SEGMENTS = new Set(['node_modules', '.git']);
const EXCLUDED_EXTENSIONS = [
  '.ojmodel',
  '.ojmodel.part',
  '.ojimage',
  '.safetensors',
  '.gguf',
  '.ggml',
  '.bin',
  '.pt',
  '.pth',
  '.onnx',
  '.ckpt',
  '.npz',
  '.h5',
  '.pkl',
];

/** 允许打包的图文件（不含权重）：SD1.5 UNet 补丁后的计算图，见 scripts/image/patch_sd15_unet_fp32_ln.py。 */
export const ALLOWED_MODEL_GRAPHS = new Set(['web/public/sd15-unet/model.onnx']);

/** 返回文件被排除的原因；未排除返回 null。 */
export function exclusionReason(path, size = 0) {
  const p = String(path).replace(/\\/g, '/');
  const parts = p.split('/');
  if (parts.some((s) => EXCLUDED_SEGMENTS.has(s))) return '依赖或版本库目录';
  const base = parts[parts.length - 1];
  if (/^\.env(\.|$)/.test(base)) return '环境变量或密钥文件';
  if (p === SITE_ZIP_NAME) return '网站包本身';
  if (ALLOWED_MODEL_GRAPHS.has(p)) return null;
  if (EXCLUDED_EXTENSIONS.some((ext) => base.toLowerCase().endsWith(ext))) return '模型或权重文件';
  const limit = p.startsWith(RUNTIME_DIR) ? MAX_RUNTIME_FILE_BYTES : MAX_SITE_FILE_BYTES;
  if (size > limit) return '文件过大';
  return null;
}

/** 是否属于白名单范围。 */
export function inSiteScope(path) {
  const p = String(path).replace(/\\/g, '/');
  if (SITE_ROOT_FILES.includes(p)) return true;
  return SITE_DIRS.some((dir) => p.startsWith(dir));
}

/**
 * @param {Array<{path: string, size?: number}>} entries 仓库内的文件（相对路径）
 * @returns {{included: string[], excluded: Array<{path: string, reason: string}>}}
 */
export function selectSiteFiles(entries) {
  const included = [];
  const excluded = [];
  for (const { path, size = 0 } of entries) {
    if (!inSiteScope(path)) continue;
    const reason = exclusionReason(path, size);
    if (reason) excluded.push({ path, reason });
    else included.push(path);
  }
  included.sort();
  return { included, excluded };
}
