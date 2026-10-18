/**
 * 网站包（静态版）包含哪些文件：纯逻辑，便于测试。
 *
 * 采用白名单：只收录网站源码、构建产物和说明文件。
 * 模型权重、导出的 .ojmodel、训练数据、环境变量与密钥一律排除。
 * 网站本身不包含任何 AI 模型：模型由浏览器从 Hugging Face 下载并缓存，
 * 或由管理员单独导出一个 .ojmodel 文件。
 */

export const SITE_ROOT_FILES = ['package.json', 'package-lock.json', 'README.md'];
export const SITE_DIRS = ['web/'];
export const SITE_ZIP_NAME = 'web/dist/site.zip';
export const MAX_SITE_FILE_BYTES = 20 * 1024 * 1024;

const EXCLUDED_SEGMENTS = new Set(['node_modules', '.git']);
const EXCLUDED_EXTENSIONS = [
  '.ojmodel',
  '.ojmodel.part',
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

/** 返回文件被排除的原因；未排除返回 null。 */
export function exclusionReason(path, size = 0) {
  const p = String(path).replace(/\\/g, '/');
  const parts = p.split('/');
  if (parts.some((s) => EXCLUDED_SEGMENTS.has(s))) return '依赖或版本库目录';
  const base = parts[parts.length - 1];
  if (/^\.env(\.|$)/.test(base)) return '环境变量或密钥文件';
  if (p === SITE_ZIP_NAME) return '网站包本身';
  if (EXCLUDED_EXTENSIONS.some((ext) => base.toLowerCase().endsWith(ext))) return '模型或权重文件';
  if (size > MAX_SITE_FILE_BYTES) return '文件过大';
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
