/**
 * 生图模型目录（纯逻辑）。模型信息来自 config.json 的 imageModels。
 * status: 'available' 表示文件齐全、可在浏览器中运行；其他状态只显示、不可选择。
 */
export function imageModelsOf(cfg) {
  return (cfg?.imageModels || []).filter((m) => m && typeof m.id === 'string');
}

export function isSelectable(model) {
  return model?.status === 'available';
}

/** 所有文件的总大小（字节）。 */
export function totalBytes(model) {
  return Object.values(model?.files || {}).reduce((sum, f) => sum + (Number(f.size) || 0), 0);
}

export function formatGB(bytes) {
  return `${(bytes / 1e9).toFixed(2)} GB`; // 十进制 GB，与 Hugging Face 页面显示一致
}

/**
 * Hugging Face 上某个文件的下载地址。file 可省略（只给路径时使用模型的仓库）；
 * file.repo 与 file.revision 可单独指定其他仓库（例如备选的 fp32 文本编码器）。
 */
export function fileUrl(model, path, file = {}) {
  // file.local：本站 public 目录中的文件（相对于网站根目录），例如修正过计算图的 UNet
  if (file.local) return localFileUrl(file.local);
  const repo = file.repo || model.repo;
  const revision = file.revision || model.revision || 'main';
  return `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
}

/**
 * 本站文件的绝对地址（在页面中按页面地址解析）。没有页面地址时（如 Node 测试）返回相对路径。
 * worker 不解析地址，由主线程把 sourceUrls 的结果传入。
 */
export function localFileUrl(rel) {
  const base = globalThis.document?.baseURI || globalThis.location?.href;
  if (!base) return `./${rel}`;
  return new URL(rel, base).href;
}

/** 运行时需要的全部地址（与 files 中的键对应）。 */
export function sourceUrls(model) {
  const out = {};
  for (const [key, f] of Object.entries(model.files || {})) out[key] = fileUrl(model, f.path, f);
  return out;
}

/** 每个运行时文件的大小（字节），键与 sourceUrls 相同；用于加载进度。 */
export function sourceSizes(model) {
  const out = {};
  for (const [key, f] of Object.entries(model.files || {})) out[key] = Number(f.size) || 0;
  return out;
}
