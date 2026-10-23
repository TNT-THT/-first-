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

/** Hugging Face 上某个文件的下载地址。 */
export function fileUrl(model, path) {
  return `https://huggingface.co/${model.repo}/resolve/${model.revision || 'main'}/${path}`;
}

/** 运行时需要的全部地址（与 files 中的键对应）。 */
export function sourceUrls(model) {
  const out = {};
  for (const [key, f] of Object.entries(model.files || {})) out[key] = fileUrl(model, f.path);
  return out;
}

/** 每个运行时文件的大小（字节），键与 sourceUrls 相同；用于加载进度。 */
export function sourceSizes(model) {
  const out = {};
  for (const [key, f] of Object.entries(model.files || {})) out[key] = Number(f.size) || 0;
  return out;
}

/**
 * 文本编码器的备选文件（config 中的 textEncoderFallback，如 fp32）。
 * 它不在 files 中：不参与下载、缓存状态与导出，只在主文件无法创建会话时由生图 worker 直接读取。
 * 没有备选文件时返回 null。
 */
export function fallbackFile(model) {
  const f = model?.textEncoderFallback;
  if (!f || typeof f.path !== 'string' || typeof f.repo !== 'string') return null;
  return {
    url: `https://huggingface.co/${f.repo}/resolve/${f.revision || 'main'}/${f.path}`,
    size: Number(f.size) || 0,
  };
}

/** 生成时 worker 需要的全部地址：files 的地址，加上备选文件（如有）。 */
export function runtimeUrls(model) {
  const urls = sourceUrls(model);
  const fb = fallbackFile(model);
  if (fb) urls.textEncoderFallback = fb.url;
  return urls;
}

/** 生成时 worker 需要的全部大小：files 的大小，加上备选文件（如有）。 */
export function runtimeSizes(model) {
  const sizes = sourceSizes(model);
  const fb = fallbackFile(model);
  if (fb) sizes.textEncoderFallback = fb.size;
  return sizes;
}
