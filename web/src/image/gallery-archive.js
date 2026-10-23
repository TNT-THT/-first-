/**
 * 生图画廊的导出 / 导入（zip 格式，纯逻辑）。
 *
 * 压缩包结构：
 *   manifest.json        格式标识、版本、每张图的提示词、尺寸、步数、引导、种子、模型、创建时间
 *   images/001-xxx.png   图片（PNG 原文件，不重新编码）
 *
 * 导入时逐项校验：格式与版本、文件路径安全、文件存在、PNG 文件头、尺寸与参数为合理的数值。
 * 不合法的条目会被跳过并计入报告，而不是导入一半后报错。
 */
import { buildStoredZip, readZip, safeEntryName } from './zip.js';

export const GALLERY_FORMAT = 'oj-ai-gallery';
export const GALLERY_VERSION = 1;
export const GALLERY_MANIFEST = 'manifest.json';
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const MAX_IMAGE_BYTES = 32 * 1024 * 1024;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

export function isPng(bytes) {
  return bytes.length >= 8 && PNG_SIGNATURE.every((b, i) => bytes[i] === b);
}

/** 文件名中不安全的字符替换为 _，并保证以 .png 结尾。 */
export function safePngName(name) {
  const base = String(name || 'image')
    .replace(/\.png$/i, '')
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .slice(0, 80) || 'image';
  return `${base}.png`;
}

/**
 * @param {Array<{data: Uint8Array, filename: string, prompt: string, width: number, height: number,
 *   steps: number, guidance: number, seed: number, model: string, createdAt: number}>} items
 * @returns {Uint8Array} zip 字节
 */
export function buildGalleryArchive(items, { now = new Date() } = {}) {
  const used = new Set();
  const entries = items.map((it, i) => {
    let file = `images/${String(i + 1).padStart(3, '0')}-${safePngName(it.filename)}`;
    while (used.has(file)) file = file.replace(/\.png$/, '-dup.png');
    used.add(file);
    return {
      file,
      prompt: String(it.prompt || ''),
      width: it.width,
      height: it.height,
      steps: it.steps,
      guidance: it.guidance,
      seed: it.seed,
      model: String(it.model || ''),
      createdAt: Number(it.createdAt) || 0,
    };
  });
  const manifest = {
    format: GALLERY_FORMAT,
    version: GALLERY_VERSION,
    exportedAt: now.toISOString(),
    count: entries.length,
    items: entries,
  };
  const files = [
    { name: GALLERY_MANIFEST, data: encoder.encode(JSON.stringify(manifest, null, 2)) },
    ...items.map((it, i) => ({ name: entries[i].file, data: it.data })),
  ];
  return buildStoredZip(files, { date: now });
}

function validNumber(v, min, max) {
  return Number.isFinite(v) && v >= min && v <= max;
}

/**
 * 读取画廊压缩包。
 * @returns {Promise<{items: Array<object>, skipped: Array<{file: string, reason: string}>}>}
 *   items 中每项包含 data（PNG 字节）与提示词等元数据。
 */
export async function readGalleryArchive(bytes) {
  const files = await readZip(bytes);
  const byName = new Map(files.map((f) => [f.name, f.data]));
  const raw = byName.get(GALLERY_MANIFEST);
  if (!raw) throw new Error('这不是本应用导出的画廊文件（缺少 manifest.json）');
  let manifest;
  try {
    manifest = JSON.parse(decoder.decode(raw));
  } catch {
    throw new Error('画廊文件的 manifest.json 无法解析');
  }
  if (manifest?.format !== GALLERY_FORMAT) throw new Error('文件格式不受支持，这不是本应用导出的画廊');
  if (!Number.isInteger(manifest.version) || manifest.version > GALLERY_VERSION) {
    throw new Error(`画廊文件版本 ${manifest.version} 高于当前支持的版本 ${GALLERY_VERSION}，请更新页面后再导入`);
  }
  if (!Array.isArray(manifest.items)) throw new Error('画廊文件的 items 无效');
  if (manifest.items.length > 2000) throw new Error('画廊文件中的图片过多（上限 2000 张）');

  const items = [];
  const skipped = [];
  for (const entry of manifest.items) {
    const file = String(entry?.file || '');
    let name;
    try {
      name = safeEntryName(file);
    } catch (err) {
      skipped.push({ file, reason: err.message });
      continue;
    }
    const data = byName.get(name);
    if (!data) {
      skipped.push({ file, reason: '压缩包中缺少该图片' });
      continue;
    }
    if (data.length > MAX_IMAGE_BYTES || !isPng(data)) {
      skipped.push({ file, reason: '不是有效的 PNG 图片' });
      continue;
    }
    if (!validNumber(entry.width, 1, 4096) || !validNumber(entry.height, 1, 4096)) {
      skipped.push({ file, reason: '图片尺寸无效' });
      continue;
    }
    items.push({
      data,
      filename: safePngName(name.split('/').pop()),
      prompt: String(entry.prompt || '').slice(0, 4000),
      width: entry.width,
      height: entry.height,
      steps: validNumber(entry.steps, 1, 100) ? entry.steps : 20,
      guidance: validNumber(entry.guidance, 0, 30) ? entry.guidance : 7.5,
      seed: validNumber(entry.seed, 0, 4294967295) ? entry.seed : 0,
      model: String(entry.model || '').slice(0, 200),
      createdAt: validNumber(entry.createdAt, 0, Number.MAX_SAFE_INTEGER) ? entry.createdAt : Date.now(),
    });
  }
  return { items, skipped };
}
