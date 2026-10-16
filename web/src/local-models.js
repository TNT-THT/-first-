/**
 * “新建 AI”登记表：记录本浏览器中通过导入（或训练结果导入）得到的模型。
 *
 * - 只保存元数据（名称、来源、缓存地址、运行库地址），模型文件本身在 Cache Storage 中；
 * - 不上传云端，换一台电脑或清空浏览器数据后需要重新导入；
 * - 与官方模型完全独立：官方列表不会被改写，新建 AI 的缓存地址也与官方地址不同。
 */
import { isLocalRecord } from './model-source.js';

const KEY = 'oj-ai.localModels';

function readAll(storage) {
  try {
    const list = JSON.parse(storage.getItem(KEY) || '[]');
    if (!Array.isArray(list)) return [];
    return list.filter((e) => e && typeof e.id === 'string' && e.record && isLocalRecord(e.record));
  } catch {
    return [];
  }
}

function writeAll(list, storage) {
  storage.setItem(KEY, JSON.stringify(list));
}

/** @returns {Array<{id: string, label: string, source: string, originId: string, createdAt: string, record: object}>} */
export function loadLocalModels(storage = globalThis.localStorage) {
  if (!storage) return [];
  return readAll(storage);
}

export function addLocalModel(entry, storage = globalThis.localStorage) {
  const list = readAll(storage).filter((e) => e.id !== entry.id);
  list.push(entry);
  writeAll(list, storage);
  return list;
}

export function removeLocalModel(id, storage = globalThis.localStorage) {
  const list = readAll(storage).filter((e) => e.id !== id);
  writeAll(list, storage);
  return list;
}

/** 供推理与缓存检查使用的模型记录（WebLLM 的 extraRecords 格式）。 */
export function localRecords(list) {
  return list.map((e) => e.record);
}
