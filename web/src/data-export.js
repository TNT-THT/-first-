/**
 * 本机数据的选择性导出（仅管理员，静态版）。纯逻辑，便于测试。
 *
 * 导出的文件只保存在你自己的电脑上，不会上传。文件可能包含对话内容，请妥善保管。
 * 永远不导出：管理员标记（oj-ai.admin）、服务端令牌（oj-ai.token）、旧版解锁标记（oj-ai.unlock）。
 * 模型权重（浏览器缓存）不属于本机数据，需要时使用“导出当前模型”。
 */

export const DATA_EXPORT_FORMAT = 'oj-ai-data';
export const DATA_EXPORT_VERSION = 1;

export const DATA_SECTIONS = [
  { id: 'sessions', key: 'oj-ai.sessions.v1', label: '会话记录（对话历史）' },
  { id: 'adaptation', key: 'oj-ai.adapt.v1', label: '管理员说明、参数与更正' },
  { id: 'knowledge', key: 'oj-ai.kb.v1', label: '本地知识库' },
  { id: 'imageAdmin', key: 'oj-ai.image.v1', label: '生图管理员参数、风格提示、更正与知识库' },
  { id: 'localModels', key: 'oj-ai.localModels', label: '新建AI清单（不含模型权重）' },
];

export const NEVER_EXPORTED_KEYS = ['oj-ai.admin', 'oj-ai.token', 'oj-ai.unlock'];

/**
 * 按所选分区生成导出对象。
 * @param {Storage|{getItem(k: string): string|null}} storage
 * @param {string[]} sectionIds 要导出的分区 ID（未知的 ID 会被忽略）
 * @returns {{format: string, version: number, exportedAt: string, sections: Object, missing: string[]}}
 */
export function buildDataExport(storage, sectionIds, { now = new Date() } = {}) {
  const wanted = new Set(sectionIds || []);
  const sections = {};
  const missing = [];
  for (const sec of DATA_SECTIONS) {
    if (!wanted.has(sec.id)) continue;
    if (NEVER_EXPORTED_KEYS.includes(sec.key)) continue;
    const raw = storage.getItem(sec.key);
    if (raw === null || raw === undefined) {
      missing.push(sec.id);
      continue;
    }
    // 尽量保存为结构化数据；无法解析时保留原始文本
    try {
      sections[sec.id] = JSON.parse(raw);
    } catch {
      sections[sec.id] = raw;
    }
  }
  return {
    format: DATA_EXPORT_FORMAT,
    version: DATA_EXPORT_VERSION,
    exportedAt: now.toISOString(),
    sections,
    missing,
  };
}

/** 默认文件名：oj-ai-data-YYYY-MM-DD.json */
export function dataExportFilename(now = new Date()) {
  const d = now.toISOString().slice(0, 10);
  return `oj-ai-data-${d}.json`;
}
