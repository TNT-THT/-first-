/**
 * 待转换的第三方模型（仅登记，尚不能在本站运行）。
 *
 * - 这些模型只有 GGUF 等非 MLC 格式，WebLLM 无法加载，因此不放进 webllm.models，
 *   也不进入 training/catalog.json 的一致性检查；它们只在下拉框中以禁用的“待转换”分组显示，并附备注。
 * - 每个待转换模型使用独立的 model_id，适应记录（更正、知识库、参数）按 model_id 分开保存，
 *   与官方原版互不影响；只有管理员显式执行“跨模型迁移”才会复制数据。
 * - 转换完成后，把条目移到 webllm.models，并在 extraRecords 中补充 model / model_lib 记录即可启用。
 */

export const PENDING_STATUS = 'pending';

/**
 * 读取 config.json 中的 webllm.pendingModels，只保留有效、未启用且 id 不与官方模型冲突的条目。
 * @param {object} webllmCfg config.json 中的 webllm 段
 * @returns {Array<{id: string, label: string, note: string, source: string}>}
 */
export function pendingModels(webllmCfg = {}) {
  const officialIds = new Set([
    ...(webllmCfg.models || []).map((m) => m.id),
    ...(webllmCfg.extraRecords || []).map((r) => r.model_id),
  ]);
  const seen = new Set();
  const out = [];
  for (const raw of Array.isArray(webllmCfg.pendingModels) ? webllmCfg.pendingModels : []) {
    const id = String(raw?.id || '').trim();
    const label = String(raw?.label || '').trim();
    if (!id || !label || raw.status !== PENDING_STATUS) continue;
    if (officialIds.has(id) || seen.has(id)) continue;
    seen.add(id);
    out.push({
      id,
      label,
      note: String(raw.note || '').trim(),
      source: String(raw.source || '').trim(),
    });
  }
  return out;
}
