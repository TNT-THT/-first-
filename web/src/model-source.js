/**
 * 模型列表与权重下载来源。
 *
 * 1. extraRecords：自定义模型记录（来自 config.json 的 webllm.extraRecords），
 *    用于保留 WebLLM 内置列表中被注释掉、但仍可使用的模型。同 model_id 的内置记录会被替换。
 * 2. mirrorOwner：设置后，所有模型都从 https://huggingface.co/<mirrorOwner>/<模型ID> 下载，
 *    即你自己在 Hugging Face 上的副本。为空时沿用官方 mlc-ai 地址。
 * 运行库（wasm）仍使用 model_lib 中指定的地址，不需要复制。
 */

const OWNER_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;

/**
 * 合并内置模型列表、自定义记录，并按需替换为自己的 Hugging Face 地址。
 * @param {Array<{model_id: string, model: string}>} builtinList WebLLM 内置的模型记录
 * @param {Array<object>} [extraRecords] 自定义模型记录，同 model_id 的内置记录会被替换
 * @param {string} [mirrorOwner] Hugging Face 用户名或组织名，留空则不替换地址
 * @returns {Array<object>} 新的 model_list
 */
export function buildModelList(builtinList, extraRecords = [], mirrorOwner = '') {
  const extras = Array.isArray(extraRecords) ? extraRecords : [];
  const extraIds = new Set(extras.map((r) => r.model_id));
  const merged = [...builtinList.filter((r) => !extraIds.has(r.model_id)), ...extras];

  const owner = String(mirrorOwner || '').trim();
  if (!owner) return merged;
  if (!OWNER_PATTERN.test(owner)) {
    throw new Error('mirrorOwner 只能包含字母、数字、点、下划线和短横线，且不能包含斜杠。');
  }
  return merged.map((record) => ({
    ...record,
    model: `https://huggingface.co/${owner}/${record.model_id}`,
  }));
}
