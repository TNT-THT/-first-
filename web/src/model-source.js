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
 * 本地“新建 AI”的虚拟地址前缀。这些模型的文件只存在于浏览器缓存中，
 * 地址只用作缓存键，不会真正访问网络，也不会被镜像设置改写。
 */
export const LOCAL_MODEL_PREFIX = 'https://local.oj-ai.invalid/models/';

export function isLocalRecord(record) {
  return String(record?.model || '').startsWith(LOCAL_MODEL_PREFIX);
}

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
  return merged.map((record) =>
    isLocalRecord(record)
      ? record
      : { ...record, model: `https://huggingface.co/${owner}/${record.model_id}` }
  );
}

/** 默认上下文窗口（token）。可在 config.json 的 webllm.contextWindowTokens 中修改。 */
export const DEFAULT_CONTEXT_WINDOW = 65536;

/** 显存不足时依次尝试的更小窗口（token）。 */
export const CONTEXT_WINDOW_LADDER = [65536, 32768, 16384, 8192, 4096];

/**
 * 加载时尝试的窗口序列：先请求的窗口，再依次更小的窗口（只取比它小的档位）。
 * 非法或缺省的请求值使用默认窗口。
 */
export function contextWindowLadder(target) {
  const t = Number(target);
  const top = Number.isFinite(t) && t > 0 ? Math.floor(t) : DEFAULT_CONTEXT_WINDOW;
  return [...new Set([top, ...CONTEXT_WINDOW_LADDER.filter((w) => w < top)])];
}

/**
 * 为每条模型记录设置运行时的 context_window_size（WebLLM 会把 overrides 合并进模型配置，
 * 并据此分配 KV 缓存）。不修改原记录；带有滑动窗口的记录保持不变。
 */
export function applyContextWindow(records, tokens) {
  const n = Number(tokens);
  if (!Number.isFinite(n) || n <= 0) return records;
  return records.map((r) => {
    if (Number(r.overrides?.sliding_window_size) > 0) return r;
    return { ...r, overrides: { ...(r.overrides || {}), context_window_size: Math.floor(n) } };
  });
}

/**
 * 是否为显存不足一类的加载错误（此时换用更小的窗口重试可能成功）。
 * 其他错误（如 WebGPU 不可用、shader-f16 不支持、网络错误）直接报告，不重试。
 */
export function isMemoryError(err) {
  const name = String(err?.name || '');
  const msg = String(err?.message || err || '');
  return (
    name === 'DeviceLostError' ||
    name === 'GPUOutOfMemoryError' ||
    /out of memory|\bOOM\b|device was lost|createBuffer|maxBufferSize/i.test(msg)
  );
}
