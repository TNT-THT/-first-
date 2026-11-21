/**
 * 用量统计核心（纯逻辑，无 DOM 依赖；网页与服务端共用，保证两边口径一致）。
 *
 * 设计目标：省空间、O(1) 记录、不脆弱。
 * - 只存【聚合计数】，不存每次调用的完整记录：按模型汇总（调用数、失败数、
 *   提示/生成 Token），外加每模型最近若干天的日汇总；另保留一小段“近期日志”
 *   （定长环形列表）供面板查看。数据体积有硬上限，不会随使用时长无限增长。
 * - recordUsage 只做哈希表更新与定长裁剪，单次 O(1)（裁剪仅在超限时发生，
 *   且规模被上限锁住）；汇总查询只排序模型列表（模型种类数很小）。
 * - 所有读取都经过 normalizeUsageStats 的防御性清洗：存储被写坏、旧版本或
 *   人工篡改时一律回退为合法结构，绝不抛错、绝不让坏数据毒化后续统计。
 */

export const USAGE_STORAGE_KEY = 'oj-ai.usage.v1';
export const USAGE_STATS_VERSION = 1;
/** 近期日志最多保留条数（再旧的只留在聚合计数里）。 */
export const MAX_RECENT_ENTRIES = 60;
/** 每模型保留的日汇总天数。 */
export const MAX_USAGE_DAYS = 31;
/** 最多统计的模型种类数（超出时淘汰最久未用的，防导入模型刷爆存储）。 */
export const MAX_USAGE_MODELS = 100;

const MAX_COUNT = 1e15;

export function emptyUsageStats() {
  return { v: USAGE_STATS_VERSION, models: {}, recent: [] };
}

/** 把任意值清洗为非负整数计数（天文数字封顶，防溢出与坏数据）。 */
export function cleanCount(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v <= 0) return 0;
  return Math.min(MAX_COUNT, Math.floor(v));
}

/** 本地日期键 YYYY-MM-DD（按本地时区，与用户看日历的直觉一致）。 */
export function dayKeyOf(ts) {
  const d = new Date(Number.isFinite(ts) ? ts : Date.now());
  const pad = (x) => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function cleanDayMap(raw) {
  if (!raw || typeof raw !== 'object') return {};
  const out = {};
  for (const [day, v] of Object.entries(raw)) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || !v || typeof v !== 'object') continue;
    out[day] = {
      calls: cleanCount(v.calls),
      promptTokens: cleanCount(v.promptTokens),
      completionTokens: cleanCount(v.completionTokens),
    };
  }
  // 只留最近 MAX_USAGE_DAYS 天（键按日期字典序即时间序）
  const keys = Object.keys(out).sort();
  while (keys.length > MAX_USAGE_DAYS) delete out[keys.shift()];
  return out;
}

function cleanRecent(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const e of raw) {
    if (!e || typeof e !== 'object') continue;
    out.push({
      t: Number.isFinite(Number(e.t)) ? Number(e.t) : 0,
      model: typeof e.model === 'string' && e.model ? e.model.slice(0, 200) : 'unknown',
      source: e.source === 'api' ? 'api' : 'web',
      promptTokens: cleanCount(e.promptTokens),
      completionTokens: cleanCount(e.completionTokens),
      status: e.status === 'error' || e.status === 'stopped' ? e.status : 'ok',
    });
    if (out.length >= MAX_RECENT_ENTRIES) break;
  }
  return out;
}

/**
 * 把任意来源的数据（localStorage / JSON 文件）清洗为合法统计结构。
 * 任何异常输入都回退为对应部分的空值，永不抛错。
 */
export function normalizeUsageStats(raw) {
  const out = emptyUsageStats();
  if (!raw || typeof raw !== 'object') return out;
  if (raw.models && typeof raw.models === 'object') {
    for (const [id, m] of Object.entries(raw.models)) {
      if (!id || typeof id !== 'string' || !m || typeof m !== 'object') continue;
      out.models[id.slice(0, 200)] = {
        calls: cleanCount(m.calls),
        errors: cleanCount(m.errors),
        promptTokens: cleanCount(m.promptTokens),
        completionTokens: cleanCount(m.completionTokens),
        lastUsedAt: Number.isFinite(Number(m.lastUsedAt)) ? Number(m.lastUsedAt) : 0,
        days: cleanDayMap(m.days),
      };
    }
    pruneModels(out);
  }
  out.recent = cleanRecent(raw.recent);
  return out;
}

/** 模型种类超限时淘汰最久未用的（罕见路径，O(模型数) 可接受）；keepId 永不淘汰。 */
function pruneModels(stats, keepId = '') {
  const ids = Object.keys(stats.models).filter((id) => id !== keepId);
  const excess = ids.length + (keepId && stats.models[keepId] ? 1 : 0) - MAX_USAGE_MODELS;
  if (excess <= 0) return;
  ids
    .sort((a, b) => (stats.models[a].lastUsedAt || 0) - (stats.models[b].lastUsedAt || 0))
    .slice(0, excess)
    .forEach((id) => delete stats.models[id]);
}

function pruneDays(modelStats) {
  const keys = Object.keys(modelStats.days).sort();
  while (keys.length > MAX_USAGE_DAYS) delete modelStats.days[keys.shift()];
}

/**
 * 记录一次用量。就地更新并返回 stats。
 * entry: { model, source: 'web'|'api', promptTokens, completionTokens, status: 'ok'|'error'|'stopped' }
 * status 为 error 时只增加调用/失败计数，不计 Token（失败通常没有完整用量）；
 * stopped 表示用户中途停止，已产生部分的 Token 照常计入。
 */
export function recordUsage(stats, entry = {}, now = Date.now()) {
  const target = stats && typeof stats === 'object' && stats.models && typeof stats.models === 'object'
    ? stats
    : emptyUsageStats();
  const model = typeof entry.model === 'string' && entry.model.trim() ? entry.model.trim().slice(0, 200) : 'unknown';
  const status = entry.status === 'error' || entry.status === 'stopped' ? entry.status : 'ok';
  const promptTokens = status === 'error' ? 0 : cleanCount(entry.promptTokens);
  const completionTokens = status === 'error' ? 0 : cleanCount(entry.completionTokens);

  let m = target.models[model];
  if (!m) {
    m = target.models[model] = { calls: 0, errors: 0, promptTokens: 0, completionTokens: 0, lastUsedAt: 0, days: {} };
    pruneModels(target, model);
    m = target.models[model];
  }
  m.calls += 1;
  if (status === 'error') m.errors += 1;
  m.promptTokens += promptTokens;
  m.completionTokens += completionTokens;
  m.lastUsedAt = now;

  const day = dayKeyOf(now);
  const d = m.days[day] || (m.days[day] = { calls: 0, promptTokens: 0, completionTokens: 0 });
  d.calls += 1;
  d.promptTokens += promptTokens;
  d.completionTokens += completionTokens;
  pruneDays(m);

  if (!Array.isArray(target.recent)) target.recent = [];
  target.recent.unshift({
    t: now,
    model,
    source: entry.source === 'api' ? 'api' : 'web',
    promptTokens,
    completionTokens,
    status,
  });
  if (target.recent.length > MAX_RECENT_ENTRIES) target.recent.length = MAX_RECENT_ENTRIES;
  target.v = USAGE_STATS_VERSION;
  return target;
}

/** 清空统计（返回新的空结构，不改动入参引用之外的状态）。 */
export function clearUsageStats() {
  return emptyUsageStats();
}

/**
 * 汇总为面板可直接渲染的结构：总计 + 每模型行（按总 Token 降序） + 最近 N 天序列。
 * models 规模被 MAX_USAGE_MODELS 锁住，排序开销可忽略。
 */
export function summarizeUsage(stats, { days = 7, now = Date.now() } = {}) {
  const s = normalizeUsageStats(stats);
  const dayCount = Math.min(31, Math.max(1, Math.floor(days) || 7));
  const dayKeys = [];
  for (let i = dayCount - 1; i >= 0; i--) dayKeys.push(dayKeyOf(now - i * 86400000));
  const lastDays = dayKeys.map((day) => ({ day, calls: 0, promptTokens: 0, completionTokens: 0, totalTokens: 0 }));

  let totalCalls = 0;
  let totalErrors = 0;
  let promptTokens = 0;
  let completionTokens = 0;
  const models = [];
  for (const [id, m] of Object.entries(s.models)) {
    totalCalls += m.calls;
    totalErrors += m.errors;
    promptTokens += m.promptTokens;
    completionTokens += m.completionTokens;
    const dayMap = new Map(lastDays.map((d) => [d.day, d]));
    for (const [day, dv] of Object.entries(m.days)) {
      const slot = dayMap.get(day);
      if (slot) {
        slot.calls += dv.calls;
        slot.promptTokens += dv.promptTokens;
        slot.completionTokens += dv.completionTokens;
        slot.totalTokens = slot.promptTokens + slot.completionTokens;
      }
    }
    const today = m.days[dayKeyOf(now)] || { calls: 0, promptTokens: 0, completionTokens: 0 };
    models.push({
      id,
      calls: m.calls,
      errors: m.errors,
      promptTokens: m.promptTokens,
      completionTokens: m.completionTokens,
      totalTokens: m.promptTokens + m.completionTokens,
      lastUsedAt: m.lastUsedAt,
      today: { ...today, totalTokens: today.promptTokens + today.completionTokens },
    });
  }
  models.sort((a, b) => b.totalTokens - a.totalTokens || b.calls - a.calls || a.id.localeCompare(b.id));
  return {
    totalCalls,
    totalErrors,
    promptTokens,
    completionTokens,
    totalTokens: promptTokens + completionTokens,
    models,
    lastDays,
    recent: s.recent,
  };
}

/** 从 Web Storage 读取（损坏/不可用时返回空统计，永不抛错）。 */
export function loadUsageStats(storage, key = USAGE_STORAGE_KEY) {
  try {
    const raw = storage?.getItem(key);
    if (!raw) return emptyUsageStats();
    return normalizeUsageStats(JSON.parse(raw));
  } catch {
    return emptyUsageStats();
  }
}

/** 写入 Web Storage（配额满/隐私模式等失败时静默忽略，统计不能影响聊天）。 */
export function saveUsageStats(stats, storage, key = USAGE_STORAGE_KEY) {
  if (!storage || typeof storage.setItem !== 'function') return false;
  try {
    storage.setItem(key, JSON.stringify(normalizeUsageStats(stats)));
    return true;
  } catch {
    return false;
  }
}

/** 千分位格式化（如 12,345）；非法值显示 0。 */
export function formatTokens(n) {
  const v = cleanCount(n);
  return v.toLocaleString('en-US');
}
