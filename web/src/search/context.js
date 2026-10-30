/**
 * 搜索结果整理成提示词 / 界面卡片（纯逻辑，便于测试）。
 *
 * 设计要点：
 * - 搜索结果只在“当次提问”中使用（拼进最后一条用户消息），不会写进历史，也不占之后几轮的上下文；
 *   历史里保存的是精简后的来源清单（标题、地址、摘要前若干字），仅用于回看与导出。
 * - 结果一律按纯文本处理：标题、摘要都去掉 HTML，地址只允许 http(s)，避免搜索结果被当成 HTML 注入。
 * - 总长度有上限，避免小模型的 4K 窗口被资料占满。
 */

/** 只允许 http(s) 链接；其他（javascript:、data: 等）一律丢弃。 */
export function sanitizeUrl(url) {
  try {
    const u = new URL(String(url || ''));
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return '';
    return u.href;
  } catch {
    return '';
  }
}

export function hostOf(url) {
  try {
    return new URL(String(url)).host.replace(/^www\./, '');
  } catch {
    return '';
  }
}

function oneLine(text, max) {
  const s = String(text || '')
    .replace(/\s+/g, ' ')
    .trim();
  return s.length > max ? `${s.slice(0, Math.max(0, max - 1))}…` : s;
}

/** 单条结果归一化：标题缺失时用域名，摘要截断。 */
export function normalizeResult(result, { maxSnippetChars = 240 } = {}) {
  const url = sanitizeUrl(result?.url);
  if (!url) return null;
  const title = oneLine(result?.title || '', 120) || hostOf(url) || '未命名来源';
  const snippet = oneLine(result?.snippet || '', maxSnippetChars);
  const extract = oneLine(result?.extract || '', 800);
  return {
    title,
    url,
    snippet,
    ...(extract ? { extract } : {}),
    provider: String(result?.provider || ''),
  };
}

/** 合并多个搜索源的结果：按地址去重、丢弃非法地址、限制条数。 */
export function mergeResults(results, { limit = 5, maxSnippetChars = 240 } = {}) {
  const out = [];
  const seen = new Set();
  for (const item of Array.isArray(results) ? results : []) {
    const r = normalizeResult(item, { maxSnippetChars });
    if (!r) continue;
    const key = r.url.replace(/#.*$/, '');
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(r);
    if (out.length >= Math.max(1, limit)) break;
  }
  return out;
}

/**
 * 多个搜索源的结果轮流取（round-robin），避免排在前面的来源占满全部名额。
 * groups: [{id, results}]；同一地址只保留一次。
 */
export function interleaveByProvider(groups, { limit = 5, maxSnippetChars = 240 } = {}) {
  const lists = (Array.isArray(groups) ? groups : []).map((g) => ({
    id: g?.id || '',
    items: (Array.isArray(g?.results) ? g.results : []).map((r) =>
      normalizeResult({ ...r, provider: r?.provider || g?.id || '' }, { maxSnippetChars }),
    ),
  }));
  const out = [];
  const seen = new Set();
  for (let round = 0; out.length < Math.max(1, limit); round += 1) {
    let added = false;
    for (const list of lists) {
      const item = list.items[round];
      if (!item) continue;
      added = true;
      const key = item.url.replace(/#.*$/, '');
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(item);
      if (out.length >= Math.max(1, limit)) break;
    }
    if (!added) break;
  }
  return out;
}

/** 保存到历史 / 导出的精简版本（不含正文，控制 localStorage 体积）。 */
export function compactSources(results, { limit = 5 } = {}) {
  return (Array.isArray(results) ? results : [])
    .slice(0, Math.max(1, limit))
    .map((r) => {
      const url = sanitizeUrl(r?.url);
      // 标题可能为空（个别接口只返回正文）：用域名兜底，避免界面上出现一个空白条目
      const title = oneLine(r?.title || '', 120) || hostOf(url) || url;
      return {
        title,
        url,
        snippet: oneLine(r?.snippet || '', 160),
        provider: String(r?.provider || ''),
      };
    })
    .filter((r) => r.url);
}

/**
 * 每个来源的结果状态（ok / empty / error）→ 精简后随消息保存，用于在回答下方说明
 * “为什么只搜到一个来源”。不包含任何请求内容。
 */
export function compactAttempts(attempts, { limit = 6 } = {}) {
  return (Array.isArray(attempts) ? attempts : [])
    .slice(0, Math.max(1, limit))
    .map((a) => ({
      id: String(a?.id || ''),
      status: ['ok', 'empty', 'error'].includes(a?.status) ? a.status : 'error',
      count: Number.isFinite(a?.count) ? a.count : 0,
      message: a?.message ? oneLine(a.message, 120) : '',
    }))
    .filter((a) => a.id);
}

function pad(n) {
  return String(n).padStart(2, '0');
}

function stamp(now) {
  const d = now instanceof Date ? now : new Date(now);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/**
 * 拼成发送给模型的资料块。
 * 明确告诉模型：这些是浏览器刚抓到的网页内容，可能不完整；要标注来源；资料里没有的不要编。
 */
export function buildSearchBlock(
  results,
  { query = '', mode = 'quick', provider = '', now = new Date(), maxChars = 1600 } = {},
) {
  const list = (Array.isArray(results) ? results : []).filter((r) => sanitizeUrl(r?.url));
  if (!list.length) return '';
  const head =
    `【联网搜索结果】今天是 ${stamp(now)}。以下是浏览器刚刚抓取的公开网页内容（查询：${oneLine(query, 80)}，来源：${provider || '网络'}）。` +
    '资料可能不完整、过时或有误：请优先参考这些资料回答，并在用到时用 [1]、[2] 这样的编号标注来源；资料中没有的信息不要编造。' +
    '特别注意：资料没有提到某件事，不等于这件事没有发生——不要仅凭资料里没写就断言“尚未公布 / 不存在 / 未颁发”；' +
    '如果资料与问题无关或不足以回答，请直接说明“没有搜到可靠信息”，并给出可查证的线索（例如官方页面名称）。' +
    '这些资料不是用户的原话。';
  const lines = [head];
  let used = 0;
  for (let i = 0; i < list.length; i += 1) {
    const r = list[i];
    const body = mode === 'smart' && r.extract ? r.extract : r.snippet;
    const entry = [`[${i + 1}] ${oneLine(r.title, 120)}`, `${r.url}`, body].filter(Boolean).join('\n');
    if (used + entry.length > maxChars && i > 0) break;
    lines.push(entry);
    used += entry.length;
  }
  return lines.join('\n\n');
}

/**
 * 把资料块拼到“最后一条用户消息”里（两种后端都接受这种形状，不需要额外的 system 角色）。
 * 只在本次请求中使用，不修改会话里保存的消息。
 */
export function withSearchContext(messages, block) {
  if (!block || !Array.isArray(messages) || !messages.length) return messages;
  let lastUser = -1;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (messages[i]?.role === 'user') {
      lastUser = i;
      break;
    }
  }
  if (lastUser < 0) return messages;
  const out = messages.map((m) => ({ role: m.role, content: m.content }));
  out[lastUser].content = `${block}\n\n【我的问题】\n${out[lastUser].content}`;
  return out;
}

/**
 * 界面用的来源状态列表：[{id, label, status, count, message}]，按 providerLabels 转成中文名。
 * 只在有来源“无结果 / 失败”时才值得显示。
 */
export function describeAttempts(attempts, { providerLabels = {} } = {}) {
  return (Array.isArray(attempts) ? attempts : []).map((a) => ({
    id: a?.id || '',
    label: providerLabels[a?.id] || a?.id || '来源',
    status: ['ok', 'empty', 'error'].includes(a?.status) ? a.status : 'error',
    count: Number.isFinite(a?.count) ? a.count : 0,
    message: a?.message ? oneLine(a.message, 120) : '',
  }));
}

/** 界面与状态栏里的一句话摘要，例如“已联网搜索：5 个来源（维基百科）”。 */
export function searchSummary(search, { providerLabels = {} } = {}) {
  if (!search) return '';
  if (search.status === 'searching') return `正在联网搜索：${oneLine(search.query, 40) || '…'}`;
  if (search.status === 'error') return `联网搜索失败${search.reason ? `（${search.reason}）` : ''}，本条回答未使用搜索结果`;
  const n = Array.isArray(search.results) ? search.results.length : 0;
  if (!n) return '联网搜索没有找到结果，本条回答未使用搜索结果';
  // 多个搜索源合并时，标注所有贡献了结果的来源
  const ids = Array.isArray(search.providers) && search.providers.length ? search.providers : [search.provider];
  const label = ids
    .filter(Boolean)
    .map((id) => providerLabels[id] || id)
    .join(' + ') || '网络';
  const mode = search.mode === 'smart' ? '智能搜索' : '快速搜索';
  // 有来源没有结果或请求失败时明说，避免让人以为只搜了一个网站
  const attempts = Array.isArray(search.attempts) ? search.attempts : [];
  const bad = attempts.filter((a) => a.status && a.status !== 'ok');
  const tail = bad.length
    ? `；${bad
        .slice(0, 3)
        .map((a) => `${providerLabels[a.id] || a.id}（${a.status === 'error' ? '失败' : '无结果'}）`)
        .join('、')}`
    : '';
  return `${mode}：${n} 条结果（${label}${tail}）`;
}
