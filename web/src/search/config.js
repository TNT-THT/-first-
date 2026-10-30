/**
 * 联网资料检索的配置与偏好（纯逻辑，便于测试）。
 *
 * 搜索请求在浏览器中直接发送给公开接口；因此默认来源只是一组可跨域的
 * 公开资料源，并不等同于覆盖整个互联网的搜索引擎。要检索更广泛网页，
 * 需要部署者配置受信任的 SearXNG / 自定义搜索服务。
 *
 * 三份数据分开保存：
 * 1. 站点配置：config.json 的 `webSearch`；
 * 2. 用户偏好：本浏览器的开关与模式（localStorage `oj-ai.search.v1`）；
 * 3. 管理员搜索源设置：自建服务地址与正文读取模板（仅本机）。
 */

export const SEARCH_PREF_KEY = 'oj-ai.search.v1';
export const SEARCH_SETTINGS_KEY = 'oj-ai.search.settings.v1';

/** 搜索模式：quick = 资料源摘要；smart = 补充可审计的正文摘录。 */
export const SEARCH_MODES = ['quick', 'smart'];
export const SEARCH_MODE_LABELS = { quick: '快速检索', smart: '深入检索' };

/** 免费、免密钥、允许跨域的公开资料源。 */
export const KEYLESS_PROVIDERS = ['wikipedia', 'stackexchange', 'hackernews', 'duckduckgo'];
/** 需要管理员自己填地址的进阶来源。 */
export const ADVANCED_PROVIDERS = ['searxng', 'custom'];
export const ALL_PROVIDERS = [...KEYLESS_PROVIDERS, ...ADVANCED_PROVIDERS];

export const DEFAULT_WEB_SEARCH = {
  enabled: true,
  defaultOn: false,
  defaultMode: 'quick',
  maxResults: 5,
  maxSnippetChars: 240,
  // 单个请求与一次完整检索的预算分开，避免一个慢源拖住回答。
  timeoutMs: 8000,
  totalTimeoutMs: 12000,
  // 仅页面内内存缓存；缓存本身不写入 localStorage（聊天历史另有可审计保存）。
  cacheTtlMs: 5 * 60 * 1000,
  providers: ['wikipedia', 'stackexchange', 'hackernews'],
  wikipedia: { lang: 'zh', fallbackLang: 'en' },
  stackexchange: { site: 'stackoverflow' },
  searxng: { baseUrl: '' },
  custom: { endpoint: '' },
  smart: { maxPages: 3, readerTemplate: '' },
  maxContextChars: { quick: 1600, smart: 3200 },
};

const MAX_CONFIG_URL_CHARS = 2048;

function clampInt(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.round(n)));
}

function str(value, fallback = '') {
  return typeof value === 'string' ? value.trim() : fallback;
}

function isLoopbackHost(hostname) {
  const host = String(hostname || '').toLowerCase().replace(/^\[|\]$/g, '');
  return host === 'localhost' || host === '::1' || /^127(?:\.\d{1,3}){3}$/.test(host);
}

/**
 * 公开服务只允许 HTTPS；HTTP 仅允许本机调试服务，避免把查询词发到明文网络。
 * `cleanBaseUrl` 用于会在后续代码中自行添加路径/参数的 API 地址，因此去掉末尾斜杠。
 */
export function cleanBaseUrl(value) {
  const raw = str(value);
  if (!raw || raw.length > MAX_CONFIG_URL_CHARS) return '';
  try {
    const u = new URL(raw);
    if (u.username || u.password) return '';
    if (u.protocol !== 'https:' && !(u.protocol === 'http:' && isLoopbackHost(u.hostname))) return '';
    return u.href.replace(/\/+$/, '');
  } catch {
    return '';
  }
}

/**
 * 正文读取模板会把编码后的原文地址拼在末尾，或者用 `{url}` 占位。
 * 与 base URL 不同，模板的尾部 `/` / `=` 是语义的一部分，绝不能裁掉。
 */
export function cleanReaderTemplate(value) {
  const raw = str(value);
  if (!raw || raw.length > MAX_CONFIG_URL_CHARS) return '';
  const hasPlaceholder = raw.includes('{url}');
  const probe = hasPlaceholder ? raw.replaceAll('{url}', 'https%3A%2F%2Fexample.invalid%2F') : raw;
  try {
    const u = new URL(probe);
    if (u.username || u.password) return '';
    if (u.protocol !== 'https:' && !(u.protocol === 'http:' && isLoopbackHost(u.hostname))) return '';
    // 保留用户原本的占位符；普通模板使用 URL 规范化结果以保留根路径的 `/`。
    if (hasPlaceholder) return raw;
    const normalized = u.href;
    // 普通前缀必须明确有一个拼接边界，例如 /、?、& 或 =。
    if (!/[/?&=:#]$/.test(normalized)) return '';
    return normalized;
  } catch {
    return '';
  }
}

/** 由安全的正文模板生成请求地址。 */
export function readerUrlFor(template, sourceUrl) {
  const safeTemplate = cleanReaderTemplate(template);
  if (!safeTemplate) return '';
  const encoded = encodeURIComponent(String(sourceUrl || ''));
  if (!encoded) return '';
  return safeTemplate.includes('{url}') ? safeTemplate.replaceAll('{url}', encoded) : `${safeTemplate}${encoded}`;
}

/**
 * 把 config.json 的 `webSearch` 归一化：字段缺失或写错时回落到默认值，数值夹到允许范围。
 * 站点配置不可信（可能被随意修改），因此这里不用展开合并，而是逐项校验。
 */
export function normalizeWebSearch(raw) {
  const src = raw && typeof raw === 'object' ? raw : {};
  const d = DEFAULT_WEB_SEARCH;
  const providers = Array.isArray(src.providers)
    ? [...new Set(src.providers.filter((p) => typeof p === 'string').map((p) => p.trim()))].filter((p) =>
        ALL_PROVIDERS.includes(p),
      )
    : [];
  const wiki = src.wikipedia && typeof src.wikipedia === 'object' ? src.wikipedia : {};
  const se = src.stackexchange && typeof src.stackexchange === 'object' ? src.stackexchange : {};
  const searx = src.searxng && typeof src.searxng === 'object' ? src.searxng : {};
  const custom = src.custom && typeof src.custom === 'object' ? src.custom : {};
  const smart = src.smart && typeof src.smart === 'object' ? src.smart : {};
  const ctx = src.maxContextChars && typeof src.maxContextChars === 'object' ? src.maxContextChars : {};
  const langOk = (v) => (typeof v === 'string' && /^[a-z]{2,3}(-[a-z]{2,8})?$/i.test(v.trim()) ? v.trim() : '');
  return {
    enabled: src.enabled !== false,
    defaultOn: src.defaultOn === true,
    defaultMode: SEARCH_MODES.includes(src.defaultMode) ? src.defaultMode : d.defaultMode,
    maxResults: clampInt(src.maxResults, 1, 10, d.maxResults),
    maxSnippetChars: clampInt(src.maxSnippetChars, 80, 600, d.maxSnippetChars),
    timeoutMs: clampInt(src.timeoutMs, 2000, 20000, d.timeoutMs),
    totalTimeoutMs: clampInt(src.totalTimeoutMs, 3000, 30000, d.totalTimeoutMs),
    cacheTtlMs: clampInt(src.cacheTtlMs, 0, 15 * 60 * 1000, d.cacheTtlMs),
    providers: providers.length ? providers : [...d.providers],
    wikipedia: {
      lang: langOk(wiki.lang) || d.wikipedia.lang,
      fallbackLang: langOk(wiki.fallbackLang) || d.wikipedia.fallbackLang,
    },
    stackexchange: {
      site: typeof se.site === 'string' && /^[a-z0-9.]+$/.test(se.site.trim()) ? se.site.trim() : d.stackexchange.site,
    },
    searxng: { baseUrl: cleanBaseUrl(searx.baseUrl) },
    custom: { endpoint: cleanBaseUrl(custom.endpoint) },
    smart: {
      maxPages: clampInt(smart.maxPages, 0, 5, d.smart.maxPages),
      readerTemplate: cleanReaderTemplate(smart.readerTemplate),
    },
    maxContextChars: {
      quick: clampInt(ctx.quick, 400, 12000, d.maxContextChars.quick),
      smart: clampInt(ctx.smart, 400, 16000, d.maxContextChars.smart),
    },
  };
}

/** 读取用户偏好（可能是空对象；损坏时按空对象处理）。 */
export function loadSearchPrefs(storage = globalThis.localStorage) {
  try {
    const raw = storage.getItem(SEARCH_PREF_KEY);
    if (!raw) return {};
    const data = JSON.parse(raw);
    if (!data || typeof data !== 'object' || Array.isArray(data)) return {};
    const out = {};
    if (typeof data.enabled === 'boolean') out.enabled = data.enabled;
    if (SEARCH_MODES.includes(data.mode)) out.mode = data.mode;
    return out;
  } catch {
    return {};
  }
}

export function saveSearchPrefs(prefs, storage = globalThis.localStorage) {
  try {
    const out = {};
    if (typeof prefs?.enabled === 'boolean') out.enabled = prefs.enabled;
    if (SEARCH_MODES.includes(prefs?.mode)) out.mode = prefs.mode;
    storage.setItem(SEARCH_PREF_KEY, JSON.stringify(out));
  } catch {
    // 存储不可用时忽略：本次页面内仍按当前开关生效
  }
}

/** 合并“站点默认值”与“本机偏好”，得到本次实际生效的开关与模式。 */
export function resolveSearchPrefs(config, stored = {}) {
  const cfg = config || DEFAULT_WEB_SEARCH;
  return {
    enabled: typeof stored.enabled === 'boolean' ? stored.enabled : !!cfg.defaultOn,
    mode: SEARCH_MODES.includes(stored.mode) ? stored.mode : cfg.defaultMode,
  };
}

/** 读取管理员的搜索源设置（SearXNG 地址、自定义接口、正文读取模板）。 */
export function loadProviderSettings(storage = globalThis.localStorage) {
  try {
    const raw = storage.getItem(SEARCH_SETTINGS_KEY);
    const data = raw ? JSON.parse(raw) : {};
    if (!data || typeof data !== 'object') return { searxngBase: '', customEndpoint: '', readerTemplate: '' };
    return {
      searxngBase: cleanBaseUrl(data.searxngBase),
      customEndpoint: cleanBaseUrl(data.customEndpoint),
      readerTemplate: cleanReaderTemplate(data.readerTemplate),
    };
  } catch {
    return { searxngBase: '', customEndpoint: '', readerTemplate: '' };
  }
}

export function saveProviderSettings(settings, storage = globalThis.localStorage) {
  try {
    storage.setItem(
      SEARCH_SETTINGS_KEY,
      JSON.stringify({
        searxngBase: cleanBaseUrl(settings?.searxngBase),
        customEndpoint: cleanBaseUrl(settings?.customEndpoint),
        readerTemplate: cleanReaderTemplate(settings?.readerTemplate),
      }),
    );
  } catch {
    // 忽略：设置只在本机生效
  }
}

/**
 * 进阶来源（需要管理员填地址）。填写后与免费来源并行检索、合并呈现；
 * 不让一个不受信任的自定义源单独垄断本次答案。
 */
export function planAdvanced(config, settings = {}) {
  const cfg = config || DEFAULT_WEB_SEARCH;
  const plan = [];
  const add = (id, url) => {
    if (url && !plan.some((p) => p.id === id)) plan.push({ id, url });
  };
  add('custom', cleanBaseUrl(settings.customEndpoint) || cfg.custom.endpoint);
  add('searxng', cleanBaseUrl(settings.searxngBase) || cfg.searxng.baseUrl);
  return plan;
}

/** 免费公开资料源，来自 config.json 的 providers。 */
export function planKeyless(config) {
  const cfg = config || DEFAULT_WEB_SEARCH;
  return (cfg.providers || [])
    .filter((id) => KEYLESS_PROVIDERS.includes(id))
    .map((id) => ({ id, url: '' }));
}

/** 所有实际启用的来源。进阶源在前只是展示与结果排序优先级，不再排他。 */
export function planProviders(config, settings = {}) {
  return [...planAdvanced(config, settings), ...planKeyless(config)];
}

export const PROVIDER_LABELS = {
  wikipedia: '维基百科',
  stackexchange: 'Stack Overflow',
  hackernews: 'Hacker News',
  duckduckgo: 'DuckDuckGo 即时答案',
  searxng: 'SearXNG',
  custom: '自定义接口',
};

/** 搜索功能是否可用（站点未关闭、且至少有一个能用的搜索源）。 */
export function searchAvailable(config, settings = {}) {
  const cfg = config || DEFAULT_WEB_SEARCH;
  return !!cfg.enabled && planProviders(cfg, settings).length > 0;
}
