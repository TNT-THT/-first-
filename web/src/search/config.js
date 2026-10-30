/**
 * 联网搜索 / 智能搜索的配置与偏好（纯逻辑，便于测试）。
 *
 * **前提：没有 API Key，也没有独立服务器。** 所有搜索都在浏览器里直接请求公开接口。
 *
 * 三份数据分开保存：
 * 1. 站点配置：config.json 的 `webSearch`，由部署者决定有没有这个功能、默认是否开启、结果条数上限；
 * 2. 用户偏好：本浏览器的开关与模式（localStorage `oj-ai.search.v1`），所有用户可改；
 * 3. 管理员搜索源设置（可选，进阶）：自建 SearXNG 地址、自定义接口地址（`oj-ai.search.settings.v1`），仅本机。
 *
 * 默认只使用“免费、免密钥、允许跨域”的公开接口，开箱即用：
 *   维基百科（概念与算法）、Stack Exchange / Stack Overflow（具体编程问题）、哔哩哔哩（中文内容）。
 * 它们会并行请求并合并结果。管理员另填了 SearXNG / 自定义接口时，优先用它们（能搜整个网页）。
 */

export const SEARCH_PREF_KEY = 'oj-ai.search.v1';
export const SEARCH_SETTINGS_KEY = 'oj-ai.search.settings.v1';

/** 搜索模式：quick = 只用搜索结果摘要；smart = 额外抓取正文摘要再交给模型。 */
export const SEARCH_MODES = ['quick', 'smart'];
export const SEARCH_MODE_LABELS = { quick: '快速搜索', smart: '智能搜索' };

/** 免费、免密钥、允许跨域的公开接口。名字里的“免费”指不需要注册或付费。 */
export const KEYLESS_PROVIDERS = ['wikipedia', 'stackexchange', 'hackernews', 'bilibili', 'duckduckgo'];
/** 需要管理员自己填地址的进阶来源。 */
export const ADVANCED_PROVIDERS = ['searxng', 'custom'];
export const ALL_PROVIDERS = [...KEYLESS_PROVIDERS, ...ADVANCED_PROVIDERS];

export const DEFAULT_WEB_SEARCH = {
  enabled: true,
  defaultOn: false,
  defaultMode: 'quick',
  maxResults: 5,
  maxSnippetChars: 240,
  timeoutMs: 8000,
  providers: ['wikipedia', 'stackexchange', 'bilibili'],
  wikipedia: { lang: 'zh', fallbackLang: 'en' },
  stackexchange: { site: 'stackoverflow' },
  searxng: { baseUrl: '' },
  custom: { endpoint: '' },
  smart: { maxPages: 3, readerTemplate: '' },
  maxContextChars: { quick: 1600, smart: 3200 },
};

function clampInt(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, Math.round(n)));
}

function str(value, fallback = '') {
  return typeof value === 'string' ? value.trim() : fallback;
}

/** 只接受 http(s) 地址；返回去掉末尾斜杠的地址，非法时返回空串。 */
export function cleanBaseUrl(value) {
  const raw = str(value);
  if (!raw) return '';
  try {
    const u = new URL(raw);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') return '';
    return u.href.replace(/\/+$/, '');
  } catch {
    return '';
  }
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
      readerTemplate: cleanBaseUrl(smart.readerTemplate),
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

/** 读取管理员的搜索源设置（SearXNG 地址、自定义接口、正文抓取模板）。 */
export function loadProviderSettings(storage = globalThis.localStorage) {
  try {
    const raw = storage.getItem(SEARCH_SETTINGS_KEY);
    const data = raw ? JSON.parse(raw) : {};
    if (!data || typeof data !== 'object') return { searxngBase: '', customEndpoint: '', readerTemplate: '' };
    return {
      searxngBase: cleanBaseUrl(data.searxngBase),
      customEndpoint: cleanBaseUrl(data.customEndpoint),
      readerTemplate: cleanBaseUrl(data.readerTemplate),
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
        readerTemplate: cleanBaseUrl(settings?.readerTemplate),
      }),
    );
  } catch {
    // 忽略：设置只在本机生效
  }
}

/**
 * 进阶来源（需要管理员填地址）。填写后优先使用，能搜到整个网页。
 * 顺序：自定义接口 > SearXNG。
 */
export function planAdvanced(config, settings = {}) {
  const cfg = config || DEFAULT_WEB_SEARCH;
  const plan = [];
  const add = (id, url) => {
    if (url && !plan.some((p) => p.id === id)) plan.push({ id, url });
  };
  add('custom', settings.customEndpoint || cfg.custom.endpoint);
  add('searxng', settings.searxngBase || cfg.searxng.baseUrl);
  return plan;
}

/**
 * 免费来源（免密钥、允许跨域），来自 config.json 的 providers。
 * 它们会并行请求并合并结果，任何一个失败都不影响其它来源。
 */
export function planKeyless(config) {
  const cfg = config || DEFAULT_WEB_SEARCH;
  return (cfg.providers || [])
    .filter((id) => KEYLESS_PROVIDERS.includes(id))
    .map((id) => ({ id, url: '' }));
}

/**
 * 展示 / 说明用的完整搜索源列表：进阶来源在前（优先），免费来源在后（并行合并）。
 */
export function planProviders(config, settings = {}) {
  return [...planAdvanced(config, settings), ...planKeyless(config)];
}

export const PROVIDER_LABELS = {
  wikipedia: '维基百科',
  stackexchange: 'Stack Overflow',
  hackernews: 'Hacker News',
  bilibili: '哔哩哔哩',
  duckduckgo: 'DuckDuckGo',
  searxng: 'SearXNG',
  custom: '自定义接口',
};

/** 搜索功能是否可用（站点未关闭、且至少有一个能用的搜索源）。 */
export function searchAvailable(config, settings = {}) {
  const cfg = config || DEFAULT_WEB_SEARCH;
  return !!cfg.enabled && planProviders(cfg, settings).length > 0;
}
