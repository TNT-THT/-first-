// 联网搜索：配置、查询提取、搜索源解析、资料块与编排（纯逻辑，不访问真实网络）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  cleanBaseUrl,
  loadProviderSettings,
  loadSearchPrefs,
  normalizeWebSearch,
  planProviders,
  resolveSearchPrefs,
  saveProviderSettings,
  saveSearchPrefs,
  searchAvailable,
} from '../web/src/search/config.js';
import { buildSearchQuery, wikipediaLangs } from '../web/src/search/query.js';
import {
  parseCustom,
  parseDuckDuckGo,
  parseHackerNews,
  parseSearxng,
  parseStackExchange,
  parseWikipedia,
  stripHtml,
} from '../web/src/search/providers.js';
import {
  buildSearchBlock,
  compactSources,
  interleaveByProvider,
  mergeResults,
  sanitizeUrl,
  searchSummary,
  withSearchContext,
} from '../web/src/search/context.js';
import { runWebSearch } from '../web/src/search/index.js';

function memoryStorage(initial = {}) {
  const data = { ...initial };
  return {
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = String(v);
    },
  };
}

/** 按 URL 片段路由的假 fetch；值为 Error 时抛出，为函数时按 URL 计算响应。 */
function fakeFetch(routes) {
  return async (url) => {
    const key = Object.keys(routes).find((k) => String(url).includes(k));
    if (!key) throw new Error(`unexpected url: ${url}`);
    const value = typeof routes[key] === 'function' ? routes[key](String(url)) : routes[key];
    if (value instanceof Error) throw value;
    return {
      ok: true,
      status: 200,
      json: async () => value,
      text: async () => String(value),
    };
  };
}

const WIKI_PAYLOAD = {
  query: {
    search: [
      { title: '二分查找', snippet: '<span class="searchmatch">二分</span>查找是一种算法', pageid: 1 },
      { title: '时间复杂度', snippet: '描述运行时间', pageid: 2 },
    ],
  },
};

// ---------- 配置 ----------
test('站点配置缺失时使用默认值（三个免费免密钥来源、关闭、快速搜索）', () => {
  const cfg = normalizeWebSearch(undefined);
  assert.equal(cfg.enabled, true);
  assert.equal(cfg.defaultOn, false);
  assert.equal(cfg.defaultMode, 'quick');
  assert.equal(cfg.maxResults, 5);
  assert.deepEqual(cfg.providers, ['wikipedia', 'stackexchange', 'hackernews']);
  assert.equal(cfg.wikipedia.lang, 'zh');
  assert.equal(cfg.stackexchange.site, 'stackoverflow');
});

test('站点配置的数值夹到允许范围，未知搜索源被丢弃', () => {
  const cfg = normalizeWebSearch({
    enabled: false,
    defaultOn: true,
    defaultMode: 'deep',
    maxResults: 99,
    timeoutMs: 10,
    providers: ['wikipedia', 'evil', 'searxng', 'wikipedia'],
    wikipedia: { lang: 'ZH-CN' },
    maxContextChars: { quick: 10, smart: 999999 },
  });
  assert.equal(cfg.enabled, false);
  assert.equal(cfg.defaultOn, true);
  assert.equal(cfg.defaultMode, 'quick');
  assert.equal(cfg.maxResults, 10);
  assert.equal(cfg.timeoutMs, 2000);
  assert.deepEqual(cfg.providers, ['wikipedia', 'searxng']);
  assert.deepEqual(
    normalizeWebSearch({ providers: ['stackexchange', 'hackernews', 'duckduckgo', 'bing'] }).providers,
    ['stackexchange', 'hackernews', 'duckduckgo'],
  );
  assert.equal(cfg.wikipedia.lang, 'ZH-CN');
  assert.equal(cfg.maxContextChars.quick, 400);
  assert.equal(cfg.maxContextChars.smart, 16000);
});

test('地址只接受 http(s)，并去掉末尾斜杠', () => {
  assert.equal(cleanBaseUrl('https://searx.example.com/'), 'https://searx.example.com');
  assert.equal(cleanBaseUrl('  http://127.0.0.1:8080  '), 'http://127.0.0.1:8080');
  assert.equal(cleanBaseUrl('javascript:alert(1)'), '');
  assert.equal(cleanBaseUrl('ftp://example.com'), '');
  assert.equal(cleanBaseUrl(''), '');
});

test('用户偏好：默认取站点默认值，保存后覆盖；损坏的存储按默认处理', () => {
  const cfg = normalizeWebSearch({ defaultOn: true, defaultMode: 'smart' });
  const storage = memoryStorage();
  assert.deepEqual(resolveSearchPrefs(cfg, loadSearchPrefs(storage)), { enabled: true, mode: 'smart' });
  saveSearchPrefs({ enabled: false, mode: 'quick' }, storage);
  assert.deepEqual(resolveSearchPrefs(cfg, loadSearchPrefs(storage)), { enabled: false, mode: 'quick' });

  storage.setItem('oj-ai.search.v1', '{坏掉的 JSON');
  assert.deepEqual(loadSearchPrefs(storage), {});
  const broken = {
    getItem() {
      throw new Error('denied');
    },
    setItem() {
      throw new Error('denied');
    },
  };
  assert.deepEqual(loadSearchPrefs(broken), {});
  assert.doesNotThrow(() => saveSearchPrefs({ enabled: true }, broken));
});

test('管理员搜索源设置只保存合法地址', () => {
  const storage = memoryStorage();
  assert.deepEqual(loadProviderSettings(storage), { searxngBase: '', customEndpoint: '', readerTemplate: '' });
  saveProviderSettings({ searxngBase: 'https://s.example.com/', customEndpoint: 'javascript:x', readerTemplate: '' }, storage);
  assert.deepEqual(loadProviderSettings(storage), {
    searxngBase: 'https://s.example.com',
    customEndpoint: '',
    readerTemplate: '',
  });
});

test('搜索源顺序：管理员配置的自定义接口 > SearXNG > 站点配置；维基百科始终可作后备', () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia'] });
  assert.deepEqual(planProviders(cfg, {}).map((p) => p.id), ['wikipedia']);
  assert.deepEqual(planProviders(cfg, { searxngBase: 'https://s.example.com' }).map((p) => p.id), [
    'searxng',
    'wikipedia',
  ]);
  assert.deepEqual(
    planProviders(cfg, { customEndpoint: 'https://c.example.com', searxngBase: 'https://s.example.com' }).map((p) => p.id),
    ['custom', 'searxng', 'wikipedia'],
  );
  // 站点配置里没有维基百科、也没有任何管理员设置时：没有可用搜索源
  const none = normalizeWebSearch({ providers: ['searxng'] });
  assert.equal(searchAvailable(none, {}), false);
  assert.equal(searchAvailable(normalizeWebSearch({ enabled: false }), {}), false);
});

// ---------- 查询提取 ----------
test('搜索查询会去掉代码块、行内代码、链接与代码行', () => {
  const text = [
    '为什么这段代码会 TLE？',
    '',
    '```cpp',
    '#include <bits/stdc++.h>',
    'int main() { return 0; }',
    '```',
    '题目要求：https://hydro.ac/p/1',
    '    int n; cin >> n;',
  ].join('\n');
  const q = buildSearchQuery(text);
  assert.equal(q, '为什么这段代码会 TLE？');
});

test('整条都是代码时返回空查询（此时不搜索）', () => {
  assert.equal(buildSearchQuery('```py\nprint(1)\n```'), '');
  assert.equal(buildSearchQuery('int main() { return 0; }'), '');
  assert.equal(buildSearchQuery('   '), '');
});

test('查询最长 maxChars，超长句被截断', () => {
  const q = buildSearchQuery('很长的问题'.repeat(50), { maxChars: 20 });
  assert.equal(q.length, 20);
});

test('维基百科语言：中文问题先中文站，英文问题先英文站', () => {
  assert.deepEqual(wikipediaLangs('二分查找怎么实现'), ['zh', 'en']);
  assert.deepEqual(wikipediaLangs('binary search'), ['en', 'zh']);
  assert.deepEqual(wikipediaLangs('二分搜索 binary'), ['zh', 'en']);
});

// ---------- 解析 ----------
test('维基百科结果：去掉高亮标签，地址指向对应语言的条目', () => {
  const results = parseWikipedia(WIKI_PAYLOAD, { lang: 'zh' });
  assert.equal(results.length, 2);
  assert.equal(results[0].title, '二分查找');
  assert.equal(results[0].snippet, '二分查找是一种算法');
  assert.equal(results[0].url, 'https://zh.wikipedia.org/wiki/%E4%BA%8C%E5%88%86%E6%9F%A5%E6%89%BE');
  assert.equal(results[0].provider, 'wikipedia');
});

test('SearXNG 与自定义接口的字段名容错（title/url/snippet 的常见别名）', () => {
  const searx = parseSearxng({
    results: [
      { title: 'A', url: 'https://a.example.com', content: '摘要 A' },
      { title: '', link: 'https://b.example.com', snippet: '摘要 B' },
      { title: '没有地址' },
    ],
  });
  assert.deepEqual(searx.map((r) => r.url), ['https://a.example.com', 'https://b.example.com']);
  assert.equal(searx[1].title, 'https://b.example.com');

  const custom = parseCustom({ data: [{ name: 'C', href: 'https://c.example.com', description: '摘要 C' }] });
  assert.equal(custom[0].title, 'C');
  assert.equal(custom[0].snippet, '摘要 C');

  assert.deepEqual(parseCustom({ nope: 1 }), []);
  assert.equal(stripHtml('&lt;b&gt;x&lt;/b&gt; &amp; y'), '<b>x</b> & y');
});

// ---------- 合并与资料块 ----------
test('只允许 http(s) 地址，合并时去重并限制条数', () => {
  assert.equal(sanitizeUrl('https://a.example.com/x'), 'https://a.example.com/x');
  assert.equal(sanitizeUrl('javascript:alert(1)'), '');
  assert.equal(sanitizeUrl('data:text/html,hi'), '');
  assert.equal(sanitizeUrl('not a url'), '');

  const merged = mergeResults(
    [
      { title: 'A', url: 'https://a.example.com/1', snippet: 'a' },
      { title: 'A2', url: 'https://a.example.com/1#frag', snippet: 'a2' },
      { title: 'B', url: 'https://b.example.com/2', snippet: 'b' },
      { title: '非法', url: 'javascript:alert(1)' },
      { title: 'C', url: 'https://c.example.com/3' },
    ],
    { limit: 2 },
  );
  assert.deepEqual(merged.map((r) => r.url), ['https://a.example.com/1', 'https://b.example.com/2']);
  assert.equal(merged[0].title, 'A');
});

test('资料块包含编号、地址、时间与“资料不是用户原话”的说明，且受字数上限约束', () => {
  const now = new Date(2026, 9, 30, 9, 5);
  const results = mergeResults([
    { title: '二分查找', url: 'https://zh.wikipedia.org/wiki/a', snippet: '摘要一' },
    { title: '时间复杂度', url: 'https://zh.wikipedia.org/wiki/b', snippet: '摘要二' },
  ]);
  const block = buildSearchBlock(results, { query: '二分查找', mode: 'quick', provider: '维基百科', now });
  assert.match(block, /【联网搜索结果】/);
  assert.match(block, /2026-10-30 09:05/);
  assert.match(block, /查询：二分查找/);
  assert.match(block, /\[1\] 二分查找/);
  assert.match(block, /https:\/\/zh\.wikipedia\.org\/wiki\/a/);
  assert.match(block, /资料不是用户的原话/);

  const short = buildSearchBlock(results, { query: 'q', maxChars: 120 });
  assert.ok(short.length < block.length + 400);
  assert.equal(buildSearchBlock([], {}), '');
});

test('智能搜索优先使用正文摘要', () => {
  const results = [{ title: 'A', url: 'https://a.example.com', snippet: '短摘要', extract: '更长的正文摘要' }];
  const block = buildSearchBlock(results, { query: 'q', mode: 'smart' });
  assert.match(block, /更长的正文摘要/);
  const quick = buildSearchBlock(results, { query: 'q', mode: 'quick' });
  assert.match(quick, /短摘要/);
  assert.equal(quick.includes('更长的正文摘要'), false);
});

test('资料块只拼到最后一条用户消息，且不修改原数组', () => {
  const messages = [
    { role: 'user', content: '第一问' },
    { role: 'assistant', content: '第一答' },
    { role: 'user', content: '第二问' },
  ];
  const out = withSearchContext(messages, '【资料】X');
  assert.equal(out[0].content, '第一问');
  assert.equal(out[1].content, '第一答');
  assert.match(out[2].content, /^【资料】X\n\n【我的问题】\n第二问$/);
  assert.equal(messages[2].content, '第二问');
  assert.deepEqual(withSearchContext(messages, ''), messages);
});

test('历史中保存的来源清单会被裁剪并丢弃非法地址', () => {
  const saved = compactSources(
    [
      { title: 'A', url: 'https://a.example.com', snippet: 'x'.repeat(400) },
      { title: 'B', url: 'javascript:alert(1)' },
    ],
    { limit: 5 },
  );
  assert.equal(saved.length, 1);
  assert.equal(saved[0].snippet.length <= 161, true);
});

test('来源摘要文案：搜索中 / 失败 / 成功', () => {
  assert.match(searchSummary({ status: 'searching', query: '二分查找' }), /正在联网搜索：二分查找/);
  assert.match(searchSummary({ status: 'error', reason: 'HTTP 500' }), /联网搜索失败（HTTP 500）/);
  assert.match(
    searchSummary({ status: 'done', mode: 'smart', provider: 'wikipedia', results: [{ url: 'https://a' }] }, {
      providerLabels: { wikipedia: '维基百科' },
    }),
    /智能搜索：1 个来源（维基百科）/,
  );
});

// ---------- 编排 ----------
test('runWebSearch：维基百科成功返回结果，并把查询与模式带回来', async () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia'] });
  const res = await runWebSearch('二分查找的时间复杂度是多少？', {
    config: cfg,
    prefs: { enabled: true, mode: 'quick' },
    settings: {},
    fetchImpl: fakeFetch({ 'list=search': WIKI_PAYLOAD }),
  });
  assert.equal(res.ok, true);
  assert.equal(res.provider, 'wikipedia');
  assert.equal(res.results.length, 2);
  assert.equal(res.query, '二分查找的时间复杂度是多少？');
});

test('runWebSearch：第一个搜索源失败时自动换下一个，并记录错误', async () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia'] });
  const res = await runWebSearch('二分查找', {
    config: cfg,
    prefs: { enabled: true, mode: 'quick' },
    settings: { searxngBase: 'https://searx.example.com' },
    fetchImpl: fakeFetch({
      'searx.example.com': new Error('HTTP 403'),
      'list=search': WIKI_PAYLOAD,
    }),
  });
  assert.equal(res.ok, true);
  assert.equal(res.provider, 'wikipedia');
  assert.equal(res.errors[0].id, 'searxng');
  assert.match(res.errors[0].message, /403/);
});

test('runWebSearch：配置关闭、没有查询、全部失败时都给出原因', async () => {
  const disabled = await runWebSearch('二分查找', {
    config: normalizeWebSearch({ enabled: false }),
    prefs: { enabled: true, mode: 'quick' },
    fetchImpl: fakeFetch({}),
  });
  assert.equal(disabled.ok, false);
  assert.match(disabled.reason, /关闭/);

  const empty = await runWebSearch('```cpp\nint main(){}\n```', {
    config: normalizeWebSearch({}),
    prefs: { enabled: true, mode: 'quick' },
    fetchImpl: fakeFetch({}),
  });
  assert.equal(empty.ok, false);
  assert.match(empty.reason, /没有/);

  const failed = await runWebSearch('二分查找', {
    config: normalizeWebSearch({}),
    prefs: { enabled: true, mode: 'quick' },
    fetchImpl: fakeFetch({ 'list=search': new Error('网络不可用') }),
  });
  assert.equal(failed.ok, false);
  assert.match(failed.reason, /网络不可用/);
});

test('runWebSearch：搜索源返回空结果时继续尝试下一个', async () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia'] });
  const res = await runWebSearch('二分查找', {
    config: cfg,
    prefs: { enabled: true, mode: 'quick' },
    settings: { customEndpoint: 'https://c.example.com/search' },
    fetchImpl: fakeFetch({
      'c.example.com': { results: [] },
      'list=search': WIKI_PAYLOAD,
    }),
  });
  assert.equal(res.ok, true);
  assert.equal(res.provider, 'wikipedia');
  assert.equal(res.errors[0].id, 'custom');
});

test('runWebSearch：智能模式会补抓正文摘要', async () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia'], smart: { maxPages: 2 } });
  const res = await runWebSearch('二分查找', {
    config: cfg,
    prefs: { enabled: true, mode: 'smart' },
    fetchImpl: fakeFetch({
      'list=search': WIKI_PAYLOAD,
      'prop=extracts': {
        query: { pages: { 1: { title: '二分查找', extract: '二分查找是一种在有序数组中查找目标的方法。' } } },
      },
    }),
  });
  assert.equal(res.ok, true);
  assert.equal(res.mode, 'smart');
  assert.match(res.results[0].extract, /有序数组/);
  const block = buildSearchBlock(res.results, { query: res.query, mode: res.mode });
  assert.match(block, /有序数组/);
});

test('runWebSearch：用户停止时抛出 AbortError，不吞掉取消', async () => {
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    () =>
      runWebSearch('二分查找', {
        config: normalizeWebSearch({}),
        prefs: { enabled: true, mode: 'quick' },
        fetchImpl: fakeFetch({ 'list=search': WIKI_PAYLOAD }),
        signal: controller.signal,
      }),
    (err) => err.name === 'AbortError',
  );
});

// ---------- 免费、免密钥来源 ----------
test('免费来源并行请求并合并：结果按来源轮流取，某个来源失败不影响其它', async () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia', 'stackexchange', 'hackernews'] });
  const fetchImpl = fakeFetch({
    'list=search': {
      query: {
        search: [
          { title: '二分查找', snippet: '维基条目', pageid: 1 },
          { title: '时间复杂度', snippet: '维基条目 2', pageid: 2 },
        ],
      },
    },
    'api.stackexchange.com': new Error('CORS 被拦截'),
    'hn.algolia.com': {
      hits: [
        { title: 'Binary search explained', url: 'https://example.com/bs', objectID: '1', story_text: '讨论文本' },
        { title: 'Ask HN: 面试题', objectID: '2', story_text: '没有外链时用讨论页地址' },
      ],
    },
  });
  const res = await runWebSearch('二分查找', {
    config: cfg,
    prefs: { enabled: true, mode: 'quick' },
    fetchImpl,
  });
  assert.equal(res.ok, true);
  assert.deepEqual(res.providers, ['wikipedia', 'hackernews']);
  // 轮流取：维基 1 条、HN 1 条、维基 1 条…
  assert.equal(res.results[0].title, '二分查找');
  assert.equal(res.results[1].title, 'Binary search explained');
  assert.equal(res.results[2].title, '时间复杂度');
  assert.equal(res.results.length, 4);
  assert.deepEqual(
    res.results.map((r) => r.title),
    ['二分查找', 'Binary search explained', '时间复杂度', 'Ask HN: 面试题'],
  );
  assert.equal(res.results.find((r) => r.title === 'Ask HN: 面试题').url, 'https://news.ycombinator.com/item?id=2');
  assert.ok(res.errors.some((e) => e.id === 'stackexchange'));
  assert.equal(res.provider, 'wikipedia');
});

test('管理员配置了 SearXNG / 自定义接口时优先使用，不再请求免费来源', async () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia', 'stackexchange', 'hackernews'] });
  const calls = [];
  const fetchImpl = async (url) => {
    calls.push(String(url));
    if (String(url).includes('searx.example.com')) {
      return { ok: true, json: async () => ({ results: [{ title: 'S', url: 'https://s.example.com/1', content: '摘要' }] }) };
    }
    throw new Error(`不该请求：${url}`);
  };
  const res = await runWebSearch('二分查找', {
    config: cfg,
    prefs: { enabled: true, mode: 'quick' },
    settings: { searxngBase: 'https://searx.example.com' },
    fetchImpl,
  });
  assert.equal(res.ok, true);
  assert.deepEqual(res.providers, ['searxng']);
  assert.equal(calls.length, 1);
});

test('进阶来源失败时回落到免费来源', async () => {
  const cfg = normalizeWebSearch({ providers: ['wikipedia'] });
  const res = await runWebSearch('二分查找', {
    config: cfg,
    prefs: { enabled: true, mode: 'quick' },
    settings: { searxngBase: 'https://searx.example.com' },
    fetchImpl: fakeFetch({
      'searx.example.com': new Error('HTTP 403'),
      'list=search': WIKI_PAYLOAD,
    }),
  });
  assert.equal(res.ok, true);
  assert.deepEqual(res.providers, ['wikipedia']);
  assert.equal(res.errors[0].id, 'searxng');
});

test('Stack Exchange 与 Hacker News 的解析：去掉 HTML、按上限截断摘要', () => {
  const so = parseStackExchange(
    {
      items: [
        {
          title: 'How does &quot;binary search&quot; work?',
          link: 'https://stackoverflow.com/q/1',
          body: '<p>Use <code>mid</code> and compare...' + 'x'.repeat(500) + '</p>',
          score: 10,
        },
        { title: '没有链接', body: 'x' },
      ],
    },
    { maxSnippetChars: 60 },
  );
  assert.equal(so.length, 1);
  assert.equal(so[0].title, 'How does "binary search" work?');
  assert.ok(so[0].snippet.length <= 60);
  assert.ok(so[0].snippet.startsWith('Use mid and compare…') || so[0].snippet.startsWith('Use mid and compare'));
  assert.equal(so[0].provider, 'stackexchange');

  const hn = parseHackerNews({
    hits: [
      { title: 'Show HN: 搜索', url: 'https://example.com/a', objectID: '10', story_text: '<p>说明</p>' },
      { title: 'Ask HN', objectID: '11' },
    ],
  });
  assert.equal(hn[0].snippet, '说明');
  assert.equal(hn[1].url, 'https://news.ycombinator.com/item?id=11');
});

test('DuckDuckGo 即时答案：摘要与相关主题（默认不启用）', () => {
  const ddg = parseDuckDuckGo({
    Heading: 'Binary search',
    AbstractText: 'In computer science, binary search is...',
    AbstractURL: 'https://en.wikipedia.org/wiki/Binary_search_algorithm',
    RelatedTopics: [
      { FirstURL: 'https://example.com/1', Text: '相关主题一' },
      { Topics: [{ FirstURL: 'https://example.com/2', Text: '相关主题二' }] },
      { Text: '没有地址' },
    ],
  });
  assert.equal(ddg.length, 3);
  assert.equal(ddg[0].provider, 'duckduckgo');
  assert.equal(ddg[2].url, 'https://example.com/2');
  const cfg = normalizeWebSearch(undefined);
  assert.equal(cfg.providers.includes('duckduckgo'), false);
});

test('interleaveByProvider：轮流取、去重、受条数上限约束', () => {
  const merged = interleaveByProvider(
    [
      { id: 'a', results: [{ title: 'A1', url: 'https://a/1' }, { title: 'A2', url: 'https://a/2' }] },
      { id: 'b', results: [{ title: 'B1', url: 'https://b/1' }, { title: 'B2', url: 'https://b/2' }] },
    ],
    { limit: 3 },
  );
  assert.deepEqual(merged.map((r) => r.title), ['A1', 'B1', 'A2']);
  const deduped = interleaveByProvider(
    [
      { id: 'a', results: [{ title: 'A1', url: 'https://x/1' }] },
      { id: 'b', results: [{ title: 'B1', url: 'https://x/1#frag' }] },
    ],
    { limit: 5 },
  );
  assert.equal(deduped.length, 1);
  assert.deepEqual(interleaveByProvider([], { limit: 5 }), []);
});

test('来源摘要：多个搜索源时一起列出', () => {
  const text = searchSummary(
    { status: 'done', mode: 'quick', provider: 'wikipedia', providers: ['wikipedia', 'stackexchange'], results: [{ url: 'https://a' }] },
    { providerLabels: { wikipedia: '维基百科', stackexchange: 'Stack Overflow' } },
  );
  assert.match(text, /快速搜索：1 个来源（维基百科 \+ Stack Overflow）/);
});
