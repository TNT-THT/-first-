// 工具面板与窄屏模型切换：在 jsdom 中加载真实的 web/index.html + main.js 做冒烟测试。
//
// 这里不模拟 WebGPU（本机模型无法在 Node 中加载），只验证界面逻辑：
// 工具面板的展开 / 收起、三张卡片、开关状态与本机偏好、窄屏下拉框与侧边栏同步、
// 以及“发送 → 联网资料检索 → 来源卡片 → 来源写入会话”的完整流程（搜索用假的 fetch）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { JSDOM } from 'jsdom';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const html = readFileSync(join(root, 'web', 'index.html'), 'utf8');
const config = JSON.parse(readFileSync(join(root, 'web', 'public', 'config.json'), 'utf8'));

const dom = new JSDOM(html, { url: 'https://example.com/app/', pretendToBeVisual: true, runScripts: 'outside-only' });
const { window } = dom;

globalThis.window = window;
globalThis.document = window.document;
Object.defineProperty(globalThis, 'navigator', { value: window.navigator, configurable: true });
Object.defineProperty(globalThis, 'location', { value: window.location, configurable: true });
globalThis.localStorage = window.localStorage;
globalThis.HTMLElement = window.HTMLElement;
globalThis.Element = window.Element;
globalThis.Node = window.Node;
globalThis.Event = window.Event;
globalThis.CustomEvent = window.CustomEvent;
globalThis.DOMParser = window.DOMParser;
globalThis.MutationObserver = window.MutationObserver;
globalThis.getComputedStyle = window.getComputedStyle;
globalThis.Blob = window.Blob;
globalThis.File = window.File;
globalThis.requestAnimationFrame = (cb) => setTimeout(cb, 0);
globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
globalThis.matchMedia = (q) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {} });
window.matchMedia = globalThis.matchMedia;
if (!window.HTMLDialogElement.prototype.showModal) {
  window.HTMLDialogElement.prototype.showModal = function showModal() {
    this.open = true;
  };
}
if (!window.HTMLDialogElement.prototype.close) {
  window.HTMLDialogElement.prototype.close = function close() {
    this.open = false;
  };
}

/** 假的 fetch：站点配置走本地文件，维基百科搜索返回固定结果，其余请求报错。 */
const WIKI = {
  query: { search: [{ title: '二分查找', snippet: '一种在有序数组中查找的算法', pageid: 1 }] },
};
const SO = {
  items: [
    {
      title: 'What is the time complexity of binary search?',
      link: 'https://stackoverflow.com/questions/1',
      body: '<p>O(log n) for a sorted array.</p>',
    },
  ],
};
const HN = { hits: [{ title: 'Binary search, explained', url: 'https://example.com/bs', objectID: '1', story_text: '讨论' }] };
globalThis.fetch = async (url) => {
  const u = String(url);
  if (u.includes('wikipedia.org') && u.includes('list=search')) return { ok: true, json: async () => WIKI };
  if (u.includes('api.stackexchange.com')) return { ok: true, json: async () => SO };
  if (u.includes('hn.algolia.com')) return { ok: true, json: async () => HN };
  if (u.includes('config.json')) return { ok: true, json: async () => config };
  throw new Error(`unexpected fetch: ${u}`);
};
window.fetch = globalThis.fetch;

// main.js 顶部有 Vite 才能处理的 CSS 导入与 import.meta.env：读取源码后替换，再动态导入。
const cssImport = /^import\s+['"][^'"]+\.css['"];\s*$/gm;
const smokePath = join(root, 'web', 'src', `.ui-tools-smoke-${process.pid}.mjs`);
const source = readFileSync(join(root, 'web', 'src', 'main.js'), 'utf8')
  .replace(cssImport, '')
  .replaceAll('import.meta.env.VITE_ENGINE', 'undefined');
writeFileSync(smokePath, source);
try {
  await import(smokePath);
} finally {
  unlinkSync(smokePath);
}
await new Promise((r) => setTimeout(r, 300));

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const tick = (ms = 10) => new Promise((r) => setTimeout(r, ms));

/** 等待异步 UI 状态，避免在并行运行其它测试文件时依赖固定 400ms 时序。 */
async function waitFor(check, { timeout = 1500, interval = 20 } = {}) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const value = check();
    if (value) return value;
    await tick(interval);
  }
  return check();
}

test('启动后两个模型下拉框内容一致，窄屏下拉框可用', () => {
  assert.equal($('#login').hidden, true, '静态模式应直接进入聊天');
  assert.equal($('#main').hidden, false);
  assert.equal($$('#modelSelect option').length, config.webllm.models.length);
  assert.equal($$('#modelSelectCompact option').length, config.webllm.models.length);
  assert.equal($('#modelSelectCompact').hidden, false);
  // 窄屏下拉框显示短名称（不带“（非推理·小·…”），完整标签保留在 dataset 与悬停提示中
  assert.match($$('#modelSelectCompact option')[0].textContent, /^Qwen2\.5 3B$/);
  assert.equal($$('#modelSelectCompact option')[0].dataset.label, config.webllm.models[0].label);
});

test('工具面板：三行工具、可展开卡片、开关写入本机偏好', async () => {
  assert.equal($('#toolList').children.length, 3, '应有思考模式、滑动窗口、联网资料检索三项');
  assert.equal($('#toolsPanel').hidden, true);
  $('#toolsBtn').click();
  assert.equal($('#toolsPanel').hidden, false, '点击工具按钮应展开面板');
  assert.equal($('#toolsBtn').getAttribute('aria-expanded'), 'true');

  // 点行展开对应卡片
  $('#toolList [data-tool-open="search"]').click();
  assert.equal($('#toolCard').hidden, false, '点行应展开卡片');
  assert.match($('#toolCard').textContent, /搜索方式/);
  // 默认资料源不需要密钥，但产品不再把它们伪称为完整互联网资料检索。
  assert.match($('#toolCard').textContent, /维基百科、Stack Overflow、Hacker News/);
  assert.match($('#toolCard').textContent, /不是完整互联网索引/);
  assert.match($('#toolCard').textContent, /疑似密钥.*跳过联网检索/);
  // 非管理员不显示自建搜索源设置（用不上，也不该被误以为必须配置）
  assert.equal($('#toolCard .search-advanced'), null);
  assert.equal($$('#toolCard [data-search-mode]').length, 2);
  assert.equal($$('#toolCard [data-search-mode].is-active').length, 1);

  // 切换搜索方式：写入本机偏好，卡片同步
  $('#toolCard [data-search-mode="smart"]').click();
  assert.equal(JSON.parse(localStorage.getItem('oj-ai.search.v1')).mode, 'smart');
  assert.equal($('#toolCard [data-search-mode="smart"]').getAttribute('aria-pressed'), 'true');

  // 开启联网资料检索：角标显示已开启数量
  const searchSwitch = $('#toolList [data-tool-toggle="search"]');
  assert.equal(searchSwitch.disabled, false);
  searchSwitch.checked = true;
  searchSwitch.dispatchEvent(new window.Event('change', { bubbles: true }));
  await tick();
  assert.equal(JSON.parse(localStorage.getItem('oj-ai.search.v1')).enabled, true);
  assert.equal($('#toolsBadge').hidden, false);
  assert.equal($('#toolsBadge').textContent, '1');

  // 默认模型不是推理模型：思考开关禁用，但仍能看到原因
  const thinkSwitch = $('#toolList [data-tool-toggle="thinking"]');
  assert.equal(thinkSwitch.disabled, true);
  const thinkRow = $('#toolList [data-tool-open="thinking"]').closest('.tool-item');
  assert.equal(thinkRow.classList.contains('is-unavailable'), true);

  // 滑动窗口开关
  const slidingSwitch = $('#toolList [data-tool-toggle="sliding"]');
  assert.equal(slidingSwitch.disabled, false);
  slidingSwitch.checked = true;
  slidingSwitch.dispatchEvent(new window.Event('change', { bubbles: true }));
  await tick();
  assert.equal(localStorage.getItem('oj-ai.sliding.v1'), 'on');
  assert.equal($('#toolsBadge').textContent, '2');

  // 展开另一张卡片时，切换按钮与卡片都跟着刷新
  $('#toolList [data-tool-open="sliding"]').click();
  assert.match($('#toolCard').textContent, /滑动窗口/);
  assert.equal($$('#toolList .tool-item.is-open').length, 1);

  // 点击面板外部收起
  document.body.click();
  assert.equal($('#toolsPanel').hidden, true);
});

test('模型下拉框同步：在窄屏下拉框中切换，侧边栏跟着变并记为默认模型', async () => {
  const target = config.webllm.models[3].id;
  $('#modelSelectCompact').value = target;
  $('#modelSelectCompact').dispatchEvent(new window.Event('change', { bubbles: true }));
  await tick();
  assert.equal($('#modelSelect').value, target);
  assert.equal(localStorage.getItem('oj-ai.defaultModel.v1'), target);
});

test('发送时先联网资料检索：来源卡片显示在回答下方，并随会话保存', async () => {
  $('#input').value = '二分查找的时间复杂度是多少？请解释一下这段代码为什么 TLE。';
  $('#composer').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
  const assistant = await waitFor(() => [...document.querySelectorAll('.msg.assistant')].pop());
  assert.ok(assistant, '应出现助手消息');
  const card = await waitFor(() => assistant.querySelector('.search-card')?.dataset.key?.startsWith('done|') && assistant.querySelector('.search-card'));
  assert.ok(card, '回答下方应出现资料来源卡片');
  // 三个默认资料源并行请求并合并，摘要列出全部贡献了资料的来源。
  assert.match(card.querySelector('summary').textContent, /深入检索：3 条资料（维基百科 \+ Stack Overflow \+ Hacker News）/);
  const links = [...card.querySelectorAll('.search-sources a')];
  assert.equal(links.length, 3);
  assert.equal(links[0].textContent, '二分查找');
  assert.equal(links[0].getAttribute('href'), 'https://zh.wikipedia.org/wiki/%E4%BA%8C%E5%88%86%E6%9F%A5%E6%89%BE');
  assert.equal(links[1].getAttribute('href'), 'https://stackoverflow.com/questions/1');
  assert.equal(links[2].getAttribute('href'), 'https://example.com/bs');

  // “各来源情况”：多来源时默认折叠列出，让人看到确实搜了三个网站
  const statusBox = card.querySelector('.search-attempts-box');
  assert.ok(statusBox, '多来源时应显示各来源情况');
  const statuses = [...statusBox.querySelectorAll('.search-attempts li')].map((li) => li.textContent);
  assert.equal(statuses.length, 3);
  assert.ok(statuses.every((t) => /返回 \d+ 条资料/.test(t)), statuses.join('/'));

  // 来源清单写入本机会话：刷新页面后仍能回看
  const store = JSON.parse(localStorage.getItem('oj-ai.sessions.v1'));
  const saved = store.sessions[0].messages.find((m) => m.role === 'assistant' && m.search);
  assert.ok(saved, '来源应随会话保存');
  assert.equal(saved.search.status, 'done');
  assert.equal(saved.search.provider, 'wikipedia');
  assert.deepEqual(saved.search.providers, ['wikipedia', 'stackexchange', 'hackernews']);
  assert.equal(saved.search.results[0].title, '二分查找');
  // 各来源状态也随会话保存（不额外发请求）
  assert.deepEqual(
    saved.search.attempts.map((a) => [a.id, a.status]),
    [
      ['wikipedia', 'ok'],
      ['stackexchange', 'ok'],
      ['hackernews', 'ok'],
    ],
  );

  // Node 中没有 WebGPU：回答应给出可读的错误，而不是静默失败
  assert.match(assistant.textContent, /WebGPU/);
});

test('联网资料检索全部失败时：卡片写明每个来源的错误与网络提示，回答照常生成', async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes('wikipedia.org') || u.includes('api.stackexchange.com') || u.includes('hn.algolia.com')) {
      throw new TypeError('Failed to fetch');
    }
    if (u.includes('config.json')) return { ok: true, json: async () => config };
    throw new Error(`unexpected fetch: ${u}`);
  };
  window.fetch = globalThis.fetch;
  try {
    $('#toolsBtn').click();
    const searchSwitch = $('#toolList [data-tool-toggle="search"]');
    if (!searchSwitch.checked) {
      searchSwitch.checked = true;
      searchSwitch.dispatchEvent(new window.Event('change', { bubbles: true }));
      await tick();
    }
    document.body.click();
    $('#input').value = '二分查找的时间复杂度？';
    $('#composer').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));

    const assistant = await waitFor(() => [...document.querySelectorAll('.msg.assistant')].pop());
    const card = await waitFor(() => assistant?.querySelector('.search-card')?.dataset.key?.startsWith('error|') && assistant.querySelector('.search-card'));
    assert.ok(card, '失败时也应显示资料来源卡片');
    assert.equal(card.classList.contains('is-error'), true);
    assert.match(card.querySelector('summary').textContent, /联网资料检索未使用/);
    const failures = [...card.querySelectorAll('.search-failures li')].map((li) => li.textContent);
    assert.equal(failures.length, 3);
    assert.ok(failures.some((t) => t.startsWith('维基百科：')));
    assert.ok(failures.some((t) => t.startsWith('Stack Overflow：')));
    assert.match(card.textContent, /公开资料 API/);
  } finally {
    globalThis.fetch = realFetch;
    window.fetch = realFetch;
  }
});

test('部分来源无结果 / 失败时：卡片列出各来源情况，回答照常生成', async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes('wikipedia.org') && u.includes('list=search')) {
      return { ok: true, json: async () => WIKI };
    }
    if (u.includes('api.stackexchange.com')) return { ok: true, json: async () => ({ items: [] }) };
    if (u.includes('hn.algolia.com')) throw new TypeError('Failed to fetch');
    if (u.includes('config.json')) return { ok: true, json: async () => config };
    throw new Error(`unexpected fetch: ${u}`);
  };
  window.fetch = globalThis.fetch;
  try {
    $('#input').value = '二分查找的时间复杂度？';
    $('#composer').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));

    const assistant = await waitFor(() => [...document.querySelectorAll('.msg.assistant')].pop());
    const card = await waitFor(() => assistant?.querySelector('.search-card')?.dataset.key?.startsWith('done|') && assistant.querySelector('.search-card'));
    assert.ok(card, '有结果时也应显示资料来源卡片');
    assert.equal(card.classList.contains('is-error'), false);

    // 只有维基百科有结果，但卡片明确列出另外两个来源的空/失败状态
    const statuses = [...card.querySelectorAll('.search-attempts li')].map((li) => li.textContent);
    assert.equal(statuses.length, 3);
    assert.equal(statuses[0], '维基百科：返回 1 条资料');
    assert.equal(statuses[1], 'Stack Overflow：没有相关结果');
    assert.match(statuses[2], /^Hacker News：(请求失败|Failed to fetch)/);

    // 状态随会话保存，刷新后仍能看到“为什么只有一个来源”
    const store = JSON.parse(localStorage.getItem('oj-ai.sessions.v1'));
    const saved = store.sessions[0].messages.filter((m) => m.role === 'assistant' && m.search).pop();
    assert.deepEqual(
      saved.search.attempts.map((a) => a.status),
      ['ok', 'empty', 'error'],
    );
  } finally {
    globalThis.fetch = realFetch;
    window.fetch = realFetch;
  }
});
