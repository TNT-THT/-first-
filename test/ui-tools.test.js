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
  assert.equal($('#login'), null, '入口登录页已删除，静态模式直接进入聊天');
  assert.equal($('#main').hidden, false);
  assert.equal($$('#modelSelect option').length, config.webllm.models.length + config.webllm.pendingModels.length);
  assert.equal($$('#modelSelectCompact option').length, config.webllm.models.length + config.webllm.pendingModels.length);
  const pendingOpt = $('#modelSelect option[value="Qwen3.8-27B"]');
  assert.ok(pendingOpt, '待转换分组应展示 Qwen3.8 27B');
  assert.equal(pendingOpt.disabled, true, '待转换模型必须禁用，不能被选中');
  assert.equal(pendingOpt.parentElement.label, '待转换（暂不可用）');
  assert.equal($('#modelSelectCompact').hidden, false);
  // 窄屏下拉框显示短名称（不带“（非推理·小·…”），完整标签保留在 dataset 与悬停提示中
  assert.match($$('#modelSelectCompact option')[0].textContent, /^Qwen2\.5 3B$/);
  assert.equal($$('#modelSelectCompact option')[0].dataset.label, config.webllm.models[0].label);
});

test('管理员面板入口：静态版名称明确，点击打开本机管理员面板', () => {
  assert.equal($('#roleBtn').textContent.trim(), '管理员面板');
  assert.equal($('#roleBtn').hidden, false);
  assert.equal($('#adminPanelBtn').hidden, true, '纯静态版没有服务端管理员面板入口');
  assert.equal($('#adminPanelBtn').textContent.trim(), 'API 管理', '服务端控制台入口应与本机管理员面板区分名称');
  assert.equal($('#localApiInfo').hidden, true, '静态版管理员面板不显示本机 API 地址（没有本机服务）');
  assert.ok($('#apiAccessBase'), 'API 接入应单独显示基础地址（Base URL），避免用户把完整对话接口填进软件');
  assert.ok($('#apiAdminCorsOrigins'), 'API 管理的参数控制台应提供跨域网站白名单设置');
  assert.match(document.body.textContent, /允许调用 API 的网站白名单/);
  assert.match(document.body.textContent, /不要填完整对话接口/);
  assert.equal($('#rolePanel').hidden, true);
  $('#roleBtn').click();
  assert.equal($('#rolePanel').hidden, false, '点击管理员面板应打开本机管理员面板');
  assert.match($('#rolePanel').textContent, /本机管理员面板/);
  assert.match($('#rolePanel').textContent, /这里管理当前浏览器里的模型/);
  $('#roleBtn').click();
  assert.equal($('#rolePanel').hidden, true);
});

test('网站包下载：点击后立即给出反馈，并走浏览器原生下载（先 HEAD 确认存在）', async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, options = {}) => {
    const u = String(url);
    if (/site-[a-z0-9-]+\.zip$/i.test(u)) {
      assert.equal(options.method, 'HEAD', '下载前应先用 HEAD 确认网站包存在，而不是先把整个包读进内存');
      return { ok: true, status: 200 };
    }
    return realFetch(url, options);
  };
  window.fetch = globalThis.fetch;
  localStorage.setItem('oj-ai.admin', JSON.stringify({ exp: Date.now() + 60_000 }));
  try {
    $('#roleBtn').click(); // 打开管理员面板，刷新网站包按钮的可用状态
    const btn = document.querySelector('[data-site-package="site-win-x64.zip"]');
    assert.equal(btn.disabled, false);
    btn.click();
    const status = await waitFor(() => /已开始下载网站包 site-win-x64\.zip/.test($('#statusText').textContent) && $('#statusText').textContent);
    assert.ok(status, '点击后应立即提示已开始下载，并指引用户看浏览器下载进度');
    assert.match(btn.textContent, /下载/, '按钮文字应即时变化，给出点击反馈');
  } finally {
    localStorage.removeItem('oj-ai.admin');
    if (!$('#rolePanel').hidden) $('#roleBtn').click();
    globalThis.fetch = realFetch;
    window.fetch = realFetch;
  }
});

test('管理员面板直接改参数：填数字保存即写入本机覆盖，清空一项恢复默认', async () => {
  const modelId = config.webllm.defaultModel;
  localStorage.setItem('oj-ai.admin', JSON.stringify({ exp: Date.now() + 60_000 }));
  try {
    $('#roleBtn').click(); // 打开管理员面板，渲染参数表单
    const form = document.querySelector('#paramOverrides .param-form');
    assert.ok(form, '管理员面板应有直接改参数的表单，不必再去对话里发参数指令');
    assert.equal(form.querySelectorAll('[data-param-key]').length, 6, '应有温度、重复惩罚、最大输出与三个窗口共 6 项');
    form.querySelector('[data-param-key="temperature"]').value = '0.9';
    form.querySelector('[data-param-key="maxTokens"]').value = '1234';
    form.dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    const saved = await waitFor(() => {
      const adapt = JSON.parse(localStorage.getItem('oj-ai.adapt.v1') || '{"models":{}}');
      return adapt.models[modelId]?.params?.temperature === 0.9 && adapt.models[modelId].params;
    });
    assert.ok(saved, '保存后参数应写入本机覆盖');
    assert.equal(saved.maxTokens, 1234);
    const status = await waitFor(() => /已保存「/.test($('#statusText').textContent) && $('#statusText').textContent);
    assert.ok(status, '保存后状态栏应给出已保存反馈');

    // 表单已按新覆盖重渲染：把温度清空并保存，应去掉该项覆盖（恢复默认）
    const form2 = document.querySelector('#paramOverrides .param-form');
    assert.equal(form2.querySelector('[data-param-key="temperature"]').value, '0.9', '已有覆盖应预填在输入框中');
    form2.querySelector('[data-param-key="temperature"]').value = '';
    form2.dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }));
    const afterReset = await waitFor(() => {
      const adapt = JSON.parse(localStorage.getItem('oj-ai.adapt.v1') || '{"models":{}}');
      const params = adapt.models[modelId]?.params || {};
      return !('temperature' in params) && params;
    });
    assert.ok(afterReset, '清空并保存后应恢复默认');
    assert.equal(afterReset.maxTokens, 1234, '未清空的项应保持不变');
    const log = JSON.parse(localStorage.getItem('oj-ai.adapt.v1')).models[modelId].log || [];
    assert.ok(log.some((e) => e.source === 'panel'), '变更记录应标明来源为管理员面板');
  } finally {
    localStorage.removeItem('oj-ai.admin');
    localStorage.removeItem('oj-ai.adapt.v1');
    if (!$('#rolePanel').hidden) $('#roleBtn').click();
  }
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
  // 三个公开 API 并行；摘要披露各来源合计命中的资料数。
  // 结果按相关性排序，来源顺序不再固定；摘要只保证三个来源都被列出。
  assert.match(
    card.querySelector('summary').textContent,
    /深入检索：3 条资料（(?:维基百科|Stack Overflow|Hacker News)(?: \+ (?:维基百科|Stack Overflow|Hacker News)){2}）/,
  );
  const links = [...card.querySelectorAll('.search-sources a')];
  assert.equal(links.length, 3);
  const byTitle = new Map(links.map((a) => [a.textContent, a.getAttribute('href')]));
  assert.equal(byTitle.get('二分查找'), 'https://zh.wikipedia.org/wiki/%E4%BA%8C%E5%88%86%E6%9F%A5%E6%89%BE');
  assert.ok([...byTitle.values()].includes('https://stackoverflow.com/questions/1'));
  assert.ok([...byTitle.values()].includes('https://example.com/bs'));

  // “各来源情况”：多来源时默认折叠列出，让人看到各公开 API 均被检查
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
  // 结果按相关性排序，第一条可能是任一来源；只断言来源集合与资料内容。
  assert.equal(saved.search.results.length, 3);
  assert.deepEqual([...saved.search.providers].sort(), ['hackernews', 'stackexchange', 'wikipedia']);
  assert.ok(saved.search.results.some((r) => r.title === '二分查找'));
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
    assert.ok(failures.some((t) => t.startsWith('Hacker News：')));
    assert.match(card.textContent, /浏览器直连的公开 API/);
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

    // 只有维基百科有结果，但卡片明确列出其余公开 API 的状态。
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

test('用量统计面板：点击打开，含网页用量卡片与静态版 API 提示', async () => {
  const btn = $('#usageBtn');
  assert.ok(btn, '顶部应有用量统计入口');
  btn.click();
  const dlg = $('#usageDialog');
  assert.equal(dlg.open, true, '点击应打开用量统计面板');
  assert.match(dlg.textContent, /网页模型用量/);
  assert.match(dlg.textContent, /API 调用用量/);
  assert.equal($('#usageWebCards').children.length, 4, '应有总调用/提示/生成/总 Token 四张汇总卡');
  assert.equal($('#usageWebDays').children.length, 7, '应有最近 7 天趋势条');
  assert.match($('#usageApiHint').textContent, /静态版/, '静态版应说明没有 API 统计');
  $('#usageClose').click();
  assert.equal(dlg.open, false, '关闭按钮应收起面板');
});

test('一站式训练包按钮：点击后走 Release 下载并立即反馈', async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, options = {}) => {
    const u = String(url);
    if (/oj-training\.zip$/.test(u)) {
      assert.equal(options.method, 'HEAD');
      assert.match(u, /releases\/latest\/download\/oj-training\.zip$/, '训练包应走 GitHub Release 直达链接');
      return { ok: true, status: 200 };
    }
    return realFetch(url, options);
  };
  window.fetch = globalThis.fetch;
  localStorage.setItem('oj-ai.admin', JSON.stringify({ exp: Date.now() + 60_000 }));
  try {
    $('#roleBtn').click();
    const btn = $('#trainingPackageBtn');
    assert.equal(btn.disabled, false, '管理员应可点击训练包按钮');
    btn.click();
    const status = await waitFor(() => /已开始下载一站式训练包 oj-training\.zip/.test($('#statusText').textContent) && $('#statusText').textContent);
    assert.ok(status, '点击训练包后应立即提示已开始下载');
  } finally {
    localStorage.removeItem('oj-ai.admin');
    if (!$('#rolePanel').hidden) $('#roleBtn').click();
    globalThis.fetch = realFetch;
    window.fetch = realFetch;
  }
});
