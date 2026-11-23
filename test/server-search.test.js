// 服务端网页搜索（假搜索引擎 + 假网页，不访问真实网络）与前端 websearch 来源接入。
import test from 'node:test';
import assert from 'node:assert/strict';
import { parseBingRss, parseBrave, parseDuckDuckGoHtml, parseTavily, searchWeb } from '../server/search.js';
import { planProviders, PROVIDER_LABELS } from '../web/src/search/config.js';
import { parseServerWeb } from '../web/src/search/providers.js';

test('解析器：Tavily / Brave / Bing RSS / DDG HTML', () => {
  assert.equal(parseTavily({ results: [{ title: 'T', url: 'https://a.com', content: '正文' }] })[0].provider, 'tavily');
  assert.equal(parseBrave({ web: { results: [{ title: 'T', url: 'https://a.com', description: 'd' }] } })[0].provider, 'brave');
  const rss = '<rss><channel><item><title>标题</title><link>https://a.com/x</link><description>摘要</description></item></channel></rss>';
  assert.equal(parseBingRss(rss)[0].url, 'https://a.com/x');
  const html = '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fb.com%2Fy">结果</a>';
  assert.equal(parseDuckDuckGoHtml(html)[0].url, 'https://b.com/y');
});

test('searchWeb：优先 Tavily 并抓正文、去重', async () => {
  const fetchImpl = async (url, opts = {}) => {
    const u = String(url);
    if (u.includes('api.tavily.com')) {
      assert.equal(opts.method, 'POST');
      return new Response(JSON.stringify({
        results: [
          { title: 'Qwen 文档', url: 'https://Qwen.ai/docs', content: '短', score: 0.9 },
          { title: 'Qwen 文档', url: 'https://Qwen.ai/docs', content: '短', score: 0.9 },
        ],
      }), { headers: { 'content-type': 'application/json' } });
    }
    if (u === 'https://Qwen.ai/docs') {
      return new Response('<html><head><style>x</style></head><body><h1>Qwen</h1><p>这是正文内容，介绍模型用法与部署。</p><script>bad()</script></body></html>', {
        headers: { 'content-type': 'text/html' },
      });
    }
    throw new Error(`不应访问：${u}`);
  };
  const out = await searchWeb({ query: 'Qwen 部署', fetchImpl, env: { TAVILY_API_KEY: 'tvly-x' } });
  assert.equal(out.provider, 'tavily');
  assert.equal(out.results.length, 1);
  assert.match(out.results[0].content, /这是正文内容/);
  assert.doesNotMatch(out.results[0].content, /bad\(\)/);
});

test('searchWeb：无密钥时走 Bing RSS 兜底', async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes('bing.com/search')) {
      return new Response('<rss><channel><item><title>新闻</title><link>https://news.example/a</link><description>摘要文字</description></item></channel></rss>', { headers: { 'content-type': 'application/rss+xml' } });
    }
    if (u.includes('duckduckgo')) return new Response('<html></html>', { headers: { 'content-type': 'text/html' } });
    if (u === 'https://news.example/a') {
      return new Response('<body><p>新闻正文内容。</p></body>', { headers: { 'content-type': 'text/html' } });
    }
    throw new Error(`不应访问：${u}`);
  };
  const out = await searchWeb({ query: '新闻', fetchImpl, env: {} });
  assert.equal(out.provider, 'keyless');
  assert.equal(out.results[0].url, 'https://news.example/a');
});

test('前端：软件版启用 websearch 来源且结果归一化带正文', () => {
  const plan = planProviders(null, { serverSearch: true });
  assert.equal(plan[0].id, 'websearch');
  assert.equal(PROVIDER_LABELS.websearch, '网页搜索');
  const rows = parseServerWeb({ results: [{ title: 'T', url: 'https://a.com', snippet: 's', content: '正文内容' }] });
  assert.equal(rows[0].provider, 'websearch');
  assert.equal(rows[0].extract, '正文内容');
});
