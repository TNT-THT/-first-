import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderMarkdown } from '../web/src/markdown.js';
import { isAllowedOrigin } from '../web/src/origin.js';
import { buildContext } from '../web/src/store.js';
import { validateMessages } from '../server/chat.js';
import { createToken, verifyToken, checkPassword } from '../server/auth.js';

test('Markdown：代码块带语言标签与复制按钮，并做语法高亮', () => {
  const html = renderMarkdown('```cpp\nint main(){ return 0; }\n```');
  assert.match(html, /class="code-block"/);
  assert.match(html, /<span class="code-lang">cpp<\/span>/);
  assert.match(html, /data-action="copy"/);
  assert.match(html, /hljs-keyword/);
});

test('Markdown：python / java 等常见语言别名正常', () => {
  assert.match(renderMarkdown('```py\nprint(1)\n```'), /language-python/);
  assert.match(renderMarkdown('```java\nclass A {}\n```'), /language-java/);
});

test('Markdown：行内与行间 LaTeX 公式（$、$$、\\(\\)、\\[\\]）', () => {
  assert.match(renderMarkdown('复杂度 $O(n\\log n)$'), /class="katex"/);
  assert.match(renderMarkdown('\\(\\sum_{i=1}^n a_i\\)'), /class="katex"/);
  assert.match(renderMarkdown('$$\nx^2\n$$'), /katex-display/);
  assert.match(renderMarkdown('\\[x^2\\]'), /katex-display/);
});

test('Markdown：货币符号与代码中的 $ 不会被当作公式', () => {
  const money = renderMarkdown('价格 $5 和 $10');
  assert.ok(!money.includes('katex'));
  const code = renderMarkdown('代码里的 `$x$` 不是公式');
  assert.ok(!code.includes('katex'));
  assert.match(code, /<code class="inline-code">\$x\$<\/code>/);
});

test('Markdown：禁止原始 HTML，防止注入', () => {
  const html = renderMarkdown('<img src=x onerror=alert(1)>\n\n<script>alert(2)</script>');
  assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('<script'));
  assert.match(html, /&lt;script&gt;/);
});

test('Markdown：链接在新标签页打开', () => {
  const html = renderMarkdown('[题解](https://example.com)');
  assert.match(html, /target="_blank"/);
  assert.match(html, /rel="noopener noreferrer"/);
});

test('origin 白名单：精确匹配、通配子域名、拒绝伪造域名', () => {
  const p = ['https://hydro.ac', 'https://*.hydro.ac'];
  assert.equal(isAllowedOrigin('https://hydro.ac', p), true);
  assert.equal(isAllowedOrigin('https://oj.hydro.ac', p), true);
  assert.equal(isAllowedOrigin('https://a.b.hydro.ac', p), true);
  assert.equal(isAllowedOrigin('https://evilhydro.ac', p), false);
  assert.equal(isAllowedOrigin('https://hydro.ac.evil.com', p), false);
  assert.equal(isAllowedOrigin('http://hydro.ac', p), false);
  assert.equal(isAllowedOrigin('null', p), false);
  assert.equal(isAllowedOrigin('https://anything.com', ['*']), true);
  assert.equal(isAllowedOrigin('https://hydro.ac', []), false);
});

test('多轮上下文：保留最近消息、以用户消息开头、跳过错误与空消息', () => {
  const msgs = [
    { role: 'user', content: '第一题 代码' },
    { role: 'assistant', content: '回答一' },
    { role: 'user', content: '为什么 TLE？' },
    { role: 'assistant', content: '', error: false },
    { role: 'user', content: '再看看' },
  ];
  const ctx = buildContext(msgs, { maxMessages: 10 });
  assert.equal(ctx[0].role, 'user');
  assert.equal(ctx.at(-1).content, '再看看');
  assert.ok(ctx.some((m) => m.content === '第一题 代码'));
  assert.ok(!ctx.some((m) => m.content === ''));

  const limited = buildContext(msgs, { maxMessages: 2 });
  assert.equal(limited.length, 2);
  assert.equal(limited[0].role, 'user');
});

test('消息校验：合法数组通过，只保留 role/content 字段', () => {
  const out = validateMessages([{ role: 'user', content: 'hi', extra: 1 }]);
  assert.deepEqual(out, [{ role: 'user', content: 'hi' }]);
});

test('令牌：签发、验证、过期、篡改', () => {
  const { token } = createToken('secret', 60_000, 1000);
  assert.equal(verifyToken('secret', token, 2000), true);
  assert.equal(verifyToken('secret', token, 100_000), false, '过期');
  assert.equal(verifyToken('other', token, 2000), false, '错误密钥');
  assert.equal(verifyToken('secret', token + 'x', 2000), false, '篡改');
  assert.equal(verifyToken('secret', undefined, 2000), false);
});

test('密码比较', () => {
  assert.equal(checkPassword('abcdefgh', 'abcdefgh'), true);
  assert.equal(checkPassword('abcdefgh', 'abcdefg'), false);
  assert.equal(checkPassword('', ''), false);
  assert.equal(checkPassword('abcdefgh', undefined), false);
});
