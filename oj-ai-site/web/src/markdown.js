import MarkdownIt from 'markdown-it';
import hljs from 'highlight.js/lib/common';
import katex from 'katex';

const LANG_ALIAS = {
  'c++': 'cpp',
  cc: 'cpp',
  hpp: 'cpp',
  py: 'python',
  py3: 'python',
  js: 'javascript',
  ts: 'typescript',
  golang: 'go',
  sh: 'bash',
};

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * 行内 / 行间数学公式规则，支持：$...$、$$...$$、\(...\)、\[...\]
 * 必须放在 markdown-it 的 escape 规则之前，否则 "\(" 会被当作转义字符吃掉。
 * 代码块与行内代码由 markdown-it 先行消费，因此代码里的 $ 不会被误判为公式。
 */
function mathInlinePlugin(state, silent) {
  const src = state.src;
  const start = state.pos;
  const ch = src[start];
  let open;
  let close;
  let display;

  if (ch === '$') {
    if (src[start + 1] === '$') {
      open = '$$';
      close = '$$';
      display = true;
    } else {
      open = '$';
      close = '$';
      display = false;
    }
  } else if (ch === '\\' && (src[start + 1] === '(' || src[start + 1] === '[')) {
    display = src[start + 1] === '[';
    open = display ? '\\[' : '\\(';
    close = display ? '\\]' : '\\)';
  } else {
    return false;
  }

  const contentStart = start + open.length;
  let end = -1;
  let i = contentStart;
  while (i < src.length) {
    const idx = src.indexOf(close, i);
    if (idx === -1) break;
    if (close.startsWith('$') && src[idx - 1] === '\\') {
      i = idx + 1;
      continue;
    }
    end = idx;
    break;
  }
  if (end === -1) return false;

  const content = src.slice(contentStart, end);
  if (!content.trim()) return false;
  // 公式内容不能跨越行内代码的反引号，否则 "$5 ... `$x$`" 会被错误配对
  if (content.includes('`')) return false;
  // 简单区分货币符号：$5 和 $10 这类不当作公式
  if (open === '$') {
    if (/^\s/.test(content) || /\s$/.test(content)) return false;
    if (/\d/.test(src[end + 1] || '')) return false;
  }

  if (!silent) {
    const token = state.push('math_inline', '', 0);
    token.content = content.trim();
    token.markup = open;
    token.meta = { display };
  }
  state.pos = end + close.length;
  return true;
}

function renderMath(tokens, idx) {
  const t = tokens[idx];
  try {
    return katex.renderToString(t.content, {
      displayMode: t.meta.display,
      throwOnError: false,
      strict: 'ignore',
      output: 'htmlAndMathml',
    });
  } catch {
    return `<code>${escapeHtml(t.content)}</code>`;
  }
}

export function createMarkdownRenderer() {
  const md = new MarkdownIt({
    html: false, // 禁止原始 HTML，防止模型输出注入脚本
    linkify: true,
    breaks: true, // 聊天场景中单个换行也保留为换行
    typographer: false,
  });

  md.inline.ruler.before('escape', 'math_inline', mathInlinePlugin);
  md.renderer.rules.math_inline = renderMath;

  // 代码块：语言标签 + 复制按钮（点击事件由外层委托处理）
  md.renderer.rules.fence = (tokens, idx) => {
    const token = tokens[idx];
    const rawLang = (token.info || '').trim().split(/\s+/)[0].toLowerCase();
    const alias = LANG_ALIAS[rawLang] || rawLang;
    const known = alias && hljs.getLanguage(alias) ? alias : null;
    const body = known
      ? hljs.highlight(token.content, { language: known, ignoreIllegals: true }).value
      : escapeHtml(token.content);
    const label = known || rawLang || 'text';
    return (
      `<div class="code-block">` +
      `<div class="code-head"><span class="code-lang">${escapeHtml(label)}</span>` +
      `<button type="button" class="copy-btn" data-action="copy">复制</button></div>` +
      `<pre><code class="hljs language-${escapeHtml(label)}">${body}</code></pre></div>`
    );
  };

  // 行内代码同样做转义（markdown-it 默认已处理，这里仅补充 class）
  md.renderer.rules.code_inline = (tokens, idx) =>
    `<code class="inline-code">${escapeHtml(tokens[idx].content)}</code>`;

  // 链接在新标签页打开，并阻止 opener 访问
  const defaultLinkOpen =
    md.renderer.rules.link_open ||
    ((tokens, idx, options, _env, self) => self.renderToken(tokens, idx, options));
  md.renderer.rules.link_open = (tokens, idx, options, env, self) => {
    tokens[idx].attrSet('target', '_blank');
    tokens[idx].attrSet('rel', 'noopener noreferrer');
    return defaultLinkOpen(tokens, idx, options, env, self);
  };

  return md;
}

const renderer = createMarkdownRenderer();

export function renderMarkdown(text) {
  return renderer.render(text || '');
}
