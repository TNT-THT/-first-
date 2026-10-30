/**
 * 对话导出：把当前会话整理为 Markdown 文本（所有用户可用，仅在本机生成与下载，不上传）。
 *
 * - 只导出可见的消息：用户消息、AI 回答（标注回答的 AI 名称）、固定提示（以引用形式输出）。
 * - 不导出推理过程（thinking）与出错的消息，避免把临时内容当作对话记录。
 * - 联网资料检索的来源（标题与链接）附在对应回答之后；搜索摘要不导出。
 */

function pad(n) {
  return String(n).padStart(2, '0');
}

/** 形如 2026-10-20 14:05，用于文档内的导出时间。 */
export function formatExportTime(now = new Date()) {
  const d = now instanceof Date ? now : new Date(now);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** 形如 20261020-1405，用于文件名。 */
function stampForFile(now = new Date()) {
  const d = now instanceof Date ? now : new Date(now);
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

/** 生成安全的文件名：去掉路径与系统保留字符，限制长度。 */
export function conversationFilename(title, now = new Date()) {
  const base =
    String(title || '')
      .replace(/[\\/:*?"<>|\r\n\t]+/g, '_')
      .trim()
      .slice(0, 40) || '对话';
  return `${base}-${stampForFile(now)}.md`;
}

/** 把多行文本逐行加上引用符号 "> "。 */
function quote(text) {
  return text
    .split('\n')
    .map((line) => (line ? `> ${line}` : '>'))
    .join('\n');
}

/**
 * 会话 → Markdown 文本。session 为 {title, messages}；消息格式与 store 中一致。
 */
export function buildConversationMarkdown(session, { now = new Date() } = {}) {
  const title = String(session?.title || '').trim() || '未命名对话';
  const lines = [`# ${title}`, '', `导出时间：${formatExportTime(now)}`, ''];
  for (const m of session?.messages || []) {
    if (!m || m.error) continue;
    const content = String(m.content || '').trim();
    if (!content) continue;
    if (m.role === 'user') {
      lines.push('## 我', '', content, '');
    } else if (m.fixed) {
      lines.push(quote(`系统提示：${content}`), '');
    } else {
      const head = m.modelLabel ? `## 回答（${m.modelLabel}）` : '## 回答';
      lines.push(head, '', content, '');
      // 联网资料检索的来源：只在当轮使用，导出时作为参考链接附在回答之后
      const sources = Array.isArray(m.search?.results) ? m.search.results.filter((r) => r?.url) : [];
      if (sources.length) {
        lines.push('参考来源：', '');
        sources.forEach((r, i) => lines.push(`${i + 1}. [${r.title || r.url}](${r.url})`));
        lines.push('');
      }
    }
  }
  return lines.join('\n');
}
