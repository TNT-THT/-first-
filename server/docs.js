/**
 * 对外接入信息与文档页（模式一 API + 模式二链接）。
 * 只包含公开信息，绝不包含任何密钥；示例中的密钥均为占位符。
 */

export const API_VERSION = '1.0.0';

export function escapeHtml(s) {
  return String(s ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

export function buildApiInfo(cfg) {
  const base = cfg.publicBaseUrl || '';
  const abs = (p) => (base ? `${base}${p}` : p);
  return {
    appName: cfg.appName,
    version: API_VERSION,
    model: cfg.llmModel || null,
    baseUrl: base || null,
    endpoints: {
      health: abs('/healthz'),
      config: abs('/api/config'),
      info: abs('/api/info'),
      docs: abs('/api/docs'),
      website: abs('/'),
      models: abs('/v1/models'),
      chatCompletions: abs('/v1/chat/completions'),
      legacyChat: abs('/api/chat'),
    },
    auth: {
      website: '打开网站链接，用 APP_PASSWORD 登录（会话令牌保存在浏览器）',
      api: cfg.apiKeys?.length
        ? 'Authorization: Bearer <API_KEY> 或 x-api-key（服务端 API_KEYS 配置；管理员也可在网页面板创建动态 Key）'
        : cfg.apiAdminPassword
          ? '由 API 管理员在网站“API 接入”面板创建并分发 Authorization: Bearer <API_KEY>'
          : '未配置 API_KEYS 或 API_ADMIN_PASSWORD；外部调用请先配置静态 API_KEYS 或启用 API 管理员面板创建 Key。普通网站登录令牌不能访问 /v1/*。',
      apiKeyConfigured: !!cfg.apiKeys?.length,
      apiKeyAuthAvailable: !!cfg.apiKeys?.length || !!cfg.apiAdminPassword,
      keyManagerEnabled: !!cfg.apiAdminPassword,
      keyManager: abs('/api/docs#api-key-manager'),
    },
    link: {
      chat: abs('/?q=<问题>'),
      chatAndSend: abs('/?q=<问题>&send=1'),
      embed: abs('/?embed=1'),
      params: {
        q: '预填问题（别名 prompt / text / question），最多 20000 字',
        send: '1=打开后自动发送（仅可信链接使用）',
        model: '指定模型（仅浏览器内 WebLLM 模式有效，服务端模式忽略）',
        embed: '1=嵌入模式（隐藏侧边栏，适合 iframe）',
      },
    },
  };
}

export function buildDocsHtml(cfg) {
  const base = cfg.publicBaseUrl || 'https://ai.example.com';
  const model = cfg.llmModel || 'your-model';
  const curl = `curl ${base}/v1/chat/completions \\
  -H "Authorization: Bearer <API_KEY>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "两数之和怎么做？"}],
    "stream": false
  }'`;
  const curlStream = `curl -N ${base}/v1/chat/completions \\
  -H "Authorization: Bearer <API_KEY>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "messages": [{"role": "user", "content": "两数之和怎么做？"}],
    "stream": true
  }'`;
  const python = `# pip install openai
from openai import OpenAI
client = OpenAI(base_url="${base}/v1", api_key="<API_KEY>")
r = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "两数之和怎么做？"}],
)
print(r.choices[0].message.content)`;
  const js = `const res = await fetch("${base}/v1/chat/completions", {
  method: "POST",
  headers: {
    "Authorization": "Bearer <API_KEY>",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "${model}",
    messages: [{ role: "user", content: "两数之和怎么做？" }],
  }),
});
const data = await res.json();
console.log(data.choices[0].message.content);`;
  const link = `${base}/?q=${encodeURIComponent('两数之和怎么做？')}`;
  const linkSend = `${link}&send=1`;
  const embed = `<iframe src="${escapeHtml(base)}/?embed=1" width="420" height="640" style="border:0" title="${escapeHtml(cfg.appName)}"></iframe>`;
  const modelsCurl = `curl ${base}/v1/models -H "Authorization: Bearer <API_KEY>"`;

  const section = (title, body) => `<section><h2>${escapeHtml(title)}</h2>${body}</section>`;
  const pre = (code) => `<pre>${escapeHtml(code)}</pre>`;

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(cfg.appName)} · 远程接入文档</title>
<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.6}pre{background:#f6f8fa;padding:1rem;overflow:auto;border-radius:8px}code{background:#f6f8fa;padding:.1em .3em;border-radius:4px}pre code{background:none;padding:0}a{color:#0969da}</style>
</head><body>
<h1>${escapeHtml(cfg.appName)} · 远程接入文档</h1>
<p>本服务由自建服务器独立提供，不依赖 GitHub Pages、Hugging Face 等第三方平台托管。
同一台服务器同时提供两种接入方式：<strong>模式一 API</strong>（程序调用）与 <strong>模式二链接</strong>（浏览器打开 / iframe 嵌入）。
当前模型：<code>${escapeHtml(model)}</code> · 版本 <code>${API_VERSION}</code> ·
<a href="/api/info">机器可读信息（JSON）</a> · <a href="/">打开网站</a></p>

${section('模式一：自建服务器 API（OpenAI 兼容）', `
<p>鉴权：<code>Authorization: Bearer &lt;API_KEY&gt;</code> 或 <code>x-api-key: &lt;API_KEY&gt;</code>。密钥可在服务器环境变量 <code>API_KEYS</code> 中配置，也可由管理员使用独立的 <code>API_ADMIN_PASSWORD</code> 在网站「API 接入」面板中为用户或应用创建并分发。
普通网站登录令牌不能访问 <code>/v1/*</code>。支持流式（<code>stream: true</code>，SSE）与非流式两种返回；跨域调用受 <code>CORS_ORIGINS</code> 控制（默认 <code>*</code>）。</p>
${pre(modelsCurl)}
${pre(curl)}
<p>流式：</p>${pre(curlStream)}
<p>Python（OpenAI SDK）：</p>${pre(python)}
<p>JavaScript（fetch）：</p>${pre(js)}
<p><strong>安全提示：</strong>不要把 API Key 写进公开网页、浏览器扩展或移动端应用；这些场景可被用户查看密钥，建议由你自己的后端转发。
跨域浏览器请求受 <code>CORS_ORIGINS</code> 控制，但 CORS 不是密钥保护机制。</p>
<p>错误格式为 OpenAI 风格：<code>{"error": {"message": "...", "type": "invalid_request_error"}}</code>。
请求字段：<code>model</code>（须与服务端模型一致）、<code>messages</code>（含 system，纯文本）、<code>stream</code>、<code>temperature</code>（0–2）、<code>max_tokens</code> / <code>max_completion_tokens</code>（由 <code>API_MAX_TOKENS</code> 限制；未指定时应用服务端上限），以及兼容服务商支持的额外生成参数。工具调用、多模态内容暂不支持。</p>
`)}

${section('网页 API 调试台与管理员参数控制台', `
<p>只有输入独立的 <code>API_ADMIN_PASSWORD</code> 并完成服务端验证后，才能在聊天页顶部的「管理员面板」中使用网页调试台、服务端参数控制台和 API Key 管理。调试台使用短期管理员令牌；普通网站 <code>APP_PASSWORD</code> 会话不能调用 <code>/v1/*</code>。</p>
<p>参数控制台可以即时修改上游 OpenAI 兼容地址、模型、服务端 API Key、系统提示、默认 temperature、token 上限与供应商支持的 JSON 生成参数。设置写入权限受限的 <code>API_SETTINGS_FILE</code>（默认 <code>data/api-settings.json</code>）；Key 不会从管理 API 回传。普通网站登录口令、管理员口令与 CORS 等安全参数仍由服务器环境变量管理。</p>
<p>管理员可以按用户或应用创建、命名和撤销独立 API Key，再安全地分发给非管理员使用者。本站没有个人账号、账号级配额或用量账单；普通用户拿到自己的 Key 后可在其他网站后端或软件中调用本接口，不必使用网站登录令牌。</p>
`)}

${section('网站内管理 API Key', `
<div id="api-key-manager"></div>
<p>配置 <code>API_ADMIN_PASSWORD</code> 并重启服务器后，登录网站，在聊天页顶部点击「管理员面板」并完成管理员验证，即可为每位用户或应用创建单独命名的 Key，或随时撤销。
创建时完整 Key 只显示一次，请立即安全分发并保存在调用方受控后端；网站只保存 SHA-256 摘要，之后不能再次查看明文。误关页面或遗失时请撤销旧 Key 并创建新 Key。
此管理口令不要与普通网站访问口令相同，也不要把 API Key 放在公开网页、浏览器 JS 或手机 App 中。</p>
`)}

${section('模式二：网页链接接入（直接打开）', `
<p>任何人打开链接即可使用（服务端模式需先输入站点口令登录一次）：</p>
${pre(link)}
<p>打开后自动发送（仅可信链接使用，链接里带了问题并会自动提问）：</p>
${pre(linkSend)}
<p>参数：<code>q</code> 预填问题（别名 <code>prompt</code>/<code>text</code>/<code>question</code>）、<code>send=1</code> 自动发送、<code>model</code> 指定模型（仅浏览器内模式）、<code>embed=1</code> 嵌入模式。</p>
`)}

${section('模式二：网页链接接入（iframe 嵌入到其他网站）', `
<p>把下面代码粘贴到你的页面即可。嵌入范围由服务端 <code>FRAME_ANCESTORS</code> 控制（默认 <code>*</code>），
父页面还可通过 <code>postMessage</code> 发送 <code>{type: "oj-ai:fill", text: "题目", send: false}</code> 填入输入框
（来源须在 <code>ALLOWED_PARENT_ORIGINS</code> 白名单内）。</p>
${pre(embed)}
`)}

${section('自建部署（不依赖其他平台）', `
<p>在本仓库根目录：</p>
${pre(`cp .env.example .env   # 填写 APP_PASSWORD、SESSION_SECRET、API_ADMIN_PASSWORD、PUBLIC_BASE_URL
# API 管理员可在网页参数控制台修改服务端模型配置，并按用户/应用创建或撤销 Key
# 也可用 API_KEYS 配置静态密钥；设置文件 API_SETTINGS_FILE 默认写入 data/api-settings.json
# 模型：自建 Ollama（docker compose --profile local-llm）或自有 vLLM / OpenAI 兼容接口
npm run build:remote
npm start                  # 或 docker compose up -d --build
# 对外请用 Nginx/Caddy 提供 HTTPS（见 deploy/nginx.conf），并关闭 proxy_buffering 以支持流式输出`)}
<p>健康检查：<code>GET /healthz</code>；公开配置：<code>GET /api/config</code>（不含密钥）。</p>
`)}
</body></html>`;
}
