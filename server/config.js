import { randomBytes } from 'node:crypto';
import path from 'node:path';

/**
 * 所有配置均来自环境变量，详见 .env.example。
 * 上游模型供应商的 LLM_API_KEY 只存在于服务端；网站创建的调用方 API Key 仅在创建成功时显示一次。
 */
export function loadConfig(env = process.env) {
  const cfg = {
    port: Number(env.PORT || 3000),
    host: env.HOST || '0.0.0.0',

    // 登录密码（必填）。留空则拒绝启动，避免被陌生人白嫖你的 API 额度。
    appPassword: env.APP_PASSWORD || '',
    // 用于签名令牌的密钥；未设置时每次启动随机生成（重启后需要重新登录）。
    sessionSecret: env.SESSION_SECRET || randomBytes(32).toString('hex'),
    sessionSecretIsRandom: !env.SESSION_SECRET,
    tokenTtlHours: Number(env.TOKEN_TTL_HOURS || 168),

    // OpenAI 兼容接口（DeepSeek / 通义千问 DashScope / vLLM / Ollama 等均可）
    llmBaseUrl: (env.LLM_BASE_URL || '').replace(/\/+$/, ''),
    llmApiKey: env.LLM_API_KEY || '',
    llmModel: env.LLM_MODEL || '',
    llmTemperature: Number(env.LLM_TEMPERATURE ?? 0.3),

    systemPrompt:
      env.SYSTEM_PROMPT ||
      '你是一个有帮助的 AI 助手，可以回答各类问题。请使用用户提问的语言回答。' +
        '数学公式使用 LaTeX（行内用 $...$，独立公式用 $$...$$），代码使用带语言标注的 Markdown 代码块。',

    // 允许嵌入本站的父页面来源，写入 CSP frame-ancestors。'*' 表示任何站点都可以嵌入。
    frameAncestors: env.FRAME_ANCESTORS || '*',

    // 联网资料检索在浏览器内直接请求资料源，因此这些来源必须写进 CSP 的 connect-src（空格分隔）。
    // 默认放行公开资料 API：维基百科、Stack Exchange、Hacker News（以及可选的 DuckDuckGo）；它们不是完整网页搜索。
    // 使用受信任的 SearXNG、自定义接口或正文读取服务时，把每个服务 origin 追加进来（例如 https://searx.example.com）。
    cspConnectSrc:
      env.CSP_CONNECT_SRC ||
      'https://*.wikipedia.org https://wikipedia.org https://api.stackexchange.com https://hn.algolia.com https://api.duckduckgo.com',

    // 允许通过 postMessage 向输入框填充内容的父页面来源。支持 https://*.example.com 通配写法。
    // 设为 '*' 则接受任意来源。
    allowedParentOrigins: (env.ALLOWED_PARENT_ORIGINS || 'https://hydro.ac,https://*.hydro.ac')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),

    // 部署在 Nginx / 代理工具等反向代理之后时设为 1，以便正确获取客户端 IP（用于登录限流）。
    trustProxy: env.TRUST_PROXY ? Number(env.TRUST_PROXY) : false,

    appName: env.APP_NAME || 'OJ 刷题助手',

    // ---- 对外 API（模式一：自建服务器 API，OpenAI 兼容 /v1/*）----
    // API_KEYS：逗号分隔的外部调用密钥，可配置多个；也支持单值 API_KEY。
    // 外部程序以 API Key 调用 /v1；也可由管理员通过网站面板创建独立 Key。
    apiKeys: parseList(env.API_KEYS, env.API_KEY),
    // API Key 管理页面的独立管理员口令。必须与 APP_PASSWORD 区分，且至少 16 位；留空会禁用网页生成/撤销功能。
    apiAdminPassword: env.API_ADMIN_PASSWORD || '',
    // 网页创建的 API Key 只保存摘要；该文件/目录需要持久化备份，并限制仅服务器用户可读。
    apiKeysFile: path.resolve(env.API_KEYS_FILE || path.join(process.cwd(), 'data', 'api-keys.json')),
    // 管理员控制台修改的服务端模型参数持久化文件（API Key 以 0600 权限保存，不会由管理 API 回传）。
    apiSettingsFile: path.resolve(env.API_SETTINGS_FILE || path.join(process.cwd(), 'data', 'api-settings.json')),
    // 供应商特定生成参数默认值，可由 API 管理员控制台修改。
    defaultGenerationParams: {},

    // CORS：允许浏览器跨域调用 /v1/* 与 /api/* 的来源。'*' 表示任意来源（默认）。
    // 收紧时写逗号分隔的 origin 列表，例如 https://oj.example.com,https://*.example.com
    corsOrigins: parseList(env.CORS_ORIGINS) || ['*'],

    // 对外公开地址（不带末尾斜杠），用于生成链接/嵌入/API 示例；留空时文档用相对路径。
    publicBaseUrl: (env.PUBLIC_BASE_URL || '').replace(/\/+$/, ''),

    // 单次回复最大 token 上限（外部 API 的 max_tokens 超过会被拒绝，避免刷爆额度）。
    apiMaxTokens: Number(env.API_MAX_TOKENS || 8192),
  };
  return cfg;
}

function parseList(...values) {
  const out = [];
  for (const v of values) {
    if (!v) continue;
    for (const part of String(v).split(',')) {
      const s = part.trim();
      if (s && !out.includes(s)) out.push(s);
    }
  }
  return out.length ? out : null;
}

export function validateConfig(cfg) {
  const errors = [];
  if (!cfg.appPassword) errors.push('缺少 APP_PASSWORD（登录密码）');
  if (cfg.appPassword && cfg.appPassword.length < 8) errors.push('APP_PASSWORD 长度至少 8 位');
  if (cfg.apiAdminPassword && cfg.apiAdminPassword.length < 16) {
    errors.push('API_ADMIN_PASSWORD 长度至少 16 位（或者留空以禁用网页管理员控制台）');
  }
  if (cfg.apiAdminPassword && cfg.appPassword && cfg.apiAdminPassword === cfg.appPassword) {
    errors.push('API_ADMIN_PASSWORD 必须与 APP_PASSWORD 不同');
  }
  if (!Number.isInteger(cfg.apiMaxTokens) || cfg.apiMaxTokens < 1 || cfg.apiMaxTokens > 1_000_000) {
    errors.push('API_MAX_TOKENS 必须是 1 到 1000000 之间的整数');
  }
  if (!Number.isFinite(cfg.llmTemperature) || cfg.llmTemperature < 0 || cfg.llmTemperature > 2) {
    errors.push('LLM_TEMPERATURE 必须是 0 到 2 之间的数字');
  }
  return errors;
}
