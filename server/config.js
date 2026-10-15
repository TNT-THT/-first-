import { randomBytes } from 'node:crypto';

/**
 * 所有配置均来自环境变量，详见 .env.example。
 * API Key 只存在于服务端，绝不会下发到浏览器。
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

    // 允许通过 postMessage 向输入框填充内容的父页面来源。支持 https://*.example.com 通配写法。
    // 设为 '*' 则接受任意来源。
    allowedParentOrigins: (env.ALLOWED_PARENT_ORIGINS || 'https://hydro.ac,https://*.hydro.ac')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),

    // 部署在 Nginx / 代理工具等反向代理之后时设为 1，以便正确获取客户端 IP（用于登录限流）。
    trustProxy: env.TRUST_PROXY ? Number(env.TRUST_PROXY) : false,

    appName: env.APP_NAME || 'OJ 刷题助手',
  };
  return cfg;
}

export function validateConfig(cfg) {
  const errors = [];
  if (!cfg.appPassword) errors.push('缺少 APP_PASSWORD（登录密码）');
  if (cfg.appPassword && cfg.appPassword.length < 8) errors.push('APP_PASSWORD 长度至少 8 位');
  return errors;
}
