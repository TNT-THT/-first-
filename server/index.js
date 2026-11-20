import express from 'express';
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';
import { loadConfig, validateConfig } from './config.js';
import { createToken, verifyToken, checkPassword, checkApiKey, extractBearer } from './auth.js';
import {
  validateMessages,
  validateChatCompletionBody,
  proxyChatStream,
  collectChatCompletion,
  HttpError,
} from './chat.js';
import { buildApiInfo, buildDocsHtml, API_VERSION } from './docs.js';
import { ApiKeyStore } from './api-key-store.js';
import { ApiSettingsStore } from './settings-store.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB_DIST = path.resolve(__dirname, '..', 'web', 'dist');

/** 简单的内存登录限流：每个 IP 15 分钟内最多 10 次失败尝试。 */
function createLoginLimiter({ windowMs = 15 * 60 * 1000, max = 10 } = {}) {
  const hits = new Map();
  return {
    blocked(ip) {
      const now = Date.now();
      const rec = hits.get(ip);
      if (!rec || now - rec.start > windowMs) return false;
      return rec.count >= max;
    },
    fail(ip) {
      const now = Date.now();
      const rec = hits.get(ip);
      if (!rec || now - rec.start > windowMs) hits.set(ip, { start: now, count: 1 });
      else rec.count += 1;
    },
    reset(ip) {
      hits.delete(ip);
    },
  };
}

/** OpenAI 风格错误体（供 /v1/* 使用）。 */
function openAiError(res, status, message, type = 'invalid_request_error') {
  return res.status(status).json({ error: { message, type, code: null } });
}

function originMatches(origin, pattern) {
  if (!origin || origin === 'null') return false;
  if (pattern === '*') return true;
  let parsed;
  try {
    parsed = new URL(origin);
  } catch {
    return false;
  }
  const wildcard = String(pattern).match(/^(https?):\/\/\*\.(.+)$/i);
  if (wildcard) {
    try {
      const base = new URL(`${wildcard[1]}://${wildcard[2]}`);
      return (
        parsed.protocol === base.protocol &&
        parsed.port === base.port &&
        parsed.hostname !== base.hostname &&
        parsed.hostname.endsWith(`.${base.hostname}`)
      );
    } catch {
      return false;
    }
  }
  try {
    return parsed.origin === new URL(pattern).origin;
  } catch {
    return false;
  }
}

export function createApp(cfg = loadConfig()) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', cfg.trustProxy);
  app.use(express.json({ limit: '256kb' }));
  app.use(['/api/admin/api-keys', '/api/admin/settings'], (_req, res, next) => {
    res.setHeader('Cache-Control', 'no-store');
    next();
  });

  // ---- CORS（供模式一外部 API 的浏览器跨域调用；同源网页不受影响）----
  const corsOrigins = cfg.corsOrigins?.length ? cfg.corsOrigins : ['*'];
  app.use((req, res, next) => {
    if (!req.path.startsWith('/api/') && !req.path.startsWith('/v1/') && req.path !== '/healthz') {
      return next();
    }
    const origin = req.get('origin') || '';
    const allowAll = corsOrigins.includes('*');
    const allowed = allowAll || (origin && corsOrigins.some((pattern) => originMatches(origin, pattern)));
    if (allowed) {
      res.setHeader('Access-Control-Allow-Origin', allowAll ? '*' : origin);
      res.setHeader('Vary', 'Origin');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, x-api-key');
      res.setHeader('Access-Control-Max-Age', '86400');
    }
    if (req.method === 'OPTIONS') return res.status(204).end();
    return next();
  });

  // ---- 安全响应头 ----
  // 注意：这里刻意【不】设置 X-Frame-Options，否则浏览器会禁止被 iframe 嵌入。
  // 嵌入范围改由 CSP 的 frame-ancestors 控制。
  app.use((req, res, next) => {
    res.setHeader(
      'Content-Security-Policy',
      [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        `connect-src 'self'${cfg.cspConnectSrc ? ` ${cfg.cspConnectSrc}` : ''}`,
        "object-src 'none'",
        "base-uri 'self'",
        `frame-ancestors ${cfg.frameAncestors}`,
      ].join('; ')
    );
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    next();
  });

  const limiter = createLoginLimiter();
  const apiKeyAdminLimiter = createLoginLimiter();
  const apiKeyStore = new ApiKeyStore(cfg.apiKeysFile);
  const apiSettingsStore = new ApiSettingsStore(cfg.apiSettingsFile);
  apiSettingsStore.apply(cfg);
  // 使用独立签名密钥空间；普通网页登录令牌不能冒充 API Key 管理员会话。
  const apiKeyAdminSecret = `${cfg.sessionSecret}:api-key-admin-v1`;

  const requireAuth = (req, res, next) => {
    const token = extractBearer(req);
    if (!verifyToken(cfg.sessionSecret, token)) {
      return res.status(401).json({ error: '未登录或登录已过期' });
    }
    next();
  };

  // /v1 只接受静态/管理员签发的 API Key 或短期管理员会话；普通网站登录不能作为外部 API 凭证。
  const requireApiAuth = async (req, res, next) => {
    const token = extractBearer(req);
    try {
      if (
        checkApiKey(cfg.apiKeys, token) ||
        verifyToken(apiKeyAdminSecret, token) ||
        (await apiKeyStore.verify(token))
      ) {
        return next();
      }
      return openAiError(res, 401, '未提供有效的 API Key');
    } catch (err) {
      return next(err);
    }
  };

  const requireApiKeysAdmin = (req, res, next) => {
    if (!cfg.apiAdminPassword) {
      return res.status(503).json({ error: '网页 API Key 管理未启用，请在服务器 .env 配置 API_ADMIN_PASSWORD 并重启。' });
    }
    const token = extractBearer(req);
    if (!verifyToken(apiKeyAdminSecret, token)) {
      return res.status(401).json({ error: 'API Key 管理会话无效或已过期，请重新验证管理员口令。' });
    }
    return next();
  };

  app.get('/healthz', (_req, res) =>
    res.json({ ok: true, version: API_VERSION, model: cfg.llmModel || null, modelConfigured: !!(cfg.llmBaseUrl && cfg.llmModel) })
  );

  // 公开配置：前端据此决定 postMessage 的白名单等，不含任何密钥
  app.get('/api/config', (_req, res) => {
    res.json({
      appName: cfg.appName,
      model: cfg.llmModel || null,
      allowedParentOrigins: cfg.allowedParentOrigins,
      publicBaseUrl: cfg.publicBaseUrl || null,
      apiBaseUrl: cfg.publicBaseUrl ? `${cfg.publicBaseUrl}/v1` : '/v1',
      apiKeyAuth: !!cfg.apiKeys?.length || !!cfg.apiAdminPassword,
      apiKeyManagerEnabled: !!cfg.apiAdminPassword,
      apiDefaultTemperature: cfg.llmTemperature,
      apiMaxTokens: cfg.apiMaxTokens,
      version: API_VERSION,
    });
  });

  // 网站端 API Key 管理：独立管理员口令登录，列表只返回元数据，创建时明文仅返回一次。
  app.post('/api/admin/api-keys/login', (req, res) => {
    if (!cfg.apiAdminPassword) {
      return res.status(503).json({ error: '网页 API Key 管理未启用，请在服务器 .env 配置 API_ADMIN_PASSWORD 并重启。' });
    }
    const ip = req.ip;
    if (apiKeyAdminLimiter.blocked(ip)) {
      return res.status(429).json({ error: '验证次数过多，请 15 分钟后再试。' });
    }
    if (!checkPassword(cfg.apiAdminPassword, req.body?.password)) {
      apiKeyAdminLimiter.fail(ip);
      return res.status(401).json({ error: '管理员口令错误。' });
    }
    apiKeyAdminLimiter.reset(ip);
    const { token, expiresAt } = createToken(apiKeyAdminSecret, 8 * 60 * 60 * 1000);
    return res.json({ token, expiresAt });
  });

  app.get('/api/admin/api-keys', requireApiKeysAdmin, async (_req, res, next) => {
    try {
      return res.json({ keys: await apiKeyStore.list(), maxKeys: 100, environmentKeys: (cfg.apiKeys || []).length });
    } catch (err) {
      return next(err);
    }
  });

  app.post('/api/admin/api-keys', requireApiKeysAdmin, async (req, res, next) => {
    try {
      if (req.body?.label !== undefined && typeof req.body.label !== 'string') {
        return res.status(400).json({ error: 'label 必须是字符串。' });
      }
      const created = await apiKeyStore.create(req.body?.label || '');
      return res.status(201).json(created);
    } catch (err) {
      if (/最多只能创建/.test(err.message)) return res.status(400).json({ error: err.message });
      console.error('[api-keys] create failed:', err.message);
      return next(err);
    }
  });

  app.delete('/api/admin/api-keys/:id', requireApiKeysAdmin, async (req, res, next) => {
    try {
      const revoked = await apiKeyStore.revoke(req.params.id);
      if (!revoked) return res.status(404).json({ error: '未找到此 API Key。' });
      return res.json({ revoked: true });
    } catch (err) {
      console.error('[api-keys] revoke failed:', err.message);
      return next(err);
    }
  });

  // 服务端参数控制台：仅 API 管理员会话可读写；运行时立即生效，并持久化到权限受限的 JSON 文件。
  app.get('/api/admin/settings', requireApiKeysAdmin, (_req, res) => {
    return res.json(apiSettingsStore.snapshot(cfg));
  });

  app.put('/api/admin/settings', requireApiKeysAdmin, async (req, res, next) => {
    try {
      return res.json(await apiSettingsStore.update(req.body, cfg));
    } catch (err) {
      if (err instanceof HttpError) return res.status(err.status).json({ error: err.message });
      console.error('[api-settings] update failed:', err.message);
      return next(err);
    }
  });

  app.delete('/api/admin/settings', requireApiKeysAdmin, async (_req, res, next) => {
    try {
      return res.json(await apiSettingsStore.reset(cfg));
    } catch (err) {
      console.error('[api-settings] reset failed:', err.message);
      return next(err);
    }
  });

  // 机器可读接入信息 + 人类可读文档页（模式一 API / 模式二链接）
  app.get('/api/info', (_req, res) => res.json(buildApiInfo(cfg)));
  app.get('/api/docs', (_req, res) => {
    res.type('html').setHeader('Cache-Control', 'no-cache');
    res.send(buildDocsHtml(cfg));
  });

  app.post('/api/login', (req, res) => {
    const ip = req.ip;
    if (limiter.blocked(ip)) {
      return res.status(429).json({ error: '尝试次数过多，请 15 分钟后再试' });
    }
    const { password } = req.body || {};
    if (!checkPassword(cfg.appPassword, password)) {
      limiter.fail(ip);
      return res.status(401).json({ error: '密码错误' });
    }
    limiter.reset(ip);
    const { token, expiresAt } = createToken(cfg.sessionSecret, cfg.tokenTtlHours * 3600 * 1000);
    res.json({ token, expiresAt });
  });

  app.get('/api/session', requireAuth, (_req, res) => res.json({ ok: true }));

  app.post('/api/chat', requireAuth, async (req, res) => {
    try {
      const messages = validateMessages(req.body?.messages);
      const ac = new AbortController();
      res.on('close', () => {
        if (!res.writableFinished) ac.abort();
      });
      await proxyChatStream({ cfg, messages, res, signal: ac.signal });
    } catch (err) {
      const status = err instanceof HttpError ? err.status : 500;
      if (!(err instanceof HttpError)) console.error('[chat] unexpected:', err);
      if (res.headersSent) {
        res.end();
      } else {
        res.status(status).json({ error: err.message || '服务器内部错误' });
      }
    }
  });

  // ---- 模式一：OpenAI 兼容 API（/v1/*）----
  const modelObject = () => ({
    id: cfg.llmModel,
    object: 'model',
    created: 0,
    owned_by: 'oj-ai-assistant',
  });

  app.get('/v1/models', requireApiAuth, (_req, res) => {
    res.json({ object: 'list', data: cfg.llmModel ? [modelObject()] : [] });
  });

  app.get('/v1/models/:id', requireApiAuth, (req, res) => {
    if (!cfg.llmModel || req.params.id !== cfg.llmModel) {
      return openAiError(res, 404, `模型不存在：${req.params.id}`, 'model_not_found');
    }
    return res.json(modelObject());
  });

  app.post('/v1/chat/completions', requireApiAuth, async (req, res) => {
    const ac = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) ac.abort();
    });
    try {
      const { messages, stream, temperature, maxTokens, params } = validateChatCompletionBody(req.body, cfg);
      // 即使调用方未提交 max_tokens，也应用服务端上限，避免无限制生成消耗服务器额度。
      const outputLimit = maxTokens ?? cfg.apiMaxTokens ?? 8192;
      if (stream) {
        await proxyChatStream({ cfg, messages, res, signal: ac.signal, temperature, maxTokens: outputLimit, params, normalize: true });
        return;
      }
      const result = await collectChatCompletion({ cfg, messages, temperature, maxTokens: outputLimit, params, signal: ac.signal });
      if (!result) return;
      const promptTokens = messages.reduce((n, m) => n + Math.ceil(m.content.length / 4), 0);
      const completionTokens = Math.ceil(result.content.length / 4);
      res.json({
        id: result.id,
        object: 'chat.completion',
        created: result.created,
        model: result.model,
        choices: [
          {
            index: 0,
            message: { role: 'assistant', content: result.content },
            finish_reason: result.finishReason,
          },
        ],
        usage: {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens,
        },
      });
    } catch (err) {
      const status = err instanceof HttpError ? err.status : 500;
      if (!(err instanceof HttpError)) console.error('[v1/chat] unexpected:', err);
      if (res.headersSent) {
        res.end();
      } else {
        openAiError(res, status, err.message || '服务器内部错误', status === 404 ? 'model_not_found' : 'invalid_request_error');
      }
    }
  });

  // ---- 前端静态资源 ----
  if (existsSync(WEB_DIST)) {
    app.use(
      express.static(WEB_DIST, {
        index: false,
        setHeaders(res, filePath) {
          if (filePath.endsWith('index.html')) res.setHeader('Cache-Control', 'no-cache');
          else if (filePath.includes(`${path.sep}assets${path.sep}`)) {
            res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');
          }
        },
      })
    );
    app.get('*', (req, res, next) => {
      if (req.path.startsWith('/api/') || req.path.startsWith('/v1/')) return next();
      res.setHeader('Cache-Control', 'no-cache');
      res.sendFile(path.join(WEB_DIST, 'index.html'));
    });
  } else {
    app.get('/', (_req, res) =>
      res.type('text/plain').send('前端尚未构建，请先执行 npm run build:remote')
    );
  }

  app.use((err, req, res, _next) => {
    const apiError = (status, message) =>
      req.path.startsWith('/v1/')
        ? openAiError(res, status, message)
        : res.status(status).json({ error: message });
    if (err.type === 'entity.parse.failed') return apiError(400, '请求体不是合法 JSON');
    if (err.type === 'entity.too.large') return apiError(413, '请求体过大');
    console.error(err);
    return apiError(500, '服务器内部错误');
  });

  return app;
}

// 直接运行时启动服务（测试时通过 import 使用 createApp，不会自动监听）
const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  // 原生 `npm start` 也读取项目根目录的 .env；Docker Compose 的 env_file 环境变量优先不被覆盖。
  dotenv.config({ path: path.resolve(process.cwd(), '.env') });
  const cfg = loadConfig();
  const errors = validateConfig(cfg);
  if (errors.length) {
    console.error('启动失败：\n - ' + errors.join('\n - '));
    process.exit(1);
  }
  if (cfg.sessionSecretIsRandom) {
    console.warn('[warn] 未设置 SESSION_SECRET，本次使用随机密钥，重启后需重新登录。');
  }
  if (!cfg.llmBaseUrl || !cfg.llmModel) {
    console.warn('[warn] 未配置 LLM_BASE_URL / LLM_MODEL，聊天功能暂不可用。');
  }
  if (!cfg.apiKeys?.length && !cfg.apiAdminPassword) {
    console.warn('[warn] 未配置 API_KEYS 或 API_ADMIN_PASSWORD，外部 API（/v1/*）仅接受网页登录令牌；如需供外部系统调用，请配置其中一项。');
  }
  createApp(cfg).listen(cfg.port, cfg.host, () => {
    console.log(`${cfg.appName} 已启动: http://${cfg.host}:${cfg.port}`);
    console.log(`  网站链接: http://${cfg.host}:${cfg.port}/`);
    console.log(`  API 入口: http://${cfg.host}:${cfg.port}/v1/chat/completions`);
    console.log(`  接入文档: http://${cfg.host}:${cfg.port}/api/docs`);
  });
}
