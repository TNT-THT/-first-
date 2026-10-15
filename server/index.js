import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { existsSync } from 'node:fs';
import { loadConfig, validateConfig } from './config.js';
import { createToken, verifyToken, checkPassword } from './auth.js';
import { validateMessages, proxyChatStream, HttpError } from './chat.js';

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

export function createApp(cfg = loadConfig()) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', cfg.trustProxy);
  app.use(express.json({ limit: '256kb' }));

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
        "connect-src 'self'",
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

  const requireAuth = (req, res, next) => {
    const header = req.get('authorization') || '';
    const token = header.startsWith('Bearer ') ? header.slice(7) : '';
    if (!verifyToken(cfg.sessionSecret, token)) {
      return res.status(401).json({ error: '未登录或登录已过期' });
    }
    next();
  };

  app.get('/healthz', (_req, res) => res.json({ ok: true }));

  // 公开配置：前端据此决定 postMessage 的白名单等，不含任何密钥
  app.get('/api/config', (_req, res) => {
    res.json({
      appName: cfg.appName,
      model: cfg.llmModel || null,
      allowedParentOrigins: cfg.allowedParentOrigins,
    });
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
      if (req.path.startsWith('/api/')) return next();
      res.setHeader('Cache-Control', 'no-cache');
      res.sendFile(path.join(WEB_DIST, 'index.html'));
    });
  } else {
    app.get('/', (_req, res) =>
      res.type('text/plain').send('前端尚未构建，请先执行 npm run build')
    );
  }

  app.use((err, _req, res, _next) => {
    if (err.type === 'entity.parse.failed') return res.status(400).json({ error: '请求体不是合法 JSON' });
    if (err.type === 'entity.too.large') return res.status(413).json({ error: '请求体过大' });
    console.error(err);
    res.status(500).json({ error: '服务器内部错误' });
  });

  return app;
}

// 直接运行时启动服务（测试时通过 import 使用 createApp，不会自动监听）
const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
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
  createApp(cfg).listen(cfg.port, cfg.host, () => {
    console.log(`${cfg.appName} 已启动: http://${cfg.host}:${cfg.port}`);
  });
}
