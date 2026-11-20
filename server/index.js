import express from 'express';
import { randomUUID } from 'node:crypto';
import { createServer } from 'node:http';
import { BrowserModelBridge, BRIDGE_SUPPORTED_GENERATION_PARAMS } from './browser-model-bridge.js';
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
// 网站包里静态版与服务版网页共存（web/dist 与 web/dist-api），可用 OJ_WEB_DIST 指定服务版目录。
const WEB_DIST = process.env.OJ_WEB_DIST
  ? path.resolve(process.cwd(), process.env.OJ_WEB_DIST)
  : path.resolve(__dirname, '..', 'web', 'dist');

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

export function createApp(cfg = loadConfig(), { browserBridge = null } = {}) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', cfg.trustProxy);
  app.use(express.json({ limit: '256kb' }));
  app.use(['/api/admin/api-keys', '/api/admin/settings'], (_req, res, next) => {
    res.setHeader('Cache-Control', 'no-store');
    next();
  });

  // ---- CORS（仅供模式一外部 API /v1 的浏览器跨域调用；同源网页不受影响）----
  // 白名单每次请求实时读取 cfg：管理员在「API 管理」中保存后立即生效，无需重启。
  // 空数组表示不允许任何跨域浏览器调用（同源请求不受 CORS 限制，不受影响）。
  // 注意：跨域许可只发给 /v1，绝不发给 /api/*——否则白名单里的第三方网站可以
  // 绕过用户直接调用自助申请 Key、登录等本机接口（本站网页是同源的，不需要 CORS）。
  app.use((req, res, next) => {
    if (!req.path.startsWith('/v1/')) {
      return next();
    }
    const corsOrigins = Array.isArray(cfg.corsOrigins) ? cfg.corsOrigins : ['*'];
    const origin = req.get('origin') || '';
    const allowAll = corsOrigins.includes('*');
    const allowed = allowAll || (origin && corsOrigins.some((pattern) => originMatches(origin, pattern)));
    if (allowed) {
      res.setHeader('Access-Control-Allow-Origin', allowAll ? '*' : origin);
      res.setHeader('Vary', 'Origin, Access-Control-Request-Private-Network');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type, x-api-key');
      res.setHeader('Access-Control-Max-Age', '86400');
      // 公网 HTTPS 网页调用本机回环/局域网 API 时，Chromium 的 Private Network Access
      // 预检会带 Access-Control-Request-Private-Network；只有白名单内的来源才返回许可。
      // 新版 Chrome 改用本地网络访问权限弹窗，此响应头对旧版预检保持兼容、无副作用。
      if (req.method === 'OPTIONS' && req.get('access-control-request-private-network') === 'true') {
        res.setHeader('Access-Control-Allow-Private-Network', 'true');
      }
    }
    if (req.method === 'OPTIONS') return res.status(204).end();
    return next();
  });

  // ---- 安全响应头 ----
  // 注意：这里刻意【不】设置 X-Frame-Options，否则浏览器会禁止被 iframe 嵌入。
  // 嵌入范围改由 CSP 的 frame-ancestors 控制。
  // 本机桥接模式下，网页要在浏览器里编译 WebAssembly 并从 Hugging Face 下载模型权重，
  // 因此仅在该模式放行 'wasm-unsafe-eval' 与模型下载域名，否则网页模型根本加载不了。
  const scriptSrc = browserBridge ? "script-src 'self' 'wasm-unsafe-eval'" : "script-src 'self'";
  const connectSrc = `'self'${cfg.cspConnectSrc ? ` ${cfg.cspConnectSrc}` : ''}${
    browserBridge ? ' https://huggingface.co https://*.huggingface.co https://*.hf.co' : ''
  }`;
  app.use((req, res, next) => {
    res.setHeader(
      'Content-Security-Policy',
      [
        "default-src 'self'",
        scriptSrc,
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        `connect-src ${connectSrc}`,
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
  const publicKeyApplyLimiter = createLoginLimiter({ windowMs: 10 * 60 * 1000, max: 3 });
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

  app.get('/healthz', (_req, res) => {
    const bridgeStatus = browserBridge?.status();
    return res.json({
      ok: true,
      version: API_VERSION,
      model: bridgeStatus?.models?.[0]?.id || cfg.llmModel || null,
      modelConfigured: bridgeStatus ? bridgeStatus.connected : !!(cfg.llmBaseUrl && cfg.llmModel),
      localBrowserBridge: !!browserBridge,
      browserModelConnected: bridgeStatus?.connected ?? null,
    });
  });

  // 公开配置：前端据此决定 postMessage 的白名单等，不含任何密钥
  app.get('/api/local-model-bridge/status', (_req, res) => {
    if (!browserBridge) return res.status(404).json({ error: '本机浏览器模型 API 未启用。' });
    return res.json(browserBridge.status());
  });

  // 本机 API 申请：仅本机桥接部署开启；每个 IP 每 10 分钟最多申请 3 把。
  // 只允许本站网页（同源）申请：带 Origin 的浏览器请求若来源与本站不一致，一律拒绝，
  // 防止被加入跨域白名单的第三方网站绕过用户自行申请 Key。
  app.post('/api/api-keys/apply', async (req, res, next) => {
    if (!browserBridge) return res.status(404).json({ error: '本机 API Key 申请未启用。' });
    const origin = req.get('origin');
    if (origin) {
      let originHost = '';
      try {
        originHost = new URL(origin).host;
      } catch {
        /* 无法解析的 Origin 按跨源处理 */
      }
      if (!originHost || originHost !== req.get('host')) {
        return res.status(403).json({ error: '只允许在本机网页中申请 API Key。' });
      }
    }
    const ip = req.ip;
    if (publicKeyApplyLimiter.blocked(ip)) {
      return res.status(429).json({ error: '申请次数过多，请 10 分钟后再试。' });
    }
    publicKeyApplyLimiter.fail(ip);
    try {
      const label = typeof req.body?.label === 'string' ? req.body.label : '网页自助申请';
      return res.status(201).json(await apiKeyStore.create(label));
    } catch (err) {
      if (/最多只能创建/.test(err.message)) return res.status(429).json({ error: err.message });
      return next(err);
    }
  });

  app.get('/api/config', (_req, res) => {
    res.json({
      appName: cfg.appName,
      model: cfg.llmModel || null,
      allowedParentOrigins: cfg.allowedParentOrigins,
      publicBaseUrl: cfg.publicBaseUrl || null,
      apiBaseUrl: cfg.publicBaseUrl ? `${cfg.publicBaseUrl}/v1` : '/v1',
      apiKeyAuth: !!cfg.apiKeys?.length || !!cfg.apiAdminPassword,
      apiKeyManagerEnabled: !!cfg.apiAdminPassword,
      localBrowserBridge: !!browserBridge,
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
  app.get('/api/info', (_req, res) => res.json(buildApiInfo(cfg, browserBridge?.status())));
  app.get('/api/docs', (_req, res) => {
    res.type('html').setHeader('Cache-Control', 'no-cache');
    res.send(buildDocsHtml(cfg, browserBridge?.status()));
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
      if (browserBridge) {
        // 本机桥接模式下没有服务端上游：网页自身的聊天也交给本机网页的 WebGPU 模型。
        // 网页可在请求中指定 model（模型选择栏的选择，仅影响这一条请求）；
        // 不指定时沿用桥接上报的第一个模型。API 请求选什么模型不会改掉网页自己的选择。
        const bridgeStatus = browserBridge.status();
        if (!bridgeStatus.connected) {
          return res.status(503).json({ error: '本机模型尚未连接：请保持本网页打开，待模型连接后再发送。' });
        }
        const requestedModel = typeof req.body?.model === 'string' && req.body.model ? req.body.model : '';
        if (requestedModel && !bridgeStatus.models?.some((m) => m.id === requestedModel)) {
          return res.status(404).json({ error: `模型不存在或本机网页未提供该模型：${requestedModel}` });
        }
        const model = requestedModel || bridgeStatus.models?.[0]?.id;
        if (!model) {
          return res.status(503).json({ error: '本机模型尚未连接：请保持本网页打开，待模型连接后再发送。' });
        }
        res.status(200);
        res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.flushHeaders?.();
        const id = `chatcmpl-${randomUUID().replaceAll('-', '')}`;
        const created = Math.floor(Date.now() / 1000);
        const emit = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);
        emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] });
        await browserBridge.run(
          // 不传 temperature/maxTokens：让浏览器按该模型的本机参数（管理员面板/预设）决定，
          // 服务端只给输出硬上限，避免服务端默认值把本机面板的温度与最大输出“压住”。
          { model, messages, hardMaxTokens: cfg.apiMaxTokens ?? 8192 },
          {
            signal: ac.signal,
            onDelta(delta) {
              emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { content: delta }, finish_reason: null }] });
            },
          },
        );
        if (!ac.signal.aborted && !res.destroyed) {
          emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
          res.write('data: [DONE]\n\n');
          res.end();
        }
        return;
      }
      await proxyChatStream({ cfg, messages, res, signal: ac.signal });
    } catch (err) {
      const status = err instanceof HttpError
        ? err.status
        : (Number.isInteger(err?.status) && err.status >= 400 && err.status <= 599 ? err.status : 500);
      if (!(err instanceof HttpError)) {
        if (Number.isInteger(err?.status)) console.warn(`[chat] 请求失败（${status}）：${err.message || '未知错误'}`);
        else console.error('[chat] unexpected:', err);
      }
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

  const availableApiModels = () => browserBridge
    ? browserBridge.status().models.map((model) => ({ id: model.id, object: 'model', created: 0, owned_by: 'oj-ai-assistant', name: model.label }))
    : cfg.llmModel ? [modelObject()] : [];

  // 兼容把完整对话接口误填为 Base URL 的客户端：它们会在其后拼接 /models 取列表，
  // 请求 /v1/chat/completions/models 时同样返回模型列表，避免用户被莫名其妙的 404 卡住。
  app.get(['/v1/models', '/v1/chat/completions/models'], requireApiAuth, (_req, res) => {
    res.json({ object: 'list', data: availableApiModels() });
  });

  app.get('/v1/models/:id', requireApiAuth, (req, res) => {
    const model = availableApiModels().find((entry) => entry.id === req.params.id);
    if (!model) return openAiError(res, 404, `模型不存在或本机网页尚未连接：${req.params.id}`, 'model_not_found');
    return res.json(model);
  });

  // 同上兼容：Base URL 误填为完整对话接口时，客户端会请求 /v1/chat/completions/chat/completions，一并处理。
  app.post(['/v1/chat/completions', '/v1/chat/completions/chat/completions'], requireApiAuth, async (req, res) => {
    const ac = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) ac.abort();
    });
    try {
      const requestCfg = browserBridge ? { ...cfg, llmModel: '' } : cfg;
      const { messages, stream, temperature, maxTokens, params, model: requestedModel } = validateChatCompletionBody(req.body, requestCfg);
      // 即使调用方未提交 max_tokens，也应用服务端上限，避免无限制生成消耗服务器额度。
      const outputLimit = maxTokens ?? cfg.apiMaxTokens ?? 8192;
      if (browserBridge) {
        if (!browserBridge.status().connected) {
          return openAiError(res, 503, '本机模型网页尚未连接。请在这台电脑打开服务网页，并保持网页打开。', 'service_unavailable');
        }
        const available = availableApiModels();
        const model = requestedModel || available[0]?.id;
        if (!model || !available.some((entry) => entry.id === model)) {
          return openAiError(res, 404, `模型不存在或本机网页未提供该模型：${model || '(未指定)'}`, 'model_not_found');
        }
        // 本机引擎只支持一部分生成参数：不支持的提前明确拒绝（400），不静默丢弃假装生效
        const bridgeParams = { ...(cfg.defaultGenerationParams || {}), ...params };
        const unsupportedParams = Object.keys(bridgeParams).filter(
          (key) => !BRIDGE_SUPPORTED_GENERATION_PARAMS.includes(key),
        );
        if (unsupportedParams.length) {
          return openAiError(
            res,
            400,
            `本机模型不支持生成参数：${unsupportedParams.join('、')}。支持的参数：${BRIDGE_SUPPORTED_GENERATION_PARAMS.join('、')}。`,
          );
        }
        const id = `chatcmpl-${randomUUID().replaceAll('-', '')}`;
        const created = Math.floor(Date.now() / 1000);
        let content = '';
        if (stream) {
          res.status(200);
          res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
          res.setHeader('Cache-Control', 'no-cache, no-transform');
          res.setHeader('Connection', 'keep-alive');
          res.flushHeaders?.();
          const emit = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`);
          emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }] });
          await browserBridge.run({
            model,
            messages,
            // 调用方没指定时不传 temperature/maxTokens，让本机模型参数决定；outputLimit 只作硬上限
            temperature,
            maxTokens,
            hardMaxTokens: outputLimit,
            params: bridgeParams,
            systemPrompt: cfg.systemPrompt,
          }, {
            signal: ac.signal,
            onDelta(delta) {
              content += delta;
              emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { content: delta }, finish_reason: null }] });
            },
          });
          if (!ac.signal.aborted && !res.destroyed) {
            emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] });
            res.write('data: [DONE]\n\n');
            res.end();
          }
          return;
        }
        const result = await browserBridge.run({
          model,
          messages,
          temperature,
          maxTokens,
          hardMaxTokens: outputLimit,
          params: bridgeParams,
          systemPrompt: cfg.systemPrompt,
        }, {
          signal: ac.signal,
          onDelta(delta) { content += delta; },
        });
        if (ac.signal.aborted || res.destroyed) return;
        const promptTokens = messages.reduce((n, m) => n + Math.ceil(m.content.length / 4), 0);
        const completionTokens = Math.ceil(content.length / 4);
        return res.json({
          id, object: 'chat.completion', created, model,
          choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: result.finishReason || 'stop' }],
          usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
        });
      }
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
      const status = err instanceof HttpError
        ? err.status
        : (Number.isInteger(err?.status) && err.status >= 400 && err.status <= 599 ? err.status : 500);
      if (!(err instanceof HttpError)) {
        // 带 status 的是可预期的运行失败（本机网页未连接、模型生成失败、超时等）：
        // 只记一行真实原因，不打印误导性的包装堆栈；其余才是未预期异常。
        if (Number.isInteger(err?.status)) console.warn(`[v1/chat] 请求失败（${status}）：${err.message || '未知错误'}`);
        else console.error('[v1/chat] unexpected:', err);
      }
      if (res.headersSent) {
        // 流式响应已经发头：把失败原因作为 SSE error 事件发出去再结束，
        // 否则调用方只会看到流莫名中断（或一句通用文案），拿不到真实原因。
        try {
          res.write(`data: ${JSON.stringify({ error: { message: err.message || '服务器内部错误', type: 'invalid_request_error', code: null } })}\n\n`);
        } catch { /* 连接已断开 */ }
        res.end();
      } else {
        openAiError(res, status, err.message || '服务器内部错误', status === 404 ? 'model_not_found' : 'invalid_request_error');
      }
    }
  });

  // /v1 下未匹配到的路径：返回 OpenAI 风格的 JSON 404 并给出填写提示，
  // 避免 Express 默认的 HTML 错误页让客户端（如自动获取模型列表的软件）报出看不懂的错。
  app.use('/v1', (req, res) =>
    openAiError(
      res,
      404,
      `接口不存在：${req.method} ${req.path}。请确认软件里的 Base URL 以 /v1 结尾（例如 http://127.0.0.1:3000/v1）：模型列表是 GET /v1/models，对话是 POST /v1/chat/completions；不要把完整的对话接口地址当作 Base URL。`,
      'invalid_request_error',
    ),
  );

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
  const bridgeEnabled = process.env.LOCAL_BROWSER_BRIDGE === '1';
  if (bridgeEnabled) {
    // 桥接模式按“同一台电脑”设计：自助申请 Key 与模型桥接都没有公网级鉴权。
    // 若误把服务绑到局域网/公网地址，任何能连上的人都能申请 Key 并使用本机显卡，
    // 因此默认拒绝启动；确需如此部署时须显式设置 OJ_ALLOW_PUBLIC_BRIDGE=1 自行承担风险。
    const loopbackHosts = new Set(['127.0.0.1', '::1', '[::1]', 'localhost']);
    if (!loopbackHosts.has(String(cfg.host).toLowerCase()) && process.env.OJ_ALLOW_PUBLIC_BRIDGE !== '1') {
      console.error(
        `[start] 本机网页桥接只能绑定本机回环地址（当前 HOST=${cfg.host}）。` +
          '如确需绑定其他地址，请设置 OJ_ALLOW_PUBLIC_BRIDGE=1 并自行做好网络隔离与访问控制。',
      );
      process.exit(1);
    }
  }
  if (bridgeEnabled) {
    console.log('[local] 本机网页模型桥接已启用：打开本网页并保持打开，即可聊天并使用 /v1 API（无需配置 LLM_BASE_URL / LLM_MODEL）。');
  } else if (!cfg.llmBaseUrl || !cfg.llmModel) {
    console.warn('[warn] 未配置 LLM_BASE_URL / LLM_MODEL，聊天功能暂不可用。');
  }
  if (!cfg.apiKeys?.length && !cfg.apiAdminPassword) {
    console.warn('[warn] 未配置 API_KEYS 或 API_ADMIN_PASSWORD，外部 API（/v1/*）仅接受网页登录令牌；如需供外部系统调用，请配置其中一项。');
  }
  const browserBridge = process.env.LOCAL_BROWSER_BRIDGE === '1' ? new BrowserModelBridge() : null;
  const server = createServer(createApp(cfg, { browserBridge }));
  browserBridge?.attach(server);
  server.listen(cfg.port, cfg.host, () => {
    console.log(`${cfg.appName} 已启动: http://${cfg.host}:${cfg.port}`);
    console.log(`  网站链接: http://${cfg.host}:${cfg.port}/`);
    console.log(`  API 入口: http://${cfg.host}:${cfg.port}/v1/chat/completions`);
    console.log(`  接入文档: http://${cfg.host}:${cfg.port}/api/docs`);
  });
}
