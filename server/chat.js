import { SEARCH_EVIDENCE_SYSTEM_GUARD } from '../web/src/search/context.js';

export const LIMITS = {
  maxMessages: 60,
  maxMessageChars: 20000,
  maxTotalChars: 80000,
};

export class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/**
 * 把 OpenAI 的富文本 content 归一化为纯文本。
 * 不少客户端（尤其多模态客户端）发的是分段数组：[{ type: 'text', text: '…' }, …]，
 * 以前这里只认字符串，分段消息会直接 400，导致“请求传不进来、AI 没有响应”。
 * 文本段按顺序拼接；只有图片等非文本段时明确报 400（本机模型仅支持文本）。
 * assistant 的 content 为 null（工具调用占位等）按空文本处理，历史回放更宽容。
 */
export function normalizeMessageContent(content, { role, index }) {
  const at = `第 ${index + 1} 条消息`;
  if (typeof content === 'string') return content;
  if (content === null || content === undefined) {
    if (role === 'assistant') return '';
    throw new HttpError(400, `${at}内容为空`);
  }
  if (Array.isArray(content)) {
    const texts = [];
    let hasMedia = false;
    for (const part of content) {
      if (typeof part === 'string') { texts.push(part); continue; }
      if (!part || typeof part !== 'object') continue;
      if ((part.type === 'text' || part.type === 'input_text') && typeof part.text === 'string') {
        texts.push(part.text);
      } else if (part.type === 'image_url' || part.type === 'input_image' || part.type === 'image') {
        hasMedia = true;
      }
    }
    const text = texts.join('');
    if (!text && hasMedia) {
      throw new HttpError(400, `${at}包含图片等非文本内容，本机模型仅支持文本消息，请去掉图片后重试`);
    }
    return text;
  }
  throw new HttpError(400, `${at}格式错误`);
}

/**
 * 校验并清洗对话历史。
 * 默认（网页 /api/chat）只允许 user / assistant 角色；
 * allowSystem=true 时（外部 /v1/chat/completions）额外允许 system / developer，
 * developer 会归一化为 system。服务端自己的系统提示仍会插在最前面。
 */
export function validateMessages(input, { allowSystem = false } = {}) {
  if (!Array.isArray(input) || input.length === 0) {
    throw new HttpError(400, 'messages 必须是非空数组');
  }
  if (input.length > LIMITS.maxMessages) {
    throw new HttpError(400, `单次对话最多 ${LIMITS.maxMessages} 条消息`);
  }
  let total = 0;
  const out = input.map((m, i) => {
    if (!m || typeof m !== 'object') throw new HttpError(400, `第 ${i + 1} 条消息格式错误`);
    let role = m.role;
    if (role === 'developer' && allowSystem) role = 'system';
    const allowed = allowSystem ? ['user', 'assistant', 'system'] : ['user', 'assistant'];
    if (!allowed.includes(role)) {
      throw new HttpError(400, `第 ${i + 1} 条消息角色非法`);
    }
    const content = normalizeMessageContent(m.content, { role, index: i });
    // 用户消息必须有实际内容；assistant 历史允许空文本（占位/被中断的回复）
    if (role === 'user' && content.trim() === '') {
      throw new HttpError(400, `第 ${i + 1} 条消息内容为空`);
    }
    if (content.length > LIMITS.maxMessageChars) {
      throw new HttpError(400, `第 ${i + 1} 条消息过长`);
    }
    total += content.length;
    return { role, content };
  }).filter((m) => m.role === 'user' || m.content.trim() !== '');
  if (total > LIMITS.maxTotalChars) throw new HttpError(400, '对话总长度超过限制，请新建会话');
  if (out[out.length - 1].role !== 'user') throw new HttpError(400, '最后一条消息必须来自用户');
  return out;
}

/** 由服务端单独处理或禁止传递的生成字段（大小写不敏感）。 */
const RESERVED_GENERATION_PARAMS = new Set([
  'model',
  'messages',
  'stream',
  'temperature',
  'max_tokens',
  'maxtokens',
  'max_completion_tokens',
  'authorization',
  'api_key',
  'apikey',
  'tools',
  'tool_choice',
]);

/**
 * 复制并校验 OpenAI 兼容供应商可接受的额外生成参数。
 * 支持标准采样项及供应商自定义项，但保留模型路由、消息、鉴权和输出上限等服务端控制字段。
 */
export function validateGenerationParams(input, { maxBytes = 16 * 1024 } = {}) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new HttpError(400, '生成参数必须是 JSON 对象');
  }
  const cloneValue = (value, depth = 0) => {
    if (depth > 5) throw new HttpError(400, '生成参数嵌套层级不能超过 5 层');
    if (value === null || typeof value === 'boolean') return value;
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) throw new HttpError(400, '生成参数不能包含非有限数字');
      return value;
    }
    if (typeof value === 'string') {
      if (value.length > 8192) throw new HttpError(400, '单个生成参数字符串不能超过 8192 字符');
      return value;
    }
    if (Array.isArray(value)) {
      if (value.length > 128) throw new HttpError(400, '生成参数数组不能超过 128 项');
      return value.map((item) => cloneValue(item, depth + 1));
    }
    if (typeof value === 'object') {
      const result = {};
      for (const [key, child] of Object.entries(value)) {
        if (
          !/^[A-Za-z][A-Za-z0-9_.-]{0,63}$/.test(key) ||
          ['__proto__', 'prototype', 'constructor'].includes(key.toLowerCase()) ||
          /(?:authorization|api[_-]?key|access[_-]?token|password|secret|credential)/i.test(key)
        ) {
          throw new HttpError(400, `生成参数名称非法：${key.slice(0, 80)}`);
        }
        result[key] = cloneValue(child, depth + 1);
      }
      return result;
    }
    throw new HttpError(400, '生成参数仅支持 JSON 值');
  };

  const result = {};
  for (const [key, value] of Object.entries(input)) {
    if (!/^[A-Za-z][A-Za-z0-9_.-]{0,63}$/.test(key)) {
      throw new HttpError(400, `生成参数名称非法：${key.slice(0, 80)}`);
    }
    if (RESERVED_GENERATION_PARAMS.has(key.toLowerCase())) {
      throw new HttpError(400, `生成参数 ${key} 由服务端单独控制，不能作为额外参数传入`);
    }
    if (
      /(?:authorization|api[_-]?key|access[_-]?token|password|secret|credential)/i.test(key) ||
      ['tools', 'tool_choice'].includes(key.toLowerCase())
    ) {
      throw new HttpError(400, `不支持或不允许覆盖生成参数：${key}`);
    }
    result[key] = cloneValue(value);
  }
  if (Buffer.byteLength(JSON.stringify(result), 'utf8') > maxBytes) {
    throw new HttpError(400, `生成参数 JSON 不能超过 ${maxBytes} 字节`);
  }
  return result;
}

/**
 * 校验 OpenAI 兼容的 /v1/chat/completions 请求体。
 * 返回 { messages, stream, temperature, maxTokens, params, model }。
 * model 若与服务端配置不一致返回 404（与 OpenAI 的 model_not_found 语义一致）。
 */
export function validateChatCompletionBody(body, cfg = {}) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new HttpError(400, '请求体必须是 JSON 对象');
  const messages = validateMessages(body.messages, { allowSystem: true });

  const model = body.model === undefined || body.model === null ? '' : String(body.model);
  if (model && cfg.llmModel && model !== cfg.llmModel) {
    throw new HttpError(404, `模型不存在：${model}`);
  }

  let temperature;
  if (body.temperature !== undefined && body.temperature !== null) {
    temperature = Number(body.temperature);
    if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
      throw new HttpError(400, 'temperature 必须是 0 到 2 之间的数字');
    }
  }

  let maxTokens;
  const maxFields = ['max_tokens', 'max_completion_tokens', 'maxTokens'].filter(
    (key) => body[key] !== undefined && body[key] !== null,
  );
  if (maxFields.length > 1) throw new HttpError(400, 'max_tokens、max_completion_tokens 与 maxTokens 只能指定一个');
  if (maxFields.length) {
    maxTokens = Number(body[maxFields[0]]);
    const cap = cfg.apiMaxTokens || LIMITS.maxTotalChars;
    if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > cap) {
      throw new HttpError(400, `max_tokens 必须是 1 到 ${cap} 之间的整数`);
    }
  }

  const reserved = new Set([
    'model',
    'messages',
    'stream',
    'temperature',
    'max_tokens',
    'max_completion_tokens',
    'maxTokens',
  ]);
  const generationParams = {};
  for (const [key, value] of Object.entries(body)) {
    if (!reserved.has(key)) generationParams[key] = value;
  }

  return {
    messages,
    stream: body.stream === true,
    temperature,
    maxTokens,
    params: validateGenerationParams(generationParams),
    model: model || cfg.llmModel || '',
  };
}

function upstreamMessages(cfg, messages) {
  return [{ role: 'system', content: `${cfg.systemPrompt}\n\n${SEARCH_EVIDENCE_SYSTEM_GUARD}` }, ...messages];
}

async function fetchUpstream({ cfg, messages, temperature, maxTokens, params, signal }) {
  if (!cfg.llmBaseUrl || !cfg.llmModel) {
    throw new HttpError(503, '服务端尚未配置 LLM_BASE_URL / LLM_MODEL');
  }
  const controller = new AbortController();
  const onClientGone = () => controller.abort();
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener('abort', onClientGone, { once: true });

  // 等待上游开始响应的超时：上游卡住时不能让请求无限挂着（流式正文阶段不计时）
  const timeoutMs = Number(cfg.upstreamTimeoutMs) > 0 ? Number(cfg.upstreamTimeoutMs) : 60_000;
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  let upstream;
  try {
    upstream = await fetch(`${cfg.llmBaseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        accept: 'text/event-stream',
        ...(cfg.llmApiKey ? { authorization: `Bearer ${cfg.llmApiKey}` } : {}),
      },
      body: JSON.stringify({
        ...cfg.defaultGenerationParams,
        ...params,
        model: cfg.llmModel,
        messages: upstreamMessages(cfg, messages),
        temperature: temperature ?? cfg.llmTemperature,
        ...(maxTokens ? { max_tokens: maxTokens } : {}),
        stream: true,
      }),
      signal: controller.signal,
    });
  } catch (err) {
    signal?.removeEventListener('abort', onClientGone);
    if (timedOut) {
      throw new HttpError(504, `模型服务响应超时（${Math.round(timeoutMs / 1000)} 秒内未开始返回），请稍后重试。`);
    }
    if (controller.signal.aborted) return { aborted: true };
    throw new HttpError(502, `无法连接模型服务：${err.cause?.code || err.message}`);
  } finally {
    clearTimeout(timer);
  }
  return { upstream, controller, onClientGone };
}

async function assertUpstreamOk(upstream) {
  if (!upstream.ok) {
    const detail = (await upstream.text().catch(() => '')).slice(0, 300);
    console.error(`[chat] upstream ${upstream.status}: ${detail}`);
    throw new HttpError(502, `模型服务返回错误（HTTP ${upstream.status}）`);
  }
}

/** 给上游 SSE chunk 补齐 OpenAI 客户端期望的 id / created / model / object 字段。 */
export function normalizeChunk(json, { id, created, model }) {
  if (!json || typeof json !== 'object' || !Array.isArray(json.choices)) return json;
  const out = { ...json };
  if (!out.id) out.id = id;
  if (!out.object) out.object = 'chat.completion.chunk';
  if (typeof out.created !== 'number') out.created = created;
  if (!out.model) out.model = model;
  return out;
}

function newCompletionId() {
  return `chatcmpl-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * 将请求转发到 OpenAI 兼容的 /chat/completions（stream=true），并把上游 SSE 流式写回客户端。
 * 客户端断开时会取消上游请求，避免浪费 token。
 * normalize=true 时会给每个 chunk 补齐 id/created/model（供 /v1 外部调用方）。
 * onText 可选：每段输出文本回调一次（仅 normalize 路径解析得到），供用量统计累计。
 */
export async function proxyChatStream({ cfg, messages, res, signal, temperature, maxTokens, params, normalize = false, onText = null }) {
  const fetched = await fetchUpstream({ cfg, messages, temperature, maxTokens, params, signal });
  if (fetched.aborted) return;
  const { upstream, controller, onClientGone } = fetched;
  try {
    await assertUpstreamOk(upstream);
  } catch (err) {
    signal?.removeEventListener('abort', onClientGone);
    throw err;
  }

  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  // 告诉 Nginx 等反向代理不要缓冲，否则流式输出会变成“一次性吐出”
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders?.();

  const id = newCompletionId();
  const created = Math.floor(Date.now() / 1000);
  let buf = '';
  const decoder = new TextDecoder();
  const emitLine = (line) => {
    if (!normalize) {
      res.write(`${line}\n`);
      return;
    }
    const trimmed = line.trimEnd();
    if (!trimmed.startsWith('data:')) {
      res.write(`${line}\n`);
      return;
    }
    const data = trimmed.slice(5).trim();
    if (!data || data === '[DONE]') {
      res.write(`${line}\n`);
      return;
    }
    try {
      const json = normalizeChunk(JSON.parse(data), { id, created, model: cfg.llmModel });
      // 用量统计钩子：仅统计输出文本长度，不改变转发内容；回调异常不影响转发。
      if (onText) {
        const delta = json.choices?.[0]?.delta?.content;
        if (typeof delta === 'string' && delta) {
          try { onText(delta); } catch { /* 统计失败不影响转发 */ }
        }
      }
      res.write(`data: ${JSON.stringify(json)}\n`);
    } catch {
      res.write(`${line}\n`);
    }
  };

  try {
    for await (const chunk of upstream.body) {
      if (res.writableEnded) break;
      if (!normalize) {
        res.write(Buffer.from(chunk));
      } else {
        buf += decoder.decode(chunk, { stream: true });
        let nl;
        while ((nl = buf.indexOf('\n')) !== -1) {
          emitLine(buf.slice(0, nl));
          buf = buf.slice(nl + 1);
        }
      }
    }
    if (normalize && buf) emitLine(buf);
  } catch (err) {
    if (!controller.signal.aborted) console.error('[chat] stream error:', err.message);
  } finally {
    signal?.removeEventListener('abort', onClientGone);
    if (!res.writableEnded) res.end();
  }
}

/**
 * 非流式收集：内部仍用上游 stream=true，收集全部 delta 后返回 OpenAI 格式的完整回复。
 * 返回 { id, created, model, content, finishReason }。
 */
export async function collectChatCompletion({ cfg, messages, temperature, maxTokens, params, signal }) {
  const fetched = await fetchUpstream({ cfg, messages, temperature, maxTokens, params, signal });
  if (fetched.aborted) return null;
  const { upstream, controller, onClientGone } = fetched;
  try {
    await assertUpstreamOk(upstream);
    const id = newCompletionId();
    const created = Math.floor(Date.now() / 1000);
    let content = '';
    let finishReason = 'stop';
    const decoder = new TextDecoder();
    let buf = '';
    const handleLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return;
      const data = trimmed.slice(5).trim();
      if (!data || data === '[DONE]') return;
      let json;
      try {
        json = JSON.parse(data);
      } catch {
        return;
      }
      const choice = json.choices?.[0];
      if (!choice) return;
      const delta = choice.delta?.content ?? choice.message?.content ?? '';
      if (typeof delta === 'string') content += delta;
      if (choice.finish_reason) finishReason = choice.finish_reason;
    };
    for await (const chunk of upstream.body) {
      if (signal?.aborted) break;
      buf += decoder.decode(chunk, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) !== -1) {
        handleLine(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
      }
    }
    if (buf) handleLine(buf);
    // 上游“成功但空内容”多半是模型/网关异常：明确报错，不要把空回复当成功返回
    if (!content.trim()) {
      throw new HttpError(502, '模型服务返回了空内容，请检查模型服务状态或更换模型后重试。');
    }
    return { id, created, model: cfg.llmModel, content, finishReason };
  } finally {
    signal?.removeEventListener('abort', onClientGone);
  }
}
