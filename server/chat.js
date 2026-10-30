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

/** 校验并清洗前端传来的对话历史，只允许 user / assistant 角色。 */
export function validateMessages(input) {
  if (!Array.isArray(input) || input.length === 0) {
    throw new HttpError(400, 'messages 必须是非空数组');
  }
  if (input.length > LIMITS.maxMessages) {
    throw new HttpError(400, `单次对话最多 ${LIMITS.maxMessages} 条消息`);
  }
  let total = 0;
  const out = input.map((m, i) => {
    if (!m || typeof m !== 'object') throw new HttpError(400, `第 ${i + 1} 条消息格式错误`);
    if (m.role !== 'user' && m.role !== 'assistant') {
      throw new HttpError(400, `第 ${i + 1} 条消息角色非法`);
    }
    if (typeof m.content !== 'string' || m.content.trim() === '') {
      throw new HttpError(400, `第 ${i + 1} 条消息内容为空`);
    }
    if (m.content.length > LIMITS.maxMessageChars) {
      throw new HttpError(400, `第 ${i + 1} 条消息过长`);
    }
    total += m.content.length;
    return { role: m.role, content: m.content };
  });
  if (total > LIMITS.maxTotalChars) throw new HttpError(400, '对话总长度超过限制，请新建会话');
  if (out[out.length - 1].role !== 'user') throw new HttpError(400, '最后一条消息必须来自用户');
  return out;
}

/**
 * 将请求转发到 OpenAI 兼容的 /chat/completions（stream=true），并把上游 SSE 原样流式写回客户端。
 * 客户端断开时会取消上游请求，避免浪费 token。
 */
export async function proxyChatStream({ cfg, messages, res, signal }) {
  if (!cfg.llmBaseUrl || !cfg.llmModel) {
    throw new HttpError(503, '服务端尚未配置 LLM_BASE_URL / LLM_MODEL');
  }

  const controller = new AbortController();
  const onClientGone = () => controller.abort();
  signal?.addEventListener('abort', onClientGone, { once: true });

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
        model: cfg.llmModel,
        messages: [{ role: 'system', content: `${cfg.systemPrompt}\n\n${SEARCH_EVIDENCE_SYSTEM_GUARD}` }, ...messages],
        temperature: cfg.llmTemperature,
        stream: true,
      }),
      signal: controller.signal,
    });
  } catch (err) {
    if (controller.signal.aborted) return;
    throw new HttpError(502, `无法连接模型服务：${err.cause?.code || err.message}`);
  }

  if (!upstream.ok) {
    const detail = (await upstream.text().catch(() => '')).slice(0, 300);
    console.error(`[chat] upstream ${upstream.status}: ${detail}`);
    throw new HttpError(502, `模型服务返回错误（HTTP ${upstream.status}）`);
  }

  res.status(200);
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  // 告诉 Nginx 等反向代理不要缓冲，否则流式输出会变成“一次性吐出”
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders?.();

  try {
    for await (const chunk of upstream.body) {
      if (res.writableEnded) break;
      res.write(Buffer.from(chunk));
    }
  } catch (err) {
    if (!controller.signal.aborted) console.error('[chat] stream error:', err.message);
  } finally {
    signal?.removeEventListener('abort', onClientGone);
    if (!res.writableEnded) res.end();
  }
}
