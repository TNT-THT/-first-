import http from 'node:http';

/**
 * 模拟 OpenAI 兼容的 /v1/chat/completions（SSE 流式），用于测试，不访问外部网络。
 * - 普通提问：逐段返回 "你好，世界$O(n)$"
 * - 消息中包含 "__ERROR__"：返回 500，用于测试错误处理
 * - 消息中包含 "__SLOW__"：延迟 1.2 秒再开始输出，用于测试加载状态
 */
export function startMockLLM() {
  const requests = [];

  const respond = (res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    const pieces = ['你好', '，', '世界', '$O(n)$'];
    let i = 0;
    const tick = () => {
      if (i < pieces.length) {
        const chunk = { choices: [{ delta: { content: pieces[i++] } }] };
        res.write(`data: ${JSON.stringify(chunk)}\n\n`);
        setTimeout(tick, 5);
      } else {
        res.write('data: [DONE]\n\n');
        res.end();
      }
    };
    tick();
  };

  const server = http.createServer((req, res) => {
    let raw = '';
    req.on('data', (c) => (raw += c));
    req.on('end', () => {
      if (req.method !== 'POST' || req.url !== '/v1/chat/completions') {
        res.writeHead(404).end();
        return;
      }
      const body = JSON.parse(raw);
      requests.push({ headers: req.headers, body });
      const last = body.messages[body.messages.length - 1]?.content || '';
      if (last.includes('__ERROR__')) {
        res.writeHead(500, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'boom sk-LEAKED-KEY-SHOULD-NOT-REACH-CLIENT' }));
        return;
      }
      const delay = last.includes('__SLOW__') ? 1200 : 0;
      setTimeout(() => respond(res), delay);
    });
  });

  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}/v1`,
        requests,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}
