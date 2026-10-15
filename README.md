# -first-
The first repository of me,and first of "炸药工厂"

# OJ 刷题助手（可嵌入 iframe 的开源大模型聊天站）

一个可以部署为**专属网址**的 AI 聊天站点，专为算法刷题场景设计：

- **可被 iframe 嵌入**：不设置 `X-Frame-Options`，通过 CSP `frame-ancestors` 控制嵌入范围，可嵌入 Better Hydro 等 OJ 的 400–450px 侧边栏。
- **嵌入模式**：`?embed=1` 一键隐藏自带侧边栏、Logo 和设置按钮。
- **postMessage 填题**：父页面可把题目文本填入输入框（不自动发送，除非显式指定）。
- **Markdown + 代码高亮 + 复制按钮 + LaTeX 公式**（KaTeX，本地打包，不依赖外部 CDN）。
- **多轮对话记忆**：会话保存在浏览器本地，发送时携带最近的上下文，可以直接追问“为什么 TLE”。
- **密码登录**：令牌放在请求头中，不依赖 Cookie，保证跨站 iframe 也能正常使用。
- **API Key 只在服务端**：浏览器不会拿到模型密钥。
- **流式输出**：逐字显示，并且在 Nginx 等反向代理后也不会被缓冲。

模型使用任意 **OpenAI 兼容接口**，因此可以接入开源模型（Qwen、DeepSeek、Llama 等）的云端 API，也可以用 vLLM / Ollama 自建。

## 快速开始（本地）

需要 Node.js 18.18 及以上版本。

```bash
cp .env.example .env        # 按注释填写 APP_PASSWORD、LLM_BASE_URL、LLM_MODEL、LLM_API_KEY
npm install
npm run build               # 构建前端到 web/dist
npm start                   # 默认监听 0.0.0.0:3000
```

打开 `http://localhost:3000`，输入 `APP_PASSWORD` 即可使用。

前端开发模式（热更新，接口转发到 3000 端口）：先执行 `npm start`，再执行 `npm run dev:web`。

运行测试：`npm test`（不需要真实 API Key，使用内置的模拟上游）。

## 部署到专属网址

GitHub Pages 只能托管静态文件，无法保存 API Key 和校验密码，因此本项目需要一台服务器（VPS、云主机均可）。

1. 准备域名（如 `ai.example.com`），解析到服务器 IP。
2. 服务器上安装 Docker，克隆仓库，编写 `.env`（参考 `.env.example`，务必修改 `APP_PASSWORD` 并设置 `SESSION_SECRET`）。
3. 启动：`docker compose up -d --build`（容器只监听 `127.0.0.1:3000`）。
4. 配置 HTTPS 反向代理，参考 `deploy/nginx.conf`（关闭 `proxy_buffering`，不添加 `X-Frame-Options`）。可以用 Certbot 申请证书，也可以直接用 Caddy 自动签发。
5. 访问 `https://ai.example.com` 登录即可。

## 嵌入到 Better Hydro 侧边栏

在脚本里创建 iframe（注意 `allow="clipboard-write"` 以便代码复制按钮可用）：

```html
<iframe
  src="https://ai.example.com/?embed=1"
  style="width:100%;height:100%;border:0"
  allow="clipboard-write"></iframe>
```

把题目发送到输入框（父页面 → iframe）：

```js
iframe.contentWindow.postMessage(
  { type: 'oj-ai:fill', text: 题目文本, mode: 'replace' /* 或 'append' */, send: false /* true 则直接发送 */ },
  'https://ai.example.com'   // 目标 origin 必须写明，不要用 '*'
);
```

iframe 会回复 `{ type: 'oj-ai:filled' }`；iframe 启动后会向父页面发送 `{ type: 'oj-ai:ready' }`（需要浏览器提供 referrer）。

安全说明：只有 `ALLOWED_PARENT_ORIGINS` 中列出的父页面 origin 才能填充输入框；其他来源的消息会被忽略。

注意：同一站点之间共享登录状态；跨站嵌入时，浏览器可能把 iframe 的存储分区隔离，需要在 iframe 内重新输入一次密码。

## 配置项

| 变量 | 说明 |
|---|---|
| `APP_PASSWORD` | 登录密码，必填，至少 8 位 |
| `SESSION_SECRET` | 令牌签名密钥，生产环境必须设置 |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI 兼容接口地址、密钥、模型名 |
| `SYSTEM_PROMPT` | 自定义系统提示词，默认是算法竞赛教练角色 |
| `FRAME_ANCESTORS` | 允许嵌入本站的父页面（CSP），默认 `*` |
| `ALLOWED_PARENT_ORIGINS` | 允许 postMessage 填充输入框的父页面来源 |
| `TRUST_PROXY` | 部署在反向代理后设为 `1`，用于正确识别客户端 IP（登录限流） |
| `TOKEN_TTL_HOURS` | 登录有效期，默认 168 小时 |

完整说明见 `.env.example`。

## 目录结构

```
server/        Express 后端：登录鉴权、模型流式代理、安全响应头
web/           前端（Vite + 原生 JS）：Markdown / KaTeX / 高亮 / 会话 / 嵌入模式
test/          单元测试与服务端集成测试（含模拟上游）
deploy/        Nginx 反向代理示例
Dockerfile     生产镜像
```
