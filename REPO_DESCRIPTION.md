# 仓库描述：-first-（OJ 刷题助手）

## 一句话简介（GitHub About 推荐）

> 可嵌入 iframe 的 OJ 刷题 AI 助手：支持浏览器 WebGPU 本地推理，也支持自建 Node.js 服务器提供 OpenAI 兼容 API（API Key）与可分享网页链接 / iframe；可完全自托管本地模型，不依赖第三方托管平台提供远程服务。

**GitHub Topics 建议：** `oj` `ai-assistant` `webllm` `webgpu` `qwen` `iframe` `self-hosted` `openai-compatible` `llm` `algorithm` `hydro`

---

## 项目定位

本仓库是作者的第一个仓库（`TNT-THT/-first-`，即“炸药工厂”的第一个项目），核心是一个专为算法刷题（Online Judge）设计的 AI 助手网站，包名 `oj-ai-assistant`。

项目提供两种运行方式。默认静态模式下，开源大模型直接在用户浏览器里通过 WebGPU 运行（WebLLM），对话内容不会离开本机；静态文件可部署到任意自有 Web 服务器，第三方静态托管只是可选方式。

**远程接入则由自建 Node.js 服务器独立提供**：服务端模式使用 Express 托管网站，并提供 API Key 鉴权的 OpenAI 兼容接口（`/v1/models`、`/v1/chat/completions`，流式 / 非流式），以及网页分享链接（`?q=...`、`&send=1`）和 iframe 嵌入（`?embed=1`）。完全自托管时可让服务端连接同机 Ollama / vLLM，不需要依赖第三方 AI API 或托管平台。仓库还包括独立的 Python 训练工具链。

## 核心功能

- **浏览器内推理**：基于 `@mlc-ai/web-llm 0.2.86`，模型首次从 Hugging Face 下载后缓存在浏览器本地，之后离线可用；对话、历史、知识库全部保存在本机 localStorage。
- **自建服务器 API**：OpenAI 兼容 `/v1/models` 与 `/v1/chat/completions`，支持 SSE 流式和非流式响应、`Authorization: Bearer` / `x-api-key`、CORS；API 管理员验证后可使用网页调试台、服务端模型参数控制台和 API Key 管理；管理员可按用户/应用创建独立 Key 并分发，非管理员用户可用自己的 Key 从其他网站或软件调用 API。动态 Key 只存摘要、明文创建时展示一次，也可用 `API_KEYS` 配置静态密钥。`/api/docs` 与 `/api/info` 提供人类 / 机器可读接入说明。
- **网页链接接入**：分享 `?q=...` 预填问题、`&send=1` 自动发送，或 `?embed=1` iframe 嵌入；网站面板可生成分享链接和 iframe 片段，父页面可按 `ALLOWED_PARENT_ORIGINS` 白名单通过 `postMessage` 填题。
- **32 个可选模型**（下拉切换，列表在 `web/public/config.json` 的 `webllm.models` 中维护）：Qwen / Qwen-Coder / Qwen3 / Qwen3.5、Google Gemma、Microsoft Phi-4 mini、Mistral Ministral 3、DeepSeek-R1 蒸馏版、Meta Llama（大/超大级）；含推理模型（思考模式）与非推理模型、q4f16/q4f32 量化版本，标签标注尺寸、显存需求与“纯文本”。
- **对话体验**：Markdown + 代码高亮 + 一键复制 + KaTeX 公式、引用回复、重新生成、思考过程折叠、历史会话折叠加载、导出对话为 Markdown、多 AI 对话（回复标注来源 AI）、重复输出检测、滑动窗口 / 历史截断与 KV 缓存复用、上下文与思考窗口预算管理。
- **工具面板**：思考模式、滑动窗口、联网资料检索统一收在输入框「工具」按钮中，偏好保存在本浏览器。
- **联网资料检索（可选，默认关闭）**：浏览器直连维基百科、Stack Overflow、Hacker News 公开 API（可扩展 SearXNG / 自定义接口），快速/深入两种模式，带来源卡片与 `[编号]` 引用审计；含敏感信息时 fail-closed 不发请求。
- **图像生成**：浏览器内运行 Stable Diffusion 1.5（ONNX Runtime Web），含调度器、画廊存档、模型下载续传等完整管线。
- **模型导出 / 导入（管理员）**：已下载模型可导出为 `.ojmodel` 文件（`OJMD` 魔数 + JSON 头 + 原始数据），在另一台电脑导入为“新建AI”离线使用；导入模型使用独立虚拟地址，不覆盖官方模型。
- **AI 适应（管理员反馈，仅本机）**：管理员可写入附加说明、参数覆盖、错误说法更正、本地知识库（最高优先级注入系统提示），并做答案字符串检查；可导出“固化数据”用训练工具把知识固化进权重。
- **网站包与本机数据导出（管理员）**：`npm run build` 自动打包 `oj-ai-site.zip`（含 Windows/macOS/Linux 一键启动器），以及会话/知识库等本机数据的 JSON 导出；不含任何模型与密钥。
- **LoRA 训练（独立运行时）**：`training/` 提供从选模型、环境检查、数据准备、训练、转换到打包 `.ojmodel` 的一站式流程（需 GPU），训练结果通过网页“导入训练结果”加入。

## 技术栈

| 层 | 技术 |
|---|---|
| 前端 | Vite 5、原生 JS（ES Module）、WebLLM 0.2.86、ONNX Runtime Web 1.30、KaTeX、markdown-it、highlight.js、DOMPurify |
| 服务端（远程模式） | Node ≥18.18、Express 4、dotenv（`server/`，含网页登录、独立 API 管理员认证、API Key 与参数持久化、OpenAI 兼容 API、流式代理、跨域与接入文档） |
| 训练 | Python、PyTorch、MLC-LLM（`training/`，含 `ojtrain.py` CLI） |
| 部署 | 自建 Node.js / Docker + Nginx（`Dockerfile`、`docker-compose.yml`、`deploy/nginx.conf`）；静态模式也可独立部署 |
| 测试 | Node 内置 `node --test`（50+ 测试文件）、Python `unittest`（`training/tests/`） |

## 目录结构

```
.
├── web/                  # 前端站点（Vite）
│   ├── src/              # 源码：main.js、引擎（engines/）、图像生成（image/）、联网检索（search/）、下载（download/）等
│   ├── public/config.json# 模型列表、系统提示、管理员口令哈希、嵌入白名单等站点配置
│   └── index.html
├── server/               # 可选服务端模式：Express 应用、鉴权、聊天代理
├── training/             # LoRA 微调一站式工具（ojtrain.py 入口、catalog、打包）
├── scripts/              # 构建脚本：网站包打包、口令设置、一键启动器（launcher/）、SD1.5 工具
├── test/                 # Node 单元/集成测试（含 SD1.5、ONNX、搜索、适应等 fixtures）
├── docs/                 # 管理员操作指南、联网搜索评估文档
├── deploy/               # Nginx 配置
├── .github/workflows/    # GitHub Pages 自动部署
├── Dockerfile / docker-compose.yml
└── README.md             # 完整使用说明（功能、部署、嵌入、服务端模式等）
```

## 部署方式

1. **本地静态运行**：`npm run dev:web` 开发预览；`npm run build` 构建并生成网站包，解压后双击启动器（Windows `.bat` / macOS `.command` / Linux `python3 serve-site.py`）即可本机使用。静态构建也可放到任意静态 Web 服务器，但它不提供服务器 API。
2. **自建远程服务（推荐用于 API / 链接接入）**：复制 `.env.example` 为 `.env`，配置 `APP_PASSWORD`、`SESSION_SECRET`、`API_ADMIN_PASSWORD`、API Key、自有模型服务与 `PUBLIC_BASE_URL`；执行 `npm run build:remote && npm start`，或 `docker compose up -d --build`。通过 Nginx/Caddy 对外提供 HTTPS 后，网站、API 和接入文档都由自己的服务器提供。
3. **自托管本地模型**：在同机运行 Ollama / vLLM；Docker Compose 可启用 `local-llm` profile。模型推理发生在自己的服务器，不需要调用第三方模型 API。

## 安全与隐私说明

- 默认静态模式下推理与数据均在本机完成，不产生 API 费用、不上传对话。
- 服务端模式中的模型供应商 Key 可通过服务器环境变量或 API 管理员控制台设置，始终留在服务器端；控制台配置文件以 `0600` 权限保存且管理 API 不会回显密钥。外部 API 始终需要 API Key，生产部署应使用强随机密钥与 HTTPS。
- 管理员校验为浏览器端加盐哈希比对，仅作 UI 门槛，不是服务端鉴权；远程网站访问通过服务端 `APP_PASSWORD` 登录。
- 嵌入范围由 CSP `frame-ancestors` 控制（环境变量可收紧），刻意不设置 `X-Frame-Options` 以便 iframe 嵌入；`postMessage` 填题另受来源白名单控制。
- 自动发送链接 `send=1` 会在打开后立即提交 URL 中的问题，避免在不可信场景传播敏感内容；API Key 不应写入公开网页代码。
- 联网检索的查询词会发送给所选公开资料源（非匿名），敏感信息会触发 fail-closed。

## 适用场景

- OJ 刷题时的在线答疑与思路讲解（可嵌入 OJ 侧边栏，题目一键填入）；
- 无服务器预算、希望零成本自托管的个人/小团队 AI 助手站点；
- 离线环境下通过 `.ojmodel` 导入导出分发模型；
- 用自有 GPU 对常用模型做 LoRA 微调并回灌网页使用。
