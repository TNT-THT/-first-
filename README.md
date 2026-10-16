# -first-
The first repository of me,and first of "炸药工厂"

# OJ 刷题助手（可嵌入 iframe 的开源大模型站点）

一个专为算法刷题设计的 AI 助手，**默认是纯静态网站，部署在 GitHub Pages 上即可使用，不需要服务器，也不需要任何 API Key**。
模型（Qwen2.5-Coder，开源）直接在你的浏览器里通过 WebGPU 运行，对话内容不会离开你的电脑。

## 功能

- **专属网址**：部署到 GitHub Pages 后得到固定地址，例如 `https://tnt-tht.github.io/-first-/`。
- **访问口令**：打开网站需要输入口令（只保存加盐哈希，明文不在仓库里）。
- **可被 iframe 嵌入**：不设置 `X-Frame-Options`，可嵌入 Better Hydro 等 OJ 的 400–450px 侧边栏。
- **嵌入模式**：`?embed=1` 一键隐藏自带侧边栏、Logo 和模型设置。
- **postMessage 填题**：父页面可以把题目文本填入输入框（默认不自动发送）。
- **Markdown、代码高亮、一键复制、LaTeX 公式**（KaTeX，本地打包）。
- **多轮对话记忆**：会话保存在浏览器本地，可以直接追问“为什么 TLE”。
- **模型可选**（下拉框切换，列表在 `web/public/config.json` 的 `webllm.models` 中维护）。标签中的尺寸按参数量划分：
  极小 ≤1B、小 1–4B、中 4–9B、大 ≥10B。共 30 个：非推理 24 个、推理 6 个。
  - 厂商：Qwen（阿里）、Meta Llama、Google Gemma、Mistral、Microsoft Phi、Hugging Face SmolLM2、AI2 OLMo、TinyLlama、Stability StableLM、Nous Hermes、DeepSeek（仅 R1 蒸馏版，均为推理模型）。
  - 大型：Llama 2 13B（需约 12 GB 显存，多数设备无法运行）。
  - DeepSeek 的小型非推理模型（如 deepseek-coder-1.3b-instruct）不在 WebLLM 内置列表中，无法直接使用。
- **自定义模型记录**：`webllm.extraRecords` 用于保留 WebLLM 内置列表中被注释掉、但仍可使用的模型（当前是 DeepSeek-R1 蒸馏 1.5B）。WebLLM 官方注释提示该模型可能存在正确性问题，请以实际输出为准。
- **推理模型的思考过程会被隐藏**：Qwen3、DeepSeek-R1 等模型输出的 `<think>` 内容不展示，只显示最终回答，也不计入历史上下文。
- **系统提示词**可在 `config.json` 的 `systemPrompt` 中自由修改，默认不限制话题与回答格式。

## 硬件与浏览器要求

- 需要 **WebGPU**：桌面端最新版 Chrome 或 Edge。
- 显存 / 共享显存建议（WebLLM 估算值，q4f16 量化）：0.5B 约 0.9 GB，1.5B 约 1.6 GB，1.7B 约 2.0 GB，3B 约 2.5 GB，4B 约 3.4 GB，7B / 8B 约 5–5.7 GB。
- 首次使用时需要从 Hugging Face 下载模型权重（文件较大，耗时取决于网络），下载后由浏览器缓存，之后不再需要下载。
- 推理模型（Qwen3、DeepSeek-R1）会先输出较长的思考过程，默认 `maxTokens` 为 1024，可能不足以写完答案；如遇到“只输出了思考过程”的提示，可在 `config.json` 中调大 `maxTokens`，或换用非推理模型。
- 中国大陆网络访问 Hugging Face 可能需要代理工具，请先确认代理已开启并允许访问 `huggingface.co`。
- 如果浏览器不支持 WebGPU，页面会在状态栏给出提示，而不是静默失败。

## 部署到 GitHub Pages（推荐，零成本）

1. 仓库 **Settings → Pages**：Source 选择 **GitHub Actions**。
2. 推送代码到 `main` 或 `arena/64ed15eb-first` 分支，`.github/workflows/pages.yml` 会自动测试、构建并发布。
3. 访问 `https://tnt-tht.github.io/-first-/`。

> 注意：GitHub Pages 不允许自定义响应头，因此无法精细控制嵌入来源。如果发现 iframe 嵌入被拦截，请改用 Netlify 或 Cloudflare Pages（导入本仓库即可，`web/public/_headers` 已经配置好 `frame-ancestors`）。

## 访问口令

口令保存在 `web/public/config.json` 的 `access` 字段中，只存储加盐的 SHA-256 哈希。

```bash
npm run set-password              # 自动生成随机口令并打印（请妥善保存）
npm run set-password -- "新口令"   # 使用指定口令（至少 8 位）
```

修改后提交并推送，等待 Actions 完成即可生效。

**安全说明**：纯静态站点的口令只是“软门禁”，挡住随意访问，并不是服务端鉴权。由于推理在浏览器本地完成，站点本身不产生 API 费用，所以这个门禁主要用于保护你自己的使用环境。

## 本地运行

```bash
npm install
npm run dev:web        # 开发模式，http://localhost:5173
npm run build          # 构建纯静态版本到 web/dist
npm test               # 运行测试
```

## 嵌入到 Better Hydro 侧边栏

```html
<iframe
  src="https://tnt-tht.github.io/-first-/?embed=1"
  style="width:100%;height:100%;border:0"
  allow="clipboard-write"></iframe>
```

把题目发送到输入框（父页面 → iframe）：

```js
iframe.contentWindow.postMessage(
  { type: 'oj-ai:fill', text: 题目文本, mode: 'replace' /* 或 'append' */, send: false /* true 则直接发送 */ },
  'https://tnt-tht.github.io'   // 目标 origin 必须写明，不要用 '*'
);
```

iframe 会回复 `{ type: 'oj-ai:filled' }`，启动后会向父页面发送 `{ type: 'oj-ai:ready' }`。
只有 `web/public/config.json` 中 `allowedParentOrigins` 列出的父页面才能填充输入框（默认 `https://hydro.ac` 及其子域）。

**已知限制**：
- 模型权重存放在浏览器的存储中，并按“顶层站点”隔离。也就是说，在 Better Hydro 的页面里嵌入时，需要在那个站点下再下载一次模型。
- 跨站嵌入时口令需要在 iframe 内重新输入一次。

## 可选：服务端模式（自有 API 或自建模型）

如果你以后有 OpenAI 兼容接口（DeepSeek、通义千问 DashScope、自建 vLLM / Ollama 等），也可以使用服务端模式。
这种模式需要一台服务器，API Key 只保存在服务端。

```bash
cp .env.example .env     # 填写 APP_PASSWORD、LLM_BASE_URL、LLM_MODEL、LLM_API_KEY 等
npm run build:remote     # 构建服务端版前端（使用服务端登录与流式接口）
npm start                # 监听 0.0.0.0:3000
```

也可以用 `docker compose up -d --build` 部署（容器只监听 `127.0.0.1:3000`，再用 `deploy/nginx.conf` 配置 HTTPS 反向代理，注意关闭 `proxy_buffering`）。

## 目录结构

```
web/                 前端（Vite + 原生 JS）
  public/config.json 站点配置：模型列表、系统提示词、口令哈希、允许嵌入的父页面
  public/_headers    Netlify / Cloudflare Pages 的响应头配置
  src/engines/       推理后端：webllm（浏览器内，默认）与 remote（服务端）
server/              可选的服务端（登录鉴权、OpenAI 兼容接口流式代理）
scripts/             口令设置脚本
test/                单元测试与服务端集成测试
deploy/              Nginx 反向代理示例（服务端模式）
.github/workflows/   GitHub Pages 自动部署
```
