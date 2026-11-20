# 单台电脑本机运行网页与 API

本指南把网站和 OpenAI 兼容 API 都运行在同一台电脑上。API 只绑定 `127.0.0.1`，只有这台电脑上的浏览器和软件能访问；不需要公网域名、VPS 或第三方托管平台。

> **模型说明：**本机 API 是 Node.js 服务端，它会把请求转发到 `LLM_BASE_URL`。默认目标是本机 Ollama；浏览器静态版内置的 WebLLM 和服务端 API 是两套独立的推理引擎，API 不会直接调用另一个浏览器标签页里的 WebLLM。

## 准备

1. 安装 Node.js 18.18 或更高版本。
2. 若不打算使用云模型 API，在同一台电脑安装并启动 [Ollama](https://ollama.com/)；没有自定义 `.env` 模型配置时，本机启动器默认使用 `qwen2.5:3b`。首次下载该示例模型：

   ```bash
   ollama pull qwen2.5:3b
   ```

   也可以使用其他本机 OpenAI 兼容模型服务；在网站参数控制台或项目根目录 `.env` 中设置正确的 `LLM_BASE_URL` 和模型名。使用 Ollama 时默认地址是 `http://127.0.0.1:11434/v1`。

## 启动

在项目根目录打开终端，运行：

```bash
npm run start:local-api
```

该命令会在缺少依赖时运行 `npm ci`，构建服务端网页版本，再启动本机 Node.js 服务。第一次启动时会生成被 Git 忽略的 `.local-api.json`，设置本机网站口令、管理员口令和随机会话密钥；生成的口令会在终端显示一次。若项目已有有效 `.env` 配置，则尽量沿用。服务会强制绑定到 `127.0.0.1`，不会向局域网或公网开放。

保持该终端运行，在**同一台电脑**打开：

- 网页：`http://127.0.0.1:3000`
- API Base URL：`http://127.0.0.1:3000/v1`
- 对话接口：`http://127.0.0.1:3000/v1/chat/completions`

如果 3000 端口已被占用，可设置 `LOCAL_API_PORT` 后启动，例如：

```bash
LOCAL_API_PORT=3001 npm run start:local-api
```

Windows PowerShell：

```powershell
$env:LOCAL_API_PORT = "3001"
npm run start:local-api
```

服务运行期间不要关闭终端；按 `Ctrl+C` 停止。电脑休眠、停止 Node 服务或停止 Ollama 后，API 就不能生成回答。

## 创建并使用 API Key

1. 在本机网页打开 **API 接入**，用首次启动时显示的 `API_ADMIN_PASSWORD` 验证。
2. 在 **API Key 管理**创建一把 Key。完整 Key 只在创建时显示一次；请复制到同一台电脑上运行的 API 客户端。
3. 在兼容 OpenAI 的客户端填写 Base URL `http://127.0.0.1:3000/v1` 和这把 Key；模型名使用页面显示的名称。

本机命令行测试示例：

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H 'Authorization: Bearer <API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5:3b","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

Windows 可在 PowerShell 中使用 `curl.exe`，或在兼容 OpenAI 的软件里填写相同地址和 Key。

## 安全和限制

- 此本机模式**不会**让其他电脑通过 LAN、GitHub Pages 或互联网访问 API；`127.0.0.1` 只指向本机。
- 不要把 `.env`、`.local-api.json` 或创建的 API Key 提交到 Git，也不要把 Key 放到公开网页前端。
- 第一次下载 Ollama 模型需要网络和磁盘空间；下载完成后，推理可由本机 Ollama 完成，不必调用云模型 API。
- 如果只打开 `https://tnt-tht.github.io/-first-/`，那仍是 GitHub Pages 的静态版，不会连接本机 API。必须打开 `http://127.0.0.1:3000`。
- 若要让其他电脑访问，需要改成局域网/公网部署并配置相应的访问控制与 HTTPS；本指南刻意不开放网络监听。
