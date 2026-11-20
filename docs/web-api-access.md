# 网页远程接入、API 调试、参数控制台与 Key 管理

本文说明服务端模式中三种不同凭据的用途，以及管理员如何在网页中调试 API、修改服务端参数并为普通用户创建可用于第三方软件的独立 API Key。

## 1. 访问凭据与权限

| 凭据/身份 | 用途 | 能做什么 |
|---|---|---|
| `APP_PASSWORD` | 网站共用访问口令 | 进入当前网站聊天界面；它不是个人账号，也不能用作 `/v1` API Key。 |
| `API_ADMIN_PASSWORD` | 服务端 API 管理员口令 | 解锁网页 API 调试台、全站 API 参数控制台以及 API Key 创建/撤销。 |
| API Key | 外部 API 凭据 | 调用方以 `Bearer` 或 `x-api-key` 调用本站 `/v1` 接口，可用于后端程序、OpenAI SDK、API 客户端等。 |
| 浏览器“管理员面板”中的本机管理员密钥 | 原有本机功能的 UI 权限 | 仅控制静态版本机模型导入/导出等界面，不是服务端身份验证，不能解锁 API 管理功能。 |

本站目前没有个人账号、注册、每人配额或用量账单。管理员可以按用户或应用分别创建、命名和撤销 API Key，再安全地把对应 Key 交给使用者。服务端只保存动态 Key 的摘要，不能再次显示明文；Key 标签只是管理员备注，不会自动验证真实使用者身份。

普通用户不需要 `API_ADMIN_PASSWORD`，也不需要先在网站登录，便可在自己使用的外部软件中使用管理员分发的 API Key。普通网站会话令牌不能访问 `/v1`。

## 2. 一次性服务端准备

在项目根目录复制 `.env.example` 为 `.env`，至少配置网站访问口令、独立的管理员口令、模型服务和 API 地址：

```dotenv
APP_PASSWORD=替换为网站访问口令
SESSION_SECRET=使用 openssl rand -hex 32 生成
API_ADMIN_PASSWORD=使用 openssl rand -base64 32 生成的独立口令
PUBLIC_BASE_URL=https://你的域名
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=qwen2.5:7b
```

`API_ADMIN_PASSWORD` 至少 16 个字符，且必须与 `APP_PASSWORD` 不同。不要将模型供应商的 `LLM_API_KEY`、外部调用用的 `API_KEYS` 或 API 管理口令提交到 Git。配置后运行：

```bash
npm run build:remote
npm start
```

也可使用 `docker compose up -d --build`。Docker Compose 将 `/app/data` 挂载到 `api-keys-data` 持久卷，其中保存网站生成的 API Key 摘要和管理员控制台设置。原生运行的默认路径分别为 `data/api-keys.json` 与 `data/api-settings.json`；可用 `API_KEYS_FILE`、`API_SETTINGS_FILE` 改路径。这些文件及目录不应提交到 Git或开放给其他用户读取。

## 3. 管理员网页控制台

1. 打开服务端模式网站。若网站配置了 `APP_PASSWORD`，进入聊天页仍需输入这个共用网站口令。
2. 点聊天页顶部 **API 管理**（它会直接打开到管理员控制面板位置；也可以从 **API 接入**打开同一窗口）。
3. 在 **管理员控制面板**输入 `API_ADMIN_PASSWORD`。成功后同一个短期管理员令牌会解锁网页 API 调试台、服务端参数控制台和 API Key 管理。
4. 关闭接入窗口或点 **锁定管理面板**会清除本页面内存中的管理员令牌和调试结果；令牌不会写入本地存储或可复制的请求 JSON。

服务端会逐个检查这些管理员 API 请求。静态版浏览器“管理员面板”只是客户端软门槛，不会被当作服务端管理员身份。

### 3.1 网页 API 调试台

管理员可填写可选 system 提示和 user 消息、temperature、`max_tokens`，选择流式 SSE 或完整 JSON，停止生成、查看/复制请求和响应。调试请求使用管理员短期令牌；普通网站登录会话不能调用 `/v1`。

### 3.2 服务端参数控制台

控制台可以修改并立即应用以下全站设置：

- `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY`；
- `SYSTEM_PROMPT`、默认 `LLM_TEMPERATURE`、`API_MAX_TOKENS`；
- 一组额外的 JSON 生成参数默认值，例如 `top_p`、`frequency_penalty`、`presence_penalty`、`stop`、`seed`，以及所选兼容服务商支持的其他参数。

这些是全站默认值；外部 API 请求可以覆盖额外生成参数，但 `API_MAX_TOKENS` 始终是服务端硬上限。模型路由、消息、流式模式、鉴权字段和输出上限不能放在额外参数 JSON 中覆盖。工具调用目前不支持。

设置保存到 `API_SETTINGS_FILE` 指定的 JSON 文件，并立即作用于当前进程；重启后自动加载。API 管理响应只返回“上游 API Key 是否已配置”，不会回显密钥。若要清理管理员设置并恢复进程启动时从 `.env` 读取的值，点 **恢复服务器 .env 配置**。

控制台写入的上游 API Key 会以明文保存在服务器端设置文件中（文件权限限制为 `0600`，目录为 `0700`），这与 `.env` 中的密钥属于同等敏感信息。请确保服务器账户与持久化 Docker 卷安全；不要把该文件打包或公开。

控制台不允许更改 `APP_PASSWORD`、`API_ADMIN_PASSWORD`、`SESSION_SECRET`、CORS 白名单、服务器监听地址或 Key 存储路径。这些安全/部署参数仍须通过服务器环境变量管理。

## 4. 普通用户：在其他软件中调用自己的 API Key

站点所有者或 API 管理员先在 **API Key 管理**里按用户或应用创建一把独立 Key，并通过安全渠道把完整 Key 交给对应使用者。Key 只在创建时显示一次，创建后无法恢复明文；泄露或遗失时撤销旧 Key 并创建新 Key。

API 根地址通常是 `https://你的域名/v1`；管理面板中的完整 Chat Completions 地址为 `https://你的域名/v1/chat/completions`。兼容 OpenAI 的软件中填写：

- **Base URL**：`https://你的域名/v1`
- **API Key**：管理员分发的那把 Key
- **Model**：管理面板显示的模型名称

命令行测试：

```bash
curl https://你的域名/v1/chat/completions \
  -H 'Authorization: Bearer <API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{"model":"你的模型名","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

也支持 `x-api-key: <API_KEY>`。接口包括 `GET /v1/models`、`GET /v1/models/:id` 与 `POST /v1/chat/completions`，支持 `stream: true` 的 SSE 和 `stream: false` 的 JSON 响应。请求支持 system/user/assistant 消息、`temperature`、`max_tokens`（兼容 `max_completion_tokens`），以及兼容服务商接受的额外 JSON 生成参数。最大输出受服务端 `API_MAX_TOKENS` 限制。纯文本 API 暂不支持图像/音频、多模态内容或工具调用。完整 Python / JavaScript 示例可见 `/api/docs`，机器可读信息见 `/api/info`。

### 网页与跨域调用

服务器默认 `CORS_ORIGINS=*`，便于在其他网站的浏览器中接入；生产环境可在 `.env` 将它收紧为可信来源列表，也可在 **API 管理 → 管理员参数控制台** 的「允许调用 API 的网站白名单」中直接增删（每行一个来源，保存后立即生效并持久化；恢复 .env 配置会回到启动值）。白名单内的公网网页调用本机/局域网 API 时，服务端还会对浏览器的本地网络预检返回许可；浏览器自身可能仍会弹出本地网络访问询问，需要用户允许。本机包默认白名单为本站与 `https://pd.geneluo.fun`（纸上谈兵）。CORS 不是密钥保护机制：任何写入网页 JavaScript 的 API Key 都可能被访问者查看。优先在你控制的后端或受信任的软件中保存 Key；不要把 Key 放进公开网页、浏览器扩展或移动端包。

## 5. API Key 管理

1. 配置 `API_ADMIN_PASSWORD` 并启动服务。
2. 打开顶部 **API 管理**，完成管理员验证。
3. 在 **API Key 管理**中为每位用户/应用输入可识别的标签并创建 Key。
4. 立即复制明文并安全地交给对应使用者；服务端不会再次展示。
5. 使用列表中的标签确认对象，需要停用时单独撤销对应 Key。

最多可创建 100 把动态 Key。环境变量 `API_KEYS` 配置的静态 Key 不显示在列表中，也不能在网页撤销；轮换它们需改服务器环境变量。API Key 管理不提供账号级配额、调用统计或计费。

## 6. 分享链接与 iframe

**API 接入**面板也提供网页分享工具：

- `?q=...`：打开网站时预填问题；`prompt`、`text`、`question` 是别名。
- 勾选“打开链接后自动发送”会附加 `send=1`。
- iframe 代码沿用预填问题和自动发送选项，并启用 `embed=1`。

问题会作为 URL 参数传播，可能进入浏览器历史、反向代理日志或分析系统。不要把账号凭据、私密数据或不适合自动提交的内容放入分享链接。iframe 是否能加载由服务端 `FRAME_ANCESTORS` 决定；跨域父页面通过 `postMessage` 填题则另由 `ALLOWED_PARENT_ORIGINS` 控制。

## 7. 故障排查

- **看不到服务端的“API 管理”按钮：**确认打开的是服务端构建（`npm run build:remote`）提供的页面，并重新构建/发布前端；默认纯静态 WebLLM 模式的“管理员面板”只管当前浏览器本机功能，没有服务端参数/Key 控制台。
- **网页调试台/参数控制台提示未启用：**检查 `API_ADMIN_PASSWORD` 是否配置且至少 16 位；修改环境变量后需要重启服务。
- **网页调试台提示 401：**管理员令牌可能已过期或窗口已锁定；重新输入 `API_ADMIN_PASSWORD`。
- **外部软件收到 401：**确认使用管理员分发的 API Key，而不是 `APP_PASSWORD`、网站登录令牌或 `API_ADMIN_PASSWORD`；检查 Key 是否已被撤销。
- **外部软件能连服务器，但浏览器跨域失败：**检查 `CORS_ORIGINS` 和反向代理是否透传 `OPTIONS`、`Authorization`、`Content-Type`、`x-api-key`。
- **API 返回模型未配置或上游连接失败：**检查控制台中的 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY`，并确认 Node.js 服务端能访问该地址。
- **上游模型拒绝某个额外参数：**不同兼容服务商支持的扩展不完全相同；删除该参数或按上游 API 文档调整。
- **容器重启后设置丢失：**确认 `api-keys-data` 持久卷仍挂载，并备份该卷；不要删除该卷或 `data/` 文件。
- **iframe 无法显示：**检查 `FRAME_ANCESTORS` 和反向代理的 CSP 配置；生产环境建议只允许可信父页面来源。
