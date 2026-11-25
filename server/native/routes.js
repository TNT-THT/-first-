/**
 * 本机大模型（llama.cpp）的 HTTP 接口：
 * - GET    /api/native/status            状态与模型目录（公开，不含本机路径）
 * - GET    /api/native/models/:id/quants 仓库可下载量化（需网站登录）
 * - POST   /api/native/chat              网页聊天转发（需网站登录，自动加载已下载模型）
 * - POST   /api/native/binary            准备 llama.cpp 运行库（需 API 管理员）
 * - POST   /api/native/models/:id/download 下载权重（需 API 管理员）
 * - POST   /api/native/models/:id/quants-preview 浏览器取到文件树后在本地分组（需网站登录）
 * - POST   /api/native/models/:id/browser-download/* 浏览器取数、软件落盘的下载通道（需 API 管理员）
 * - DELETE /api/native/models/:id        删除权重（需 API 管理员）
 * - POST   /api/native/load | /unload    加载/卸载模型（需 API 管理员）
 *
 * 另导出 streamNativeToHttp / nativeCompletionJson 供 /v1 与 /api/chat 复用。
 */
import { randomUUID } from 'node:crypto';
import { describeQuant, estimateRamGB } from './catalog.js';
import { parseHfRepoInput } from './formats.js';
import { createThinkFolder, foldMessageReasoning } from './llama.js';
import { validateMessages } from '../chat.js';

/** 量化档位的展示视图：名字之外还要有类型、是否可直接使用，别让用户猜。 */
function quantView(g) {
  return {
    quant: g.quant,
    files: g.files.length,
    totalBytes: g.totalBytes,
    sizeGB: Math.round((g.totalBytes / 1e9) * 10) / 10,
    ramGB: estimateRamGB(g.totalBytes),
    ...describeQuant(g.quant),
  };
}

/** 浏览器通道回传的搜索结果清洗：只留判定需要的字段，非法仓库丢弃。 */
function cleanSearchResults(list) {
  const out = [];
  for (const r of Array.isArray(list) ? list.slice(0, 50) : []) {
    const repo = parseHfRepoInput(r?.repo || r?.id);
    if (!repo) continue;
    out.push({
      repo,
      name: typeof r.name === 'string' ? r.name : repo.split('/').pop(),
      author: typeof r.author === 'string' ? r.author : repo.split('/')[0],
      downloads: Math.max(0, Number(r.downloads) || 0),
      likes: Math.max(0, Number(r.likes) || 0),
      pipeline: typeof r.pipeline === 'string' ? r.pipeline : '',
      library: typeof r.library === 'string' ? r.library : '',
      tags: Array.isArray(r.tags) ? r.tags.filter((t) => typeof t === 'string').slice(0, 30) : [],
      gated: !!r.gated,
      updatedAt: typeof r.updatedAt === 'string' ? r.updatedAt : '',
    });
  }
  return out;
}

/** 仓库详情的展示视图（服务端直连与浏览器通道共用，保证两条路看到的一样）。 */
function infoView(info) {
  return {
    repo: info.repo,
    name: info.name,
    downloads: info.downloads,
    likes: info.likes,
    pipeline: info.pipeline,
    formats: info.formats,
    variants: info.variants,
    route: info.route,
    totalGB: Math.round((info.totalBytes / 1e9) * 10) / 10,
    files: info.files.slice(0, 60).map((f) => ({ path: f.path, size: f.size })),
    quants: (info.quants || []).map((q) => ({
      quant: q.quant,
      totalBytes: q.totalBytes,
      sizeGB: Math.round((q.totalBytes / 1e9) * 10) / 10,
      typeLabel: q.typeLabel,
      bitsPerWeight: q.bitsPerWeight,
      directUse: q.directUse,
      useNote: q.useNote,
      qualityNote: q.qualityNote,
    })),
  };
}

function errJson(res, err, fallback = '本机大模型服务出错') {
  const status = Number.isInteger(err?.status) ? err.status : 500;
  return res.status(status).json({ error: err?.message || fallback });
}

/**
 * 把 llama-server 的流式响应转成 OpenAI 风格 SSE 写给客户端。
 * reasoning_content 会折叠为 <think>…</think>，与网页端思考展示逻辑一致。
 * @returns {Promise<{content: string, usage: object|null}>}
 */
export async function streamNativeToHttp(engine, { model, messages, temperature, maxTokens, signal, res, onText, thinking }) {
  const upstream = await engine.chatCompletion({ model, messages, stream: true, temperature, maxTokens, signal, thinking });
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
  const folder = createThinkFolder();
  let content = '';
  let usage = null;
  let upstreamError = '';
  let sawFinish = false;
  let buffer = '';
  const decoder = new TextDecoder();
  for await (const chunk of upstream.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line.startsWith('data:')) continue;
      const payload = line.slice(5).trim();
      if (payload === '[DONE]') continue;
      let data;
      try {
        data = JSON.parse(payload);
      } catch {
        continue;
      }
      if (data.usage) usage = data.usage;
      // llama.cpp 在生成中途失败时会在流里塞一条 {"error":…}（HTTP 仍是 200）：
      // 以前这里直接忽略，网页只看到回答流莫名中断、什么原因都没有。
      if (data.error) {
        upstreamError = String(data.error.message || data.error || '生成失败');
        continue;
      }
      const delta = data.choices?.[0]?.delta || {};
      const text = folder.push(delta);
      if (text) {
        content += text;
        onText?.(text);
        emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] });
      }
      const finish = data.choices?.[0]?.finish_reason;
      if (finish) {
        sawFinish = true;
        emit({ id, object: 'chat.completion.chunk', created, model, choices: [{ index: 0, delta: {}, finish_reason: finish }] });
      }
    }
  }
  if (upstreamError) {
    // 明确告诉客户端“是上游生成失败”，而不是让它把空/半截回答当成正常结束
    try {
      res.write(`data: ${JSON.stringify({ error: { message: `llama.cpp 生成失败：${upstreamError}`, type: 'server_error', code: null } })}\n\n`);
    } catch { /* 连接已断开 */ }
  } else if (!content && !sawFinish) {
    // 上游既没内容也没结束标记（版本不兼容/流被截断）：同样给出可执行的原因
    try {
      res.write(`data: ${JSON.stringify({ error: { message: 'llama.cpp 没有返回任何内容：请到“本机大模型”面板查看日志尾部，确认模型已正确加载。', type: 'server_error', code: null } })}\n\n`);
    } catch { /* 连接已断开 */ }
  }
  res.write('data: [DONE]\n\n');
  res.end();
  return { content, usage, error: upstreamError };
}

/** 非流式：返回 OpenAI 风格的完整响应对象。 */
export async function nativeCompletionJson(engine, { model, messages, temperature, maxTokens, signal, thinking }) {
  const upstream = await engine.chatCompletion({ model, messages, stream: false, temperature, maxTokens, signal, thinking });
  const data = await upstream.json();
  const choice = data.choices?.[0] || {};
  const message = foldMessageReasoning(choice.message || { role: 'assistant', content: '' });
  return {
    id: data.id || `chatcmpl-${randomUUID().replaceAll('-', '')}`,
    object: 'chat.completion',
    created: data.created || Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message, finish_reason: choice.finish_reason || 'stop' }],
    usage: data.usage || null,
    content: typeof message.content === 'string' ? message.content : '',
  };
}

export function registerNativeRoutes(app, { engine, requireSession, requireAdmin }) {
  app.get('/api/native/status', async (_req, res) => {
    try {
      return res.json(await engine.status());
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.get('/api/native/models/:id/quants', requireSession, async (req, res) => {
    try {
      const model = engine.resolveModel(req.params.id);
      if (!model) return res.status(404).json({ error: `未知模型：${req.params.id}` });
      const grouped = await engine.availableQuants(req.params.id);
      return res.json({ id: model.id, quants: grouped.map(quantView) });
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/chat', requireSession, async (req, res) => {
    const ac = new AbortController();
    res.on('close', () => {
      if (!res.writableFinished) ac.abort();
    });
    try {
      const model = typeof req.body?.model === 'string' ? req.body.model : '';
      if (!engine.hasModel(model)) return res.status(404).json({ error: `未知模型：${model || '(未指定)'}` });
      const messages = validateMessages(req.body?.messages, { allowSystem: true });
      const temperature = Number.isFinite(req.body?.temperature) ? req.body.temperature : undefined;
      const maxTokens = Number.isInteger(req.body?.max_tokens) ? req.body.max_tokens : undefined;
      if (req.body?.stream === false) {
        const out = await nativeCompletionJson(engine, { model, messages, temperature, maxTokens, signal: ac.signal });
        const { content: _content, ...json } = out;
        return res.json(json);
      }
      await streamNativeToHttp(engine, { model, messages, temperature, maxTokens, signal: ac.signal, res });
      return undefined;
    } catch (err) {
      if (res.headersSent) {
        try {
          res.write(`data: ${JSON.stringify({ error: { message: err.message || '生成失败', type: 'server_error', code: null } })}\n\n`);
        } catch { /* 连接已断开 */ }
        return res.end();
      }
      return errJson(res, err);
    }
  });

  // 准备 / 更新 llama.cpp 运行库。
  // body: { force?: true, tag?: 'b8421' }
  //   force —— 已有二进制也重新下载。没有这个开关时“准备运行库”在已装旧版的情况下
  //            是空操作，用户永远升不到支持新模型架构的版本。
  //   tag   —— 锁定到指定构建号。新版在某个系统上跑不起来时，这是唯一的出路。
  app.post('/api/native/binary', requireAdmin, async (req, res) => {
    try {
      const force = req.body?.force === true || req.body?.force === 'true';
      const tag = typeof req.body?.tag === 'string' ? req.body.tag.trim() : '';
      const got = await engine.ensureBinary({ force, tag });
      return res.json({
        ok: true,
        source: got.source || 'downloaded',
        version: got.version || '',
        job: engine.binaryJob || null,
      });
    } catch (err) {
      return errJson(res, err);
    }
  });

  // 运行库健康度：这个 llama.cpp 在本机能不能跑、版本够不够新。
  // 界面据此在用户点“加载”之前就提示，而不是等他等半天看到一个崩溃码。
  app.get('/api/native/binary/health', requireAdmin, async (req, res) => {
    try {
      const forArch = typeof req.query?.arch === 'string' ? req.query.arch : '';
      return res.json(await engine.checkBinaryHealth({ forArch }));
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/models/:id/download', requireAdmin, async (req, res) => {
    try {
      const job = await engine.download(req.params.id, req.body?.quant);
      return res.status(202).json({ id: job.id, quant: job.quant, state: job.state });
    } catch (err) {
      return errJson(res, err);
    }
  });

  // 浏览器通道（与普通网页版同一条网络路）：量化清单由浏览器取文件树、
  // 软件只负责分组计算，服务端直连 HF 失败时不再一上来就死。
  app.post('/api/native/models/:id/quants-preview', requireSession, async (req, res) => {
    try {
      const model = engine.resolveModel(req.params.id);
      if (!model) return res.status(404).json({ error: `未知模型：${req.params.id}` });
      const files = Array.isArray(req.body?.files) ? req.body.files : [];
      const grouped = engine.groupQuantsFromFiles(req.params.id, files);
      return res.json({ id: model.id, quants: grouped.map(quantView) });
    } catch (err) {
      return errJson(res, err);
    }
  });

  // 浏览器通道下载：浏览器把权重字节分块传给软件，软件按偏移 append 到
  // .part 并在收尾时核对大小转正。断网/关页面后再点下载按偏移续传。
  app.post('/api/native/models/:id/browser-download/prepare', requireAdmin, async (req, res) => {
    try {
      const out = await engine.prepareBrowserDownload(req.params.id, req.body?.quant, req.body?.files);
      return res.json(out);
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/models/:id/browser-download/chunk', requireAdmin, async (req, res) => {
    try {
      const file = decodeURIComponent(String(req.get('x-bd-file') || ''));
      const quant = String(req.get('x-bd-quant') || '');
      const expected = Number(req.get('x-bd-offset'));
      const parts = [];
      let total = 0;
      for await (const chunk of req) {
        total += chunk.length;
        if (total > 64 * 1024 * 1024) {
          return res.status(413).json({ error: '单块过大，请更新软件后重试。' });
        }
        parts.push(chunk);
      }
      const out = await engine.receiveBrowserChunk(
        req.params.id,
        quant,
        file,
        Number.isFinite(expected) ? expected : undefined,
        Buffer.concat(parts),
      );
      return res.json(out);
    } catch (err) {
      if (Number.isFinite(err?.received)) {
        return res.status(Number.isInteger(err.status) ? err.status : 409)
          .json({ error: err.message, received: err.received });
      }
      return errJson(res, err);
    }
  });

  app.post('/api/native/models/:id/browser-download/finish', requireAdmin, async (req, res) => {
    try {
      const out = await engine.finishBrowserDownload(req.params.id, req.body?.quant);
      return res.json(out);
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/models/:id/browser-download/fail', requireAdmin, async (req, res) => {
    try {
      return res.json(engine.failBrowserDownload(req.params.id, req.body?.quant, req.body?.error));
    } catch (err) {
      return errJson(res, err);
    }
  });

  // 本地导入（浏览器用自带下载器把文件下到电脑 → 在软件里导入，全程不碰 HF 网络）：
  // begin 登记清单 → file 按偏移收片（可续传）→ finish 自动识别格式落地登记。
  app.post('/api/native/import/begin', requireAdmin, async (req, res) => {
    try {
      const out = await engine.beginImport(req.body?.importId, req.body?.files);
      return res.json(out);
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.get('/api/native/import/:importId/status', requireAdmin, async (req, res) => {
    try {
      return res.json(await engine.importSessionStatus(req.params.importId));
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/import/:importId/file', requireAdmin, async (req, res) => {
    try {
      const relPath = decodeURIComponent(String(req.get('x-import-path') || ''));
      const expected = Number(req.get('x-import-offset'));
      const parts = [];
      let total = 0;
      for await (const chunk of req) {
        total += chunk.length;
        if (total > 64 * 1024 * 1024) {
          return res.status(413).json({ error: '单片过大，请更新软件后重试。' });
        }
        parts.push(chunk);
      }
      const out = await engine.writeImportSlice(
        req.params.importId,
        relPath,
        Number.isFinite(expected) ? expected : undefined,
        Buffer.concat(parts),
      );
      return res.json(out);
    } catch (err) {
      if (Number.isFinite(err?.received)) {
        return res.status(Number.isInteger(err.status) ? err.status : 409)
          .json({ error: err.message, received: err.received });
      }
      return errJson(res, err);
    }
  });

  app.post('/api/native/import/:importId/finish', requireAdmin, async (req, res) => {
    try {
      const out = await engine.finishImport(req.params.importId, { name: req.body?.name });
      return res.status(201).json(out);
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.delete('/api/native/import/:importId', requireAdmin, async (req, res) => {
    try {
      return res.json(await engine.cancelImport(req.params.importId));
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.delete('/api/native/models/:id', requireAdmin, async (req, res) => {
    try {
      if (!engine.hasModel(req.params.id)) return res.status(404).json({ error: `未知模型：${req.params.id}` });
      return res.json(await engine.deleteModel(req.params.id, req.body?.quant));
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/load', requireAdmin, async (req, res) => {
    try {
      const { id, quant, contextTokens, gpuLayers } = req.body || {};
      const status = await engine.load(id, { quant, contextTokens, gpuLayers });
      return res.json(status);
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/unload', requireAdmin, async (_req, res) => {
    try {
      return res.json(await engine.unload());
    } catch (err) {
      return errJson(res, err);
    }
  });

  // ---- Hugging Face 桥接：搜索 → 拉取 → 点击使用（小白模式） ----
  // 搜索与详情只需网站登录（只读 Hub 公开信息）；下载/转换/云端登记/Token 需管理员。

  app.get('/api/native/hf/search', requireSession, async (req, res) => {
    try {
      const q = typeof req.query.q === 'string' ? req.query.q : '';
      const repo = parseHfRepoInput(q);
      if (repo) {
        // 粘贴的是仓库链接/owner-repo：直接给详情，不用再搜
        const info = await engine.hfInfo(repo);
        return res.json({ direct: repo, results: [{ repo, name: info.name, author: info.author, downloads: info.downloads, likes: info.likes, formats: info.formats }] });
      }
      const results = await engine.searchHf(q, { limit: req.query.limit, sort: req.query.sort });
      return res.json({ results });
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.get('/api/native/hf/model', requireSession, async (req, res) => {
    try {
      const repo = parseHfRepoInput(req.query.repo);
      if (!repo) return res.status(400).json({ error: '请提供有效的 Hugging Face 仓库（owner/repo 或完整链接）。' });
      const info = await engine.hfInfo(repo);
      return res.json(infoView(info));
    } catch (err) {
      return errJson(res, err);
    }
  });

  // 浏览器通道（与普通网页版同一条网络路）：搜索与详情由浏览器从 Hub 取数，
  // 本地服务只做格式识别、路线判定、登记等本地计算，不再要求软件直连 HF。
  // 浏览器取数失败时前端才回退上面的服务端直连接口。
  app.post('/api/native/hf/search-meta', requireSession, async (req, res) => {
    try {
      const results = engine.searchFromMeta(cleanSearchResults(req.body?.results));
      return res.json({ results });
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/hf/model-meta', requireSession, async (req, res) => {
    try {
      const repo = parseHfRepoInput(req.body?.repo);
      if (!repo) return res.status(400).json({ error: '请提供有效的 Hugging Face 仓库（owner/repo 或完整链接）。' });
      const info = engine.hfInfoFromMeta(repo, req.body?.meta || {}, req.body?.files);
      return res.json(infoView(info));
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/hf/use', requireAdmin, async (req, res) => {
    try {
      const repo = parseHfRepoInput(req.body?.repo);
      if (!repo) return res.status(400).json({ error: '请提供有效的 Hugging Face 仓库（owner/repo 或完整链接）。' });
      const mode = ['auto', 'local', 'python', 'cloud'].includes(req.body?.mode) ? req.body.mode : 'auto';
      // 浏览器已取到详情时优先用本地判定，不再为判定而直连 HF
      const pre = req.body?.info && typeof req.body.info === 'object' ? req.body.info : null;
      const info = pre ? engine.hfInfoFromMeta(repo, pre.meta || {}, pre.files) : null;
      const out = await engine.useRepo(repo, mode, { info, viaBrowser: !!info || req.body?.viaBrowser === true });
      return res.status(202).json(out);
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/hf/python', requireAdmin, async (req, res) => {
    try {
      const id = typeof req.body?.id === 'string' ? req.body.id : '';
      const job = await engine.loadPython(id);
      return res.status(202).json({ id: job.id, state: job.state });
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/hf/prepare-runner', requireAdmin, async (_req, res) => {
    try {
      const job = await engine.prepareRunnerTool();
      return res.status(202).json({ state: job.state });
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/hf/convert', requireAdmin, async (req, res) => {
    try {
      const id = typeof req.body?.id === 'string' ? req.body.id : '';
      const job = await engine.convert(id);
      return res.status(202).json({ id: job.id, state: job.state });
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/hf/prepare-converter', requireAdmin, async (_req, res) => {
    try {
      const job = await engine.prepareConverterTool();
      return res.status(202).json({ state: job.state });
    } catch (err) {
      return errJson(res, err);
    }
  });

  app.post('/api/native/hf/token', requireAdmin, async (req, res) => {
    try {
      if (typeof req.body?.token !== 'string') return res.status(400).json({ error: 'token 必须是字符串（留空表示清除）。' });
      return res.json(await engine.setHfToken(req.body.token));
    } catch (err) {
      return errJson(res, err);
    }
  });
}
