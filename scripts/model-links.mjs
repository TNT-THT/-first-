#!/usr/bin/env node
/**
 * 生成 docs/model-downloads.md：网页下拉框中每个模型的权重下载地址清单。
 *
 * 地址以 @mlc-ai/web-llm 的预编译记录与 web/public/config.json 的 extraRecords 为准，
 * 与网页运行时真正访问的地址一致。正常使用时网页会自动下载并缓存这些模型，
 * 这份清单供需要手动下载、做镜像或核对来源的用户查阅。
 *
 * 用法：node scripts/model-links.mjs
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const cfg = JSON.parse(readFileSync(join(root, 'web', 'public', 'config.json'), 'utf8'));
const { prebuiltAppConfig } = await import(join(root, 'node_modules', '@mlc-ai', 'web-llm', 'lib', 'index.js'));

const byId = new Map(prebuiltAppConfig.model_list.map((r) => [r.model_id, r]));
for (const extra of cfg.webllm.extraRecords || []) byId.set(extra.model_id, extra);

const lines = [
  '# 模型下载链接',
  '',
  '网页在首次选用某个模型时，会自动从下表地址下载权重并缓存在浏览器中（断网后可离线使用），**平时不需要手动下载**。',
  '这份清单供以下场景查阅：手动预下载/校验、把权重镜像到自己的 Hugging Face 仓库（配合 `config.json` 的 `webllm.mirrorOwner`）、核对模型来源。',
  '',
  '- 权重仓库：模型权重、tokenizer 与 `mlc-chat-config.json` 所在仓库，WebLLM 运行时从这里下载；',
  '- 运行库：WebLLM 的 WebGPU 运行库（.wasm），同一架构的模型共用；',
  '- 显存为 WebLLM 记录的 `vram_required_MB`，与网页下拉标签中的数值一致（估计值）。',
  '',
  '| 网页下拉名称 | 模型 ID | 权重仓库（下载地址） | 运行库 | 显存（GB） |',
  '| --- | --- | --- | --- | --- |',
];
for (const m of cfg.webllm.models) {
  const rec = byId.get(m.id);
  if (!rec) throw new Error(`找不到模型记录：${m.id}`);
  const repo = rec.model.replace(/\/?$/, '/');
  const vram = rec.vram_required_MB ? (rec.vram_required_MB / 1024).toFixed(1) : '—';
  lines.push(`| ${m.label} | \`${m.id}\` | ${repo} | ${rec.model_lib} | ${vram} |`);
}
lines.push(
  '',
  '说明：第三方“去审查”模型的权重仓库为其作者的 Hugging Face 仓库（见表中地址），运行库仍为 MLC 官方运行库。',
  '本文件由 `node scripts/model-links.mjs` 按当前模型清单生成，模型增删后请重新生成。',
  '',
);
writeFileSync(join(root, 'docs', 'model-downloads.md'), `${lines.join('\n')}\n`);
console.log(`已生成 docs/model-downloads.md（${cfg.webllm.models.length} 个模型）`);
