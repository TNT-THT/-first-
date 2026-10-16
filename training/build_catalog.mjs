#!/usr/bin/env node
/**
 * 生成 training/catalog.json：网页下拉框中每个模型对应的训练信息。
 *
 * 事实字段来自仓库内的 web/public/config.json 与 @mlc-ai/web-llm 的内置记录
 * （model_lib、overrides、vram_required_MB、原始 MLC 地址）。
 * 原始 Hugging Face 仓库、对话模板、参数量与显存估计是整理者的判断，
 * 沙箱无法访问 huggingface.co 核对，因此每条都标记为 verified: false。
 *
 * 用法：node training/build_catalog.mjs            # 写入 training/catalog.json
 *       node training/build_catalog.mjs --check    # 与现有文件比较，不一致则退出码 1（供测试使用）
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');
const CATALOG_PATH = join(here, 'catalog.json');
const CONFIG_PATH = join(root, 'web', 'public', 'config.json');
const WEBLLM_PATH = join(root, 'node_modules', '@mlc-ai', 'web-llm', 'lib', 'index.js');

// 原始（未量化）Hugging Face 仓库与 MLC 对话模板。整理者的判断，未经核对。
// 对话模板名必须与 MLC-LLM 版本中的模板名一致，训练前请对照官方 mlc-chat-config.json。
const UPSTREAM = {
  'Qwen2.5-Coder-0.5B-Instruct-q4f16_1-MLC': ['Qwen/Qwen2.5-Coder-0.5B-Instruct', 'qwen2', 0.5],
  'Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC': ['Qwen/Qwen2.5-Coder-1.5B-Instruct', 'qwen2', 1.5],
  'Qwen2.5-Coder-3B-Instruct-q4f16_1-MLC': ['Qwen/Qwen2.5-Coder-3B-Instruct', 'qwen2', 3.1],
  'Qwen2.5-Coder-7B-Instruct-q4f16_1-MLC': ['Qwen/Qwen2.5-Coder-7B-Instruct', 'qwen2', 7.6],
  'Qwen2.5-3B-Instruct-q4f16_1-MLC': ['Qwen/Qwen2.5-3B-Instruct', 'qwen2', 3.1],
  'Qwen2.5-7B-Instruct-q4f16_1-MLC': ['Qwen/Qwen2.5-7B-Instruct', 'qwen2', 7.6],
  'Qwen3-1.7B-q4f16_1-MLC': ['Qwen/Qwen3-1.7B', 'chatml', 2.0],
  'Qwen3-4B-q4f16_1-MLC': ['Qwen/Qwen3-4B', 'chatml', 4.0],
  'Qwen3-8B-q4f16_1-MLC': ['Qwen/Qwen3-8B', 'chatml', 8.2],
  'Qwen3.5-4B-q4f16_1-MLC': ['Qwen/Qwen3.5-4B', 'chatml', 4.0],
  'Llama-3.2-1B-Instruct-q4f16_1-MLC': ['meta-llama/Llama-3.2-1B-Instruct', 'llama-3', 1.2],
  'Llama-3.2-3B-Instruct-q4f16_1-MLC': ['meta-llama/Llama-3.2-3B-Instruct', 'llama-3', 3.2],
  'Llama-3.1-8B-Instruct-q4f16_1-MLC': ['meta-llama/Llama-3.1-8B-Instruct', 'llama-3', 8.0],
  'Llama-2-13b-chat-hf-q4f16_1-MLC': ['meta-llama/Llama-2-13b-chat-hf', 'llama-2', 13.0],
  'gemma3-1b-it-q4f16_1-MLC': ['google/gemma-3-1b-it', 'gemma3_instruction', 1.0],
  'gemma-2-2b-it-q4f16_1-MLC': ['google/gemma-2-2b-it', 'gemma_instruction', 2.6],
  'gemma-2-9b-it-q4f16_1-MLC': ['google/gemma-2-9b-it', 'gemma_instruction', 9.2],
  'Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC': ['mistralai/Ministral-3-3B-Instruct-2512-BF16', 'mistral_default', 3.4],
  'Mistral-7B-Instruct-v0.3-q4f16_1-MLC': ['mistralai/Mistral-7B-Instruct-v0.3', 'mistral_default', 7.2],
  'Phi-4-mini-instruct-q4f16_1-MLC': ['microsoft/Phi-4-mini-instruct', 'phi-3', 3.8],
  'Phi-3.5-mini-instruct-q4f16_1-MLC': ['microsoft/Phi-3.5-mini-instruct', 'phi-3', 3.8],
  'SmolLM2-360M-Instruct-q4f16_1-MLC': ['HuggingFaceTB/SmolLM2-360M-Instruct', 'chatml', 0.36],
  'SmolLM2-1.7B-Instruct-q4f16_1-MLC': ['HuggingFaceTB/SmolLM2-1.7B-Instruct', 'chatml', 1.7],
  'OLMo-2-0425-1B-Instruct-q4f16_1-MLC': ['allenai/OLMo-2-0425-1B-Instruct', 'chatml', 1.5],
  'OLMo-2-1124-7B-Instruct-q4f16_1-MLC': ['allenai/OLMo-2-1124-7B-Instruct', 'chatml', 7.3],
  'TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC': ['TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'tinyllama_v1_0', 1.1],
  'stablelm-2-zephyr-1_6b-q4f16_1-MLC': ['stabilityai/stablelm-2-zephyr-1_6b', 'stablelm-2', 1.6],
  'Hermes-3-Llama-3.2-3B-q4f16_1-MLC': ['NousResearch/Hermes-3-Llama-3.2-3B', 'chatml', 3.2],
  'DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC': ['deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'llama-3', 8.0],
  'DeepSeek-R1-Distill-Qwen-7B-q4f16_1-MLC': ['deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'deepseek_r1_qwen', 7.6],
};

// 需要 Hugging Face 账号并同意许可协议（训练前需设置 HF_TOKEN）。
const GATED_PREFIXES = ['meta-llama/', 'google/gemma'];

// 下拉框标签格式：“名称（推理·小）” 或 “名称（非推理·极小·默认）”
export function parseLabel(label) {
  const inner = (label.match(/（([^）]*)）/) || [])[1] || '';
  const parts = inner.split('·');
  const reasoning = !parts[0].startsWith('非推理') && parts[0].startsWith('推理');
  const sizeClass = ['极小', '小', '中', '大'].find((s) => parts.includes(s)) || '';
  return { reasoning, sizeClass };
}

export function buildCatalog({ configPath = CONFIG_PATH, webllmPath = WEBLLM_PATH } = {}) {
  const config = JSON.parse(readFileSync(configPath, 'utf8'));
  return import(pathToFileURL(webllmPath).href).then(({ prebuiltAppConfig }) => {
    const byId = new Map(prebuiltAppConfig.model_list.map((r) => [r.model_id, r]));
    const entries = [];
    for (const item of config.webllm.models) {
      const rec = byId.get(item.id);
      if (!rec) throw new Error(`web-llm 中找不到记录：${item.id}`);
      const up = UPSTREAM[item.id];
      if (!up) throw new Error(`缺少原始仓库映射：${item.id}`);
      const [hfRepo, convTemplate, paramsB] = up;
      const { reasoning, sizeClass } = parseLabel(item.label);
      const quant = (item.id.match(/-(q\d+f\d+(?:_\d+)?)-MLC/) || [])[1] || null;
      const overrides = rec.overrides || {};
      entries.push({
        id: item.id,
        label: item.label,
        size_class: sizeClass,
        reasoning,
        hf_repo: hfRepo,
        hf_gated: GATED_PREFIXES.some((p) => hfRepo.startsWith(p)),
        params_b: paramsB,
        train_vram_gb_min: Math.round(paramsB * 2 + 4),
        quantization: quant,
        conv_template: convTemplate,
        context_window_size: overrides.context_window_size ?? null,
        overrides,
        vram_required_MB: rec.vram_required_MB ?? null,
        source_model: rec.model,
        model_lib_url: rec.model_lib,
        wasm_file: rec.model_lib.split('/').pop(),
        verified: false,
      });
    }
    return {
      note: '由 training/build_catalog.mjs 生成。事实字段来自 web-llm 记录；hf_repo、conv_template、params_b、train_vram_gb_min 为整理者的判断，未经核对（verified 均为 false）。',
      webllm_version: '0.2.86',
      models: entries,
    };
  });
}

async function main(argv) {
  const catalog = await buildCatalog();
  const text = JSON.stringify(catalog, null, 2) + '\n';
  if (argv.includes('--check')) {
    const current = readFileSync(CATALOG_PATH, 'utf8');
    if (current !== text) {
      console.error('training/catalog.json 与网页配置不一致，请运行：node training/build_catalog.mjs');
      process.exit(1);
    }
    console.log('catalog.json 已与网页配置同步');
    return;
  }
  writeFileSync(CATALOG_PATH, text);
  console.log(`已写入 ${CATALOG_PATH}（${catalog.models.length} 个模型）`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main(process.argv.slice(2)).catch((err) => {
    console.error(err.message || err);
    process.exit(1);
  });
}
