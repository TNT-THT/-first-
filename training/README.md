# 训练运行时（LoRA 微调 → 打包 → 导入“新建AI”）

浏览器里的 WebLLM 只能做推理，无法微调。所以训练放在一个**独立的运行时**里完成：
它运行在有 NVIDIA GPU 的电脑（或云 GPU）上，训练结果打包成 `.ojmodel` 文件，再由管理员在网页里通过“**导入训练结果**”加入“新建AI”分组。

```
prepare_dataset.py  →  train_lora.py  →  (MLC 转换)  →  pack_ojmodel.py  →  网页：导入训练结果
  整理数据             LoRA 微调+合并       量化为 WebLLM 格式     打包为 .ojmodel       新建AI
```

这套流程与网页互不依赖：网页端不会调用训练代码，也不会上传任何数据。

## 为什么是这个方案

- 训练需要 GPU 和 PyTorch，不适合放进静态网页。
- 只训练 **LoRA 适配器**（参数量很少），再合并进基础模型，显存和时间都远小于全量微调。
- 合并后的模型**仍是同一架构**，所以可以复用官方 WebLLM 的 wasm 运行库，不需要重新编译 wasm。
  这一点很关键：只要不改模型结构（层数、隐藏维度、RoPE 参数等），就只需要替换权重。

## 环境要求

| 项目 | 说明 |
| --- | --- |
| GPU | NVIDIA，显存建议 ≥ 8 GB（1.5B 模型，并开启 `--grad-checkpointing`）；3B 及以上需要更大显存 |
| Python | 3.10+ |
| 依赖 | `pip install -r training/requirements.txt`（PyTorch 请按 pytorch.org 的说明选择 CUDA 版本） |
| 基础模型 | 必须是 WebLLM 已支持的架构，例如 `Qwen/Qwen2.5-Coder-1.5B-Instruct`（对应 `mlc-ai/Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC`） |

> 显存需求是估算值，沙箱中没有 GPU，未经实际验证。

## 第 1 步：整理数据

每行一个 JSON，支持三种格式（可混合）：

```json
{"prompt": "题目或问题", "response": "回答"}
{"instruction": "...", "input": "可选补充", "output": "回答"}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

```bash
python3 training/prepare_dataset.py --input raw.jsonl --output train.jsonl
```

脚本会检查角色顺序（以 assistant 回复结尾）、去掉空样本、丢弃超长样本，并打印丢弃原因。
示例见 `training/examples/sample.jsonl`。

> 训练数据的版权和隐私由你自己负责。不要使用你没有权利使用的内容。

## 第 2 步：LoRA 微调并合并

```bash
python3 training/train_lora.py \
  --base Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --data train.jsonl \
  --output-dir runs/coder15-lora \
  --merged-dir runs/coder15-merged \
  --epochs 2 --learning-rate 2e-4 --lora-r 16 --lora-alpha 32 \
  --grad-checkpointing
```

- 只对 assistant 的回复计算损失，问题部分不参与训练。
- `--output-dir` 保存 LoRA 适配器和 `training_meta.json`（记录样本数、步数、loss、参数）。
- `--merged-dir` 保存合并后的完整 HF 模型，供下一步转换。

## 第 3 步：转换为 WebLLM（MLC）格式

这一步使用 MLC-LLM 工具链，需要单独安装，并且版本要与网页使用的 WebLLM 0.2.86 对应的运行库兼容。
**本仓库没有验证这一步**（沙箱无法访问 mlc.ai），命令以 MLC 官方文档为准：

```bash
# 量化权重（q4f16_1 与网页中其他模型一致）
mlc_llm convert_weight runs/coder15-merged --quantization q4f16_1 -o mlc/weights
# 生成 mlc-chat-config.json；上下文长度要与官方 wasm 的编译值一致（这里是 4096）
mlc_llm gen_config runs/coder15-merged --quantization q4f16_1 \
  --conv-template qwen2 --context-window-size 4096 -o mlc/weights
```

输出目录必须包含：`mlc-chat-config.json`、`tensor-cache.json`、`params_shard_*.bin`，以及 `tokenizer.json` 或 `tokenizer.model`。
WebLLM 0.2.86 使用的索引文件名是 `tensor-cache.json`；如果你的 MLC 版本输出了不同的文件名，需要核对内容后再改名。

**注意**：不要修改 `mlc-chat-config.json` 中与架构有关的字段（例如 `rope_theta`、`tie_word_embeddings`、层数）。
这些值已经编译进 wasm，改了会导致输出乱码（见项目 README 中 R1 1.5B 的说明）。

## 第 4 步：打包为 `.ojmodel`

官方 wasm 地址可以在 `web/public/config.json` 对应模型的 WebLLM 记录里找到（即 `model_lib`）。
以 Qwen2.5-Coder-1.5B 为例，它使用的是 `Qwen2-1.5B-Instruct-q4f16_1_cs1k-webgpu.wasm`。

```bash
cat > record.json <<'JSON'
{"overrides": {"context_window_size": 4096}, "vram_required_MB": 1630}
JSON

python3 training/pack_ojmodel.py \
  --dir mlc/weights \
  --wasm libs/Qwen2-1.5B-Instruct-q4f16_1_cs1k-webgpu.wasm \
  --model-lib-url https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Qwen2-1.5B-Instruct-q4f16_1_cs1k-webgpu.wasm \
  --model-id coder15-tuned-v1 \
  --label "我的刷题AI v1" \
  --record-json record.json \
  --out coder15-tuned-v1.ojmodel
```

- `--model-lib-url` 必须与 WebLLM 实际使用的地址一致，否则导入后找不到运行库。
- `--model-id` 只能包含字母、数字、`.`、`_`、`-`。
- 打包只读取文件，不修改输入目录。

## 第 5 步：在网页中导入

1. 打开网页，点击聊天顶部的“更改权限”，输入管理员密钥。
2. 点“导入训练结果”，选择 `.ojmodel` 文件。
3. 导入后，模型出现在下拉框的“新建AI”分组中，名称后带“（训练结果）”。

导入的模型与官方模型的缓存分开保存（使用本地虚拟地址），不会覆盖官方模型，也不会上传云端。

## 测试

```bash
# 数据整理与打包（只需要 Python 标准库）
python3 -m unittest discover -s training/tests -p "test_*.py"
# 训练冒烟测试需要 torch / transformers / peft；未安装时自动跳过
# 跨语言检查（打包结果能被网页端导入）
npm test
```

训练冒烟测试使用 `tests/make_tiny_model.py` 生成的随机极小模型，在 CPU 上跑通“构造标签 → LoRA 训练 → 合并 → 重新加载”。
它只验证代码路径，**不代表任何真实模型的质量**。

## 已验证与未验证

已验证（在沙箱中，CPU）：
- 数据整理的各种输入格式与丢弃规则（单元测试）。
- `pack_ojmodel.py` 的文件布局、偏移、错误处理，并且输出能被网页端 `importModelFile` 导入（`test/training-pack.test.js`）。
- `train_lora.py` 在极小随机模型上跑通：只对 assistant 回复计算损失、LoRA 训练、合并、保存、重新加载。

未验证：
- 真实 GPU 训练（显存占用、速度、效果）。
- MLC 转换步骤（需要 mlc-llm，沙箱无法安装）。
- 转换后的模型在浏览器中的加载与生成质量。
- 训练结果在 iframe（嵌入）模式下的导入。

## 文件

| 文件 | 作用 |
| --- | --- |
| `prepare_dataset.py` | 把原始问答整理为 `messages` 格式 |
| `train_lora.py` | LoRA 微调，可选合并输出完整模型 |
| `pack_ojmodel.py` | 把 MLC 输出目录打包为 `.ojmodel`（只用标准库） |
| `requirements.txt` | 训练依赖（GPU 环境） |
| `examples/sample.jsonl` | 示例数据 |
| `tests/` | Python 单元测试、训练冒烟测试、极小模型生成器 |
