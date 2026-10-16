# 训练一站式工具（LoRA 微调 → 打包 → 导入网页）

所有训练前的准备都在 `training/` 目录里完成：生成配置、创建 Python 环境、检查 GPU、下载或导入基础模型、导入数据、训练、转换、打包。最终得到一个 `.ojmodel` 文件，在网页的“导入训练结果”中加入“新建AI”。

网页端不会调用这里的任何代码，也不会上传数据。

```
init → setup → doctor → import-data / import-model → run（fetch → data → train → convert → pack）→ 网页导入
```

所有命令都在**仓库根目录**执行，入口是 `python3 training/ojtrain.py`。

## 1. 第一次准备

```bash
python3 training/ojtrain.py init --name my-coder          # 生成 training/config.json（个人配置，不进仓库）
python3 training/ojtrain.py setup                         # 创建 training/.venv 并安装 PyTorch 等依赖
training/.venv/bin/python training/ojtrain.py doctor      # 检查环境
```

`setup` 会安装 PyTorch。请先确认显卡驱动和 CUDA 版本。如果需要特定 CUDA 版本的 PyTorch，可以先在 `training/.venv` 中按 pytorch.org 的说明手动安装对应版本，然后再运行 `setup`（pip 会跳过已满足的依赖）。

`doctor` 输出每一项的状态：

| 状态 | 含义 |
| --- | --- |
| `OK` | 通过 |
| `WARN` | 可以继续，但某些步骤会失败或很慢（例如没有 GPU、缺少 mlc_llm） |
| `FAIL` | 必须先解决（例如 Python 版本过低、配置文件有误） |

## 2. 导入或下载基础模型

二选一：

```bash
# 方式 A：使用本机已有的模型目录（不复制，只记录路径）
python3 training/ojtrain.py import-model /path/to/Qwen2.5-Coder-1.5B-Instruct

# 方式 B：不做任何操作，由 run 自动从 Hugging Face 下载到 training/models/
```

模型目录需要包含 `config.json`、`*.safetensors` 权重，以及 `tokenizer.json` 或 `tokenizer.model`。

下载私有或受限模型时，请在本机设置环境变量 `HF_TOKEN`。不要把 token 写进文件或发给别人。

## 3. 导入数据

支持 `.jsonl`、`.json`（数组）、`.csv`（带表头）。每条样本可以是以下格式之一：

```json
{"prompt": "题目或问题", "response": "回答"}
{"instruction": "...", "input": "可选补充", "output": "回答"}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

CSV 的表头可以是 `prompt,response`，或 `instruction,input,output`。

```bash
python3 training/ojtrain.py import-data 我的数据.csv
```

导入后文件会复制到 `training/data/`，并把配置中的 `data.raw` 指向它。

流程会自动：
- 检查角色顺序（对话必须以 assistant 回复结尾）。
- 丢弃空样本和超过 `data.max_chars` 的样本，并打印丢弃原因。
- 按 `data.seed` 随机切出验证集（比例由 `data.val_ratio` 决定，默认 5%）。验证集不参与训练，只用于计算验证 loss，帮你判断是否过拟合。

数据太少（少于 20 条）时不切分验证集。

## 4. 配置说明

配置文件是 `training/config.json`，由 `init` 从 `training/config.example.json` 生成。相对路径都相对于配置文件所在目录。

| 字段 | 说明 | 默认值 |
| --- | --- | --- |
| `name` | 本次训练的名称，决定工作目录 `training/work/<name>/` | `coder15-tuned` |
| `base.hf_repo` | 基础模型的 Hugging Face 仓库 | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| `base.local_dir` | 本机模型目录；设置后优先使用它 | `null` |
| `data.raw` | 原始数据文件 | `data/raw.jsonl` |
| `data.val_ratio` | 验证集比例（0 到 0.5） | `0.05` |
| `data.max_chars` | 单条样本的最大字符数，超过则丢弃 | `8000` |
| `data.seed` | 切分验证集用的随机种子 | `42` |
| `train.epochs` | 训练轮数 | `2` |
| `train.learning_rate` | 学习率 | `0.0002` |
| `train.lora_r` / `lora_alpha` / `lora_dropout` | LoRA 参数 | `16` / `32` / `0.05` |
| `train.target_modules` | 作用于哪些层 | q/k/v/o 投影 |
| `train.max_length` | 超过该 token 数的样本跳过 | `1024` |
| `train.batch_size` × `grad_accum` | 每步样本数与梯度累积，相乘得到有效 batch | `1` × `8` |
| `train.dtype` | `auto` / `bf16` / `fp16` / `fp32` | `auto` |
| `train.grad_checkpointing` | 省显存，速度变慢 | `true` |
| `mlc.quantization` / `conv_template` / `context_window_size` | MLC 转换参数 | `q4f16_1` / `qwen2` / `4096` |
| `mlc.model_lib_url` | 网页使用的 wasm 运行库地址（必须与 WebLLM 一致） | Qwen2-1.5B 的官方地址 |
| `mlc.wasm_path` / `wasm_url` | 本地 wasm 路径；缺失时从 `wasm_url` 下载 | `libs/...wasm` |
| `pack.model_id` | 模型 id（只含字母、数字、`._-`） | `coder15-tuned-v1` |
| `pack.label` | 网页“新建AI”中显示的名称 | `我的刷题AI v1` |
| `pack.vram_required_MB` | 显存提示 | `1630` |

修改配置后直接运行即可，不需要重新 `init`。

## 5. 一键运行

```bash
python3 training/ojtrain.py run                 # 执行全部步骤
python3 training/ojtrain.py status              # 查看每一步的状态
python3 training/ojtrain.py run --from train    # 从某一步继续
python3 training/ojtrain.py run --force         # 忽略已完成的记录，全部重新执行
```

每一步完成后都会记录到 `training/work/<name>/state.json`，中断后不会重复已完成的步骤。

| 步骤 | 做什么 | 输出 |
| --- | --- | --- |
| `doctor` | 环境检查（失败项会停止流程） | 检查报告 |
| `fetch` | 确认或下载基础模型 | `training/models/...` |
| `data` | 整理数据，切分训练集和验证集 | `work/<name>/data/train.jsonl`、`val.jsonl` |
| `train` | LoRA 微调，合并进基础模型 | `work/<name>/lora/`、`work/<name>/merged/` |
| `convert` | 用 mlc_llm 量化并转换为 WebLLM 格式 | `work/<name>/mlc/` |
| `pack` | 打包为 `.ojmodel` | `work/<name>/<model_id>.ojmodel` |

`train` 结束后，训练元数据写在 `work/<name>/lora/training_meta.json`，包括样本数、步数、训练 loss、验证 loss 和耗时。

### convert 步骤需要手动安装 mlc_llm

如果没有安装 `mlc_llm`，`convert` 会**暂停**（状态 `waiting`），并打印需要执行的命令。这不算失败，前面的步骤不会被重做。安装 MLC-LLM 后运行：

```bash
python3 training/ojtrain.py run --from convert
```

如果转换在别的机器上完成，把输出目录（包含 `tensor-cache.json`、`mlc-chat-config.json`、分词器和权重分片）完整复制到 `training/work/<name>/mlc/`，然后同样运行 `run --from convert`。流程会检测到结果并直接继续。

MLC-LLM 的版本需要与网页使用的 WebLLM 0.2.86 对应的运行库兼容。转换输出必须包含 `tensor-cache.json`、权重分片、分词器和 `mlc-chat-config.json`。**这一步本仓库没有验证过**，命令以 MLC 官方文档为准。

不要修改 `mlc-chat-config.json` 中与架构有关的字段（例如 `rope_theta`、`tie_word_embeddings`、层数）。这些值已经编译进 wasm，改了会导致输出乱码。

## 6. 导入网页

1. 打开网页，点聊天顶部的“更改权限”，输入管理员密钥，升级为管理员。
2. 点“导入训练结果”，选择 `work/<name>/<model_id>.ojmodel`。
3. 导入后，模型出现在下拉框的“新建AI”分组中，名称后带“（训练结果）”。

导入的模型与官方模型的缓存分开保存，不会覆盖官方模型，也不会上传云端。

## 7. 时间与显存（估算，未实测）

以 1.5B 模型、3000 条样本、每条约 300 token、2 个 epoch 为例（约 180 万训练 token）：RTX 4060 级显卡约 50 分钟，高端显卡约 20–25 分钟。模型加载约 1–2 分钟，合并保存约 2–5 分钟，MLC 转换约 5–10 分钟，打包约 1–2 分钟。

1.5B 模型建议 8 GB 以上显存（开启 `grad_checkpointing`）。3B 模型耗时约为 1.5B 的 2 倍，7B 约为 5 倍，且需要至少 24 GB 显存。

以上数字是按公开的 LoRA 吞吐数据（RTX 4060 上 Qwen2.5-1.5B 约 630 token/s）推算的估计值，不是在本仓库环境中实测的。

## 8. 测试

```bash
npm test                  # 网页与打包的测试，其中包含跨语言检查：打包结果能被网页端导入
npm run test:training     # 训练工具的 Python 测试；未安装 PyTorch 时训练冒烟测试会跳过
```

`tests/test_train_smoke.py` 使用随机初始化的极小模型（`tests/make_tiny_model.py`）在 CPU 上跑通“构造标签 → LoRA 训练 → 合并 → 重新加载”。它只验证代码路径，不代表任何真实模型的质量。

## 已验证与未验证

已验证（沙箱中，无 GPU）：
- 配置校验、路径解析、`init` / `import-data` / `import-model` 等命令。
- 数据导入（jsonl / json / csv）、角色校验、切分训练集与验证集。
- 环境检查的输出与退出码。
- 流水线的步骤顺序、断点续跑、`mlc_llm` 缺失时暂停、失败时停止。
- `pack_ojmodel.py` 的输出能被网页端 `importModelFile` 导入（跨语言测试）。
- LoRA 训练、合并、重新加载（极小随机模型，CPU）。

未验证：
- `setup` 在真实机器上安装 PyTorch 的过程。
- 从 Hugging Face 下载基础模型（沙箱无法访问 huggingface.co）。
- 真实 GPU 训练的速度、显存占用和效果。
- MLC 转换步骤（沙箱无法安装 mlc-llm）。
- 转换后的模型在浏览器中的加载和生成质量。

## 文件

| 文件 | 作用 |
| --- | --- |
| `ojtrain.py` | 一站式命令行入口 |
| `config.py` / `config.example.json` | 配置的读取、校验、路径解析与模板 |
| `doctor.py` | 环境检查 |
| `fetch_model.py` | 下载或导入基础模型，下载运行库并检查 wasm 文件头 |
| `prepare_dataset.py` | 导入、校验、切分数据 |
| `pipeline.py` | 步骤编排、状态记录、断点续跑 |
| `train_lora.py` | LoRA 微调，可选验证集与合并输出 |
| `pack_ojmodel.py` | 把 MLC 输出打包为 `.ojmodel`（只用标准库） |
| `requirements.txt` | 训练依赖（GPU 环境） |
| `examples/sample.jsonl` | 示例数据 |
| `tests/` | 单元测试、流水线测试、训练冒烟测试、极小模型生成器 |

个人文件不进入仓库（已在 `.gitignore` 中忽略）：`config.json`、`data/`、`models/`、`work/`、`libs/`、`.venv/`。
