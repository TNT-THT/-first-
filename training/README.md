# 训练一站式工具（LoRA 微调 → 打包 → 导入网页）

在自己的机器上，用 GPU 对网页下拉框里的模型做 LoRA 微调，打包成 `.ojmodel`，再导入网页的“新建AI”。训练前的准备都在 `training/` 中完成：选择模型、创建 Python 环境、检查环境、下载或导入基础模型、导入数据、训练、转换、打包。

网页端不会调用这里的任何代码，也不会上传数据。

```
models → init → setup → doctor → import-model / import-data → run（fetch → data → train → convert → pack）→ 网页“导入训练结果”
```

所有命令都在**仓库根目录**执行，入口是 `python3 training/ojtrain.py`。

## 0. 能训练哪些模型

| 类别 | 怎么选 | 运行库 | 备注 |
| --- | --- | --- | --- |
| 网页下拉框中的 29 个模型 | `init --model <id>` | 沿用官方运行库，**不需要编译** | 清单见 `catalog.json`，`ojtrain.py models` 查看 |
| 仅训练目录的超大模型（Qwen2.5 32B、Yi 1.5 34B） | `init --hf-repo <仓库> --conv-template <模板名> --context-window-size 4096` | 在本机用 `mlc_llm compile` 编译 | 不在网页下拉框中；`init --model` 会拒绝并给出上述命令；显存约 19–20 GB（估算），训练显存需 70 GB 以上 |
| 其他 Hugging Face 模型 | `init --hf-repo 组织/模型 --conv-template <模板名>` | 在本机用 `mlc_llm compile` 编译 | 需要 MLC-LLM 支持该架构，且 transformers 能加载 |

**“所有模型”的含义**：下拉框中的 29 个模型都使用同一套流程，因为训练后的权重与官方模型的架构、量化、形状完全相同，所以可以复用同一个 wasm 运行库。清单之外的模型取决于两件事：MLC-LLM 是否支持该架构，以及 transformers 能否加载它。并不是每个 Hugging Face 模型都能用。

**验证范围**：流程对下拉框中的 29 个模型都已接好，但沙箱中只用极小随机模型验证了代码路径（多种架构的训练与合并、打包与导入），**没有在任何真实模型上跑通**。仅训练目录的 Qwen2.5 32B 与 Yi 1.5 34B 只有估算的显存数值，没有 WebLLM 记录可核对，也需要本机编译。训练 70B 模型需要远超消费级显卡的显存，本流程不适用于它们。第一次使用某个模型时，请先按第 5 节“需要核对的部分”检查。

运行库只与架构和量化有关，与权重无关。例如 Qwen2.5-7B 与 Qwen2.5-Coder-7B 共用一个 wasm（均使用 Qwen2-7B 的库）。

## 1. 第一次准备

```bash
python3 training/ojtrain.py models                                  # 列出可训练的模型
python3 training/ojtrain.py init --model Qwen2.5-3B-Instruct-q4f16_1-MLC   # 从清单生成配置
python3 training/ojtrain.py setup                                   # 创建 training/.venv 并安装依赖
training/.venv/bin/python training/ojtrain.py doctor                # 检查环境
```

`setup` 会安装 PyTorch。请先确认显卡驱动和 CUDA 版本。如果需要特定 CUDA 版本的 PyTorch，可以先在 `training/.venv` 中按 pytorch.org 的说明手动安装对应版本，然后再运行 `setup`（pip 会跳过已满足的依赖）。

`doctor` 输出每一项的状态：

| 状态 | 含义 |
| --- | --- |
| `OK` | 通过 |
| `WARN` | 可以继续，但某些步骤会失败或很慢（例如没有 GPU、缺少 mlc_llm、自定义模型尚未编译） |
| `FAIL` | 必须先解决（例如 Python 版本过低、配置文件有误） |

## 2. 导入或下载基础模型

二选一：

```bash
# 方式 A：使用本机已有的模型目录（不复制，只记录路径）
python3 training/ojtrain.py import-model /path/to/模型目录

# 方式 B：不做任何操作，由 run 自动从 Hugging Face 下载到 training/models/
```

模型目录需要包含 `config.json`、`*.safetensors` 权重，以及 `tokenizer.json` 或 `tokenizer.model`。

**注意**：本机的基础模型必须与配置对应的原始仓库是同一个模型（同架构、同尺寸）。例如清单中的 Qwen2.5-3B 对应 `Qwen/Qwen2.5-Coder-1.5B-Instruct`，不能换成 7B 的目录，否则运行库与权重对不上，网页端会加载失败。

gated 模型（Llama、Gemma 等）需要在 Hugging Face 上同意许可协议，并在本机设置环境变量 `HF_TOKEN`。不要把 token 写进文件或发给别人。清单中标记 `gated` 的模型会在 `models` 输出中提示。

## 3. 导入数据

支持 `.jsonl`、`.json`（数组）、`.csv`（带表头）。每条样本可以是以下格式之一：

```json
{"prompt": "题目或问题", "response": "回答"}
{"instruction": "...", "input": "可选补充", "output": "回答"}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

`messages` 格式可以包含 `system` 消息。对于不接受 system 角色的对话模板（例如 Gemma），训练时会自动把 system 内容并入第一条用户消息。

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

配置文件是 `training/config.json`，由 `init` 生成。相对路径都相对于配置文件所在目录。

| 字段 | 说明 | 默认值 |
| --- | --- | --- |
| `name` | 本次训练的名称，决定工作目录 `training/work/<name>/` | 由模型 id 推导，如 `Qwen2.5-3B-Instruct-tuned` |
| `catalog_id` | 来自清单的模型 id（自定义模型为 null） | — |
| `base.hf_repo` | 基础模型的 Hugging Face 仓库 | 由清单或 `--hf-repo` 决定 |
| `base.local_dir` | 本机模型目录；设置后优先使用它 | `null` |
| `data.raw` | 原始数据文件 | `data/raw.jsonl` |
| `data.val_ratio` | 验证集比例（0 到 0.5） | `0.05` |
| `data.max_chars` | 单条样本的最大字符数，超过则丢弃 | `8000` |
| `data.seed` | 切分验证集用的随机种子 | `42` |
| `train.epochs` | 训练轮数 | `2` |
| `train.learning_rate` | 学习率 | `0.0002` |
| `train.lora_r` / `lora_alpha` / `lora_dropout` | LoRA 参数 | `16` / `32` / `0.05` |
| `train.target_modules` | 期望挂载 LoRA 的模块；模型中不存在时自动改用实际存在的注意力投影（如 Phi-3 的 `qkv_proj`） | q/k/v/o 投影 |
| `train.max_length` | 超过该 token 数的样本跳过 | `1024` |
| `train.batch_size` × `grad_accum` | 每步样本数与梯度累积，相乘得到有效 batch | `1` × `8` |
| `train.dtype` | `auto` / `bf16` / `fp16` / `fp32` | `auto` |
| `train.grad_checkpointing` | 省显存，速度变慢 | `true` |
| `mlc.quantization` | MLC 量化方式 | `q4f16_1` |
| `mlc.conv_template` | MLC 对话模板名（清单中的值待核对，见第 5 节） | 由清单决定 |
| `mlc.context_window_size` | 上下文长度（与官方记录一致，清单模型通常为 4096） | `4096` |
| `mlc.compile` | `true` 时在 convert 步骤用 `mlc_llm compile` 编译运行库 | 清单模型为 `false` |
| `mlc.model_lib_url` | 网页使用的运行库地址；清单模型为官方地址，自定义模型为本地地址 | — |
| `mlc.wasm_path` / `wasm_url` | 本地 wasm 路径；缺失时清单模型从 `wasm_url` 下载 | `libs/...wasm` |
| `pack.model_id` | 模型 id（只含字母、数字、`._-`） | `<name>-v1` |
| `pack.label` | 网页“新建AI”中显示的名称 | `<模型名> 训练版 v1` |
| `pack.vram_required_MB` | 显存提示（清单模型来自官方记录） | 清单决定 |
| `pack.source_model` | 文件地址的基础路径（清单模型为官方 MLC 仓库） | 清单决定 |
| `pack.overrides` | 与官方记录一致的覆盖项（如 `max_history_size`） | `{}` |

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
| `convert` | 用 mlc_llm 量化并转换为 WebLLM 格式；自定义模型还要编译运行库 | `work/<name>/mlc/`（+ 自定义模型的 wasm） |
| `pack` | 打包为 `.ojmodel`（自定义模型会把 wasm 一起打进文件） | `work/<name>/<model_id>.ojmodel` |

`train` 结束后，训练元数据写在 `work/<name>/lora/training_meta.json`，包括实际使用的 LoRA 模块、样本数、步数、训练 loss、验证 loss 和耗时。

### convert 步骤需要手动安装 mlc_llm

如果没有安装 `mlc_llm`，`convert` 会**暂停**（状态 `waiting`），并打印需要执行的命令。这不算失败，前面的步骤不会被重做。安装 MLC-LLM 后运行：

```bash
python3 training/ojtrain.py run --from convert
```

转换成功后，所用参数（量化、模板、上下文、是否编译）会记录在 `work/<name>/mlc/convert_params.json`。之后修改这些参数，需要用 `run --from convert --force` 重新转换；`--from` 只会从该步继续，已完成的步骤不会重跑。

如果转换在别的机器上完成，把输出目录（包含 `tensor-cache.json`、`mlc-chat-config.json`、分词器和权重分片；自定义模型还要有 wasm）完整复制到 `training/work/<name>/mlc/`（wasm 放到配置中的 `wasm_path`），然后同样运行 `run --from convert`。流程会检测到结果并直接继续。

MLC-LLM 的版本需要与网页使用的 WebLLM 0.2.86 对应的运行库兼容。**这一步本仓库没有验证过**，命令以 MLC 官方文档为准。

不要修改 `mlc-chat-config.json` 中与架构有关的字段（例如 `rope_theta`、`tie_word_embeddings`、层数）。这些值已经编译进 wasm，改了会导致输出乱码。

### 需要核对的部分（训练前请检查）

- **对话模板名**：清单中的 `conv_template` 是整理者的判断，**未经核对**（`catalog.json` 中 `verified` 均为 false）。它决定网页端如何拼接对话格式。请对照官方仓库的 `mlc-chat-config.json`（`conv_template` 字段）确认。如果不一致，修改配置中的 `mlc.conv_template` 后运行 `run --from convert --force`。
- **原始仓库**：清单中的 `hf_repo` 同样未经核对。如果下载时报 404，请到 Hugging Face 搜索正确的仓库名，修改 `base.hf_repo`。
- **少数较新的模型**（例如 Qwen3.5、Ministral 3）可能是多模态或混合结构，`AutoModelForCausalLM` 不一定能加载。训练前请先用 `doctor` 和一个小数据做一次 `run --from train` 试跑。

## 6. 自定义模型（清单之外）

```bash
python3 training/ojtrain.py init --hf-repo 组织/模型名 \
    --conv-template chatml --context-window-size 4096 --quantization q4f16_1
```

这条路径会在本机编译运行库，要求：
- MLC-LLM 支持该模型的架构（如 Llama、Qwen2/3、Mistral、Phi-3、Gemma 等常见架构）。不支持的架构会在 convert 步骤报错。
- 对话模板名与 MLC-LLM 版本中的模板一致（必须手动指定）。
- 编译需要 MLC-LLM 的完整工具链（TVM、WebGPU 的 emscripten 环境），按 MLC 官方文档安装。

编译产物的 wasm 会打进 `.ojmodel`，`model_lib` 使用本地地址 `https://local.oj-ai.invalid/libs/...`。网页导入时把 wasm 写入运行库缓存，之后离线可用。

## 7. 导入网页

1. 打开网页，点聊天顶部的“更改权限”，输入管理员密钥，升级为管理员。
2. 点“导入训练结果”，选择 `work/<name>/<model_id>.ojmodel`。
3. 导入后，模型出现在下拉框的“新建AI”分组中，名称后带“（训练结果）”。

导入的模型与官方模型的缓存分开保存，不会覆盖官方模型，也不会上传云端。

## 7a. 固化（可选）：把知识库与更正训练进权重

网页端的知识库与更正默认只作为提示词使用。若希望把它们写进模型权重，可以按下面三步操作：

1. 网页中“更改权限”后，在“AI 适应记录”里对该 AI 点“导出固化数据”，得到 `solidify-<模型>.jsonl`。
2. 在本机运行（需要 NVIDIA GPU，耗时见第 8 节估算）：

   ```bash
   python3 training/ojtrain.py solidify solidify-<模型>.jsonl
   ```

   命令会导入数据，并从“数据”步骤起强制重新执行 数据 → 训练 → 转换 → 打包（即使之前已完成，因为数据已改变）。结果是 `work/<name>/<model_id>.ojmodel`。
3. 在网页“导入训练结果”中导入。

注意事项：

- **只有记录了对应问题的条目才会导出**。网页记录反馈时才会保存触发它的问题，早期记录没有问题，会计入“跳过”。
- 每条样本是 `问题 → 内容`：更正的内容是“新说法”，知识库条目的内容是反馈原文。反馈原文不一定是标准回答，导出的文件应先人工检查（每行一条 JSON，可以直接编辑或删除）。网页端不做自动过滤。
- **这是对“更正不写入 `.ojmodel`”规则的明确例外**：训练结果会把更正与知识库写进权重，但 `.ojmodel` 本身仍然不包含它们的文字记录。
- 训练结果是新的模型 id，显示在“新建AI”中。原 AI 的提示词层记录不会自动套用到它；需要时请在新 AI 上重新给出反馈。
- 固化只是可选步骤。不做固化时，提示词层照常生效。

## 8. 时间与显存（估算，未实测）

**训练时基础模型不做量化**（使用 fp16/bf16），因此训练显存远高于网页推理所需。经验上，显存需求约为“基础权重（每 1B 参数约 2 GB）+ 4 GB 左右开销”。清单中的 `train_vram_gb_min` 即按此估计，实际以 `doctor` 与试跑为准。

| 模型规模 | 训练显存（估计） | 备注 |
| --- | --- | --- |
| 0.5–1.5B | 约 5–8 GB | 8 GB 显卡可以跑，建议开启 `grad_checkpointing` |
| 3–4B | 约 10–12 GB | |
| 7–8B | 约 18–20 GB | 需要 24 GB 显卡 |
| 13B | 约 30 GB | 需要 40 GB 以上显卡 |

以 1.5B 模型、3000 条样本、每条约 300 token、2 个 epoch 为例（约 180 万训练 token）：RTX 4060 级显卡约 50 分钟，高端显卡约 20–25 分钟。模型加载约 1–2 分钟，合并保存约 2–5 分钟，MLC 转换约 5–10 分钟，打包约 1–2 分钟。3B 模型耗时约为 1.5B 的 2 倍，7B 约为 5 倍。

以上数字是按公开的 LoRA 吞吐数据（RTX 4060 上 Qwen2.5-1.5B 约 630 token/s）推算的估计值，不是在本仓库环境中实测的。

## 9. 测试

```bash
npm test                  # 网页与打包的测试；包含清单与 web-llm 记录的一致性检查，以及跨语言导入检查
npm run test:training     # 训练工具的 Python 测试；未安装 PyTorch 时训练冒烟测试会跳过
node training/build_catalog.mjs          # 网页模型列表变化后，重新生成 catalog.json
node training/build_catalog.mjs --check  # 只检查是否同步
```

`tests/test_train_smoke.py` 使用随机初始化的极小模型（`tests/make_tiny_model.py`，支持 qwen2、llama、mistral、phi3、qwen3、gemma2 六种架构，并可模拟拒绝 system 角色的模板）在 CPU 上跑通“构造标签 → LoRA 训练 → 合并 → 重新加载”。它只验证代码路径，不代表任何真实模型的质量。

## 已验证与未验证

已验证（沙箱中，无 GPU）：
- 配置校验、路径解析、`models` / `init --model` / `init --hf-repo` 等命令。
- 清单与网页下拉框、web-llm 记录一致（`npm test` 中的检查）。
- 数据导入（jsonl / json / csv）、角色校验、切分训练集与验证集。
- 环境检查的输出与退出码（包括编译模式下运行库为 WARN）。
- 流水线的步骤顺序、断点续跑、`mlc_llm` 缺失时暂停、失败时停止、编译命令与产物检查。
- LoRA 训练、合并、重新加载：qwen2、llama、mistral、phi3（自动改用 `o_proj` + `qkv_proj`）、qwen3、gemma2，以及拒绝 system 角色的模板（CPU，极小随机模型）。
- 清单模型的打包结果（沿用官方运行库）与自定义模型的打包结果（本地运行库）都能被网页端的 `importModelFile` 导入，并保留 `context_window_size` 与 `max_history_size` 覆盖项。
- 网页端 WebLLM 加载 wasm 时会先查缓存，因此导入的本地运行库不会访问 `local.oj-ai.invalid`（阅读 web-llm 源码确认，未在浏览器中运行）。

未验证：
- `setup` 在真实机器上安装 PyTorch 的过程。
- 从 Hugging Face 下载基础模型（沙箱无法访问 huggingface.co）。清单中的原始仓库名与对话模板名**未核对**。
- 真实 GPU 训练的速度、显存占用和效果。
- MLC 转换与 `mlc_llm compile` 步骤（沙箱无法安装 mlc-llm；流水线使用桩程序验证接线）。
- 清单中各模型转换后在浏览器里的加载和生成质量。
- 具体某个真实模型（如 Qwen3.5、Ministral 3）是否能被 transformers 训练。

## 文件

| 文件 | 作用 |
| --- | --- |
| `ojtrain.py` | 一站式命令行入口 |
| `catalog.json` / `build_catalog.mjs` | 网页下拉框中 29 个模型，加上仅训练目录的 2 个超大模型的训练信息；生成脚本从 web-llm 记录读取事实字段 |
| `catalog.py` | 把清单模型或自定义模型写入配置 |
| `config.py` / `config.example.json` | 配置的读取、校验、路径解析与模板 |
| `doctor.py` | 环境检查 |
| `fetch_model.py` | 下载或导入基础模型，下载运行库并检查 wasm 文件头 |
| `prepare_dataset.py` | 导入、校验、切分数据 |
| `pipeline.py` | 步骤编排、状态记录、断点续跑、编译与打包 |
| `train_lora.py` | LoRA 微调（自动识别目标模块、处理 system 角色），可选验证集与合并输出 |
| `pack_ojmodel.py` | 把 MLC 输出（及可选 wasm）打包为 `.ojmodel`（只用标准库） |
| `requirements.txt` | 训练依赖（GPU 环境） |
| `examples/sample.jsonl` | 示例数据 |
| `tests/` | 单元测试、流水线测试、清单测试、训练冒烟测试（多架构）、极小模型生成器 |

个人文件不进入仓库（已在 `.gitignore` 中忽略）：`config.json`、`data/`、`models/`、`work/`、`libs/`、`.venv/`。
