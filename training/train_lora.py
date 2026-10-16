#!/usr/bin/env python3
"""
LoRA 微调运行时（PyTorch + transformers + PEFT）。需要 NVIDIA GPU（显存建议 ≥ 8 GB，1.5B 模型）。

输入：prepare_dataset.py 生成的 JSONL（每行 {"messages": [...]}）。
只对 assistant 的回复计算损失（其他部分标记为 -100），这样模型学的是“怎么回答”。

输出：
  --output-dir   LoRA 适配器（adapter）、分词器和 training_meta.json
  --merged-dir   可选：把适配器合并进基础权重后的完整 HF 模型（用于后续 MLC 转换，见 README）

示例：
  python3 train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --data train.jsonl \
      --output-dir runs/qwen15-lora --merged-dir runs/qwen15-merged --epochs 2

注意：此脚本在本仓库的沙箱中没有 GPU，无法做真实训练；只做过 CPU 冒烟测试（见 tests/make_tiny_model.py）。
"""
import argparse
import json
import os
import random
import sys
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LoRA 微调（仅对 assistant 回复计算损失）")
    p.add_argument("--base", required=True, help="基础模型：HF 仓库 id 或本地目录")
    p.add_argument("--data", required=True, help="训练数据 JSONL（messages 格式）")
    p.add_argument("--output-dir", required=True, help="保存 LoRA 适配器的目录")
    p.add_argument("--merged-dir", default=None, help="可选：保存合并后的完整模型")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1, help="覆盖 epochs（用于快速测试）")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=1024, help="超过该长度的样本会被跳过")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    p.add_argument("--grad-checkpointing", action="store_true", help="省显存，速度变慢")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU（仅用于冒烟测试）")
    return p.parse_args(argv)


def load_examples(path):
    """读取 messages 格式的 JSONL。原始数据请先用 prepare_dataset.py 整理。"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as err:
                raise SystemExit(f"{path} 第 {lineno} 行不是合法 JSON：{err}。请先运行 prepare_dataset.py 整理数据。")
            if not isinstance(obj, dict) or "messages" not in obj:
                raise SystemExit(f"{path} 第 {lineno} 行缺少 messages 字段。请先运行 prepare_dataset.py 整理数据。")
            examples.append(obj)
    return examples


def _chat_ids(tokenizer, messages, add_generation_prompt):
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    # 新版 transformers 返回 dict-like（含 input_ids），旧版直接返回列表
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    return list(out)


def build_features(examples, tokenizer, max_length):
    """把对话转换为 input_ids 与 labels。只有 assistant 的回复位置保留标签。"""
    features = []
    skipped = 0
    for ex in examples:
        msgs = ex["messages"]
        ids = _chat_ids(tokenizer, msgs, add_generation_prompt=False)
        labels = [-100] * len(ids)
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            start = len(_chat_ids(tokenizer, msgs[:i], add_generation_prompt=True))
            end = len(_chat_ids(tokenizer, msgs[: i + 1], add_generation_prompt=False))
            for j in range(start, min(end, len(ids))):
                labels[j] = ids[j]
        if max_length and len(ids) > max_length:
            skipped += 1
            continue
        if all(v == -100 for v in labels):
            skipped += 1
            continue
        features.append({"input_ids": ids, "labels": labels})
    return features, skipped


def make_collator(pad_id):
    import torch

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attention = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attention.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention),
        }

    return collate


def pick_dtype(name, cpu):
    import torch

    if cpu or name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit("基础模型的分词器没有 chat_template，无法构造对话格式。")

    examples = load_examples(args.data)
    features, skipped = build_features(examples, tokenizer, args.max_length)
    if not features:
        raise SystemExit("没有可训练的样本（全部超长或没有 assistant 回复）。")
    print(f"训练样本：{len(features)} 条，跳过 {skipped} 条", flush=True)

    dtype = pick_dtype(args.dtype, args.cpu)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    use_cuda = torch.cuda.is_available() and not args.cpu
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="no",
        bf16=use_cuda and dtype == torch.bfloat16,
        fp16=use_cuda and dtype == torch.float16,
        report_to=[],
        seed=args.seed,
        remove_unused_columns=False,
        use_cpu=not use_cuda,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=features,
        data_collator=make_collator(tokenizer.pad_token_id),
    )
    started = time.time()
    result = trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.merged_dir:
        merged = model.merge_and_unload()
        merged = merged.to(dtype)
        os.makedirs(args.merged_dir, exist_ok=True)
        merged.save_pretrained(args.merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.merged_dir)

    meta = {
        "base": args.base,
        "examples": len(features),
        "skipped": skipped,
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "seconds": round(time.time() - started, 1),
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "target_modules": sorted(args.target_modules.split(","))},
        "seed": args.seed,
        "dtype": str(dtype).replace("torch.", ""),
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    args = parse_args(argv)
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
