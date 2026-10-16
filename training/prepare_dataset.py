#!/usr/bin/env python3
"""
把训练数据整理成统一的聊天格式（每行一个 JSON：{"messages": [...]}），并切出验证集。

支持的输入文件：
  - .jsonl：每行一个 JSON 对象
  - .json ：一个 JSON 数组，数组中每个元素是一个对象
  - .csv  ：第一行是表头，列名见下方“每条样本的格式”

每条样本可以是以下三种格式之一：
  1. {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  2. {"prompt": "题目或问题", "response": "回答"}           （可选 "system"）
  3. {"instruction": "...", "input": "可选补充", "output": "回答"}

规则：
  - system 只能出现在第一条；之后必须是 user / assistant 交替，并以 assistant 结尾。
  - 内容去掉首尾空白后不能为空；超过 max_chars 的样本会被丢弃。
  - 验证集按 seed 随机抽取（可复现），用于训练后比较效果；样本太少时不切分。
  - 只使用标准库，不需要 GPU。

用法：
  python3 prepare_dataset.py --input raw.jsonl --output train.jsonl --val-output val.jsonl --val-ratio 0.05
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

ROLES = {"system", "user", "assistant"}
MIN_ITEMS_FOR_VAL = 20


class SkipExample(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _text(value, field):
    if not isinstance(value, str):
        raise SkipExample(f"{field} 不是字符串")
    text = value.strip()
    if not text:
        raise SkipExample(f"{field} 为空")
    return text


def to_messages(obj):
    """把任一支持的输入格式转换为 messages 列表（未校验）。"""
    if not isinstance(obj, dict):
        raise SkipExample("不是 JSON 对象")
    if "messages" in obj:
        msgs = obj["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise SkipExample("messages 为空或不是列表")
        out = []
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in ROLES:
                raise SkipExample("messages 中有无效的 role")
            out.append({"role": m["role"], "content": _text(m.get("content"), "content")})
        return out

    out = []
    if "system" in obj and str(obj["system"] or "").strip():
        out.append({"role": "system", "content": str(obj["system"]).strip()})
    if "prompt" in obj and "response" in obj:
        out.append({"role": "user", "content": _text(obj["prompt"], "prompt")})
        out.append({"role": "assistant", "content": _text(obj["response"], "response")})
        return out
    if "instruction" in obj and "output" in obj:
        question = _text(obj["instruction"], "instruction")
        extra = obj.get("input")
        if isinstance(extra, str) and extra.strip():
            question = f"{question}\n\n{extra.strip()}"
        out.append({"role": "user", "content": question})
        out.append({"role": "assistant", "content": _text(obj["output"], "output")})
        return out
    raise SkipExample("无法识别的格式（需要 messages、prompt/response 或 instruction/output）")


def validate(messages):
    """检查角色顺序：可选 system，之后 user/assistant 交替，以 assistant 结尾。"""
    body = messages
    if messages and messages[0]["role"] == "system":
        body = messages[1:]
    if any(m["role"] == "system" for m in body):
        raise SkipExample("system 只能出现在第一条")
    if not body:
        raise SkipExample("没有对话内容")
    for i, m in enumerate(body):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            raise SkipExample("角色没有按 user/assistant 交替")
    if body[-1]["role"] != "assistant":
        raise SkipExample("对话必须以 assistant 回复结尾")


def process_objects(objs, max_chars=8000):
    """把已解析的对象转换为样本。返回 (保留的样本列表, 丢弃原因计数)。"""
    kept = []
    skipped = Counter()
    for obj in objs:
        try:
            messages = to_messages(obj)
            validate(messages)
            if sum(len(m["content"]) for m in messages) > max_chars:
                raise SkipExample(f"超过 {max_chars} 字符")
        except SkipExample as err:
            skipped[err.reason] += 1
            continue
        kept.append({"messages": messages})
    return kept, skipped


def process(lines, max_chars=8000):
    """处理 JSONL 的行。返回 (保留的样本列表, 丢弃原因计数)。"""
    objs = []
    skipped = Counter()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            skipped["JSON 解析失败"] += 1
    kept, sk = process_objects(objs, max_chars)
    skipped.update(sk)
    return kept, skipped


def load_objects(path):
    """读取 .jsonl / .json / .csv，返回 (对象列表, 解析失败计数)。"""
    ext = os.path.splitext(path)[1].lower()
    skipped = Counter()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = []
            for row in csv.DictReader(f):
                cleaned = {k.strip(): (v or "").strip() for k, v in row.items() if k}
                if any(cleaned.values()):
                    rows.append(cleaned)
        return rows, skipped
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(".json 文件必须是数组（每个元素是一条样本）")
        return data, skipped
    if ext in (".jsonl", ".txt", ""):
        objs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    objs.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped["JSON 解析失败"] += 1
        return objs, skipped
    raise ValueError(f"不支持的文件类型：{ext}（请使用 .jsonl、.json 或 .csv）")


def split(items, val_ratio=0.0, seed=42):
    """按 seed 随机切出验证集。返回 (训练集, 验证集)。"""
    if val_ratio <= 0 or len(items) < MIN_ITEMS_FOR_VAL:
        return list(items), []
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n_val = max(1, round(len(items) * val_ratio))
    val_idx = set(order[:n_val])
    train = [items[i] for i in range(len(items)) if i not in val_idx]
    val = [items[i] for i in sorted(val_idx)]
    return train, val


def write_jsonl(path, items):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_dataset(raw_path, out_dir, val_ratio=0.0, max_chars=8000, seed=42):
    """完整流程：读取 → 校验 → 切分 → 写出 train.jsonl / val.jsonl。返回统计信息。"""
    objs, parse_skipped = load_objects(raw_path)
    kept, skipped = process_objects(objs, max_chars)
    skipped.update(parse_skipped)
    if not kept:
        raise ValueError("没有可用的样本，请检查输入文件和格式。")
    train, val = split(kept, val_ratio, seed)
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    write_jsonl(train_path, train)
    if val:
        write_jsonl(val_path, val)
    elif os.path.exists(val_path):
        os.remove(val_path)
    return {
        "train_path": train_path,
        "val_path": val_path if val else None,
        "train": len(train),
        "val": len(val),
        "skipped": dict(skipped),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="整理训练数据为统一的 messages 格式（支持 jsonl / json / csv）")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="训练集输出路径（.jsonl）")
    p.add_argument("--val-output", default=None, help="验证集输出路径（.jsonl），与 --val-ratio 配合使用")
    p.add_argument("--val-ratio", type=float, default=0.0, help="验证集比例，例如 0.05")
    p.add_argument("--max-chars", type=int, default=8000, help="单条样本的最大字符数")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    objs, parse_skipped = load_objects(args.input)
    kept, skipped = process_objects(objs, args.max_chars)
    skipped.update(parse_skipped)
    if not kept:
        print("没有可用的样本，请检查输入格式。", file=sys.stderr)
        return 1
    train, val = split(kept, args.val_ratio, args.seed)
    write_jsonl(args.output, train)
    if val:
        if not args.val_output:
            print("指定了 --val-ratio 但没有 --val-output，验证集将被丢弃。", file=sys.stderr)
        else:
            write_jsonl(args.val_output, val)
    print(f"训练集 {len(train)} 条，验证集 {len(val)} 条，丢弃 {sum(skipped.values())} 条")
    for reason, n in skipped.most_common():
        print(f"  - {reason}：{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
