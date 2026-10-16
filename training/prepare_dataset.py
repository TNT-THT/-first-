#!/usr/bin/env python3
"""
把训练数据整理成统一的聊天格式（每行一个 JSON：{"messages": [...]}）。

支持的输入（每行一个 JSON 对象，三选一）：
  1. {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  2. {"prompt": "题目或问题", "response": "回答"}           （可选 "system"）
  3. {"instruction": "...", "input": "可选补充", "output": "回答"}

规则：
  - system 只能出现在第一条；之后必须是 user / assistant 交替，并以 assistant 结尾。
  - 内容去掉首尾空白后不能为空；超过 --max-chars 的样本会被丢弃。
  - 只使用标准库，不需要 GPU。

用法：
  python3 prepare_dataset.py --input raw.jsonl --output train.jsonl
"""
import argparse
import json
import sys
from collections import Counter

ROLES = {"system", "user", "assistant"}


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
    if "system" in obj and str(obj["system"]).strip():
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


def process(lines, max_chars=8000):
    """返回 (保留的样本列表, 丢弃原因计数)。"""
    kept = []
    skipped = Counter()
    for lineno, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            messages = to_messages(obj)
            validate(messages)
            if sum(len(m["content"]) for m in messages) > max_chars:
                raise SkipExample(f"超过 {max_chars} 字符")
        except json.JSONDecodeError:
            skipped["JSON 解析失败"] += 1
            continue
        except SkipExample as err:
            skipped[err.reason] += 1
            continue
        kept.append({"messages": messages})
    return kept, skipped


def main(argv=None):
    p = argparse.ArgumentParser(description="整理训练数据为统一的 messages 格式")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-chars", type=int, default=8000, help="单条样本的最大字符数")
    args = p.parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as f:
        kept, skipped = process(f, args.max_chars)
    if not kept:
        print("没有可用的样本，请检查输入格式。", file=sys.stderr)
        return 1
    with open(args.output, "w", encoding="utf-8") as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"保留 {len(kept)} 条，丢弃 {sum(skipped.values())} 条")
    for reason, n in skipped.most_common():
        print(f"  - {reason}：{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
