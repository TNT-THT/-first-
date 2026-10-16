#!/usr/bin/env python3
"""
把 WebLLM/MLC 格式的模型目录打包成 .ojmodel 文件，供网页“导入训练结果”使用。

用法：
  python3 pack_ojmodel.py --dir mlc_out/ --wasm libs/Qwen2-1.5B-q4f16_1-webgpu.wasm \
      --model-lib-url https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/Qwen2-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm \
      --model-id my-tuned-1.5b --label "我的微调版" --out my-tuned-1.5b.ojmodel

只使用标准库。文件格式必须与 web/src/model-file.js 完全一致：
  4 字节魔数 "OJMD" + uint32 小端的头部长度 + UTF-8 JSON 头部 + 各文件的原始数据。
头部中的偏移相对于数据区起点。
"""
import argparse
import json
import os
import re
import struct
import sys
from datetime import datetime, timezone

MAGIC = b"OJMD"
FORMAT = "oj-ai-model"
FORMAT_VERSION = 1
PREFIX_BYTES = 8
MODEL_SCOPE = "webllm/model"
CONFIG_SCOPE = "webllm/config"
WASM_SCOPE = "webllm/wasm"
OPTIONAL_RECORD_FIELDS = ("overrides", "vram_required_MB", "low_resource_required")


def clean_model_url(model_url):
    """与 web/src/model-file.js 中的 cleanModelUrl 规则一致。"""
    url = str(model_url)
    if not url.endswith("/"):
        url += "/"
    if not re.search(r".+/resolve/.+/", url):
        url += "resolve/main/"
    return url


def _file_url(base, name):
    return base + name


def _require(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少 {what}：{path}")
    return path


def collect_entries(model_dir, model_base, model_lib_url, wasm_path):
    """按 web/src/model-file.js 的规则收集条目：权重索引、分片、分词器、配置、运行库。"""
    entries = []

    tensor_cache = _require(os.path.join(model_dir, "tensor-cache.json"), "权重索引 tensor-cache.json")
    with open(tensor_cache, "r", encoding="utf-8") as f:
        index = json.load(f)
    shard_names = []
    for rec in index.get("records", []):
        name = rec.get("dataPath")
        if name and name not in shard_names:
            shard_names.append(name)
    if not shard_names:
        raise ValueError("tensor-cache.json 中没有任何权重分片（records 为空）")

    model_files = ["tensor-cache.json"] + shard_names
    tokenizer = None
    for name in ("tokenizer.json", "tokenizer.model"):
        if os.path.isfile(os.path.join(model_dir, name)):
            tokenizer = name
            break
    if tokenizer is None:
        raise FileNotFoundError("缺少分词器：需要 tokenizer.json 或 tokenizer.model")
    model_files.append(tokenizer)

    for name in model_files:
        _require(os.path.join(model_dir, name), f"权重文件 {name}")
        entries.append({"scope": MODEL_SCOPE, "path": os.path.join(model_dir, name), "url": _file_url(model_base, name)})

    config_path = _require(os.path.join(model_dir, "mlc-chat-config.json"), "mlc-chat-config.json")
    entries.append({"scope": CONFIG_SCOPE, "path": config_path, "url": _file_url(model_base, "mlc-chat-config.json")})

    _require(wasm_path, "运行库 wasm")
    entries.append({"scope": WASM_SCOPE, "path": wasm_path, "url": model_lib_url})
    return entries


def pack(
    model_dir,
    out_path,
    *,
    model_id,
    label,
    model_lib_url,
    wasm_path,
    source_model,
    record_extra=None,
    created_at=None,
):
    """打包并返回 {"entries": n, "bytes": total}。"""
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", model_id or ""):
        raise ValueError("model_id 只能包含字母、数字、点、下划线和连字符（最长 128 字符）")
    for url_name, url in (("source_model", source_model), ("model_lib_url", model_lib_url)):
        if not str(url).startswith("https://"):
            raise ValueError(f"{url_name} 必须是 https 地址")

    base = clean_model_url(source_model)
    entries = collect_entries(model_dir, base, model_lib_url, wasm_path)

    meta = []
    offset = 0
    for e in entries:
        size = os.path.getsize(e["path"])
        meta.append({"scope": e["scope"], "url": e["url"], "size": size, "offset": offset})
        offset += size

    source = {"model": source_model, "model_lib": model_lib_url}
    for key in OPTIONAL_RECORD_FIELDS:
        if record_extra and key in record_extra:
            source[key] = record_extra[key]

    header = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "modelId": model_id,
        "label": label[:200],
        "source": source,
        "createdAt": created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "entries": meta,
    }
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")

    tmp_path = out_path + ".part"
    with open(tmp_path, "wb") as out:
        out.write(MAGIC + struct.pack("<I", len(header_bytes)))
        out.write(header_bytes)
        for e in entries:
            with open(e["path"], "rb") as src:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
    os.replace(tmp_path, out_path)
    return {"entries": len(meta), "bytes": PREFIX_BYTES + len(header_bytes) + offset}


def _parse_args(argv):
    p = argparse.ArgumentParser(description="把 MLC 模型目录打包为 .ojmodel（可在网页“导入训练结果”）")
    p.add_argument("--dir", required=True, help="MLC 输出目录（含 tensor-cache.json、分片、分词器、mlc-chat-config.json）")
    p.add_argument("--wasm", required=True, help="与该模型架构匹配的 wasm 运行库文件")
    p.add_argument("--model-lib-url", required=True, help="该 wasm 在网页中使用的 https 地址（必须与 WebLLM 的 model_lib 一致）")
    p.add_argument("--model-id", required=True, help="模型 id（只含字母、数字、._-）")
    p.add_argument("--label", required=True, help="在“新建AI”中显示的名称")
    p.add_argument(
        "--source-model",
        default=None,
        help="来源模型地址（https）。默认 https://huggingface.co/oj-ai-training/<model-id>，仅用于标识，导入时不会下载",
    )
    p.add_argument("--record-json", default=None, help="可选：WebLLM 记录的 JSON 文件，取其中 overrides 等字段")
    p.add_argument("--out", required=True, help="输出的 .ojmodel 路径")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    record_extra = None
    if args.record_json:
        with open(args.record_json, "r", encoding="utf-8") as f:
            record_extra = json.load(f)
    source_model = args.source_model or f"https://huggingface.co/oj-ai-training/{args.model_id}"
    result = pack(
        args.dir,
        args.out,
        model_id=args.model_id,
        label=args.label,
        model_lib_url=args.model_lib_url,
        wasm_path=args.wasm,
        source_model=source_model,
        record_extra=record_extra,
    )
    print(f"已写入 {args.out}：{result['entries']} 个文件，{result['bytes'] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as err:
        print(f"打包失败：{err}", file=sys.stderr)
        sys.exit(1)
