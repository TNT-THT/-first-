"""
模型清单：把网页下拉框中的模型映射为训练配置。

两条路径：
1. 清单模型（apply_catalog）：使用官方记录中的运行库（model_lib），不编译。
   训练后的权重与官方模型同架构、同量化、同形状，因此可以复用同一个 wasm。
2. 自定义模型（apply_custom）：清单之外的架构。用 mlc_llm compile 在本机生成运行库，
   并把 wasm 一起打进 .ojmodel，model_lib 使用本地地址（网页导入时写入运行库缓存）。
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(HERE, "catalog.json")
LOCAL_HOST = "https://local.oj-ai.invalid"
QUANT_RE = re.compile(r"^q\d+f\d+(_\d+)?$")


class CatalogError(ValueError):
    pass


def load(path=CATALOG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["models"]


def find(model_id, path=CATALOG_PATH):
    for entry in load(path):
        if entry["id"] == model_id:
            return entry
    raise CatalogError(f"清单中没有模型 {model_id}。运行 `ojtrain.py models` 查看可选模型，或用 --hf-repo 使用自定义模型。")


def display_name(label):
    """“Qwen3 1.7B（推理·小）” → “Qwen3 1.7B”。"""
    return label.split("（")[0].strip() or label


def default_name(model_id):
    """官方 id 去掉 -MLC 与量化后缀，再加 -tuned：Qwen2.5-Coder-1.5B-Instruct-tuned。"""
    base = re.sub(r"-MLC$", "", model_id)
    base = re.sub(r"-q\d+f\d+(_\d+)?$", "", base)
    return f"{base}-tuned"


def apply_catalog(cfg, entry, name=None):
    """把清单中的模型写入配置（原地修改并返回 cfg）。"""
    if not entry.get("in_dropdown", True):
        raise CatalogError(
            f"{entry['id']} 只在训练目录中，网页下拉框没有它，也没有官方运行库，必须在本机编译。"
            f"请使用：init --hf-repo {entry['hf_repo']} --conv-template {entry['conv_template']} --context-window-size 4096"
        )
    name = name or default_name(entry["id"])
    model_id = f"{name}-v1"
    cfg["name"] = name
    cfg["catalog_id"] = entry["id"]
    cfg["base"] = {"hf_repo": entry["hf_repo"], "local_dir": None}
    cfg["mlc"] = {
        "quantization": entry["quantization"],
        "conv_template": entry["conv_template"],
        "context_window_size": entry["context_window_size"],
        "compile": False,
        "model_lib_url": entry["model_lib_url"],
        "wasm_path": f"libs/{entry['wasm_file']}",
        "wasm_url": entry["model_lib_url"],
    }
    cfg["pack"] = {
        "model_id": model_id,
        "label": f"{display_name(entry['label'])} 训练版 v1",
        "vram_required_MB": entry.get("vram_required_MB"),
        "source_model": entry["source_model"],
        "overrides": dict(entry.get("overrides") or {}),
    }
    return cfg


def apply_custom(cfg, *, hf_repo, conv_template, context_window_size, quantization="q4f16_1", name=None):
    """清单之外的模型：本机编译运行库。"""
    if not hf_repo or not re.match(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$", hf_repo):
        raise CatalogError("--hf-repo 格式应为 “组织/模型名”")
    if not QUANT_RE.match(quantization or ""):
        raise CatalogError("--quantization 格式应类似 q4f16_1")
    if not conv_template:
        raise CatalogError("自定义模型必须指定 --conv-template（对话模板名，需与 MLC-LLM 版本一致）")
    if not context_window_size or int(context_window_size) <= 0:
        raise CatalogError("--context-window-size 必须是正整数")
    name = name or re.sub(r"[^A-Za-z0-9._-]", "-", hf_repo.split("/")[-1]).strip("-") + "-tuned"
    model_id = f"{name}-v1"
    cfg["name"] = name
    cfg["catalog_id"] = None
    cfg["base"] = {"hf_repo": hf_repo, "local_dir": None}
    cfg["mlc"] = {
        "quantization": quantization,
        "conv_template": conv_template,
        "context_window_size": int(context_window_size),
        "compile": True,
        "model_lib_url": f"{LOCAL_HOST}/libs/{model_id}.wasm",
        "wasm_path": f"libs/{model_id}.wasm",
        "wasm_url": None,
    }
    cfg["pack"] = {
        "model_id": model_id,
        "label": f"{name} 训练版 v1",
        "vram_required_MB": None,
        "source_model": f"{LOCAL_HOST}/models/{model_id}/",
        "overrides": {},
    }
    return cfg


def summary_lines(models):
    """`ojtrain.py models` 的输出。"""
    lines = []
    for m in models:
        kind = "推理" if m["reasoning"] else "非推理"
        gated = "（需 HF_TOKEN 与许可）" if m["hf_gated"] else ""
        lines.append(f"{m['id']}")
        lines.append(f"    {m['label']}  {m['size_class'] or '-'} · {kind}")
        lines.append(f"    原始仓库：{m['hf_repo']}{gated}  参数约 {m['params_b']}B")
        lines.append(f"    训练显存估计：≥ {m['train_vram_gb_min']} GB  对话模板：{m['conv_template']}（待核对）")
        if m.get("in_dropdown", True):
            lines.append(f"    推理显存：约 {m['inference_vram_gb']} GB（WebLLM 记录）")
        else:
            lines.append(
                f"    推理显存：约 {m['inference_vram_gb']} GB（估算）· 仅训练目录，不在网页下拉框中，需本机编译："
                f"init --hf-repo {m['hf_repo']} --conv-template {m['conv_template']} --context-window-size 4096"
            )
    return lines
