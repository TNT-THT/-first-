"""
训练配置：读取、校验、生成默认配置，并解析各步骤使用的路径。

配置文件默认是 training/config.json（由 `ojtrain.py init` 从 config.example.json 生成）。
相对路径都相对于配置文件所在目录。
"""
import json
import os
import re
import shutil

import catalog

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, "config.example.json")
DEFAULT_CONFIG = os.path.join(HERE, "config.json")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
HF_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
DTYPES = ("auto", "bf16", "fp16", "fp32")


class ConfigError(ValueError):
    pass


def load_example():
    with open(EXAMPLE, "r", encoding="utf-8") as f:
        return json.load(f)


def write_config(path, cfg):
    """写入配置文件（去掉以下划线开头的内部字段）。"""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
        f.write("\n")


def init_config(path=DEFAULT_CONFIG, name=None, force=False, model=None, custom=None):
    """生成配置。model 为清单中的模型 id；custom 为自定义模型的参数字典（见 catalog.apply_custom）。"""
    if os.path.exists(path) and not force:
        raise ConfigError(f"配置文件已存在：{path}（如需覆盖，请加 --force）")
    cfg = load_example()
    if model:
        cfg = catalog.apply_catalog(cfg, catalog.find(model), name=name)
    elif custom:
        cfg = catalog.apply_custom(cfg, name=name, **custom)
    elif name:
        cfg["name"] = name
        cfg["pack"]["model_id"] = f"{name}-v1"
    write_config(path, cfg)
    return path


def validate(cfg):
    """返回错误信息列表（中文），空列表表示通过。"""
    errs = []

    def need(cond, msg):
        if not cond:
            errs.append(msg)

    name = cfg.get("name")
    need(isinstance(name, str) and NAME_RE.match(name or ""), "name 只能包含字母、数字、点、下划线和连字符（最长 64 字符）")

    base = cfg.get("base") or {}
    hf_repo = base.get("hf_repo")
    local_dir = base.get("local_dir")
    need(bool(hf_repo) or bool(local_dir), "base 中必须设置 hf_repo 或 local_dir 之一")
    if hf_repo:
        need(bool(HF_REPO_RE.match(hf_repo)), "base.hf_repo 格式应为 “组织/模型名”，例如 Qwen/Qwen2.5-Coder-1.5B-Instruct")

    data = cfg.get("data") or {}
    need(isinstance(data.get("raw"), str) and data.get("raw"), "data.raw 必须是数据文件路径")
    ratio = data.get("val_ratio", 0)
    need(isinstance(ratio, (int, float)) and 0 <= ratio < 0.5, "data.val_ratio 应在 0 到 0.5 之间")
    need(isinstance(data.get("max_chars", 8000), int) and data.get("max_chars", 8000) > 0, "data.max_chars 必须是正整数")

    train = cfg.get("train") or {}
    need(_pos(train.get("epochs")), "train.epochs 必须大于 0")
    need(_pos(train.get("learning_rate")), "train.learning_rate 必须大于 0")
    need(_pos_int(train.get("lora_r")), "train.lora_r 必须是正整数")
    need(_pos_int(train.get("lora_alpha")), "train.lora_alpha 必须是正整数")
    need(_pos_int(train.get("max_length")), "train.max_length 必须是正整数")
    need(_pos_int(train.get("batch_size")), "train.batch_size 必须是正整数")
    need(_pos_int(train.get("grad_accum")), "train.grad_accum 必须是正整数")
    tm = train.get("target_modules")
    need(isinstance(tm, list) and tm and all(isinstance(x, str) and x for x in tm), "train.target_modules 必须是非空的模块名列表")
    need(train.get("dtype", "auto") in DTYPES, f"train.dtype 只能是 {', '.join(DTYPES)}")

    mlc = cfg.get("mlc") or {}
    need(isinstance(mlc.get("compile", False), bool), "mlc.compile 必须是 true 或 false")
    need(isinstance(mlc.get("quantization"), str) and mlc.get("quantization"), "mlc.quantization 不能为空")
    need(isinstance(mlc.get("conv_template"), str) and mlc.get("conv_template"), "mlc.conv_template 不能为空")
    need(_pos_int(mlc.get("context_window_size")), "mlc.context_window_size 必须是正整数")
    need(str(mlc.get("model_lib_url", "")).startswith("https://"), "mlc.model_lib_url 必须是 https 地址")
    need(isinstance(mlc.get("wasm_path"), str) and mlc.get("wasm_path"), "mlc.wasm_path 不能为空")
    if mlc.get("wasm_url"):
        need(str(mlc["wasm_url"]).startswith("https://"), "mlc.wasm_url 必须是 https 地址")

    pack = cfg.get("pack") or {}
    need(bool(MODEL_ID_RE.match(str(pack.get("model_id", "")))), "pack.model_id 只能包含字母、数字、._-（最长 128 字符）")
    need(isinstance(pack.get("label"), str) and 0 < len(pack.get("label", "")) <= 200, "pack.label 不能为空，且不超过 200 字符")
    if pack.get("source_model"):
        need(str(pack["source_model"]).startswith("https://"), "pack.source_model 必须是 https 地址")
    need(pack.get("overrides", {}) is None or isinstance(pack.get("overrides", {}), dict), "pack.overrides 必须是对象")
    need(cfg.get("catalog_id") is None or isinstance(cfg.get("catalog_id"), str), "catalog_id 必须是字符串或空")
    return errs


def _pos(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def load_config(path=DEFAULT_CONFIG):
    if not os.path.isfile(path):
        raise ConfigError(f"找不到配置文件：{path}。请先运行 python3 training/ojtrain.py init")
    with open(path, "r", encoding="utf-8") as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as err:
            raise ConfigError(f"配置文件不是合法 JSON：{err}") from err
    errs = validate(cfg)
    if errs:
        raise ConfigError("配置有误：\n  - " + "\n  - ".join(errs))
    cfg["_root"] = os.path.dirname(os.path.abspath(path))
    cfg["_path"] = os.path.abspath(path)
    return cfg


def paths(cfg):
    """解析所有步骤用到的绝对路径。"""
    root = cfg["_root"]
    name = cfg["name"]
    work = os.path.join(root, "work", name)
    base = cfg["base"]
    if base.get("local_dir"):
        base_dir = os.path.join(root, base["local_dir"])
    else:
        base_dir = os.path.join(root, "models", base["hf_repo"].replace("/", "__"))
    model_id = cfg["pack"]["model_id"]
    return {
        "root": root,
        "work": work,
        "state": os.path.join(work, "state.json"),
        "base": os.path.normpath(base_dir),
        "data_raw": os.path.normpath(os.path.join(root, cfg["data"]["raw"])),
        "data_dir": os.path.join(work, "data"),
        "train_file": os.path.join(work, "data", "train.jsonl"),
        "val_file": os.path.join(work, "data", "val.jsonl"),
        "lora": os.path.join(work, "lora"),
        "merged": os.path.join(work, "merged"),
        "mlc": os.path.join(work, "mlc"),
        "wasm": os.path.normpath(os.path.join(root, cfg["mlc"]["wasm_path"])),
        "ojmodel": os.path.join(work, f"{model_id}.ojmodel"),
    }
