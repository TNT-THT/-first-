#!/usr/bin/env python3
"""
训练前环境检查：Python、依赖、GPU、磁盘空间、网络、MLC 工具、配置与文件是否齐全。

每一项返回 OK / WARN / FAIL：
  - FAIL：必须先解决，否则无法训练（例如 Python 版本过低、配置文件有误）。
  - WARN：可以继续，但某些步骤会失败或很慢（例如没有 GPU、缺少 mlc_llm）。
"""
import importlib.util
import os
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

import config as cfgmod
import fetch_model

OK, WARN, FAIL = "OK", "WARN", "FAIL"
REQUIRED_PACKAGES = ("torch", "transformers", "peft", "accelerate", "safetensors", "tokenizers", "huggingface_hub")


@dataclass
class Check:
    name: str
    status: str
    detail: str


def check_python(min_version=(3, 10)):
    cur = sys.version_info[:3]
    text = f"Python {cur[0]}.{cur[1]}.{cur[2]}"
    if cur[:2] < min_version:
        return Check("Python", FAIL, f"{text}，需要 {min_version[0]}.{min_version[1]} 或更高版本")
    return Check("Python", OK, text)


def check_packages(names=REQUIRED_PACKAGES):
    missing = [n for n in names if importlib.util.find_spec(n) is None]
    if missing:
        return Check("Python 依赖", WARN, f"缺少：{', '.join(missing)}（运行 ojtrain.py setup 安装）")
    return Check("Python 依赖", OK, "已安装全部训练依赖")


def check_gpu():
    if importlib.util.find_spec("torch") is None:
        return Check("GPU", WARN, "未安装 PyTorch，无法检测显卡")
    import torch

    if not torch.cuda.is_available():
        return Check("GPU", WARN, "未检测到 CUDA 显卡，训练会非常慢或无法进行")
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 1024**3
    detail = f"{props.name}，显存 {gb:.1f} GB"
    if gb < 8:
        return Check("GPU", WARN, detail + "（1.5B 模型建议 8 GB 以上）")
    return Check("GPU", OK, detail)


def check_disk(path, need_gb=30):
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free / 1024**3
    detail = f"可用 {free:.1f} GB（建议至少 {need_gb} GB：基础模型、合并模型、MLC 输出各占一份）"
    return Check("磁盘空间", OK if free >= need_gb else WARN, detail)


def check_network(url="https://huggingface.co", timeout=5, opener=urllib.request.urlopen):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with opener(req, timeout=timeout):
            pass
        return Check("网络（Hugging Face）", OK, "可以访问")
    except (urllib.error.URLError, OSError, ValueError) as err:
        return Check("网络（Hugging Face）", WARN, f"无法访问 {url}（{err}）。可改用 import-model 导入本地模型目录")


def check_mlc():
    if shutil.which("mlc_llm"):
        return Check("MLC 转换工具", OK, shutil.which("mlc_llm"))
    return Check("MLC 转换工具", WARN, "未找到 mlc_llm 命令（转换步骤需要，见 training/README.md）")


def check_config(path):
    """返回 (Check, cfg)；配置无效时 cfg 为 None。"""
    try:
        cfg = cfgmod.load_config(path)
    except cfgmod.ConfigError as err:
        return Check("配置文件", FAIL, str(err).replace("\n", " ")), None
    return Check("配置文件", OK, path), cfg


def check_project_files(cfg):
    p = cfgmod.paths(cfg)
    checks = []
    problems = fetch_model.check_model_dir(p["base"])
    if problems:
        checks.append(Check("基础模型", WARN, f"尚未就绪（{'；'.join(problems)}）。运行 fetch 或 import-model"))
    else:
        checks.append(Check("基础模型", OK, p["base"]))
    if os.path.isfile(p["data_raw"]):
        checks.append(Check("训练数据", OK, p["data_raw"]))
    else:
        checks.append(Check("训练数据", WARN, f"找不到 {p['data_raw']}。运行 import-data 导入"))
    if os.path.isfile(p["wasm"]):
        checks.append(Check("运行库（wasm）", OK, p["wasm"]))
    elif cfg["mlc"].get("compile"):
        checks.append(Check("运行库（wasm）", WARN, "尚未编译，convert 步骤会用 mlc_llm compile 生成（需要 MLC-LLM）"))
    elif cfg["mlc"].get("wasm_url"):
        checks.append(Check("运行库（wasm）", WARN, "尚未下载，打包时会自动从 wasm_url 下载"))
    else:
        checks.append(Check("运行库（wasm）", FAIL, f"找不到 {p['wasm']}，且没有设置 wasm_url"))
    return checks


def run_checks(config_path=cfgmod.DEFAULT_CONFIG, offline=False, opener=urllib.request.urlopen):
    checks = [check_python(), check_packages(), check_gpu(), check_disk(cfgmod.HERE)]
    if not offline:
        checks.append(check_network(opener=opener))
    checks.append(check_mlc())
    check, cfg = check_config(config_path)
    checks.append(check)
    if cfg is not None:
        checks.extend(check_project_files(cfg))
    return checks


def format_report(checks):
    width = max(len(c.name) for c in checks)
    lines = [f"[{c.status:4}] {c.name.ljust(width)}  {c.detail}" for c in checks]
    counts = {s: sum(1 for c in checks if c.status == s) for s in (OK, WARN, FAIL)}
    lines.append(f"合计：{counts[OK]} 项通过，{counts[WARN]} 项警告，{counts[FAIL]} 项失败")
    return "\n".join(lines)


def has_failure(checks):
    return any(c.status == FAIL for c in checks)


if __name__ == "__main__":
    checks = run_checks()
    print(format_report(checks))
    sys.exit(1 if has_failure(checks) else 0)
