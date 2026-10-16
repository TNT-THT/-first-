#!/usr/bin/env python3
"""
获取基础模型和运行库（wasm）到本地。

- fetch_hf：从 Hugging Face 下载模型（需要 huggingface_hub；公开模型无需登录）。
  如果需要访问私有或受限模型，请在本机设置环境变量 HF_TOKEN，不要把 token 写进文件或发给别人。
- import_local：导入本机已有的模型目录（只检查，不复制）。
- download_file：下载运行库（wasm），并检查文件头是 wasm 格式（避免把 404 页面当成 wasm 保存）。
"""
import glob
import os
import shutil
import urllib.request

WASM_MAGIC = b"\x00asm"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")
HF_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "merges.txt"]


def check_model_dir(path):
    """检查目录是否是可用的 HF 模型目录。返回问题列表，空列表表示通过。"""
    if not os.path.isdir(path):
        return [f"目录不存在：{path}"]
    problems = []
    if not os.path.isfile(os.path.join(path, "config.json")):
        problems.append("缺少 config.json")
    if not glob.glob(os.path.join(path, "*.safetensors")):
        problems.append("缺少权重文件（*.safetensors）")
    if not any(os.path.isfile(os.path.join(path, name)) for name in TOKENIZER_FILES):
        problems.append("缺少分词器文件（tokenizer.json 或 tokenizer.model）")
    return problems


def import_local(src, dest=None):
    """使用本机已有的模型目录。dest 为空时直接使用 src，不复制。"""
    problems = check_model_dir(src)
    if problems:
        raise ValueError(f"模型目录不完整：{'；'.join(problems)}")
    if dest is None or os.path.abspath(dest) == os.path.abspath(src):
        return os.path.abspath(src)
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return os.path.abspath(dest)


def fetch_hf(repo_id, dest, revision="main"):
    """从 Hugging Face 下载模型到 dest（已存在且完整时不会重复下载）。"""
    if not check_model_dir(dest):
        return os.path.abspath(dest)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise RuntimeError("缺少 huggingface_hub，请先运行 ojtrain.py setup") from err
    os.makedirs(dest, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=dest,
        allow_patterns=HF_ALLOW_PATTERNS,
    )
    problems = check_model_dir(dest)
    if problems:
        raise RuntimeError(f"下载后的模型目录不完整：{'；'.join(problems)}")
    return os.path.abspath(dest)


def download_file(url, dest, opener=urllib.request.urlopen, timeout=60):
    """下载文件到 dest。已存在且是 wasm 时跳过。返回绝对路径。"""
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        with open(dest, "rb") as f:
            if f.read(4) == WASM_MAGIC or not dest.endswith(".wasm"):
                return os.path.abspath(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    with opener(url, timeout=timeout) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    if dest.endswith(".wasm"):
        with open(tmp, "rb") as f:
            magic = f.read(4)
        if magic != WASM_MAGIC:
            os.remove(tmp)
            raise ValueError(f"下载的内容不是 wasm 文件（可能是错误页面）：{url}")
    os.replace(tmp, dest)
    return os.path.abspath(dest)
