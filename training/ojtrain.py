#!/usr/bin/env python3
"""
训练一站式命令行入口。所有准备工作都在 training/ 目录中完成。

典型流程（在仓库根目录执行）：
  python3 training/ojtrain.py init                       # 生成 training/config.json
  python3 training/ojtrain.py setup                      # 创建 training/.venv 并安装依赖
  python3 training/ojtrain.py doctor                     # 检查 GPU、依赖、磁盘、网络
  python3 training/ojtrain.py import-data 我的数据.csv     # 导入训练数据（jsonl / json / csv）
  python3 training/ojtrain.py import-model 本地模型目录    # 或者让流水线自动下载基础模型
  python3 training/ojtrain.py run                        # 一键执行：获取模型 → 整理数据 → 训练 → 转换 → 打包
  python3 training/ojtrain.py status                     # 查看每一步的状态

中断后可以继续：run --from train；重新执行某一步：run --from train --force。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import catalog
import config as cfgmod
import doctor
import fetch_model
import pipeline as pl

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")
VENV = os.path.join(HERE, ".venv")


def _load(args):
    return cfgmod.load_config(args.config)


def cmd_init(args):
    if args.model and args.hf_repo:
        raise ValueError("--model 与 --hf-repo 只能二选一")
    custom = None
    if args.hf_repo:
        custom = {
            "hf_repo": args.hf_repo,
            "conv_template": args.conv_template,
            "context_window_size": args.context_window_size,
            "quantization": args.quantization,
        }
    path = cfgmod.init_config(args.config, name=args.name, force=args.force, model=args.model, custom=custom)
    cfg = cfgmod.load_config(path)
    print(f"已生成配置文件：{path}")
    print(f"基础模型：{cfg['base']['hf_repo']}")
    if cfg["mlc"]["compile"]:
        print("运行库：自定义架构，convert 步骤会在本机用 mlc_llm compile 编译。")
    else:
        print(f"运行库：沿用官方 {cfg['mlc']['wasm_path'].split('/')[-1]}，无需编译。")
    print("请确认 base 指向的模型可访问（gated 模型需设置 HF_TOKEN），再运行 run。")
    return 0


def cmd_models(args):
    models = catalog.load()
    if args.json:
        print(json.dumps(models, ensure_ascii=False, indent=2))
        return 0
    in_dd = sum(1 for m in models if m.get("in_dropdown", True))
    print(f"可训练的模型（共 {len(models)} 个：网页下拉框 {in_dd} 个，另有仅训练目录 {len(models) - in_dd} 个）：")
    print("\n".join(catalog.summary_lines(models)))
    print("\n用法：python3 training/ojtrain.py init --model <上面的 id>")
    print("清单之外的模型：init --hf-repo 组织/模型 --conv-template <模板名> --context-window-size 4096")
    return 0


def cmd_setup(args):
    if not os.path.isdir(VENV) or args.recreate:
        subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    py = os.path.join(VENV, "bin", "python")
    if os.name == "nt":
        py = os.path.join(VENV, "Scripts", "python.exe")
    subprocess.run([py, "-m", "pip", "install", "-r", REQUIREMENTS], check=True)
    print(f"依赖已安装到 {VENV}")
    print("之后请使用该环境运行：" + (f"{py} training/ojtrain.py run" if os.name != "nt" else f"{py} training\\ojtrain.py run"))
    return 0


def cmd_doctor(args):
    checks = doctor.run_checks(args.config, offline=args.offline)
    print(doctor.format_report(checks))
    return 1 if doctor.has_failure(checks) else 0


def cmd_import_data(args):
    cfg = _load(args)
    if not os.path.isfile(args.file):
        raise ValueError(f"找不到文件：{args.file}")
    ext = os.path.splitext(args.file)[1].lower()
    if ext not in (".jsonl", ".json", ".csv"):
        raise ValueError("只支持 .jsonl、.json、.csv 文件")
    dest_dir = os.path.join(cfg["_root"], "data")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(args.file))
    if os.path.abspath(args.file) != os.path.abspath(dest):
        shutil.copyfile(args.file, dest)
    cfg["data"]["raw"] = os.path.relpath(dest, cfg["_root"]).replace(os.sep, "/")
    cfgmod.write_config(cfg["_path"], cfg)
    print(f"已导入 {dest}，并把 data.raw 设置为 {cfg['data']['raw']}")
    return 0


def cmd_import_model(args):
    cfg = _load(args)
    src = os.path.abspath(args.dir)
    problems = fetch_model.check_model_dir(src)
    if problems:
        raise ValueError("模型目录不完整：" + "；".join(problems))
    rel = os.path.relpath(src, cfg["_root"])
    cfg["base"]["local_dir"] = src if rel.startswith("..") else rel.replace(os.sep, "/")
    cfgmod.write_config(cfg["_path"], cfg)
    print(f"已设置基础模型目录：{cfg['base']['local_dir']}")
    return 0


def _single(step):
    def run(args):
        cfg = _load(args)
        results = pl.Pipeline(cfg).run(steps=[step], force=args.force)
        return 1 if results.get(step) == "failed" else 0

    return run


def cmd_run(args):
    cfg = _load(args)
    p = pl.Pipeline(cfg)
    results = p.run(steps=p.plan(args.from_step), force=args.force)
    print("\n状态汇总：")
    for step in pl.STEPS:
        if step in results:
            print(f"  {step.ljust(8)} {results[step]}")
    if "failed" in results.values():
        return 1
    if "waiting" in results.values():
        print("\n流程已暂停。完成上面提示的手动步骤后，运行：python3 training/ojtrain.py run --from " + [s for s, v in results.items() if v == "waiting"][0])
    return 0


def cmd_status(args):
    cfg = _load(args)
    p = pl.Pipeline(cfg)
    state = p.load_state()
    print(f"配置：{cfg['_path']}")
    print(f"工作目录：{p.p['work']}")
    for step in pl.STEPS:
        info = state["steps"].get(step)
        mark = f"已完成（{info['at']}）" if info and info.get("done") else "未完成"
        print(f"  {step.ljust(8)} {mark}")
    print(f"模型文件：{p.p['ojmodel'] if os.path.isfile(p.p['ojmodel']) else '尚未生成'}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="ojtrain", description="训练一站式工具（LoRA 微调 → 打包 → 导入网页）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add(name, func, help_text, config=True):
        sp = sub.add_parser(name, help=help_text)
        if config:
            sp.add_argument("--config", default=cfgmod.DEFAULT_CONFIG, help="配置文件（默认 training/config.json）")
        sp.set_defaults(func=func)
        return sp

    sp = add("init", cmd_init, "生成配置 training/config.json（可直接选用网页下拉框中的模型）", config=False)
    sp.add_argument("--config", default=cfgmod.DEFAULT_CONFIG)
    sp.add_argument("--name", default=None, help="训练名称（同时作为默认模型 id）")
    sp.add_argument("--force", action="store_true", help="覆盖已有配置")
    sp.add_argument("--model", default=None, help="清单中的模型 id（见 ojtrain.py models）")
    sp.add_argument("--hf-repo", default=None, help="清单之外的模型：Hugging Face 仓库，例如 组织/模型名")
    sp.add_argument("--conv-template", default=None, help="自定义模型的对话模板名（需与 MLC-LLM 版本一致）")
    sp.add_argument("--context-window-size", type=int, default=4096, help="自定义模型的上下文长度")
    sp.add_argument("--quantization", default="q4f16_1", help="自定义模型的量化方式")

    sp = add("models", cmd_models, "列出可训练的模型（来自网页下拉框）", config=False)
    sp.add_argument("--json", action="store_true", help="以 JSON 输出")

    add("setup", cmd_setup, "创建 training/.venv 并安装依赖", config=False).add_argument("--recreate", action="store_true")
    add("doctor", cmd_doctor, "检查环境：GPU、依赖、磁盘、网络、配置").add_argument("--offline", action="store_true", help="跳过网络检查")
    sp = add("import-data", cmd_import_data, "导入训练数据（jsonl / json / csv）")
    sp.add_argument("file")
    sp = add("import-model", cmd_import_model, "使用本机已有的基础模型目录（不下载）")
    sp.add_argument("dir")

    add("fetch", _single("fetch"), "下载或确认基础模型").add_argument("--force", action="store_true")
    add("data", _single("data"), "整理数据，切分训练集和验证集").add_argument("--force", action="store_true")
    add("train", _single("train"), "LoRA 微调并合并").add_argument("--force", action="store_true")
    add("convert", _single("convert"), "用 mlc_llm 转换为 WebLLM 格式").add_argument("--force", action="store_true")
    add("pack", _single("pack"), "打包为 .ojmodel").add_argument("--force", action="store_true")

    sp = add("run", cmd_run, "一键执行全部步骤（可从某一步开始）")
    sp.add_argument("--from", dest="from_step", choices=pl.STEPS, default=None)
    sp.add_argument("--force", action="store_true", help="忽略已完成的记录，全部重新执行")

    add("status", cmd_status, "查看各步骤状态")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (cfgmod.ConfigError, pl.StepError, ValueError, RuntimeError, OSError) as err:
        print(f"错误：{err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
