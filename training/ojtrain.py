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
import os
import shutil
import subprocess
import sys

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
    path = cfgmod.init_config(args.config, name=args.name, force=args.force)
    print(f"已生成配置文件：{path}")
    print("请按需修改 base（基础模型）、data（数据）、pack（模型名称和显示名）后再运行 run。")
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

    sp = add("init", cmd_init, "生成默认配置 training/config.json", config=False)
    sp.add_argument("--config", default=cfgmod.DEFAULT_CONFIG)
    sp.add_argument("--name", default=None, help="训练名称（同时作为默认模型 id）")
    sp.add_argument("--force", action="store_true", help="覆盖已有配置")

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
