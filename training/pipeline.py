#!/usr/bin/env python3
"""
一站式流水线：doctor → fetch → data → train → convert → pack。

- 每一步完成后记录到 work/<name>/state.json，中断后可以从断点继续。
- convert 需要 mlc_llm；没有安装时该步骤会“暂停”（状态 waiting），并打印需要执行的命令，
  不会把整个流程标记为失败。
- 所有外部命令都通过 runner 执行，便于测试。
"""
import datetime
import json
import os
import shutil
import subprocess
import sys

import config as cfgmod
import doctor
import fetch_model
import pack_ojmodel
from prepare_dataset import build_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ("doctor", "fetch", "data", "train", "convert", "pack")


class StepError(RuntimeError):
    pass


class StepPending(Exception):
    """步骤需要人工操作（例如安装 mlc_llm），流程暂停但不算失败。"""


class Pipeline:
    def __init__(self, cfg, runner=subprocess.run, log=print, python=sys.executable):
        self.cfg = cfg
        self.p = cfgmod.paths(cfg)
        self.runner = runner
        self.log = log
        self.python = python

    # ---------- 状态 ----------
    def load_state(self):
        if not os.path.isfile(self.p["state"]):
            return {"steps": {}}
        with open(self.p["state"], "r", encoding="utf-8") as f:
            return json.load(f)

    def save_state(self, state):
        os.makedirs(self.p["work"], exist_ok=True)
        with open(self.p["state"], "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def is_done(self, step):
        return bool(self.load_state()["steps"].get(step, {}).get("done"))

    def mark_done(self, step):
        state = self.load_state()
        state["steps"][step] = {"done": True, "at": datetime.datetime.now().isoformat(timespec="seconds")}
        self.save_state(state)

    def clear(self, step):
        state = self.load_state()
        state["steps"].pop(step, None)
        self.save_state(state)

    # ---------- 运行 ----------
    def plan(self, from_step=None):
        if from_step is None:
            return list(STEPS)
        if from_step not in STEPS:
            raise StepError(f"未知步骤：{from_step}（可选：{', '.join(STEPS)}）")
        return list(STEPS[STEPS.index(from_step):])

    def run(self, steps=None, force=False):
        """按顺序执行步骤。返回 {步骤: 状态}，状态为 done / skipped / waiting / failed。"""
        results = {}
        for step in steps or STEPS:
            if self.is_done(step) and not force:
                self.log(f"[跳过] {step}：已完成（使用 --force 可重新执行）")
                results[step] = "skipped"
                continue
            if force:
                self.clear(step)
            self.log(f"[开始] {step}")
            try:
                getattr(self, f"step_{step}")()
            except StepPending as err:
                self.log(f"[暂停] {step}：{err}")
                results[step] = "waiting"
                break
            except (StepError, RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as err:
                self.log(f"[失败] {step}：{err}")
                results[step] = "failed"
                break
            self.mark_done(step)
            self.log(f"[完成] {step}")
            results[step] = "done"
        return results

    # ---------- 各步骤 ----------
    def step_doctor(self):
        checks = doctor.run_checks(self.cfg["_path"], offline=True)
        self.log(doctor.format_report(checks))
        if doctor.has_failure(checks):
            raise StepError("环境检查有失败项，请先按上面的提示修复")

    def step_fetch(self):
        base = self.cfg["base"]
        if base.get("local_dir"):
            fetch_model.import_local(self.p["base"])
            self.log(f"使用本地模型：{self.p['base']}")
            return
        fetch_model.fetch_hf(base["hf_repo"], self.p["base"], revision=base.get("revision", "main"))
        self.log(f"模型已就绪：{self.p['base']}")

    def step_data(self):
        d = self.cfg["data"]
        info = build_dataset(
            self.p["data_raw"],
            self.p["data_dir"],
            val_ratio=d.get("val_ratio", 0.0),
            max_chars=d.get("max_chars", 8000),
            seed=d.get("seed", 42),
        )
        self.log(f"训练集 {info['train']} 条，验证集 {info['val']} 条，丢弃 {sum(info['skipped'].values())} 条")

    def train_command(self):
        t = self.cfg["train"]
        cmd = [
            self.python, os.path.join(HERE, "train_lora.py"),
            "--base", self.p["base"],
            "--data", self.p["train_file"],
            "--output-dir", self.p["lora"],
            "--merged-dir", self.p["merged"],
            "--epochs", str(t["epochs"]),
            "--learning-rate", str(t["learning_rate"]),
            "--lora-r", str(t["lora_r"]),
            "--lora-alpha", str(t["lora_alpha"]),
            "--lora-dropout", str(t.get("lora_dropout", 0.05)),
            "--target-modules", ",".join(t["target_modules"]),
            "--max-length", str(t["max_length"]),
            "--batch-size", str(t["batch_size"]),
            "--grad-accum", str(t["grad_accum"]),
            "--dtype", t.get("dtype", "auto"),
            "--seed", str(self.cfg["data"].get("seed", 42)),
        ]
        if os.path.isfile(self.p["val_file"]):
            cmd += ["--val-data", self.p["val_file"]]
        if t.get("grad_checkpointing"):
            cmd.append("--grad-checkpointing")
        return cmd

    def step_train(self):
        self.runner(self.train_command(), check=True)

    def convert_commands(self):
        m = self.cfg["mlc"]
        out = self.p["mlc"]
        return [
            ["mlc_llm", "convert_weight", self.p["merged"], "--quantization", m["quantization"], "-o", out],
            ["mlc_llm", "gen_config", self.p["merged"], "--quantization", m["quantization"],
             "--conv-template", m["conv_template"], "--context-window-size", str(m["context_window_size"]), "-o", out],
        ]

    def step_convert(self):
        out = self.p["mlc"]
        if self._mlc_output_ready():
            self.log(f"已找到转换结果：{out}")
            return
        if not shutil.which("mlc_llm"):
            cmds = "\n".join("  " + " ".join(c) for c in self.convert_commands())
            raise StepPending(
                "未找到 mlc_llm。请先安装 MLC-LLM 并执行下面的命令；"
                "也可以把别处转换好的结果放到 " + out + "。完成后运行：\n"
                "  python3 training/ojtrain.py run --from convert\n" + cmds
            )
        for cmd in self.convert_commands():
            self.runner(cmd, check=True)
        if not self._mlc_output_ready():
            raise StepError("转换输出中缺少 tensor-cache.json 或 mlc-chat-config.json；请核对 MLC 版本的输出文件名")

    def _mlc_output_ready(self):
        out = self.p["mlc"]
        return os.path.isfile(os.path.join(out, "tensor-cache.json")) and os.path.isfile(
            os.path.join(out, "mlc-chat-config.json")
        )

    def step_pack(self):
        m = self.cfg["mlc"]
        pk = self.cfg["pack"]
        if not os.path.isfile(self.p["wasm"]):
            if not m.get("wasm_url"):
                raise StepError(f"找不到运行库 {self.p['wasm']}，且没有设置 wasm_url")
            self.log(f"下载运行库：{m['wasm_url']}")
            fetch_model.download_file(m["wasm_url"], self.p["wasm"])
        source = pk.get("source_model") or f"https://huggingface.co/oj-ai-training/{pk['model_id']}"
        record_extra = {"overrides": {"context_window_size": m["context_window_size"]}}
        if pk.get("vram_required_MB"):
            record_extra["vram_required_MB"] = pk["vram_required_MB"]
        result = pack_ojmodel.pack(
            self.p["mlc"],
            self.p["ojmodel"],
            model_id=pk["model_id"],
            label=pk["label"],
            model_lib_url=m["model_lib_url"],
            wasm_path=self.p["wasm"],
            source_model=source,
            record_extra=record_extra,
        )
        self.log(f"已生成 {self.p['ojmodel']}（{result['entries']} 个文件，{result['bytes'] / 1e9:.2f} GB）")
        self.log("下一步：在网页中点“更改权限”升级为管理员，然后点“导入训练结果”选择该文件。")
