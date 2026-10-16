import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfgmod  # noqa: E402
import pipeline as pl  # noqa: E402

WASM_URL = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Demo.wasm"


def make_model(path):
    os.makedirs(path, exist_ok=True)
    for name, content in (("config.json", "{}"), ("tokenizer.json", "{}")):
        with open(os.path.join(path, name), "w") as f:
            f.write(content)
    with open(os.path.join(path, "model.safetensors"), "wb") as f:
        f.write(b"0" * 16)
    return path


def make_mlc(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "tensor-cache.json"), "w") as f:
        json.dump({"records": [{"dataPath": "params_shard_0.bin"}]}, f)
    with open(os.path.join(path, "params_shard_0.bin"), "wb") as f:
        f.write(os.urandom(256))
    for name in ("tokenizer.json", "mlc-chat-config.json"):
        with open(os.path.join(path, name), "w") as f:
            f.write("{}")
    return path


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.logs = []
        self.calls = []
        self.path = os.path.join(self.root, "config.json")
        cfgmod.init_config(self.path, name="t1")
        self.cfg = cfgmod.load_config(self.path)
        self.cfg["base"]["local_dir"] = make_model(os.path.join(self.root, "base"))
        self.cfg["mlc"]["wasm_path"] = "libs/m.wasm"
        os.makedirs(os.path.join(self.root, "libs"))
        with open(os.path.join(self.root, "libs", "m.wasm"), "wb") as f:
            f.write(b"\x00asm\x01\x00\x00\x00")
        raw_dir = os.path.join(self.root, "data")
        os.makedirs(raw_dir)
        with open(os.path.join(raw_dir, "raw.jsonl"), "w", encoding="utf-8") as f:
            for i in range(30):
                f.write(json.dumps({"prompt": f"问{i}", "response": f"答{i}"}, ensure_ascii=False) + "\n")
        self.cfg["data"]["val_ratio"] = 0.1
        self.cfg["train"]["grad_checkpointing"] = True

    def tearDown(self):
        self.tmp.cleanup()

    def pipe(self, runner=None):
        def runner_default(cmd, check=True):
            self.calls.append(cmd)

        return pl.Pipeline(self.cfg, runner=runner or runner_default, log=self.logs.append, python="PY")

    # ---------- 计划 ----------
    def test_plan(self):
        p = self.pipe()
        self.assertEqual(p.plan(), list(pl.STEPS))
        self.assertEqual(p.plan("train"), ["train", "convert", "pack"])
        with self.assertRaisesRegex(pl.StepError, "未知步骤"):
            p.plan("bogus")

    # ---------- 各步骤 ----------
    def test_data_then_train_command(self):
        p = self.pipe()
        self.assertEqual(p.run(["data"])["data"], "done")
        self.assertTrue(os.path.isfile(p.p["train_file"]))
        self.assertTrue(os.path.isfile(p.p["val_file"]))
        cmd = p.train_command()
        self.assertEqual(cmd[0], "PY")
        self.assertIn("--val-data", cmd)
        self.assertIn("--grad-checkpointing", cmd)
        self.assertEqual(cmd[cmd.index("--target-modules") + 1], "q_proj,k_proj,v_proj,o_proj")
        self.assertEqual(cmd[cmd.index("--base") + 1], self.cfg["base"]["local_dir"])

    def test_train_runs_through_runner_and_skips_second_time(self):
        p = self.pipe()
        p.run(["data"])
        self.assertEqual(p.run(["train"])["train"], "done")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(p.run(["train"])["train"], "skipped")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(p.run(["train"], force=True)["train"], "done")
        self.assertEqual(len(self.calls), 2)

    def test_convert_waits_without_mlc(self):
        p = self.pipe()
        original = pl.shutil.which
        pl.shutil.which = lambda name: None
        try:
            results = p.run(["convert"])
        finally:
            pl.shutil.which = original
        self.assertEqual(results["convert"], "waiting")
        self.assertTrue(any("mlc_llm convert_weight" in line for line in self.logs))
        self.assertFalse(p.is_done("convert"))

    def test_convert_with_mlc_writes_then_checks_output(self):
        p = self.pipe()
        original = pl.shutil.which
        pl.shutil.which = lambda name: "/usr/bin/mlc_llm"

        def fake_runner(cmd, check=True):
            self.calls.append(cmd)
            if cmd[1] == "convert_weight":
                make_mlc(p.p["mlc"])

        p.runner = fake_runner
        try:
            self.assertEqual(p.run(["convert"])["convert"], "done")
        finally:
            pl.shutil.which = original
        self.assertEqual([c[1] for c in self.calls], ["convert_weight", "gen_config"])

    def test_convert_detects_missing_output(self):
        p = self.pipe()
        original = pl.shutil.which
        pl.shutil.which = lambda name: "/usr/bin/mlc_llm"
        try:
            self.assertEqual(p.run(["convert"])["convert"], "failed")
        finally:
            pl.shutil.which = original

    def test_convert_accepts_existing_output_without_mlc(self):
        p = self.pipe()
        make_mlc(p.p["mlc"])
        original = pl.shutil.which
        pl.shutil.which = lambda name: None
        try:
            self.assertEqual(p.run(["convert"])["convert"], "done")
        finally:
            pl.shutil.which = original
        self.assertEqual(self.calls, [])

    def test_pack_produces_importable_header(self):
        p = self.pipe()
        make_mlc(p.p["mlc"])
        self.assertEqual(p.run(["pack"])["pack"], "done")
        with open(p.p["ojmodel"], "rb") as f:
            data = f.read()
        self.assertEqual(data[:4], b"OJMD")
        (hlen,) = struct.unpack("<I", data[4:8])
        header = json.loads(data[8 : 8 + hlen])
        self.assertEqual(header["modelId"], "t1-v1")
        self.assertEqual(header["label"], self.cfg["pack"]["label"])
        self.assertEqual(header["source"]["model_lib"], self.cfg["mlc"]["model_lib_url"])
        self.assertEqual(header["source"]["overrides"], {"context_window_size": 4096})
        self.assertEqual(header["source"]["vram_required_MB"], self.cfg["pack"]["vram_required_MB"])

    def test_pack_downloads_missing_wasm(self):
        p = self.pipe()
        make_mlc(p.p["mlc"])
        os.remove(p.p["wasm"])
        src = os.path.join(self.root, "remote.wasm")
        with open(src, "wb") as f:
            f.write(b"\x00asm\x01\x00\x00\x00")
        import fetch_model

        original = fetch_model.download_file
        fetch_model.download_file = lambda url, dest, **kw: original(
            "file://" + src, dest, opener=lambda u, timeout=60: open(src, "rb")
        )
        try:
            self.assertEqual(p.run(["pack"])["pack"], "done")
        finally:
            fetch_model.download_file = original
        self.assertTrue(os.path.isfile(p.p["wasm"]))

    def test_failure_stops_the_pipeline(self):
        self.cfg["base"]["local_dir"] = os.path.join(self.root, "missing-model")
        p = self.pipe()
        results = p.run(["fetch", "data"])
        self.assertEqual(results, {"fetch": "failed"})
        self.assertFalse(os.path.exists(p.p["train_file"]))

    def test_doctor_step_passes_with_valid_config(self):
        p = self.pipe()
        self.assertEqual(p.run(["doctor"])["doctor"], "done")

    def test_state_file_records_steps(self):
        p = self.pipe()
        p.run(["data"])
        with open(p.p["state"], encoding="utf-8") as f:
            state = json.load(f)
        self.assertTrue(state["steps"]["data"]["done"])
        self.assertIn("at", state["steps"]["data"])

    def test_full_run_stops_at_waiting_convert(self):
        p = self.pipe()
        original = pl.shutil.which
        pl.shutil.which = lambda name: None
        try:
            results = p.run(p.plan())
        finally:
            pl.shutil.which = original
        self.assertEqual(results["data"], "done")
        self.assertEqual(results["train"], "done")
        self.assertEqual(results["convert"], "waiting")
        self.assertNotIn("pack", results)


if __name__ == "__main__":
    unittest.main()
