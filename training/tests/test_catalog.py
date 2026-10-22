import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import catalog  # noqa: E402
import config as cfgmod  # noqa: E402


class CatalogDataTest(unittest.TestCase):
    def setUp(self):
        self.models = catalog.load()

    def test_covers_all_dropdown_models(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "web", "public", "config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        ids = [m["id"] for m in cfg["webllm"]["models"]]
        in_dd = [m["id"] for m in self.models if m.get("in_dropdown", True)]
        self.assertEqual(sorted(ids), sorted(in_dd))
        self.assertEqual(len(in_dd), 32)  # 含 3 个第三方去审查模型
        self.assertEqual(len(self.models), 34)  # 另有 2 个仅训练目录的超大模型

    def test_catalog_only_models_are_not_in_dropdown(self):
        only = [m for m in self.models if not m.get("in_dropdown", True)]
        self.assertEqual(sorted(m["id"] for m in only), ["Qwen2.5-32B-Instruct", "Yi-1.5-34B-Chat"])
        for m in only:
            with self.subTest(model=m["id"]):
                self.assertIsNone(m["model_lib_url"])
                self.assertEqual(m["size_class"], "超大")
                self.assertEqual(m["inference_vram_source"], "estimate")

    def test_every_entry_is_complete(self):
        for m in self.models:
            with self.subTest(model=m["id"]):
                self.assertTrue(m["hf_repo"].count("/") == 1)
                self.assertIn(m["size_class"], ("超大", "极小", "小", "中", "大"))
                if m.get("in_dropdown", True):
                    self.assertTrue(m["model_lib_url"].startswith("https://"))
                    self.assertTrue(m["wasm_file"].endswith(".wasm"))
                self.assertRegex(m["quantization"], r"^q[34]f(16|32)|^q3f16")  # 70B 使用 q3f16 以适应显存
                self.assertTrue(m["conv_template"])
                self.assertFalse(m["verified"])  # 原始仓库与模板名尚未核对

    def test_labels_parse_reasoning_and_size(self):
        by_id = {m["id"]: m for m in self.models}
        self.assertTrue(by_id["Qwen3-8B-q4f16_1-MLC"]["reasoning"])
        self.assertFalse(by_id["Qwen2.5-3B-Instruct-q4f16_1-MLC"]["reasoning"])
        self.assertEqual(by_id["Llama-2-13b-chat-hf-q4f16_1-MLC"]["size_class"], "大")
        self.assertEqual(by_id["gemma-2-9b-it-q4f32_1-MLC"]["size_class"], "大")
        self.assertEqual(by_id["Llama-3.1-70B-Instruct-q3f16_1-MLC"]["size_class"], "超大")
        self.assertEqual(by_id["Llama-3.1-70B-Instruct-q3f16_1-MLC"]["inference_vram_gb"], 30.4)
        self.assertEqual(by_id["Qwen2.5-0.5B-Instruct-q4f16_1-MLC"]["size_class"], "极小")
        self.assertTrue(by_id["Qwen3.5-9B-q4f16_1-MLC"]["reasoning"])
        self.assertEqual(by_id["Qwen3-8B-q4f32_1-MLC"]["quantization"], "q4f32_1")
        self.assertEqual(len([m for m in self.models if m.get("in_dropdown", True) and m["size_class"] == "超大"]), 2)

    def test_gated_models_are_flagged(self):
        by_id = {m["id"]: m for m in self.models}
        self.assertTrue(by_id["gemma-2-9b-it-q4f16_1-MLC"]["hf_gated"])
        self.assertTrue(by_id["gemma3-1b-it-q4f16_1-MLC"]["hf_gated"])
        self.assertFalse(by_id["Qwen2.5-3B-Instruct-q4f16_1-MLC"]["hf_gated"])

    def test_find_unknown_model_gives_hint(self):
        with self.assertRaisesRegex(catalog.CatalogError, "ojtrain.py models"):
            catalog.find("no-such-model")


class ApplyCatalogTest(unittest.TestCase):
    def test_catalog_model_reuses_official_library(self):
        entry = catalog.find("Qwen2.5-3B-Instruct-q4f16_1-MLC")
        cfg = catalog.apply_catalog(cfgmod.load_example(), entry)
        self.assertEqual(cfg["name"], "Qwen2.5-3B-Instruct-tuned")
        self.assertEqual(cfg["pack"]["model_id"], "Qwen2.5-3B-Instruct-tuned-v1")
        self.assertEqual(cfg["base"]["hf_repo"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertFalse(cfg["mlc"]["compile"])
        self.assertEqual(cfg["mlc"]["model_lib_url"], entry["model_lib_url"])
        self.assertEqual(cfg["mlc"]["wasm_url"], entry["model_lib_url"])
        self.assertEqual(cfg["mlc"]["wasm_path"], "libs/" + entry["wasm_file"])
        self.assertEqual(cfg["mlc"]["context_window_size"], 4096)
        self.assertEqual(cfg["pack"]["source_model"], entry["source_model"])
        self.assertEqual(cfg["catalog_id"], entry["id"])
        self.assertEqual(cfgmod.validate(cfg), [])

    def test_overrides_are_carried_into_pack(self):
        entry = catalog.find("Qwen2.5-3B-Instruct-q4f16_1-MLC")
        cfg = catalog.apply_catalog(cfgmod.load_example(), entry)
        self.assertEqual(cfg["pack"]["overrides"], {"context_window_size": 4096})
        self.assertEqual(cfgmod.validate(cfg), [])

    def test_every_catalog_model_gives_valid_config(self):
        for entry in catalog.load():
            if not entry.get("in_dropdown", True):
                continue
            with self.subTest(model=entry["id"]):
                cfg = catalog.apply_catalog(cfgmod.load_example(), entry)
                self.assertEqual(cfgmod.validate(cfg), [])

    def test_catalog_only_model_refuses_official_library(self):
        entry = catalog.find("Qwen2.5-32B-Instruct")
        with self.assertRaisesRegex(catalog.CatalogError, "--hf-repo Qwen/Qwen2.5-32B-Instruct"):
            catalog.apply_catalog(cfgmod.load_example(), entry)

    def test_custom_model_compiles_locally(self):
        cfg = catalog.apply_custom(
            cfgmod.load_example(), hf_repo="acme/Tiny-Coder", conv_template="chatml", context_window_size=2048,
        )
        self.assertEqual(cfg["name"], "Tiny-Coder-tuned")
        self.assertTrue(cfg["mlc"]["compile"])
        self.assertTrue(cfg["mlc"]["model_lib_url"].startswith(catalog.LOCAL_HOST))
        self.assertEqual(cfg["mlc"]["wasm_path"], "libs/Tiny-Coder-tuned-v1.wasm")
        self.assertIsNone(cfg["mlc"]["wasm_url"])
        self.assertEqual(cfg["pack"]["source_model"], catalog.LOCAL_HOST + "/models/Tiny-Coder-tuned-v1/")
        self.assertEqual(cfgmod.validate(cfg), [])

    def test_custom_model_requires_template_and_valid_repo(self):
        with self.assertRaisesRegex(catalog.CatalogError, "conv-template"):
            catalog.apply_custom(cfgmod.load_example(), hf_repo="a/b", conv_template="", context_window_size=2048)
        with self.assertRaisesRegex(catalog.CatalogError, "组织/模型名"):
            catalog.apply_custom(cfgmod.load_example(), hf_repo="just-a-name", conv_template="chatml", context_window_size=2048)
        with self.assertRaisesRegex(catalog.CatalogError, "q4f16"):
            catalog.apply_custom(cfgmod.load_example(), hf_repo="a/b", conv_template="chatml", context_window_size=2048, quantization="fp32")


class InitWithModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_model_writes_valid_config(self):
        cfgmod.init_config(self.path, model="Phi-4-mini-instruct-q4f16_1-MLC")
        cfg = cfgmod.load_config(self.path)
        self.assertEqual(cfg["base"]["hf_repo"], "microsoft/Phi-4-mini-instruct")

    def test_init_custom_writes_compile_mode(self):
        cfgmod.init_config(self.path, custom={"hf_repo": "acme/x-1b", "conv_template": "chatml",
                                              "context_window_size": 2048, "quantization": "q4f16_1"})
        cfg = cfgmod.load_config(self.path)
        self.assertTrue(cfg["mlc"]["compile"])
        self.assertEqual(cfg["name"], "x-1b-tuned")


if __name__ == "__main__":
    unittest.main()
