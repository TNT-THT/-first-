import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfgmod  # noqa: E402


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_example_is_valid(self):
        self.assertEqual(cfgmod.validate(cfgmod.load_example()), [])

    def test_init_writes_and_refuses_overwrite(self):
        cfgmod.init_config(self.path, name="my-run")
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["name"], "my-run")
        self.assertEqual(data["pack"]["model_id"], "my-run-v1")
        with self.assertRaisesRegex(cfgmod.ConfigError, "已存在"):
            cfgmod.init_config(self.path)
        cfgmod.init_config(self.path, force=True)

    def test_load_adds_root_and_strips_internal_keys_on_write(self):
        cfgmod.init_config(self.path)
        cfg = cfgmod.load_config(self.path)
        self.assertEqual(cfg["_root"], self.tmp.name)
        cfgmod.write_config(self.path, cfg)
        with open(self.path, encoding="utf-8") as f:
            self.assertNotIn("_root", json.load(f))

    def test_missing_file_gives_hint(self):
        with self.assertRaisesRegex(cfgmod.ConfigError, "init"):
            cfgmod.load_config(self.path)

    def test_invalid_fields_are_all_reported(self):
        cfgmod.init_config(self.path)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        data["name"] = "bad name/x"
        data["train"]["lora_r"] = 0
        data["train"]["target_modules"] = []
        data["pack"]["model_id"] = "a/b"
        data["mlc"]["model_lib_url"] = "http://insecure"
        data["base"] = {"hf_repo": None, "local_dir": None}
        errs = cfgmod.validate(data)
        for needle in ("name", "lora_r", "target_modules", "model_id", "model_lib_url", "hf_repo 或 local_dir"):
            self.assertTrue(any(needle in e for e in errs), needle)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        with self.assertRaisesRegex(cfgmod.ConfigError, "配置有误"):
            cfgmod.load_config(self.path)

    def test_bool_is_not_a_number(self):
        cfgmod.init_config(self.path)
        cfg = cfgmod.load_config(self.path)
        cfg["train"]["epochs"] = True
        self.assertIn("train.epochs 必须大于 0", cfgmod.validate(cfg))

    def test_paths_are_relative_to_config_dir(self):
        cfgmod.init_config(self.path, name="r1")
        cfg = cfgmod.load_config(self.path)
        p = cfgmod.paths(cfg)
        root = self.tmp.name
        self.assertEqual(p["work"], os.path.join(root, "work", "r1"))
        self.assertEqual(p["base"], os.path.join(root, "models", "Qwen__Qwen2.5-Coder-1.5B-Instruct"))
        self.assertEqual(p["data_raw"], os.path.join(root, "data", "raw.jsonl"))
        self.assertTrue(p["ojmodel"].endswith("r1-v1.ojmodel"))

    def test_local_dir_overrides_hf_repo(self):
        cfgmod.init_config(self.path)
        cfg = cfgmod.load_config(self.path)
        cfg["base"]["local_dir"] = "/abs/model"
        self.assertEqual(cfgmod.paths(cfg)["base"], "/abs/model")


if __name__ == "__main__":
    unittest.main()
