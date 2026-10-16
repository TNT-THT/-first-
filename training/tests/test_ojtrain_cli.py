import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ojtrain  # noqa: E402


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = ojtrain.main(argv)
    return code, out.getvalue()


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.cfg = os.path.join(self.root, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_init_status_and_refuse_overwrite(self):
        code, out = run(["init", "--config", self.cfg, "--name", "cli-run"])
        self.assertEqual(code, 0)
        self.assertIn("已生成配置文件", out)
        code, _ = run(["init", "--config", self.cfg])
        self.assertEqual(code, 1)
        code, out = run(["status", "--config", self.cfg])
        self.assertEqual(code, 0)
        self.assertIn("未完成", out)

    def test_import_data_copies_and_updates_config(self):
        run(["init", "--config", self.cfg])
        src = os.path.join(self.root, "mydata.csv")
        with open(src, "w", encoding="utf-8") as f:
            f.write("prompt,response\n问,答\n")
        code, out = run(["import-data", src, "--config", self.cfg])
        self.assertEqual(code, 0, out)
        self.assertTrue(os.path.isfile(os.path.join(self.root, "data", "mydata.csv")))
        with open(self.cfg, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["data"]["raw"], "data/mydata.csv")

    def test_import_data_rejects_other_types(self):
        run(["init", "--config", self.cfg])
        src = os.path.join(self.root, "x.xlsx")
        with open(src, "w") as f:
            f.write("x")
        code, _ = run(["import-data", src, "--config", self.cfg])
        self.assertEqual(code, 1)

    def test_import_model_sets_local_dir(self):
        run(["init", "--config", self.cfg])
        model = os.path.join(self.root, "my-base")
        os.makedirs(model)
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            with open(os.path.join(model, name), "w") as f:
                f.write("{}")
        code, out = run(["import-model", model, "--config", self.cfg])
        self.assertEqual(code, 0, out)
        with open(self.cfg, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["base"]["local_dir"], "my-base")

    def test_import_model_rejects_incomplete_dir(self):
        run(["init", "--config", self.cfg])
        code, _ = run(["import-model", self.root, "--config", self.cfg])
        self.assertEqual(code, 1)

    def test_doctor_reports_and_exit_code(self):
        code, out = run(["doctor", "--config", self.cfg, "--offline"])
        self.assertEqual(code, 1)  # 配置文件不存在 → FAIL
        self.assertIn("配置文件", out)
        run(["init", "--config", self.cfg])
        code, out = run(["doctor", "--config", self.cfg, "--offline"])
        self.assertEqual(code, 0, out)
        self.assertIn("基础模型", out)

    def test_run_from_convert_pauses_without_mlc(self):
        import shutil

        run(["init", "--config", self.cfg])
        original = shutil.which
        shutil.which = lambda name: None
        try:
            code, out = run(["run", "--from", "convert", "--config", self.cfg])
        finally:
            shutil.which = original
        self.assertEqual(code, 0)
        self.assertIn("流程已暂停", out)

    def test_missing_config_gives_error_code(self):
        code, _ = run(["run", "--config", os.path.join(self.root, "none.json")])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
