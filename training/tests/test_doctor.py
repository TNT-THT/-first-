import os
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config as cfgmod  # noqa: E402
import doctor  # noqa: E402


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_python_version_gate(self):
        self.assertEqual(doctor.check_python((3, 10)).status, doctor.OK)
        self.assertEqual(doctor.check_python((99, 0)).status, doctor.FAIL)

    def test_missing_packages_is_warning(self):
        c = doctor.check_packages(("definitely_not_installed_pkg_xyz",))
        self.assertEqual(c.status, doctor.WARN)
        self.assertIn("definitely_not_installed_pkg_xyz", c.detail)

    def test_disk_check_uses_existing_ancestor(self):
        c = doctor.check_disk(os.path.join(self.tmp.name, "does", "not", "exist"), need_gb=0)
        self.assertEqual(c.status, doctor.OK)

    def test_network_ok_and_failure(self):
        ok = doctor.check_network(opener=lambda req, timeout=5: open(os.devnull))
        self.assertEqual(ok.status, doctor.OK)

        def boom(req, timeout=5):
            raise urllib.error.URLError("no route")

        bad = doctor.check_network(opener=boom)
        self.assertEqual(bad.status, doctor.WARN)
        self.assertIn("import-model", bad.detail)

    def test_missing_config_is_failure_and_skips_project_checks(self):
        checks = doctor.run_checks(os.path.join(self.tmp.name, "missing.json"), offline=True)
        cfg_check = [c for c in checks if c.name == "配置文件"][0]
        self.assertEqual(cfg_check.status, doctor.FAIL)
        self.assertTrue(doctor.has_failure(checks))
        self.assertFalse(any(c.name == "基础模型" for c in checks))

    def test_project_checks_with_valid_config(self):
        path = os.path.join(self.tmp.name, "config.json")
        cfgmod.init_config(path)
        checks = doctor.run_checks(path, offline=True)
        by_name = {c.name: c for c in checks}
        self.assertEqual(by_name["配置文件"].status, doctor.OK)
        self.assertEqual(by_name["基础模型"].status, doctor.WARN)  # 尚未下载
        self.assertEqual(by_name["运行库（wasm）"].status, doctor.WARN)  # 有 wasm_url，可自动下载
        self.assertNotIn("网络（Hugging Face）", by_name)

    def test_wasm_without_url_is_failure(self):
        path = os.path.join(self.tmp.name, "config.json")
        cfgmod.init_config(path)
        cfg = cfgmod.load_config(path)
        cfg["mlc"]["wasm_url"] = None
        checks = doctor.check_project_files(cfg)
        self.assertEqual({c.name: c.status for c in checks}["运行库（wasm）"], doctor.FAIL)

    def test_check_config_always_returns_pair(self):
        missing = doctor.check_config(os.path.join(self.tmp.name, "nope.json"))
        self.assertIsInstance(missing, tuple)
        self.assertEqual(missing[0].status, doctor.FAIL)
        self.assertIsNone(missing[1])
        path = os.path.join(self.tmp.name, "config.json")
        cfgmod.init_config(path)
        ok = doctor.check_config(path)
        self.assertEqual(ok[0].status, doctor.OK)
        self.assertEqual(ok[1]["name"], "coder15-tuned")

    def test_compile_mode_wasm_is_warning_not_failure(self):
        path = os.path.join(self.tmp.name, "config.json")
        cfgmod.init_config(path, custom={"hf_repo": "acme/x-1b", "conv_template": "chatml",
                                         "context_window_size": 2048, "quantization": "q4f16_1"})
        cfg = cfgmod.load_config(path)
        by_name = {c.name: c.status for c in doctor.check_project_files(cfg)}
        self.assertEqual(by_name["运行库（wasm）"], doctor.WARN)

    def test_report_counts(self):
        checks = [doctor.Check("a", doctor.OK, "x"), doctor.Check("b", doctor.WARN, "y"), doctor.Check("c", doctor.FAIL, "z")]
        report = doctor.format_report(checks)
        self.assertIn("1 项通过，1 项警告，1 项失败", report)
        self.assertTrue(doctor.has_failure(checks))
        self.assertFalse(doctor.has_failure(checks[:2]))


if __name__ == "__main__":
    unittest.main()
