import io
import os
import sys
import tempfile
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fetch_model as fm  # noqa: E402


def make_model(path, tokenizer="tokenizer.json", weights=True, config=True):
    os.makedirs(path, exist_ok=True)
    if config:
        with open(os.path.join(path, "config.json"), "w") as f:
            f.write("{}")
    if weights:
        with open(os.path.join(path, "model.safetensors"), "wb") as f:
            f.write(b"0" * 16)
    if tokenizer:
        with open(os.path.join(path, tokenizer), "w") as f:
            f.write("{}")
    return path


class FetchModelTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_check_model_dir_reports_each_problem(self):
        self.assertEqual(fm.check_model_dir(os.path.join(self.root, "nope")), [f"目录不存在：{os.path.join(self.root, 'nope')}"])
        d = make_model(os.path.join(self.root, "m1"), tokenizer=None, weights=False, config=False)
        problems = fm.check_model_dir(d)
        self.assertEqual(len(problems), 3)
        self.assertEqual(fm.check_model_dir(make_model(os.path.join(self.root, "m2"))), [])
        self.assertEqual(fm.check_model_dir(make_model(os.path.join(self.root, "m3"), tokenizer="tokenizer.model")), [])

    def test_import_local_without_copy(self):
        src = make_model(os.path.join(self.root, "src"))
        self.assertEqual(fm.import_local(src), os.path.abspath(src))
        self.assertFalse(os.path.exists(os.path.join(self.root, "copy")))

    def test_import_local_copies_when_dest_given(self):
        src = make_model(os.path.join(self.root, "src"))
        dest = os.path.join(self.root, "dest")
        fm.import_local(src, dest)
        self.assertEqual(fm.check_model_dir(dest), [])

    def test_import_local_rejects_incomplete(self):
        src = make_model(os.path.join(self.root, "bad"), weights=False)
        with self.assertRaisesRegex(ValueError, "不完整"):
            fm.import_local(src)

    def test_fetch_hf_skips_when_complete(self):
        d = make_model(os.path.join(self.root, "done"))
        # 已完整时不应调用 huggingface_hub（这里若调用会因为没有网络而失败）
        self.assertEqual(fm.fetch_hf("Some/Model", d), os.path.abspath(d))

    def test_download_wasm_from_file_url_and_skip_second_time(self):
        src = os.path.join(self.root, "remote.wasm")
        with open(src, "wb") as f:
            f.write(b"\x00asm" + b"\x01\x00\x00\x00" + b"x" * 20)
        dest = os.path.join(self.root, "libs", "model.wasm")
        calls = []

        def opener(url, timeout=60):
            calls.append(url)
            return open(src, "rb")

        fm.download_file("file://" + src, dest, opener=opener)
        self.assertTrue(os.path.isfile(dest))
        fm.download_file("file://" + src, dest, opener=opener)
        self.assertEqual(len(calls), 1, "已存在的 wasm 不应重复下载")

    def test_download_rejects_html_error_page(self):
        src = os.path.join(self.root, "page.html")
        with open(src, "w") as f:
            f.write("<html>404</html>")
        dest = os.path.join(self.root, "x.wasm")
        with self.assertRaisesRegex(ValueError, "不是 wasm"):
            fm.download_file("file://" + src, dest, opener=lambda url, timeout=60: open(src, "rb"))
        self.assertFalse(os.path.exists(dest))
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_download_propagates_http_error(self):
        def opener(url, timeout=60):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO())

        with self.assertRaises(urllib.error.HTTPError):
            fm.download_file("https://example.invalid/x.wasm", os.path.join(self.root, "y.wasm"), opener=opener)


if __name__ == "__main__":
    unittest.main()
