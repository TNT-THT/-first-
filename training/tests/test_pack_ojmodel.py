import json
import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pack_ojmodel as pk  # noqa: E402

WASM_URL = "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_86/base/Demo.wasm"


def make_model_dir(root, shards=("params_shard_0.bin", "params_shard_1.bin"), tokenizer="tokenizer.json"):
    d = os.path.join(root, "mlc")
    os.makedirs(d)
    index = {"records": [{"dataPath": s} for s in shards]}
    with open(os.path.join(d, "tensor-cache.json"), "w") as f:
        json.dump(index, f)
    for s in shards:
        with open(os.path.join(d, s), "wb") as f:
            f.write(os.urandom(1000))
    if tokenizer:
        with open(os.path.join(d, tokenizer), "w") as f:
            f.write("{}")
    with open(os.path.join(d, "mlc-chat-config.json"), "w") as f:
        json.dump({"model_type": "qwen2"}, f)
    wasm = os.path.join(root, "model.wasm")
    with open(wasm, "wb") as f:
        f.write(b"\0asm" + os.urandom(64))
    return d, wasm


def read_pack(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"OJMD"
    (hlen,) = struct.unpack("<I", data[4:8])
    header = json.loads(data[8 : 8 + hlen].decode("utf-8"))
    return header, data[8 + hlen :]


class PackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_layout_and_offsets(self):
        d, wasm = make_model_dir(self.root)
        out = os.path.join(self.root, "out.ojmodel")
        res = pk.pack(
            d, out, model_id="my-1b", label="我的模型", model_lib_url=WASM_URL, wasm_path=wasm,
            source_model="https://huggingface.co/oj-ai-training/my-1b",
            record_extra={"overrides": {"context_window_size": 4096}, "vram_required_MB": 2000},
        )
        header, data = read_pack(out)
        self.assertEqual(res["entries"], len(header["entries"]))
        self.assertEqual(header["format"], "oj-ai-model")
        self.assertEqual(header["version"], 1)
        self.assertEqual(header["modelId"], "my-1b")
        self.assertEqual(header["label"], "我的模型")
        self.assertEqual(header["source"]["model_lib"], WASM_URL)
        self.assertEqual(header["source"]["overrides"], {"context_window_size": 4096})
        self.assertEqual(header["source"]["vram_required_MB"], 2000)
        # 偏移相对于数据区，且各段首尾相接
        expected = 0
        for e in header["entries"]:
            self.assertEqual(e["offset"], expected)
            expected += e["size"]
        self.assertEqual(expected, len(data))
        scopes = {(e["scope"], e["url"].rsplit("/", 1)[-1]) for e in header["entries"]}
        self.assertIn(("webllm/model", "tensor-cache.json"), scopes)
        self.assertIn(("webllm/model", "params_shard_1.bin"), scopes)
        self.assertIn(("webllm/config", "mlc-chat-config.json"), scopes)
        wasm_entry = [e for e in header["entries"] if e["scope"] == "webllm/wasm"][0]
        self.assertEqual(wasm_entry["url"], WASM_URL)
        # 数据内容与源文件一致
        first = header["entries"][0]
        with open(os.path.join(d, "tensor-cache.json"), "rb") as f:
            self.assertEqual(data[first["offset"] : first["offset"] + first["size"]], f.read())

    def test_urls_follow_clean_model_url_rules(self):
        self.assertEqual(pk.clean_model_url("https://huggingface.co/a/b"), "https://huggingface.co/a/b/resolve/main/")
        self.assertEqual(pk.clean_model_url("https://huggingface.co/a/b/resolve/main/"), "https://huggingface.co/a/b/resolve/main/")

    def test_accepts_tokenizer_model_instead_of_json(self):
        d, wasm = make_model_dir(self.root, tokenizer="tokenizer.model")
        out = os.path.join(self.root, "t.ojmodel")
        pk.pack(d, out, model_id="x", label="x", model_lib_url=WASM_URL, wasm_path=wasm, source_model="https://huggingface.co/a/x")
        header, _ = read_pack(out)
        self.assertTrue(any(e["url"].endswith("tokenizer.model") for e in header["entries"]))

    def test_rejects_missing_shard_and_tokenizer(self):
        d, wasm = make_model_dir(self.root, tokenizer=None)
        with self.assertRaisesRegex(FileNotFoundError, "分词器"):
            pk.pack(d, os.path.join(self.root, "a"), model_id="x", label="x", model_lib_url=WASM_URL, wasm_path=wasm, source_model="https://huggingface.co/a/x")
        d2, wasm2 = make_model_dir(os.path.join(self.root, "two"))
        os.remove(os.path.join(d2, "params_shard_1.bin"))
        with self.assertRaisesRegex(FileNotFoundError, "params_shard_1.bin"):
            pk.pack(d2, os.path.join(self.root, "b"), model_id="x", label="x", model_lib_url=WASM_URL, wasm_path=wasm2, source_model="https://huggingface.co/a/x")

    def test_rejects_bad_id_and_non_https(self):
        d, wasm = make_model_dir(self.root)
        out = os.path.join(self.root, "c")
        with self.assertRaisesRegex(ValueError, "model_id"):
            pk.pack(d, out, model_id="bad/id", label="x", model_lib_url=WASM_URL, wasm_path=wasm, source_model="https://huggingface.co/a/x")
        with self.assertRaisesRegex(ValueError, "https"):
            pk.pack(d, out, model_id="ok", label="x", model_lib_url=WASM_URL, wasm_path=wasm, source_model="http://x/y")
        self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
