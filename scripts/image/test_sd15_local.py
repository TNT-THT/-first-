"""本机生图脚本的测试（标准库 unittest）。

运行：python3 -m unittest scripts/image/test_sd15_local.py   （需要 numpy、onnxruntime、regex）
其中固定期望值来自网页版（JavaScript）的同一算法，用于交叉检查：
分词结果、随机噪声、PNDM 时间步。端到端测试使用 test/fixtures/sd15-tiny 的微型模型，只验证流程与形状。
"""
import json
import os
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sd15_local as L  # noqa: E402

FIXTURES = os.path.join(HERE, "..", "..", "test", "fixtures", "sd15-tiny")
SCHED = {"num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear",
         "skip_prk_steps": True, "steps_offset": 1, "prediction_type": "epsilon", "timestep_spacing": "leading",
         "set_alpha_to_one": False}


def tiny_vocab():
    """与网页版测试相同的小词表：字节字符 + 两条合并规则。"""
    chars = sorted(set(L.BYTE_MAP.values()), key=lambda c: [ord(x) for x in c])
    vocab, i = {}, 0
    for c in chars:
        vocab[c] = i
        i += 1
        vocab[c + "</w>"] = i
        i += 1
    vocab["<|startoftext|>"] = i
    i += 1
    vocab["<|endoftext|>"] = i
    i += 1
    vocab["ca"] = i
    i += 1
    vocab["cat</w>"] = i
    return vocab, "#version: 0.2\nc a\nca t</w>\n"


class TokenizerTest(unittest.TestCase):
    def setUp(self):
        vocab, merges = tiny_vocab()
        self.tok = L.ClipTokenizer(vocab, merges)

    def test_matches_web_tokenizer_for_a_cat(self):
        # 网页版 tinyTokenizer 对 'a cat' 的输出（前 6 个 id）
        self.assertEqual(self.tok.encode("a cat").tolist()[:6], [512, 129, 515, 513, 513, 513])

    def test_length_and_padding(self):
        ids = self.tok.encode("x" * 200)
        self.assertEqual(len(ids), 77)
        self.assertEqual(ids[-1], self.tok.eos)

    def test_bos_eos(self):
        ids = self.tok.tokenize("")
        self.assertEqual(ids, [self.tok.bos, self.tok.eos])


class NoiseAndSchedulerTest(unittest.TestCase):
    def test_noise_matches_web_for_seed_7(self):
        # 网页版 gaussianNoise(7, 4*32*32) 的前 4 个值（来自 Node 运行结果）
        x = L.gaussian_noise(7, 4 * 32 * 32)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.size, 4096)
        expected = [2.7593729496002197, 1.1319811344146729, -0.06805135309696198, -0.20517154037952423]
        np.testing.assert_allclose(x[:4], expected, rtol=0, atol=1e-6)
        self.assertTrue(np.array_equal(x, L.gaussian_noise(7, 4 * 32 * 32)))

    def test_timesteps_match_web(self):
        s = L.PndmScheduler(SCHED)
        self.assertEqual(s.set_timesteps(5), [801, 601, 601, 401, 201, 1])
        self.assertEqual(len(s.set_timesteps(20)), 21)  # PLMS 多一步（与 diffusers 一致）

    def test_rejects_bad_steps(self):
        with self.assertRaises(ValueError):
            L.PndmScheduler(SCHED).set_timesteps(1)


class PngTest(unittest.TestCase):
    def test_png_roundtrip_header_and_pixels(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[0, 0] = [255, 0, 0]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.png")
            L.write_png(path, rgb)
            with open(path, "rb") as f:
                data = f.read()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        w, h, bd, ct = struct.unpack(">IIBB", data[16:26])
        self.assertEqual((w, h, bd, ct), (3, 2, 8, 2))
        idat_len = struct.unpack(">I", data[33:37])[0]
        raw = zlib.decompress(data[41:41 + idat_len])
        self.assertEqual(raw[0:1], b"\x00")
        self.assertEqual(raw[1:4], bytes([255, 0, 0]))

    def test_chw_to_rgb_rounding(self):
        chw = np.array([[[-1.0, 1.0]], [[0.0, 0.0]], [[0.0, 0.0]]], dtype=np.float32)
        rgb = L.chw_to_rgb(chw.reshape(-1), 2, 1)
        self.assertEqual(rgb[0, 0].tolist(), [0, 128, 128])
        self.assertEqual(rgb[0, 1].tolist(), [255, 128, 128])


@unittest.skipUnless(os.path.isdir(FIXTURES), "需要 test/fixtures/sd15-tiny")
class PipelineTest(unittest.TestCase):
    def test_tiny_model_end_to_end_shape(self):
        import onnxruntime as ort

        vocab, merges = tiny_vocab()
        with tempfile.TemporaryDirectory() as d:
            for sub in ("tokenizer", "scheduler"):
                os.makedirs(os.path.join(d, sub))
            with open(os.path.join(d, "tokenizer", "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vocab, f)
            with open(os.path.join(d, "tokenizer", "merges.txt"), "w", encoding="utf-8") as f:
                f.write(merges)
            with open(os.path.join(d, "scheduler", "scheduler_config.json"), "w", encoding="utf-8") as f:
                json.dump(SCHED, f)
            for rel in ("text_encoder/model.onnx", "unet/model.onnx", "unet/weights.pb", "vae_decoder/model.onnx"):
                os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
                with open(os.path.join(FIXTURES, rel), "rb") as src, open(os.path.join(d, rel), "wb") as dst:
                    dst.write(src.read())
            pipe = L.load_pipeline(d, ["CPUExecutionProvider"], ort)
            chw = L.generate(pipe, "a cat", width=256, height=256, steps=3, guidance=7.5, seed=7)
            self.assertEqual(chw.shape, (3, 256, 256))
            self.assertTrue(np.all(np.isfinite(chw)))
            again = L.generate(pipe, "a cat", width=256, height=256, steps=3, guidance=7.5, seed=7)
            self.assertTrue(np.array_equal(chw, again), "同一种子应得到同样的结果")


if __name__ == "__main__":
    unittest.main()
