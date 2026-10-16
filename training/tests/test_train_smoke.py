"""
CPU 冒烟测试：用随机初始化的极小模型跑通 build_features → LoRA 训练 → 合并 → 重新加载。
需要 torch / transformers / peft；未安装时自动跳过。它不验证模型质量。
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

try:
    import peft  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401

    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "未安装 torch/transformers/peft，跳过训练冒烟测试")
class TrainSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import make_tiny_model
        import prepare_dataset

        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = make_tiny_model.build(os.path.join(cls.tmp.name, "base"))
        sample = os.path.join(HERE, "..", "examples", "sample.jsonl")
        cls.data = os.path.join(cls.tmp.name, "train.jsonl")
        with open(sample, encoding="utf-8") as f:
            kept, _ = prepare_dataset.process(f)
        with open(cls.data, "w", encoding="utf-8") as f:
            for item in kept:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_loss_only_on_assistant_replies(self):
        from transformers import AutoTokenizer
        import train_lora

        tok = AutoTokenizer.from_pretrained(self.base)
        examples = train_lora.load_examples(self.data)
        features, skipped = train_lora.build_features(examples, tok, max_length=0)
        self.assertEqual(skipped, 0)
        for feat, ex in zip(features, examples):
            labeled = [i for i, v in zip(feat["input_ids"], feat["labels"]) if v != -100]
            text = tok.decode(labeled)
            for m in ex["messages"]:
                if m["role"] == "assistant":
                    self.assertIn(m["content"], text)
                else:
                    self.assertNotIn(m["content"], text)

    def test_train_merge_and_reload(self):
        import train_lora
        from transformers import AutoModelForCausalLM

        out = os.path.join(self.tmp.name, "run")
        merged = os.path.join(self.tmp.name, "merged")
        rc = train_lora.main([
            "--base", self.base, "--data", self.data, "--output-dir", out, "--merged-dir", merged,
            "--max-steps", "2", "--batch-size", "2", "--grad-accum", "1", "--cpu",
            "--lora-r", "4", "--lora-alpha", "8", "--target-modules", "q_proj,v_proj",
        ])
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "training_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.assertEqual(meta["steps"], 2)
        self.assertTrue(os.path.isfile(os.path.join(out, "adapter_model.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(merged, "model.safetensors")))
        self.assertTrue(os.path.isfile(os.path.join(merged, "tokenizer.json")))

        model = AutoModelForCausalLM.from_pretrained(merged)
        self.assertFalse(any("lora_" in k for k in model.state_dict()))
        ids = torch.tensor([[1, 2, 3]])
        self.assertEqual(model(ids).logits.shape[-1], model.config.vocab_size)



@unittest.skipUnless(HAVE_DEPS, "未安装 torch/transformers/peft，跳过训练冒烟测试")
class ArchitectureTest(unittest.TestCase):
    """不同架构的极小随机模型都能完成 LoRA 训练与合并；不验证模型质量。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = os.path.join(cls.tmp.name, "data.jsonl")
        with open(cls.data, "w", encoding="utf-8") as f:
            for i in range(6):
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": "你是刷题助手"},
                    {"role": "user", "content": f"题{i}"},
                    {"role": "assistant", "content": f"答{i}"},
                ]}, ensure_ascii=False) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _train(self, arch, reject_system=False):
        import make_tiny_model
        import train_lora

        base = os.path.join(self.tmp.name, f"base-{arch}-{int(reject_system)}")
        make_tiny_model.build(base, arch=arch, reject_system=reject_system)
        out = os.path.join(self.tmp.name, f"run-{arch}-{int(reject_system)}")
        merged = os.path.join(self.tmp.name, f"merged-{arch}-{int(reject_system)}")
        rc = train_lora.main([
            "--base", base, "--data", self.data, "--output-dir", out, "--merged-dir", merged,
            "--max-steps", "1", "--batch-size", "1", "--grad-accum", "1", "--cpu", "--dtype", "fp32",
            "--lora-r", "4", "--lora-alpha", "8", "--target-modules", "q_proj,k_proj,v_proj,o_proj",
        ])
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "training_meta.json"), encoding="utf-8") as f:
            return json.load(f), merged

    def test_qwen_style_uses_q_k_v_o(self):
        meta, merged = self._train("qwen2")
        self.assertEqual(meta["lora"]["target_modules"], ["k_proj", "o_proj", "q_proj", "v_proj"])
        self.assertTrue(os.path.isfile(os.path.join(merged, "model.safetensors")))

    def test_each_architecture_trains_and_merges(self):
        for arch in ("llama", "mistral", "qwen3", "gemma2"):
            with self.subTest(arch=arch):
                meta, merged = self._train(arch)
                self.assertEqual(meta["steps"], 1)
                self.assertTrue(os.path.isfile(os.path.join(merged, "model.safetensors")))

    def test_phi3_falls_back_to_fused_projections(self):
        meta, _ = self._train("phi3")
        self.assertEqual(meta["lora"]["target_modules"], ["o_proj", "qkv_proj"])

    def test_template_without_system_role(self):
        meta, _ = self._train("gemma2", reject_system=True)
        self.assertEqual(meta["steps"], 1)
        self.assertEqual(meta["examples"], 6)

    def test_unknown_module_request_falls_back_with_note(self):
        import io
        import contextlib
        import train_lora
        from transformers import AutoModelForCausalLM

        base = os.path.join(self.tmp.name, "base-qwen2-0")
        model = AutoModelForCausalLM.from_pretrained(base)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            chosen = train_lora.resolve_target_modules(model, ["no_such_proj"])
        self.assertEqual(sorted(chosen), ["k_proj", "o_proj", "q_proj", "v_proj"])
        self.assertIn("no_such_proj", out.getvalue())


if __name__ == "__main__":
    unittest.main()
