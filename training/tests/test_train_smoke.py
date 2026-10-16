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


if __name__ == "__main__":
    unittest.main()
