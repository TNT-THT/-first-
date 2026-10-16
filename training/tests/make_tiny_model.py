#!/usr/bin/env python3
"""
生成一个随机初始化的极小因果语言模型和分词器，用于 CPU 上的冒烟测试。
它不是有用的模型，只用于验证训练、合并、打包的代码路径能跑通。

支持的架构（对应 transformers 中的配置类）：qwen2、llama、mistral、phi3、qwen3、gemma2。
reject_system=True 时，对话模板会拒绝 system 角色（与 Gemma 的模板行为一致）。

用法：python3 make_tiny_model.py OUT_DIR [ARCH]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "examples", "sample.jsonl")
ARCHS = ("qwen2", "llama", "mistral", "phi3", "qwen3", "gemma2")

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)
TEMPLATE_NO_SYSTEM = (
    "{% for m in messages %}{% if m['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def _config(arch, vocab_size):
    from transformers import (
        Gemma2Config,
        LlamaConfig,
        MistralConfig,
        Phi3Config,
        Qwen2Config,
        Qwen3Config,
    )

    common = dict(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    if arch == "qwen2":
        return Qwen2Config(**common)
    if arch == "llama":
        return LlamaConfig(**common)
    if arch == "mistral":
        return MistralConfig(**common)
    if arch == "phi3":
        return Phi3Config(pad_token_id=0, **common)
    if arch == "qwen3":
        return Qwen3Config(head_dim=8, **common)
    if arch == "gemma2":
        return Gemma2Config(head_dim=8, **common)
    raise ValueError(f"不支持的架构：{arch}（可选：{', '.join(ARCHS)}）")


def build(out_dir, arch="qwen2", reject_system=False):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    with open(SAMPLE, "r", encoding="utf-8") as f:
        corpus = [line for line in f if line.strip()]

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(corpus * 20, trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", eos_token="<|endoftext|>", pad_token="<|endoftext|>"
    )
    fast.chat_template = TEMPLATE_NO_SYSTEM if reject_system else TEMPLATE
    os.makedirs(out_dir, exist_ok=True)
    fast.save_pretrained(out_dir)

    cfg = _config(arch, len(fast))
    AutoModelForCausalLM.from_config(cfg).save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(2)
    arch = sys.argv[2] if len(sys.argv) == 3 else "qwen2"
    build(sys.argv[1], arch=arch)
    print(f"已生成 {sys.argv[1]}（架构：{arch}）")
