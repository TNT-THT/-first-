#!/usr/bin/env python3
"""
生成一个随机初始化的极小 Qwen2 模型和分词器，用于 CPU 上的冒烟测试。
它不是有用的模型，只用于验证训练、合并、打包的代码路径能跑通。

用法：python3 make_tiny_model.py OUT_DIR
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "..", "examples", "sample.jsonl")

TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def build(out_dir):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

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
    fast.chat_template = TEMPLATE
    fast.save_pretrained(out_dir)

    cfg = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    Qwen2ForCausalLM(cfg).save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    os.makedirs(sys.argv[1], exist_ok=True)
    build(sys.argv[1])
    print(f"已生成 {sys.argv[1]}")
