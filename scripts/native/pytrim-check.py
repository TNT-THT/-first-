"""自检：runner.py 的对话裁剪（test/python-runner-trim.test.js 调用）。

用假 tokenizer 走一遍 trim_messages_for_window：40 条历史 + system + 最后问题，
断言保留 system、保留最新消息、且裁剪后不以助手消息开头。
"""
import importlib.util
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runner", root / "server" / "native" / "python" / "runner.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class FakeTok:
    model_max_length = 4096
    chat_template = "x"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in msgs)

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": list(range(max(1, len(text))))}

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)


mod.STATE["tokenizer"] = FakeTok()
messages = [{"role": "system", "content": "S"}]
messages += [
    {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 300}
    for i in range(40)
]
messages.append({"role": "user", "content": "FINAL"})
out = mod.trim_messages_for_window(messages)
assert out[0]["role"] == "system", out[0]
assert out[-1]["content"] == "FINAL", out[-1]
assert out[1]["role"] != "assistant", "裁剪后不能以助手消息开头"
print("ok", len(out))
sys.exit(0)
