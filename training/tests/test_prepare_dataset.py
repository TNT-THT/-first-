import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import prepare_dataset as pd  # noqa: E402


class PrepareDatasetTest(unittest.TestCase):
    def test_prompt_response_format(self):
        kept, skipped = pd.process(['{"prompt": "问", "response": "答"}'])
        self.assertEqual(kept, [{"messages": [{"role": "user", "content": "问"}, {"role": "assistant", "content": "答"}]}])
        self.assertEqual(sum(skipped.values()), 0)

    def test_instruction_with_input(self):
        kept, _ = pd.process(['{"instruction": "判断", "input": "s=abba", "output": "回文"}'])
        self.assertEqual(kept[0]["messages"][0]["content"], "判断\n\ns=abba")

    def test_system_message_kept_first(self):
        kept, _ = pd.process(['{"system": "你是助手", "prompt": "问", "response": "答"}'])
        roles = [m["role"] for m in kept[0]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_rejects_bad_roles_and_order(self):
        cases = {
            '{"messages": [{"role": "assistant", "content": "a"}, {"role": "user", "content": "q"}]}': "角色没有按 user/assistant 交替",
            '{"messages": [{"role": "user", "content": "q"}]}': "对话必须以 assistant 回复结尾",
            '{"messages": [{"role": "user", "content": "q"}, {"role": "system", "content": "s"}, {"role": "assistant", "content": "a"}]}': "system 只能出现在第一条",
            '{"messages": [{"role": "tool", "content": "q"}]}': "messages 中有无效的 role",
        }
        for line, reason in cases.items():
            kept, skipped = pd.process([line])
            self.assertEqual(kept, [], line)
            self.assertIn(reason, skipped, line)

    def test_skips_empty_and_too_long_and_invalid_json(self):
        lines = [
            '{"prompt": "  ", "response": "x"}',
            '{"prompt": "' + "字" * 50 + '", "response": "x"}',
            "not json",
            "",
        ]
        kept, skipped = pd.process(lines, max_chars=20)
        self.assertEqual(kept, [])
        self.assertEqual(skipped["prompt 为空"], 1)
        self.assertEqual(skipped["JSON 解析失败"], 1)
        self.assertEqual(sum(v for k, v in skipped.items() if k.startswith("超过")), 1)

    def test_sample_file_is_valid(self):
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "sample.jsonl")
        with open(path, encoding="utf-8") as f:
            kept, skipped = pd.process(f)
        self.assertEqual(len(kept), 3)
        self.assertEqual(sum(skipped.values()), 3)


if __name__ == "__main__":
    unittest.main()
