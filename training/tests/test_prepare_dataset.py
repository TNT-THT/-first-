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


import json as _json  # noqa: E402
import tempfile  # noqa: E402


class LoadAndSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def test_csv_with_bom_and_blank_rows(self):
        path = self._write("d.csv", "\ufeffprompt,response\n问一,答一\n,\n问二,答二\n")
        objs, skipped = pd.load_objects(path)
        self.assertEqual(len(objs), 2)
        kept, _ = pd.process_objects(objs)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[1]["messages"][0]["content"], "问二")

    def test_json_array(self):
        path = self._write("d.json", _json.dumps([{"prompt": "问", "response": "答"}], ensure_ascii=False))
        objs, _ = pd.load_objects(path)
        self.assertEqual(len(objs), 1)

    def test_json_must_be_array(self):
        path = self._write("d.json", '{"prompt": "问"}')
        with self.assertRaisesRegex(ValueError, "数组"):
            pd.load_objects(path)

    def test_unsupported_extension(self):
        path = self._write("d.xlsx", "x")
        with self.assertRaisesRegex(ValueError, "不支持"):
            pd.load_objects(path)

    def test_split_is_deterministic_and_disjoint(self):
        items = [{"messages": [{"role": "user", "content": str(i)}]} for i in range(100)]
        train_a, val_a = pd.split(items, 0.1, seed=7)
        train_b, val_b = pd.split(items, 0.1, seed=7)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 10)
        self.assertEqual(len(train_a) + len(val_a), 100)
        self.assertFalse({id(x) for x in train_a} & {id(x) for x in val_a})

    def test_small_dataset_is_not_split(self):
        items = [{"messages": []}] * 5
        train, val = pd.split(items, 0.5)
        self.assertEqual(len(train), 5)
        self.assertEqual(val, [])

    def test_build_dataset_writes_files_and_stats(self):
        rows = [{"prompt": f"问{i}", "response": f"答{i}"} for i in range(40)] + [{"prompt": "", "response": "x"}]
        path = self._write("raw.jsonl", "\n".join(_json.dumps(r, ensure_ascii=False) for r in rows))
        out = os.path.join(self.dir, "out")
        info = pd.build_dataset(path, out, val_ratio=0.1, seed=1)
        self.assertEqual(info["train"] + info["val"], 40)
        self.assertEqual(info["skipped"], {"prompt 为空": 1})
        self.assertTrue(os.path.isfile(os.path.join(out, "train.jsonl")))
        self.assertTrue(os.path.isfile(os.path.join(out, "val.jsonl")))

    def test_build_dataset_without_split_removes_stale_val(self):
        path = self._write("raw.jsonl", '{"prompt": "问", "response": "答"}')
        out = os.path.join(self.dir, "out")
        os.makedirs(out)
        with open(os.path.join(out, "val.jsonl"), "w") as f:
            f.write("stale")
        info = pd.build_dataset(path, out, val_ratio=0.0)
        self.assertIsNone(info["val_path"])
        self.assertFalse(os.path.exists(os.path.join(out, "val.jsonl")))

    def test_build_dataset_rejects_when_nothing_valid(self):
        path = self._write("raw.jsonl", '{"prompt": "", "response": ""}')
        with self.assertRaisesRegex(ValueError, "没有可用的样本"):
            pd.build_dataset(path, self.dir)
