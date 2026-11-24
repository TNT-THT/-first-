// runner.py（原格式本地运行）的对话裁剪：历史超过模型窗口时 transformers 会直接抛错。
// 用一个假 tokenizer 跑 scripts/native/pytrim-check.py，验证裁剪结果。
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

test('runner.py：超窗历史被裁到保留 system + 最新消息，且不以助手消息开头', () => {
  let stdout = '';
  try {
    stdout = execFileSync('python3', [path.join(here, '..', 'scripts', 'native', 'pytrim-check.py')], { encoding: 'utf8' });
  } catch (err) {
    if (err.code === 'ENOENT') return; // 没装 Python 的环境跳过（CI 的 Node 环境不保证有）
    throw err;
  }
  assert.match(stdout, /^ok \d+$/m);
});
