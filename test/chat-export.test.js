// 对话导出：Markdown 内容、文件名与安全处理（所有用户可用，仅在本机生成）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildConversationMarkdown, conversationFilename, formatExportTime } from '../web/src/chat-export.js';

const NOW = new Date(2026, 9, 20, 14, 5);

test('导出时间与文件名格式', () => {
  assert.equal(formatExportTime(NOW), '2026-10-20 14:05');
  assert.equal(conversationFilename('你好 世界', NOW), '你好 世界-20261020-1405.md');
});

test('文件名去掉路径与保留字符，空标题使用默认名', () => {
  assert.equal(conversationFilename('a/b\\c:d*e?f"g<h>i|j\nk', NOW), 'a_b_c_d_e_f_g_h_i_j_k-20261020-1405.md');
  assert.equal(conversationFilename('', NOW), '对话-20261020-1405.md');
  assert.equal(conversationFilename('x'.repeat(80), NOW).startsWith('x'.repeat(40) + '-'), true);
});

test('导出内容：用户消息、标注模型的回答、固定提示；不含推理过程与出错消息', () => {
  const md = buildConversationMarkdown(
    {
      title: '测试对话',
      messages: [
        { role: 'user', content: '什么是二分查找？' },
        { role: 'assistant', content: '二分查找是……', model: 'm1', modelLabel: '测试AI' },
        { role: 'assistant', content: '', error: '失败', model: 'm1' },
        { role: 'user', content: '太啰嗦了' },
        { role: 'assistant', content: '已修改参数：最大输出长度 默认 → 400', fixed: true },
      ],
    },
    { now: NOW },
  );
  assert.match(md, /^# 测试对话\n\n导出时间：2026-10-20 14:05\n/);
  assert.match(md, /## 我\n\n什么是二分查找？/);
  assert.match(md, /## 回答（测试AI）\n\n二分查找是……/);
  assert.match(md, /> 系统提示：已修改参数：最大输出长度 默认 → 400/);
  assert.equal(md.includes('失败'), false);
  assert.equal(md.includes('thinking'), false);
});

test('空会话也能导出标题与导出时间', () => {
  const md = buildConversationMarkdown({ title: '', messages: [] }, { now: NOW });
  assert.match(md, /^# 未命名对话/);
});

test('多行固定提示逐行加引用符号', () => {
  const md = buildConversationMarkdown(
    { title: 't', messages: [{ role: 'assistant', content: '第一行\n第二行', fixed: true }] },
    { now: NOW },
  );
  assert.match(md, /> 系统提示：第一行\n>\n?> 第二行|> 系统提示：第一行\n> 第二行/);
});
