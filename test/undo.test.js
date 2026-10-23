// 对话撤回：只撤回最后一轮，并把问题交还给输入框。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { undoLastTurn, undoableTurnIndex } from '../web/src/store.js';

const u = (content, extra = {}) => ({ role: 'user', content, ts: 1, ...extra });
const a = (content, extra = {}) => ({ role: 'assistant', content, ts: 2, ...extra });

test('撤回：删除最后一问一答，返回原问题', () => {
  const s = { title: '第一问', messages: [u('第一问'), a('回答一'), u('第二问'), a('回答二')] };
  const r = undoLastTurn(s);
  assert.deepEqual(r, { question: '第二问', recorded: false });
  assert.deepEqual(s.messages.map((m) => m.content), ['第一问', '回答一']);
  assert.equal(s.title, '第一问');
});

test('撤回：回复失败（只剩问题或错误回复）时同样可以撤回', () => {
  const s = { title: '题目', messages: [u('题目'), a('出错了', { error: true })] };
  assert.equal(undoLastTurn(s).question, '题目');
  assert.equal(s.messages.length, 0);
  assert.equal(s.title, '新会话');
  const t = { title: '题目', messages: [u('另一题')] };
  assert.equal(undoLastTurn(t).question, '另一题');
  assert.equal(t.messages.length, 0);
});

test('撤回：没有可撤回的轮次返回 null，且不修改会话', () => {
  assert.equal(undoLastTurn({ messages: [] }), null);
  assert.equal(undoLastTurn(null), null);
  const s = { messages: [a('孤立的回复')] };
  assert.equal(undoLastTurn(s), null);
  assert.equal(s.messages.length, 1);
  assert.equal(undoableTurnIndex({ messages: [u('a'), u('b')] }), 1);
});

test('撤回：曾记录反馈或更正的轮次会标记 recorded，供界面提示', () => {
  const s = { messages: [u('更正：xxx', { recorded: true }), a('已记录')] };
  assert.equal(undoLastTurn(s).recorded, true);
});

test('撤回：可以连续撤回，每次只撤回一轮', () => {
  const s = { messages: [u('1'), a('1'), u('2'), a('2'), u('3'), a('3')] };
  assert.equal(undoLastTurn(s).question, '3');
  assert.equal(undoLastTurn(s).question, '2');
  assert.deepEqual(s.messages.map((m) => m.content), ['1', '1']);
});
