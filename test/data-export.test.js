// 本机数据的选择性导出（管理员，静态版）。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildDataExport, DATA_SECTIONS, dataExportFilename, NEVER_EXPORTED_KEYS } from '../web/src/data-export.js';

function memoryStorage(init = {}) {
  const m = new Map(Object.entries(init));
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
  };
}

const NOW = new Date('2026-10-18T08:00:00Z');

test('只导出所选分区，并解析为结构化数据', () => {
  const s = memoryStorage({
    'oj-ai.sessions.v1': JSON.stringify({ sessions: [{ id: 'a' }] }),
    'oj-ai.kb.v1': JSON.stringify({ models: {} }),
  });
  const out = buildDataExport(s, ['sessions'], { now: NOW });
  assert.equal(out.format, 'oj-ai-data');
  assert.equal(out.version, 1);
  assert.equal(out.exportedAt, NOW.toISOString());
  assert.deepEqual(out.sections, { sessions: { sessions: [{ id: 'a' }] } });
  assert.deepEqual(out.missing, []);
});

test('所选分区在本机没有数据时记入 missing，不生成空条目', () => {
  const out = buildDataExport(memoryStorage({}), ['sessions', 'knowledge'], { now: NOW });
  assert.deepEqual(out.sections, {});
  assert.deepEqual(out.missing, ['sessions', 'knowledge']);
});

test('无法解析为 JSON 的内容保留原始文本', () => {
  const s = memoryStorage({ 'oj-ai.localModels': 'not json' });
  const out = buildDataExport(s, ['localModels'], { now: NOW });
  assert.equal(out.sections.localModels, 'not json');
});

test('未知的分区 ID 被忽略；管理员标记、令牌永远不导出', () => {
  const s = memoryStorage({
    'oj-ai.admin': '{"exp":1}',
    'oj-ai.token': 'secret-token',
    'oj-ai.unlock': '1',
    'oj-ai.sessions.v1': '[]',
  });
  const out = buildDataExport(s, ['admin', 'token', 'unlock', 'nope', 'sessions'], { now: NOW });
  assert.deepEqual(Object.keys(out.sections), ['sessions']);
  const text = JSON.stringify(out);
  assert.ok(!text.includes('secret-token'));
  assert.ok(!text.includes('"exp"'));
});

test('分区表与排除列表保持一致：分区 key 不包含被禁止的键', () => {
  for (const sec of DATA_SECTIONS) assert.ok(!NEVER_EXPORTED_KEYS.includes(sec.key), sec.key);
  assert.deepEqual(
    DATA_SECTIONS.map((s) => s.id),
    ['sessions', 'adaptation', 'knowledge', 'localModels'],
  );
});

test('文件名包含日期', () => {
  assert.equal(dataExportFilename(NOW), 'oj-ai-data-2026-10-18.json');
});
