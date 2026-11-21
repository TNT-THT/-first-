// 用量统计核心：聚合口径、上限裁剪、坏数据清洗、网页/服务端两种存储的往返。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  MAX_RECENT_ENTRIES,
  MAX_USAGE_DAYS,
  MAX_USAGE_MODELS,
  USAGE_STORAGE_KEY,
  clearUsageStats,
  dayKeyOf,
  emptyUsageStats,
  loadUsageStats,
  normalizeUsageStats,
  recordUsage,
  saveUsageStats,
  summarizeUsage,
} from '../web/src/usage-stats.js';
import { UsageStatsStore } from '../server/usage-stats.js';

const T0 = new Date('2026-11-20T10:00:00').getTime();
const DAY = 86400000;

function memStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    _map: map,
  };
}

test('normalizeUsageStats：垃圾输入回退为空结构，永不抛错', () => {
  for (const bad of [null, undefined, 42, 'x', [], { models: 'nope' }, { models: { a: null } }]) {
    assert.deepEqual(normalizeUsageStats(bad), emptyUsageStats());
  }
  const cleaned = normalizeUsageStats({
    v: 99,
    models: {
      m1: { calls: '3', errors: -2, promptTokens: 10.9, completionTokens: Number.NaN, lastUsedAt: 'x', days: { '2026-11-20': { calls: 2, promptTokens: 5, completionTokens: 6 }, bogus: { calls: 9 } } },
    },
    recent: [{ t: 'bad', model: 7, promptTokens: -1, status: 'weird' }, null, { t: 5, model: 'm1', source: 'api', promptTokens: 3, completionTokens: 4, status: 'error' }],
  });
  assert.equal(cleaned.models.m1.calls, 3);
  assert.equal(cleaned.models.m1.errors, 0);
  assert.equal(cleaned.models.m1.promptTokens, 10);
  assert.equal(cleaned.models.m1.days['2026-11-20'].calls, 2);
  assert.equal(cleaned.models.m1.days.bogus, undefined);
  assert.equal(cleaned.recent.length, 2);
  assert.equal(cleaned.recent[0].status, 'ok', '非法 status 应清洗为 ok');
  assert.equal(cleaned.recent[0].model, 'unknown');
  assert.equal(cleaned.recent[1].source, 'api');
});

test('recordUsage：按模型与日期聚合，失败不计 Token、停止计已产生 Token', () => {
  const s = emptyUsageStats();
  recordUsage(s, { model: 'a', source: 'web', promptTokens: 100, completionTokens: 50 }, T0);
  recordUsage(s, { model: 'a', source: 'web', promptTokens: 20, completionTokens: 10 }, T0 + 1000);
  recordUsage(s, { model: 'a', source: 'web', promptTokens: 999, completionTokens: 999, status: 'error' }, T0 + 2000);
  recordUsage(s, { model: 'b', source: 'api', promptTokens: 10, completionTokens: 5, status: 'stopped' }, T0 + DAY);
  const sum = summarizeUsage(s, { now: T0 + DAY });
  assert.equal(sum.totalCalls, 4);
  assert.equal(sum.totalErrors, 1);
  assert.equal(sum.promptTokens, 130);
  assert.equal(sum.completionTokens, 65);
  const a = sum.models.find((m) => m.id === 'a');
  assert.equal(a.calls, 3);
  assert.equal(a.errors, 1);
  assert.equal(a.today.calls, 0, '次日查看时今日应为新一天');
  const b = sum.models.find((m) => m.id === 'b');
  assert.equal(b.today.calls, 1);
  assert.equal(b.totalTokens, 15, 'stopped 应计入已产生的 Token');
  assert.equal(sum.recent.length, 4);
  assert.equal(sum.recent[0].model, 'b', '近期日志按时间倒序');
});

test('recordUsage：近期日志定长、日汇总与模型种类都有上限', () => {
  const s = emptyUsageStats();
  for (let i = 0; i < MAX_RECENT_ENTRIES + 20; i++) {
    recordUsage(s, { model: 'a', promptTokens: 1, completionTokens: 1 }, T0 + i * 1000);
  }
  assert.equal(s.recent.length, MAX_RECENT_ENTRIES);
  assert.equal(s.models.a.calls, MAX_RECENT_ENTRIES + 20, '聚合计数不受近期日志上限影响');

  const s2 = emptyUsageStats();
  for (let d = 0; d < MAX_USAGE_DAYS + 9; d++) {
    recordUsage(s2, { model: 'a', promptTokens: 1, completionTokens: 1 }, T0 + d * DAY);
  }
  assert.equal(Object.keys(s2.models.a.days).length, MAX_USAGE_DAYS);
  assert.ok(s2.models.a.days[dayKeyOf(T0 + (MAX_USAGE_DAYS + 8) * DAY)], '应保留最近一天');

  const s3 = emptyUsageStats();
  for (let i = 0; i < MAX_USAGE_MODELS + 5; i++) {
    recordUsage(s3, { model: `m${i}`, promptTokens: 1, completionTokens: 1 }, T0 + i * 1000);
  }
  assert.equal(Object.keys(s3.models).length, MAX_USAGE_MODELS);
  assert.ok(s3.models[`m${MAX_USAGE_MODELS + 4}`], '最新模型不应被淘汰');
  assert.equal(s3.models.m0, undefined, '最久未用的模型应被淘汰');
});

test('summarizeUsage：按总 Token 降序、最近 7 天序列对齐', () => {
  const s = emptyUsageStats();
  recordUsage(s, { model: 'small', promptTokens: 10, completionTokens: 10 }, T0);
  recordUsage(s, { model: 'big', promptTokens: 500, completionTokens: 500 }, T0 - 3 * DAY);
  const sum = summarizeUsage(s, { now: T0 });
  assert.deepEqual(sum.models.map((m) => m.id), ['big', 'small']);
  assert.equal(sum.lastDays.length, 7);
  assert.equal(sum.lastDays[6].day, dayKeyOf(T0));
  assert.equal(sum.lastDays[3].totalTokens, 1000);
  assert.equal(sum.lastDays[6].totalTokens, 20);
});

test('网页存储往返：保存后可读回，损坏 JSON 回退为空', () => {
  const storage = memStorage();
  const s = emptyUsageStats();
  recordUsage(s, { model: 'a', promptTokens: 7, completionTokens: 8 }, T0);
  assert.equal(saveUsageStats(s, storage), true);
  const back = loadUsageStats(storage);
  assert.equal(back.models.a.calls, 1);
  assert.equal(back.models.a.completionTokens, 8);
  storage._map.set(USAGE_STORAGE_KEY, '{broken json');
  assert.deepEqual(loadUsageStats(storage), emptyUsageStats());
  assert.deepEqual(loadUsageStats(null), emptyUsageStats(), '无 storage 时也不抛错');
  assert.equal(saveUsageStats(s, null), false);
});

test('UsageStatsStore：落盘后可读回，损坏文件从空开始，clear 立即生效', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'oj-usage-'));
  try {
    const file = join(dir, 'usage.json');
    const store = new UsageStatsStore(file);
    store.record({ model: 'api-model', promptTokens: 30, completionTokens: 12 });
    store.record({ model: 'api-model', promptTokens: 0, completionTokens: 0, status: 'error' });
    store.flush();
    const raw = JSON.parse(await readFile(file, 'utf8'));
    assert.equal(raw.models['api-model'].calls, 2);

    const reopened = new UsageStatsStore(file);
    const snap = reopened.snapshot();
    assert.equal(snap.source, 'api');
    assert.equal(snap.totalCalls, 2);
    assert.equal(snap.totalErrors, 1);
    assert.equal(snap.totalTokens, 42);
    assert.equal(snap.recent[0].source, 'api');

    const cleared = reopened.clear();
    assert.equal(cleared.totalCalls, 0);
    assert.equal(new UsageStatsStore(file).snapshot().totalCalls, 0, 'clear 应已落盘');

    await writeFile(file, 'not json at all');
    assert.equal(new UsageStatsStore(file).snapshot().totalCalls, 0, '损坏文件不应抛错');

    const mem = new UsageStatsStore('');
    mem.record({ model: 'x', promptTokens: 1, completionTokens: 1 });
    assert.equal(mem.snapshot().totalTokens, 2, '仅内存模式可用');
    mem.flush();
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('clearUsageStats 返回空结构且不影响原对象', () => {
  const s = emptyUsageStats();
  recordUsage(s, { model: 'a', promptTokens: 1, completionTokens: 1 }, T0);
  const cleared = clearUsageStats();
  assert.deepEqual(cleared, emptyUsageStats());
  assert.equal(s.models.a.calls, 1);
});
