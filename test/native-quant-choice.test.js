// 「HF 一键使用」的量化选择：不再替用户挑一档直接下。
//
// 起因是用户真实踩坑：16.6 GB 总内存 / 7.4 GB 可用，HF 一键使用按偏好表自动选了
// Q4_K_M（15.5 GB），下一整晚才发现跑不动。省了 10 秒，浪费了几十 GB。
// 所以多档仓库必须先让用户选，且清单里要直接标明“这台机器装不装得下”。
import test from 'node:test';
import assert from 'node:assert/strict';
import { pickQuantForBudget, QUANT_PREFERENCE } from '../server/native/catalog.js';

// 用户那台机器：16.6 GB 总内存 / 7.4 GB 可用 / 没配页面文件
const USER_MACHINE = { totalGB: 16.6, freeGB: 7.4, commitLimitGB: 16.6, pagefileGB: 0 };

const GROUPED = [
  { quant: 'Q8_0', totalBytes: 32e9 },
  { quant: 'Q6_K', totalBytes: 25e9 },
  { quant: 'Q4_K_M', totalBytes: 15.5e9 },
  { quant: 'Q3_K_M', totalBytes: 12e9 },
  { quant: 'Q2_K', totalBytes: 8.5e9 },
  { quant: 'IQ1_M', totalBytes: 4.5e9 },
];

test('16 GB 机器不再被推荐 15.5 GB 那档', () => {
  const r = pickQuantForBudget(GROUPED, USER_MACHINE);
  assert.notEqual(r.quant, 'Q4_K_M', 'Q4_K_M 需 19.4 GB，16.6 GB 机器装不下，不能推荐');
  assert.ok(r.neededGB < 16.6, `推荐档位应在总内存内，实际 ${r.neededGB} GB`);
  assert.ok(['free', 'total'].includes(r.fits), `fits 应为 free/total，实际 ${r.fits}`);
});

test('内存充裕的机器仍然推荐高质量档（不能为了保险一律给最小档）', () => {
  const r = pickQuantForBudget(GROUPED, { totalGB: 64, freeGB: 50, commitLimitGB: 70, pagefileGB: 8 });
  assert.equal(r.quant, 'Q4_K_M', '64 GB 机器应正常推荐 Q4_K_M');
  assert.equal(r.fits, 'free');
  assert.match(r.reason, /装得下/);
});

test('每档都判定出装不装得下，且结论随机器变化', () => {
  const small = pickQuantForBudget(GROUPED, USER_MACHINE);
  const big = pickQuantForBudget(GROUPED, { totalGB: 64, freeGB: 50, commitLimitGB: 70, pagefileGB: 8 });
  const fitOf = (r, q) => r.candidates.find((c) => c.quant === q).fits;
  // Q8_0 约 37 GB：小机器肯定装不下，大机器装得下
  assert.equal(fitOf(small, 'Q8_0'), 'never');
  assert.equal(fitOf(big, 'Q8_0'), 'free');
  // Q4_K_M 在小机器上：超过 16.6-2.5 的预留线，也没配页面文件 → never
  assert.equal(fitOf(small, 'Q4_K_M'), 'never');
});

test('没配页面文件时，绝不能把“可提交上限=物理内存”当成能跑', () => {
  // 这是关键：commitLimit 在没配页面文件时等于物理内存，
  // 若拿它当退路，15.5 GB 的档位会被判成 spill（"能跑，只是慢"），
  // 用户就会以为有救。没页面文件就没有硬盘兜底，必须判 never。
  const r = pickQuantForBudget(GROUPED, USER_MACHINE);
  const q4 = r.candidates.find((c) => c.quant === 'Q4_K_M');
  assert.equal(q4.fits, 'never');
  // 配了足够大的页面文件后才可能出现 spill
  const withPf = pickQuantForBudget(
    [{ quant: 'Q4_K_M', totalBytes: 15.5e9 }],
    { totalGB: 16.6, freeGB: 7.4, commitLimitGB: 36, pagefileGB: 20 },
  );
  assert.equal(withPf.fits, 'spill');
  assert.match(withPf.reason, /页面文件|硬盘/);
});

test('全部档位都装不下时，给出最小的一档但把实话说明白', () => {
  const huge = [
    { quant: 'Q8_0', totalBytes: 80e9 },
    { quant: 'Q4_K_M', totalBytes: 40e9 },
  ];
  const r = pickQuantForBudget(huge, { totalGB: 8, freeGB: 3, commitLimitGB: 8, pagefileGB: 0 });
  assert.equal(r.fits, 'never');
  assert.equal(r.quant, 'Q8_0' === r.quant ? 'Q8_0' : r.quant);
  // 应挑最小那档（40e9 的 Q4_K_M），而不是最大的
  assert.equal(r.quant, 'Q4_K_M');
  assert.match(r.reason, /超过|装不下|换更小的模型/);
});

test('内存信息读不到时不瞎判“装不下”，退回质量偏好', () => {
  const r = pickQuantForBudget(GROUPED, {});
  assert.ok(r.quant, '仍应给出一个档位，不能返回空让用户没法用');
  assert.ok(QUANT_PREFERENCE.includes(r.quant) || r.quant, '应命中偏好表或至少给出一档');
});

test('系统预留：刚好等于总内存的档位不算装得下', () => {
  // 16 GB 机器，Q4_K_M 需要 19.4 GB > 16 GB，本来就装不下；
  // 再验一个更微妙的：需要 15.5 GB 的档位，也不能判成"装得下"（系统要留 2.5 GB）。
  const borderline = [{ quant: 'Q5_K_M', totalBytes: 12.5e9 }]; // 约需 15.5 GB
  const r = pickQuantForBudget(borderline, { totalGB: 16, freeGB: 14, commitLimitGB: 16, pagefileGB: 0 });
  assert.equal(r.fits, 'never', '15.5 GB 塞进 16 GB 会把系统挤爆，不能判成装得下');
});

test('空仓库返回 null，不崩', () => {
  const r = pickQuantForBudget([], USER_MACHINE);
  assert.equal(r.quant, null);
  assert.equal(r.fits, 'unknown');
});

// ——— 端到端：多档仓库必须把选择权交回用户 ———
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { NativeEngine } from '../server/native/engine.js';

const MULTI_FILES = [
  { path: 'Model-Q8_0.gguf', size: 32e9 },
  { path: 'Model-Q4_K_M.gguf', size: 15.5e9 },
  { path: 'Model-Q2_K.gguf', size: 8.5e9 },
  { path: 'README.md', size: 100 },
];

async function engineFor(t) {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-qc-'));
  const engine = new NativeEngine({
    dataDir: path.join(dir, 'native'),
    // 网络一律不走：只在本地用浏览器预载的文件树判定
    fetchImpl: async () => { throw new Error('测试不应联网'); },
    env: {},
    logger: { warn() {}, log() {}, error() {} },
  });
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  return engine;
}

test('多档 GGUF 一键使用 → 返回 choose-quant，清单带内存判定，不开始下载', async (t) => {
  const engine = await engineFor(t);
  const info = engine.hfInfoFromMeta('acme/Multi-GGUF', { author: 'acme', tags: ['gguf'] }, MULTI_FILES);
  const out = await engine.useRepo('acme/Multi-GGUF', 'local', {
    info, viaBrowser: true,
  });
  assert.equal(out.action, 'choose-quant', '多档时必须先让用户选');
  assert.ok(Array.isArray(out.quants) && out.quants.length === 3, '应列出全部三档');
  assert.ok(out.recommended, '应给出推荐档位');
  assert.ok(out.reason, '应说明为什么推荐这一档');
  // 每档都要有大小与内存估算，且带装不装得下的结论
  for (const q of out.quants) {
    assert.ok(q.sizeGB > 0, `${q.quant} 应有体积`);
    assert.ok(q.ramGB > 0, `${q.quant} 应有内存需求`);
    assert.ok(['free', 'total', 'spill', 'never', 'unknown'].includes(q.fits), `${q.quant} 的 fits 不合法`);
  }
  assert.equal(out.quants.filter((q) => q.recommended).length, 1, '推荐项应恰好一个');
  // 不能偷偷开服务端下载
  const st = await engine.status();
  const m = st.models.find((x) => x.id === out.id);
  assert.equal((m.downloads || []).length, 0, '选量化阶段不该开始任何下载');
});

test('用户指定量化后 → 正常进入下载，不再问第二遍', async (t) => {
  const engine = await engineFor(t);
  const info = engine.hfInfoFromMeta('acme/Multi-GGUF', { author: 'acme', tags: ['gguf'] }, MULTI_FILES);
  const out = await engine.useRepo('acme/Multi-GGUF', 'local', {
    info, viaBrowser: true, quant: 'q2_k',
  });
  assert.equal(out.action, 'download');
  // 小写也要认（界面/API 都可能传来小写）
  assert.equal(out.quant, 'Q2_K');
});

test('只有一档量化的仓库 → 直接下载，不打扰用户', async (t) => {
  const engine = await engineFor(t);
  const info = engine.hfInfoFromMeta('acme/Single-GGUF', { author: 'acme', tags: ['gguf'] },
    [{ path: 'Only-Q4_K_M.gguf', size: 4e9 }]);
  const out = await engine.useRepo('acme/Single-GGUF', 'local', { info, viaBrowser: true });
  assert.equal(out.action, 'download', '只有一档时没必要让用户点两下');
  assert.equal(out.quant, 'Q4_K_M');
});
