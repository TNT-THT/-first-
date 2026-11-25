// 页面文件（pagefile / swap）探测：**「多少字节」和「知不知道」是两件事**。
//
// 起因是用户问「运行 AI 会调用系统工作集吗」。答完发现 v1.0.21 的量化判定里
// 有一个真 bug：sysmem.js 的 windowsCommit 返回 pagefileBytes: null，注释写着
// “由调用方用提交上限 - 物理内存推算”——但调用方根本没实现这个推算，直接
// `Number(commit?.pagefileBytes) || 0` 把 null 压成了 0。
//
// 后果：Windows（也就是用户的机器）上 pagefileGB 恒为 0，于是 fits:'spill'
// （🐌 靠页面文件硬顶）**永远不可能出现**——配了 30 GB 页面文件、勉强能跑的档位
// 被一律标成 ⛔「装不下」。方向安全（不会骗人说能跑），但没说实话。
import test from 'node:test';
import assert from 'node:assert/strict';
import { readMemoryBudget } from '../server/native/sysmem.js';
import { pickQuantForBudget } from '../server/native/catalog.js';

// 用户那台机器：16.6 GB 总内存 / 7.4 GB 可用
const BASE = { totalGB: 16.6, freeGB: 7.4, commitLimitGB: 16.6 };
const GROUPED = [
  { quant: 'Q4_K_M', totalBytes: 15.5e9 }, // 约需 19.4 GB
  { quant: 'Q3_K_M', totalBytes: 12e9 },
  { quant: 'Q2_K', totalBytes: 8.5e9 },
];
const fitOf = (r, q) => r.candidates.find((c) => c.quant === q)?.fits;

test('pagefileBytes 三态：0 = 确认没配，null = 查不到，不能被压成同一个值', () => {
  for (const platform of ['win32', 'linux', 'darwin', 'freebsd', null]) {
    const b = readMemoryBudget({ forcePlatform: platform });
    assert.ok(b.pagefileBytes === null || b.pagefileBytes >= 0, `${platform}: 页面文件只能是 null 或非负`);
    // pagefileKnown 必须与 pagefileBytes 保持一致
    assert.equal(b.pagefileKnown, b.pagefileBytes !== null, `${platform}: pagefileKnown 与实测值不一致`);
  }
});

test('配了页面文件 → 该档判 spill（能跑但慢），不是 never', () => {
  const r = pickQuantForBudget(GROUPED, {
    ...BASE, pagefileGB: 20, commitLimitGB: 36, commitFreeGB: 30,
  });
  // 19.4 GB 装不进 16.6-2.5 的物理内存线，但 20 GB 页面文件兜得住
  assert.equal(fitOf(r, 'Q4_K_M'), 'spill');
});

test('没配页面文件 → 同一档判 never（没有硬盘兜底）', () => {
  const r = pickQuantForBudget(GROUPED, { ...BASE, pagefileGB: 0 });
  assert.equal(fitOf(r, 'Q4_K_M'), 'never');
});

test('页面文件查不到（null）→ 不能当成“有兜底”，判 never 并说明可能偏保守', () => {
  const r = pickQuantForBudget(GROUPED, { ...BASE, pagefileGB: null });
  assert.equal(fitOf(r, 'Q4_K_M'), 'never', '查不到时不能声称能靠页面文件跑');
  // 文案要说清是“没探测到”，而不是断言“没配”
  const only = pickQuantForBudget([{ quant: 'Q4_K_M', totalBytes: 15.5e9 }], { ...BASE, pagefileGB: null });
  assert.match(only.reason, /未能探测/);
  assert.doesNotMatch(only.reason, /未配页面文件/, '查不到 ≠ 没配，不能下这个结论');
});

test('页面文件够大但余量被吃光 → 仍判 never（上限够 ≠ 现在申请得到）', () => {
  // 每次踩坑都值得写成用例：以前只看「可提交上限」，30 GB 页面文件看着很够，
  // 但其它程序已经吃掉 30 GB，模型照样申请不到内存，加载时直接 bad_alloc。
  const r = pickQuantForBudget(GROUPED, {
    ...BASE, pagefileGB: 20, commitLimitGB: 36, commitFreeGB: 1,
  });
  assert.equal(fitOf(r, 'Q4_K_M'), 'never');
});

test('未配页面文件的文案要写明“没有硬盘兜底”', () => {
  const r = pickQuantForBudget([{ quant: 'Q4_K_M', totalBytes: 15.5e9 }], { ...BASE, pagefileGB: 0 });
  assert.match(r.reason, /未配页面文件|没有硬盘兜底/);
});

// ——— Windows 的解析：三种 PowerShell 返回必须分出三种结论 ———
// 沙箱是 Linux，跑不了 PowerShell，所以把“解析”这一步单独抽出来验证。
// 这段逻辑以前是 `Number(j.pagefileMB)`，而 Number(null) === 0 ——
// 查询失败会被当成“确认没配页面文件”，结论从“可能偏保守”变成“肯定没兜底”。
function parseWindowsPagefile(j) {
  let pagefileBytes = null;
  if (j.pagefileKnown === true) {
    const mb = Number(j.pagefileMB);
    pagefileBytes = Number.isFinite(mb) && mb >= 0 ? Math.round(mb * 1024 * 1024) : 0;
  }
  return pagefileBytes;
}

test('Windows 解析：查询成功且有页面文件 → 字节数', () => {
  assert.equal(parseWindowsPagefile({ pagefileKnown: true, pagefileMB: 20480 }), 20480 * 1024 * 1024);
});

test('Windows 解析：查询成功但没有页面文件 → 0（确认没配）', () => {
  assert.equal(parseWindowsPagefile({ pagefileKnown: true, pagefileMB: null }), 0);
  assert.equal(parseWindowsPagefile({ pagefileKnown: true, pagefileMB: 0 }), 0);
});

test('Windows 解析：查询失败 → null（不知道），绝不能变成 0', () => {
  // 这是本次修复的核心：Number(null) === 0，旧写法会把“不知道”说成“没有”
  assert.equal(parseWindowsPagefile({ pagefileKnown: false, pagefileMB: null }), null);
  assert.equal(parseWindowsPagefile({}), null, '缺少 pagefileKnown 时按“不知道”处理');
});
