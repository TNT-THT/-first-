// “能不能用硬盘/SSD 当内存”：唯一真实机制是页面文件（swap），对应指标是提交上限。
// 这组测试锁住三件事：① 探测不到就返回 null，绝不编数字；② 物理内存 / 提交上限 /
// 页面文件三种“不够”给出不同结论；③ 页面文件建议值算得对。
import test from 'node:test';
import assert from 'node:assert/strict';
import { planPagefile, readMemoryBudget, STORAGE_AS_MEMORY_NOTE } from '../server/native/sysmem.js';
import { NativeEngine } from '../server/native/engine.js';

test('内存预算：任何平台都不抛错，拿不到的信息就是 null/0，不编造', () => {
  for (const platform of ['win32', 'linux', 'darwin', 'freebsd', null]) {
    const b = readMemoryBudget({ forcePlatform: platform });
    assert.ok(b.totalBytes > 0, `${platform}: 物理内存应能读到`);
    assert.ok(b.freeBytes >= 0);
    assert.ok(b.commitLimitBytes === null || b.commitLimitBytes > 0);
    assert.equal(typeof b.canSpill, 'boolean');
    assert.equal(typeof b.source, 'string');
  }
});

test('内存预算：pagefileBytes 是实测值，不从“提交上限 − 物理内存”反推', () => {
  const b = readMemoryBudget();
  if (!b.commitLimitBytes) return; // 环境不支持，跳过
  // 这条用例最初写成了“commitLimit <= total 时 pagefileBytes 必为 0”，
  // 结果在 GitHub Actions 上失败：runner 有 3 GB swap，但 overcommit 模式下
  // CommitLimit 反而小于物理内存。用差值反推会得出负数或错值，所以改成
  // 只断言“不变量”：
  assert.ok(b.pagefileBytes >= 0, '页面文件大小不能是负数');
  // 有页面文件就应该标记为“硬盘能兜底”
  assert.equal(b.canSpill, b.pagefileBytes > 0);
  // 三者关系必须自洽：能兜底意味着提交上限 > 物理内存；不能兜底时最多等于物理内存
  if (b.canSpill) {
    assert.ok(b.commitLimitBytes >= b.totalBytes,
      `有页面文件时提交上限应不小于物理内存：${b.commitLimitBytes} vs ${b.totalBytes}`);
  }
});

test('内存预算：overcommit 下调紧的内核（提交上限 < 物理内存）也不能算出负数', async () => {
  // 直接验证算法本身：模拟 GitHub runner 那种“有 swap 但 CommitLimit 偏小”的输入，
  // planPagefile 不能因为提交上限小而给出负的建议值。
  const { planPagefile } = await import('../server/native/sysmem.js');
  const plan = planPagefile({ neededGB: 20, totalGB: 16, pagefileGB: 3 });
  assert.ok(plan.requiredPagefileGB >= 0, `建议值不能为负：${plan.requiredPagefileGB}`);
  assert.ok(plan.requiredCommitGB >= 20, '要有余量');
});

test('页面文件建议：按“需求 ×1.2 余量 − 物理内存”算，且给到具体数字', () => {
  // 用户机器：需 17.4 GB，物理 16.6 GB，没配页面文件
  const plan = planPagefile({ neededGB: 17.4, totalGB: 16.6, pagefileGB: 0 });
  assert.equal(plan.enough, false);
  assert.equal(plan.requiredCommitGB, 20.9); // 17.4 × 1.2
  assert.equal(plan.requiredPagefileGB, 4.3); // 20.9 − 16.6
  assert.match(plan.tip, /页面文件调到约 4\.3 GB/);
  assert.match(plan.tip, /慢|换更小的量化/, '必须说清代价，不能只说“能跑”');
  // 现有页面文件已经够：不该再让用户去改
  const ok = planPagefile({ neededGB: 17.4, totalGB: 16.6, pagefileGB: 32 });
  assert.equal(ok.enough, true);
  assert.match(ok.tip, /已经够/);
  // 物理内存本来就装得下：不该提页面文件
  const fits = planPagefile({ neededGB: 8, totalGB: 32, pagefileGB: 0 });
  assert.equal(fits.enough, true);
  assert.equal(fits.requiredPagefileGB, 0);
  // 信息不全时返回 null，不要瞎算
  assert.equal(planPagefile({ neededGB: 0, totalGB: 16 }), null);
  assert.equal(planPagefile({ neededGB: 16, totalGB: 0 }), null);
  assert.equal(planPagefile({}), null);
});

test('实话文案：必须同时说清机制、代价和正解（不能暗示硬盘能当内存用）', () => {
  assert.match(STORAGE_AS_MEMORY_NOTE, /页面文件|swap/);
  assert.match(STORAGE_AS_MEMORY_NOTE, /慢/);
  assert.match(STORAGE_AS_MEMORY_NOTE, /换更小的量化/);
});

test('内存体检：物理内存装不下但页面文件能兜住时，结论是“能跑但很慢”，不是“必须换机器”', async () => {
  const engine = new NativeEngine({ dataDir: '/tmp/oj-sysmem-t', logger: { warn() {}, log() {}, error() {} } });
  const r = engine._memoryReport({ totalBytes: 13.5e9, blockCount: 48 }, 0, 8192);
  assert.ok(r.neededGB > r.totalGB, '这个用例要求需求真的超过物理内存');
  // 结论必须分情况，且不能自相矛盾
  assert.equal(r.exceedsTotal, true);
  if (r.exceedsCommit) {
    assert.match(r.summary, /上限都超过|必须换更小的量化|换机器/);
  } else {
    assert.match(r.summary, /换页|慢/);
  }
  // 一定要带上“页面文件/swap 可提交上限”这个数字，用户才知道底线在哪
  if (r.commitLimitGB) assert.match(r.summary, /可提交上限/);
  // 并且给出可操作的页面文件建议
  assert.ok(r.pagefilePlan === null || typeof r.pagefilePlan.requiredPagefileGB === 'number');
});

test('内存体检：小模型不会因为提交上限偏低而被误报“装不下”', async () => {
  const engine = new NativeEngine({ dataDir: '/tmp/oj-sysmem-t2', logger: { warn() {}, log() {}, error() {} } });
  const tiny = engine._memoryReport(1.5e9, 0, 4096);
  assert.equal(tiny.exceedsTotal, false);
  assert.equal(tiny.tight, false);
  assert.ok(!/必须换更小的量化/.test(tiny.summary), tiny.summary);
});

// ---- 用户机器的真实参数（16.6 GB 总内存 / 7.4 GB 可用），把边界情形固化下来 ----
const USER_MACHINE = (pagefileGB) => {
  const totalGB = 16.6;
  return {
    totalBytes: totalGB * 1e9,
    freeBytes: 7.4 * 1e9,
    commitLimitBytes: (totalGB + pagefileGB) * 1e9,
    commitFreeBytes: (totalGB + pagefileGB - 2) * 1e9,
    pagefileBytes: pagefileGB * 1e9,
    canSpill: pagefileGB > 0,
    source: 'test',
  };
};
const Q2K_WEIGHTS = { totalBytes: 13.5e9, blockCount: 48 }; // 35B-A3B 的 Q2_K

test('用户机器：没配页面文件时，结论是“系统装不下，必须换量化”，并给出要开多大页面文件', () => {
  const engine = new NativeEngine({ dataDir: '/tmp/oj-sysmem-u1', logger: { warn() {}, log() {}, error() {} } });
  const r = engine._memoryReport(Q2K_WEIGHTS, 0, 8192, { budget: USER_MACHINE(0) });
  assert.equal(r.totalGB, 16.6);
  assert.equal(r.freeGB, 7.4);
  assert.equal(r.pagefileGB, 0);
  assert.equal(r.canSpill, false);
  assert.equal(r.neededGB, 17.4);
  // 需求 17.4 > 物理 16.6 > 提交上限 16.6 → 必须换量化
  assert.equal(r.exceedsTotal, true);
  assert.equal(r.exceedsCommit, true);
  assert.equal(r.spillOnly, false);
  assert.match(r.summary, /上限都超过|必须换更小的量化/);
  // 页面文件建议 ≈ 17.4×1.2 − 16.6 = 4.3 GB
  assert.equal(r.pagefilePlan.requiredPagefileGB, 4.3);
  assert.equal(r.pagefilePlan.enough, false);
});

test('用户机器：配了 32 GB 页面文件后，结论变成“能跑但极慢”，不再说“装不下”', () => {
  const engine = new NativeEngine({ dataDir: '/tmp/oj-sysmem-u2', logger: { warn() {}, log() {}, error() {} } });
  const r = engine._memoryReport(Q2K_WEIGHTS, 0, 8192, { budget: USER_MACHINE(32) });
  assert.equal(r.canSpill, true);
  assert.equal(r.spillOnly, true, '物理内存装不下但页面文件能兜住');
  assert.equal(r.exceedsCommit, false);
  assert.match(r.summary, /页面文件还能兜住/);
  assert.match(r.summary, /慢|换页|超时/, '必须说清代价');
  assert.match(r.summary, /建议换更小的量化/, '仍要给出正解');
  // 这句话绝不能出现：用户已经被“装不下”误导过一次
  assert.ok(!/就算关掉所有程序也装不下/.test(r.summary));
});

test('用户机器：换到真装得下的量化后，必须明确说“够用 / 关掉程序即可”，而不是继续喊换量化', () => {
  const engine = new NativeEngine({ dataDir: '/tmp/oj-sysmem-u3', logger: { warn() {}, log() {}, error() {} } });
  // 5.5 GB 权重 + 4096 上下文 → 需 7.9 GB：总内存 16.6 够，但当前只有 7.4 可用
  const tight = engine._memoryReport({ totalBytes: 5.5e9, blockCount: 32 }, 0, 4096, { budget: USER_MACHINE(0) });
  assert.equal(tight.exceedsTotal, false, '装得进物理内存就不能判“装不下”');
  assert.equal(tight.exceedsFree, true);
  assert.match(tight.summary, /关掉其它占内存的程序/);
  assert.ok(!/必须换更小的量化/.test(tight.summary), '总内存够的时候不该叫用户换量化');

  // 再小一档（3.5 GB 权重）：当前可用也够，必须直说够用
  const fits = engine._memoryReport({ totalBytes: 3.5e9, blockCount: 32 }, 0, 4096, { budget: USER_MACHINE(0) });
  assert.equal(fits.exceedsTotal, false);
  assert.equal(fits.exceedsFree, false);
  assert.equal(fits.tight, false);
  assert.match(fits.summary, /内存够用/);
});
