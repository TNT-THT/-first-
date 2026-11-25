// CPU 指令集探测与“该装哪个 llama.cpp 构建包”。
//
// 背景：旧版 llama.cpp 按指令集分包（avx512/avx2/avx/noavx），挑错档位就是
// “非法指令”或零输出崩溃。用户实测“只有 avx2 版能用”，而 b4000 确实只发这四种。
// 要求是“任何机器都能装上能用的版本”，所以这里不存任何 CPU 型号表，
// 全靠运行时实测（cpu.js 的 binaryRuns）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import {
  binaryRuns, CPU_LEVELS, detectCpuLevel, findCpuVariants, installBestCpuVariant, parseCpuFeatures,
} from '../server/native/cpu.js';
import { pickReleaseAsset } from '../server/native/llama.js';

/** b4000 那种“按指令集分包”的资产列表（取自真实发布）。 */
const B4000_ASSETS = [
  'llama-b4000-bin-win-avx-x64.zip',
  'llama-b4000-bin-win-avx2-x64.zip',
  'llama-b4000-bin-win-avx512-x64.zip',
  'llama-b4000-bin-win-noavx-x64.zip',
  'llama-b4000-bin-win-kompute-x64.zip',
  'llama-b4000-bin-win-openblas-x64.zip',
  'llama-b4000-bin-win-cuda-cu12.2.0-x64.zip',
  'cudart-llama-bin-win-cu12.2.0-x64.zip',
].map((name) => ({ name }));

/** 新版那种单包列表。 */
const MODERN_ASSETS = [
  'llama-b12686-bin-win-cpu-x64.zip',
  'llama-b12686-bin-win-cuda-13.5-x64.zip',
  'llama-b12686-bin-win-vulkan-x64.zip',
  'cudart-llama-bin-win-cuda-13.5-x64.zip',
].map((name) => ({ name }));

const win = { platform: 'win32', arch: 'x64' };

test('挑包：新版单包直接命中 bin-win-cpu，不会误挑 cuda/vulkan', () => {
  assert.equal(pickReleaseAsset(MODERN_ASSETS, win).name, 'llama-b12686-bin-win-cpu-x64.zip');
});

test('挑包：旧版按指令集分包时按 CPU 档位挑，绝不随手 grab 一个', () => {
  // 这是本次要修的核心问题：以前找不到 bin-win-cpu 就 /bin-win-.*-x64/ 随便挑，
  // 可能挑中 cuda 版（没 N 卡直接崩）或 avx512 版（老 CPU 非法指令）。
  const cases = [
    ['avx512', 'llama-b4000-bin-win-avx512-x64.zip'],
    ['avx2', 'llama-b4000-bin-win-avx2-x64.zip'],
    ['avx', 'llama-b4000-bin-win-avx-x64.zip'],
    ['noavx', 'llama-b4000-bin-win-noavx-x64.zip'],
  ];
  for (const [cpuLevel, want] of cases) {
    assert.equal(pickReleaseAsset(B4000_ASSETS, { ...win, cpuLevel }).name, want, `${cpuLevel} 应挑 ${want}`);
  }
  // 未知档位时默认 avx2（2013 年后的 x64 CPU 基本都支持），而且挑出来的必须是 CPU 版
  const fallback = pickReleaseAsset(B4000_ASSETS, { ...win, cpuLevel: 'mystery' });
  assert.equal(fallback.name, 'llama-b4000-bin-win-avx2-x64.zip');
  assert.ok(!/cuda|kompute|openblas/.test(fallback.name), `绝不能挑 GPU/实验性包：${fallback.name}`);
});

test('挑包：不支持的高档位会自动降级到能用的那一档', () => {
  // 只有 avx 和 noavx 两种包（模拟只发了这两档的构建）
  const only = [
    'llama-b4000-bin-win-avx-x64.zip',
    'llama-b4000-bin-win-noavx-x64.zip',
  ].map((name) => ({ name }));
  // 想要 avx512，没有 → 一路降到 avx
  assert.equal(pickReleaseAsset(only, { ...win, cpuLevel: 'avx512' }).name, 'llama-b4000-bin-win-avx-x64.zip');
});

test('实测：一个能跑和一个跑不了的二进制，binaryRuns 判断正确', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-cpu-'));
  const good = path.join(dir, 'good');
  const bad = path.join(dir, 'bad');
  await writeFile(good, '#!/usr/bin/env node\nconsole.log("version: 4000 (abc)");\n');
  await writeFile(bad, '#!/usr/bin/env node\nprocess.exit(3221225477);\n');
  await chmod(good, 0o755);
  await chmod(bad, 0o755);
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });

  assert.equal(binaryRuns(good).ok, true);
  const badResult = binaryRuns(bad);
  assert.equal(badResult.ok, false);
  assert.equal(badResult.silent, true, '零输出崩溃必须被识别为 silent');
  // 不存在的文件也不能抛异常
  assert.equal(binaryRuns(path.join(dir, 'nope')).ok, false);
});

test('实测：从高到低试，第一个能跑的就是这台机器的上限', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-cpu2-'));
  const mk = async (level, works) => {
    const p = path.join(dir, level);
    await writeFile(p, works
      ? '#!/usr/bin/env node\nconsole.log("version: 4000");\n'
      : '#!/usr/bin/env node\nprocess.exit(3221225477);\n');
    await chmod(p, 0o755);
    return { level, path: p };
  };
  // avx512 跑不了、avx2 能跑（正是用户那台机器的情形）
  const candidates = [await mk('avx512', false), await mk('avx2', true), await mk('avx', true)];
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });

  const detected = detectCpuLevel(candidates);
  assert.equal(detected.level, 'avx2', 'avx512 跑不了就该落到 avx2');
  assert.equal(detected.probed, true);
  assert.equal(detected.results[0].level, 'avx512');
  assert.equal(detected.results[0].ok, false);
});

test('实测：全都跑不了时返回 unknown，而不是假装成功', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-cpu3-'));
  const bad = path.join(dir, 'bad');
  await writeFile(bad, '#!/usr/bin/env node\nprocess.exit(1);\n');
  await chmod(bad, 0o755);
  t.after(async () => { await rm(dir, { recursive: true, force: true }); });
  const r = detectCpuLevel([{ level: 'avx2', path: bad }]);
  assert.equal(r.level, 'unknown');
  assert.equal(r.probed, true);
});

test('安装策略：装完实测，跑不了就降一档重来（不依赖型号表）', async () => {
  // 模拟：avx512 装上跑不了，avx2 装上能跑
  const tried = [];
  const result = await installBestCpuVariant(['avx512', 'avx2', 'avx', 'noavx'], async (level) => {
    tried.push(level);
    return level === 'avx2';
  });
  assert.equal(result.ok, true);
  assert.equal(result.level, 'avx2');
  assert.deepEqual(tried, ['avx512', 'avx2'], '应按从高到低顺序试，到能跑为止');
});

test('安装策略：全部档位都失败时如实返回失败', async () => {
  const result = await installBestCpuVariant(['avx2', 'avx'], async () => false);
  assert.equal(result.ok, false);
  assert.equal(result.level, null);
  assert.equal(result.tried.length, 2);
});

test('安装策略：发布里没有的档位不浪费时间去试', async () => {
  const tried = [];
  await installBestCpuVariant(['avx'], async (level) => { tried.push(level); return true; });
  assert.deepEqual(tried, ['avx'], '只有 avx 可用时不该先去试 avx512/avx2');
});

test('目录扫描：能从 llama-b4000-bin-win-avx2-x64 这种目录名认出档位', async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), 'oj-cpu4-'));
  for (const level of ['avx512', 'avx2', 'noavx']) {
    const d = path.join(root, `llama-b4000-bin-win-${level}-x64`);
    await mkdir(d, { recursive: true });
    await writeFile(path.join(d, 'llama-server.exe'), 'x');
  }
  t.after(async () => { await rm(root, { recursive: true, force: true }); });
  const found = await findCpuVariants(root);
  const levels = found.map((f) => f.level).sort();
  assert.deepEqual(levels, ['avx2', 'avx512', 'noavx']);
});

test('CPU 特性解析：读得出 AVX/AVX2，读不到就返回 null（不猜）', () => {
  const f = parseCpuFeatures('system_info: CPU : SSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1');
  assert.deepEqual(f, { avx: true, avx2: true, avx512: false });
  assert.equal(parseCpuFeatures('没有任何特性行'), null);
  assert.equal(parseCpuFeatures(''), null);
  // 只有 AVX 没有 AVX2 的老 CPU 也要判对
  assert.deepEqual(parseCpuFeatures('CPU : SSE3 = 1 | AVX = 1 | AVX2 = 0'), { avx: true, avx2: false, avx512: false });
});

test('档位顺序必须是 avx512 → avx2 → avx → noavx（从快到慢）', () => {
  assert.deepEqual(CPU_LEVELS, ['avx512', 'avx2', 'avx', 'noavx']);
});
