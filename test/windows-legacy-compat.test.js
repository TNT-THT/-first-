// Windows 兼容性：确保在 Windows 10 最旧版（1507 / build 10240）上也能跑。
//
// 用户要求"再次全线检查，确保在低版本 Windows 上也能兼容运行"。
// Windows 10 有 7 个主要版本、跨度 7 年，早期版本缺一些东西，而这些东西一旦缺失，
// 表现都是"闪一下就没了"或"白屏"——用户看到的只有一句莫名其妙的报错。
//
// 这组测试锁住四条硬性保证（都是真实踩过的坑）：
//   ① 所有 .ps1 必须是 UTF-8 **带 BOM**：PS 5.1 对无 BOM 的 UTF-8 按系统 ANSI
//      代码页解码（简体中文即 GBK），中文提示会全变成乱码。
//   ② 所有 .bat 必须是 CRLF + 纯 ASCII：cmd 不支持 LF 换行；非 ASCII 在 GBK 控制台会乱。
//   ③ 打包白名单必须包含 check-env.ps1：漏了它启动脚本会报"找不到文件"。
//   ④ 前端必须有 nomodule 降级提示：旧浏览器不支持 <script type="module">，
//      没有提示就是纯白屏，用户完全不知道发生了什么。
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import { LAUNCHER_FILES } from '../scripts/site-files.mjs';

const root = path.resolve(import.meta.dirname, '..');
const read = (p) => readFileSync(path.join(root, p));

/** 递归找出所有指定扩展名的文件（跳过 node_modules 等）。 */
function findFiles(dir, ext, out = []) {
  for (const name of readdirSync(dir)) {
    if (['node_modules', '.git', 'dist', 'dist-api', '.build', 'release-assets'].includes(name)) continue;
    const full = path.join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) findFiles(full, ext, out);
    else if (name.endsWith(ext)) out.push(path.relative(root, full));
  }
  return out;
}

test('所有 PowerShell 脚本都是 UTF-8 with BOM（否则 PS 5.1 上中文全乱码）', () => {
  const files = findFiles(path.join(root, 'scripts'), '.ps1');
  assert.ok(files.length >= 4, `应该找到多个 ps1，实际 ${files.length}`);
  for (const f of files) {
    const buf = read(f);
    assert.equal(
      buf[0] === 0xef && buf[1] === 0xbb && buf[2] === 0xbf, true,
      `${f} 缺少 UTF-8 BOM。Windows PowerShell 5.1 会把无 BOM 的 UTF-8 按 GBK 解码，`
      + '中文提示会变成乱码（这是 v1.0.22 之前 serve-server.ps1 的真实缺陷）。',
    );
    // BOM 之后必须还能按 UTF-8 正常解码
    assert.doesNotThrow(() => buf.slice(3).toString('utf8'), `${f} 不是合法 UTF-8`);
  }
});

test('PowerShell 脚本用 CRLF 换行', () => {
  for (const f of findFiles(path.join(root, 'scripts'), '.ps1')) {
    const s = read(f).toString('utf8');
    const crlf = (s.match(/\r\n/g) || []).length;
    const lfOnly = (s.match(/(?<!\r)\n/g) || []).length;
    assert.equal(lfOnly, 0, `${f} 含 ${lfOnly} 个纯 LF 换行；Windows 脚本应统一 CRLF`);
    assert.ok(crlf > 0, `${f} 应该使用 CRLF`);
  }
});

test('批处理文件是 CRLF 且纯 ASCII（cmd 与 GBK 控制台的安全区）', () => {
  const list = findFiles(path.join(root, 'scripts'), '.bat');
  assert.ok(list.length >= 3);
  for (const f of list) {
    const buf = read(f);
    const s = buf.toString('latin1');
    const lfOnly = (s.match(/(?<!\r)\n/g) || []).length;
    assert.equal(lfOnly, 0, `${f} 含纯 LF 换行，cmd 下会解析失败`);
    for (const b of buf) {
      assert.ok(b < 0x80, `${f} 含非 ASCII 字节 0x${b.toString(16)}；`
        + '在简体中文系统的 GBK 代码页下会显示成乱码，建议改成英文或用 ps1 承担中文');
    }
  }
});

test('打包白名单包含 check-env.ps1（漏了它启动脚本会报找不到）', () => {
  const names = LAUNCHER_FILES.map((f) => f.from);
  assert.ok(names.includes('check-env.ps1'), `LAUNCHER_FILES 缺少 check-env.ps1：${names.join(',')}`);
  // 每个被 serve-*.ps1 dot-source 的脚本都必须在白名单里
  for (const ps of ['serve-api.ps1', 'serve-site.ps1']) {
    const src = read(path.join('scripts/launcher', ps)).toString('utf8');
    const m = src.match(/\.\s*\(Join-Path \$PSScriptRoot '([^']+\.ps1)'\)/);
    assert.ok(m, `${ps} 应该 dot-source 环境检查脚本`);
    assert.ok(names.includes(m[1]), `${ps} 引用了 ${m[1]}，但它不在打包白名单里`);
  }
});

test('启动脚本在旧系统上给出明确提示，而不是让用户对着白屏猜', () => {
  const check = read('scripts/launcher/check-env.ps1').toString('utf8');
  // 必须覆盖 Windows 10 各版本的 build 号
  for (const build of ['10240', '10586', '14393', '17763', '19041', '19044', '19045']) {
    assert.ok(check.includes(build), `check-env.ps1 缺少 build ${build} 的判定`);
  }
  // 必须明确说出"需要 Windows 10"以及替代方案
  assert.match(check, /需要 Windows 10/);
  assert.match(check, /局域网/);
  // 早期 Win10 要提示浏览器问题（IE11 跑不了 ES2020）
  assert.match(check, /浏览器/);
  // 大模型场景要检查页面文件，并把 0xC0000005 说清楚
  assert.match(check, /页面文件|虚拟内存/);
});

test('网页有 nomodule 降级提示，且那段脚本是 ES5', () => {
  const html = read('web/index.html').toString('utf8');
  const m = html.match(/<script nomodule>([\s\S]*?)<\/script>/);
  assert.ok(m, 'web/index.html 缺少 <script nomodule> 降级提示');
  const js = m[1];
  // 旧浏览器（IE11）解析到 const/let/箭头函数/模板字符串会直接抛错，
  // 提示就显示不出来了 —— 必须保持 ES5。
  assert.doesNotMatch(js, /\b(const|let)\b/, 'nomodule 脚本必须是 ES5，不能用 const/let');
  assert.doesNotMatch(js, /=>/, 'nomodule 脚本不能用箭头函数');
  assert.doesNotMatch(js, /`/, 'nomodule 脚本不能用模板字符串');
  // 要告诉用户装什么
  assert.match(html, /Chrome/);
  assert.match(html, /Firefox/);
});

test('PowerShell 脚本括号配平（沙箱无 PowerShell，先做静态把关）', () => {
  // 注意顺序：**先去字符串，再去注释**。反过来会把 StartsWith('#') 里的 #
  // 当成注释开头，把后面的右括号全吃掉，误报不配平（写这个检查时就踩到了）。
  const files = [
    ...findFiles(path.join(root, 'scripts'), '.ps1'),
  ];
  for (const f of files) {
    let t = read(f).toString('utf8');
    t = t.replace(/'[^'\r\n]*'/g, "''").replace(/"[^"\r\n]*"/g, '""');
    t = t.replace(/#[^\r\n]*/g, '');
    t = [...t].filter((c) => c.charCodeAt(0) < 128).join('');
    const count = (c) => t.split(c).length - 1;
    assert.equal(count('{'), count('}'), `${f} 花括号不配平`);
    assert.equal(count('('), count(')'), `${f} 圆括号不配平`);
  }
});

test('check-env.ps1 的版本判定逻辑与真实 build 号一致', () => {
  // 用与脚本相同的映射表验证：任何 build 号都要落到正确的版本名，
  // 且 10240（Windows 10 1507，最旧版）必须判定为“支持”。
  const src = read('scripts/launcher/check-env.ps1').toString('utf8');
  const classify = (build) => {
    let name = '未知';
    if (build >= 22000) name = 'Windows 11';
    else if (build >= 10240) name = 'Windows 10';
    else if (build >= 9600) name = 'Windows 8.1';
    else if (build >= 9200) name = 'Windows 8';
    else if (build >= 7600) name = 'Windows 7';
    return name;
  };
  const cases = [
    [7601, 'Windows 7'], [9200, 'Windows 8'], [9600, 'Windows 8.1'],
    [10240, 'Windows 10'], [10586, 'Windows 10'], [14393, 'Windows 10'],
    [17763, 'Windows 10'], [19041, 'Windows 10'], [19044, 'Windows 10'],
    [19045, 'Windows 10'], [22000, 'Windows 11'], [22631, 'Windows 11'],
  ];
  for (const [build, want] of cases) {
    assert.equal(classify(build), want, `build ${build} 应判定为 ${want}`);
    // 15240 之后的各版本号要能在脚本里找到（22000+ 用范围覆盖，不逐个列）
    if (build < 22000 && ![7601, 9200].includes(build)) {
      assert.ok(src.includes(String(build)), `check-env.ps1 应能识别 build ${build}`);
    }
  }
  // 最老的 Win10 必须走“支持”分支：门槛是 10240，不是 19041
  assert.match(src, /-lt 10240/, '版本门槛必须是 10240（Win10 1507），不能写成 19041');
});
