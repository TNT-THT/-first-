// 软件版窗口策略：优先独立应用窗口（Edge/Chrome --app），不打包任何新运行时；
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {
  appWindowArgs,
  expandEnvVars,
  findAppBrowser,
  openAppWindow,
  pickOpenPlan,
} from '../standalone/app-window.mjs';

const URL = 'http://127.0.0.1:3210/';

test('expandEnvVars：展开 %VAR% 且不区分大小写', () => {
  assert.equal(expandEnvVars('%ProgramFiles%\\Edge', { programfiles: 'C:\\PF' }), 'C:\\PF\\Edge');
  assert.equal(expandEnvVars('%NOPE%\\x', {}), '\\x');
});

test('findAppBrowser：Windows 优先 Edge，认 OJ_APP_BROWSER 覆盖', () => {
  const env = { 'ProgramFiles(x86)': 'C:\\PF86', ProgramFiles: 'C:\\PF' };
  const edge = path.join('C:\\PF86', 'Microsoft\\Edge\\Application\\msedge.exe');
  assert.equal(findAppBrowser({ platform: 'win32', env, exists: (p) => p === edge }), edge);
  assert.equal(findAppBrowser({ platform: 'win32', env: { OJ_APP_BROWSER: 'D:\\my\\edge.exe' }, exists: () => false }), 'D:\\my\\edge.exe');
});

test('findAppBrowser：Linux 用 which 探测，找不到返回 null', () => {
  const fakeWhich = (cmd, args) => ({ status: cmd === 'which' && args[0] === 'google-chrome' ? 0 : 1 });
  assert.equal(findAppBrowser({ platform: 'linux', env: {}, spawnSyncImpl: fakeWhich }), 'google-chrome');
  assert.equal(findAppBrowser({ platform: 'linux', env: {}, spawnSyncImpl: () => ({ status: 1 }) }), null);
});

test('pickOpenPlan：none / browser / app 与无 Chromium 时的回退', () => {
  assert.deepEqual(pickOpenPlan(URL, { env: { OJ_NO_OPEN_BROWSER: '1' } }), { kind: 'none' });
  assert.equal(pickOpenPlan(URL, { mode: 'none' }).kind, 'none');
  assert.equal(pickOpenPlan(URL, { mode: 'browser', platform: 'linux' }).kind, 'browser');
  const app = pickOpenPlan(URL, { platform: 'linux', env: {}, spawnSyncImpl: () => ({ status: 0 }) });
  assert.equal(app.kind, 'app');
  assert.ok(app.args.includes(`--app=${URL}`));
  const fallback = pickOpenPlan(URL, { platform: 'linux', env: {}, spawnSyncImpl: () => ({ status: 1 }) });
  assert.deepEqual(fallback, { kind: 'browser', command: 'xdg-open', args: [URL] });
});

test('appWindowArgs：是独立窗口参数而不是普通打开', () => {
  const args = appWindowArgs(URL);
  assert.equal(args[0], `--app=${URL}`);
  assert.ok(args.some((a) => a.startsWith('--window-size=')));
});

test('openAppWindow：按 app 方案启动，spawn 抛错时回退默认浏览器', () => {
  const calls = [];
  const child = { once() {}, unref() {} };
  const spawnImpl = (command, args) => { calls.push([command, args]); return child; };
  const plan = openAppWindow(URL, { platform: 'linux', env: {}, spawnSyncImpl: () => ({ status: 0 }), spawnImpl });
  assert.equal(plan.kind, 'app');
  assert.equal(calls.length, 1);

  const calls2 = [];
  let first = true;
  const flaky = (command, args) => {
    calls2.push([command, args]);
    if (first) { first = false; throw new Error('ENOENT'); }
    return child;
  };
  const plan2 = openAppWindow(URL, { platform: 'linux', env: {}, spawnSyncImpl: () => ({ status: 0 }), spawnImpl: flaky });
  assert.equal(plan2.kind, 'browser');
  assert.equal(plan2.fallback, true);
  assert.equal(calls2[1][0], 'xdg-open');

  // none 时完全不启动任何进程
  const calls3 = [];
  const plan3 = openAppWindow(URL, { env: { OJ_OPEN_MODE: 'none' }, spawnImpl: (c, a) => { calls3.push([c, a]); return child; } });
  assert.equal(plan3.kind, 'none');
  assert.equal(calls3.length, 0);
});
