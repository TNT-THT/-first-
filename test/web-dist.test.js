// 回归：软件版/独立软件启动后打开首页必须是现成界面，绝不能出现
// “前端尚未构建，请先执行 npm run build:remote”这类要用户先跑命令的提示。
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp, MISSING_WEB_HTML, resolveWebDist } from '../server/index.js';
import { loadConfig } from '../server/config.js';

test('resolveWebDist：显式指定优先，缺文件时按 dist-api → dist 顺延', () => {
  const baseDir = path.join(path.sep, 'srv');
  const api = path.resolve(baseDir, '..', 'web', 'dist-api');
  const dist = path.resolve(baseDir, '..', 'web', 'dist');
  const custom = path.join(path.sep, 'custom-web');
  const only = (...ok) => (p) => ok.includes(p);

  // 显式目录有 index.html 时直接使用（独立软件启动器的通道）
  assert.equal(
    resolveWebDist({ webDist: custom, env: {}, baseDir, exists: only(path.join(custom, 'index.html')) }),
    custom,
  );
  // 显式目录缺文件时不死守，继续找现成产物
  assert.equal(
    resolveWebDist({ webDist: custom, env: {}, baseDir, exists: only(path.join(api, 'index.html'), path.join(dist, 'index.html')) }),
    api,
  );
  // 环境变量通道
  assert.equal(
    resolveWebDist({ env: { OJ_WEB_DIST: custom }, baseDir, exists: only(path.join(custom, 'index.html'), path.join(dist, 'index.html')) }),
    custom,
  );
  // 只有静态版产物时用静态版
  assert.equal(
    resolveWebDist({ env: {}, baseDir, exists: only(path.join(dist, 'index.html')) }),
    dist,
  );
  // 全都没有时返回首个候选（交给兜底页处理），且候选是 dist-api
  assert.equal(resolveWebDist({ env: {}, baseDir, exists: () => false }), api);
});

test('loadConfig：OJ_WEB_DIST 被接住，不再静默丢失', () => {
  const cfg = loadConfig({ OJ_WEB_DIST: '/somewhere/web' });
  assert.equal(cfg.webDist, '/somewhere/web');
  assert.equal(loadConfig({}).webDist, '');
});

test('兜底页：不含任何要用户执行的开发命令', () => {
  assert.ok(!/npm run|build:remote/.test(MISSING_WEB_HTML));
  assert.match(MISSING_WEB_HTML, /双击即可使用/);
});

test('createApp：通过配置拿到网页目录时直接提供界面（软件版真实路径）', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-webdist-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  await writeFile(path.join(dir, 'index.html'), '<!doctype html><title>软件界面</title><div id="app">ok</div>');
  const cfg = loadConfig({
    APP_PASSWORD: 'test-password',
    SESSION_SECRET: 'test-secret-test-secret',
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
    USAGE_STATS_FILE: path.join(dir, 'usage.json'),
    OJ_WEB_DIST: dir,
  });
  assert.equal(cfg.webDist, dir);
  const app = createApp(cfg, {});
  const server = await new Promise((resolve) => {
    const srv = app.listen(0, '127.0.0.1', () => resolve(srv));
  });
  t.after(() => new Promise((r) => server.close(r)));
  const base = `http://127.0.0.1:${server.address().port}`;

  const home = await fetch(`${base}/`);
  assert.equal(home.status, 200);
  const html = await home.text();
  assert.match(html, /软件界面/);
  assert.ok(!html.includes('npm run build:remote'), '首页绝不能再让用户先执行构建命令');

  // SPA 深链同样落回界面
  const deep = await fetch(`${base}/some/deep/link`);
  assert.equal(deep.status, 200);
  assert.match(await deep.text(), /软件界面/);
});
