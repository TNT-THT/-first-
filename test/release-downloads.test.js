// GitHub Release 下载链接配置：归一化、URL 拼接与 config.json 现行配置守卫。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  DEFAULT_RELEASE_REPO,
  SITE_PACKAGE_ASSETS,
  TRAINING_PACKAGE_ASSET,
  isReleaseAsset,
  isValidReleaseRepo,
  isValidReleaseTag,
  normalizeReleaseConfig,
  releaseAssetUrl,
  releasePageUrl,
} from '../web/src/release-downloads.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

test('normalizeReleaseConfig：缺省指向本仓库的 latest', () => {
  assert.deepEqual(normalizeReleaseConfig(undefined), { repo: DEFAULT_RELEASE_REPO, tag: 'latest' });
  assert.deepEqual(normalizeReleaseConfig(null), { repo: DEFAULT_RELEASE_REPO, tag: 'latest' });
  assert.deepEqual(normalizeReleaseConfig({}), { repo: DEFAULT_RELEASE_REPO, tag: 'latest' });
  assert.deepEqual(normalizeReleaseConfig({ tag: '' }), { repo: DEFAULT_RELEASE_REPO, tag: 'latest' });
});

test('normalizeReleaseConfig：部署者可改仓库与固定版本，非法值回退', () => {
  assert.deepEqual(normalizeReleaseConfig({ repo: 'my-org/my-fork', tag: 'v1.2.0' }), {
    repo: 'my-org/my-fork',
    tag: 'v1.2.0',
  });
  assert.equal(normalizeReleaseConfig({ repo: 'not a repo' }).repo, DEFAULT_RELEASE_REPO);
  assert.equal(normalizeReleaseConfig({ repo: 'https://evil.example/x' }).repo, DEFAULT_RELEASE_REPO);
  assert.equal(normalizeReleaseConfig({ repo: 'a/b/c' }).repo, DEFAULT_RELEASE_REPO);
  assert.equal(normalizeReleaseConfig({ tag: '../escape' }).tag, 'latest');
  assert.equal(normalizeReleaseConfig({ tag: 'v1/2' }).tag, 'latest');
  assert.equal(normalizeReleaseConfig({ tag: 'latest' }).tag, 'latest');
});

test('isValidReleaseRepo / isValidReleaseTag', () => {
  assert.equal(isValidReleaseRepo('TNT-THT/-first-'), true);
  assert.equal(isValidReleaseRepo('owner/name.with.dots'), true);
  assert.equal(isValidReleaseRepo('noslash'), false);
  assert.equal(isValidReleaseRepo(''), false);
  assert.equal(isValidReleaseTag('v1.0.0'), true);
  assert.equal(isValidReleaseTag('..' ), false);
  assert.equal(isValidReleaseTag('a/b'), false);
});

test('releaseAssetUrl：latest 与固定版本的路径不同', () => {
  assert.equal(
    releaseAssetUrl(undefined, 'site-win-x64.zip'),
    `https://github.com/${DEFAULT_RELEASE_REPO}/releases/latest/download/site-win-x64.zip`,
  );
  assert.equal(
    releaseAssetUrl({ repo: 'o/r', tag: 'v2.0.0' }, 'oj-training.zip'),
    'https://github.com/o/r/releases/download/v2.0.0/oj-training.zip',
  );
  assert.equal(
    releaseAssetUrl({ repo: 'bad repo', tag: 'v2' }, 'site-linux-x64.zip'),
    `https://github.com/${DEFAULT_RELEASE_REPO}/releases/download/v2/site-linux-x64.zip`,
  );
});

test('releasePageUrl：latest 与固定版本', () => {
  assert.equal(releasePageUrl(undefined), `https://github.com/${DEFAULT_RELEASE_REPO}/releases/latest`);
  assert.equal(releasePageUrl({ tag: 'v3.1.0' }), `https://github.com/${DEFAULT_RELEASE_REPO}/releases/tag/v3.1.0`);
});

test('isReleaseAsset：四平台网站包与训练包', () => {
  for (const asset of SITE_PACKAGE_ASSETS) assert.equal(isReleaseAsset(asset), true);
  assert.equal(isReleaseAsset(TRAINING_PACKAGE_ASSET), true);
  assert.equal(isReleaseAsset('config.json'), false);
  assert.equal(isReleaseAsset('../site-win-x64.zip'), false);
});

test('config.json 现行发布配置：指向本仓库 latest', () => {
  const cfg = JSON.parse(readFileSync(join(root, 'web', 'public', 'config.json'), 'utf8'));
  assert.deepEqual(normalizeReleaseConfig(cfg.release), { repo: 'TNT-THT/-first-', tag: 'latest' });
});
