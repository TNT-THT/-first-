/**
 * GitHub Release 下载链接配置与拼接（纯逻辑，便于测试）。
 *
 * 网站包（四平台）与一站式训练包不再随 GitHub Pages 一起分发，而是发布在
 * 仓库的 GitHub Release 中：网页按钮按 config.json 的 `release` 配置生成直达
 * 下载链接。部署者可把 `release.repo` 改成自己的仓库，把 `release.tag` 改成
 * 固定版本号（如 "v1.2.0"）以锁定版本；缺省使用最新版（latest）。
 */

/** 缺省的 Release 所在仓库（owner/name）。 */
export const DEFAULT_RELEASE_REPO = 'TNT-THT/-first-';
/** 四平台网站包的文件名，与 scripts/site-files.mjs 的 siteZipName() 一致。 */
export const SITE_PACKAGE_ASSETS = [
  'site-win-x64.zip',
  'site-darwin-arm64.zip',
  'site-darwin-x64.zip',
  'site-linux-x64.zip',
];
/** 一站式训练包的文件名，与 scripts/pack-training.mjs 一致。 */
export const TRAINING_PACKAGE_ASSET = 'oj-training.zip';
/** 独立软件「LLM托管平台——天蓝色1.0」的文件名，与 standalone/build-standalone.mjs 一致。 */
export const STANDALONE_PACKAGE_ASSET = 'LLM托管平台-天蓝色1.0.zip';
export const STANDALONE_EXE_ASSET = 'LLM托管平台-天蓝色1.0.exe';
/** 独立软件的两个下载选项：完整压缩包与单文件 exe。 */
export const STANDALONE_ASSETS = [STANDALONE_PACKAGE_ASSET, STANDALONE_EXE_ASSET];

/** 是否为合法的 GitHub 仓库标识 owner/name（避免把任意字符串拼进 URL）。 */
export function isValidReleaseRepo(repo) {
  return /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(String(repo || ''));
}

/** 是否为合法的 Release 标签（单个路径段）。 */
export function isValidReleaseTag(tag) {
  return /^[A-Za-z0-9_.-]+$/.test(String(tag || '')) && tag !== '.' && tag !== '..';
}

/**
 * 归一化 config.json 的 `release` 配置。
 * repo 非法时回退到缺省仓库；tag 缺省/非法时用 latest。
 * @param {unknown} raw
 * @returns {{ repo: string, tag: string }}
 */
export function normalizeReleaseConfig(raw) {
  const cfg = raw && typeof raw === 'object' ? raw : {};
  const repo = isValidReleaseRepo(cfg.repo) ? cfg.repo : DEFAULT_RELEASE_REPO;
  const tag = cfg.tag == null || cfg.tag === '' ? 'latest' : String(cfg.tag);
  return { repo, tag: tag === 'latest' || isValidReleaseTag(tag) ? tag : 'latest' };
}

/**
 * Release 资源直达下载地址。
 * tag 为 latest 时用 /releases/latest/download/，固定版本用 /releases/download/<tag>/。
 */
export function releaseAssetUrl(release, asset) {
  const { repo, tag } = normalizeReleaseConfig(release);
  const name = encodeURIComponent(String(asset || ''));
  return tag === 'latest'
    ? `https://github.com/${repo}/releases/latest/download/${name}`
    : `https://github.com/${repo}/releases/download/${encodeURIComponent(tag)}/${name}`;
}

/** Release 列表/版本页地址（查看全部下载）。 */
export function releasePageUrl(release) {
  const { repo, tag } = normalizeReleaseConfig(release);
  return tag === 'latest'
    ? `https://github.com/${repo}/releases/latest`
    : `https://github.com/${repo}/releases/tag/${encodeURIComponent(tag)}`;
}

/** 是否为 Release 中发布的资源文件名（网站包、训练包或独立软件）。 */
export function isReleaseAsset(file) {
  return SITE_PACKAGE_ASSETS.includes(file) || file === TRAINING_PACKAGE_ASSET || STANDALONE_ASSETS.includes(file);
}
