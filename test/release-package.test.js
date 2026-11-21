// GitHub Release 打包：训练包内容白名单、发布资源命名一致性、模型下载清单与 CI 配置守卫。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TRAINING_PACKAGE_NAME, packTraining, selectTrainingFiles } from '../scripts/pack-training.mjs';
import { siteZipName } from '../scripts/site-files.mjs';
import { SITE_PACKAGE_ASSETS, TRAINING_PACKAGE_ASSET } from '../web/src/release-downloads.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

/** 读取 zip 中央目录中的条目名。 */
function zipEntryNames(buf) {
  let eocd = -1;
  for (let i = buf.length - 22; i >= 0; i--) {
    if (buf.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  assert.notEqual(eocd, -1, '应能找到 zip 中央目录');
  const count = buf.readUInt16LE(eocd + 10);
  let p = buf.readUInt32LE(eocd + 16);
  const names = [];
  for (let n = 0; n < count; n++) {
    const nameLen = buf.readUInt16LE(p + 28);
    const extraLen = buf.readUInt16LE(p + 30);
    const commentLen = buf.readUInt16LE(p + 32);
    names.push(buf.toString('utf8', p + 46, p + 46 + nameLen));
    p += 46 + nameLen + extraLen + commentLen;
  }
  return names;
}

test('发布资源命名：网页按钮、打包脚本与训练包名保持一致', () => {
  assert.deepEqual(SITE_PACKAGE_ASSETS, [
    siteZipName('win-x64'),
    siteZipName('darwin-arm64'),
    siteZipName('darwin-x64'),
    siteZipName('linux-x64'),
  ]);
  assert.equal(TRAINING_PACKAGE_ASSET, TRAINING_PACKAGE_NAME);

  const html = readFileSync(join(root, 'web', 'index.html'), 'utf8');
  for (const asset of SITE_PACKAGE_ASSETS) {
    assert.ok(html.includes(`data-site-package="${asset}"`), `index.html 应有 ${asset} 的下载按钮`);
  }
  assert.ok(html.includes('id="trainingPackageBtn"'), 'index.html 应有一站式训练包按钮');
  assert.ok(html.includes('id="releasePageLink"'), 'index.html 应有 Release 列表页链接');
});

test('训练包白名单：脚本/清单/示例入包，配置、数据与测试不入包', () => {
  const { included, excluded } = selectTrainingFiles([
    'ojtrain.py',
    'requirements.txt',
    'README.md',
    'catalog.json',
    'config.example.json',
    'config.json',
    'examples/sample.jsonl',
    'examples/notes.md',
    'tests/test_train.py',
    'data/train.jsonl',
    'models/base/config.json',
    '__pycache__/ojtrain.cpython-312.pyc',
    'doctor.pyc',
    'runs/out/adapter_model.safetensors',
  ]);
  for (const want of ['ojtrain.py', 'requirements.txt', 'README.md', 'catalog.json', 'config.example.json', 'examples/sample.jsonl', 'examples/notes.md']) {
    assert.ok(included.includes(want), `${want} 应进入训练包`);
  }
  for (const bad of ['config.json', 'tests/test_train.py', 'data/train.jsonl', 'models/base/config.json', '__pycache__/ojtrain.cpython-312.pyc', 'doctor.pyc', 'runs/out/adapter_model.safetensors']) {
    assert.ok(excluded.includes(bad), `${bad} 不应进入训练包`);
  }
});

test('训练包实打包：包含训练全链路文件且没有测试与缓存', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'oj-training-pack-'));
  try {
    const outPath = join(dir, TRAINING_PACKAGE_NAME);
    const { fileCount } = packTraining({ projectRoot: root, outPath });
    const names = zipEntryNames(await readFile(outPath));
    assert.equal(names.length, fileCount);
    for (const want of [
      'training/ojtrain.py',
      'training/requirements.txt',
      'training/README.md',
      'training/catalog.json',
      'training/config.example.json',
      'training/fetch_model.py',
      'training/train_lora.py',
      'training/pack_ojmodel.py',
      'training/pipeline.py',
      'training/doctor.py',
      'training/examples/sample.jsonl',
    ]) {
      assert.ok(names.includes(want), `训练包应包含 ${want}`);
    }
    for (const name of names) {
      assert.ok(!name.includes('/tests/') && !name.includes('__pycache__') && !name.endsWith('.pyc'), `训练包不应包含 ${name}`);
      assert.notEqual(name, 'training/config.json', '训练包不应包含个人配置');
    }
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('模型下载清单：39 个网页模型全部有与运行时一致的下载地址', async () => {
  const cfg = JSON.parse(readFileSync(join(root, 'web', 'public', 'config.json'), 'utf8'));
  const { prebuiltAppConfig } = await import(join(root, 'node_modules', '@mlc-ai', 'web-llm', 'lib', 'index.js'));
  const byId = new Map(prebuiltAppConfig.model_list.map((r) => [r.model_id, r]));
  for (const extra of cfg.webllm.extraRecords || []) byId.set(extra.model_id, extra);

  const md = readFileSync(join(root, 'docs', 'model-downloads.md'), 'utf8');
  assert.equal(cfg.webllm.models.length, 39, '网页下拉应保持 39 个模型');
  for (const m of cfg.webllm.models) {
    const rec = byId.get(m.id);
    assert.ok(rec, `${m.id} 应有 WebLLM 记录`);
    assert.ok(md.includes(`\`${m.id}\``), `模型下载清单应包含 ${m.id}`);
    assert.ok(md.includes(rec.model.replace(/\/?$/, '/')), `模型下载清单应包含 ${m.id} 的权重仓库地址`);
    assert.ok(md.includes(rec.model_lib), `模型下载清单应包含 ${m.id} 的运行库地址`);
  }
});

test('Release 工作流：发布五件资产且文件名与网页/脚本一致', () => {
  const yml = readFileSync(join(root, '.github', 'workflows', 'release.yml'), 'utf8');
  for (const asset of [...SITE_PACKAGE_ASSETS, TRAINING_PACKAGE_ASSET]) {
    assert.ok(yml.includes(asset), `release.yml 应发布 ${asset}`);
  }
  assert.match(yml, /tags:.*v\*/, 'release.yml 应由 v* 标签触发');
  assert.match(yml, /contents: write/, 'release.yml 需要写权限创建 Release');
});
