// 本机大模型目录与量化解析：目录完整性、三种真实仓库布局的量化识别、归档解压、
// llama.cpp 发布资源选择、思考过程折叠。全部离线，不访问网络。
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import {
  NATIVE_MODELS,
  estimateRamGB,
  extractQuant,
  groupFilesByQuant,
  isNativeModelId,
  nativeModelById,
  pickDefaultQuant,
} from '../server/native/catalog.js';
import { extractTarGz, extractZip } from '../server/native/archive.js';
import { createThinkFolder, foldMessageReasoning, pickLlamaRelease, pickReleaseAsset } from '../server/native/llama.js';
import { buildZip } from '../scripts/lib/zip.mjs';

test('目录：id 唯一、包含 27B 与 100~1000B 梯队、仓库地址合法', () => {
  const ids = NATIVE_MODELS.map((m) => m.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const m of NATIVE_MODELS) {
    assert.match(m.repo, /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/, m.id);
    assert.ok(m.paramsB >= 27, `${m.id} 不应小于 27B`);
    assert.ok(m.license, `${m.id} 应标注许可证`);
  }
  assert.ok(nativeModelById('Qwen3.8-27B'), '必须包含用户点名的 Qwen3.8-27B');
  const big = NATIVE_MODELS.filter((m) => m.paramsB >= 100);
  assert.ok(big.length >= 5, `100B 以上至少 5 个，实际 ${big.length}`);
  assert.ok(Math.max(...NATIVE_MODELS.map((m) => m.paramsB)) >= 600, '应包含 600B+ 旗舰');
  assert.ok(isNativeModelId('DeepSeek-R1-671B'));
  assert.ok(!isNativeModelId('Qwen3-8B-q4f16_1-MLC'), 'WebLLM 模型不能混进本机目录');
});

test('extractQuant：兼容根目录文件、点分隔、按量化分子目录三种布局', () => {
  assert.equal(extractQuant('Qwen3.8-27B-Q4_K_M.gguf'), 'Q4_K_M');
  assert.equal(extractQuant('Mixtral-8x22B-Instruct-v0.1.Q4_K_M-00001-of-00002.gguf'), 'Q4_K_M');
  assert.equal(extractQuant('Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf'), 'Q4_K_M');
  assert.equal(extractQuant('DeepSeek-R1-Q2_K/DeepSeek-R1-00001-of-00002.gguf'), 'Q2_K');
  assert.equal(extractQuant('Meta-Llama-3.1-405B-Instruct-Q4_K_M/Meta-Llama-3.1-405B-Instruct-Q4_K_M-00001-of-00005.gguf'), 'Q4_K_M');
  assert.equal(extractQuant('Llama-4-Scout-17B-16E-Instruct-UD-IQ1_M.gguf'), 'UD-IQ1_M');
  assert.equal(extractQuant('DeepSeek-R1-UD-Q2_K_XL/DeepSeek-R1-UD-Q2_K_XL-00001-of-00004.gguf'), 'UD-Q2_K_XL');
  assert.equal(extractQuant('README.md'), null);
});

test('groupFilesByQuant：分片排序、排除 mmproj、Q2_K 与 Q2_K_L 分开', () => {
  const grouped = groupFilesByQuant([
    { path: 'M-Q4_K_M-00002-of-00002.gguf', size: 12 },
    { path: 'M-Q4_K_M-00001-of-00002.gguf', size: 10 },
    { path: 'M-Q2_K.gguf', size: 5 },
    { path: 'M-Q2_K_L.gguf', size: 6 },
    { path: 'mmproj-F16.gguf', size: 99 },
    { path: 'README.md', size: 1 },
  ]);
  const byQuant = new Map(grouped.map((g) => [g.quant, g]));
  assert.deepEqual([...byQuant.keys()].sort(), ['Q2_K', 'Q2_K_L', 'Q4_K_M']);
  assert.deepEqual(byQuant.get('Q4_K_M').files.map((f) => f.path), [
    'M-Q4_K_M-00001-of-00002.gguf',
    'M-Q4_K_M-00002-of-00002.gguf',
  ]);
  assert.equal(byQuant.get('Q4_K_M').totalBytes, 22);
  assert.equal(pickDefaultQuant(grouped), 'Q4_K_M');
});

test('estimateRamGB：随权重与上下文增长', () => {
  const small = estimateRamGB(16e9, 8192);
  const big = estimateRamGB(130e9, 8192);
  assert.ok(big > small && small > 16);
  assert.ok(estimateRamGB(16e9, 65536) > small);
});

test('zip 解压：与本站 buildZip 往返一致，拒绝路径穿越', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-native-zip-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const zip = buildZip([
    { name: 'bin/llama-server', data: Buffer.from('ELF-FAKE') },
    { name: 'bin/lib.ggml.so', data: Buffer.from('LIB') },
  ]);
  const written = await extractZip(zip, dir);
  assert.deepEqual(written.sort(), ['bin/lib.ggml.so', 'bin/llama-server']);
  assert.equal(await readFile(path.join(dir, 'bin', 'llama-server'), 'utf8'), 'ELF-FAKE');
  const evil = buildZip([{ name: '../evil.txt', data: Buffer.from('x') }]);
  await assert.rejects(() => extractZip(evil, dir), /不安全路径/);
});

/** 手工拼一个最小 ustar 条目（只用于测试 tar.gz 解压）。 */
function tarEntry(name, data) {
  const header = Buffer.alloc(512);
  header.write(name, 0, 100);
  header.write('0000644\0', 100, 8);
  header.write('0000000\0', 108, 8);
  header.write('0000000\0', 116, 8);
  header.write(`${data.length.toString(8).padStart(11, '0')}\0`, 124, 12);
  header.write('00000000000\0', 136, 12);
  header.write('        ', 148, 8);
  header.write('0', 156, 1);
  header.write('ustar\0', 257, 6);
  let sum = 0;
  for (const b of header) sum += b;
  header.write(`${sum.toString(8).padStart(6, '0')}\0 `, 148, 8);
  const body = Buffer.alloc(Math.ceil(data.length / 512) * 512);
  data.copy(body);
  return Buffer.concat([header, body]);
}

test('tar.gz 解压：llama.cpp 的 ubuntu 包布局', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-native-tar-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const tar = Buffer.concat([
    tarEntry('llama-b1234/llama-server', Buffer.from('ELF-FAKE')),
    tarEntry('llama-b1234/libllama.so', Buffer.from('LIB')),
    Buffer.alloc(1024),
  ]);
  const written = await extractTarGz(gzipSync(tar), dir);
  assert.deepEqual(written.sort(), ['llama-b1234/libllama.so', 'llama-b1234/llama-server']);
  assert.equal(await readFile(path.join(dir, 'llama-b1234', 'llama-server'), 'utf8'), 'ELF-FAKE');
});

test('pickLlamaRelease：跳过非 b 系列发布，挑最新带资产的 b 系列构建', () => {
  const releases = [
    { tag_name: 'v1.0.0', assets: [{ name: 'nightly-tag.txt' }] },
    { tag_name: 'b7001', assets: [] },
    { tag_name: 'b7000', assets: [{ name: 'llama-b7000-bin-win-cpu-x64.zip' }] },
    { tag_name: 'b6999', assets: [{ name: 'llama-b6999-bin-win-cpu-x64.zip' }] },
  ];
  assert.equal(pickLlamaRelease(releases)?.tag_name, 'b7000');
  assert.equal(pickLlamaRelease([{ tag_name: 'v1.0.0', assets: [{ name: 'x' }] }]), null);
  assert.equal(pickLlamaRelease(null), null);
});

test('pickReleaseAsset：按平台挑 llama.cpp 官方构建', () => {
  const assets = [
    { name: 'llama-b7000-bin-win-cpu-x64.zip' },
    { name: 'llama-b7000-bin-win-cuda-12.4-x64.zip' },
    { name: 'llama-b7000-bin-macos-arm64.tar.gz' },
    { name: 'llama-b7000-bin-ubuntu-x64.tar.gz' },
  ].map((a) => ({ ...a, browser_download_url: `https://example.com/${a.name}` }));
  assert.equal(pickReleaseAsset(assets, { platform: 'win32' }).name, 'llama-b7000-bin-win-cpu-x64.zip');
  assert.equal(pickReleaseAsset(assets, { platform: 'win32', variant: 'cuda' }).name, 'llama-b7000-bin-win-cuda-12.4-x64.zip');
  assert.equal(pickReleaseAsset(assets, { platform: 'darwin', arch: 'arm64' }).name, 'llama-b7000-bin-macos-arm64.tar.gz');
  assert.equal(pickReleaseAsset(assets, { platform: 'linux', arch: 'x64' }).name, 'llama-b7000-bin-ubuntu-x64.tar.gz');
  assert.equal(pickReleaseAsset([], { platform: 'linux' }), null);
});

test('createThinkFolder / foldMessageReasoning：思考过程折叠为 <think>', () => {
  const folder = createThinkFolder();
  let out = '';
  out += folder.push({ reasoning_content: '先想', content: '' });
  out += folder.push({ reasoning_content: '再想', content: '' });
  out += folder.push({ reasoning_content: '', content: '答案' });
  out += folder.push({ content: '继续' });
  assert.equal(out, '<think>先想再想</think>答案继续');

  const plain = createThinkFolder();
  assert.equal(plain.push({ content: '直接答' }), '直接答');

  const folded = foldMessageReasoning({ role: 'assistant', reasoning_content: '想', content: '答' });
  assert.equal(folded.content, '<think>想</think>答');
  assert.equal('reasoning_content' in folded, false);
  assert.equal(foldMessageReasoning({ role: 'assistant', content: '答' }).content, '答');
});
