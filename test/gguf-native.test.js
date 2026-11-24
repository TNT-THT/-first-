// GGUF 元数据解析与“加载入口分片”选择：
// - 真实 GGUF 头里分词器词表就有几 MB，以前只读前 1MB → 超大模型永远“层数未知”；
// - 分片模型必须把第一个分片交给 llama.cpp，以前按字典序可能挑到中段分片。
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { parseGgufBlockCount, readGgufBlockCount } from '../server/native/gguf.js';
import { isShardFile, pickEntryFile } from '../server/native/catalog.js';

const gstr = (s) => {
  const b = Buffer.from(s, 'utf8');
  const h = Buffer.alloc(8);
  h.writeBigUInt64LE(BigInt(b.length));
  return Buffer.concat([h, b]);
};
const u32 = (v) => {
  const b = Buffer.alloc(4);
  b.writeUInt32LE(v);
  return b;
};

/** 拼一个 GGUF 头：词表数组放在 block_count 之前（真实文件的顺序），可指定大小。 */
function buildGgufHeader({ blockCount = 80, vocabCount = 1000, tensorCount = 500 } = {}) {
  const parts = [];
  const kv = (key, type, payload) => parts.push(gstr(key), u32(type), payload);
  const tokens = [];
  for (let i = 0; i < vocabCount; i += 1) tokens.push(gstr(`token-${i}-${'x'.repeat(8)}`));
  const arrHead = Buffer.alloc(12);
  arrHead.writeUInt32LE(8, 0); // STRING
  arrHead.writeBigUInt64LE(BigInt(tokens.length), 4);
  kv('tokenizer.ggml.tokens', 9, Buffer.concat([arrHead, ...tokens]));
  kv('general.architecture', 8, gstr('llama'));
  kv('llama.block_count', 4, u32(blockCount));
  kv('llama.context_length', 4, u32(32768));
  const head = Buffer.alloc(24);
  head.write('GGUF', 0, 'ascii');
  head.writeUInt32LE(3, 4);
  head.writeBigUInt64LE(BigInt(tensorCount), 8);
  head.writeBigUInt64LE(4n, 16);
  return Buffer.concat([head, ...parts]);
}

test('GGUF 层数：小文件直接解析', () => {
  assert.equal(parseGgufBlockCount(buildGgufHeader({ blockCount: 64 })), 64);
});

test('GGUF 层数：几 MB 的词表（超大模型常见）也要能越过，不再“层数未知”', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-gguf-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'big.gguf');
  const buf = buildGgufHeader({ blockCount: 80, vocabCount: 152064 }); // 约 4.3 MB 头
  assert.ok(buf.length > 4 * 1024 * 1024);
  await writeFile(file, buf);
  assert.equal(await readGgufBlockCount(file), 80);
  // 旧实现只读 1MB：同样文件读不到 → 界面上算不出每层显存
  assert.equal(await readGgufBlockCount(file, { maxBytes: 1024 * 1024 }), null);
});

test('GGUF 层数：块边界处的跨块读取不会重复/错位', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-gguf-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'edge.gguf');
  await writeFile(file, buildGgufHeader({ blockCount: 33, vocabCount: 3000 }));
  // 小块（10 字节）强制大量跨块拼接，最容易暴露“重复读同一段”的错位
  assert.equal(await readGgufBlockCount(file, { chunkBytes: 10 }), 33);
});

test('GGUF 层数：坏文件/非 GGUF 返回 null 不抛错', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-gguf-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const file = path.join(dir, 'bad.gguf');
  await writeFile(file, 'NOT-A-GGUF-FILE-AT-ALL');
  assert.equal(await readGgufBlockCount(file), null);
  assert.equal(parseGgufBlockCount(Buffer.alloc(0)), null);
  assert.equal(parseGgufBlockCount(Buffer.from('GGUF')), null);
});

test('pickEntryFile：分片模型必须返回第 1 个分片（而不是字典序第一个）', () => {
  assert.equal(isShardFile('M-Q4_K_M-00001-of-00002.gguf'), true);
  assert.equal(isShardFile('M-Q4_K_M.gguf'), false);
  // 10 个分片：字典序会把 -00010- 排到 -00002- 前面
  const ten = Array.from({ length: 10 }, (_, i) => ({ path: `M-Q4_K_M-${String(i + 1).padStart(5, '0')}-of-00010.gguf` }));
  assert.equal(pickEntryFile([...ten].reverse()).path, 'M-Q4_K_M-00001-of-00010.gguf');
  // 按量化分子目录布局时，同一目录里还有别的文件
  assert.equal(
    pickEntryFile([
      { path: 'README.md' },
      { path: 'Q4_K_M/M-00002-of-00003.gguf' },
      { path: 'Q4_K_M/M-00001-of-00003.gguf' },
      { path: 'mmproj-F16.gguf' },
    ]).path,
    'Q4_K_M/M-00001-of-00003.gguf',
  );
  // 单文件量化直接用它
  assert.equal(pickEntryFile([{ path: 'M-Q4_K_M.gguf' }, { path: 'a.txt' }]).path, 'M-Q4_K_M.gguf');
  assert.equal(pickEntryFile([{ path: 'mmproj-F16.gguf' }]), null);
  assert.equal(pickEntryFile([]), null);
  assert.equal(pickEntryFile(null), null);
});

// ---- llama.cpp 运行库：官方发布里同时有 cudart-llama-…（只有 CUDA 运行库）----
import { pickCudartAsset, pickReleaseAsset } from '../server/native/llama.js';

const asset = (name) => ({ name, browser_download_url: `https://example.com/${name}` });

test('pickReleaseAsset：CUDA 变体不能被 cudart-llama-…（运行库包）截胡', () => {
  const assets = [
    asset('cudart-llama-b9-bin-win-cuda-13.5-x64.zip'),
    asset('llama-b9-bin-win-cpu-x64.zip'),
    asset('llama-b9-bin-win-cuda-13.5-x64.zip'),
    asset('llama-b9-bin-win-vulkan-x64.zip'),
    asset('llama-b9-bin-ubuntu-x64.tar.gz'),
    asset('llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz'),
    asset('llama-b9-bin-macos-arm64.tar.gz'),
  ];
  assert.equal(pickReleaseAsset(assets, { platform: 'win32', variant: 'cuda' }).name, 'llama-b9-bin-win-cuda-13.5-x64.zip');
  assert.equal(pickReleaseAsset(assets, { platform: 'win32' }).name, 'llama-b9-bin-win-cpu-x64.zip');
  assert.equal(pickReleaseAsset(assets, { platform: 'win32', variant: 'vulkan' }).name, 'llama-b9-bin-win-vulkan-x64.zip');
  // Linux 以前完全忽略变体，OJ_LLAMA_VARIANT=cuda 拿到的是 CPU 包
  assert.equal(pickReleaseAsset(assets, { platform: 'linux', variant: 'cuda' }).name, 'llama-b9-bin-ubuntu-cuda-13.5-x64.tar.gz');
  assert.equal(pickReleaseAsset(assets, { platform: 'linux' }).name, 'llama-b9-bin-ubuntu-x64.tar.gz');
  assert.equal(pickReleaseAsset(assets, { platform: 'darwin', arch: 'arm64' }).name, 'llama-b9-bin-macos-arm64.tar.gz');
  // 只有 cudart 包时不能把它当主包
  assert.equal(pickReleaseAsset([asset('cudart-llama-b9-bin-win-cuda-13.5-x64.zip')], { platform: 'win32', variant: 'cuda' }), null);
});

test('pickCudartAsset：CUDA 主包要能配上同版本的 cudart 运行库', () => {
  const assets = [
    asset('llama-b9-bin-win-cuda-13.5-x64.zip'),
    asset('cudart-llama-b9-bin-win-cuda-13.5-x64.zip'),
  ];
  assert.equal(
    pickCudartAsset(assets, asset('llama-b9-bin-win-cuda-13.5-x64.zip')).name,
    'cudart-llama-b9-bin-win-cuda-13.5-x64.zip',
  );
  assert.equal(pickCudartAsset(assets, asset('llama-b9-bin-win-cpu-x64.zip')), null);
});

// ---- llama-server 启动环境：CUDA 版的 cudart 依赖必须能被动态链接器找到 ----
test('LlamaProcess：非 Windows 启动时把 bin 目录加进 LD_LIBRARY_PATH/DYLD_LIBRARY_PATH', async () => {
  const { LlamaProcess } = await import("../server/native/llama.js");
  const calls = [];
  const fakeChild = () => {
    const handlers = {};
    return {
      stdout: { on() {} },
      stderr: { on() {} },
      once: (ev, fn) => { handlers[ev] = fn; },
      kill() {},
    };
  };
  const server = new LlamaProcess({
    spawnImpl: (bin, args, opts) => { calls.push({ bin, args, opts }); return fakeChild(); },
    fetchImpl: async () => { throw new Error('not listening'); },
    logger: { warn() {}, log() {}, error() {} },
  });
  // 不让 waitHealthy 拖住：只验证 spawn 参数，随后主动 stop
  const p = server.start({
    binary: '/data/native/bin/llama-server',
    modelFile: '/data/native/m/Q4_K_M/M-00001-of-00002.gguf',
    modelId: 'x', quant: 'Q4_K_M', contextTokens: 4096, gpuLayers: 0, port: 18999,
  }).catch(() => {});
  await new Promise((r) => setTimeout(r, 50));
  await server.stop();
  await p;
  assert.equal(calls.length, 1);
  const { args, opts } = calls[0];
  if (process.platform === 'win32') return;
  const varName = process.platform === 'darwin' ? 'DYLD_LIBRARY_PATH' : 'LD_LIBRARY_PATH';
  assert.ok(opts.env[varName].split(path.delimiter)[0] === '/data/native/bin', `首个搜索路径应是 bin 目录，实际 ${opts.env[varName]}`);
  // 关键参数仍在：分片入口 + flash attention
  assert.match(args[args.indexOf('-m') + 1], /-00001-of-00002\.gguf$/);
  assert.equal(args[args.indexOf('--flash-attn') + 1], 'on');
});

test('pickReleaseAsset：CUDA/Vulkan 也要按当前架构挑包（arm64 机器不能装 x64 包）', async () => {
  const { pickReleaseAsset } = await import('../server/native/llama.js');
  const assets = [
    { name: 'llama-b1-bin-win-cuda-12.4-x64.zip' },
    { name: 'llama-b1-bin-win-cuda-12.4-arm64.zip' },
    { name: 'llama-b1-bin-ubuntu-cuda-12.8-x64.tar.gz' },
    { name: 'llama-b1-bin-ubuntu-cuda-12.8-arm64.tar.gz' },
    { name: 'llama-b1-bin-ubuntu-vulkan-x64.tar.gz' },
  ];
  assert.equal(pickReleaseAsset(assets, { platform: 'win32', arch: 'arm64', variant: 'cuda' }).name, 'llama-b1-bin-win-cuda-12.4-arm64.zip');
  assert.equal(pickReleaseAsset(assets, { platform: 'linux', arch: 'arm64', variant: 'cuda' }).name, 'llama-b1-bin-ubuntu-cuda-12.8-arm64.tar.gz');
  assert.equal(pickReleaseAsset(assets, { platform: 'linux', arch: 'x64', variant: 'vulkan' }).name, 'llama-b1-bin-ubuntu-vulkan-x64.tar.gz');
  // 请求的变体不存在时回落到纯 CPU 包，而不是返回 null 让用户看到“没有预编译包”
  assert.equal(pickReleaseAsset([{ name: 'llama-b1-bin-ubuntu-x64.tar.gz' }], { platform: 'linux', arch: 'x64', variant: 'cuda' }).name, 'llama-b1-bin-ubuntu-x64.tar.gz');
});

test('LlamaProcess：extraArgs 追加到 llama-server 参数末尾（MoE 专家卸载等高级用法）', async () => {
  const { LlamaProcess } = await import('../server/native/llama.js');
  const calls = [];
  const server = new LlamaProcess({
    spawnImpl: (bin, args, opts) => {
      calls.push({ bin, args, opts });
      return { stdout: { on() {} }, stderr: { on() {} }, once() {}, kill() {} };
    },
    fetchImpl: async () => { throw new Error('not listening'); },
    logger: { warn() {}, log() {}, error() {} },
  });
  const p = server.start({
    binary: '/x/llama-server', modelFile: '/x/m.gguf', modelId: 'x', quant: 'Q4_K_M',
    contextTokens: 8192, gpuLayers: 99, port: 18998,
    extraArgs: ['--n-cpu-moe', '12', ''], // 空串必须被过滤掉，否则 llama-server 会把它当参数值
  }).catch(() => {});
  await new Promise((r) => setTimeout(r, 30));
  await server.stop();
  await p;
  const { args } = calls[0];
  assert.deepEqual(args.slice(-2), ['--n-cpu-moe', '12']);
  assert.ok(!args.includes(''));
  assert.equal(args[args.indexOf('-ngl') + 1], '99');
});
