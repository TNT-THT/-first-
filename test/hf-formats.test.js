// 多格式识别与运行路线 + HF 搜索/详情桥接（假 Hub API，不访问真实网络）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { detectFormats, detectVariants, hfModelId, optionsForFormats, parseHfRepoInput, routeForFormats, safeDirName } from '../server/native/formats.js';
import { fetchModelInfo, searchModels } from '../server/native/hf.js';

test('detectFormats：按文件与标签识别主流格式', () => {
  assert.deepEqual(detectFormats([{ path: 'model-Q4_K_M.gguf' }]), ['gguf']);
  assert.ok(detectFormats([{ path: 'model.safetensors' }, { path: 'config.json' }]).includes('safetensors'));
  assert.ok(detectFormats([{ path: 'pytorch_model.bin' }]).includes('pytorch'));
  assert.ok(detectFormats([{ path: 'onnx/model.onnx' }]).includes('onnx'));
  assert.ok(detectFormats([{ path: 'mlc-chat-config.json' }, { path: 'params_shard_0.bin' }]).includes('mlc'));
  assert.ok(detectFormats([], { tags: ['gguf'] }).includes('gguf'));
  assert.ok(detectFormats([{ path: 'model.safetensors' }], { tags: ['mlx'] }).includes('mlx'));
});

test('routeForFormats：默认路线本地优先，云端只是可切换选项', () => {
  assert.equal(routeForFormats(['gguf']).route, 'local');
  assert.equal(routeForFormats(['safetensors']).route, 'convert');
  assert.equal(routeForFormats(['pytorch']).route, 'convert');
  assert.equal(routeForFormats(['safetensors'], ['AWQ']).route, 'convert');
  assert.equal(routeForFormats(['onnx']).route, 'python');
  assert.equal(routeForFormats(['mlc']).route, 'browser');
  assert.equal(routeForFormats([]).route, 'unsupported');
  assert.deepEqual(detectVariants([{ path: 'model-awq.safetensors' }]), ['AWQ']);
});

test('optionsForFormats：任何格式都有本地路线，云端恒为可选项', () => {
  const types = (f, v) => optionsForFormats(f, v).map((o) => o.type);
  assert.deepEqual(types(['gguf']), ['local-gguf', 'cloud']);
  assert.deepEqual(types(['safetensors']), ['local-convert', 'local-python', 'cloud']);
  assert.deepEqual(types(['onnx']), ['local-python', 'cloud']);
  assert.ok(types(['mlc']).includes('browser'));
  assert.ok(optionsForFormats(['onnx']).every((o) => o.type === 'cloud' || o.local));
  assert.ok(optionsForFormats(['safetensors'], ['GPTQ']).some((o) => o.local), '量化变体也必须能本地运行');
});

test('parseHfRepoInput / hfModelId / safeDirName', () => {
  assert.equal(parseHfRepoInput('https://huggingface.co/bartowski/Qwen3-32B-GGUF'), 'bartowski/Qwen3-32B-GGUF');
  assert.equal(parseHfRepoInput('someone/Model-GGUF'), 'someone/Model-GGUF');
  assert.equal(parseHfRepoInput('就想搜个 llama'), null);
  assert.equal(hfModelId('a/b'), 'hf/a/b');
  assert.equal(safeDirName('hf/a/b'), 'hf_a_b');
});

test('searchModels：归一化 Hub 搜索结果', async () => {
  const fetchImpl = async (url) => {
    assert.match(String(url), /api\/models\?/);
    return new Response(JSON.stringify([
      { id: 'someone/Tiny-GGUF', author: 'someone', downloads: 123, likes: 4, pipeline_tag: 'text-generation', tags: ['gguf'], gated: false },
      { id: 'secret/hidden', private: true },
    ]), { headers: { 'content-type': 'application/json' } });
  };
  const out = await searchModels({ query: 'tiny', fetchImpl });
  assert.equal(out.length, 1);
  assert.equal(out[0].repo, 'someone/Tiny-GGUF');
  assert.equal(out[0].downloads, 123);
});

test('fetchModelInfo：文件树 + 总大小', async () => {
  const fetchImpl = async (url) => {
    const u = String(url);
    if (u.includes('/tree/')) {
      return new Response(JSON.stringify([
        { type: 'file', path: 'model.safetensors', size: 100, lfs: { size: 100 } },
        { type: 'file', path: 'config.json', size: 10 },
      ]), { headers: { 'content-type': 'application/json' } });
    }
    return new Response(JSON.stringify({ author: 'someone', downloads: 5, tags: [] }), { headers: { 'content-type': 'application/json' } });
  };
  const info = await fetchModelInfo('someone/Tiny-ST', { fetchImpl });
  assert.equal(info.totalBytes, 110);
  assert.equal(info.files.length, 2);
});
