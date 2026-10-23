// 生图纯逻辑单元测试：尺寸、随机数、PNDM 调度器、CLIP 分词器、float16、参数与模型目录。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_SIZE, RATIOS, normalizeSize, sizeForRatio, sizeProblem, snapSize } from '../web/src/image/sizes.js';
import { gaussianNoise, mulberry32 } from '../web/src/image/rng.js';
import { createPndmScheduler } from '../web/src/image/pndm.js';
import { bytesToUnicode, createClipTokenizer, parseMerges } from '../web/src/image/clip-tokenizer.js';
import { floatToHalfBits, halfBitsToFloat, encodeForType, decodeFromType } from '../web/src/image/tensor-types.js';
import {
  IMAGE_STORAGE_KEY,
  MAX_IMAGE_HINTS,
  addImageHint,
  buildImagePrompt,
  clearImageParams,
  effectiveImageSettings,
  getImageSettings,
  parseImageParams,
  removeImageHint,
  setImageParams,
  stripImageParamLine,
} from '../web/src/image/params.js';
import { fileUrl, formatGB, imageModelsOf, isSelectable, sourceUrls, totalBytes } from '../web/src/image/catalog.js';

const here = path.dirname(fileURLToPath(import.meta.url));

function memoryStorage(init = {}) {
  const data = { ...init };
  return {
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => {
      data[k] = String(v);
    },
    removeItem: (k) => {
      delete data[k];
    },
    _data: data,
  };
}

// ---------- 尺寸 ----------

test('尺寸：比例与长边换算（竖向与横向都正确）', () => {
  assert.deepEqual(sizeForRatio('1:1', 512), { width: 512, height: 512 });
  assert.deepEqual(sizeForRatio('4:3', 512), { width: 512, height: 384 });
  assert.deepEqual(sizeForRatio('3:4', 512), { width: 384, height: 512 });
  assert.deepEqual(sizeForRatio('16:9', 768), { width: 768, height: 432 });
  assert.deepEqual(sizeForRatio('9:16', 768), { width: 432, height: 768 });
  assert.deepEqual(sizeForRatio('3:4', 768), { width: 576, height: 768 });
});

test('尺寸：所有比例结果都在范围内且为 8 的倍数', () => {
  for (const r of RATIOS) {
    for (const long of [256, 300, 512, 640, 768]) {
      const { width, height } = sizeForRatio(r.id, long);
      assert.equal(sizeProblem(width, height), null, `${r.id} ${long} → ${width}×${height}`);
    }
  }
});

test('尺寸：未知比例回到默认尺寸', () => {
  assert.deepEqual(sizeForRatio('2:7', 512), { ...DEFAULT_SIZE });
});

test('尺寸：snapSize 夹取范围并取整到 8 的倍数', () => {
  assert.equal(snapSize(100), 256);
  assert.equal(snapSize(1000), 768);
  assert.equal(snapSize(500), 504);
  assert.equal(snapSize(512), 512);
  assert.equal(snapSize('abc'), null);
});

test('尺寸：normalizeSize 非数值时使用默认尺寸', () => {
  assert.deepEqual(normalizeSize(100, 1000), { width: 256, height: 768 });
  assert.deepEqual(normalizeSize(undefined, 512), { ...DEFAULT_SIZE });
});

test('尺寸：sizeProblem 给出中文说明', () => {
  assert.equal(sizeProblem(512, 512), null);
  assert.match(sizeProblem(500, 512), /8 的倍数/);
  assert.match(sizeProblem(1024, 512), /256–768/);
  assert.match(sizeProblem(512.5, 512), /整数/);
});

// ---------- 随机数 ----------

test('随机数：同一种子得到相同噪声，不同种子不同', () => {
  const a = gaussianNoise(42, 64);
  const b = gaussianNoise(42, 64);
  const c = gaussianNoise(43, 64);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.notDeepEqual(Array.from(a), Array.from(c));
});

test('随机数：mulberry32 的输出在 [0, 1) 内且确定', () => {
  const u = mulberry32(7);
  const first = [u(), u(), u()];
  const u2 = mulberry32(7);
  assert.deepEqual([u2(), u2(), u2()], first);
  for (const v of first) assert.ok(v >= 0 && v < 1);
});

test('随机数：标准正态的均值约为 0、方差约为 1', () => {
  const x = gaussianNoise(2026, 200000);
  let mean = 0;
  for (const v of x) mean += v;
  mean /= x.length;
  let variance = 0;
  for (const v of x) variance += (v - mean) ** 2;
  variance /= x.length;
  assert.ok(Math.abs(mean) < 0.01, `mean=${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.02, `variance=${variance}`);
});

// ---------- PNDM ----------

test('PNDM：n=4 的时间步与 diffusers（leading、steps_offset=1）一致', () => {
  const s = createPndmScheduler({ skip_prk_steps: true, steps_offset: 1, timestep_spacing: 'leading' });
  assert.deepEqual(s.setTimesteps(4), [751, 501, 501, 251, 1]);
});

test('PNDM：n=20 时间步共 21 个（步数 + 1），按降序，首尾正确', () => {
  const s = createPndmScheduler({ skip_prk_steps: true, steps_offset: 1 });
  const ts = s.setTimesteps(20);
  assert.deepEqual(ts, [
    951, 901, 901, 851, 801, 751, 701, 651, 601, 551, 501, 451, 401, 351, 301, 251, 201, 151, 101, 51, 1,
  ]);
  assert.equal(ts.length, 21);
});

test('PNDM：步数小于 2 或配置不受支持时报错', () => {
  const s = createPndmScheduler({ skip_prk_steps: true });
  assert.throws(() => s.setTimesteps(1), /不小于 2/);
  assert.throws(() => s.setTimesteps(2.5), /不小于 2/);
  assert.throws(() => createPndmScheduler({ skip_prk_steps: false }), /skip_prk_steps/);
  assert.throws(() => createPndmScheduler({ skip_prk_steps: true, prediction_type: 'v_prediction' }), /prediction_type/);
  assert.throws(() => createPndmScheduler({ skip_prk_steps: true, timestep_spacing: 'trailing' }), /timestep_spacing/);
});

test('PNDM：模型输出为 0 时，第一步只按 sqrt(ᾱ_prev / ᾱ_t) 缩放样本（scaled_linear 配置）', () => {
  const config = { skip_prk_steps: true, steps_offset: 1, beta_schedule: 'scaled_linear', beta_start: 0.00085, beta_end: 0.012 };
  const s = createPndmScheduler(config);
  const ts = s.setTimesteps(20);
  // 独立计算 ᾱ（与 diffusers 的 scaled_linear 定义相同）
  const N = 1000;
  const a = Math.sqrt(0.00085);
  const b = Math.sqrt(0.012);
  const ac = new Float64Array(N);
  let acc = 1;
  for (let i = 0; i < N; i += 1) {
    const beta = ((a + ((b - a) * i) / (N - 1)) ** 2);
    acc *= 1 - beta;
    ac[i] = acc;
  }
  const sample = Float64Array.from([0.5, -1.25, 2]);
  const out = s.step(new Float64Array(3), ts[0], sample);
  const t = ts[0];
  const prev = t - 50;
  const coeff = Math.sqrt(ac[prev] / ac[t]);
  for (let i = 0; i < 3; i += 1) assert.ok(Math.abs(out[i] - coeff * sample[i]) < 1e-9, `i=${i}`);
});

test('PNDM：必须先调用 setTimesteps', () => {
  const s = createPndmScheduler({ skip_prk_steps: true });
  assert.throws(() => s.step(new Float64Array(2), 1, new Float64Array(2)), /setTimesteps/);
});

// ---------- CLIP 分词器 ----------

function miniTokenizer(extraVocab = {}, merges = 'c a\nca t</w>\n') {
  const vocab = {
    '<|startoftext|>': 49406,
    '<|endoftext|>': 49407,
    'c': 1,
    'a': 2,
    't</w>': 3,
    'a</w>': 4,
    'ca': 5,
    'cat</w>': 6,
    ...extraVocab,
  };
  return createClipTokenizer({ vocab, merges: parseMerges(merges), maxLength: 77 });
}

test('分词器：BPE 合并后得到 cat</w>，并加上 BOS 与 EOS', () => {
  const tok = miniTokenizer();
  assert.deepEqual(tok.tokenize('Cat'), [49406, 6, 49407]);
});

test('分词器：encode 补齐到 77，补齐值为 EOS（pad）', () => {
  const tok = miniTokenizer();
  const ids = tok.encode('cat');
  assert.equal(ids.length, 77);
  assert.equal(ids[0], 49406);
  assert.equal(ids[1], 6);
  assert.equal(ids[2], 49407);
  assert.equal(ids[3], 49407);
  assert.equal(ids[76], 49407);
});

test('分词器：超过 77 个 token 时截断，并保留 EOS', () => {
  const tok = miniTokenizer();
  const long = Array.from({ length: 200 }, () => 'a').join(' ');
  const ids = tok.encode(long);
  assert.equal(ids.length, 77);
  assert.equal(ids[0], 49406);
  assert.equal(ids[76], 49407);
  assert.equal(ids[75], 4); // 'a</w>' 的编号；每个 'a' 都是独立的词
});

test('分词器：合并空白、忽略大小写', () => {
  const tok = miniTokenizer();
  assert.deepEqual(tok.tokenize('  CAT\n\t'), tok.tokenize('cat'));
});

test('分词器：字节映射与 GPT-2 一致（空格映射为 Ġ）', () => {
  const map = bytesToUnicode();
  assert.equal(map.size, 256);
  assert.equal(map.get(32), 'Ġ');
  assert.equal(map.get(33), '!');
});

test('分词器：未登录的字符使用 unk（默认等于 EOS），不会抛错', () => {
  const tok = miniTokenizer();
  const ids = tok.tokenize('z');
  assert.deepEqual(ids, [49406, 49407, 49407]);
});

// ---------- float16 ----------

test('float16：常见值的位模式与往返精度', () => {
  assert.equal(floatToHalfBits(1), 0x3c00);
  assert.equal(floatToHalfBits(-2), 0xc000);
  assert.equal(floatToHalfBits(0), 0);
  assert.equal(floatToHalfBits(65504), 0x7bff);
  assert.equal(floatToHalfBits(70000), 0x7c00);
  assert.equal(halfBitsToFloat(0x3c00), 1);
  assert.equal(halfBitsToFloat(0x0001), 2 ** -24);
  for (const v of [0.1, -3.25, 123.5, 0.000123]) {
    const back = halfBitsToFloat(floatToHalfBits(v));
    assert.ok(Math.abs(back - v) <= Math.abs(v) * 1e-3 + 1e-7, `${v} → ${back}`);
  }
});

test('张量类型：float16 编码与解码往返', () => {
  const src = Float32Array.from([0.5, -1, 0.25, 2]);
  const enc = encodeForType('float16', src);
  assert.ok(enc instanceof Uint16Array);
  assert.deepEqual(Array.from(decodeFromType('float16', enc)), Array.from(src));
  assert.throws(() => encodeForType('bfloat16', src), /不支持/);
});

// ---------- 参数与提示词 ----------

test('参数行：解析、夹取范围与取整', () => {
  assert.deepEqual(parseImageParams('画一只猫\n参数：width=768, height=512, steps=25, guidance=6, seed=42'), {
    width: 768,
    height: 512,
    steps: 25,
    guidance: 6,
    seed: 42,
  });
  assert.deepEqual(parseImageParams('设置: steps=0, width=1000, guidance=99'), { steps: 2, width: 768, guidance: 20 });
  assert.equal(parseImageParams('参数：foo=1, bar=2'), null);
  assert.equal(parseImageParams('没有参数行'), null);
});

test('参数行：stripImageParamLine 去掉参数行，保留描述', () => {
  assert.equal(stripImageParamLine('一只猫\n参数：steps=20'), '一只猫');
});

test('参数存储：按模型分别保存，清除只影响该模型', () => {
  const storage = memoryStorage();
  setImageParams('m1', { steps: 30, width: 512 }, { storage });
  setImageParams('m2', { steps: 10 }, { storage });
  assert.deepEqual(getImageSettings('m1', { storage }).params, { steps: 30, width: 512 });
  clearImageParams('m1', { storage });
  assert.deepEqual(getImageSettings('m1', { storage }).params, {});
  assert.deepEqual(getImageSettings('m2', { storage }).params, { steps: 10 });
  assert.ok(storage._data[IMAGE_STORAGE_KEY]);
});

test('参数存储：损坏的 JSON 视为空', () => {
  const storage = memoryStorage({ [IMAGE_STORAGE_KEY]: '{not json' });
  assert.deepEqual(getImageSettings('x', { storage }), { params: {}, hints: [] });
});

test('生效设置：管理员参数覆盖界面控件，尺寸取整', () => {
  const ui = { width: 512, height: 512, steps: 20, guidance: 7.5, seed: null };
  assert.deepEqual(effectiveImageSettings(ui, { steps: 30, width: 760 }), {
    width: 760,
    height: 512,
    steps: 30,
    guidance: 7.5,
    seed: null,
  });
  assert.equal(effectiveImageSettings(ui, {}).steps, 20);
});

test('风格提示：去重、删除、上限 30 条（丢弃最旧）', () => {
  const storage = memoryStorage();
  const first = addImageHint('m', ' 水彩风格 ', { storage });
  assert.equal(first.text, '水彩风格');
  assert.equal(addImageHint('m', '水彩风格', { storage }), null);
  assert.equal(addImageHint('m', '   ', { storage }), null);
  assert.equal(removeImageHint('m', first.id, { storage }), true);
  for (let i = 0; i < MAX_IMAGE_HINTS + 5; i += 1) addImageHint('m', `提示${i}`, { storage });
  const hints = getImageSettings('m', { storage }).hints;
  assert.equal(hints.length, MAX_IMAGE_HINTS);
  assert.equal(hints[0].text, '提示5');
});

test('提示词：用户描述在前，风格提示在后，逗号分隔', () => {
  assert.equal(buildImagePrompt('一只猫', [{ text: '水彩' }, { text: '柔和光线' }]), '一只猫, 水彩, 柔和光线');
  assert.equal(buildImagePrompt('  ', [{ text: '水彩' }]), '水彩');
});

// ---------- 模型目录 ----------

test('模型目录：config.json 中的生图模型条目完整、可选择性正确', () => {
  const cfg = JSON.parse(fs.readFileSync(path.join(here, '..', 'web', 'public', 'config.json'), 'utf8'));
  const models = imageModelsOf(cfg);
  const sd = models.find((m) => m.id === 'sd15-nmkd-fp16');
  const flux = models.find((m) => m.id === 'flux2-klein-4b-onnx-webgpu');
  assert.ok(sd && flux);
  assert.equal(isSelectable(sd), true);
  assert.equal(isSelectable(flux), false, 'FLUX 尚未接入，不可选择');
  for (const key of ['textEncoder', 'unet', 'unetWeights', 'vaeDecoder', 'vocab', 'merges', 'scheduler']) {
    assert.ok(sd.files[key], `缺少文件 ${key}`);
    assert.ok(sd.files[key].size > 0, `${key} 大小无效`);
  }
  assert.equal(totalBytes(sd), Object.values(sd.files).reduce((s, f) => s + f.size, 0));
  assert.match(formatGB(totalBytes(sd)), /GB$/);
  const urls = sourceUrls(sd);
  assert.equal(urls.unetWeights, fileUrl(sd, 'unet/weights.pb'));
  assert.match(urls.vocab, /^https:\/\/huggingface\.co\/nmkd\/stable-diffusion-1\.5-onnx-fp16\/resolve\/main\/tokenizer\/vocab\.json$/);
});

test('模型目录：没有配置时返回空列表', () => {
  assert.deepEqual(imageModelsOf({}), []);
  assert.deepEqual(imageModelsOf(undefined), []);
});
