// 管理员的更正、知识库、说明与风格提示迁移到生图 AI（仅本机）。
// 覆盖：来自聊天 AI 的说明与更正迁入生图 AI、生图 AI 之间的复制、去重与上限、更正需要密钥验证的开关。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  IMAGE_STORAGE_KEY,
  MAX_IMAGE_NOTES,
  addImageHint,
  copyIntoImageModel,
  getImageSettings,
  imageAdminModelIds,
} from '../web/src/image/params.js';
import { listCorrections, listKnowledge, listAdaptation, recordFeedback } from '../web/src/adaptation.js';

function memoryStorage() {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    raw: () => m.get(IMAGE_STORAGE_KEY) ?? null,
  };
}

test('迁移更正需要管理员密钥验证（allowCorrections）', () => {
  const storage = memoryStorage();
  assert.throws(
    () => copyIntoImageModel('sd15-nmkd-fp16', { corrections: [{ text: '猫是橘色的' }] }, { storage }),
    /需要先通过管理员密钥验证/,
  );
  assert.equal(storage.raw(), null, '未验证时不写入任何内容');
});

test('复制而非移动：来源不变；目标得到条目并记录来源；重复文字跳过', () => {
  const storage = memoryStorage();
  addImageHint('src-image', '水彩风格', { storage });
  const r1 = copyIntoImageModel(
    'tiny-sd-onnx-webgpu',
    {
      corrections: [{ text: '猫是橘色的', from: 'src-image' }],
      knowledge: [{ text: '白天的天空是蓝色', from: 'src-image' }],
      hints: [{ text: '水彩风格', from: 'src-image' }],
    },
    { storage, allowCorrections: true },
  );
  assert.deepEqual(r1, { corrections: 1, knowledge: 1, hints: 1, duplicate: 0, overflow: 0 });
  const target = getImageSettings('tiny-sd-onnx-webgpu', { storage });
  assert.equal(target.corrections[0].text, '猫是橘色的');
  assert.equal(target.corrections[0].from, 'src-image');
  assert.equal(getImageSettings('src-image', { storage }).hints.length, 1, '来源保持不变');

  const r2 = copyIntoImageModel(
    'tiny-sd-onnx-webgpu',
    { knowledge: [{ text: '白天的天空是蓝色' }] },
    { storage },
  );
  assert.equal(r2.knowledge, 0);
  assert.equal(r2.duplicate, 1);
  assert.equal(getImageSettings('tiny-sd-onnx-webgpu', { storage }).knowledge.length, 1);
});

test('没有新增内容时不改动存储（只有重复项）', () => {
  const storage = memoryStorage();
  copyIntoImageModel('a', { knowledge: [{ text: 'x' }] }, { storage });
  const before = storage.raw();
  const r = copyIntoImageModel('a', { knowledge: [{ text: 'x' }] }, { storage });
  assert.equal(r.duplicate, 1);
  assert.equal(storage.raw(), before);
});

test('上限：超过每类上限的条目不写入，并计入 overflow；已有条目不被丢弃', () => {
  const storage = memoryStorage();
  const items = Array.from({ length: MAX_IMAGE_NOTES + 3 }, (_, i) => ({ text: `知识${i}` }));
  const r = copyIntoImageModel('big', { knowledge: items }, { storage });
  assert.equal(r.knowledge, MAX_IMAGE_NOTES);
  assert.equal(r.overflow, 3);
  const kb = getImageSettings('big', { storage }).knowledge;
  assert.equal(kb.length, MAX_IMAGE_NOTES);
  assert.equal(kb[0].text, '知识0', '最早的条目保留，不被挤出');
});

test('聊天 AI 的更正、知识库与说明可以迁入生图 AI（说明作为知识库条目）', () => {
  const storage = memoryStorage();
  const adaptStorage = memoryStorage();
  recordFeedback('Qwen3-4B-q4f16_1-MLC', '更正：太阳是方形的', {
    storage: adaptStorage,
    allowCorrections: true,
    question: '太阳什么样？',
  });
  recordFeedback('Qwen3-4B-q4f16_1-MLC', '这个说法要注意：月亮不发光', { storage: adaptStorage, question: '月亮？' });
  const corr = listCorrections('Qwen3-4B-q4f16_1-MLC', { storage: adaptStorage });
  const kb = listKnowledge('Qwen3-4B-q4f16_1-MLC', { storage: adaptStorage });
  const notes = listAdaptation('Qwen3-4B-q4f16_1-MLC', { storage: adaptStorage }).notes;
  assert.equal(corr.length, 1);
  assert.equal(notes.length, 1);
  assert.ok(kb.length >= 1, '说明会同步写入知识库');

  const r = copyIntoImageModel(
    'sd15-nmkd-fp16',
    {
      corrections: corr.map((c) => ({ text: c.text, from: 'Qwen3-4B-q4f16_1-MLC' })),
      knowledge: [...kb, ...notes].map((k) => ({ text: k.text, from: 'Qwen3-4B-q4f16_1-MLC' })),
    },
    { storage, allowCorrections: true },
  );
  assert.equal(r.corrections, 1);
  // 说明与知识库中相同的文字只写入一次
  assert.equal(r.knowledge, new Set([...kb, ...notes].map((k) => k.text)).size);
  assert.equal(r.duplicate, [...kb, ...notes].length - r.knowledge);
  assert.ok(imageAdminModelIds({ storage }).includes('sd15-nmkd-fp16'));
});

test('列出生图 AI 时只包含有更正、知识库或风格提示的条目', () => {
  const storage = memoryStorage();
  // 只有参数的生图 AI 不算（迁移只针对管理员记录）
  storage.setItem(IMAGE_STORAGE_KEY, JSON.stringify({ models: { p: { params: { steps: 20 }, hints: [], corrections: [], knowledge: [] } } }));
  assert.deepEqual(imageAdminModelIds({ storage }), []);
  addImageHint('h', '风格', { storage });
  assert.deepEqual(imageAdminModelIds({ storage }), ['h']);
});
