// 生图读取模型文件：完整缓存直接读；下载到一半只补下缺少的字节（不重下已保存的部分）；什么都没有才从网络读取。
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { sourceUrls } from '../web/src/image/catalog.js';
import { IMAGE_CACHE, IMAGE_PART_CACHE, downloadImageModel } from '../web/src/image/model-files.js';
import { openModelFile } from '../web/src/image/model-source.js';

function memoryCaches() {
  const store = new Map();
  return {
    _store: store,
    async open(name) {
      if (!store.has(name)) store.set(name, new Map());
      const m = store.get(name);
      return {
        async keys() {
          return [...m.keys()].map((url) => ({ url }));
        },
        async match(url) {
          const key = typeof url === 'string' ? url : url.url;
          if (!m.has(key)) return undefined;
          const e = m.get(key);
          return new Response(e.bytes.slice(), { status: 200, headers: e.headers });
        },
        async put(url, res) {
          if (!res.ok) throw new TypeError('bad response');
          m.set(url, { bytes: new Uint8Array(await res.arrayBuffer()), headers: Object.fromEntries(res.headers) });
        },
        async delete(url) {
          return m.delete(url);
        },
      };
    },
  };
}

const DATA = Uint8Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]); // 10 字节
const model = {
  id: 'source-test',
  label: '读取测试',
  family: 'sd15',
  repo: 'org/repo',
  revision: 'main',
  status: 'available',
  files: { weights: { path: 'weights.pb', size: 10 } },
};
const URL_W = sourceUrls(model).weights;
const instantSleep = async () => {};

function rangeFetch(calls, { failAfter = null } = {}) {
  return async (url, opts = {}) => {
    const range = opts.headers?.Range || null;
    calls.push(range);
    if (range) {
      const m = /bytes=(\d+)-(\d+)/.exec(range);
      return new Response(DATA.slice(Number(m[1]), Number(m[2]) + 1), { status: 206 });
    }
    if (failAfter !== null) {
      // 先送 failAfter 字节，然后断开（模拟网络中断）
      let sent = false;
      const stream = new ReadableStream(
        {
          pull(c) {
            if (!sent) {
              sent = true;
              c.enqueue(DATA.slice(0, failAfter));
              return;
            }
            c.error(new TypeError('network error'));
          },
        },
        { highWaterMark: 0 },
      );
      return new Response(stream, { status: 200 });
    }
    return new Response(DATA.slice(), { status: 200 });
  };
}

async function bytesOf(res) {
  return new Uint8Array(await res.arrayBuffer());
}

test('完整缓存：直接读取，不访问网络', async () => {
  const storage = memoryCaches();
  const calls = [];
  await downloadImageModel(model, { storage, fetchImpl: rangeFetch(calls), sleep: instantSleep });
  calls.length = 0;
  const res = await openModelFile(URL_W, 10, { storage, fetchImpl: rangeFetch(calls) });
  assert.deepEqual(await bytesOf(res), DATA);
  assert.deepEqual(calls, [], '完整缓存不发起请求');
});

test('下载到一半：只补下缺少的字节（Range），已保存的 4 字节不重下，结果正确', async () => {
  const storage = memoryCaches();
  // 第一次下载：收到 4 字节后中断；sleep 直接抛出取消，使下载停在这里（已保存 4 字节）
  await assert.rejects(
    downloadImageModel(model, {
      storage,
      fetchImpl: rangeFetch([], { failAfter: 4 }),
      sleep: async () => {
        throw Object.assign(new Error('已取消下载'), { name: 'AbortError' });
      },
    }),
    (err) => err.name === 'AbortError',
  );
  const parts = await storage.open(IMAGE_PART_CACHE);
  assert.ok((await parts.keys()).length > 0, '已保存部分分段');

  const calls = [];
  const res = await openModelFile(URL_W, 10, { storage, fetchImpl: rangeFetch(calls) });
  assert.deepEqual(await bytesOf(res), DATA);
  assert.deepEqual(calls, ['bytes=4-9'], '只请求缺少的字节，不整个重读');
  const cache = await storage.open(IMAGE_CACHE);
  assert.ok(await cache.match(URL_W), '补齐后写入完整缓存');
});

test('什么都没有：从网络读取（与原来一致），不写入完整缓存', async () => {
  const storage = memoryCaches();
  const calls = [];
  const res = await openModelFile(URL_W, 10, { storage, fetchImpl: rangeFetch(calls) });
  assert.deepEqual(await bytesOf(res), DATA);
  assert.deepEqual(calls, [null]);
  const cache = await storage.open(IMAGE_CACHE);
  assert.equal(await cache.match(URL_W), undefined, '不在生成时写入完整缓存');
});

test('没有存储接口（如无 Cache API）：直接从网络读取', async () => {
  const calls = [];
  const res = await openModelFile(URL_W, 10, { storage: null, fetchImpl: rangeFetch(calls) });
  assert.deepEqual(await bytesOf(res), DATA);
  assert.deepEqual(calls, [null]);
});
