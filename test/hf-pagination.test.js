// Hugging Face 文件树分页与大文件下载健壮性：
// Hub 树接口一次最多 1000 条并在 Link 头给下一页游标。超大模型仓库轻松过千条，
// 只读第一页会静默截断（量化少档、分片只下一部分，加载时才报缺文件）。
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { downloadFile, fetchRepoFiles, parseLinkNext } from '../server/native/hf.js';
import { fetchRepoTreeViaBrowser, parseLinkNext as parseLinkNextBrowser } from '../web/src/engines/native-client.js';

const page = (prefix, from, to, next) => new Response(
  JSON.stringify(Array.from({ length: to - from }, (_, i) => ({
    type: 'file', path: `${prefix}-${from + i}-of-9999.gguf`, size: 10, lfs: { size: 10, oid: 'a'.repeat(64) },
  }))),
  { headers: { 'content-type': 'application/json', ...(next ? { link: `<https://huggingface.co/api/models/o/r/tree/main?recursive=true&cursor=${next}>; rel="next"` } : {}) } },
);

test('parseLinkNext：只认 rel="next"，没有就返回空', () => {
  assert.equal(
    parseLinkNext('<https://huggingface.co/api/models/o/r/tree/main?cursor=abc>; rel="next", <https://x/?cursor=z>; rel="prev"'),
    'https://huggingface.co/api/models/o/r/tree/main?cursor=abc',
  );
  assert.equal(parseLinkNext('<https://x/?cursor=z>; rel="prev"'), '');
  assert.equal(parseLinkNext(''), '');
  assert.equal(parseLinkNext(undefined), '');
  assert.equal(parseLinkNextBrowser('<https://y/?cursor=q>; rel="next"'), 'https://y/?cursor=q');
});

test('服务端 fetchRepoFiles：跟随 rel="next" 取回全部多页文件并去重', async () => {
  const seen = [];
  const fetchImpl = async (url) => {
    seen.push(String(url));
    if (!String(url).includes('cursor=')) return page('A', 0, 3, 'p2');
    if (String(url).includes('cursor=p2')) return page('B', 3, 6, 'p3');
    if (String(url).includes('cursor=p3')) return page('C', 6, 8, '');
    throw new Error(`意外请求：${url}`);
  };
  const files = await fetchRepoFiles('o/r', { fetchImpl });
  assert.equal(files.length, 8);
  assert.equal(seen.length, 3);
  assert.deepEqual(files.map((f) => f.path), [
    'A-0-of-9999.gguf', 'A-1-of-9999.gguf', 'A-2-of-9999.gguf',
    'B-3-of-9999.gguf', 'B-4-of-9999.gguf', 'B-5-of-9999.gguf',
    'C-6-of-9999.gguf', 'C-7-of-9999.gguf',
  ]);
  assert.ok(files.every((f) => f.size === 10 && f.sha256.length === 64));
});

test('服务端 fetchRepoFiles：游标原地打转时不死循环', async () => {
  let calls = 0;
  const fetchImpl = async (url) => {
    calls += 1;
    if (!String(url).includes('cursor=')) return page('A', 0, 2, 'same');
    return page('B', 2, 4, 'same');
  };
  const files = await fetchRepoFiles('o/r', { fetchImpl });
  assert.equal(files.length, 4);
  assert.equal(calls, 2);
});

test('浏览器 fetchRepoTreeViaBrowser：同样跟随分页（大仓库不再被截断）', async () => {
  const seen = [];
  const fetchImpl = async (url) => {
    seen.push(String(url));
    if (!String(url).includes('cursor=')) return page('A', 0, 2, 'n1');
    return page('B', 2, 4, '');
  };
  const files = await fetchRepoTreeViaBrowser('o/r', { fetchImpl });
  assert.equal(files.length, 4);
  assert.equal(seen.length, 2);
  assert.equal(files.at(-1).path, 'B-3-of-9999.gguf');
});

test('浏览器 fetchRepoTreeViaBrowser：门控仓库 401 给中文说明（提示填 Token）', async () => {
  await assert.rejects(
    () => fetchRepoTreeViaBrowser('o/r', { fetchImpl: async () => new Response('denied', { status: 401 }) }),
    /需要授权.*Token/,
  );
});

test('downloadFile：传输中断自动重试并按偏移续传，不从头开始', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-dl-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const dest = path.join(dir, 'model.gguf');
  const content = Buffer.from('0123456789abcdef');
  let calls = 0;
  const fetchImpl = async (url, opts) => {
    calls += 1;
    const range = opts.headers?.range || '';
    const from = range ? Number(range.match(/bytes=(\d+)-/)?.[1] || 0) : 0;
    if (calls === 1) {
      // 第一次只给一半就断流
      return new Response(content.subarray(0, 8), {
        status: 200,
        headers: { 'content-length': '8' },
      });
    }
    return new Response(content.subarray(from), {
      status: from > 0 ? 206 : 200,
      headers: { 'content-length': String(content.length - from) },
    });
  };
  const progress = [];
  const out = await downloadFile({
    url: 'https://example.com/x.gguf', dest, fetchImpl, expectedSize: content.length,
    retries: 3, retryDelayMs: 1, onProgress: (p) => progress.push(p.downloaded),
  });
  assert.equal(out.bytes, content.length);
  assert.ok(calls >= 2);
  assert.equal(await readFile(dest, 'utf8'), content.toString());
  assert.ok(progress.length > 0);
  await assert.rejects(() => stat(`${dest}.part`), /ENOENT/);
});

test('downloadFile：CDN 没有 content-length 时以 Hub 声明大小校验，截断文件不转正', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-dl-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const dest = path.join(dir, 'short.gguf');
  // 一直只给 5 字节、永远到不了 10：必须报“不完整”，绝不能把半截文件当成功
  const fetchImpl = async () => new Response('abcde', { status: 200 });
  await assert.rejects(
    () => downloadFile({ url: 'https://example.com/y.gguf', dest, fetchImpl, expectedSize: 10, retries: 1, retryDelayMs: 1 }),
    /不完整/,
  );
  await assert.rejects(() => stat(dest), /ENOENT/);
});

test('downloadFile：已完整的分片直接跳过，不再浪费流量', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-dl-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const dest = path.join(dir, 'done.gguf');
  await writeFile(dest, 'hello');
  let calls = 0;
  const out = await downloadFile({
    url: 'https://example.com/z.gguf', dest, expectedSize: 5,
    fetchImpl: async () => { calls += 1; throw new Error('不应请求'); },
  });
  assert.equal(out.skipped, true);
  assert.equal(calls, 0);
});

test('downloadFile：大小不符的历史坏文件会被丢掉重下', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-dl-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const dest = path.join(dir, 'bad.gguf');
  await writeFile(dest, 'trunc'); // 5 字节，但声明 10
  const out = await downloadFile({
    url: 'https://example.com/w.gguf', dest, expectedSize: 10, retryDelayMs: 1,
    fetchImpl: async () => new Response('0123456789', { status: 200, headers: { 'content-length': '10' } }),
  });
  assert.equal(out.bytes, 10);
  assert.equal(await readFile(dest, 'utf8'), '0123456789');
});

test('downloadFile：403/404 立即失败并给中文说明，不无谓重试', async (t) => {
  const dir = await mkdtemp(path.join(tmpdir(), 'oj-dl-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  let calls = 0;
  const fetchImpl = async () => {
    calls += 1;
    return new Response('no', { status: 403 });
  };
  await assert.rejects(
    () => downloadFile({ url: 'https://example.com/g.gguf', dest: path.join(dir, 'g.gguf'), fetchImpl, retries: 3, retryDelayMs: 1 }),
    /HTTP 403.*Token/,
  );
  assert.equal(calls, 1);
});
