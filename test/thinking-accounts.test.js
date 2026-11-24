// 本批新功能：思考开关真的关掉、账号存储隔离、GGUF 层数解析。
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createApp } from '../server/index.js';
import { loadConfig } from '../server/config.js';
import { startMockLLM } from './mock-llm.js';
import { AccountStore } from '../server/accounts.js';
import { parseGgufBlockCount } from '../server/native/gguf.js';

let llm;
let server;
let base;
let dir;

before(async () => {
  llm = await startMockLLM();
  dir = await mkdtemp(path.join(tmpdir(), 'oj-thinking-'));
  const cfg = loadConfig({
    SESSION_SECRET: 'thinking-test-secret-0123456789',
    LLM_BASE_URL: llm.url,
    LLM_API_KEY: 'sk-test',
    LLM_MODEL: 'qwen-test',
    API_KEYS: 'sk-external-1',
    OJ_ACCOUNTS_FILE: path.join(dir, 'accounts.json'),
    API_KEYS_FILE: path.join(dir, 'keys.json'),
    API_SETTINGS_FILE: path.join(dir, 'settings.json'),
  });
  const app = createApp(cfg);
  await new Promise((resolve) => { server = app.listen(0, '127.0.0.1', resolve); });
  base = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await new Promise((r) => server.close(r));
  await llm.close();
  await rm(dir, { recursive: true, force: true });
});

async function chat(body, headers = {}) {
  const r = await fetch(`${base}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
  });
  assert.equal(r.status, 200);
  await r.text();
  return llm.requests[llm.requests.length - 1].body;
}

test('思考开关：thinking:false 时上游收到 enable_thinking=false（真的不思考）', async () => {
  const upstream = await chat({ messages: [{ role: 'user', content: 'hi' }], thinking: false });
  assert.equal(upstream.chat_template_kwargs?.enable_thinking, false);
});

test('思考开关：不传或 true 时不注入关闭参数', async () => {
  const a = await chat({ messages: [{ role: 'user', content: 'hi' }] });
  assert.equal(a.chat_template_kwargs?.enable_thinking, undefined);
  const b = await chat({ messages: [{ role: 'user', content: 'hi' }], thinking: true });
  assert.equal(b.chat_template_kwargs?.enable_thinking, undefined);
});

test('思考开关：/v1 的 chat_template_kwargs.enable_thinking=false 被尊重', async () => {
  const r = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer sk-external-1' },
    body: JSON.stringify({
      model: 'qwen-test',
      messages: [{ role: 'user', content: 'hi' }],
      chat_template_kwargs: { enable_thinking: false },
    }),
  });
  assert.equal(r.status, 200);
  await r.text();
  const upstream = llm.requests[llm.requests.length - 1].body;
  assert.equal(upstream.chat_template_kwargs?.enable_thinking, false);
});

test('账号存储：注册、验密、重复名拒绝、数据隔离与文件权限', async () => {
  const file = path.join(dir, 'unit-accounts.json');
  const store = new AccountStore(file);
  const a = await store.register('alice', 'alice-pass-123');
  assert.ok(a.id);
  await assert.rejects(() => store.register('alice', 'alice-pass-123'), /已经被注册/);
  await assert.rejects(() => store.register('al', 'short'), /密码/);
  assert.equal(await store.verify('alice', 'wrong-password'), null);
  const ok = await store.verify('alice', 'alice-pass-123');
  assert.equal(ok.id, a.id);

  await store.putData(a.id, { v: 1, sessions: { sessions: [1] }, settings: {} });
  const b = await store.register('bob', 'bob-pass-12345');
  assert.equal((await store.getData(b.id)).payload, null, '别的账号读不到 alice 的数据');
  assert.deepEqual((await store.getData(a.id)).payload.settings, {});
  assert.equal((await stat(file)).mode & 0o777, 0o600);
});

test('GGUF 层数解析：读 block_count，坏文件返回 null', () => {
  // 最小 GGUF 头：magic + version + tensor_count + kv_count + 一条 kv(block_count)
  const key = Buffer.from('llama.block_count', 'utf8');
  const header = Buffer.alloc(4 + 4 + 8 + 8);
  header.write('GGUF', 0, 'ascii');
  header.writeUInt32LE(3, 4);
  header.writeBigUInt64LE(0n, 8); // tensor_count
  header.writeBigUInt64LE(1n, 16); // kv_count
  const kv = Buffer.alloc(8 + key.length + 4 + 4);
  kv.writeBigUInt64LE(BigInt(key.length), 0);
  key.copy(kv, 8);
  kv.writeUInt32LE(4, 8 + key.length); // UINT32 类型
  kv.writeUInt32LE(32, 12 + key.length); // 32 层
  const buf = Buffer.concat([header, kv]);
  assert.equal(parseGgufBlockCount(buf), 32);
  assert.equal(parseGgufBlockCount(Buffer.from('NOPE........')), null);
  assert.equal(parseGgufBlockCount(Buffer.alloc(4)), null);
});
