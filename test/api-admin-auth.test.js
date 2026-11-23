// API 管理员验证持久化：验证一次后（含关面板、刷新）不用再验证，
// 只有主动锁定或令牌过期才需要重新验证。
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  API_ADMIN_SESSION_KEY,
  clearApiAdminSession,
  loadApiAdminSession,
  saveApiAdminSession,
} from '../web/src/api-admin-auth.js';

function fakeStorage() {
  const map = new Map();
  return {
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, String(v)),
    removeItem: (k) => map.delete(k),
    map,
  };
}

test('保存后可恢复（模拟刷新页面/重开面板）', () => {
  const storage = fakeStorage();
  saveApiAdminSession(storage, { token: 'v1.123.sig', expiresAt: 9999 });
  assert.deepEqual(loadApiAdminSession(storage, 1000), { token: 'v1.123.sig', expiresAt: 9999 });
});

test('已过期返回 null 并顺手清除', () => {
  const storage = fakeStorage();
  saveApiAdminSession(storage, { token: 'v1.123.sig', expiresAt: 500 });
  assert.equal(loadApiAdminSession(500), null);
  assert.equal(loadApiAdminSession(storage, 400)?.token, 'v1.123.sig'); // 再次保存前已被清
  saveApiAdminSession(storage, { token: 'v1.123.sig', expiresAt: 500 });
  assert.equal(loadApiAdminSession(storage, 500), null);
  assert.equal(storage.map.has(API_ADMIN_SESSION_KEY), false, '过期会话应被清除');
});

test('锁定（clear）后必须重新验证', () => {
  const storage = fakeStorage();
  saveApiAdminSession(storage, { token: 'v1.123.sig', expiresAt: 9999 });
  clearApiAdminSession(storage);
  assert.equal(loadApiAdminSession(storage, 1000), null);
});

test('存储不可用/数据损坏时不抛错', () => {
  const broken = {
    getItem() { throw new Error('denied'); },
    setItem() { throw new Error('denied'); },
    removeItem() { throw new Error('denied'); },
  };
  saveApiAdminSession(broken, { token: 'x', expiresAt: 1 });
  assert.equal(loadApiAdminSession(broken), null);
  clearApiAdminSession(broken);

  const storage = fakeStorage();
  storage.setItem(API_ADMIN_SESSION_KEY, '{not json');
  assert.equal(loadApiAdminSession(storage), null);
  saveApiAdminSession(storage, { token: '', expiresAt: 10 });
  assert.equal(loadApiAdminSession(storage), null, '空令牌不保存');
});
