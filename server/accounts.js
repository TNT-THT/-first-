/**
 * 网页账号（服务器端部署专用）：开放自助注册，账号之间互不可见。
 *
 * - 密码只存 scrypt 摘要（每账号独立盐），永不明文落盘、永不回传。
 * - 每个账号一份数据：对话记录与参数设定（前端整体同步的一个 JSON 对象），
 *   换设备登录同一账号即可恢复；模型与权重是服务器共用的，不进账号数据。
 * - 存储为单个 JSON 文件（与 api-keys.json 同级的数据目录），写入走临时文件+重命名。
 */
import { randomBytes, scryptSync, timingSafeEqual } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';

export const MAX_ACCOUNT_DATA_BYTES = 8 * 1024 * 1024; // 单账号同步数据上限（对话+设定）
const NAME_RE = /^[A-Za-z0-9_\u4e00-\u9fa5][A-Za-z0-9_\u4e00-\u9fa5.-]{1,31}$/;

function hashPassword(password, salt = randomBytes(16).toString('hex')) {
  const hash = scryptSync(String(password), salt, 32).toString('hex');
  return { salt, hash };
}

function checkPassword(user, password) {
  if (!user || typeof password !== 'string') return false;
  const { hash } = hashPassword(password, user.salt);
  const a = Buffer.from(hash, 'hex');
  const b = Buffer.from(user.hash, 'hex');
  return a.length === b.length && timingSafeEqual(a, b);
}

export function validateAccountName(name) {
  const n = String(name || '').trim();
  if (!NAME_RE.test(n)) return { ok: false, error: '账号名需为 2~32 个字符（中文、字母、数字、下划线、点、横线）。' };
  return { ok: true, name: n };
}

export function validateAccountPassword(password) {
  const p = String(password || '');
  if (p.length < 6 || p.length > 128) return { ok: false, error: '密码长度需为 6~128 位。' };
  return { ok: true };
}

export class AccountStore {
  constructor(file) {
    this.file = file;
    this.data = null; // { users: [{id, name, salt, hash, createdAt}], data: { [userId]: { payload, updatedAt } } }
    this.queue = Promise.resolve();
  }

  async _load() {
    if (this.data) return this.data;
    let parsed = null;
    try {
      parsed = JSON.parse(await readFile(this.file, 'utf8'));
    } catch {
      parsed = null;
    }
    this.data = {
      users: Array.isArray(parsed?.users) ? parsed.users : [],
      data: parsed?.data && typeof parsed.data === 'object' ? parsed.data : {},
    };
    return this.data;
  }

  async _save() {
    await mkdir(path.dirname(this.file), { recursive: true });
    const tmp = `${this.file}.tmp-${process.pid}`;
    await writeFile(tmp, `${JSON.stringify(this.data)}\n`, { mode: 0o600 });
    await rename(tmp, this.file);
  }

  /** 串行化写操作，避免并发注册/同步互相覆盖。 */
  _mutate(fn) {
    const run = this.queue.then(async () => {
      await this._load();
      const out = fn(this.data);
      await this._save();
      return out;
    });
    this.queue = run.catch(() => {});
    return run;
  }

  _find(data, name) {
    const lower = String(name || '').trim().toLowerCase();
    return data.users.find((u) => u.name.toLowerCase() === lower) || null;
  }

  async register(name, password) {
    const nameCheck = validateAccountName(name);
    if (!nameCheck.ok) throw Object.assign(new Error(nameCheck.error), { status: 400 });
    const passCheck = validateAccountPassword(password);
    if (!passCheck.ok) throw Object.assign(new Error(passCheck.error), { status: 400 });
    return this._mutate((data) => {
      if (this._find(data, nameCheck.name)) {
        throw Object.assign(new Error('这个账号名已经被注册了，换一个试试。'), { status: 409 });
      }
      const { salt, hash } = hashPassword(password);
      const user = {
        id: `u_${randomBytes(9).toString('base64url')}`,
        name: nameCheck.name,
        salt,
        hash,
        createdAt: new Date().toISOString(),
      };
      data.users.push(user);
      return { id: user.id, name: user.name };
    });
  }

  async verify(name, password) {
    const data = await this._load();
    const user = this._find(data, name);
    if (!checkPassword(user, password)) return null;
    return { id: user.id, name: user.name };
  }

  async getData(userId) {
    const data = await this._load();
    const rec = data.data[userId];
    return rec ? { payload: rec.payload ?? null, updatedAt: rec.updatedAt || 0 } : { payload: null, updatedAt: 0 };
  }

  async putData(userId, payload) {
    const text = JSON.stringify(payload ?? null);
    if (text.length > MAX_ACCOUNT_DATA_BYTES) {
      throw Object.assign(new Error('账号数据过大，已拒绝保存（对话太多时可删除旧会话）。'), { status: 413 });
    }
    return this._mutate((data) => {
      const updatedAt = Date.now();
      data.data[userId] = { payload: payload ?? null, updatedAt };
      // 只保留仍存在的账号的数据，防文件无限增长
      const alive = new Set(data.users.map((u) => u.id));
      for (const key of Object.keys(data.data)) if (!alive.has(key)) delete data.data[key];
      return { updatedAt };
    });
  }

  get fileExists() {
    return existsSync(this.file);
  }
}
