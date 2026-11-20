import { createHash, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile, chmod } from 'node:fs/promises';
import path from 'node:path';

const MAX_KEYS = 100;

/**
 * 持久化的站点 API Key 管理器。
 * 明文 Key 只在创建时返回一次；磁盘仅保存 SHA-256 摘要和展示用的末四位。
 * 管理接口之外，调用方永远无法读取现有 Key。
 */
export class ApiKeyStore {
  constructor(filePath) {
    this.filePath = filePath;
    this.keys = [];
    this.loaded = false;
    this.loading = null;
    this.mutationQueue = Promise.resolve();
  }

  async #load() {
    if (this.loaded) return;
    if (!this.loading) {
      this.loading = (async () => {
        try {
          await chmod(this.filePath, 0o600).catch(() => {});
          const raw = await readFile(this.filePath, 'utf8');
          const data = JSON.parse(raw);
          if (data?.version !== 1 || !Array.isArray(data.keys)) {
            throw new Error('API Key 存储文件格式无效；为避免丢失密钥，未自动覆盖。');
          }
          this.keys = data.keys.map(validateRecord);
          this.loaded = true;
        } catch (err) {
          if (err.code === 'ENOENT') {
            this.keys = [];
            this.loaded = true;
            return;
          }
          throw err;
        }
      })().finally(() => {
        this.loading = null;
      });
    }
    await this.loading;
  }

  async #mutate(mutator) {
    const operation = this.mutationQueue.then(async () => {
      await this.#load();
      const next = [...this.keys];
      const result = mutator(next);
      if (result.changed) {
        await this.#persist(next);
        this.keys = next;
      }
      return result.value;
    });
    // 一次落盘失败不应让后续所有修改永久失败。
    this.mutationQueue = operation.catch(() => {});
    return operation;
  }

  async #persist(keys) {
    const dir = path.dirname(this.filePath);
    await mkdir(dir, { recursive: true, mode: 0o700 });
    await chmod(dir, 0o700).catch(() => {});
    const temp = `${this.filePath}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`;
    const body = `${JSON.stringify({ version: 1, keys }, null, 2)}\n`;
    try {
      await writeFile(temp, body, { encoding: 'utf8', flag: 'wx', mode: 0o600 });
      await rename(temp, this.filePath);
      await chmod(this.filePath, 0o600).catch(() => {});
    } finally {
      await rm(temp, { force: true }).catch(() => {});
    }
  }

  async verify(key) {
    if (typeof key !== 'string' || !key) return false;
    await this.#load();
    const candidate = digest(key);
    let match = false;
    for (const entry of this.keys) {
      if (safeEqual(candidate, entry.hash)) match = true;
    }
    return match;
  }

  async list() {
    await this.#load();
    return this.keys
      .map(({ id, label, hint, createdAt }) => ({ id, label, hint, createdAt }))
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }

  async create(label = '') {
    const cleanLabel = String(label ?? '').trim().replace(/[\u0000-\u001f\u007f]/g, '').slice(0, 80) || '未命名密钥';
    const key = `ojai_${randomBytes(32).toString('base64url')}`;
    const entry = {
      id: randomUUID(),
      label: cleanLabel,
      hint: key.slice(-4),
      hash: digest(key),
      createdAt: new Date().toISOString(),
    };
    await this.#mutate((keys) => {
      if (keys.length >= MAX_KEYS) throw new Error(`最多只能创建 ${MAX_KEYS} 个 API Key`);
      keys.push(entry);
      return { changed: true, value: publicRecord(entry) };
    });
    // 明文仅返回这一次；调用方不应存入持久化或日志。
    return { ...publicRecord(entry), key };
  }

  async revoke(id) {
    return this.#mutate((keys) => {
      const index = keys.findIndex((entry) => entry.id === id);
      if (index < 0) return { changed: false, value: false };
      keys.splice(index, 1);
      return { changed: true, value: true };
    });
  }
}

function validateRecord(entry) {
  if (
    !entry ||
    typeof entry.id !== 'string' ||
    typeof entry.label !== 'string' ||
    typeof entry.hint !== 'string' ||
    !/^[a-f0-9]{64}$/.test(entry.hash) ||
    typeof entry.createdAt !== 'string'
  ) {
    throw new Error('API Key 存储文件包含无效记录；为避免丢失密钥，未自动覆盖。');
  }
  return {
    id: entry.id,
    label: entry.label,
    hint: entry.hint,
    hash: entry.hash,
    createdAt: entry.createdAt,
  };
}

function publicRecord({ id, label, hint, createdAt }) {
  return { id, label, hint, createdAt };
}

function digest(value) {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function safeEqual(a, b) {
  const left = Buffer.from(String(a), 'hex');
  const right = Buffer.from(String(b), 'hex');
  return left.length === right.length && timingSafeEqual(left, right);
}
