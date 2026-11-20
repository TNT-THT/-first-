import { randomBytes } from 'node:crypto';
import { chmod, mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { validateConfig } from './config.js';
import { HttpError, validateGenerationParams } from './chat.js';

const SETTING_KEYS = new Set([
  'llmBaseUrl',
  'llmModel',
  'llmApiKey',
  'llmTemperature',
  'systemPrompt',
  'apiMaxTokens',
  'defaultGenerationParams',
]);

/**
 * 持久化管理员可编辑的服务端模型设置。
 * API Key 只写入文件（0600），管理 API 只返回是否已配置，不返回密钥本身。
 */
export class ApiSettingsStore {
  constructor(filePath) {
    this.filePath = filePath;
    this.baseValues = null;
    this.overrides = {};
    this.loaded = false;
    this.mutationQueue = Promise.resolve();
  }

  apply(cfg) {
    if (this.loaded) return;
    this.baseValues = pickRuntimeValues(cfg);
    this.overrides = {};
    try {
      const raw = readFileSync(this.filePath, 'utf8');
      const data = JSON.parse(raw);
      if (data?.version !== 1 || !isPlainObject(data.values)) {
        throw new Error('服务端参数存储文件格式无效；为避免覆盖错误配置，未自动重置。');
      }
      this.overrides = validateSettings(data.values);
      Object.assign(cfg, this.overrides);
    } catch (err) {
      if (err.code !== 'ENOENT') throw err;
    }
    if (!cfg.defaultGenerationParams) cfg.defaultGenerationParams = {};
    this.loaded = true;
  }

  snapshot(cfg) {
    return {
      llmBaseUrl: cfg.llmBaseUrl || '',
      llmModel: cfg.llmModel || '',
      llmTemperature: cfg.llmTemperature,
      systemPrompt: cfg.systemPrompt || '',
      apiMaxTokens: cfg.apiMaxTokens,
      defaultGenerationParams: cfg.defaultGenerationParams || {},
      llmApiKeyConfigured: !!cfg.llmApiKey,
    };
  }

  async update(patch, cfg) {
    this.apply(cfg);
    return this.#enqueue(async () => {
      const changes = validateSettings(patch);
      const nextOverrides = { ...this.overrides, ...changes };
      const candidate = { ...cfg, ...changes };
      const errors = validateConfig(candidate);
      if (errors.length) throw new HttpError(400, errors.join('；'));
      await this.#persist(nextOverrides);
      this.overrides = nextOverrides;
      Object.assign(cfg, changes);
      return this.snapshot(cfg);
    });
  }

  async reset(cfg) {
    this.apply(cfg);
    return this.#enqueue(async () => {
      await this.#persist({});
      this.overrides = {};
      Object.assign(cfg, this.baseValues);
      cfg.defaultGenerationParams ||= {};
      return this.snapshot(cfg);
    });
  }

  #enqueue(operation) {
    const current = this.mutationQueue.then(operation);
    this.mutationQueue = current.catch(() => {});
    return current;
  }

  async #persist(values) {
    const dir = path.dirname(this.filePath);
    await mkdir(dir, { recursive: true, mode: 0o700 });
    await chmod(dir, 0o700).catch(() => {});
    const temp = `${this.filePath}.${process.pid}.${randomBytes(6).toString('hex')}.tmp`;
    try {
      await writeFile(temp, `${JSON.stringify({ version: 1, values }, null, 2)}\n`, {
        encoding: 'utf8',
        flag: 'wx',
        mode: 0o600,
      });
      await rename(temp, this.filePath);
      await chmod(this.filePath, 0o600).catch(() => {});
    } finally {
      await rm(temp, { force: true }).catch(() => {});
    }
  }
}

function validateSettings(input) {
  if (!isPlainObject(input)) throw new HttpError(400, '设置必须是 JSON 对象');
  const result = {};
  for (const key of Object.keys(input)) {
    if (!SETTING_KEYS.has(key)) throw new HttpError(400, `不支持修改此设置：${key}`);
  }

  if (Object.hasOwn(input, 'llmBaseUrl')) {
    if (typeof input.llmBaseUrl !== 'string' || input.llmBaseUrl.length > 2048) {
      throw new HttpError(400, 'LLM_BASE_URL 必须是有效的字符串');
    }
    const value = input.llmBaseUrl.trim().replace(/\/+$/, '');
    if (value) {
      let url;
      try {
        url = new URL(value);
      } catch {
        throw new HttpError(400, 'LLM_BASE_URL 必须是有效的 HTTP 或 HTTPS 地址');
      }
      if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
        throw new HttpError(400, 'LLM_BASE_URL 仅支持不带账号、查询参数或片段的 HTTP/HTTPS 地址');
      }
      result.llmBaseUrl = url.toString().replace(/\/+$/, '');
    } else {
      result.llmBaseUrl = '';
    }
  }

  if (Object.hasOwn(input, 'llmModel')) {
    if (typeof input.llmModel !== 'string' || input.llmModel.length > 256 || /[\u0000-\u001f\u007f]/.test(input.llmModel)) {
      throw new HttpError(400, 'LLM_MODEL 必须是不超过 256 字符的普通字符串');
    }
    result.llmModel = input.llmModel.trim();
  }

  if (Object.hasOwn(input, 'llmApiKey')) {
    if (typeof input.llmApiKey !== 'string' || input.llmApiKey.length > 4096) {
      throw new HttpError(400, 'LLM_API_KEY 必须是不超过 4096 字符的字符串');
    }
    result.llmApiKey = input.llmApiKey.trim();
  }

  if (Object.hasOwn(input, 'llmTemperature')) {
    const value = input.llmTemperature;
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 2) {
      throw new HttpError(400, 'LLM_TEMPERATURE 必须是 0 到 2 之间的数字');
    }
    result.llmTemperature = value;
  }

  if (Object.hasOwn(input, 'systemPrompt')) {
    if (typeof input.systemPrompt !== 'string' || input.systemPrompt.length > 50000) {
      throw new HttpError(400, 'SYSTEM_PROMPT 必须是不超过 50000 字符的字符串');
    }
    result.systemPrompt = input.systemPrompt;
  }

  if (Object.hasOwn(input, 'apiMaxTokens')) {
    const value = input.apiMaxTokens;
    if (!Number.isInteger(value) || value < 1 || value > 1_000_000) {
      throw new HttpError(400, 'API_MAX_TOKENS 必须是 1 到 1000000 之间的整数');
    }
    result.apiMaxTokens = value;
  }

  if (Object.hasOwn(input, 'defaultGenerationParams')) {
    result.defaultGenerationParams = validateGenerationParams(input.defaultGenerationParams, { maxBytes: 16 * 1024 });
  }
  return result;
}

function pickRuntimeValues(cfg) {
  return {
    llmBaseUrl: cfg.llmBaseUrl || '',
    llmModel: cfg.llmModel || '',
    llmApiKey: cfg.llmApiKey || '',
    llmTemperature: cfg.llmTemperature,
    systemPrompt: cfg.systemPrompt || '',
    apiMaxTokens: cfg.apiMaxTokens,
    defaultGenerationParams: cfg.defaultGenerationParams || {},
  };
}

function isPlainObject(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}
