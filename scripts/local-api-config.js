import { randomBytes } from 'node:crypto';

const DEFAULT_LLM_BASE_URL = 'http://127.0.0.1:11434/v1';
const DEFAULT_LLM_MODEL = 'qwen2.5:3b';
const CHANGE_ME_PASSWORD = 'change-me-to-a-long-password';

/**
 * Resolve a safe, single-computer API configuration.
 * Existing .env values are preserved where valid, while HOST is always loopback-only.
 */
export function resolveLocalApiConfig({ baseEnv = {}, localEnv = {}, processEnv = {}, makeSecret } = {}) {
  const overlay = { ...localEnv };
  const generatedCredentials = {};
  const useSecret = makeSecret || ((key) => (
    key === 'SESSION_SECRET'
      ? randomBytes(32).toString('hex')
      : randomBytes(24).toString('base64url')
  ));
  const nonEmpty = (value) => typeof value === 'string' && value.trim() !== '';
  const pick = (...values) => values.find(nonEmpty);
  const pickValid = (key, isValid) => [processEnv[key], baseEnv[key], localEnv[key]].find(isValid);

  const appPassword = pickValid(
    'APP_PASSWORD',
    (value) => nonEmpty(value) && value.length >= 8 && value !== CHANGE_ME_PASSWORD,
  ) || createCredential('APP_PASSWORD', (value) => value.length >= 8, useSecret, overlay, generatedCredentials);

  const apiAdminPassword = pickValid(
    'API_ADMIN_PASSWORD',
    (value) => nonEmpty(value) && value.length >= 16 && value !== appPassword,
  ) || createCredential(
    'API_ADMIN_PASSWORD',
    (value) => value.length >= 16 && value !== appPassword,
    useSecret,
    overlay,
    generatedCredentials,
  );

  const sessionSecret = pickValid(
    'SESSION_SECRET',
    (value) => nonEmpty(value) && value.length >= 32,
  ) || createCredential('SESSION_SECRET', (value) => value.length >= 32, useSecret, overlay, generatedCredentials);

  const portValue = pick(
    processEnv.LOCAL_API_PORT,
    processEnv.PORT,
    baseEnv.PORT,
    localEnv.PORT,
  ) || '3000';
  const port = Number(portValue);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('本机 API 端口必须是 1 到 65535 之间的整数。');
  }

  const configuredBaseUrl = [processEnv.LLM_BASE_URL, baseEnv.LLM_BASE_URL, localEnv.LLM_BASE_URL]
    .find((value) => nonEmpty(value) && !/host\.docker\.internal/i.test(value));
  const llmBaseUrl = configuredBaseUrl || DEFAULT_LLM_BASE_URL;
  if (!configuredBaseUrl) overlay.LLM_BASE_URL = DEFAULT_LLM_BASE_URL;

  const llmModel = pick(processEnv.LLM_MODEL, baseEnv.LLM_MODEL, localEnv.LLM_MODEL) || DEFAULT_LLM_MODEL;
  if (!pick(processEnv.LLM_MODEL, baseEnv.LLM_MODEL, localEnv.LLM_MODEL)) overlay.LLM_MODEL = DEFAULT_LLM_MODEL;

  // The local launcher must never expose the API to a LAN or public interface.
  overlay.HOST = '127.0.0.1';
  overlay.PORT = String(port);
  overlay.TRUST_PROXY = '0';
  overlay.PUBLIC_BASE_URL = `http://127.0.0.1:${port}`;
  overlay.CORS_ORIGINS = `http://127.0.0.1:${port}`;

  const env = {
    ...localEnv,
    ...baseEnv,
    ...processEnv,
    APP_PASSWORD: appPassword,
    API_ADMIN_PASSWORD: apiAdminPassword,
    SESSION_SECRET: sessionSecret,
    HOST: '127.0.0.1',
    PORT: String(port),
    TRUST_PROXY: '0',
    PUBLIC_BASE_URL: `http://127.0.0.1:${port}`,
    CORS_ORIGINS: `http://127.0.0.1:${port}`,
    LLM_BASE_URL: llmBaseUrl,
    LLM_MODEL: llmModel,
  };
  delete env.LOCAL_API_PORT;

  return { env, localEnv: overlay, generatedCredentials, port, llmBaseUrl, llmModel };
}

export function serializeLocalConfig(values) {
  return `${JSON.stringify(values, null, 2)}\n`;
}

function createCredential(key, isValid, makeSecret, overlay, generatedCredentials) {
  let value = '';
  for (let attempt = 0; attempt < 5 && !isValid(value); attempt += 1) {
    value = String(makeSecret(key));
  }
  if (!isValid(value)) throw new Error(`无法为 ${key} 生成有效的本机凭据。`);
  overlay[key] = value;
  generatedCredentials[key] = value;
  return value;
}

