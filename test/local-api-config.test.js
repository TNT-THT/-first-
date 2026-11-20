import test from 'node:test';
import assert from 'node:assert/strict';
import { resolveLocalApiConfig, serializeLocalConfig } from '../scripts/local-api-config.js';

function deterministicSecret(key) {
  return `generated-${key}-${'x'.repeat(40)}`;
}

const validBaseEnv = {
  APP_PASSWORD: 'existing-site-password',
  API_ADMIN_PASSWORD: 'existing-admin-password-with-enough-length',
  SESSION_SECRET: 's'.repeat(48),
};

test('local API defaults bind to loopback and generate distinct setup credentials', () => {
  const config = resolveLocalApiConfig({ makeSecret: deterministicSecret });

  assert.equal(config.env.HOST, '127.0.0.1');
  assert.equal(config.env.PORT, '3000');
  assert.equal(config.env.PUBLIC_BASE_URL, 'http://127.0.0.1:3000');
  assert.equal(config.env.CORS_ORIGINS, 'http://127.0.0.1:3000');
  assert.equal(config.env.TRUST_PROXY, '0');
  assert.equal(config.env.LOCAL_BROWSER_BRIDGE, '1');
  assert.equal(config.llmBaseUrl, '');
  assert.equal(config.llmModel, '');
  assert.equal(config.env.APP_PASSWORD, config.generatedCredentials.APP_PASSWORD);
  assert.equal(config.env.API_ADMIN_PASSWORD, config.generatedCredentials.API_ADMIN_PASSWORD);
  assert.notEqual(config.env.APP_PASSWORD, config.env.API_ADMIN_PASSWORD);
  assert.ok(config.env.SESSION_SECRET.length >= 32);
});

test('valid existing credentials and model settings are reused, but local binding stays private', () => {
  const config = resolveLocalApiConfig({
    baseEnv: {
      ...validBaseEnv,
      HOST: '0.0.0.0',
      PORT: '4312',
      LLM_BASE_URL: 'https://models.example.test/v1',
      LLM_MODEL: 'private-model',
      LLM_API_KEY: 'provider-secret',
    },
    makeSecret: deterministicSecret,
  });

  assert.equal(config.env.HOST, '127.0.0.1');
  assert.equal(config.port, 4312);
  assert.equal(config.env.PUBLIC_BASE_URL, 'http://127.0.0.1:4312');
  assert.equal(config.env.APP_PASSWORD, validBaseEnv.APP_PASSWORD);
  assert.equal(config.env.API_ADMIN_PASSWORD, validBaseEnv.API_ADMIN_PASSWORD);
  assert.equal(config.env.SESSION_SECRET, validBaseEnv.SESSION_SECRET);
  assert.equal(config.llmBaseUrl, 'https://models.example.test/v1');
  assert.equal(config.llmModel, 'private-model');
  assert.equal(config.env.LLM_API_KEY, 'provider-secret');
  assert.deepEqual(config.generatedCredentials, {});
});

test('Docker-only address and example placeholder credentials switch to safe local defaults', () => {
  const priorLocalEnv = {
    APP_PASSWORD: 'previous-local-password',
    API_ADMIN_PASSWORD: 'previous-local-admin-password-is-long-enough',
    SESSION_SECRET: 'l'.repeat(48),
  };
  const config = resolveLocalApiConfig({
    baseEnv: {
      APP_PASSWORD: 'change-me-to-a-long-password',
      API_ADMIN_PASSWORD: '',
      SESSION_SECRET: '',
      LLM_BASE_URL: 'http://host.docker.internal:11434/v1',
    },
    localEnv: priorLocalEnv,
    makeSecret: deterministicSecret,
  });

  assert.equal(config.env.APP_PASSWORD, priorLocalEnv.APP_PASSWORD);
  assert.equal(config.env.API_ADMIN_PASSWORD, priorLocalEnv.API_ADMIN_PASSWORD);
  assert.equal(config.env.SESSION_SECRET, priorLocalEnv.SESSION_SECRET);
  assert.equal(config.llmBaseUrl, '');
  assert.deepEqual(config.generatedCredentials, {});
});

test('local port override wins and invalid ports are rejected', () => {
  const config = resolveLocalApiConfig({
    baseEnv: { PORT: '3000' },
    processEnv: { LOCAL_API_PORT: '4567' },
    makeSecret: deterministicSecret,
  });
  assert.equal(config.port, 4567);
  assert.equal(config.env.PORT, '4567');

  assert.throws(
    () => resolveLocalApiConfig({ processEnv: { LOCAL_API_PORT: '70000' }, makeSecret: deterministicSecret }),
    /1 到 65535/,
  );
});

test('local config serialization safely round-trips quoted values', () => {
  const values = {
    APP_PASSWORD: 'a local password with "quotes" and \\slashes',
    HOST: '127.0.0.1',
    SYSTEM_PROMPT: 'line one\nline two',
  };
  assert.deepEqual(JSON.parse(serializeLocalConfig(values)), values);
});
