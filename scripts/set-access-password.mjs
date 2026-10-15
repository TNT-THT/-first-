#!/usr/bin/env node
/**
 * 设置网站访问口令。口令明文不会写入仓库，只保存加盐的 SHA-256 哈希。
 *
 *   node scripts/set-access-password.mjs              # 自动生成一个随机口令并打印
 *   node scripts/set-access-password.mjs "我的口令"    # 使用指定口令
 *
 * 修改后重新执行 npm run build（或推送到 main/分支触发部署）即可生效。
 */
import { createHash, randomBytes, randomInt } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const configPath = path.join(root, 'web', 'public', 'config.json');

function randomPassword(len = 14) {
  // 去掉容易混淆的字符（0/O、1/l/I）
  const alphabet = 'abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789';
  let out = '';
  for (let i = 0; i < len; i++) out += alphabet[randomInt(alphabet.length)];
  return out;
}

const given = process.argv[2];
const password = given || randomPassword();
if (password.length < 8) {
  console.error('口令至少 8 位');
  process.exit(1);
}

const salt = randomBytes(16).toString('hex');
const hash = createHash('sha256').update(`${salt}:${password}`, 'utf8').digest('hex');

const config = JSON.parse(readFileSync(configPath, 'utf8'));
config.access = { salt, passwordSha256: hash };
writeFileSync(configPath, JSON.stringify(config, null, 2) + '\n', 'utf8');

console.log(`已更新 ${path.relative(root, configPath)}`);
console.log(`访问口令：${password}`);
console.log('（口令明文未写入仓库，请妥善保存）');
