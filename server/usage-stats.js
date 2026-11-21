/**
 * 服务端 API 用量统计：内存聚合 + 防抖落盘。
 *
 * 聚合口径与网页端共用 web/src/usage-stats.js 的纯函数核心（只存汇总与定长
 * 近期日志，空间有硬上限）。记录是同步的内存哈希更新 O(1)；落盘防抖合并，
 * 写失败只告警不影响请求处理。文件损坏时从空统计开始，不拖垮服务启动。
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  clearUsageStats,
  emptyUsageStats,
  normalizeUsageStats,
  recordUsage,
  summarizeUsage,
} from '../web/src/usage-stats.js';

export class UsageStatsStore {
  /** @param {string} [filePath] 留空表示仅内存统计（测试/临时实例）。 */
  constructor(filePath = '') {
    this.filePath = filePath || '';
    this.stats = emptyUsageStats();
    this.saveTimer = null;
    if (this.filePath && existsSync(this.filePath)) {
      try {
        this.stats = normalizeUsageStats(JSON.parse(readFileSync(this.filePath, 'utf8')));
      } catch (err) {
        console.warn(`[usage-stats] 统计文件读取失败，已从空统计开始：${err.message}`);
        this.stats = emptyUsageStats();
      }
    }
  }

  /** 记录一次 API 用量（entry 口径见 usage-stats.js 的 recordUsage）。 */
  record(entry) {
    recordUsage(this.stats, { ...entry, source: 'api' });
    this.#scheduleSave();
  }

  /** 面板用汇总（与网页端同结构，另附来源标记）。 */
  snapshot() {
    return { source: 'api', ...summarizeUsage(this.stats) };
  }

  /** 清空全部 API 统计并立即落盘。 */
  clear() {
    this.stats = clearUsageStats();
    this.#saveNow();
    return this.snapshot();
  }

  /** 立即落盘（进程退出/测试收尾时调用）。 */
  flush() {
    if (this.saveTimer) {
      clearTimeout(this.saveTimer);
      this.saveTimer = null;
    }
    this.#saveNow();
  }

  #scheduleSave() {
    if (!this.filePath || this.saveTimer) return;
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null;
      this.#saveNow();
    }, 800);
    this.saveTimer.unref?.();
  }

  #saveNow() {
    if (!this.filePath) return;
    try {
      mkdirSync(dirname(this.filePath), { recursive: true });
      const tmp = `${this.filePath}.tmp`;
      writeFileSync(tmp, JSON.stringify(this.stats));
      renameSync(tmp, this.filePath);
    } catch (err) {
      console.warn(`[usage-stats] 统计保存失败（不影响 API 服务）：${err.message}`);
    }
  }
}
