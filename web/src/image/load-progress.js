// 生图模型加载的整体进度（纯函数，便于测试）。
// 读取文件字节占进度的 90%，会话创建占 10%。读取与会话创建是交错进行的
// （每个会话创建前先读取它的文件），所以两部分都按“已完成”的量计算，
// 同一时刻的结果只会随已完成的量增加，不会因为开始某个会话而跳变。

export const LOAD_SHARE = 0.9;

/**
 * @param {object} p
 * @param {number} p.loaded 已读取的字节数（实际读入的文件）
 * @param {number} p.total 预计需要读取的字节数
 * @param {number} p.sessionsDone 已创建完成的会话数（开始创建不计入）
 * @param {number} p.sessionTotal 会话总数
 * @returns {number} 0–1 之间的整体进度
 */
export function loadFraction({ loaded, total, sessionsDone, sessionTotal }) {
  const read = total > 0 ? Math.min(1, Math.max(0, loaded) / total) : 0;
  const sessions = sessionTotal > 0 ? Math.min(1, Math.max(0, sessionsDone) / sessionTotal) : 0;
  return LOAD_SHARE * read + (1 - LOAD_SHARE) * sessions;
}
