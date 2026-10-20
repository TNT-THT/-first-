/**
 * 生成后的校验：管理员标记的“错误说法 → 正确说法”规则，用于检查 AI 的回答（纯逻辑，便于测试）。
 *
 * - 规则由管理员在某个 AI 的回复上“标记错误”时创建，按 modelId 分开保存在本浏览器的 localStorage 中，不上传。
 * - 回答生成完成后，若回答（去掉其中的正确说法后）仍包含错误说法，就在回答下方给出提示；不会重新生成或改写回答。
 * - 匹配在规范化后的文本上进行：忽略大小写、全半角差异和空白。错误说法过短（规范化后不足 2 个字符）不建立规则。
 */

export const CHECKS_KEY = 'oj-ai.checks.v1';
export const MAX_CHECK_RULES_PER_MODEL = 50;

/** 规范化：NFKC（全角转半角）、小写、去掉所有空白。 */
export function normalizeForCheck(text) {
  return String(text ?? '')
    .normalize('NFKC')
    .toLowerCase()
    .replace(/\s+/g, '');
}

function readAll(storage) {
  try {
    const data = JSON.parse(storage?.getItem(CHECKS_KEY) || 'null');
    if (!data || typeof data !== 'object' || !data.models || typeof data.models !== 'object') {
      return { models: {} };
    }
    return data;
  } catch {
    return { models: {} };
  }
}

function writeAll(data, storage) {
  try {
    storage?.setItem(CHECKS_KEY, JSON.stringify(data));
  } catch {
    // 存储不可用（如隐私模式或配额已满）时忽略，本次会话仍然有效
  }
}

/** 某模型的校验规则（最新的在后）。 */
export function listCheckRules(modelId, { storage = globalThis.localStorage } = {}) {
  const list = readAll(storage).models[modelId];
  return Array.isArray(list) ? list.map((r) => ({ ...r })) : [];
}

/** 检查输入是否可以建立规则：返回错误信息，合格则返回 null（不写入任何数据）。 */
export function checkRuleError(wrong, right) {
  const w = String(wrong ?? '').trim();
  const r = String(right ?? '').trim();
  if (!w) return '请填写错误的说法。';
  if (!r) return '请填写正确的说法。';
  const nw = normalizeForCheck(w);
  if (nw.length < 2) return '错误的说法太短（至少 2 个字符），容易误判，请写得更具体。';
  if (nw === normalizeForCheck(r)) return '错误的说法与正确的说法相同。';
  return null;
}

/**
 * 新增一条规则。若同一模型中已有相同的错误说法（规范化后相同），则更新其正确说法。
 * 返回 {rule, updated} 或 {error}。
 */
export function addCheckRule(modelId, { wrong, right }, { storage = globalThis.localStorage, now = Date.now() } = {}) {
  if (!modelId) return { error: '缺少 AI 信息，无法记录。' };
  const error = checkRuleError(wrong, right);
  if (error) return { error };
  const w = String(wrong).trim();
  const r = String(right).trim();
  const nw = normalizeForCheck(w);

  const data = readAll(storage);
  const list = Array.isArray(data.models[modelId]) ? data.models[modelId] : [];
  const idx = list.findIndex((x) => normalizeForCheck(x.wrong) === nw);
  if (idx >= 0) {
    const rule = { ...list[idx], right: r, ts: now };
    list[idx] = rule;
    data.models[modelId] = list;
    writeAll(data, storage);
    return { rule: { ...rule }, updated: true };
  }
  const rule = {
    id: `${now.toString(36)}${Math.random().toString(36).slice(2, 7)}`,
    wrong: w,
    right: r,
    ts: now,
  };
  data.models[modelId] = [...list, rule].slice(-MAX_CHECK_RULES_PER_MODEL);
  writeAll(data, storage);
  return { rule: { ...rule }, updated: false };
}

/** 删除一条规则。删除后若该模型没有规则，则移除该模型的记录。 */
export function removeCheckRule(modelId, ruleId, { storage = globalThis.localStorage } = {}) {
  const data = readAll(storage);
  const list = data.models[modelId];
  if (!Array.isArray(list)) return;
  data.models[modelId] = list.filter((r) => r.id !== ruleId);
  if (!data.models[modelId].length) delete data.models[modelId];
  writeAll(data, storage);
}

/** 紧跟在错误说法之前的否定词：如“不是 O(n)”表示否定，不计为违规。 */
const NEGATION_BEFORE = /(不是|不为|并非|并不是|而非|而不是|不等于|不|非|没有|not)$/i;

/**
 * 检查回答：返回命中的规则列表 [{rule}]。
 * 1. 先把回答中的正确说法去掉，避免“O(log n)，而不是 O(n)”中的正确说法干扰匹配；
 * 2. 错误说法前紧跟否定词的出现不计为违规（如“不是 O(n)”）；
 * 3. 其余仍是字符串匹配，换说法或同义表达会漏判，这也是保留人工标记作为兜底的原因。
 */
export function findViolations(answer, rules) {
  const a = normalizeForCheck(answer);
  if (!a || !Array.isArray(rules) || !rules.length) return [];
  const hits = [];
  for (const rule of rules) {
    const nw = normalizeForCheck(rule.wrong);
    const nr = normalizeForCheck(rule.right);
    if (nw.length < 2) continue;
    const stripped = nr ? a.split(nr).join('') : a;
    let idx = stripped.indexOf(nw);
    while (idx >= 0) {
      const before = stripped.slice(Math.max(0, idx - 6), idx);
      if (!NEGATION_BEFORE.test(before)) {
        hits.push({ rule });
        break;
      }
      idx = stripped.indexOf(nw, idx + nw.length);
    }
  }
  return hits;
}
