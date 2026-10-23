/**
 * CLIP 分词器（纯逻辑，对应 transformers 的 CLIPTokenizer，用于 SD 1.5 的文本编码器）。
 *   - 文本先转小写、合并空白，再用 CLIP 的正则切分；
 *   - 每个片段按 UTF-8 字节映射为可见字符，再做 BPE 合并（最后一个字符带 </w> 后缀）；
 *   - 输出为 [BOS, ...tokens, EOS]，超过 maxLength 时截断（保留 EOS），不足时用 pad 补齐。
 * 词表（vocab.json）与合并规则（merges.txt）由调用方传入，便于在测试中使用小词表。
 */

// CLIP 的切分规则（与 transformers 相同，忽略大小写）。
const PATTERN = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|\p{N}|[^\s\p{L}\p{N}]+/giu;

/** GPT-2 / CLIP 的 字节 → 可见字符 映射。 */
export function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i += 1) bs.push(i);
  for (let i = 161; i <= 172; i += 1) bs.push(i);
  for (let i = 174; i <= 255; i += 1) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b += 1) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const map = new Map();
  bs.forEach((b, i) => map.set(b, String.fromCodePoint(cs[i])));
  return map;
}

const BYTE_MAP = bytesToUnicode();

/** 解析 merges.txt：忽略以 # 开头的版本行与空行，返回 "a b" → 排名 的映射。 */
export function parseMerges(text) {
  const ranks = new Map();
  let rank = 0;
  for (const line of String(text).split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    ranks.set(trimmed, rank);
    rank += 1;
  }
  return ranks;
}

export function createClipTokenizer({ vocab, merges, bosToken = '<|startoftext|>', eosToken = '<|endoftext|>', padToken = eosToken, unkToken = eosToken, maxLength = 77 }) {
  const encoder = vocab instanceof Map ? vocab : new Map(Object.entries(vocab));
  const ranks = merges instanceof Map ? merges : parseMerges(merges);
  const cache = new Map();
  const bos = idOf(bosToken);
  const eos = idOf(eosToken);
  const pad = idOf(padToken);
  const unk = idOf(unkToken);

  function idOf(token) {
    const id = encoder.get(token);
    if (id === undefined) throw new Error(`CLIP 词表中缺少特殊符号 ${token}`);
    return id;
  }

  function bpe(token) {
    if (cache.has(token)) return cache.get(token);
    let word = [...token.slice(0, -1), `${token.slice(-1)}</w>`];
    let pairs = pairsOf(word);
    if (pairs.length === 0) {
      const out = [word.join('')];
      cache.set(token, out);
      return out;
    }
    for (;;) {
      let best = null;
      let bestRank = Infinity;
      for (const [a, b] of pairs) {
        const r = ranks.get(`${a} ${b}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = [a, b];
        }
      }
      if (best === null) break;
      const [first, second] = best;
      const next = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          next.push(first + second);
          i += 2;
        } else {
          next.push(word[i]);
          i += 1;
        }
      }
      word = next;
      if (word.length === 1) break;
      pairs = pairsOf(word);
    }
    cache.set(token, word);
    return word;
  }

  return {
    bos,
    eos,
    pad,
    maxLength,
    /** 文本 → 完整的 token id 数组（包含 BOS 与 EOS，不截断），用于计算占用的 token 数。 */
    rawTokenize(text) {
      const clean = String(text || '').replace(/\s+/g, ' ').trim().toLowerCase();
      const ids = [bos];
      for (const m of clean.matchAll(PATTERN)) {
        // 特殊符号直接映射为 id（与 transformers 一致），不参与 BPE
        if (m[0] === bosToken || m[0] === eosToken) {
          ids.push(m[0] === bosToken ? bos : eos);
          continue;
        }
        const bytes = new TextEncoder().encode(m[0]);
        const mapped = [...bytes].map((b) => BYTE_MAP.get(b)).join('');
        for (const piece of bpe(mapped)) {
          const id = encoder.get(piece);
          ids.push(id === undefined ? unk : id);
        }
      }
      ids.push(eos);
      return ids;
    },
    /** 文本 → token id 数组（不含补齐），包含 BOS 与 EOS，超长时截断并保留 EOS。 */
    tokenize(text) {
      const ids = this.rawTokenize(text);
      if (ids.length > maxLength) {
        return [...ids.slice(0, maxLength - 1), eos];
      }
      return ids;
    },
    /** 固定长度的 Int32Array（补齐到 maxLength），供文本编码器使用。 */
    encode(text) {
      const ids = this.tokenize(text);
      const out = new Int32Array(maxLength).fill(pad);
      out.set(ids);
      return out;
    },
  };
}

function pairsOf(word) {
  const out = [];
  for (let i = 0; i < word.length - 1; i += 1) out.push([word[i], word[i + 1]]);
  return out;
}
