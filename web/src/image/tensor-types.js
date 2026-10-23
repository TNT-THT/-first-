/**
 * 张量数据类型转换（纯逻辑）。SD 1.5 的 fp16 ONNX 模型输入输出为 float16，
 * ONNX Runtime Web 中 float16 张量的数据用 Uint16Array 存放（IEEE 754 半精度位模式）。
 */
const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** 单个 float → float16 位模式（就近舍入，偶数优先）。 */
export function floatToHalfBits(x) {
  f32[0] = x;
  const f = u32[0];
  const sign = (f >>> 16) & 0x8000;
  const exp = (f >>> 23) & 0xff;
  let mant = f & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && half & 1)) half += 1;
    return sign | half;
  }
  let half = (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && half & 1)) half += 1;
  return sign | half;
}

/** float16 位模式 → number。 */
export function halfBitsToFloat(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return sign * m * 2 ** -24;
  if (e === 0x1f) return m ? NaN : sign * Infinity;
  return sign * (1 + m / 1024) * 2 ** (e - 15);
}

/**
 * 把 Float32Array 按 ONNX 类型编码成 ORT 张量可用的数组。
 * 支持 float32、float16、int32、int64（int64 使用 BigInt64Array）。
 */
export function encodeForType(type, values) {
  switch (type) {
    case 'float32':
      return values instanceof Float32Array ? values : Float32Array.from(values);
    case 'float16': {
      const out = new Uint16Array(values.length);
      for (let i = 0; i < values.length; i += 1) out[i] = floatToHalfBits(values[i]);
      return out;
    }
    case 'int32':
      return Int32Array.from(values);
    case 'int64':
      return BigInt64Array.from(Array.from(values, (v) => BigInt(v)));
    default:
      throw new Error(`不支持的张量类型：${type}`);
  }
}

/** ORT 输出数据 → Float32Array（float16 位模式会被还原）。 */
export function decodeFromType(type, data) {
  if (type === 'float16') {
    const out = new Float32Array(data.length);
    for (let i = 0; i < data.length; i += 1) out[i] = halfBitsToFloat(data[i]);
    return out;
  }
  if (type === 'float32') return Float32Array.from(data);
  if (type === 'int32' || type === 'int64') return Float32Array.from(data, (v) => Number(v));
  throw new Error(`不支持的张量类型：${type}`);
}
