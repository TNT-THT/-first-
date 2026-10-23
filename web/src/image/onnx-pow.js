/**
 * ONNX 图改写：把指数为常量 2 的 Pow(x, 2) 改写为 Mul(x, x)。
 *
 * 背景：SD 1.5 的 CLIP 文本编码器保留了分解后的 LayerNorm（ReduceMean → Sub → Pow → …）。
 * fp16 的 Pow 节点在 WebGPU 与 CPU 后端都可能没有可用的内核（报错 “Provider type for Pow node … is not set”、
 * “Could not find an implementation for Pow(13)”）。其他模型通常在导出时已经融合了 LayerNorm，因此没有 Pow。
 *
 * 做法：直接改写 protobuf 的字节。只修改 NodeProto 的 op_type 与第二个输入，其余字节原样保留，
 * 不重新序列化权重（权重可能有数百 MB）。
 *
 * 指数的常量可能有多种存放方式，都会识别：
 *   - 初始化器（TensorProto）：raw_data，或 float_data / int32_data（fp16 的位模式）/ int64_data / double_data，
 *     打包（packed）与非打包两种编码；
 *   - Constant 节点：value（张量）、value_float、value_int 属性；
 *   - 经过 Identity 或 Cast 节点传入 Pow 的上述常量。
 * 指数不是常量 2 的 Pow 保持不变，并在结果中列出节点名。
 *
 * 线格式（onnx.proto）：ModelProto.graph = 7；GraphProto.node = 1、initializer = 5；
 * NodeProto.input = 1、output = 2、name = 3、op_type = 4、attribute = 5；
 * TensorProto.dims = 1、data_type = 2、float_data = 4、int32_data = 5、int64_data = 7、name = 8、
 * raw_data = 9、double_data = 10；AttributeProto.name = 1、f = 2、i = 3、t = 5。
 */

const WIRE_VARINT = 0;
const WIRE_I64 = 1;
const WIRE_LEN = 2;
const WIRE_I32 = 5;

// TensorProto.DataType
const DT = { FLOAT: 1, INT32: 6, INT64: 7, DOUBLE: 11, FLOAT16: 10 };

const MAX_DEPTH = 16; // Identity / Cast 链的最大长度

const te = new TextEncoder();
const td = new TextDecoder();

function readVarint(buf, pos) {
  let result = 0;
  let mul = 1;
  for (;;) {
    if (pos >= buf.length) throw new Error('ONNX 文件格式错误：变长整数被截断');
    const b = buf[pos++];
    result += (b & 0x7f) * mul;
    if (!(b & 0x80)) break;
    mul *= 128;
  }
  return [result, pos];
}

/** 解析一层字段。返回 {num, wire, s, e, vs, ve, value}：s..e 为整个字段（含标签），vs..ve 为值的字节范围。 */
function parseFields(buf) {
  const out = [];
  let pos = 0;
  while (pos < buf.length) {
    const s = pos;
    let tag;
    [tag, pos] = readVarint(buf, pos);
    const num = Math.floor(tag / 8);
    const wire = tag % 8;
    let vs;
    let ve;
    let value;
    if (wire === WIRE_VARINT) {
      [value, pos] = readVarint(buf, pos);
      vs = ve = pos;
    } else if (wire === WIRE_LEN) {
      let len;
      [len, pos] = readVarint(buf, pos);
      vs = pos;
      ve = pos + len;
      pos = ve;
    } else if (wire === WIRE_I64) {
      vs = pos;
      ve = pos + 8;
      pos = ve;
    } else if (wire === WIRE_I32) {
      vs = pos;
      ve = pos + 4;
      pos = ve;
    } else {
      throw new Error(`ONNX 文件格式错误：不支持的线类型 ${wire}`);
    }
    if (pos > buf.length) throw new Error('ONNX 文件格式错误：字段超出范围');
    out.push({ num, wire, s, e: pos, vs, ve, value });
  }
  return out;
}

function encodeVarint(n) {
  const bytes = [];
  let v = n;
  while (v >= 128) {
    bytes.push((v % 128) | 0x80);
    v = Math.floor(v / 128);
  }
  bytes.push(v);
  return Uint8Array.from(bytes);
}

/** 重新编码一个长度分隔字段（标签 + 长度 + 内容）。 */
function lenField(num, payload) {
  const head = encodeVarint(num * 8 + WIRE_LEN);
  const len = encodeVarint(payload.length);
  const out = new Uint8Array(head.length + len.length + payload.length);
  out.set(head, 0);
  out.set(len, head.length);
  out.set(payload, head.length + len.length);
  return out;
}

function concat(parts) {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

function str(buf, f) {
  return td.decode(buf.subarray(f.vs, f.ve));
}

/** 解析打包的变长整数（packed varint）。 */
function packedVarints(buf, f) {
  const out = [];
  let p = f.vs;
  while (p < f.ve) {
    let v;
    [v, p] = readVarint(buf, p);
    out.push(v);
  }
  return out;
}

/** IEEE 754 半精度 → 数值。 */
function halfToNumber(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/**
 * 解码 TensorProto 的全部元素（返回数值数组）；无法识别的类型返回 null。
 * 支持 raw_data，以及 float_data / int32_data / int64_data / double_data（打包与非打包）。
 */
function decodeTensor(buf) {
  const fields = parseFields(buf);
  let dataType = 0;
  let count = 1;
  let raw = null;
  const floats = [];
  const ints32 = [];
  const ints64 = [];
  const doubles = [];
  for (const f of fields) {
    if (f.num === 1) {
      if (f.wire === WIRE_VARINT) count *= f.value;
      else if (f.wire === WIRE_LEN) for (const d of packedVarints(buf, f)) count *= d;
    } else if (f.num === 2 && f.wire === WIRE_VARINT) dataType = f.value;
    else if (f.num === 9 && f.wire === WIRE_LEN) raw = buf.subarray(f.vs, f.ve);
    else if (f.num === 4) {
      if (f.wire === WIRE_I32) floats.push(f);
      else if (f.wire === WIRE_LEN) for (let p = f.vs; p < f.ve; p += 4) floats.push({ wire: WIRE_I32, vs: p });
    } else if (f.num === 5) {
      if (f.wire === WIRE_VARINT) ints32.push(f.value);
      else if (f.wire === WIRE_LEN) ints32.push(...packedVarints(buf, f));
    } else if (f.num === 7) {
      if (f.wire === WIRE_VARINT) ints64.push(f.value);
      else if (f.wire === WIRE_LEN) ints64.push(...packedVarints(buf, f));
    } else if (f.num === 10) {
      if (f.wire === WIRE_I64) doubles.push(f);
      else if (f.wire === WIRE_LEN) for (let p = f.vs; p < f.ve; p += 8) doubles.push({ wire: WIRE_I64, vs: p });
    }
  }
  const dv = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (raw) {
    const v = dv(raw);
    const n = raw.length;
    if (dataType === DT.FLOAT16 && n === count * 2) return Array.from({ length: count }, (_, i) => halfToNumber(v.getUint16(i * 2, true)));
    if (dataType === DT.FLOAT && n === count * 4) return Array.from({ length: count }, (_, i) => v.getFloat32(i * 4, true));
    if (dataType === DT.DOUBLE && n === count * 8) return Array.from({ length: count }, (_, i) => v.getFloat64(i * 8, true));
    if (dataType === DT.INT32 && n === count * 4) return Array.from({ length: count }, (_, i) => v.getInt32(i * 4, true));
    if (dataType === DT.INT64 && n === count * 8) return Array.from({ length: count }, (_, i) => Number(v.getBigInt64(i * 8, true)));
    return null;
  }
  const bytesOf = (f, size) => new DataView(buf.buffer, buf.byteOffset + f.vs, size);
  if (dataType === DT.FLOAT && floats.length) return floats.map((f) => bytesOf(f, 4).getFloat32(0, true));
  if (dataType === DT.DOUBLE && doubles.length) return doubles.map((f) => bytesOf(f, 8).getFloat64(0, true));
  // fp16 的数值存放在 int32_data 中（每个元素是 16 位的位模式）
  if (dataType === DT.FLOAT16 && ints32.length) return ints32.map((x) => halfToNumber(x & 0xffff));
  if (dataType === DT.INT32 && ints32.length) return ints32.slice();
  if (dataType === DT.INT64 && ints64.length) return ints64.slice();
  return null;
}

/** 读取 AttributeProto 的名称。 */
function attrName(buf, f) {
  const attrBuf = buf.subarray(f.vs, f.ve);
  const nameField = parseFields(attrBuf).find((y) => y.num === 1 && y.wire === WIRE_LEN);
  return nameField ? str(attrBuf, nameField) : '';
}

/** 解析一个节点（NodeProto 的字节）：名称、操作类型、输入、输出、属性。 */
function parseNode(buf, field) {
  const nBuf = buf.subarray(field.vs, field.ve);
  const nf = parseFields(nBuf);
  const opField = nf.find((x) => x.num === 4 && x.wire === WIRE_LEN);
  const inputs = nf.filter((x) => x.num === 1 && x.wire === WIRE_LEN);
  const outputs = nf.filter((x) => x.num === 2 && x.wire === WIRE_LEN);
  const attrs = nf.filter((x) => x.num === 5 && x.wire === WIRE_LEN);
  return {
    field,
    nBuf,
    nf,
    opField,
    op: opField ? str(nBuf, opField) : '',
    inputs,
    inputNames: inputs.map((x) => str(nBuf, x)),
    outputNames: outputs.map((x) => str(nBuf, x)),
    attrs: attrs.map((a) => ({ name: attrName(nBuf, a), f: a, buf: nBuf })),
  };
}

/** 从 Constant 节点的属性中取出数值（value / value_float / value_int）。 */
function constantValues(node) {
  for (const a of node.attrs) {
    const aBuf = a.buf.subarray(a.f.vs, a.f.ve);
    const af = parseFields(aBuf);
    if (a.name === 'value') {
      const t = af.find((y) => y.num === 5 && y.wire === WIRE_LEN);
      if (t) return decodeTensor(aBuf.subarray(t.vs, t.ve));
    } else if (a.name === 'value_float') {
      const f = af.find((y) => y.num === 2 && y.wire === WIRE_I32);
      if (f) return [new DataView(aBuf.buffer, aBuf.byteOffset + f.vs, 4).getFloat32(0, true)];
    } else if (a.name === 'value_int') {
      const i = af.find((y) => y.num === 3 && y.wire === WIRE_VARINT);
      if (i) return [i.value];
    }
  }
  return null;
}

/**
 * 改写 ONNX 模型字节：指数为常量 2 的 Pow 改为 Mul(x, x)。
 * 返回 {bytes, rewritten, kept, keptNodes}：keptNodes 为保留下来的 Pow 节点名（指数不是常量 2 或无法识别）。
 * 没有可改写的节点时 bytes 与输入相同（同一对象）。
 */
export function rewritePowToMul(modelBytes) {
  const bytes = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  const top = parseFields(bytes);
  const graphField = top.find((f) => f.num === 7 && f.wire === WIRE_LEN);
  if (!graphField) return { bytes, rewritten: 0, kept: 0, keptNodes: [] };
  const gBuf = bytes.subarray(graphField.vs, graphField.ve);
  const gFields = parseFields(gBuf);

  // 初始化器：名称 → TensorProto 字节
  const initializers = new Map();
  for (const f of gFields) {
    if (f.num === 5 && f.wire === WIRE_LEN) {
      const tBuf = gBuf.subarray(f.vs, f.ve);
      const nameField = parseFields(tBuf).find((x) => x.num === 8 && x.wire === WIRE_LEN);
      if (nameField) initializers.set(str(tBuf, nameField), tBuf);
    }
  }

  // 节点：输出名 → 节点（用于追踪 Constant / Identity / Cast）
  const nodeFields = gFields.filter((f) => f.num === 1 && f.wire === WIRE_LEN);
  const nodes = nodeFields.map((f) => parseNode(gBuf, f));
  const producers = new Map();
  for (const node of nodes) for (const o of node.outputNames) producers.set(o, node);

  // 解析一个张量名的数值（沿 Identity / Cast 向上追踪，有深度上限）
  const resolve = (name, depth = 0) => {
    if (depth > MAX_DEPTH || !name) return null;
    if (initializers.has(name)) return decodeTensor(initializers.get(name));
    const p = producers.get(name);
    if (!p) return null;
    if (p.op === 'Constant') return constantValues(p);
    if (p.op === 'Identity' || p.op === 'Cast') return resolve(p.inputNames[0], depth + 1);
    return null;
  };
  const isTwo = (values) => !!values && values.length === 1 && values[0] === 2;

  let rewritten = 0;
  const keptNodes = [];
  const rewriteOf = new Map(); // 节点字段 → 替换后的字节
  for (const node of nodes) {
    if (node.op !== 'Pow' || node.inputs.length !== 2) continue;
    if (!isTwo(resolve(node.inputNames[1]))) {
      const nameField = node.nf.find((x) => x.num === 3 && x.wire === WIRE_LEN);
      keptNodes.push(nameField ? str(node.nBuf, nameField) : node.inputNames.join(','));
      continue;
    }
    // 重写：op_type 改为 Mul；第二个输入改为第一个输入（同样的原始字节）
    const newParts = [];
    for (const x of node.nf) {
      if (x === node.opField) newParts.push(lenField(4, te.encode('Mul')));
      else if (x === node.inputs[1]) newParts.push(node.nBuf.subarray(node.inputs[0].s, node.inputs[0].e));
      else newParts.push(node.nBuf.subarray(x.s, x.e));
    }
    rewriteOf.set(node.field, lenField(1, concat(newParts)));
    rewritten += 1;
  }

  if (rewritten === 0) return { bytes, rewritten, kept: keptNodes.length, keptNodes };

  const parts = [];
  for (const f of gFields) {
    if (rewriteOf.has(f)) parts.push(rewriteOf.get(f));
    else parts.push(gBuf.subarray(f.s, f.e));
  }
  const newGraph = concat(parts);
  const topParts = [];
  for (const f of top) {
    if (f === graphField) topParts.push(lenField(7, newGraph));
    else topParts.push(bytes.subarray(f.s, f.e));
  }
  return { bytes: concat(topParts), rewritten, kept: keptNodes.length, keptNodes };
}
