/**
 * ONNX 图改写：把 Pow(x, 2) 改写为 Mul(x, x)。
 *
 * 背景：SD 1.5 的 CLIP 文本编码器保留了分解后的 LayerNorm（ReduceMean → Sub → Pow → …）。
 * 在 fp16 下，Pow 节点在 WebGPU 与 CPU 后端都可能没有可用的内核（报错 “Provider type for Pow node … is not set”、
 * “Could not find an implementation for Pow(13)”）。其他模型通常在导出时已经融合了 LayerNorm，因此没有 Pow。
 *
 * 做法：直接改写 protobuf 的字节。只修改 NodeProto 的 op_type 与第二个输入，其余字节原样保留，
 * 不解析、不重新序列化权重（权重可能有数百 MB）。只改写指数为常量 2 的 Pow；其他 Pow 保持不变。
 *
 * 线格式参考：ONNX 的 onnx.proto（ModelProto.graph = 7；GraphProto.node = 1、initializer = 5；
 * NodeProto.input = 1、op_type = 4、attribute = 5；TensorProto.dims = 1、data_type = 2、float_data = 4、
 * int32_data = 5、int64_data = 7、name = 8、raw_data = 9、double_data = 10；AttributeProto.name = 1、t = 5）。
 */

const WIRE_VARINT = 0;
const WIRE_I64 = 1;
const WIRE_LEN = 2;
const WIRE_I32 = 5;

// TensorProto.DataType
const DT = { FLOAT: 1, INT32: 6, INT64: 7, DOUBLE: 11, FLOAT16: 10 };

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
    let vs = pos;
    let ve = pos;
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

/** 属性（AttributeProto，位于 buf 的字段 f 中）的名称是否为 name。 */
function attrNameIs(buf, f, name) {
  const attrBuf = buf.subarray(f.vs, f.ve);
  const nameField = parseFields(attrBuf).find((y) => y.num === 1 && y.wire === WIRE_LEN);
  return !!nameField && str(attrBuf, nameField) === name;
}

/** 读取 TensorProto 中的单个标量值是否等于 2（只识别标量或形状为 [1] 的张量）。 */
function tensorIsTwo(buf) {
  const fields = parseFields(buf);
  let dataType = 0;
  let count = 1;
  let raw = null;
  let floats = null;
  let ints32 = null;
  let ints64 = null;
  let doubles = null;
  for (const f of fields) {
    if (f.num === 1) {
      // dims：可能是重复的 varint，也可能是打包的 varint
      if (f.wire === WIRE_VARINT) count *= f.value;
      else if (f.wire === WIRE_LEN) {
        let p = f.vs;
        while (p < f.ve) {
          let d;
          [d, p] = readVarint(buf, p);
          count *= d;
        }
      }
    } else if (f.num === 2 && f.wire === WIRE_VARINT) dataType = f.value;
    else if (f.num === 9 && f.wire === WIRE_LEN) raw = buf.subarray(f.vs, f.ve);
    else if (f.num === 4) floats = (floats || []).concat([f]);
    else if (f.num === 5) ints32 = (ints32 || []).concat([f]);
    else if (f.num === 7) ints64 = (ints64 || []).concat([f]);
    else if (f.num === 10) doubles = (doubles || []).concat([f]);
  }
  if (count !== 1) return false;
  const dv = (bytes) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (raw) {
    const v = dv(raw);
    if (dataType === DT.FLOAT16 && raw.length === 2) return v.getUint16(0, true) === 0x4000;
    if (dataType === DT.FLOAT && raw.length === 4) return v.getFloat32(0, true) === 2;
    if (dataType === DT.DOUBLE && raw.length === 8) return v.getFloat64(0, true) === 2;
    if (dataType === DT.INT32 && raw.length === 4) return v.getInt32(0, true) === 2;
    if (dataType === DT.INT64 && raw.length === 8) return v.getBigInt64(0, true) === 2n;
    return false;
  }
  // 非 raw 的存放方式：只处理单个元素的情况
  const single = (list) => (list && list.length === 1 ? list[0] : null);
  if (dataType === DT.FLOAT && floats) {
    const f = single(floats);
    return !!f && f.wire === WIRE_I32 && dv(buf.subarray(f.vs, f.ve)).getFloat32(0, true) === 2;
  }
  if (dataType === DT.DOUBLE && doubles) {
    const f = single(doubles);
    return !!f && f.wire === WIRE_I64 && dv(buf.subarray(f.vs, f.ve)).getFloat64(0, true) === 2;
  }
  if (dataType === DT.INT32 && ints32) {
    const f = single(ints32);
    return !!f && f.wire === WIRE_VARINT && f.value === 2;
  }
  if (dataType === DT.INT64 && ints64) {
    const f = single(ints64);
    return !!f && f.wire === WIRE_VARINT && f.value === 2;
  }
  return false;
}

/**
 * 改写 ONNX 模型字节：指数为常量 2 的 Pow 改为 Mul(x, x)。
 * 返回 {bytes, rewritten, kept}；没有可改写的节点时 bytes 与输入相同（同一对象）。
 */
export function rewritePowToMul(modelBytes) {
  const bytes = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  const top = parseFields(bytes);
  const graphField = top.find((f) => f.num === 7 && f.wire === WIRE_LEN);
  if (!graphField) return { bytes, rewritten: 0, kept: 0 };
  const gBuf = bytes.subarray(graphField.vs, graphField.ve);
  const gFields = parseFields(gBuf);

  // 常量：initializer 名称 → TensorProto；Constant 节点输出名 → TensorProto
  const constants = new Map();
  for (const f of gFields) {
    if (f.num === 5 && f.wire === WIRE_LEN) {
      const tBuf = gBuf.subarray(f.vs, f.ve);
      const tf = parseFields(tBuf);
      const nameField = tf.find((x) => x.num === 8 && x.wire === WIRE_LEN);
      if (nameField) constants.set(str(tBuf, nameField), tBuf);
    }
  }
  const nodeFields = gFields.filter((f) => f.num === 1 && f.wire === WIRE_LEN);
  for (const f of nodeFields) {
    const nBuf = gBuf.subarray(f.vs, f.ve);
    const nf = parseFields(nBuf);
    const op = nf.find((x) => x.num === 4 && x.wire === WIRE_LEN);
    if (!op || str(nBuf, op) !== 'Constant') continue;
    const outName = nf.find((x) => x.num === 2 && x.wire === WIRE_LEN);
    const attr = nf.find((x) => x.num === 5 && x.wire === WIRE_LEN && attrNameIs(nBuf, x, 'value'));
    if (!outName || !attr) continue;
    const aBuf = nBuf.subarray(attr.vs, attr.ve);
    const t = parseFields(aBuf).find((y) => y.num === 5 && y.wire === WIRE_LEN);
    if (t) constants.set(str(nBuf, outName), aBuf.subarray(t.vs, t.ve));
  }

  let rewritten = 0;
  let kept = 0;
  const parts = [];
  for (const f of gFields) {
    if (!(f.num === 1 && f.wire === WIRE_LEN)) {
      parts.push(gBuf.subarray(f.s, f.e));
      continue;
    }
    const nBuf = gBuf.subarray(f.vs, f.ve);
    const nf = parseFields(nBuf);
    const op = nf.find((x) => x.num === 4 && x.wire === WIRE_LEN);
    const inputs = nf.filter((x) => x.num === 1 && x.wire === WIRE_LEN);
    const isPow = op && str(nBuf, op) === 'Pow' && inputs.length === 2;
    if (!isPow) {
      parts.push(gBuf.subarray(f.s, f.e));
      continue;
    }
    const expName = str(nBuf, inputs[1]);
    const expTensor = constants.get(expName);
    if (!expTensor || !tensorIsTwo(expTensor)) {
      kept += 1;
      parts.push(gBuf.subarray(f.s, f.e));
      continue;
    }
    // 重写：op_type 改为 Mul；第二个输入改为第一个输入（同样的原始字节）
    const newParts = [];
    for (const x of nf) {
      if (x === op) newParts.push(lenField(4, te.encode('Mul')));
      else if (x === inputs[1]) newParts.push(nBuf.subarray(inputs[0].s, inputs[0].e));
      else newParts.push(nBuf.subarray(x.s, x.e));
    }
    rewritten += 1;
    parts.push(lenField(1, concat(newParts)));
  }

  if (rewritten === 0) return { bytes, rewritten, kept };

  const newGraph = concat(parts);
  const topParts = [];
  for (const f of top) {
    if (f === graphField) topParts.push(lenField(7, newGraph));
    else topParts.push(bytes.subarray(f.s, f.e));
  }
  return { bytes: concat(topParts), rewritten, kept };
}
