"""Graph-only patch: rewrite the float64 LayerNorm variance path of the SD1.5 UNet to float32.
Weights (weights.pb) are untouched; only the graph protobuf is rewritten."""
import sys
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, AttributeProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)
changed = {'cast': 0, 'const': 0, 'vi': 0}

for n in m.graph.node:
    if n.op_type == 'Cast':
        for a in n.attribute:
            if a.name == 'to' and a.i == TensorProto.DOUBLE:
                a.i = TensorProto.FLOAT
                changed['cast'] += 1
    for a in n.attribute:
        if a.type == AttributeProto.TENSOR and a.t.data_type == TensorProto.DOUBLE:
            arr = numpy_helper.to_array(a.t).astype(np.float32)
            a.t.CopyFrom(numpy_helper.from_array(arr, a.t.name))
            changed['const'] += 1

for t in m.graph.initializer:
    if t.data_type == TensorProto.DOUBLE:
        raise SystemExit('unexpected double initializer: ' + t.name)

for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
    if vi.type.tensor_type.elem_type == TensorProto.DOUBLE:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT
        changed['vi'] += 1

left = 0
for n in m.graph.node:
    for a in n.attribute:
        if a.name == 'to' and a.i == TensorProto.DOUBLE:
            left += 1
        if a.type == AttributeProto.TENSOR and a.t.data_type == TensorProto.DOUBLE:
            left += 1
print('changed', changed, 'remaining double refs', left)
onnx.save(m, dst)
print('saved', dst)
