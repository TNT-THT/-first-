#!/usr/bin/env python3
"""
生成与 Stable Diffusion 1.5（diffusers ONNX 导出）接口一致的“微型”模型，用于端到端测试。

这些模型不是真正的 SD 1.5：只复刻 输入/输出名称、数据类型、外部权重（unet/weights.pb）与
张量形状的约定，让 ONNX Runtime Web 的加载与调用路径可以在没有 1.7 GB 权重的环境中运行。

用法（需要 pip install onnx numpy）：
    python3 scripts/image/make-tiny-sd15.py

输出：
    test/fixtures/sd15-tiny/        float32 接口（输入输出均为 FLOAT，timestep 为 FLOAT）
    test/fixtures/sd15-tiny-fp16/   float16 接口（与 nmkd fp16 版本一致；timestep 为 INT64）

接口约定（与 diffusers 0.41 的 pipeline_onnx_stable_diffusion.py 一致）：
    text_encoder:  input_ids (int32, [1, 77])                 -> last_hidden_state
    unet:          sample, timestep, encoder_hidden_states    -> out_sample
    vae_decoder:   latent_sample                              -> sample
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HIDDEN = 8  # 微型文本编码器的隐藏维度（真实 SD 1.5 为 768）
SEQ = 77
ROOT = os.path.join(os.path.dirname(__file__), '..', '..', 'test', 'fixtures')
IR_VERSION = 9  # 与 onnxruntime-web 1.30 兼容
OPSET = 17


def const(name, arr):
    return numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)


def text_encoder(half):
    nodes = []
    inits = [const('w_text', np.linspace(0.01, 0.08, HIDDEN).reshape(1, 1, HIDDEN))]
    nodes.append(helper.make_node('Cast', ['input_ids'], ['ids_f'], to=TensorProto.FLOAT))
    nodes.append(helper.make_node('Unsqueeze', ['ids_f', 'ax2'], ['ids_u']))
    nodes.append(helper.make_node('Mul', ['ids_u', 'w_text'], ['mul']))
    nodes.append(helper.make_node('Tanh', ['mul'], ['hidden']))
    out = 'last_hidden_state'
    if half:
        nodes.append(helper.make_node('Cast', ['hidden'], [out], to=TensorProto.FLOAT16))
    else:
        nodes.append(helper.make_node('Identity', ['hidden'], [out]))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), name='ax2'))
    in_type = TensorProto.INT32
    graph = helper.make_graph(
        nodes,
        'text_encoder',
        [helper.make_tensor_value_info('input_ids', in_type, [1, SEQ])],
        [helper.make_tensor_value_info(out, TensorProto.FLOAT16 if half else TensorProto.FLOAT, [1, SEQ, HIDDEN])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)], ir_version=IR_VERSION)


def unet(half):
    nodes = []
    inits = [
        const('w_scale', 0.9),
        const('w_mix', 0.1),
        const('w_t', 0.01),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name='ax_c'),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name='ax_1'),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name='starts'),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name='ends'),
    ]
    # 外部权重：一个 512 元素（2 KB）的张量：超过 1 KB 阈值，因此外部存放；小常量保持内联
    inits.append(numpy_helper.from_array(np.linspace(0.5, 1.0, 512).astype(np.float32), name='w_ext'))
    if half:
        nodes.append(helper.make_node('Cast', ['sample'], ['s'], to=TensorProto.FLOAT))
    s = 's' if half else 'sample'
    if half:
        nodes.append(helper.make_node('Cast', ['encoder_hidden_states'], ['enc_f'], to=TensorProto.FLOAT))
        nodes.append(helper.make_node('Cast', ['timestep'], ['t_f'], to=TensorProto.FLOAT))
        enc = 'enc_f'
        t = 't_f'
    else:
        enc = 'encoder_hidden_states'
        t = 'timestep'
    nodes += [
        helper.make_node('ReduceMean', [enc], ['m'], axes=[1, 2], keepdims=1),
        helper.make_node('Unsqueeze', ['m', 'ax_1'], ['m4']),
        helper.make_node('Mul', [s, 'w_scale'], ['s_scaled']),
        helper.make_node('Mul', ['m4', 'w_mix'], ['m_mixed']),
        helper.make_node('Add', ['s_scaled', 'm_mixed'], ['a1']),
        helper.make_node('Mul', [t, 'w_t'], ['t_scaled']),
        helper.make_node('Add', ['a1', 't_scaled'], ['a2']),
        helper.make_node('Slice', ['w_ext', 'starts', 'ends', 'ax_c'], ['ext0']),
        helper.make_node('Mul', ['ext0', 'w_t'], ['ext_scaled']),
        helper.make_node('Add', ['a2', 'ext_scaled'], ['a3']),
    ]
    if half:
        nodes.append(helper.make_node('Cast', ['a3'], ['out_sample'], to=TensorProto.FLOAT16))
        out_type = TensorProto.FLOAT16
        in_sample = TensorProto.FLOAT16
        in_enc = TensorProto.FLOAT16
        in_t = TensorProto.INT64
    else:
        nodes.append(helper.make_node('Identity', ['a3'], ['out_sample']))
        out_type = TensorProto.FLOAT
        in_sample = TensorProto.FLOAT
        in_enc = TensorProto.FLOAT
        in_t = TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        'unet',
        [
            helper.make_tensor_value_info('sample', in_sample, ['batch', 4, 'h', 'w']),
            helper.make_tensor_value_info('timestep', in_t, [1]),
            helper.make_tensor_value_info('encoder_hidden_states', in_enc, ['batch', SEQ, HIDDEN]),
        ],
        [helper.make_tensor_value_info('out_sample', out_type, ['batch', 4, 'h', 'w'])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)], ir_version=IR_VERSION)


def vae_decoder(half):
    nodes = []
    inits = [
        const('scales', [1, 1, 8, 8]),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name='s0'),
        numpy_helper.from_array(np.array([3], dtype=np.int64), name='s3'),
        numpy_helper.from_array(np.array([1], dtype=np.int64), name='ax1'),
    ]
    src = 'latent_sample'
    if half:
        nodes.append(helper.make_node('Cast', ['latent_sample'], ['lat_f'], to=TensorProto.FLOAT))
        src = 'lat_f'
    nodes += [
        helper.make_node('Resize', [src, '', 'scales'], ['up'], mode='nearest'),
        helper.make_node('Slice', ['up', 's0', 's3', 'ax1'], ['rgb']),
        helper.make_node('Tanh', ['rgb'], ['img']),
    ]
    if half:
        nodes.append(helper.make_node('Cast', ['img'], ['sample'], to=TensorProto.FLOAT16))
    else:
        nodes.append(helper.make_node('Identity', ['img'], ['sample']))
    dtype = TensorProto.FLOAT16 if half else TensorProto.FLOAT
    graph = helper.make_graph(
        nodes,
        'vae_decoder',
        [helper.make_tensor_value_info('latent_sample', dtype, ['b', 4, 'h', 'w'])],
        [helper.make_tensor_value_info('sample', dtype, ['b', 3, 'H', 'W'])],
        initializer=inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', OPSET)], ir_version=IR_VERSION)


def save(model, path, external=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if external:
        # 与真实模型一致：权重放入同目录的 weights.pb，图文件只保留结构
        onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location='weights.pb')
    else:
        onnx.save_model(model, path)
    print('wrote', os.path.relpath(path))


def build(folder, half):
    base = os.path.join(ROOT, folder)
    save(text_encoder(half), os.path.join(base, 'text_encoder', 'model.onnx'))
    save(unet(half), os.path.join(base, 'unet', 'model.onnx'), external=True)
    save(vae_decoder(half), os.path.join(base, 'vae_decoder', 'model.onnx'))


if __name__ == '__main__':
    build('sd15-tiny', half=False)
    build('sd15-tiny-fp16', half=True)
