"""生成 test/fixtures/onnx-pow/ln_decomposed_fp16.onnx（需要 onnx 包：pip install onnx）。

模拟 SD 1.5 文本编码器中被分解的 LayerNorm（ReduceMean → Sub → Pow → ReduceMean → Add → Sqrt → Div）：
- layer0 的指数 2 是 fp16 初始化器；
- layer1 的指数 2 来自 Constant 节点（fp32 张量）；
- 最后的 Pow 指数为 3，必须保持不变。
"""
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto as T

DT = np.float16


def layer_norm(prefix, src, dst, exponent_node_inputs, nodes, inits):
    """exponent_node_inputs：指数输入名（初始化器名或 Constant 输出名已经在外部创建）。"""
    nodes += [
        helper.make_node('ReduceMean', [src], [f'{prefix}_mean'], name=f'{prefix}/ReduceMean', axes=[-1], keepdims=1),
        helper.make_node('Sub', [src, f'{prefix}_mean'], [f'{prefix}_c'], name=f'{prefix}/Sub'),
        helper.make_node('Pow', [f'{prefix}_c', exponent_node_inputs], [f'{prefix}_sq'], name=f'{prefix}/Pow'),
        helper.make_node('ReduceMean', [f'{prefix}_sq'], [f'{prefix}_var'], name=f'{prefix}/ReduceMean_1', axes=[-1], keepdims=1),
        helper.make_node('Add', [f'{prefix}_var', f'{prefix}_eps'], [f'{prefix}_ve'], name=f'{prefix}/Add'),
        helper.make_node('Sqrt', [f'{prefix}_ve'], [f'{prefix}_std'], name=f'{prefix}/Sqrt'),
        helper.make_node('Div', [f'{prefix}_c', f'{prefix}_std'], [dst], name=f'{prefix}/Div'),
    ]


def build(path):
    nodes, inits = [], []
    # layer0：指数是初始化器
    inits.append(numpy_helper.from_array(np.array(2.0, dtype=DT), 'layer0_exp'))
    inits.append(numpy_helper.from_array(np.array(1e-5, dtype=DT), 'layer0_eps'))
    layer_norm('layer0', 'x', 'h0', 'layer0_exp', nodes, inits)
    # layer1：指数由 Constant 节点提供（fp32 张量）
    nodes.append(helper.make_node('Constant', [], ['layer1_exp'], name='layer1_exp_const',
                                  value=numpy_helper.from_array(np.array(2.0, dtype=np.float32), 'layer1_exp')))
    inits.append(numpy_helper.from_array(np.array(1e-5, dtype=DT), 'layer1_eps'))
    layer_norm('layer1', 'h0', 'h1', 'layer1_exp', nodes, inits)
    # 指数为 3 的 Pow：必须保持
    inits.append(numpy_helper.from_array(np.array(3.0, dtype=DT), 'pow3_exp'))
    nodes.append(helper.make_node('Pow', ['h1', 'pow3_exp'], ['h2'], name='pow3/Pow'))
    nodes.append(helper.make_node('Identity', ['h2'], ['y'], name='out/Identity'))
    x = helper.make_tensor_value_info('x', T.FLOAT16, [1, 4, 8])
    y = helper.make_tensor_value_info('y', T.FLOAT16, [1, 4, 8])
    g = helper.make_graph(nodes, 'ln_decomposed', [x], [y], initializer=inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)




def build_encodings(outdir):
    """指数为 2 的各种存放方式（每个文件只有一个 Pow(x, e) 节点，e 为 2）。"""
    def one(name, exp_nodes, exp_name, inits=()):
        x = helper.make_tensor_value_info('x', T.FLOAT16, [1, 4, 8])
        y = helper.make_tensor_value_info('y', T.FLOAT16, [1, 4, 8])
        nodes = list(exp_nodes) + [helper.make_node('Pow', ['x', exp_name], ['y'], name='Pow_under_test')]
        g = helper.make_graph(nodes, name, [x], [y], initializer=list(inits))
        m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 14)])
        m.ir_version = 8
        onnx.save(m, f'{outdir}/{name}.onnx')

    # 初始化器：make_tensor 生成的 fp16（int32_data）与 fp32（float_data，打包）
    one('enc_fp16_int32data', [], 'e', [helper.make_tensor('e', T.FLOAT16, [], [2.0])])
    one('enc_fp32_floatdata', [], 'e', [helper.make_tensor('e', T.FLOAT, [], [2.0])])
    # Constant 节点：value_float 属性；value 张量（int32_data，形状 [1]）
    one('enc_const_valuefloat', [helper.make_node('Constant', [], ['e'], name='c', value_float=2.0)], 'e')
    one('enc_const_tensor_int32data',
        [helper.make_node('Constant', [], ['e'], name='c', value=helper.make_tensor('v', T.FLOAT16, [1], [2.0]))], 'e')
    # 指数来自图输入（运行时才知道）：必须保持不变
    x = helper.make_tensor_value_info('x', T.FLOAT16, [1, 4, 8])
    e = helper.make_tensor_value_info('e', T.FLOAT16, [])
    y = helper.make_tensor_value_info('y', T.FLOAT16, [1, 4, 8])
    g = helper.make_graph([helper.make_node('Pow', ['x', 'e'], ['y'], name='Pow_dynamic')], 'enc_dynamic_exp', [x, e], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 14)])
    m.ir_version = 8
    onnx.save(m, f'{outdir}/enc_dynamic_exp.onnx')
    # Identity 与 Cast 链
    one('enc_identity_chain', [helper.make_node('Identity', ['e0'], ['e1'], name='id'),
                               helper.make_node('Identity', ['e1'], ['e'], name='id2')], 'e',
        [numpy_helper.from_array(np.array(2.0, dtype=np.float16), 'e0')])
    one('enc_cast_chain', [helper.make_node('Cast', ['e0'], ['e'], name='cast', to=T.FLOAT16)], 'e',
        [numpy_helper.from_array(np.array(2.0, dtype=np.float32), 'e0')])


if __name__ == '__main__':
    build('test/fixtures/onnx-pow/ln_decomposed_fp16.onnx')
    build_encodings('test/fixtures/onnx-pow')
    print('ok')
