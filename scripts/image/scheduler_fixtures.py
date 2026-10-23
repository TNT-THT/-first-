#!/usr/bin/env python3
"""
生成调度器对照数据：用 diffusers 的调度器逐步计算，把结果写入 test/fixtures/image-schedulers.json。
网页端的 DPM-Solver++、DEIS、Euler 调度器（web/src/image/schedulers.js）与之逐项比较（见 test/image-schedulers.test.js）。

用法（需要 torch 与 diffusers==0.41.0）：
    python3 scripts/image/scheduler_fixtures.py

配置取自各模型仓库的 scheduler/scheduler_config.json（见 config.json 中的注释）。
输入是固定的随机数（种子 20261023），模型输出不依赖样本，因此两边的计算可以逐步对照。
"""
import inspect
import json
import pathlib

import numpy as np
import torch
from diffusers import DEISMultistepScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / 'test' / 'fixtures' / 'image-schedulers.json'
DIM = 12

# 与各仓库 scheduler_config.json 相同（去掉了 _class_name 等元数据键，diffusers 会忽略它们）
CONFIGS = {
    'dpmpp2m-linspace': (DPMSolverMultistepScheduler, {
        'algorithm_type': 'dpmsolver++', 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'beta_start': 0.00085,
        'lower_order_final': True, 'num_train_timesteps': 1000, 'prediction_type': 'epsilon',
        'set_alpha_to_one': False, 'solver_order': 2, 'solver_type': 'midpoint', 'steps_offset': 1, 'thresholding': False,
        'timestep_spacing': 'linspace', 'use_karras_sigmas': False,
    }),
    'deis-linspace': (DEISMultistepScheduler, {
        'algorithm_type': 'deis', 'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'beta_start': 0.00085,
        'lower_order_final': True, 'num_train_timesteps': 1000, 'prediction_type': 'epsilon',
        'sample_max_value': 1.0, 'set_alpha_to_one': False, 'solver_order': 2, 'solver_type': 'logrho', 'steps_offset': 1,
        'thresholding': False, 'timestep_spacing': 'linspace', 'use_karras_sigmas': False,
    }),
    'euler-trailing': (EulerDiscreteScheduler, {
        'beta_end': 0.012, 'beta_schedule': 'scaled_linear', 'beta_start': 0.00085,
        'interpolation_type': 'linear', 'num_train_timesteps': 1000, 'prediction_type': 'epsilon',
        'sample_max_value': 1.0, 'set_alpha_to_one': False, 'steps_offset': 1, 'timestep_spacing': 'trailing',
        'use_karras_sigmas': False,
    }),
}

# (调度器, 推理步数)：覆盖一步、两步、常规步数，以及 lower_order 与多阶分支（DPM 的 16 步不走 lower_order_final）
CASES = [
    ('dpmpp2m-linspace', 1), ('dpmpp2m-linspace', 2), ('dpmpp2m-linspace', 4), ('dpmpp2m-linspace', 16),
    ('deis-linspace', 1), ('deis-linspace', 2), ('deis-linspace', 5),
    ('euler-trailing', 1), ('euler-trailing', 4),
]


def main():
    rng = np.random.default_rng(20261023)
    cases = []
    for name, steps in CASES:
        cls, config = CONFIGS[name]
        # 只保留该调度器构造函数接受的键；被丢弃的键都是 diffusers 的默认值（已在下方打印，便于核对）
        accepted = inspect.signature(cls.__init__).parameters
        dropped = sorted(k for k in config if k not in accepted)
        if dropped:
            print(f'{name}：diffusers 不接受、按默认值处理的键：{", ".join(dropped)}')
        config = {k: v for k, v in config.items() if k in accepted}
        sched = cls(**config)
        sched.set_timesteps(steps)
        timesteps = [int(t) for t in sched.timesteps.tolist()]
        sigmas = [float(s) for s in sched.sigmas.tolist()]
        init_sigma = float(sched.init_noise_sigma) if hasattr(sched, 'init_noise_sigma') else 1.0
        sample0 = rng.standard_normal(DIM)
        # 每一步的模型输出（噪声预测），与样本无关
        outputs = [rng.standard_normal(DIM) for _ in timesteps]

        sample = torch.tensor(sample0, dtype=torch.float64)
        sched.set_timesteps(steps)  # 重新设置，确保计数器从零开始
        steps_out = []
        sample = sample * init_sigma
        for i, t in enumerate(timesteps):
            # Euler 要求传入 scheduler.timesteps 中的元素（张量），不接受下标
            tt = sched.timesteps[i]
            scaled = sched.scale_model_input(sample, tt) if name.startswith('euler') else sample
            out = sched.step(torch.tensor(outputs[i], dtype=torch.float64), tt, sample, return_dict=False)[0]
            steps_out.append({
                'timestep': t,
                'scaledInput': scaled.tolist(),
                'output': outputs[i].tolist(),
                'next': out.tolist(),
            })
            sample = out
        cases.append({
            'scheduler': name,
            'className': cls.__name__,
            'config': config,
            'steps': steps,
            'timesteps': timesteps,
            'sigmas': sigmas,
            'initNoiseSigma': init_sigma,
            'sample0': sample0.tolist(),
            'perStep': steps_out,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({'diffusers': '0.41.0', 'cases': cases}, indent=1) + '\n', encoding='utf-8')
    print(f'已写入 {OUT.relative_to(ROOT)}：{len(cases)} 个用例')


if __name__ == '__main__':
    main()
