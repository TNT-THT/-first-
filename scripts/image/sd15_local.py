#!/usr/bin/env python3
"""在本机运行 Stable Diffusion 1.5（ONNX），算法与网页版逐项一致。

网页版（web/src/image/）使用的 CLIP 分词器、PNDM 调度器、随机噪声（mulberry32 + Box-Muller）、
无分类器引导与解码公式，在这里逐项移植。已用网页版同一套 JavaScript 代码交叉检查：
分词结果（含中文、emoji、超长文本）、初始噪声、时间步、PNDM 每一步的输出都一致
（见 scripts/image/test_sd15_local.py）。完整图像的逐像素一致性尚未在真实模型上验证。

依赖（pip install）：
    numpy onnxruntime regex
GPU（可选）：NVIDIA 使用 onnxruntime-gpu；Windows 上的 DirectML 使用 onnxruntime-directml。

模型目录与网页版“下载到本机”的文件结构相同（即 Hugging Face 仓库的目录结构）。
建议整套使用同一个仓库（如 nmkd/stable-diffusion-1.5-onnx 的 fp32 版），避免 fp16 与 fp32 的
text_encoder/model.onnx 混放在同一目录而互相覆盖：
    <模型目录>/
        text_encoder/model.onnx
        unet/model.onnx  unet/weights.pb
        vae_decoder/model.onnx
        tokenizer/vocab.json  tokenizer/merges.txt
        scheduler/scheduler_config.json

用法示例：
    python3 scripts/image/sd15_local.py --models ./sd15 --prompt "a red fox in the snow" \
        --width 512 --height 512 --steps 20 --guidance 7.5 --seed 42 --out fox.png
"""
import argparse
import json
import math
import os
import struct
import sys
import time
import zlib

import numpy as np

MASK32 = 0xFFFFFFFF
LATENT_SCALE = 0.18215
MAX_STEPS = 50
CONTEXT_LENGTH = 77
SIZE_MIN, SIZE_MAX, SIZE_STEP = 256, 768, 8


# ---------- 随机数（与 web/src/image/rng.js 相同） ----------

def _imul(a, b):
    return ((a & MASK32) * (b & MASK32)) & MASK32


def mulberry32(seed):
    a = seed & MASK32

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & MASK32
        t = a
        t = _imul(t ^ (t >> 15), t | 1)
        t = (t ^ ((t + _imul(t ^ (t >> 7), t | 61)) & MASK32)) & MASK32
        return ((t ^ (t >> 14)) & MASK32) / 4294967296

    return nxt


def gaussian_noise(seed, n):
    """长度为 n 的标准正态样本（float32）。"""
    uniform = mulberry32(seed)
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, 2):
        u1 = max(uniform(), 1e-12)
        u2 = uniform()
        r = math.sqrt(-2 * math.log(u1))
        out[i] = r * math.cos(2 * math.pi * u2)
        if i + 1 < n:
            out[i + 1] = r * math.sin(2 * math.pi * u2)
    return out


# ---------- CLIP 分词器（与 web/src/image/clip-tokenizer.js 相同） ----------

def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_MAP = _bytes_to_unicode()


class ClipTokenizer:
    def __init__(self, vocab, merges_text, max_length=CONTEXT_LENGTH):
        import regex  # 需要 \p{L} 等 Unicode 类别

        self.pattern = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|\p{N}|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )
        self.encoder = dict(vocab)
        self.ranks = {}
        rank = 0
        for line in merges_text.split("\n"):
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            self.ranks[t] = rank
            rank += 1
        self.cache = {}
        self.bos = self.encoder["<|startoftext|>"]
        self.eos = self.encoder["<|endoftext|>"]
        self.pad = self.eos
        self.unk = self.eos
        self.max_length = max_length

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
        if not pairs:
            out = ["".join(word)]
            self.cache[token] = out
            return out
        while True:
            best, best_rank = None, float("inf")
            for a, b in pairs:
                r = self.ranks.get(f"{a} {b}")
                if r is not None and r < best_rank:
                    best_rank, best = r, (a, b)
            if best is None:
                break
            first, second = best
            nxt, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    nxt.append(first + second)
                    i += 2
                else:
                    nxt.append(word[i])
                    i += 1
            word = nxt
            if len(word) == 1:
                break
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
        self.cache[token] = word
        return word

    def tokenize(self, text):
        import re

        clean = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        ids = [self.bos]
        for m in self.pattern.finditer(clean):
            s = m.group(0)
            if s == "<|startoftext|>":
                ids.append(self.bos)
                continue
            if s == "<|endoftext|>":
                ids.append(self.eos)
                continue
            mapped = "".join(BYTE_MAP[b] for b in s.encode("utf-8"))
            for piece in self._bpe(mapped):
                ids.append(self.encoder.get(piece, self.unk))
        ids.append(self.eos)
        if len(ids) > self.max_length:
            return ids[: self.max_length - 1] + [self.eos]
        return ids

    def encode(self, text):
        ids = self.tokenize(text)
        out = np.full(self.max_length, self.pad, dtype=np.int32)
        out[: len(ids)] = ids
        return out


# ---------- PNDM 调度器（与 web/src/image/pndm.js 相同，SD 1.5 配置） ----------

class PndmScheduler:
    def __init__(self, config):
        c = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "skip_prk_steps": False,
            "set_alpha_to_one": False,
            "prediction_type": "epsilon",
            "timestep_spacing": "leading",
            "steps_offset": 0,
        }
        c.update(config)
        if not c["skip_prk_steps"]:
            raise ValueError("PNDM：本实现只支持 skip_prk_steps = true（SD 1.5 的配置）")
        if c["prediction_type"] != "epsilon":
            raise ValueError(f"PNDM：不支持 prediction_type={c['prediction_type']}")
        if c["timestep_spacing"] != "leading":
            raise ValueError(f"PNDM：不支持 timestep_spacing={c['timestep_spacing']}")
        self.c = c
        n = c["num_train_timesteps"]
        self.n_train = n
        if c["beta_schedule"] == "scaled_linear":
            a, b = math.sqrt(c["beta_start"]), math.sqrt(c["beta_end"])
            betas = np.array([(a + (b - a) * i / (n - 1)) ** 2 for i in range(n)], dtype=np.float64)
        elif c["beta_schedule"] == "linear":
            betas = np.array(
                [c["beta_start"] + (c["beta_end"] - c["beta_start"]) * i / (n - 1) for i in range(n)],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"PNDM：不支持 beta_schedule={c['beta_schedule']}")
        acp = np.empty(n, dtype=np.float64)
        acc = 1.0
        for i in range(n):
            acc *= 1 - betas[i]
            acp[i] = acc
        self.alphas_cumprod = acp
        self.final_alpha = 1.0 if c["set_alpha_to_one"] else acp[0]
        self.init_noise_sigma = 1.0
        self.timesteps = []
        self.num_inference = None
        self.ets = []
        self.counter = 0
        self.cur_sample = None

    def set_timesteps(self, n):
        if not isinstance(n, int) or n < 2:
            raise ValueError("推理步数必须是不小于 2 的整数")
        self.num_inference = n
        ratio = self.n_train // n
        base = [round(i * ratio) + self.c["steps_offset"] for i in range(n)]
        self.timesteps = list(reversed(base[:-1] + [base[-2], base[-1]]))
        self.ets = []
        self.counter = 0
        self.cur_sample = None
        return self.timesteps

    def _prev_sample(self, sample, t, prev_t, model_output):
        a_t = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha
        b_t = 1 - a_t
        b_prev = 1 - a_prev
        sample_coeff = math.sqrt(a_prev / a_t)
        denom = a_t * math.sqrt(b_prev) + math.sqrt(a_t * b_t * a_prev)
        k = (a_prev - a_t) / denom
        return sample_coeff * sample - k * model_output

    def step(self, model_output, timestep, sample):
        if self.num_inference is None:
            raise ValueError("请先调用 set_timesteps")
        ratio = self.n_train // self.num_inference
        prev = timestep - ratio
        t = timestep
        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)
        else:
            prev = timestep
            t = timestep + ratio
        s = sample
        if len(self.ets) == 1 and self.counter == 0:
            mo = model_output
            self.cur_sample = sample
        elif len(self.ets) == 1 and self.counter == 1:
            mo = (model_output + self.ets[0]) / 2
            s = self.cur_sample
            self.cur_sample = None
        elif len(self.ets) == 2:
            mo = (3 * self.ets[1] - self.ets[0]) / 2
        elif len(self.ets) == 3:
            mo = (23 * self.ets[2] - 16 * self.ets[1] + 5 * self.ets[0]) / 12
        else:
            mo = (55 * self.ets[3] - 59 * self.ets[2] + 37 * self.ets[1] - 9 * self.ets[0]) / 24
        out = self._prev_sample(s, t, prev, mo)
        self.counter += 1
        return out


# ---------- 尺寸与 PNG ----------

def check_size(width, height):
    for name, v in (("宽", width), ("高", height)):
        if not isinstance(v, int) or v < SIZE_MIN or v > SIZE_MAX or v % SIZE_STEP:
            raise ValueError(f"{name}必须是 {SIZE_MIN}–{SIZE_MAX} 之间的 {SIZE_STEP} 的倍数，当前为 {v}")


def write_png(path, rgb):
    """rgb: uint8，形状 [H, W, 3]。只用标准库写 PNG。"""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & MASK32)

    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)


def chw_to_rgb(chw, width, height):
    """[3, H, W] 的 [-1, 1] → uint8 [H, W, 3]（与网页版 chwToRgba 相同的 clip(x/2+0.5)）。"""
    x = np.clip(chw.reshape(3, height, width) / 2 + 0.5, 0, 1)
    return np.floor(x * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0)  # 与 Math.round 相同（半数向上）


# ---------- ONNX 会话 ----------

_NP_TYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
}


def _session(ort, path, providers):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so, providers=providers)


def _input_type(session, name):
    for meta in session.get_inputs():
        if meta.name == name:
            return _NP_TYPES.get(meta.type, np.float32)
    raise KeyError(name)


def load_pipeline(models_dir, providers, ort):
    def p(*parts):
        return os.path.join(models_dir, *parts)

    with open(p("tokenizer", "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(p("tokenizer", "merges.txt"), encoding="utf-8") as f:
        merges = f.read()
    with open(p("scheduler", "scheduler_config.json"), encoding="utf-8") as f:
        sched_cfg = json.load(f)
    return {
        "tokenizer": ClipTokenizer(vocab, merges),
        "scheduler_config": sched_cfg,
        "text_encoder": _session(ort, p("text_encoder", "model.onnx"), providers),
        "unet": _session(ort, p("unet", "model.onnx"), providers),
        "vae": _session(ort, p("vae_decoder", "model.onnx"), providers),
    }


def generate(pipe, prompt, *, negative="", width=512, height=512, steps=20, guidance=7.5, seed=0,
             progress=None):
    """返回 [3, H, W] 的 float32（约在 [-1, 1]），与网页版 generateSd15 相同的步骤。"""
    check_size(width, height)
    if not isinstance(steps, int) or steps < 2 or steps > MAX_STEPS:
        raise ValueError(f"步数必须是 2–{MAX_STEPS} 之间的整数")
    if guidance < 0:
        raise ValueError("引导系数必须是非负数字")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("种子必须是非负整数")
    h8, w8 = height // 8, width // 8
    use_cfg = guidance > 1
    tok = pipe["tokenizer"]
    te = pipe["text_encoder"]
    te_in = te.get_inputs()[0].name
    te_out = te.get_outputs()[0].name

    def encode(text):
        ids = tok.encode(text).reshape(1, CONTEXT_LENGTH)
        return te.run([te_out], {te_in: ids.astype(_input_type(te, te_in))})[0].astype(np.float32)

    cond = encode(prompt)
    context = cond
    if use_cfg:
        context = np.concatenate([encode(negative), cond], axis=0)  # 无条件在前，条件在后

    sched = PndmScheduler(pipe["scheduler_config"])
    latents = gaussian_noise(seed, 4 * h8 * w8).astype(np.float64) * sched.init_noise_sigma
    latents = latents.reshape(1, 4, h8, w8)
    timesteps = sched.set_timesteps(steps)

    unet = pipe["unet"]
    names = [m.name for m in unet.get_inputs()]
    unet_out = unet.get_outputs()[0].name
    sample_t = _input_type(unet, "sample")
    ts_t = _input_type(unet, "timestep")
    ctx_t = _input_type(unet, "encoder_hidden_states")
    assert "sample" in names and "timestep" in names and "encoder_hidden_states" in names

    for i, t in enumerate(timesteps):
        sample = np.concatenate([latents, latents], axis=0) if use_cfg else latents
        feeds = {
            "sample": sample.astype(sample_t),
            "timestep": np.array([t], dtype=ts_t),
            "encoder_hidden_states": context.astype(ctx_t),
        }
        noise = unet.run([unet_out], feeds)[0].astype(np.float64)
        if use_cfg:
            half = noise.shape[0] // 2
            u, c = noise[:half], noise[half:]
            guided = u + guidance * (c - u)
        else:
            guided = noise
        latents = sched.step(guided, t, latents.astype(np.float64))
        if progress:
            progress(i + 1, len(timesteps))

    vae = pipe["vae"]
    vae_in = vae.get_inputs()[0].name
    vae_out = vae.get_outputs()[0].name
    scaled = (latents / LATENT_SCALE).astype(_input_type(vae, vae_in))
    chw = vae.run([vae_out], {vae_in: scaled})[0].astype(np.float32)
    if chw.size != 3 * height * width:
        raise RuntimeError(f"解码结果尺寸不正确：{chw.size}")
    return chw.reshape(3, height, width)


def main(argv=None):
    ap = argparse.ArgumentParser(description="在本机运行 Stable Diffusion 1.5（ONNX）")
    ap.add_argument("--models", required=True, help="模型目录（与网页版下载到本机的目录结构相同）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="sd15-output.png")
    ap.add_argument("--provider", choices=["cpu", "cuda", "dml"], default="cpu",
                    help="cpu：任何电脑；cuda：NVIDIA（需要 onnxruntime-gpu）；dml：Windows 的 DirectML")
    args = ap.parse_args(argv)

    import onnxruntime as ort

    available = ort.get_available_providers()
    wanted = {"cpu": ["CPUExecutionProvider"],
              "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
              "dml": ["DmlExecutionProvider", "CPUExecutionProvider"]}[args.provider]
    providers = [p for p in wanted if p in available]
    if not providers or providers[0] == "CPUExecutionProvider" and args.provider != "cpu":
        print(f"提示：当前 ONNX Runtime 不支持 {args.provider}，改用 CPU（可用：{available}）", file=sys.stderr)
    providers = providers or ["CPUExecutionProvider"]

    t0 = time.time()
    pipe = load_pipeline(args.models, providers, ort)
    print(f"模型已加载（{time.time() - t0:.1f} 秒，执行方式：{providers[0]}）", file=sys.stderr)
    t1 = time.time()
    chw = generate(
        pipe, args.prompt, negative=args.negative, width=args.width, height=args.height,
        steps=args.steps, guidance=args.guidance, seed=args.seed,
        progress=lambda s, n: print(f"\r去噪 {s}/{n}", end="", file=sys.stderr),
    )
    print(file=sys.stderr)
    write_png(args.out, chw_to_rgb(chw, args.width, args.height))
    print(f"已保存：{args.out}（生成用时 {time.time() - t1:.1f} 秒）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
