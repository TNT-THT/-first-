#!/usr/bin/env python3
"""本地 Python 模型运行时（软件版特殊格式的本地运行路线）。

用 Transformers / ONNX Runtime 直接加载 Hugging Face 原格式权重：
- Safetensors / PyTorch bin（含 config.json 里的 AWQ/GPTQ/BNB 量化配置）：
  AutoModelForCausalLM 加载（量化由 transformers 按仓库配置处理）；
- ONNX：optimum.onnxruntime 的 ORTModelForCausalLM；
- MLX（仅 macOS）：mlx-lm。

提供与 llama-server 相同的最小 OpenAI 兼容接口（/health、/v1/chat/completions，
含流式），由 Node 引擎像管理 llama-server 一样管理本进程。
只监听 127.0.0.1，带 --api-key 校验。生成速度取决于硬件，CPU 上大模型会慢，
但能保证“下载到本地就能跑”，不依赖任何云端。
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def log(*a):
    print("[python-runner]", *a, flush=True)


def load_model(model_dir):
    import torch  # noqa: F401  (确保 torch 先就绪，报错更明确)

    files = set(os.listdir(model_dir))
    has_onnx = any(f.endswith(".onnx") for f in files)
    has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)

    if has_onnx and not has_weights:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer

        log("loading ONNX model via ONNX Runtime ...")
        model = ORTModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("loading Transformers model (first load may take a while) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


STATE = {"model": None, "tokenizer": None, "lock": threading.Lock()}


def build_inputs(messages):
    tokenizer = STATE["tokenizer"]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        text = "\n".join(parts) + "\nassistant:"
    return tokenizer(text, return_tensors="pt")


def gen_kwargs(body, max_new):
    kwargs = {"max_new_tokens": max_new, "do_sample": False}
    temp = body.get("temperature")
    if isinstance(temp, (int, float)) and temp > 0:
        kwargs.update({"do_sample": True, "temperature": float(temp)})
        top_p = body.get("top_p")
        if isinstance(top_p, (int, float)):
            kwargs["top_p"] = float(top_p)
    return kwargs


def make_handler(api_key):
    class Handler(BaseHTTPRequestHandler):
        def _auth_ok(self):
            return self.headers.get("Authorization", "") == f"Bearer {api_key}"

        def _send_json(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                if not self._auth_ok():
                    return self._send_json(401, {"error": "unauthorized"})
                ready = STATE["model"] is not None
                return self._send_json(200 if ready else 503, {"status": "ok" if ready else "loading"})
            return self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._send_json(404, {"error": "not found"})
            if not self._auth_ok():
                return self._send_json(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception as e:  # noqa: BLE001
                return self._send_json(400, {"error": f"bad json: {e}"})
            messages = body.get("messages") or []
            max_new = int(body.get("max_tokens") or 512)
            stream = bool(body.get("stream"))
            model = STATE["model"]
            tokenizer = STATE["tokenizer"]
            if model is None:
                return self._send_json(503, {"error": "model not loaded"})

            def prompt_tokens(inputs):
                try:
                    return int(inputs["input_ids"].shape[-1])
                except Exception:  # noqa: BLE001
                    return 0

            with STATE["lock"]:
                inputs = build_inputs(messages)
                n_prompt = prompt_tokens(inputs)
                device = getattr(model, "device", None)
                if device is not None:
                    inputs = inputs.to(device)
                if stream:
                    return self._stream(body, inputs, n_prompt, max_new)
                import torch

                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs(body, max_new))
                new_tokens = out[0][inputs["input_ids"].shape[-1]:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                return self._send_json(200, {
                    "id": "chatcmpl-python",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model") or "python-local",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": n_prompt,
                        "completion_tokens": int(new_tokens.shape[-1]),
                        "total_tokens": n_prompt + int(new_tokens.shape[-1]),
                    },
                })

        def _stream(self, body, inputs, n_prompt, max_new):
            from transformers import TextIteratorStreamer

            tokenizer = STATE["tokenizer"]
            model = STATE["model"]
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            kwargs = gen_kwargs(body, max_new)
            kwargs["streamer"] = streamer
            thread = threading.Thread(target=lambda: model.generate(**inputs, **kwargs), daemon=True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(obj):
                self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()

            base = {"id": "chatcmpl-python", "object": "chat.completion.chunk", "created": 0,
                    "model": body.get("model") or "python-local"}
            emit({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            count = 0
            thread.start()
            try:
                for piece in streamer:
                    if piece:
                        count += 1
                        emit({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            finally:
                thread.join(timeout=5)
            emit({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": n_prompt, "completion_tokens": count, "total_tokens": n_prompt + count}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *args):  # 静音默认访问日志
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--api-key", required=True)
    args = parser.parse_args()

    # 先起服务再加载模型：/health 在加载期间返回 503，Node 侧会继续等待。
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.api_key))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log(f"serving on 127.0.0.1:{args.port}, loading {args.model_dir} ...")
    try:
        model, tokenizer = load_model(args.model_dir)
        STATE["model"] = model
        STATE["tokenizer"] = tokenizer
        log("model ready")
    except Exception as e:  # noqa: BLE001
        log(f"model load failed: {type(e).__name__}: {e}")
        # 保持进程存活一小段时间让 Node 读到日志后由其判定失败
        import time

        time.sleep(2)
        sys.exit(1)
    try:
        thread.join()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
