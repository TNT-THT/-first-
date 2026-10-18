#!/usr/bin/env python3
"""本地启动网站（macOS / Linux，需要 Python 3）。

只监听 127.0.0.1（仅本机可访问），找到空闲端口（8080–8099）后自动打开浏览器。
按 Ctrl+C 或关闭终端窗口即停止服务。
"""
import http.server
import os
import threading
import webbrowser

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dist")

TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".wasm": "application/wasm",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
    ".zip": "application/zip",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **TYPES}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def list_directory(self, path):  # 不列出目录内容
        self.send_error(404, "Not Found")
        return None

    def log_message(self, format, *args):  # 不打印每次请求
        pass


def main():
    if not os.path.isfile(os.path.join(ROOT, "index.html")):
        print("找不到 web/dist/index.html。请先把网站包完整解压，再运行本程序。")
        return 1
    server = None
    port = 0
    for p in range(8080, 8100):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if server is None:
        print("8080–8099 端口都被占用，请关闭其他程序后重试。")
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"网站已启动：{url}")
    print("请保持本窗口打开；按 Ctrl+C 停止。")
    threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
