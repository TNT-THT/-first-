#!/bin/sh
# 双击此文件即可启动本机网站 + API（macOS）。
# 优先使用网站包内置的 Node 运行时（runtime/），无需另外安装；缺失时回退到系统 Node.js。
cd "$(dirname "$0")" || exit 1
export OJ_LOCAL_PREBUILT=1
ARCH=$(uname -m)
BUNDLED=""
case "$ARCH" in
  arm64) BUNDLED="runtime/node-darwin-arm64" ;;
  x86_64) BUNDLED="runtime/node-darwin-x64" ;;
esac

run_api() {
  echo "正在启动本机网站与 API，浏览器会自动打开；使用期间请保持本窗口打开…"
  "$@"
  return $?
}

if [ -n "$BUNDLED" ] && [ -f "$BUNDLED" ]; then
  chmod +x "$BUNDLED" 2>/dev/null
  run_api "$BUNDLED" scripts/start-local-api.mjs
elif command -v node >/dev/null 2>&1; then
  run_api node scripts/start-local-api.mjs
else
  echo "还差一步：需要先安装 Node.js（免费）。"
  echo "正在打开下载页面 https://nodejs.org/zh-cn/download ，下载 LTS 版安装即可。"
  open "https://nodejs.org/zh-cn/download" 2>/dev/null
  echo "安装完成后，重新双击本文件。"
  echo "按回车关闭。"
  read -r _
  exit 1
fi
status=$?
if [ "$status" -ne 0 ]; then
  echo "按回车关闭。"
  read -r _
fi
exit "$status"
