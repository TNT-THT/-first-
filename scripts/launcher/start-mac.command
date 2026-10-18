#!/bin/sh
# 双击此文件即可在本机打开网站（macOS，需要 Python 3）。
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "需要 Python 3。可在终端运行 xcode-select --install 安装命令行工具后重试。"
  echo "按回车关闭。"
  read -r _
  exit 1
fi
python3 serve-site.py
status=$?
if [ "$status" -ne 0 ]; then
  echo "按回车关闭。"
  read -r _
fi
exit "$status"
