@echo off
chcp 65001 >nul
rem Double-click: start the LAN server (bundled Node runtime included).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve-server.ps1"
if errorlevel 1 pause
