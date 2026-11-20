@echo off
chcp 65001 >nul
rem Double-click: start the local site + API (needs Node.js, checked inside).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve-api.ps1"
if errorlevel 1 pause
