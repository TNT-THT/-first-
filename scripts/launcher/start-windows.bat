@echo off
chcp 65001 >nul
rem Double-click this file to open the site locally (uses built-in PowerShell).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve-site.ps1"
if errorlevel 1 pause
