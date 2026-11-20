@echo off
chcp 65001 >nul
rem Fallback: static site only (no local API). Uses built-in PowerShell.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve-site.ps1"
if errorlevel 1 pause
