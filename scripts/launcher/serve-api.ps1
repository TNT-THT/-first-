# 一键启动本机网站 + API（Windows）。双击 start-windows.bat 会调用本脚本。
# 优先使用网站包内置的 Node 运行时（runtime\node-win-x64.exe），无需另外安装任何东西；
# 内置运行时缺失时回退到系统安装的 Node.js；都没有时才打开 Node 下载页面。
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:OJ_LOCAL_PREBUILT = '1'

$bundledNode = Join-Path $PSScriptRoot 'runtime\node-win-x64.exe'
$nodeCmd = $null
if (Test-Path -LiteralPath $bundledNode) {
  $nodeCmd = $bundledNode
} elseif (Get-Command node -ErrorAction SilentlyContinue) {
  $nodeCmd = 'node'
}

if (-not $nodeCmd) {
  Write-Host ''
  Write-Host '还差一步：需要先安装 Node.js（免费）。'
  Write-Host '正在为你打开下载页面：下载 LTS 版，一路点“下一步”安装即可。'
  Write-Host '安装完成后，关闭本窗口，重新双击 start-windows.bat。'
  Write-Host ''
  Start-Process 'https://nodejs.org/zh-cn/download'
  Read-Host '按回车关闭'
  exit 1
}

Write-Host ''
Write-Host '正在启动本机网站与 API，浏览器会自动打开，请稍候…'
Write-Host '使用期间请保持本窗口打开。'
Write-Host ''
& $nodeCmd 'scripts\start-local-api.mjs'
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Host ''
  Write-Host "启动失败（退出码 $code）。请把本窗口显示的报错反馈给网站提供者。"
  Read-Host '按回车关闭'
  exit $code
}
