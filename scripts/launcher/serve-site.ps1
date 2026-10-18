# 本地启动网站（Windows，使用系统自带的 PowerShell，无需安装任何东西）。
# 只监听 127.0.0.1（仅本机可访问），找到空闲端口后自动打开浏览器。
# 关闭本窗口即停止服务。
$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot 'web\dist'
$rootFull = [System.IO.Path]::GetFullPath($root)

if (-not (Test-Path -LiteralPath (Join-Path $rootFull 'index.html'))) {
  Write-Host '找不到 web\dist\index.html。请先把网站包完整解压，再运行本程序。'
  Read-Host '按回车退出'
  exit 1
}

$mime = @{
  '.html' = 'text/html; charset=utf-8'
  '.js'   = 'text/javascript; charset=utf-8'
  '.mjs'  = 'text/javascript; charset=utf-8'
  '.css'  = 'text/css; charset=utf-8'
  '.json' = 'application/json; charset=utf-8'
  '.map'  = 'application/json; charset=utf-8'
  '.wasm' = 'application/wasm'
  '.svg'  = 'image/svg+xml'
  '.png'  = 'image/png'
  '.jpg'  = 'image/jpeg'
  '.jpeg' = 'image/jpeg'
  '.ico'  = 'image/x-icon'
  '.woff' = 'font/woff'
  '.woff2' = 'font/woff2'
  '.txt'  = 'text/plain; charset=utf-8'
  '.zip'  = 'application/zip'
}

# 找一个空闲端口（8080–8099）
$listener = $null
$port = 0
$lastError = ''
foreach ($p in 8080..8099) {
  $l = New-Object System.Net.HttpListener
  $l.Prefixes.Add("http://127.0.0.1:$p/")
  try {
    $l.Start()
    $listener = $l
    $port = $p
    break
  } catch {
    # 端口被占用或无权限，记录原因后尝试下一个
    $lastError = $_.Exception.Message
  }
}
if (-not $listener) {
  Write-Host '无法在 8080–8099 端口启动本机服务器。请关闭占用端口的程序后重试。'
  Write-Host "最后一次错误：$lastError"

  Read-Host '按回车退出'
  exit 1
}

$url = "http://127.0.0.1:$port/"
Write-Host ''
Write-Host "网站已启动：$url"
Write-Host '请保持本窗口打开；关闭本窗口即停止。'
Write-Host ''
Start-Process $url

function Send-Bytes($ctx, [int]$status, [string]$type, [byte[]]$bytes) {
  $ctx.Response.StatusCode = $status
  $ctx.Response.ContentType = $type
  $ctx.Response.ContentLength64 = $bytes.Length
  if ($bytes.Length -gt 0) { $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length) }
  $ctx.Response.OutputStream.Close()
}

while ($listener.IsListening) {
  $ctx = $listener.GetContext()
  try {
    $path = [System.Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath)
    if ($path.EndsWith('/')) { $path += 'index.html' }
    $rel = $path.TrimStart('/').Replace('/', '\')
    $full = [System.IO.Path]::GetFullPath((Join-Path $rootFull $rel))
    $inside = $full.StartsWith($rootFull, [System.StringComparison]::OrdinalIgnoreCase)
    if ($inside -and (Test-Path -LiteralPath $full -PathType Leaf)) {
      $ext = [System.IO.Path]::GetExtension($full).ToLowerInvariant()
      $type = $mime[$ext]
      if (-not $type) { $type = 'application/octet-stream' }
      Send-Bytes $ctx 200 $type ([System.IO.File]::ReadAllBytes($full))
    } else {
      Send-Bytes $ctx 404 'text/plain; charset=utf-8' ([System.Text.Encoding]::UTF8.GetBytes('404 Not Found'))
    }
  } catch {
    try { Send-Bytes $ctx 500 'text/plain; charset=utf-8' ([System.Text.Encoding]::UTF8.GetBytes('500 Server Error')) } catch { }
  }
}
