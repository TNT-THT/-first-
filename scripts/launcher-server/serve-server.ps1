# 局域网服务器启动脚本（Windows）：在服务器电脑上运行模型，局域网内其他人用浏览器访问。
# start-server.bat 会调用本脚本；也可以右键“使用 PowerShell 运行”。
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# 读取同目录下可选的 server.env（每行 KEY=VALUE），例如模型上游与管理员口令配置。
$envFile = Join-Path $PSScriptRoot 'server.env'
if (Test-Path -LiteralPath $envFile) {
  foreach ($line in Get-Content -LiteralPath $envFile -Encoding UTF8) {
    $t = $line.Trim()
    if (-not $t -or $t.StartsWith('#')) { continue }
    $i = $t.IndexOf('=')
    if ($i -gt 0) {
      $k = $t.Substring(0, $i).Trim()
      $v = $t.Substring($i + 1).Trim()
      if ($k) { Set-Item -Path "Env:$k" -Value $v }
    }
  }
  Write-Host '已读取 server.env 配置。'
}

if (-not $env:HOST) { $env:HOST = '0.0.0.0' }
if (-not $env:PORT) { $env:PORT = '3000' }

$bundledNode = Join-Path $PSScriptRoot 'runtime\node-win-x64.exe'
$nodeCmd = $null
if (Test-Path -LiteralPath $bundledNode) {
  $nodeCmd = $bundledNode
} elseif (Get-Command node -ErrorAction SilentlyContinue) {
  $nodeCmd = 'node'
}
if (-not $nodeCmd) {
  Write-Host '未找到 Node.js：请安装 Node.js LTS 后重新双击 start-server.bat。'
  Read-Host '按回车关闭'
  exit 1
}

Write-Host ''
Write-Host "服务器启动中（端口 $($env:PORT)）。局域网内其他设备请打开下面任意一个地址："
try {
  Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } |
    ForEach-Object { Write-Host "  http://$($_.IPAddress):$($env:PORT)/" }
} catch { }
Write-Host "本机也可以打开： http://127.0.0.1:$($env:PORT)/"
Write-Host '使用期间请保持本窗口打开；关闭窗口即停止服务。'
Write-Host '提醒：普通聊天不需要登录；下载/加载模型等管理操作需要管理员口令（在 server.env 里设置 API_ADMIN_PASSWORD，至少 16 位）。'
Write-Host ''
& $nodeCmd 'server\index.js'
$code = $LASTEXITCODE
if ($code -ne 0) {
  Write-Host "服务器已退出（退出码 $code）。请把上面的报错反馈给提供者。"
  Read-Host '按回车关闭'
  exit $code
}
