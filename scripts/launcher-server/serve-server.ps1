# 局域网服务器启动脚本（Windows）：在服务器电脑上运行模型，局域网内其他人用浏览器访问。
# start-server.bat 会调用本脚本；也可以右键“使用 PowerShell 运行”。
# 环境检查：系统版本过低/页面文件缺失时先说清楚原因。
. (Join-Path $PSScriptRoot 'check-env.ps1')
if (-not (Test-OjEnvironment -ForBigModel)) { exit 1 }

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

# 读取同目录下可选的 server.env（每行 KEY=VALUE），例如模型上游与管理员口令配置。
$envFile = Join-Path $PSScriptRoot 'server.env'
if (Test-Path -LiteralPath $envFile) {
  # 用 .NET 读取并显式指定 UTF-8：Windows PowerShell 5.1 的 Get-Content -Encoding UTF8
  # 在**无 BOM 的文件**上会按系统 ANSI 代码页解码（简体中文系统即 GBK），
  # 于是用户手写的 server.env 里中文值会变成乱码。.NET 的方式行为一致且不回退。
  $envLines = $null
  try {
    $envLines = [System.IO.File]::ReadAllLines($envFile, [System.Text.Encoding]::UTF8)
  } catch {
    $envLines = Get-Content -LiteralPath $envFile
  }
  foreach ($line in $envLines) {
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
# 列出本机局域网地址。必须有多级回退：Get-NetIPAddress 属于 NetTCPIP 模块，
# Windows 10 早期版本（1507~1607）和精简版系统里可能缺失或未启用；
# ipconfig 从 Windows 2000 起一直存在，是最可靠的兜底。
$addrs = @()
try {
  $addrs = @(Get-NetIPAddress -AddressFamily IPv4 -ErrorAction Stop |
    Where-Object { $_.IPAddress -notlike '127.*' -and $_.PrefixOrigin -ne 'WellKnown' } |
    ForEach-Object { $_.IPAddress })
} catch { $addrs = @() }
if (-not $addrs -or $addrs.Count -eq 0) {
  # 回退：解析 ipconfig（所有 Windows 版本都有，且不受模块缺失影响）
  try {
    $addrs = @(ipconfig | Select-String -Pattern '([0-9]{1,3}(\.[0-9]{1,3}){3})' |
      ForEach-Object { $_.Matches[0].Groups[1].Value } |
      Where-Object { $_ -notlike '127.*' -and $_ -ne '255.255.255.255' } |
      Select-Object -Unique)
  } catch { $addrs = @() }
}
if ($addrs -and $addrs.Count -gt 0) {
  foreach ($a in $addrs) { Write-Host "  http://$($a):$($env:PORT)/" }
} else {
  Write-Host '  （未能自动列出局域网地址：请在命令提示符运行 ipconfig，找到“IPv4 地址”）'
}
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
