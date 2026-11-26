# 环境兼容性检查（Windows）：在启动前把关，把"跑不起来"翻译成"为什么 + 怎么办"。
#
# 背景：Windows 10 有 7 个主要版本（1507/10240 ~ 22H2/19045），跨度 7 年。
# 早期版本缺一些东西，而这些东西一旦缺失，表现都是"闪一下就没了"或"白屏"，
# 用户看到的只有一句莫名其妙的报错——和之前那些崩溃码一样，等于让他猜。
#
# 本脚本只做检查与提示，不改动系统、不阻塞能用的情况：
#   - 版本过低（< Win10 10240）→ 明确说不支持，并说明原因
#   - 早期 Win10（< 1809）→ 能跑，但提示浏览器和 IE 的问题
#   - 页面文件未配置 → 提示大模型会加载失败（0xC0000005 的常见真因）
#
# 用法（在别的脚本里 dot-source 后调用）：
#   . "$PSScriptRoot\check-env.ps1"
#   if (-not (Test-OjEnvironment -ForBigModel)) { exit 1 }
#
# 注意：本文件必须保存为 **UTF-8 with BOM**。
# Windows PowerShell 5.1 对无 BOM 的 UTF-8 会按系统 ANSI 代码页解码（简体中文系统
# 即 GBK），中文提示会变成乱码——这本身就是一次踩过的坑。

function Get-OjWindowsInfo {
  # [System.Environment]::OSVersion.Version 在 Win10 上给出 10.0.<build>.<rev>，
  # 且不依赖 WMI（WMI 服务被禁用时也能用）。真正的版本差异体现在 build 上：
  #   10240 = 1507（首个 Win10）  10586 = 1511   14393 = 1607   15063 = 1703
  #   16299 = 1709                17134 = 1803   17763 = 1809   19041 = 2004 …
  $v = [System.Environment]::OSVersion.Version
  $build = 0
  if ($v.Major -eq 10 -and $v.Build -gt 0) {
    $build = $v.Build
  } else {
    # 兜底：老系统上 OSVersion 可能只给 6.x，用注册表拿真实 build
    try {
      $build = [int](Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -Name CurrentBuildNumber -ErrorAction Stop).CurrentBuildNumber
    } catch { $build = 0 }
  }
  $name = '未知'
  if ($build -ge 22000) { $name = 'Windows 11' }
  elseif ($build -ge 10240) { $name = 'Windows 10' }
  elseif ($build -ge 9600) { $name = 'Windows 8.1' }
  elseif ($build -ge 9200) { $name = 'Windows 8' }
  elseif ($build -ge 7600) { $name = 'Windows 7' }
  # 具体版本号（人话），方便用户对照“设置 → 系统 → 关于”
  $release = ''
  # Windows 11 的 Name 已经是 'Windows 11'，release 留空避免显示成“Windows 11 Windows 11”
  elseif ($build -ge 19045) { $release = '22H2' }
  elseif ($build -ge 19044) { $release = '21H2' }
  elseif ($build -ge 19043) { $release = '21H1' }
  elseif ($build -ge 19042) { $release = '20H2' }
  elseif ($build -ge 19041) { $release = '2004' }
  elseif ($build -ge 18363) { $release = '1909' }
  elseif ($build -ge 18362) { $release = '1903' }
  elseif ($build -ge 17763) { $release = '1809' }
  elseif ($build -ge 17134) { $release = '1803' }
  elseif ($build -ge 16299) { $release = '1709' }
  elseif ($build -ge 15063) { $release = '1703' }
  elseif ($build -ge 14393) { $release = '1607' }
  elseif ($build -ge 10586) { $release = '1511' }
  elseif ($build -ge 10240) { $release = '1507' }
  return [pscustomobject]@{ Build = $build; Name = $name; Release = $release }
}

function Test-OjEnvironment {
  param(
    [switch]$ForBigModel   # 要跑本机大模型时，额外检查页面文件
  )
  $w = Get-OjWindowsInfo
  $problems = @()
  $notes = @()

  # —— 1. 系统版本 ——
  if ($w.Build -gt 0 -and $w.Build -lt 10240) {
    # Node.js 自检门槛是 Windows 8.1（9600），但本软件的网页前端用到 ES2020
    # （可选链、空值合并），Windows 8.1 自带的 IE11 完全跑不了；
    # 且新版 llama.cpp 二进制依赖 Win10 才有的系统 API。
    Write-Host ''
    Write-Host "检测到系统：$($w.Name)（内部版本 $($w.Build)）。"
    Write-Host '抱歉：本软件需要 Windows 10 或更新版本，当前系统跑不起来。' -ForegroundColor Yellow
    Write-Host ''
    Write-Host '原因：'
    Write-Host '  · 网页界面用到 ES2020 语法，Windows 8.1 自带的 IE11 不支持，会白屏。'
    Write-Host '  · 本机大模型引擎依赖 Windows 10 才有的一些系统接口。'
    Write-Host ''
    Write-Host '可以怎么办：'
    Write-Host '  · 升级到 Windows 10（微软官方仍可免费升级）。'
    Write-Host '  · 或在另一台 Windows 10/11 电脑上运行，局域网内其他设备用浏览器访问（服务器版）。'
    Write-Host ''
    Read-Host '按回车关闭'
    return $false
  }

  # —— 2. 早期 Win10：能跑，但浏览器可能拖后腿 ——
  if ($w.Build -gt 0 -and $w.Build -lt 17763) {
    $notes += "你的系统是 Windows 10 $($w.Release)（内部版本 $($w.Build)），比较旧。程序本身能跑，但请注意："
    $notes += '  · 这个版本自带的浏览器是 IE11 或早期 Edge，运行本软件的网页界面会白屏或报错。'
    $notes += '  · 请先安装新版浏览器再使用：Chrome、Edge（新版）或 Firefox 均可，免费。'
    $notes += '  · 下载地址：https://www.google.cn/chrome/  或  https://www.mozilla.org/zh-CN/firefox/'
  }

  # —— 3. 大模型：页面文件（这是 0xC0000005 最常见的真因）——
  if ($ForBigModel) {
    $pfMB = $null
    try {
      $pfMB = @(Get-CimInstance Win32_PageFileUsage -ErrorAction Stop |
        Measure-Object -Property AllocatedBaseSize -Sum).Sum
    } catch {
      try {
        $pfMB = @(Get-WmiObject Win32_PageFileUsage -ErrorAction SilentlyContinue |
          Measure-Object -Property AllocatedBaseSize -Sum).Sum
      } catch { $pfMB = $null }
    }
    if ($null -ne $pfMB -and [double]$pfMB -lt 2048) {
      $problems += @(
        '没有检测到页面文件（虚拟内存），或它小于 2 GB。',
        '  影响：加载较大的模型会因为申请不到内存而直接崩溃（现象常是 0xC0000005），',
        '        而任务管理器里内存看起来"还有空余"，非常容易误判成软件 bug。',
        '  怎么办：系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存 → 更改，',
        '          勾选"自动管理所有驱动器的分页文件大小"，或手动设到 8 GB 以上，然后重启。'
      )
    }
  }

  if ($notes.Count -gt 0) {
    Write-Host ''
    foreach ($n in $notes) { Write-Host $n -ForegroundColor Yellow }
    Write-Host ''
  }
  if ($problems.Count -gt 0) {
    Write-Host ''
    Write-Host '以下是会影响使用的问题（程序仍会继续启动）：' -ForegroundColor Yellow
    foreach ($p in $problems) { Write-Host "  $p" }
    Write-Host ''
  }
  return $true
}
