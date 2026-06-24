# ============================================================
# run_and_shutdown.ps1
# 流程：cd 到项目目录 → 跑模拟 → 成功才 提交+推送GitHub → 关机
# 用法（在 VSCode 终端或 PowerShell）：
#   .\run_and_shutdown.ps1
# 若提示禁止运行脚本，先执行一次（只需一次）：
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# ============================================================

# —— 可按需修改的设置 ——
$ProjectDir   = "C:\Users\sheesh\Desktop\bridge_DL_poly"   # 项目目录
$GridRepeats  = 50                                          # 重复次数
$Epochs       = 800                                         # 训练轮数
$ShutdownWait = 60                                          # 关机倒计时(秒)，期间可 shutdown /a 取消
# ----------------------------------------------------------

# 切换 UTF-8，避免中文表格在终端乱码
chcp 65001 > $null

# 1. 进入项目目录（关键：否则 import taylor_expand 会失败、读写文件路径会错）
Set-Location $ProjectDir
Write-Host "已进入目录：$ProjectDir" -ForegroundColor Cyan

# 2. 先确认 LayerTaylor-PR 能被正确加载（taylor_expand.py 必须在同目录）
Write-Host "检查 taylor_expand 是否可用..." -ForegroundColor Cyan
python -c "import measure_morala as m; print('LTPR enabled:', m._HAS_TAYLOR_EXPAND)"

# 3. 跑模拟
Write-Host "开始运行模拟（grid-repeats=$GridRepeats, epochs=$Epochs）..." -ForegroundColor Cyan
python measure_morala.py --grid-repeats $GridRepeats --epochs $Epochs

# 4. 只有 Python 正常结束（退出码 0）才提交、推送、关机
if ($LASTEXITCODE -eq 0) {
    Write-Host "模拟成功，开始提交到 GitHub..." -ForegroundColor Green

    git add -A
    git commit -m "模拟结果 $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    git push

    if ($LASTEXITCODE -eq 0) {
        Write-Host "推送成功。$ShutdownWait 秒后关机——取消请运行: shutdown /a" -ForegroundColor Yellow
        shutdown /s /t $ShutdownWait
    } else {
        Write-Host "git push 失败（可能需要认证），不关机，请手动检查。" -ForegroundColor Red
    }
} else {
    Write-Host "Python 运行失败（退出码 $LASTEXITCODE），不推送、不关机。" -ForegroundColor Red
}
