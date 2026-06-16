# 1. 跑模拟
python measure_morala_gpu.py --device cuda --repeats 50

# 2. 只有成功才提交、推送、关机
if ($LASTEXITCODE -eq 0) {
    git add -A
    git commit -m "模拟结果 $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
    git push
    shutdown /s /t 60
} else {
    Write-Host "Python 运行失败，不推送不关机"
}