@echo off
echo ========================================
echo   论文实验顺序训练队列
echo ========================================
echo.
echo 实验顺序:
echo   [1/6] A3 CIoU - 仅评估 (checkpoint已有)
echo   [2/6] A4 ECA+CIoU - 训练100epoch + 评估
echo   [3/6] B1 SE - 训练100epoch + 评估
echo   [4/6] B2 CBAM - 训练100epoch + 评估
echo   [5/6] B3 IoU - 训练100epoch + 评估
echo   [6/6] B4 DIoU - 训练100epoch + 评估
echo.
echo 预计总时间: ~36小时 (RTX 3070)
echo 日志位置: outputs\runs\<run_name>\logs\
echo 评估结果: outputs\runs\<run_name>\metrics\
echo.
echo 按任意键开始...
pause >nul

powershell -ExecutionPolicy Bypass -File tools\run_thesis_queue.ps1

echo.
echo 队列已完成或出错，按任意键退出...
pause >nul
