@echo off
chcp 65001 >nul
echo ============================================
echo   Blind Assist Detection Web Demo
echo ============================================
echo.

:: 用 conda run 在 vscode 环境中启动，避免环境变量干扰
start "" cmd /k "conda run -n vscode --no-capture-output python D:\project\mobileNet++\mobileNet\blind-assist-detection\web_demo\app_detect.py"

:: 等待服务启动后打开浏览器
echo 正在启动服务，请稍候...
timeout /t 4 /nobreak >nul
start http://localhost:5000
