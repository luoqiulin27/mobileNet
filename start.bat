@echo off
chcp 65001 >nul
echo ============================================
echo   Blind Assist Detection - 启动
echo ============================================
echo.

:: 检查 conda 是否可用
where conda >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 conda，请先运行 setup.bat 安装环境
    pause
    exit /b 1
)

:: 激活环境
call conda activate blind-assist
if errorlevel 1 (
    echo [错误] 环境激活失败，请先运行 setup.bat
    pause
    exit /b 1
)

:: 检查数据目录
if not exist "blind-assist-detection\data\phase1_sanpo_5class\images" (
    echo [警告] 未找到训练数据
    echo 请将数据集放在以下目录:
    echo   blind-assist-detection\data\phase1_sanpo_5class\
    echo.
    echo 如果只想启动 Web Demo，可忽略此警告
    echo.
)

:: 进入项目目录
cd blind-assist-detection

:: 启动 Web Demo
echo [启动] Web Demo...
echo 访问地址: http://localhost:5000
echo 按 Ctrl+C 停止
echo.
python web_demo\app_detect.py

pause
