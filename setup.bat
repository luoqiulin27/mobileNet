@echo off
chcp 65001 >nul
echo ============================================
echo   Blind Assist Detection - 环境安装
echo ============================================
echo.

:: 检查 conda 是否可用
where conda >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 conda，请先安装 Anaconda 或 Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

:: 检查环境是否已存在
conda env list | findstr "blind-assist" >nul
if not errorlevel 1 (
    echo [信息] 环境 blind-assist 已存在
    set /p RECREATE="是否重新创建？(y/N): "
    if /i "%RECREATE%"=="y" (
        echo [信息] 删除旧环境...
        conda env remove -n blind-assist -y
    ) else (
        echo [信息] 跳过环境创建
        goto :install_done
    )
)

echo.
echo [步骤 1/2] 创建 conda 环境...
echo 这可能需要几分钟，请耐心等待...
echo.
conda env create -f environment.yml
if errorlevel 1 (
    echo.
    echo [错误] 环境创建失败，请检查网络连接
    pause
    exit /b 1
)

:install_done
echo.
echo [步骤 2/2] 验证环境...
call conda activate blind-assist
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"
python -c "import torchvision; print(f'TorchVision {torchvision.__version__} OK')"
python -c "import cv2; print(f'OpenCV {cv2.__version__} OK')"

echo.
echo ============================================
echo   安装完成！
echo ============================================
echo.
echo 使用方法:
echo   1. 运行 start.bat 启动项目
echo   2. 或手动激活环境: conda activate blind-assist
echo.
pause
