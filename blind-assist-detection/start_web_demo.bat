@echo off
chcp 65001 >nul
echo ============================================
echo   Blind Assist Detection Web Demo
echo ============================================
echo.
echo   http://localhost:5000
echo   API: POST http://localhost:5000/api/predict
echo.

cd /d %~dp0
set PYTHONUNBUFFERED=1
D:\mlp\anaconda3\python.exe -u web_demo\app_detect.py

pause
