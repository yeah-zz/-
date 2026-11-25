@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PY="
if exist "D:\Anaconda3\python.exe" set "PY=D:\Anaconda3\python.exe"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>&1
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo 未找到 Python。请先安装带 Pillow、NumPy 的 Python 3.10+。
  pause
  exit /b 1
)
echo 正在启动山河绘 · 名家笔触实验室...
echo 浏览器地址：http://127.0.0.1:8770
"%PY%" -u master_studio\server.py --port 8770 --open-browser
if errorlevel 1 pause