@echo off
cd /d "%~dp0"
if exist "D:\Anaconda3\python.exe" (
  "D:\Anaconda3\python.exe" -c "import sys, numpy, cv2" >nul 2>&1
  if not errorlevel 1 (
    "D:\Anaconda3\python.exe" launcher.py
    exit /b
  )
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if not errorlevel 1 (
    ".venv\Scripts\python.exe" launcher.py
    exit /b
  )
)
uv run --locked python launcher.py
