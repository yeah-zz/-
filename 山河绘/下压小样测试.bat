@echo off
setlocal
cd /d "%~dp0"
"D:\Anaconda3\python.exe" -X utf8 -B "%~dp0test_brush_pressure.py" --execute %*
pause
