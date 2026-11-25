@echo off
chcp 65001 >nul
setlocal
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if errorlevel 1 goto failed
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0web\launch_workbench.ps1"
if errorlevel 1 goto failed
exit /b 0
:failed
echo.
echo Startup failed. Keep this window open to read the error above.
pause
exit /b 1
