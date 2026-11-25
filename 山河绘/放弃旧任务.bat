@echo off
setlocal
cd /d "%~dp0"
"D:\Anaconda3\python.exe" -X utf8 -B "%~dp0abandon_old_task.py" %*
set "taskExitCode=%errorlevel%"
if /i "%~1"=="--check" exit /b %taskExitCode%
pause
exit /b %taskExitCode%
