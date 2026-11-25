@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
if "%~1"=="" (
  echo 请把要提取的图片拖到这个文件上。
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_stable_line_extract.ps1" -ImagePath "%~1" -OpenPreview
if errorlevel 1 pause
