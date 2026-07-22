@echo off
setlocal
start "" powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0Install-OpenHCS.ps1"
exit /b 0
