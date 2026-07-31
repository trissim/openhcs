@echo off
setlocal
start "" powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0Install-OpenHCS.ps1" -BrandIconPath "%~dp0..\..\..\openhcs\resources\assets\openhcs.ico"
exit /b 0
