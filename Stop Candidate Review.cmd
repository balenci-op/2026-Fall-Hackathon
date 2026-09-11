@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0candidate_review_launcher.ps1" -Action Stop
exit /b %errorlevel%
