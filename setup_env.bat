@echo off
setlocal
where pwsh.exe >nul 2>&1
if errorlevel 1 (
    echo PowerShell 7 is required. Run setup_isaac61.ps1 from PowerShell 7.
    exit /b 1
)
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_isaac61.ps1" %*
exit /b %ERRORLEVEL%
