@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 C:\path\to\CROSS-MORSE [additional apply_repository_upgrade.py options]
  exit /b 2
)
set "SCRIPT_DIR=%~dp0"
set "SOURCE=%~1"
shift
py -3 "%SCRIPT_DIR%apply_repository_upgrade.py" "%SOURCE%" --make-zip %*
exit /b %ERRORLEVEL%
