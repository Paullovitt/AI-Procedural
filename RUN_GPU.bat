@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python run_gpu.py --smoke
if errorlevel 1 (
  echo.
  echo ERRO ao iniciar o runtime GPU.
  pause
  exit /b 1
)
pause
