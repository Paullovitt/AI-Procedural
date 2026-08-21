@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo AI-Procedural V14 - GPU/VRAM - sessao persistente de prompts
echo.
python prompt_session_v14.py
if errorlevel 1 (
  echo.
  echo ERRO ao iniciar a sessao V14 GPU.
  pause
  exit /b 1
)
pause
