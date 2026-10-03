@echo off
rem Start Assistant DC (LM Studio / ComfyUI / Docker are started by the launcher).
cd /d "%~dp0"
if not exist "venv\Scripts\pythonw.exe" (
  echo Not installed yet: run  powershell -ExecutionPolicy Bypass -File setup.ps1
  pause
  exit /b 1
)
start "" "venv\Scripts\pythonw.exe" "scripts\launch_all.py" %*
