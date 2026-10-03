@echo off
rem Builds the run_code sandbox image (docker/sandbox/Dockerfile).
rem   build_sandbox_image.cmd          -> assistant-sandbox:latest, no torch
rem   build_sandbox_image.cmd torch    -> adds CPU torch + torchvision (~750 MB)
setlocal
cd /d "%~dp0\.."
set TORCH=0
if /i "%1"=="torch" set TORCH=1
docker build --build-arg WITH_TORCH=%TORCH% -t assistant-sandbox:latest docker/sandbox
