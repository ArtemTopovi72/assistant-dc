@echo off
REM Build the official telegram-bot-api (tdlib) server on Windows with MSVC + vcpkg.
REM Output: C:\tools\telegram-bot-api\telegram-bot-api.exe
REM Usage: scripts\build_telegram_bot_api.cmd   (30-60 min; log in runtime\tbapi_build.log)
setlocal
set ROOT=C:\tools\tbapi-src
set OUT=C:\tools\telegram-bot-api
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 || exit /b 1

if not exist C:\tools mkdir C:\tools
if not exist %ROOT% (
  git clone --recursive --depth 1 https://github.com/tdlib/telegram-bot-api.git %ROOT% || exit /b 1
)
cd /d %ROOT%
if not exist vcpkg (
  git clone --depth 1 https://github.com/microsoft/vcpkg.git || exit /b 1
  call vcpkg\bootstrap-vcpkg.bat -disableMetrics || exit /b 1
)
vcpkg\vcpkg.exe install gperf:x64-windows openssl:x64-windows zlib:x64-windows || exit /b 1

if not exist build mkdir build
cd build
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=%ROOT%\vcpkg\scripts\buildsystems\vcpkg.cmake ^
  -DCMAKE_INSTALL_PREFIX:PATH=%OUT% .. || exit /b 1
cmake --build . --target install -j 12 || exit /b 1
REM the exe links vcpkg DLLs dynamically (z.dll, libssl, libcrypto) -- ship them beside it
copy /Y %ROOT%\vcpkg\installed\x64-windows\bin\*.dll %OUT%\bin\ >nul
echo BUILD OK: %OUT%\bin\telegram-bot-api.exe
