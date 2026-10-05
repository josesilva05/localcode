@echo off
rem start-api.bat - Qwen3.8-Flash-Next on this engine as an HTTP API: llama-server's /props and
rem OpenAI's /v1/chat/completions on http://127.0.0.1:8080, where deepseek-harness's llama.cpp
rem route (llamacpp/loaded) looks by default.
rem
rem   start-api.bat                              the model from LOCALCODE_MODEL, or the UD-Q4_K_XL GGUF
rem   start-api.bat --reasoning-effort medium    any option of tools\q38_server.py (--help lists them)
rem   start-api.bat --ctx 16384 --port 8081
rem
rem Do not run it beside localcode or llama-server: one copy of the model fills the RAM.
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

if not exist "%ROOT%build\Release\q38api.dll" (
    echo The engine library is not built yet. From %ROOT%:
    echo   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGLM53F_CUDA=ON
    echo   cmake --build build --config Release --target q38api
    goto :fail
)
"%PY%" -c "import jinja2, numpy" 2>nul
if errorlevel 1 (
    echo The server needs Python with jinja2 and numpy: %PY%
    goto :fail
)

"%PY%" "%ROOT%tools\q38_server.py" %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo the API ended with code %RC%.
    goto :fail
)
exit /b 0

:fail
echo %CMDCMDLINE% | find /i "/c" >nul && pause
exit /b 1
