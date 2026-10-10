@echo off
rem start.bat - open localcode (Qwen3.8-Flash-Next on this engine) from a double-click or a prompt.
rem
rem   start.bat                          the model from LOCALCODE_MODEL, or the UD-Q4_K_XL GGUF in
rem                                      %USERPROFILE%\models\Qwen3.8-Flash-Next\UD-Q4_K_XL
rem   start.bat D:\models\some.gguf      that GGUF (any shard of a split one)
rem   start.bat D:\models\UD-Q4_K_XL     the first shard found in that directory
rem   start.bat --reasoning low          any localcode option (localcode --help lists them)
setlocal EnableDelayedExpansion
set "ROOT=%~dp0"
set "EXE=%ROOT%build\Release\localcode.exe"

if not exist "%EXE%" (
    echo localcode is not built yet. From %ROOT%:
    echo   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGLM53F_CUDA=ON
    echo   cmake --build build --config Release -j
    goto :fail
)

rem A .gguf file or a directory holding one, given as the first argument, wins; then
rem LOCALCODE_MODEL; then the default location.
set "FIRST="
if not "%~1"=="" (
    if /i "%~x1"==".gguf" if exist "%~1" set "FIRST=%~1"
    if not defined FIRST if exist "%~1\" call :first_shard "%~1"
)
if defined FIRST set "LOCALCODE_MODEL=%FIRST%"
if not defined LOCALCODE_MODEL (
    set "DEF=%USERPROFILE%\models\Qwen3.8-Flash-Next\UD-Q4_K_XL"
    if exist "!DEF!\" call :first_shard "!DEF!"
    if defined FIRST set "LOCALCODE_MODEL=!FIRST!"
)
if not defined LOCALCODE_MODEL (
    set /p "LOCALCODE_MODEL=Model GGUF (any shard): "
)
if not exist "%LOCALCODE_MODEL%" (
    echo No GGUF at "%LOCALCODE_MODEL%".
    echo Give a .gguf file or its directory as an argument, or set LOCALCODE_MODEL.
    goto :fail
)

rem The model travels in LOCALCODE_MODEL, so a path given as the first argument is dropped
rem from what localcode receives.
set "ARGS=%*"
if not "%~1"=="" (
    set "DROP="
    if /i "%~x1"==".gguf" set "DROP=1"
    if exist "%~1\" set "DROP=1"
    if defined DROP call :drop_first %*
)

rem Files named in /file are relative to the working directory: start where the script is.
pushd "%ROOT%"
"%EXE%" %ARGS%
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo.
    echo localcode ended with code %RC%.
    goto :fail
)
exit /b 0

rem the first shard (*-00001-of-*.gguf), else the first .gguf, of a directory
:first_shard
for %%F in ("%~1\*-00001-of-*.gguf") do if not defined FIRST set "FIRST=%%~fF"
if not defined FIRST for %%F in ("%~1\*.gguf") do if not defined FIRST set "FIRST=%%~fF"
exit /b 0

:drop_first
set "ARGS="
shift
:drop_loop
if "%~1"=="" exit /b 0
set "ARGS=%ARGS% %1"
shift
goto :drop_loop

:fail
rem Keep the window open when started with a double-click, so the message can be read.
echo %CMDCMDLINE% | find /i "/c" >nul && pause
exit /b 1
