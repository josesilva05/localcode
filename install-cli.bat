@echo off
rem install-cli.bat - the `localcode` command in any terminal: cd to a folder, type localcode,
rem and the agent works in that folder.
rem
rem Writes %USERPROFILE%\.local\bin\localcode.cmd, which runs this build's localcode.exe in the
rem folder it is typed in, and puts %USERPROFILE%\.local\bin on the user PATH when it is not
rem there yet (a new terminal sees it). Run it again after moving this folder.
setlocal
set "EXE=%~dp0build\Release\localcode.exe"
set "BIN=%USERPROFILE%\.local\bin"
if not exist "%EXE%" (
    echo localcode is not built yet: cmake --build build --config Release --target localcode
    exit /b 1
)
if not exist "%BIN%" mkdir "%BIN%"
(
    echo @echo off
    echo rem localcode in the folder it is typed in ^(written by %~f0^)
    echo "%EXE%" %%*
) > "%BIN%\localcode.cmd"
echo wrote %BIN%\localcode.cmd

echo ;%PATH%; | find /i ";%BIN%;" >nul
if errorlevel 1 (
    powershell -NoProfile -Command "$p=[Environment]::GetEnvironmentVariable('Path','User'); if (($p -split ';') -notcontains $env:BIN) { [Environment]::SetEnvironmentVariable('Path', ($p.TrimEnd(';') + ';' + $env:BIN), 'User'); 'added ' + $env:BIN + ' to the user PATH: open a new terminal' }"
) else (
    echo %BIN% is on the PATH: type localcode in any folder
)
exit /b 0
