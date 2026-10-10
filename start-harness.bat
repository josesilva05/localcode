@echo off
rem start-harness.bat - the API (start-api.bat) and, as soon as the model is ready,
rem deepseek-harness's web UI opened as an app window (Chrome or Edge, a light profile of its
rem own). Closing the API (Ctrl+C or its window) stops the harness it started.
rem
rem   start-harness.bat                          defaults: API on :8080, harness on :3080
rem   start-harness.bat --reasoning-effort medium --harness-dir D:\deepseek-harness
call "%~dp0start-api.bat" --harness %*
