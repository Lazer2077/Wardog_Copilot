@echo off
cd /d "%~dp0"
start "" wscript.exe "%~dp0Wardog_launcher.vbs" collect
exit /b
