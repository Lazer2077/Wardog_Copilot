@echo off
cd /d "%~dp0"
python -m wardog_ad collect --vehicle unknown --camera-view unknown
pause
