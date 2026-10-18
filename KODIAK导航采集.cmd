@echo off
cd /d "%~dp0"
python -m wardog_ad collect --vehicle KODIAK --camera-view unknown --fps 15 --max-width 512
pause
