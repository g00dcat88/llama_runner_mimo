@echo off
setlocal
cd /d "%~dp0"
pip install flask -q 2>nul
echo.
echo  Llama Runner - http://127.0.0.1:5000
echo.
python app.py
pause
