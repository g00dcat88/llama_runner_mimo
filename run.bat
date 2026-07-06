@echo off
setlocal
cd /d "%~dp0"
pip install flask -q 2>nul

REM Disable PDL - required for Volta (V100) and Pascal (P100) GPUs
set GGML_CUDA_PDL=0

REM Auto-download/update llama.cpp binaries for detected GPU
python setup_llama.py

echo.
echo  Llama Runner - http://127.0.0.1:5000
echo.
python app.py
pause
