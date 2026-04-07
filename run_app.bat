@echo off
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv python not found at "%cd%\venv\Scripts\python.exe"
    echo Create it first, for example:
    echo     python -m venv venv
    pause
    exit /b 1
)

set "PY_EXE=%cd%\venv\Scripts\python.exe"

echo Using Python: %PY_EXE%
"%PY_EXE%" -c "import sys; print('Interpreter:', sys.executable)"

"%PY_EXE%" -c "import langgraph" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependency: langgraph
    "%PY_EXE%" -m pip install langgraph
)

echo Starting Indic Multimodal RAG...
"%PY_EXE%" -m streamlit run app.py

endlocal
