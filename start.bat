@echo off
echo ====================================
echo Notifica Customer Health Analytics
echo ====================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate venv
call venv\Scripts\activate

REM Install/update dependencies
echo Installing dependencies...
pip install -r requirements.txt -q

echo.
echo Starting dashboard...
echo.
echo Open your browser at: http://localhost:8501
echo.

streamlit run app.py

pause
