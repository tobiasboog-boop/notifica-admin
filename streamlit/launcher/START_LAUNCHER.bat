@echo off
REM Start Streamlit Launcher Service
REM Dit script start de launcher die automatisch Streamlit apps kan starten

cd /d "%~dp0"
cd ..\..

echo Starting Streamlit Launcher...
python streamlit\launcher\launcher.py

pause
