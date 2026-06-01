@echo off
setlocal

echo ===============================
echo Creating virtual environment
echo ===============================

python -m venv .venv

call .venv\Scripts\activate.bat

echo ===============================
echo Installing project dependencies
echo ===============================

pip install -r requirements.txt

echo ===============================
echo Installation complete!
echo ===============================

pause