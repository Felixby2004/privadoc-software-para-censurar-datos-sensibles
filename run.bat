@echo off
setlocal
cd /d %~dp0
if not exist .venv (
  py -3.13 -m venv .venv 2>nul || py -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Error instalando dependencias. Revisa el mensaje anterior.
  pause
  exit /b 1
)
echo.
echo Privadoc iniciado en http://127.0.0.1:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
