@echo off
rem cad-ai: start the website (http://127.0.0.1:8800). Data lives in D:\cad-ai-data.
cd /d "%~dp0"
if not exist "D:\cad-ai-data\venv\Scripts\python.exe" (
  echo Once setup.bat calistirin.
  pause & exit /b 1
)
start "" http://127.0.0.1:8800
"D:\cad-ai-data\venv\Scripts\python.exe" -m cadai.web
