@echo off
rem cad-ai: one-time setup. The virtual environment goes to D: (C: has little free space).
cd /d "%~dp0"
if not exist "D:\cad-ai-data\venv\Scripts\python.exe" py -3.12 -m venv D:\cad-ai-data\venv
"D:\cad-ai-data\venv\Scripts\python.exe" -m pip install --upgrade pip
"D:\cad-ai-data\venv\Scripts\python.exe" -m pip install -r requirements.txt
echo Bitti. start.bat ile baslatin.
pause
