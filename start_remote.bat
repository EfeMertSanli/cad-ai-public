@echo off
rem cad-ai: open the website to the network (Hamachi / Tailscale / LAN) with a password.
rem Others open http://<this computer's Hamachi IP>:8800 and log in with any user name + the password.
cd /d "%~dp0"
set /p CADAI_PASSWORD=Sifre belirleyin: 
if "%CADAI_PASSWORD%"=="" ( echo Sifre bos olamaz. & pause & exit /b 1 )
set CADAI_HOST=0.0.0.0
echo.
echo Bu bilgisayarin adresleri (Hamachi adresi 25.x.x.x ile baslar):
ipconfig | findstr /R /C:"IPv4"
echo.
echo Guvenlik duvari izni yoksa bir kez yonetici olarak calistirin:
echo   netsh advfirewall firewall add rule name="cad-ai 8800" dir=in action=allow protocol=TCP localport=8800
echo.
"D:\cad-ai-data\venv\Scripts\python.exe" -m cadai.web
pause
