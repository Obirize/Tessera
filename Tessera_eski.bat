@echo off
rem Tessera (eski arayuz) - cift tikla calistir
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m tessera %*
) else (
  echo Python ortami bulunamadi: .venv klasoru eksik.
  pause
)
