@echo off
rem Tessera - cift tikla calistir
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" ".venv\Scripts\pythonw.exe" -m tessera.baslat %*
) else (
  echo Python ortami bulunamadi: .venv klasoru eksik.
  pause
)
