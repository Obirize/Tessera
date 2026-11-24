@echo off
cd /d "%~dp0"
if "%~1"=="" (echo Gorseli bu dosyanin uzerine surukle-birak. & pause & exit /b)
".venv\Scripts\python.exe" scripts\donustur.py "%~1" 40 4
start "" "cikti\%~n1"
