@echo off
:: Make sure the file exists – *this* is the only case that should be "read error"
if not exist "%CONFIG_FILE%" exit /b 1

:: Read the line containing the keyedelayedexpansion

:: Exit if no key was provided
if "%~1"=="" exit /b 1

set "KEY=%~1"
set "VALUE="
set "CONFIG_FILE=configs\config.json"

:: Make sure the file exists – *this* is the only case that should be "read error"
if not exist "%CONFIG_FILE%" exit /b 1tlocal enabledelayedexpansion

:: Exit if no key was provided
if "%~1"=="" exit 1

set "KEY=%~1"
set "VALUE="
set "CONFIG_FILE=configs\config.json"

:: Make sure the file exists – *this* is the only case that should be “read error”
if not exist "%CONFIG_FILE%" exit 1

:: Read the line containing the key
::  – redirect stderr so a “not found” search doesn’t set ERRORLEVEL
for /f "tokens=2 delims=:," %%a in ('
    type "%CONFIG_FILE%" ^| findstr /C:"\"%KEY%\"" 2^>nul
') do (
    set "line=%%a"
    for /f "tokens=*" %%b in ("!line!") do set "VALUE=%%~b"
)

:: --- DO **NOT** test ERRORLEVEL here ---
::       (missing key is **not** an I/O error)

:: Handle the possible outcomes
if not defined VALUE      exit /b 2    :: key not present
if "!VALUE!"==""          exit /b 2
if "!VALUE!"=="null"      exit /b 3    :: key present but null

echo !VALUE!
exit /b 0
