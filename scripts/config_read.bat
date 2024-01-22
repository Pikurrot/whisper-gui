@echo off
setlocal enabledelayedexpansion

:: Check if a key was provided
if "%~1"=="" exit 1

:: Initialize variables
set KEY=%~1
set VALUE=

:: Read the line containing the key from config.json
for /f "tokens=2 delims=:," %%a in ('type configs\config.json ^| findstr /C:"\"%KEY%\""') do (
	set line=%%a
	:: Trim whitespace and quotes
	for /f "tokens=*" %%b in ("!line!") do set VALUE=%%~b
)

if %errorlevel% GEQ 1 (
	:: Couldn't read file
	exit 1
)

:: Check if VALUE was set
if not defined VALUE (
	:: Key not found
	exit 2
)
if "!VALUE!" EQU "" (
	:: Key not found
	exit 2
)
if "!VALUE!"=="null" (
	:: Key found but value is null
	exit 3
)

:: Output the value and exit
echo !VALUE!
exit 0
